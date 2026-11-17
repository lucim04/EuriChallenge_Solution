import os
import sys
import json
import time
import html
import base64
import hashlib
import math
import threading
import streamlit as st
import streamlit.components.v1 as components
import cv2
import mediapipe as mp
import av
import numpy as np
from streamlit_webrtc import webrtc_streamer, WebRtcMode

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import obtener_estilos_dibujo
import rehab_engine as _re
from rehab_engine import RehabEngine

ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")

st.set_page_config(
    page_title="EuriChallenge · Rehab Monitor",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def leer(nombre):
    with open(os.path.join(ASSETS_DIR, nombre), encoding="utf-8") as f:
        return f.read()


@st.cache_data
def logo_b64(nombre):
    with open(os.path.join(ASSETS_DIR, nombre), "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode()


EURIELEC = logo_b64("eurielec.png")
ELGATO = logo_b64("elgato.png")

# ruido para el grano del loader
GRAIN = (
    "data:image/svg+xml;utf8,"
    "<svg xmlns='http://www.w3.org/2000/svg' width='220' height='220'>"
    "<filter id='n'><feTurbulence type='fractalNoise' baseFrequency='0.85' numOctaves='3' stitchTiles='stitch'/>"
    "<feColorMatrix values='0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0.55 0'/></filter>"
    "<rect width='100%25' height='100%25' filter='url(%23n)'/></svg>"
)

estilos = leer("estilos.css").replace("__GRAIN__", GRAIN)
html_intro = leer("intro.html").replace("__EURIELEC__", EURIELEC).replace("__ELGATO__", ELGATO)
frag = leer("fondo.glsl")
js_sonido = leer("sonido.js")
version = hashlib.md5((estilos + html_intro + frag + leer("intro.js")).encode()).hexdigest()[:10]
js_intro = (
    leer("intro.js")
    .replace("__HTML__", json.dumps(html_intro))
    .replace("__FRAG__", json.dumps(frag))
    .replace("__VER__", version)
)

st.markdown(f"<style>{estilos}</style>", unsafe_allow_html=True)

st.markdown(
    f"""<div class="loader" aria-hidden="true">
<div class="loader-red"></div>
<div class="loader-logos">
<div class="logo eurielec"><img class="ghost" src="{EURIELEC}" alt=""><img class="sharp" src="{EURIELEC}" alt=""></div>
<div class="logo elgato"><img class="ghost" src="{ELGATO}" alt=""><img class="sharp" src="{ELGATO}" alt=""></div>
</div>
</div>""",
    unsafe_allow_html=True,
)

components.html(f"<script>{js_sonido}</script><script>{js_intro}</script>", height=0)

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
spec_puntos, spec_conexiones = obtener_estilos_dibujo()

# ==================================================================== GUIA FANTASMA (modo juego)
# Una sombra con las proporciones del paciente hace el ejercicio y ESPERA en cada fase:
#   va al objetivo -> espera a que el MOTOR acepte la posicion -> vuelve -> espera al reposo ->
#   si el motor ha contado la repeticion: nota y +1; si la ha anulado: se repite, con el motivo.
# Todas las reglas (umbrales, tiempos minimos, compensaciones) son las de rehab_engine.py: la guia
# solo lee el estado del motor (no lo modifica), asi que el conteo de eval.py no cambia y el panel
# y la guia siempre coinciden.
# Los textos van sin tildes porque las fuentes de OpenCV no las dibujan.
# rutina: (ejercicio, lado, repeticiones). lado: "derecho", "izquierdo" o "ambos" (abduccion)
RUTINA = [("abduccion", "derecho", 3), ("abduccion", "izquierdo", 3), ("sentadilla", None, 5)]
# rutinas que se eligen con los botones (o el Stream Deck)
RUTINAS = {
    "Rutina completa": RUTINA,
    "Hombro derecho": [("abduccion", "derecho", 5)],
    "Hombro izquierdo": [("abduccion", "izquierdo", 5)],
    "Sentadillas": [("sentadilla", None, 8)],
}
GUIA_T_ANIM_S = 2.0          # duracion de la animacion de la sombra entre posiciones
GUIA_T_PREPARACION_S = 2.0   # quieto en reposo antes de empezar cada ejercicio
GUIA_T_ENTRE_REPS_S = 1.0    # pausa de la sombra en reposo antes de cada repeticion
GUIA_T_VERDE_S = 1.0         # cuanto se mantiene la sombra en verde al acertar la posicion
GUIA_ALFA_OK = 0.85          # opacidad de la sombra cuando esta en verde (resalta mas)
GUIA_T_MOTIVO_S = 2.5        # cuanto se ve el motivo de una repeticion no valida
GUIA_TAU_VIVO_S = 0.04       # suavizado minimo de los landmarks de MediaPipe (solo quita el temblor)
GUIA_N_CALIB = 30            # fotogramas en reposo para medir al paciente (mediana, ~1 s)
GUIA_ALFA = 0.55             # opacidad de la sombra
# objetivos que ensena la sombra, sacados del motor
ABD_SOMBRA = getattr(_re, "ABD_OBJETIVO", 90.0)          # el brazo sube hasta el objetivo del motor
SENT_SOMBRA_RODILLA = _re.SENT_ABAJO - 10.0              # la sombra baja un poco mas que el umbral
FLEX_SOMBRA = 180.0 - SENT_SOMBRA_RODILLA                # flexion de rodilla que dibuja la sombra

# colores BGR
# colores del sistema de Apple en modo oscuro (BGR)
C_GUIA = (255, 132, 10)      # azul #0A84FF: la sombra marca el camino
C_OK = (88, 209, 48)         # verde #30D158: posicion correcta
C_AVISO = (10, 159, 255)     # naranja #FF9F0A: en posicion pero con compensacion
C_ROJO = (58, 69, 255)       # rojo #FF453A
C_AMARILLO = (10, 214, 255)  # amarillo #FFD60A
C_TEXTO = (255, 255, 255)
C_SECUNDARIO = (178, 170, 170)   # gris secundario (etiquetas)

# avisos para el paciente a partir de lo que mide el motor
AVISO_PLANO = "Sube el brazo por el lado, no por delante"
AVISO_CODO = "Estira el brazo"
AVISO_LATERAL = "No te ladees, tronco recto"
AVISO_ATRAS = "No te eches hacia atrás"
AVISO_HOMBRO = "Relaja el hombro, no lo subas"
AVISO_GIRO = "Ponte de frente a la cámara"
AVISO_TRONCO = "Espalda más recta, no te inclines"
AVISO_ZANCADA = "Pies a la misma altura"
AVISO_PIE = "Apoya los dos pies"
AVISO_PROFUNDA = "No bajes tanto"

# proporciones por defecto (relativas a la altura del tronco) hasta medir las del paciente
PROP_DEF = {"hombros": 0.75, "caderas": 0.55, "brazo": 0.60, "antebrazo": 0.52,
            "muslo": 0.90, "tibia": 0.85, "cabeza": 0.45}

# landmarks de MediaPipe que usa la sombra (I = izquierdo del paciente)
PUNTOS_GUIA = {"hombroI": 11, "hombroD": 12, "codoI": 13, "codoD": 14, "munecaI": 15, "munecaD": 16,
               "caderaI": 23, "caderaD": 24, "rodillaI": 25, "rodillaD": 26, "tobilloI": 27, "tobilloD": 28,
               "cabeza": 0}

# estados de la sentadilla en el motor
SENT_EN_REP = ("bajando", "abajo", "subiendo")


def _ema(previo, nuevo, dt, tau):
    if previo is None:
        return nuevo
    if dt <= 0:
        return previo
    return previo + (1.0 - math.exp(-dt / tau)) * (nuevo - previo)


def _suave(p):
    """Interpolacion suave 0..1 (empieza y acaba despacio)."""
    p = min(1.0, max(0.0, p))
    return p * p * (3 - 2 * p)


# motivos tecnicos del motor -> mensajes para el paciente
MOTIVOS_AMABLES = [
    ("inclinacion lateral", "te has inclinado hacia un lado"),
    ("hombro izquierdo encogido", "has subido el hombro"),
    ("hombro derecho encogido", "has subido el hombro"),
    ("tronco demasiado inclinado", "tronco demasiado inclinado"),
    ("zancada", "un pie delante del otro"),
    ("pierna levantada", "apoya los dos pies"),
    ("demasiado profunda", "has bajado demasiado"),
    ("cuclillas", "has bajado demasiado"),
    ("poco profunda", "baja un poco más"),
    ("muy breve abajo", "aguanta un instante abajo"),
    ("sin pausa abajo", "aguanta un instante abajo"),
    ("brazo", "brazo no detectado"),
]


def _motivo_amable(texto):
    t = (texto or "").lower()
    for clave, amable in MOTIVOS_AMABLES:
        if clave in t:
            return amable
    return texto


def _ascii(texto):
    """OpenCV no dibuja tildes ni simbolos: se quitan."""
    import unicodedata
    t = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode()
    return t.replace("  ", " ").strip()


# ==================================================================== ESTILO VISUAL (tipo Apple)
# Todo lo que se pinta sobre el video: sombra "de cristal" con halo, anillo de progreso, tarjetas
# translucidas con el fondo desenfocado y efectos al conseguir la repeticion. Texto con Pillow (trae una
# tipografia escalable, Aileron, desde la version 10.1); si se deja un .ttf en assets (por ejemplo Inter,
# licencia OFL) como fuente.ttf / fuente_negrita.ttf, se usa ese. Sin Pillow se cae a las letras de OpenCV.
from collections import OrderedDict

try:
    from PIL import Image, ImageDraw, ImageFont
    _PIL_OK = True
except Exception:  # pragma: no cover
    _PIL_OK = False

FUENTE_REGULAR = os.path.join(ASSETS_DIR, "fuente.ttf")
FUENTE_NEGRITA = os.path.join(ASSETS_DIR, "fuente_negrita.ttf")
SUPERMUESTREO = 2                 # las tarjetas se dibujan al doble y se reducen: bordes suaves
FONDO_TARJETA = (22, 22, 24, 178)  # RGBA: gris casi negro translucido (material oscuro)
BORDE_TARJETA = (255, 255, 255, 34)
DESENFOQUE_TARJETA = True          # desenfoca el video detras de las tarjetas (efecto cristal)

_fuentes = {}
_sprites = OrderedDict()


def _bgr_a_rgb(c, a=255):
    return (int(c[2]), int(c[1]), int(c[0]), int(a))


def _fuente(tam, negrita=False):
    clave = (int(tam), negrita)
    if clave not in _fuentes:
        f = None
        for ruta in ((FUENTE_NEGRITA, FUENTE_REGULAR) if negrita else (FUENTE_REGULAR,)):
            if f is None and os.path.exists(ruta):
                try:
                    f = ImageFont.truetype(ruta, int(tam))
                except Exception:
                    f = None
        if f is None:
            try:
                f = ImageFont.load_default(size=int(tam))
            except Exception:
                f = None
        _fuentes[clave] = f
    return _fuentes[clave]


def _negrita_simulada(negrita):
    return negrita and not os.path.exists(FUENTE_NEGRITA)


def _cache(clave, fabrica):
    if clave in _sprites:
        _sprites.move_to_end(clave)
        return _sprites[clave]
    s = fabrica()
    _sprites[clave] = s
    if len(_sprites) > 200:
        _sprites.popitem(last=False)
    return s


def _medir(texto, tam, negrita):
    f = _fuente(tam, negrita)
    trazo = max(1, int(tam / 26)) if _negrita_simulada(negrita) else 0
    x0, y0, x1, y1 = f.getbbox(texto, stroke_width=trazo)
    return x1 - x0, y1 - y0, x0, y0, trazo


def _texto_pil(dib, xy, texto, tam, color, negrita):
    f = _fuente(tam, negrita)
    w, h, ox, oy, trazo = _medir(texto, tam, negrita)
    rgba = _bgr_a_rgb(color[:3], color[3] if len(color) > 3 else 255)
    dib.text((xy[0] - ox, xy[1] - oy), texto, font=f, fill=rgba, stroke_width=trazo, stroke_fill=rgba)


def _a_bgra(im):
    a = np.asarray(im.resize((max(1, im.width // SUPERMUESTREO), max(1, im.height // SUPERMUESTREO)),
                             Image.LANCZOS))
    return np.ascontiguousarray(a[..., [2, 1, 0, 3]])


def _tarjeta(lineas, esc, pad=(18, 12), sep=6, punto=False, alinear="izq", ancho_min=0):
    """Tarjeta redondeada translucida con lineas de texto. lineas: [(texto, tam, color_bgr, negrita)].
    punto: deja hueco a la izquierda para un punto o anillo que se dibuja aparte (devuelve su centro)."""
    clave = ("tarjeta", tuple(lineas), round(esc, 2), pad, sep, punto, alinear, ancho_min)

    def fabrica():
        S = SUPERMUESTREO

        def medida(t, tam, c, n):
            if t == "__puntos__":                      # fila de puntos de progreso: c = (hechos, total)
                d = tam * esc * S
                return (c[1] * d + (c[1] - 1) * d * 0.7, d, 0, 0, 0)
            return _medir(t, tam * esc * S, n)
        medidas = [medida(*l) for l in lineas]
        px, py = pad[0] * esc * S, pad[1] * esc * S
        hueco = (medidas[0][1] * 1.15 + 12 * esc * S) if punto else 0
        ancho = max(m[0] for m in medidas) + 2 * px + hueco
        ancho = max(ancho, ancho_min * esc * S)
        alto = sum(m[1] for m in medidas) + sep * esc * S * (len(lineas) - 1) + 2 * py
        im = Image.new("RGBA", (int(ancho), int(alto)), (0, 0, 0, 0))
        d = ImageDraw.Draw(im)
        r = min(alto / 2, 20 * esc * S)
        d.rounded_rectangle((0, 0, ancho - 1, alto - 1), radius=r, fill=FONDO_TARJETA,
                            outline=BORDE_TARJETA, width=max(1, int(S)))
        y = py
        for (t, tam, c, n), m in zip(lineas, medidas):
            if alinear == "centro":
                x = (ancho - m[0]) / 2
            else:
                x = px + hueco
            if t == "__puntos__":
                dm = tam * esc * S
                for i in range(c[1]):
                    cx0 = x + i * dm * 1.7
                    relleno = _bgr_a_rgb(C_OK) if i < c[0] else (255, 255, 255, 60)
                    d.ellipse((cx0, y, cx0 + dm, y + dm), fill=relleno)
            else:
                _texto_pil(d, (x, y), t, tam * esc * S, c, n)
            y += m[1] + sep * esc * S
        bgra = _a_bgra(im)
        centro_punto = (int((px + hueco / 2 - 3 * esc * S) / S), int(bgra.shape[0] / 2)) if punto else None
        return bgra, centro_punto

    return _cache(clave, fabrica)


def _texto_suelto(texto, tam, color, esc, negrita=True, sombra=True):
    """Texto sin tarjeta (para los "+10" que flotan), con una sombra suave para que se lea sobre el video."""
    clave = ("texto", texto, tam, tuple(color), round(esc, 2), negrita, sombra)

    def fabrica():
        S = SUPERMUESTREO
        w, h, ox, oy, trazo = _medir(texto, tam * esc * S, negrita)
        m = int(10 * esc * S)
        im = Image.new("RGBA", (int(w + 2 * m), int(h + 2 * m)), (0, 0, 0, 0))
        if sombra:
            capa = Image.new("RGBA", im.size, (0, 0, 0, 0))
            _texto_pil(ImageDraw.Draw(capa), (m, m + 2 * esc * S), texto, tam * esc * S, (0, 0, 0, 150), negrita)
            arr = np.asarray(capa).copy()
            arr[..., 3] = cv2.GaussianBlur(arr[..., 3], (0, 0), 4 * esc * S)
            im = Image.alpha_composite(im, Image.fromarray(arr))
        _texto_pil(ImageDraw.Draw(im), (m, m), texto, tam * esc * S, color, negrita)
        return _a_bgra(im)

    return _cache(clave, fabrica)


def _pegar(img, spr, x, y, alfa=1.0, cristal=False, esc=1.0):
    """Mezcla un sprite BGRA sobre la imagen en (x, y). cristal: desenfoca antes el video que queda detras."""
    h, w = spr.shape[:2]
    H, W = img.shape[:2]
    x, y = int(x), int(y)
    x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
    if x1 <= x0 or y1 <= y0 or alfa <= 0:
        return
    s = spr[y0 - y:y1 - y, x0 - x:x1 - x]
    roi = img[y0:y1, x0:x1]
    hh, ww = roi.shape[:2]
    if cristal and DESENFOQUE_TARJETA and hh > 8 and ww > 8:
        # fondo desenfocado detras de la tarjeta: se difumina a 1/4 de resolucion (barato) y se reescala
        peq = cv2.resize(roi, (max(2, ww // 4), max(2, hh // 4)), interpolation=cv2.INTER_AREA)
        peq = cv2.GaussianBlur(peq, (0, 0), max(1.0, 2.5 * esc))
        borroso = cv2.resize(peq, (ww, hh), interpolation=cv2.INTER_LINEAR)
        np.copyto(roi, borroso, where=(s[..., 3:4] > 8))
    # mezcla con transparencia en codigo nativo de OpenCV
    w2 = s[..., 3].astype(np.float32) * (alfa / 255.0)
    w1 = 1.0 - w2
    roi[:] = cv2.blendLinear(np.ascontiguousarray(roi), np.ascontiguousarray(s[..., :3]), w1, w2)


def _anillo(img, centro, radio, grosor, frac, color, alfa=1.0, pista=True):
    """Anillo de progreso (tipo anillos de Actividad): pista tenue y arco con extremos redondeados."""
    cx, cy = int(centro[0]), int(centro[1])
    r, g = int(radio), max(2, int(grosor))
    m = r + g + 2
    H, W = img.shape[:2]
    x0, y0, x1, y1 = max(0, cx - m), max(0, cy - m), min(W, cx + m), min(H, cy + m)
    if x1 <= x0 or y1 <= y0:
        return
    roi = img[y0:y1, x0:x1]
    capa = roi.copy()
    c = (cx - x0, cy - y0)
    if pista:
        cv2.circle(capa, c, r, (255, 255, 255), g, cv2.LINE_AA)
        roi[:] = cv2.addWeighted(capa, 0.18 * alfa, roi, 1 - 0.18 * alfa, 0)
        capa = roi.copy()
    frac = max(0.0, min(1.0, frac))
    if frac > 0.005:
        fin = -90 + 360 * frac
        cv2.ellipse(capa, c, (r, r), 0, -90, fin, color, g, cv2.LINE_AA)
        for ang in (-90, fin):
            p = (int(c[0] + r * math.cos(math.radians(ang))), int(c[1] + r * math.sin(math.radians(ang))))
            cv2.circle(capa, p, g // 2, color, -1, cv2.LINE_AA)
        roi[:] = cv2.addWeighted(capa, alfa, roi, 1 - alfa, 0)


def _ease_out(p):
    p = min(1.0, max(0.0, p))
    return 1 - (1 - p) ** 3


class GuiaFantasma:
    PREPARACION, IR, ESPERA_OBJ, VOLVER, ESPERA_REP, FIN = (
        "preparacion", "ir", "espera_objetivo", "volver", "espera_reposo", "fin")

    def __init__(self, rutina=RUTINA):
        self.rutina = rutina
        self.lock = threading.RLock()     # el hilo del video y el boton de reinicio la tocan a la vez
        self.reiniciar()

    def cambiar_rutina(self, rutina):
        with self.lock:
            self.rutina = rutina
            self.reiniciar()

    def reiniciar(self):
        with self.lock:
            self.paso = 0                 # ejercicio de la rutina
            self.rep = 0                  # repeticiones validas del ejercicio actual
            self.estado = self.PREPARACION
            self.t_estado = 0.0
            self.t_cond = 0.0
            self.progreso = 0.0           # 0 = reposo, 1 = objetivo (pose de la sombra)
            self.t_prev = None
            self.vivo = {}                # landmarks del fotograma actual (pixeles)
            self.segs = {}                # longitudes medidas en este fotograma
            self.buf = {}                 # longitudes en reposo (ultimos GUIA_N_CALIB fotogramas)
            self.calib = {}               # longitudes del paciente: mediana en reposo, fijas durante la rep
            self.tobillos_c = None
            self.perfil = False           # la persona esta de lado a la camara
            self.mira = 1.0               # de perfil: +1 mira hacia la derecha de la imagen, -1 a la izquierda
            self.signo_i = 1.0            # lado de la imagen donde cae el hombro izquierdo
            self.puntos = 0
            self.notas = []               # nota 1-10 de cada repeticion valida
            self.anuladas = 0
            self.ultima_nota = None       # numero, o "X" si la repeticion no fue valida
            self.t_ultima_nota = -10.0
            self.avisos_rep = set()       # avisos vistos durante la repeticion en curso
            self.extremo = None           # angulo maximo (abduccion) o minimo (rodilla) de la repeticion
            self.cuenta_ini = 0           # repeticiones del motor al empezar la repeticion
            self.t_inicio_rep = 0.0
            self.t_contada = -10.0        # cuando el motor sumo la ultima repeticion (verde de confirmacion)
            self.ultima_linea = None      # ultima linea del registro del motor ya leida
            self.motivo, self.t_motivo = None, -10.0   # por que el motor no conto la ultima
            self.aviso = None
            self.color = C_GUIA
            self.mensaje = ""
            self.visible = False
            self.valor = None             # angulo actual del paciente (brazo o rodilla)
            self.efectos = []             # pulsos de luz y textos que suben
            self.msg_prev, self.t_msg = None, 0.0

    # ------------------------------------------------------------ ejercicio actual
    @property
    def ejercicio(self):
        return self.rutina[self.paso] if self.paso < len(self.rutina) else (None, None, 0)

    def _lados(self):
        tipo, lado, _ = self.ejercicio
        return ("izquierdo", "derecho") if lado == "ambos" else (lado,)

    # ------------------------------------------------------------ lectura del motor
    def _cuenta(self, m):
        return m.repeticiones_abduccion if self.ejercicio[0] == "abduccion" else m.repeticiones_sentadilla

    def _medida(self, m):
        """Valor del paciente para el ejercicio: angulo del brazo (min de los lados) o de la rodilla."""
        tipo = self.ejercicio[0]
        if tipo == "abduccion":
            angs = [m.abd_info[l]["ang"] for l in self._lados()]
            return None if any(a is None for a in angs) else min(angs)
        if tipo == "sentadilla":
            return m.fsm_sent.angulo
        return None

    def _en_reposo(self, m, v):
        """Reposo segun el motor: brazo por debajo de ABD_ABAJO, o rodilla por encima de su umbral de pie."""
        if v is None:
            return False
        if self.ejercicio[0] == "abduccion":
            return v < _re.ABD_ABAJO
        return v >= m.fsm_sent.arriba

    def _objetivo_aceptado(self, m):
        """El motor da la posicion por buena: el brazo de la rutina ha llegado arriba (el motor la deja
        pendiente y la cuenta al bajar), o la sentadilla cumple el tiempo minimo abajo."""
        if self.ejercicio[0] == "abduccion":
            return all(m.fsm_abd[l].estado == "arriba" and m.abd_pend.get(l) is not None
                       for l in self._lados())
        return m.fsm_sent.estado in ("abajo", "subiendo")

    def _invalidez(self, m):
        """Motivo por el que el motor ya NO va a contar la repeticion en curso (None si sigue valida).

        Lee las mismas marcas que el motor revisa al cerrar la repeticion (rehab_engine._fase_2 y
        _contar_abduccion): si alguna esta puesta, al terminar la anulara. Asi la sombra no se pone
        verde en una repeticion que no va a sumar."""
        comp = getattr(_re, "VALIDAR_COMPENSACIONES", True)
        if self.ejercicio[0] == "abduccion":
            for l in self._lados():
                if comp and m.abd_comp.get(l):
                    return f"compensacion ({_ascii(m.abd_comp[l])})"
            return None
        if getattr(m, "pie_alto_mal", False):
            return AVISO_PIE
        if getattr(m, "zancada_mal", False):
            return AVISO_ZANCADA
        if getattr(m, "tronco_mal", False):
            return AVISO_TRONCO
        if comp and getattr(m, "sent_comp", ""):
            return f"compensacion ({_ascii(m.sent_comp)})"
        minimo = m.fsm_sent.minimo
        if minimo is not None and minimo < _re.SENT_MIN_VALIDO:
            return AVISO_PROFUNDA
        return None

    def _otro_brazo_arriba(self, m):
        """En la abduccion, el brazo que NO toca esta arriba (se ha equivocado de brazo)."""
        if self.ejercicio[0] != "abduccion":
            return False
        otros = [l for l in ("izquierdo", "derecho") if l not in self._lados()]
        return any(m.fsm_abd[l].estado == "arriba" for l in otros)

    def _rep_terminada(self, m, v):
        """El motor ha cerrado la repeticion: brazo rearmado abajo, o de nuevo de pie."""
        if self.ejercicio[0] == "abduccion":
            return v is not None and v < _re.ABD_ABAJO and \
                all(m.fsm_abd[l].estado != "arriba" for l in self._lados())
        return m.fsm_sent.estado not in SENT_EN_REP

    def _avisos(self, m, v):
        """Lista de compensaciones activas (texto para el paciente), con los umbrales del motor."""
        av = []
        eps = getattr(m, "episodios", {})
        # inclinacion del tronco: el mismo episodio que apunta el motor en la fase 3 (umbral y delta t suyos);
        # la direccion la da el motor (f3_del: + delante / - atras, f3_lado: hacia un lado)
        if any(eps.get(k, {}).get("activo") for k in ("tronco", "lateral")):
            d, lado = getattr(m, "f3_del", None), getattr(m, "f3_lado", None) or 0.0
            if d is not None and d < -getattr(_re, "TRONCO_MAX_ATRAS", 15.0):
                av.append(AVISO_ATRAS)
            elif d is not None and d > lado:
                av.append(AVISO_TRONCO)
            else:
                av.append(AVISO_LATERAL)
        if self.ejercicio[0] == "abduccion":
            # (ya no se pide ponerse de frente: el motor mide la abduccion con cualquier orientacion)
            for l in self._lados():
                info = m.abd_info[l]
                if info["ang"] is not None and info["ang"] > _re.ABD_ABAJO:
                    if not info["en_plano"]:
                        av.append(AVISO_PLANO)
                    if info.get("codo") is not None and info["codo"] < _re.CODO_EXTENDIDO_MIN:
                        av.append(AVISO_CODO)
            if any(eps.get(k, {}).get("activo") for k in ("hombro_I", "hombro_D")):
                av.append(AVISO_HOMBRO)
        elif m.fsm_sent.estado in SENT_EN_REP:
            # tronco, zancada y pie solo se miden durante la repeticion: fuera de ella el motor
            # guarda el ultimo valor y no hay que hacerle caso (dejaba la guia atascada)
            tronco = getattr(m, "tronco", None)
            if tronco is not None:
                lim = _re.SENT_TRONCO_MAX if getattr(m, "tronco_modo", None) == "perfil" \
                    else getattr(_re, "SENT_TRONCO_FRENTE_MAX", _re.SENT_TRONCO_MAX)
                if tronco > lim:
                    av.append(AVISO_TRONCO)
            z = getattr(m, "zancada", None)
            if z is not None and z > _re.SENT_ZANCADA_MAX:
                av.append(AVISO_ZANCADA)
            p = getattr(m, "pie_alto", None)
            if p is not None and p > _re.SENT_PIE_LEVANTADO_MAX:
                av.append(AVISO_PIE)
            if v is not None and v < _re.SENT_MIN_VALIDO:
                av.append(AVISO_PROFUNDA)
        return list(dict.fromkeys(av))

    def _leer_registro(self, m, t):
        """Lee las lineas nuevas del registro del motor y guarda el motivo si no conto una repeticion."""
        lineas = list(m.log)
        if not lineas:
            return
        nuevas = lineas
        if self.ultima_linea in lineas:
            nuevas = lineas[len(lineas) - 1 - lineas[::-1].index(self.ultima_linea) + 1:]
        self.ultima_linea = lineas[-1]
        for linea in nuevas:
            texto = linea.split("  ", 1)[-1]
            if "no valida" in texto or "anulada" in texto or "no es sentadilla" in texto:
                motivo = texto.split(":", 1)[-1].strip() if ":" in texto else texto
                self.motivo, self.t_motivo = _ascii(motivo), t

    # ------------------------------------------------------------ proporciones del paciente
    def _medir_cuerpo(self, lm, w, h, dt, reposo):
        """Landmarks del fotograma en pixeles y longitudes del paciente.

        La sombra se construye con los puntos de ESTE fotograma (casi sin suavizar). Las longitudes
        de los segmentos se miden solo en reposo (de pie, brazos abajo: se ven enteros) y se guarda la
        mediana del ultimo segundo; durante la repeticion quedan fijas, para que no encojan cuando el
        brazo apunta a la camara o las piernas se doblan.
        """
        vivo = {}
        for nombre, i in PUNTOS_GUIA.items():
            if lm[i].visibility >= (0.3 if i in (23, 24) else 0.5):
                p = np.array([lm[i].x * w, lm[i].y * h])
                vivo[nombre] = _ema(self.vivo.get(nombre), p, dt, GUIA_TAU_VIVO_S)
        # basta con un hombro y una cadera: de perfil el lado del fondo casi no se ve
        hombros = [vivo[k] for k in ("hombroI", "hombroD") if k in vivo]
        caderas = [vivo[k] for k in ("caderaI", "caderaD") if k in vivo]
        if not hombros or not caderas:
            return False
        self.vivo = vivo
        hombros_c, caderas_c = np.mean(hombros, axis=0), np.mean(caderas, axis=0)
        self.hombros_c = hombros_c
        tobillos = [vivo[k] for k in ("tobilloI", "tobilloD") if k in vivo]
        self.tobillos_c = np.mean(tobillos, axis=0) if tobillos else None
        if "hombroI" in vivo and "hombroD" in vivo:
            self.signo_i = 1.0 if vivo["hombroI"][0] >= vivo["hombroD"][0] else -1.0
        # de perfil: falta un lado, o los hombros se ven muy estrechos frente al tronco
        alto_tronco = float(np.linalg.norm(hombros_c - caderas_c))
        ancho_h = float(np.linalg.norm(vivo["hombroI"] - vivo["hombroD"])) \
            if "hombroI" in vivo and "hombroD" in vivo else 0.0
        self.perfil = ancho_h < 0.35 * alto_tronco
        if self.perfil:
            self.mira = self._hacia_donde_mira(lm, w, vivo, hombros_c, alto_tronco) or self.mira

        def media(pares):
            vals = [np.linalg.norm(vivo[a] - vivo[b]) for a, b in pares if a in vivo and b in vivo]
            return float(np.mean(vals)) if vals else None
        segs = {"tronco": alto_tronco,
                "hombros": media((("hombroI", "hombroD"),)),
                "caderas": media((("caderaI", "caderaD"),)),
                "brazo": media((("hombroI", "codoI"), ("hombroD", "codoD"))),
                "antebrazo": media((("codoI", "munecaI"), ("codoD", "munecaD"))),
                "muslo": media((("caderaI", "rodillaI"), ("caderaD", "rodillaD"))),
                "tibia": media((("rodillaI", "tobilloI"), ("rodillaD", "tobilloD")))}
        if "cabeza" in vivo:
            segs["cabeza"] = float(np.linalg.norm(vivo["cabeza"] - hombros_c))
        self.segs = {k: v for k, v in segs.items() if v}
        if self.perfil:   # de perfil las anchuras no son reales: no se aprenden
            self.segs.pop("hombros", None)
            self.segs.pop("caderas", None)
        if reposo:
            for k, v in self.segs.items():
                b = self.buf.setdefault(k, [])
                b.append(v)
                del b[:-GUIA_N_CALIB]
                self.calib[k] = float(np.median(b))
        return True

    @staticmethod
    def _hacia_donde_mira(lm, w, vivo, hombros_c, alto_tronco):
        """De perfil, +1 si la persona mira a la derecha de la imagen y -1 a la izquierda (None si no se sabe).
        Por orden de fiabilidad: punta del pie frente al talon, rodilla adelantada al doblarla, nariz."""
        votos = []
        for talon, punta in ((29, 31), (30, 32)):
            if lm[talon].visibility >= 0.5 and lm[punta].visibility >= 0.5:
                dx = (lm[punta].x - lm[talon].x) * w
                if abs(dx) > 0.05 * alto_tronco:
                    votos.append(1.0 if dx > 0 else -1.0)
        if votos:
            return 1.0 if sum(votos) > 0 else -1.0
        for rod, tob in (("rodillaI", "tobilloI"), ("rodillaD", "tobilloD")):
            if rod in vivo and tob in vivo:
                dx = vivo[rod][0] - vivo[tob][0]
                if abs(dx) > 0.15 * alto_tronco:          # solo con la rodilla claramente doblada
                    return 1.0 if dx > 0 else -1.0
        if "cabeza" in vivo and abs(vivo["cabeza"][0] - hombros_c[0]) > 0.05 * alto_tronco:
            return 1.0 if vivo["cabeza"][0] > hombros_c[0] else -1.0
        return None

    def _p(self, nombre):
        tronco = self.calib.get("tronco") or self.segs.get("tronco") or 100.0
        return self.calib.get(nombre) or self.segs.get(nombre) or PROP_DEF[nombre] * tronco

    # ------------------------------------------------------------ pose de la sombra
    def _pose(self, abd_i, abd_d, flex):
        """Puntos 2D de la sombra (vista frontal) con los tobillos en el origen."""
        tronco, hombros, caderas = self._p("tronco"), self._p("hombros"), self._p("caderas")
        muslo, tibia = self._p("muslo"), self._p("tibia")
        f = math.radians(flex)
        beta = 0.35 * f                     # la tibia se inclina hacia delante
        alfa = f - beta                     # el muslo, el resto
        incl = 0.45 * alfa                  # el tronco acompana inclinandose
        s = self.signo_i
        P = {}
        for lado, k in (("I", s), ("D", -s)):
            P["tobillo" + lado] = np.array([k * caderas * 0.55, 0.0])
            P["rodilla" + lado] = P["tobillo" + lado] + np.array([k * 0.08 * tibia * math.sin(beta),
                                                                 -tibia * math.cos(beta)])
            P["cadera" + lado] = np.array([k * caderas / 2, P["rodilla" + lado][1] - muslo * math.cos(alfa)])
        cad_c = (P["caderaI"] + P["caderaD"]) / 2
        hom_c = cad_c + np.array([0.0, -tronco * math.cos(incl)])
        P["cabeza"] = hom_c + np.array([0.0, -self._p("cabeza")])
        for lado, k, abd in (("I", s, abd_i), ("D", -s, abd_d)):
            P["hombro" + lado] = hom_c + np.array([k * hombros / 2, 0.0])
            a = math.radians(abd)
            d = np.array([k * math.sin(a), math.cos(a)])
            P["codo" + lado] = P["hombro" + lado] + self._p("brazo") * d
            P["muneca" + lado] = P["codo" + lado] + self._p("antebrazo") * d
        return P

    def _pose_abduccion(self, p):
        """La sombra es el esqueleto real del paciente en este fotograma; solo el brazo de la rutina
        se dibuja en la posicion objetivo, saliendo de su hombro y medido respecto a su tronco."""
        V = dict(self.vivo)
        if not all(k in V for k in ("hombroI", "hombroD", "caderaI", "caderaD")):
            return V      # de perfil la abduccion no se mide: sin brazo objetivo
        hs = (V["hombroI"] + V["hombroD"]) / 2
        abajo = (V["caderaI"] + V["caderaD"]) / 2 - hs
        abajo = abajo / max(1e-6, np.linalg.norm(abajo))
        a = math.radians(ABD_SOMBRA * p)
        for l in self._lados():
            L = "I" if l == "izquierdo" else "D"
            hombro = V["hombro" + L]
            lat = hombro - hs
            lat = lat - np.dot(lat, abajo) * abajo
            lat = lat / max(1e-6, np.linalg.norm(lat))
            d = math.cos(a) * abajo + math.sin(a) * lat
            V["codo" + L] = hombro + self._p("brazo") * d
            V["muneca" + L] = V["codo" + L] + self._p("antebrazo") * d
        return V

    def _pose_perfil(self, flex):
        """Sentadilla vista de lado, en pixeles, apoyada en el tobillo real: rodilla hacia delante,
        cadera hacia atras, tronco algo inclinado y brazos al frente (mirando hacia self.mira)."""
        f, m = math.radians(flex), self.mira
        beta = 0.40 * f                     # inclinacion de la tibia hacia delante
        alfa = f - beta                     # del muslo
        incl = 0.45 * alfa                  # del tronco
        brazos = math.radians(70.0) * min(1.0, flex / max(1.0, FLEX_SOMBRA))   # brazos al frente al bajar
        tob = self.tobillos_c if self.tobillos_c is not None else self.hombros_c + np.array([0.0, 3 * self._p("tronco")])
        rod = tob + self._p("tibia") * np.array([m * math.sin(beta), -math.cos(beta)])
        cad = rod + self._p("muslo") * np.array([-m * math.sin(alfa), -math.cos(alfa)])
        eje = np.array([m * math.sin(incl), -math.cos(incl)])
        hom = cad + self._p("tronco") * eje
        cab = hom + self._p("cabeza") * eje
        d_brazo = np.array([m * math.sin(brazos), math.cos(brazos)])
        codo = hom + self._p("brazo") * d_brazo
        mun = codo + self._p("antebrazo") * d_brazo
        P = {"cabeza": cab}
        for L, dx in (("I", 0.0), ("D", 3.0)):          # las dos piernas casi superpuestas
            o = np.array([dx, 0.0])
            P.update({"tobillo" + L: tob + o, "rodilla" + L: rod + o, "cadera" + L: cad + o,
                      "hombro" + L: hom + o, "codo" + L: codo + o, "muneca" + L: mun + o})
        return P

    def _pose_actual(self):
        tipo, lado, _ = self.ejercicio
        p = _suave(self.progreso)
        if tipo == "abduccion":
            return self._pose_abduccion(p), tipo
        if self.perfil:
            return self._pose_perfil(FLEX_SOMBRA * p), "sentadilla_perfil"
        return self._pose(0.0, 0.0, FLEX_SOMBRA * p), tipo

    # ------------------------------------------------------------ avance
    def _cambiar(self, estado):
        self.estado, self.t_estado, self.t_cond = estado, 0.0, 0.0

    def _nota(self):
        """Nota 1-10 de una repeticion valida: 2 puntos menos por cada aviso que haya salido y, en la
        abduccion, hasta 2 menos si se quedo entre el umbral del motor y su objetivo (80-90 grados)."""
        nota = 10.0 - 2.0 * len(self.avisos_rep)
        if self.ejercicio[0] == "abduccion" and self.extremo is not None:
            nota -= min(2.0, max(0.0, ABD_SOMBRA - self.extremo) / 5.0)
        return int(round(min(10.0, max(1.0, nota))))

    def _empezar_rep(self, m, t):
        self.avisos_rep, self.extremo = set(), None
        self.cuenta_ini = self._cuenta(m)
        self.t_inicio_rep = t
        self._cambiar(self.IR)

    def _no_valida(self, t, motivo_reciente):
        """El motor no ha contado la repeticion: X y se repite la misma."""
        self.anuladas += 1
        self.ultima_nota, self.t_ultima_nota = "X", t
        if not motivo_reciente:
            self.motivo, self.t_motivo = "repeticion no valida", t

    def actualizar(self, motor, landmarks, t, w, h):
        with self.lock:
            self._actualizar(motor, landmarks, t, w, h)

    def _actualizar(self, motor, landmarks, t, w, h):
        dt = 0.0 if self.t_prev is None else max(0.0, t - self.t_prev)
        self.t_prev = t
        self.t_estado += dt
        lm = landmarks.landmark if hasattr(landmarks, "landmark") else landmarks
        if self.estado == self.FIN or self.paso >= len(self.rutina):
            self.visible = lm is not None and self._medir_cuerpo(lm, w, h, dt, False)
            self.estado, self.color = self.FIN, C_OK
            self.mensaje = "Pulsa «Reiniciar sesión» para volver a jugar"
            return
        tipo = self.ejercicio[0]
        with motor.lock:
            v = self._medida(motor)
            self.valor = v
            avisos = self._avisos(motor, v)
            en_rep = self._en_reposo(motor, v)
            aceptado = self._objetivo_aceptado(motor)
            terminada = self._rep_terminada(motor, v)
            otro_brazo = self._otro_brazo_arriba(motor)
            cuenta = self._cuenta(motor)
            invalida = self._invalidez(motor) if self.estado in (self.ESPERA_OBJ, self.VOLVER, self.ESPERA_REP) else None
            self._leer_registro(motor, t)
        # las longitudes se miden solo en reposo y sin compensaciones (de pie, brazos abajo)
        reposo = en_rep and not avisos and self.estado in (self.PREPARACION, self.ESPERA_REP, self.IR) \
            and self.progreso < 0.05
        self.visible = lm is not None and self._medir_cuerpo(lm, w, h, dt, reposo)
        if not self.visible:
            self.mensaje, self.color = "Colócate de cuerpo entero delante de la cámara", C_ROJO
            return
        if self.estado in (self.IR, self.ESPERA_OBJ, self.VOLVER) and v is not None:
            if tipo == "abduccion":
                self.extremo = v if self.extremo is None else max(self.extremo, v)
            else:
                self.extremo = v if self.extremo is None else min(self.extremo, v)
        if self.estado in (self.IR, self.ESPERA_OBJ):
            self.avisos_rep.update(avisos)
        self.aviso = avisos[0] if avisos else None
        motivo_reciente = self.motivo if t - self.t_motivo < GUIA_T_MOTIVO_S else None
        nombre = "abducción" if tipo == "abduccion" else "sentadilla"

        if self.estado == self.PREPARACION:
            self.progreso = 0.0
            self.t_cond = self.t_cond + dt if (en_rep and not avisos) else 0.0
            if avisos:
                self.color, self.mensaje = C_AVISO, avisos[0]
            else:
                lado = self.ejercicio[1]
                self.color = C_GUIA
                self.mensaje = f"Prepárate: {nombre}" + (f" ({lado})" if lado else "") + \
                    ("" if en_rep else (" - baja el brazo" if tipo == "abduccion" else " - ponte de pie"))
            if self.t_cond >= GUIA_T_PREPARACION_S:
                self._empezar_rep(motor, t)
        elif self.estado == self.IR:
            # pausa breve en reposo antes de que la sombra arranque
            t_mov = max(0.0, self.t_estado - GUIA_T_ENTRE_REPS_S)
            self.progreso = min(1.0, t_mov / GUIA_T_ANIM_S)
            if self.t_estado < GUIA_T_ENTRE_REPS_S:
                self.mensaje = "Prepárate…"
            else:
                self.mensaje = "Sube el brazo como la sombra" if tipo == "abduccion" else "Baja como la sombra"
            self.color = C_GUIA
            if self.progreso >= 1.0:
                self._cambiar(self.ESPERA_OBJ)
        elif self.estado == self.ESPERA_OBJ:
            self.progreso = 1.0
            rechazada = motivo_reciente and self.t_motivo >= self.t_inicio_rep
            if cuenta > self.cuenta_ini:
                # el motor ya la ha sumado (p. ej. subio antes de que la sombra lo viera): se cierra
                self._cerrar_rep(motor, t, True, motivo_reciente)
            elif aceptado and not invalida:
                # verde solo si el motor la da por buena Y no tiene ninguna marca que la vaya a anular
                self.color, self.mensaje = C_OK, "¡Muy bien!"
                p = self._punto_clave()
                if p is not None:
                    self._efecto("pulso", t, pos=p, r0=max(22.0, self._p("tronco") * 0.30), color=C_OK, dur=0.9)
                self._cambiar(self.VOLVER)
            elif rechazada and en_rep:
                # el motor no la dio por buena y el paciente ya ha vuelto al reposo: se repite
                self._no_valida(t, motivo_reciente)
                self._empezar_rep(motor, t)
            elif aceptado and invalida:
                # ha llegado abajo/arriba, pero el motor la va a anular: naranja, nunca verde
                self.color, self.mensaje = C_AVISO, "No contará: " + _motivo_amable(invalida)
            elif avisos:
                self.color, self.mensaje = C_AVISO, avisos[0]
            elif motivo_reciente:
                self.color, self.mensaje = C_AVISO, "No vale: " + _motivo_amable(motivo_reciente)
            else:
                self.color = C_GUIA
                if otro_brazo:
                    self.mensaje = f"Con el brazo {self.ejercicio[1]}"   # esta subiendo el otro brazo
                else:
                    self.mensaje = "Un poco más arriba" if tipo == "abduccion" else "Un poco más abajo y aguanta"
        elif self.estado == self.VOLVER:
            self.progreso = max(0.0, 1.0 - self.t_estado / GUIA_T_ANIM_S)
            if invalida:
                # algo la ha invalidado durante la vuelta: se quita el verde en el acto
                self.color, self.mensaje = C_AVISO, "No contará: " + _motivo_amable(invalida)
            elif self.t_estado < GUIA_T_VERDE_S:
                self.color, self.mensaje = C_OK, "¡Muy bien!"   # el verde se mantiene un momento
            else:
                self.color, self.mensaje = C_GUIA, "Ahora vuelve despacio"
            if self.progreso <= 0.0:
                self._cambiar(self.ESPERA_REP)
        elif self.estado == self.ESPERA_REP:
            self.progreso = 0.0
            if invalida:
                self.color, self.mensaje = C_AVISO, "No contará: " + _motivo_amable(invalida)
            else:
                self.color = C_GUIA
                self.mensaje = "Vuelve a la posición inicial" if tipo == "abduccion" else "Ponte de pie del todo"
            if terminada:
                self._cerrar_rep(motor, t, cuenta > self.cuenta_ini, motivo_reciente)

        # confirmacion: el motor acaba de sumar la repeticion -> verde mientras la sombra espera en reposo
        if t - self.t_contada < GUIA_T_VERDE_S and self.estado in (self.PREPARACION, self.IR) \
                and self.progreso <= 0.0 and not avisos:
            self.color, self.mensaje = C_OK, "¡Contada!"

    def _cerrar_rep(self, motor, t, valida, motivo_reciente):
        """Cierra la repeticion con lo que ha decidido el motor (valida = su contador ha subido)."""
        if valida:
            nota = self._nota()
            self.notas.append(nota)
            self.puntos += nota
            self.ultima_nota = nota
            self.rep += 1
            self.t_ultima_nota = self.t_contada = t
            p = self._punto_clave()
            if p is not None:
                self._efecto("texto", t, pos=p, texto=f"+{nota}", color=C_OK, dur=1.3)
                self._efecto("pulso", t, pos=p, r0=max(22.0, self._p("tronco") * 0.30), color=C_OK, dur=0.8)
        else:
            self._no_valida(t, motivo_reciente)   # se repite la misma repeticion
            p = self._punto_clave()
            if p is not None:
                self._efecto("pulso", t, pos=p, r0=max(22.0, self._p("tronco") * 0.30), color=C_ROJO, dur=0.7)
        if self.rep >= self.ejercicio[2]:
            self.paso, self.rep = self.paso + 1, 0
            if self.paso < len(self.rutina):
                self._cambiar(self.PREPARACION)
            else:
                self._cambiar(self.FIN)
                self.mensaje = "Pulsa «Reiniciar sesión» para volver a jugar"
        else:
            self._empezar_rep(motor, t)

    # ------------------------------------------------------------ dibujo (estilo tipo Apple)
    def _efecto(self, tipo, t, **datos):
        self.efectos.append(dict(tipo=tipo, t0=t, **datos))
        self.efectos = self.efectos[-8:]

    def _punto_clave(self):
        """Articulacion que trabaja: hombro del brazo de la rutina, o rodilla (cadera si no se ve)."""
        V = self.vivo
        if self.ejercicio[0] == "abduccion":
            for l in self._lados():
                p = V.get("hombro" + ("I" if l == "izquierdo" else "D"))
                if p is not None:
                    return p
            return None
        rod = [V[k] for k in ("rodillaI", "rodillaD") if k in V]
        if rod:
            return np.mean(rod, axis=0)
        cad = [V[k] for k in ("caderaI", "caderaD") if k in V]
        return np.mean(cad, axis=0) if cad else None

    def _fraccion_objetivo(self):
        """0..1: cuanto le falta al paciente para el objetivo (lo que llena el anillo)."""
        v = self.valor
        if v is None:
            return 0.0
        if self.ejercicio[0] == "abduccion":
            return v / max(1.0, ABD_SOMBRA)
        pie = 175.0
        return (pie - v) / max(1.0, pie - SENT_SOMBRA_RODILLA)

    def dibujar(self, img):
        with self.lock:
            if _PIL_OK and _fuente(20) is not None:
                self._dibujar(img)
            else:
                self._dibujar_basico(img)

    def _figura_q(self):
        P, tipo = self._pose_actual()
        if tipo in ("abduccion", "sentadilla_perfil"):         # ya en pixeles
            return {k: v.astype(int) for k, v in P.items()}
        ancla = self.tobillos_c if self.tobillos_c is not None else \
            self.hombros_c - (P["hombroI"] + P["hombroD"]) / 2
        return {k: (v + ancla).astype(int) for k, v in P.items()}

    def _dibujar_sombra(self, img, Q, color, exito):
        """Sombra de cristal: halo de luz difuminado (suma) + trazo claro y redondeado encima."""
        h, w = img.shape[:2]
        grosor = max(5, int(self._p("tronco") * 0.085))
        r_cabeza = int(self._p("cabeza") * 0.42)
        huesos = [("hombroI", "hombroD"), ("caderaI", "caderaD"), ("hombroI", "caderaI"), ("hombroD", "caderaD")]
        for l in ("I", "D"):
            huesos += [("hombro" + l, "codo" + l), ("codo" + l, "muneca" + l),
                       ("cadera" + l, "rodilla" + l), ("rodilla" + l, "tobillo" + l)]
        pts = np.array(list(Q.values()))
        margen = r_cabeza + grosor * 6
        x0, y0 = np.maximum(pts.min(axis=0) - margen, 0)
        x1, y1 = np.minimum(pts.max(axis=0) + margen, [w - 1, h - 1])
        if x1 <= x0 or y1 <= y0:
            return
        zona = img[y0:y1 + 1, x0:x1 + 1]
        o = np.array([x0, y0])
        R = {k: tuple(int(c) for c in (v - o)) for k, v in Q.items()}

        def trazar(lienzo, col, g):
            for a_, b_ in huesos:
                if a_ in R and b_ in R:
                    cv2.line(lienzo, R[a_], R[b_], col, g, cv2.LINE_AA)
            for k_, q in R.items():
                if k_ != "cabeza":
                    cv2.circle(lienzo, q, int(g * 0.62), col, -1, cv2.LINE_AA)
            if "cabeza" in R:
                if "hombroI" in R and "hombroD" in R:
                    cuello = tuple(((np.array(R["hombroI"]) + np.array(R["hombroD"])) // 2).tolist())
                    cv2.line(lienzo, cuello, R["cabeza"], col, g, cv2.LINE_AA)
                cv2.circle(lienzo, R["cabeza"], r_cabeza + (g - grosor) // 2, col, -1, cv2.LINE_AA)

        # halo: el trazo grueso en el color de estado, dibujado y difuminado a 1/4 de resolucion y sumado
        hz, wz = zona.shape[:2]
        k = 4
        peq = np.zeros((max(1, hz // k), max(1, wz // k), 3), np.uint8)
        intensidad = 0.95 if exito else 0.6
        col_halo = tuple(int(c * intensidad) for c in color)
        g_halo = max(1, int(grosor * (2.6 if exito else 2.0) / k))
        for a_, b_ in huesos:
            if a_ in R and b_ in R:
                cv2.line(peq, (R[a_][0] // k, R[a_][1] // k), (R[b_][0] // k, R[b_][1] // k), col_halo, g_halo, cv2.LINE_AA)
        if "cabeza" in R:
            cv2.circle(peq, (R["cabeza"][0] // k, R["cabeza"][1] // k), max(1, r_cabeza // k + g_halo // 2), col_halo, -1, cv2.LINE_AA)
        peq = cv2.GaussianBlur(peq, (0, 0), 3.0 if exito else 2.2)
        cv2.add(zona, cv2.resize(peq, (wz, hz), interpolation=cv2.INTER_LINEAR), dst=zona)
        # nucleo: el color mezclado con blanco (aspecto de luz), translucido
        claro = tuple(int(c * 0.55 + 255 * 0.45) for c in color)
        capa = zona.copy()
        trazar(capa, claro, grosor)
        alfa = 0.92 if exito else 0.62
        cv2.addWeighted(capa, alfa, zona, 1 - alfa, 0, dst=zona)

    def _dibujar(self, img):
        h, w = img.shape[:2]
        esc = max(0.8, h / 600)          # 480 px -> 0.8, 720 px -> 1.2
        ahora = self.t_prev if self.t_prev is not None else 0.0
        tipo, lado, n = self.ejercicio

        # ---------------- pantalla final
        if self.estado == self.FIN:
            img[:] = (img * 0.42).astype(np.uint8)
            media = sum(self.notas) / len(self.notas) if self.notas else 0
            estrellas = 3 if media >= 8 else 2 if media >= 5 else 1
            tit, _ = _tarjeta([("Sesión completada", 30, C_TEXTO, True),
                               ("¡Buen trabajo! Así va tu rehabilitación de hoy", 15, C_SECUNDARIO, False)],
                              esc, pad=(34, 26), sep=10, alinear="centro", ancho_min=460)
            x = (w - tit.shape[1]) // 2
            y = int(h * 0.22)
            _pegar(img, tit, x, y, cristal=True, esc=esc)
            # tres estadisticas en tarjetas
            stats = [(str(self.puntos), "puntos"), (f"{media:.1f}", "nota media"), (str(self.anuladas), "repetidas")]
            spr = [_tarjeta([(v, 30, C_TEXTO, True), (k, 13, C_SECUNDARIO, False)], esc, pad=(22, 14),
                            alinear="centro", ancho_min=130)[0] for v, k in stats]
            total = sum(s.shape[1] for s in spr) + 12 * esc * (len(spr) - 1)
            xs = (w - total) / 2
            ys = y + tit.shape[0] + 14 * esc
            for s in spr:
                _pegar(img, s, xs, ys, cristal=True, esc=esc)
                xs += s.shape[1] + 12 * esc
            # anillos de logro (uno por estrella)
            cy = int(ys + spr[0].shape[0] + 44 * esc)
            for i in range(3):
                cx = int(w / 2 + (i - 1) * 56 * esc)
                _anillo(img, (cx, cy), 18 * esc, 7 * esc, 1.0 if i < estrellas else 0.0, C_OK)
            pie, _ = _tarjeta([(self.mensaje, 15, C_SECUNDARIO, False)], esc, pad=(18, 10))
            _pegar(img, pie, (w - pie.shape[1]) // 2, h - pie.shape[0] - 24 * esc, cristal=True, esc=esc)
            return

        exito = self.color == C_OK
        # ---------------- sombra
        if self.visible:
            self._dibujar_sombra(img, self._figura_q(), self.color, exito)

        # ---------------- anillo de progreso en la articulacion que trabaja
        clave = self._punto_clave() if self.visible else None
        if clave is not None and self.estado in (self.IR, self.ESPERA_OBJ, self.VOLVER):
            radio = max(22.0, self._p("tronco") * 0.30)
            frac = self._fraccion_objetivo()
            col = C_OK if (exito or frac >= 1.0) else C_GUIA
            _anillo(img, clave, radio, max(5, radio * 0.17), frac, col)
            if self.valor is not None:
                etiqueta = f"{self.valor:.0f}°"
                spr = _texto_suelto(etiqueta, 18, C_TEXTO, esc)
                _pegar(img, spr, clave[0] - spr.shape[1] / 2, clave[1] - radio - spr.shape[0] - 2 * esc)

        # ---------------- efectos (pulso de luz y puntos que suben)
        vivos = []
        for e in self.efectos:
            dt = ahora - e["t0"]
            if dt < 0 or dt > e.get("dur", 1.2):
                continue
            vivos.append(e)
            p = dt / e.get("dur", 1.2)
            if e["tipo"] == "pulso" and e.get("pos") is not None:
                for k_, retardo in ((0, 0.0), (1, 0.18)):
                    q = max(0.0, (p - retardo) / (1 - retardo))
                    if q <= 0:
                        continue
                    r = e["r0"] * (1.0 + 1.6 * _ease_out(q))
                    _anillo(img, e["pos"], r, max(2, e["r0"] * 0.12 * (1 - q)), 1.0, e["color"],
                            alfa=(1 - q) * (0.9 if k_ == 0 else 0.5), pista=False)
            elif e["tipo"] == "texto" and e.get("pos") is not None:
                spr = _texto_suelto(e["texto"], 34, e["color"], esc)
                sube = 70 * esc * _ease_out(p)
                alfa = min(1.0, p * 6) * (1 - max(0.0, (p - 0.6) / 0.4))
                _pegar(img, spr, e["pos"][0] - spr.shape[1] / 2, e["pos"][1] - spr.shape[0] - sube, alfa=alfa)
        self.efectos = vivos

        # ---------------- tarjeta del ejercicio (arriba a la izquierda) con puntos de progreso
        nombres = {"abduccion": "ABDUCCIÓN DE HOMBRO", "sentadilla": "SENTADILLA"}
        sub = nombres.get(tipo, "") + (f" · {lado.upper()}" if lado else "")
        actual = min(self.rep + 1, n)
        tarjeta, _ = _tarjeta([(sub, 12, C_SECUNDARIO, True), (f"Repetición {actual} de {n}", 22, C_TEXTO, True),
                               ("__puntos__", 9, (self.rep, n), False)],
                              esc, pad=(18, 14), sep=8)
        m = int(18 * esc)
        _pegar(img, tarjeta, m, m, cristal=True, esc=esc)

        # ---------------- puntos (arriba a la derecha) y nota de la ultima repeticion
        tp, _ = _tarjeta([("PUNTOS", 12, C_SECUNDARIO, True), (str(self.puntos), 26, C_TEXTO, True)],
                         esc, pad=(18, 14), sep=6, alinear="centro", ancho_min=110)
        xp = w - tp.shape[1] - m
        _pegar(img, tp, xp, m, cristal=True, esc=esc)
        if self.ultima_nota is not None and ahora - self.t_ultima_nota < 2.2:
            dtn = ahora - self.t_ultima_nota
            aparece = _ease_out(dtn / 0.35)
            alfa = aparece * (1 - max(0.0, (dtn - 1.7) / 0.5))
            centro = (xp + tp.shape[1] / 2, m + tp.shape[0] + 52 * esc)
            radio = 34 * esc * (0.75 + 0.25 * aparece)
            if self.ultima_nota == "X":
                _anillo(img, centro, radio, 7 * esc, 1.0, C_ROJO, alfa=alfa)
                spr = _texto_suelto("×", 34, C_ROJO, esc)
            else:
                col = C_OK if self.ultima_nota >= 8 else C_AMARILLO if self.ultima_nota >= 5 else C_AVISO
                _anillo(img, centro, radio, 7 * esc, aparece * self.ultima_nota / 10.0, col, alfa=alfa)
                spr = _texto_suelto(str(self.ultima_nota), 26, C_TEXTO, esc)
            _pegar(img, spr, centro[0] - spr.shape[1] / 2, centro[1] - spr.shape[0] / 2, alfa=alfa)

        # ---------------- instruccion (pastilla abajo) con punto de estado o anillo de preparacion
        if self.mensaje:
            if self.mensaje != self.msg_prev:
                self.msg_prev, self.t_msg = self.mensaje, ahora
            alfa = _ease_out((ahora - self.t_msg) / 0.25)
            pas, cpunto = _tarjeta([(self.mensaje, 20, C_TEXTO, True)], esc, pad=(22, 14), punto=True)
            xm = (w - pas.shape[1]) / 2
            ym = h - pas.shape[0] - 22 * esc - (6 * esc) * (1 - alfa)
            _pegar(img, pas, xm, ym, alfa=0.4 + 0.6 * alfa, cristal=True, esc=esc)
            c = (xm + cpunto[0], ym + cpunto[1])
            if self.estado == self.PREPARACION and self.t_cond > 0:
                _anillo(img, c, 8 * esc, 3 * esc, self.t_cond / GUIA_T_PREPARACION_S, self.color)
            else:
                cv2.circle(img, (int(c[0]), int(c[1])), max(3, int(6 * esc)), self.color, -1, cv2.LINE_AA)

    def _dibujar_basico(self, img):
        """Respaldo sin Pillow: la sombra y el mensaje con las letras de OpenCV."""
        h, w = img.shape[:2]
        esc = max(0.45, h / 720)
        if self.estado != self.FIN and self.visible:
            self._dibujar_sombra(img, self._figura_q(), self.color, self.color == C_OK)
        texto = _ascii(self.mensaje or "")
        if texto:
            (tw, th), _ = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, 0.9 * esc, 2)
            cv2.putText(img, texto, ((w - tw) // 2, h - int(30 * esc)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9 * esc, self.color, 2, cv2.LINE_AA)



# modo juego: la sombra guia la rutina (False = solo el motor, como antes)
MODO_GUIA = True
# resolucion pedida a la camara: None = la del navegador (lo mas ligero). Si se ve borroso y el
# ordenador va sobrado, probar (640, 480). Mas resolucion = menos FPS = el motor ve "a saltos".
CAMARA_RESOLUCION = None


if not isinstance(st.session_state.get("rehab_motor"), RehabEngine):
    st.session_state["rehab_motor"] = RehabEngine(model_complexity=1)  # en directo, el modelo rapido
    st.session_state["rehab_tiempo"] = {
        "lock": threading.Lock(),
        "ultimo_reloj": None,
        "ultimo_frame_s": None,
        "acumulado_ms": 0.0,
    }

if not isinstance(st.session_state.get("rehab_guia"), GuiaFantasma):
    st.session_state["rehab_guia"] = GuiaFantasma()

motor = st.session_state["rehab_motor"]
guia = st.session_state["rehab_guia"]
tiempo_ctx = st.session_state["rehab_tiempo"]


def _timestamp_ms(frame):
    ahora = time.monotonic()
    frame_s = frame.time
    with tiempo_ctx["lock"]:
        reloj_prev = tiempo_ctx["ultimo_reloj"]
        frame_prev = tiempo_ctx["ultimo_frame_s"]

        nueva_sesion = reloj_prev is None or (ahora - reloj_prev > 2.5)
        if not nueva_sesion and frame_s is not None and frame_prev is not None:
            if abs((frame_s - frame_prev) - (ahora - reloj_prev)) > 2.0:
                nueva_sesion = True

        if nueva_sesion:
            motor.reiniciar()
            guia.reiniciar()
            tiempo_ctx["acumulado_ms"] = 0.0
        else:
            dt_s = (
                (frame_s - frame_prev)
                if (frame_s is not None and frame_prev is not None)
                else (ahora - reloj_prev)
            )
            tiempo_ctx["acumulado_ms"] += max(0.0, dt_s) * 1000.0

        tiempo_ctx["ultimo_reloj"] = ahora
        tiempo_ctx["ultimo_frame_s"] = frame_s
        return tiempo_ctx["acumulado_ms"]


# texto sobre el video con lo que mide el motor (poner a False para la demo final)
MOSTRAR_DEPURACION = not MODO_GUIA

# lo que se puede cambiar en directo (botones / Stream Deck). El hilo del video lo lee en cada fotograma.
if not isinstance(st.session_state.get("rehab_control"), dict):
    st.session_state["rehab_control"] = {"guia": MODO_GUIA, "depuracion": MOSTRAR_DEPURACION,
                                         "rutina": "Rutina completa"}
control = st.session_state["rehab_control"]


def dibujar_depuracion(img, lineas):
    escala = max(0.5, img.shape[0] / 720)
    y = int(30 * escala)
    for texto, ok in lineas:
        color = (140, 255, 150) if ok else (40, 60, 255)  # verde si es valido, rojo si no (BGR)
        cv2.putText(img, texto, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6 * escala, (0, 0, 0), int(4 * escala), cv2.LINE_AA)
        cv2.putText(img, texto, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6 * escala, color, max(1, int(1.5 * escala)), cv2.LINE_AA)
        y += int(26 * escala)


def procesar_frame(frame):
    img = frame.to_ndarray(format="bgr24")
    t_ms = _timestamp_ms(frame)
    landmarks = motor.procesar_fotograma(img, t_ms)
    if landmarks:
        mp_drawing.draw_landmarks(
            img,
            landmarks,
            mp_pose.POSE_CONNECTIONS,
            spec_puntos,
            spec_conexiones,
        )
    if control["guia"]:
        alto, ancho = img.shape[:2]
        guia.actualizar(motor, landmarks, t_ms / 1000.0, ancho, alto)
        guia.dibujar(img)
    if control["depuracion"]:
        dibujar_depuracion(img, motor.texto_depuracion())
    return av.VideoFrame.from_ndarray(img, format="bgr24")


st.markdown(
    f"""<div class="rv-wrap rv d1"><div class="topbar">
<img src="{EURIELEC}" alt="Eurielec">
<div class="topbar-r"><button class="link" data-act="open-tutorial">Tutorial</button><img src="{ELGATO}" alt="Elgato"></div>
</div></div>
<div class="rv-wrap rv d2"><div class="title" role="heading" aria-level="1" aria-label="Rehab Monitor"><span class="t-keep t-r">R</span><span class="t-fade t-ehab">ehab&nbsp;</span><span class="t-keep t-m">M</span><span class="t-fade t-onitor">onitor</span></div></div>
<div class="rv-wrap rv d3"><p class="sub">Reto de Ingeniería Biomédica · EESTEC LC Madrid &amp; Eurielec · ETSIT, UPM</p></div>
<div class="rv-wrap rv d4"><p class="cue"><span>Pulsa <strong>START</strong> a la derecha para iniciar la monitorización biomecánica en tiempo real.</span></p></div>""",
    unsafe_allow_html=True,
)

col_izq, col_der = st.columns([1, 4], gap="medium")
with col_izq:
    st.markdown(
        """<div class="rv-wrap panel-anchor rv d4"><aside class="panel-lateral">
  <div class="panel-lateral-head">
    <span class="panel-tag">Métricas &amp; Control</span>
    <span class="panel-status">En espera</span>
  </div>
  <div class="panel-lateral-body">
    <div class="metric-slot s1">
      <span class="slot-k">Ángulo articular</span>
      <span class="slot-v">--°</span>
    </div>
    <div class="metric-slot s2">
      <span class="slot-k">Rango ROM</span>
      <span class="slot-v">--</span>
    </div>
    <div class="metric-slot s3">
      <span class="slot-k">Repeticiones</span>
      <span class="slot-v">0</span>
    </div>
  </div>
  <div class="panel-live" aria-live="off">
    <div class="live-head">Telemetría en vivo</div>
    <ul class="live-vec">
      <li><span>FPS</span><b data-k="fps">--</b></li>
      <li><span>Latencia</span><b data-k="lat">--</b></li>
      <li><span>Resolución</span><b data-k="res">--</b></li>
      <li><span>Brazo</span><b data-k="lado">--</b></li>
      <li><span>Fase</span><b data-k="fase">--</b></li>
      <li><span>Puntos</span><b data-k="vis">--</b></li>
    </ul>
    <div class="live-head">Vectores x, y, vis</div>
    <ul class="live-vec">
      <li><span>hombro</span><b data-k="v-hombro">--</b></li>
      <li><span>codo</span><b data-k="v-codo">--</b></li>
      <li><span>muñeca</span><b data-k="v-muneca">--</b></li>
    </ul>
    <div class="live-head">Registro</div>
    <ol class="live-log" data-k="log"></ol>
  </div>
  <div class="panel-hint">
    Motor lógico desacoplado en rehab_engine.py (evaluación CLI vía eval.py).
  </div>
</aside></div>""",
        unsafe_allow_html=True,
    )

with col_izq:
    datos_vivo = st.empty()

with col_der:
    ctx = webrtc_streamer(
        key="rehab-cam",
        mode=WebRtcMode.SENDRECV,
        video_frame_callback=procesar_frame,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        # sin audio (no se usa y cuesta CPU); resolucion la que da el navegador salvo que se fije arriba
        media_stream_constraints={
            "video": True if CAMARA_RESOLUCION is None else
            {"width": {"ideal": CAMARA_RESOLUCION[0]}, "height": {"ideal": CAMARA_RESOLUCION[1]}},
            "audio": False,
        },
    )
    def nueva_sesion():
        # el hilo del video lo detecta como sesion nueva en el siguiente fotograma y reinicia motor,
        # guia y reloj el mismo (sin pisarse con el procesado)
        with tiempo_ctx["lock"]:
            tiempo_ctx["ultimo_reloj"] = None
            tiempo_ctx["ultimo_frame_s"] = None
            tiempo_ctx["acumulado_ms"] = 0.0
        if not ctx.state.playing:
            motor.reiniciar()
            guia.reiniciar()

    # controles de la sesion. El Stream Deck los pulsa con atajos de teclado (ver PUENTE_STREAM_DECK);
    # las etiquetas son fijas porque el script las busca por su texto
    b = st.columns(8)
    if b[0].button("Reiniciar sesión", help="Ctrl+Opt+1 · Pone a cero los contadores y empieza la rutina"):
        nueva_sesion()
    if b[1].button("Modo juego", help="Ctrl+Opt+3 · Activa o quita la guía fantasma"):
        control["guia"] = not control["guia"]
    if b[2].button("Depuración", help="Ctrl+Opt+4 · Muestra u oculta lo que mide el motor sobre el vídeo"):
        control["depuracion"] = not control["depuracion"]
    for col, nombre in zip(b[3:], ("Hombro derecho", "Hombro izquierdo", "Sentadillas", "Rutina completa")):
        atajo = {"Hombro derecho": 5, "Hombro izquierdo": 6, "Sentadillas": 7, "Rutina completa": 8}[nombre]
        if col.button(nombre, help=f"Ctrl+Opt+{atajo} · Elige la rutina y empieza de cero"):
            control["rutina"] = nombre
            guia.cambiar_rutina(RUTINAS[nombre])
            nueva_sesion()
    # estado, dentro de la barra flotante (la columna 8 viaja con los botones)
    b[7].caption(f"{control['rutina']} · juego: {'sí' if control['guia'] else 'no'} · "
                 f"depuración: {'sí' if control['depuracion'] else 'no'}")

# puente Stream Deck -> app: cada tecla del Stream Deck envia Ctrl+Option+numero y este script pulsa el
# boton correspondiente (o START/STOP de la camara). Escucha tambien dentro de los iframes, porque al
# pulsar START el foco se queda en el de la camara.
# Ademas convierte la fila de botones en una barra flotante fija abajo: con la camara encendida la
# interfaz amplia el video y la fila quedaba fuera de la pantalla.
PUENTE_STREAM_DECK = """
(() => {
  const top = window.parent, doc = top.document;
  // version del puente: una version nueva sustituye a la anterior sin recargar la pagina
  const VERSION = 4;
  if (top.__puenteVersion === VERSION) return;
  top.__puenteVersion = VERSION;
  (top.__puenteIntervalos || []).forEach((i) => top.clearInterval(i));
  top.__puenteIntervalos = [];
  const ACCIONES = {
    Digit1: 'Reiniciar sesión', Digit2: '__camara__', Digit3: 'Modo juego', Digit4: 'Depuración',
    Digit5: 'Hombro derecho', Digit6: 'Hombro izquierdo', Digit7: 'Sentadillas', Digit8: 'Rutina completa',
  };
  const pulsar = (texto) => {
    const b = Array.from(doc.querySelectorAll('button')).find((x) => x.textContent.trim() === texto);
    if (b) b.click();
  };
  const camara = () => {
    for (const f of doc.querySelectorAll('iframe')) {
      let d = null;
      try { d = f.contentDocument; } catch (e) { continue; }
      if (!d) continue;
      const b = Array.from(d.querySelectorAll('button')).find((x) => /^(start|stop)$/i.test(x.textContent.trim()));
      if (b) { b.click(); return; }
    }
  };
  const alPulsar = (e) => {
    if (!(e.ctrlKey && e.altKey)) return;          // Ctrl + Option (en Mac Option = Alt)
    const accion = ACCIONES[e.code];                // e.code: no cambia con Option (e.key si)
    if (!accion) return;
    e.preventDefault();
    if (accion === '__camara__') camara(); else pulsar(accion);
  };
  // un unico oyente por documento que siempre llama a la version vigente del puente
  top.__puenteAlPulsar = alPulsar;
  const enganchar = (d) => {
    if (!d || d.__puenteOyente) return;
    d.__puenteOyente = true;
    d.addEventListener('keydown', (e) => top.__puenteAlPulsar && top.__puenteAlPulsar(e), true);
  };
  enganchar(doc);
  top.__puenteIntervalos.push(top.setInterval(() => {
    for (const f of doc.querySelectorAll('iframe')) { try { enganchar(f.contentDocument); } catch (e) {} }
  }, 1000));

  // ---- barra flotante con los botones (siempre visible, con la camara encendida o apagada)
  const imp = (el, prop, valor) => el.style.setProperty(prop, valor, 'important');
  const fijarBarra = () => {
    const b = Array.from(doc.querySelectorAll('button')).find((x) => x.textContent.trim() === 'Reiniciar sesión');
    const fila = b && b.closest('[data-testid="stHorizontalBlock"]');
    if (!fila) return;
    const estilos = {
      position: 'fixed', left: '50%', bottom: '14px', transform: 'translateX(-50%)', 'z-index': '9999',
      width: 'auto', 'max-width': 'calc(100vw - 32px)', margin: '0', padding: '8px 10px',
      display: 'flex', 'flex-wrap': 'wrap', 'justify-content': 'center', 'align-items': 'center', gap: '8px',
      background: 'rgba(13, 13, 20, 0.85)', border: '1px solid rgba(243, 240, 232, 0.26)',
      'border-radius': '14px', 'backdrop-filter': 'blur(6px)',
    };
    for (const [k, v] of Object.entries(estilos)) imp(fila, k, v);
    for (const c of fila.querySelectorAll(':scope > [data-testid="column"]')) {
      for (const [k, v] of Object.entries({ flex: '0 0 auto', width: 'auto', 'min-width': '0', 'max-width': 'none',
        display: 'block', 'align-items': 'center' })) imp(c, k, v);
    }
    for (const x of fila.querySelectorAll('button')) {
      imp(x, 'white-space', 'nowrap'); imp(x, 'padding', '4px 12px'); imp(x, 'min-height', '0');
    }
    for (const x of fila.querySelectorAll('[data-testid="stCaptionContainer"], small')) {
      imp(x, 'white-space', 'nowrap'); imp(x, 'margin', '0 4px'); imp(x, 'color', 'rgba(243, 240, 232, 0.75)');
    }
  };
  fijarBarra();
  top.__puenteIntervalos.push(top.setInterval(fijarBarra, 500));   // Streamlit repinta la fila en cada recarga
})();
"""
components.html(f"<script>{PUENTE_STREAM_DECK}</script>", height=0)


def publicar(payload):
    # el panel lo lee intro.js desde este nodo oculto
    dato = html.escape(json.dumps(payload), quote=True)
    datos_vivo.markdown(
        f'<div class="eur-live" data-json="{dato}"></div>', unsafe_allow_html=True
    )


estaba_activo = st.session_state.get("camara_activa", False)
st.session_state["camara_activa"] = ctx.state.playing
if not ctx.state.playing and estaba_activo:
    motor.reiniciar()
    guia.reiniciar()
    with tiempo_ctx["lock"]:
        tiempo_ctx["ultimo_reloj"] = None
        tiempo_ctx["ultimo_frame_s"] = None
        tiempo_ctx["acumulado_ms"] = 0.0

# bucle de telemetria: mientras la camara este encendida empujamos datos al panel
while ctx.state.playing:
    publicar(motor.instantanea())
    time.sleep(0.3)
publicar({"live": False})