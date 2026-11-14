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
C_GUIA = (255, 220, 90)      # cian: la sombra marca el camino
C_OK = (40, 255, 40)         # verde intenso: posicion correcta
C_AVISO = (0, 150, 255)      # naranja: en posicion pero con compensacion
C_ROJO = (60, 60, 230)
C_AMARILLO = (0, 220, 255)

# avisos para el paciente a partir de lo que mide el motor
AVISO_PLANO = "Sube el brazo por el lado, no por delante"
AVISO_CODO = "Estira el brazo"
AVISO_LATERAL = "No te ladees, tronco recto"
AVISO_HOMBRO = "Relaja el hombro, no lo subas"
AVISO_GIRO = "Ponte de frente a la camara"
AVISO_TRONCO = "Espalda mas recta, no te inclines"
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


def _ascii(texto):
    """OpenCV no dibuja tildes ni simbolos: se quitan."""
    import unicodedata
    t = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode()
    return t.replace("  ", " ").strip()


class GuiaFantasma:
    PREPARACION, IR, ESPERA_OBJ, VOLVER, ESPERA_REP, FIN = (
        "preparacion", "ir", "espera_objetivo", "volver", "espera_reposo", "fin")

    def __init__(self, rutina=RUTINA):
        self.rutina = rutina
        self.lock = threading.RLock()     # el hilo del video y el boton de reinicio la tocan a la vez
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
        lateral = getattr(m, "f3_lateral", None)
        if lateral is not None and lateral > _re.INCLINACION_MAX:
            av.append(AVISO_LATERAL)
        if self.ejercicio[0] == "abduccion":
            if m.torso_rotacion > (_re.TORSO_GIRO_MAX_3D if m.modo_3d else _re.TORSO_ROTACION_MAX):
                av.append(AVISO_GIRO)
            for l in self._lados():
                info = m.abd_info[l]
                if info["ang"] is not None and info["ang"] > _re.ABD_ABAJO:
                    if not info["en_plano"]:
                        av.append(AVISO_PLANO)
                    if info.get("codo") is not None and info["codo"] < _re.CODO_EXTENDIDO_MIN:
                        av.append(AVISO_CODO)
            eps = getattr(m, "episodios", {})
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
        if not (lm[11].visibility >= 0.5 and lm[12].visibility >= 0.5 and
                lm[23].visibility >= 0.3 and lm[24].visibility >= 0.3):
            return False
        vivo = {}
        for nombre, i in PUNTOS_GUIA.items():
            if lm[i].visibility >= (0.3 if i in (23, 24) else 0.5):
                p = np.array([lm[i].x * w, lm[i].y * h])
                vivo[nombre] = _ema(self.vivo.get(nombre), p, dt, GUIA_TAU_VIVO_S)
        self.vivo = vivo
        hombros_c = (vivo["hombroI"] + vivo["hombroD"]) / 2
        caderas_c = (vivo["caderaI"] + vivo["caderaD"]) / 2
        self.hombros_c = hombros_c
        self.tobillos_c = (vivo["tobilloI"] + vivo["tobilloD"]) / 2 \
            if "tobilloI" in vivo and "tobilloD" in vivo else None
        self.signo_i = 1.0 if vivo["hombroI"][0] >= vivo["hombroD"][0] else -1.0

        def media(pares):
            vals = [np.linalg.norm(vivo[a] - vivo[b]) for a, b in pares if a in vivo and b in vivo]
            return float(np.mean(vals)) if vals else None
        segs = {"tronco": float(np.linalg.norm(hombros_c - caderas_c)),
                "hombros": float(np.linalg.norm(vivo["hombroI"] - vivo["hombroD"])),
                "caderas": float(np.linalg.norm(vivo["caderaI"] - vivo["caderaD"])),
                "brazo": media((("hombroI", "codoI"), ("hombroD", "codoD"))),
                "antebrazo": media((("codoI", "munecaI"), ("codoD", "munecaD"))),
                "muslo": media((("caderaI", "rodillaI"), ("caderaD", "rodillaD"))),
                "tibia": media((("rodillaI", "tobilloI"), ("rodillaD", "tobilloD")))}
        if "cabeza" in vivo:
            segs["cabeza"] = float(np.linalg.norm(vivo["cabeza"] - hombros_c))
        self.segs = {k: v for k, v in segs.items() if v}
        if reposo:
            for k, v in self.segs.items():
                b = self.buf.setdefault(k, [])
                b.append(v)
                del b[:-GUIA_N_CALIB]
                self.calib[k] = float(np.median(b))
        return True

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

    def _pose_actual(self):
        tipo, lado, _ = self.ejercicio
        p = _suave(self.progreso)
        if tipo == "abduccion":
            return self._pose_abduccion(p), tipo
        abd_i = abd_d = 0.0
        flex = 0.0
        if tipo == "abduccion":
            if lado in ("izquierdo", "ambos"):
                abd_i = ABD_SOMBRA * p
            if lado in ("derecho", "ambos"):
                abd_d = ABD_SOMBRA * p
        elif tipo == "sentadilla":
            flex = FLEX_SOMBRA * p
        return self._pose(abd_i, abd_d, flex), tipo

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
            self.mensaje = "Pulsa 'Reiniciar sesion' para volver a jugar"
            return
        tipo = self.ejercicio[0]
        with motor.lock:
            v = self._medida(motor)
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
            self.mensaje, self.color = "Colocate de frente, de cuerpo entero", C_ROJO
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
        nombre = "abduccion" if tipo == "abduccion" else "sentadilla"

        if self.estado == self.PREPARACION:
            self.progreso = 0.0
            self.t_cond = self.t_cond + dt if (en_rep and not avisos) else 0.0
            if avisos:
                self.color, self.mensaje = C_AVISO, avisos[0]
            else:
                lado = self.ejercicio[1]
                self.color = C_GUIA
                self.mensaje = f"Preparate: {nombre}" + (f" ({lado})" if lado else "") + \
                    ("" if en_rep else (" - baja el brazo" if tipo == "abduccion" else " - ponte de pie"))
            if self.t_cond >= GUIA_T_PREPARACION_S:
                self._empezar_rep(motor, t)
        elif self.estado == self.IR:
            # pausa breve en reposo antes de que la sombra arranque
            t_mov = max(0.0, self.t_estado - GUIA_T_ENTRE_REPS_S)
            self.progreso = min(1.0, t_mov / GUIA_T_ANIM_S)
            if self.t_estado < GUIA_T_ENTRE_REPS_S:
                self.mensaje = "Preparate..."
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
                self.color, self.mensaje = C_OK, "Muy bien!"
                self._cambiar(self.VOLVER)
            elif rechazada and en_rep:
                # el motor no la dio por buena y el paciente ya ha vuelto al reposo: se repite
                self._no_valida(t, motivo_reciente)
                self._empezar_rep(motor, t)
            elif aceptado and invalida:
                # ha llegado abajo/arriba, pero el motor la va a anular: naranja, nunca verde
                self.color, self.mensaje = C_AVISO, "No contara: " + invalida
            elif avisos:
                self.color, self.mensaje = C_AVISO, avisos[0]
            elif motivo_reciente:
                self.color, self.mensaje = C_AVISO, "No vale: " + motivo_reciente
            else:
                self.color = C_GUIA
                if otro_brazo:
                    self.mensaje = f"Con el brazo {self.ejercicio[1]}"   # esta subiendo el otro brazo
                else:
                    self.mensaje = "Un poco mas arriba" if tipo == "abduccion" else "Un poco mas abajo y aguanta"
        elif self.estado == self.VOLVER:
            self.progreso = max(0.0, 1.0 - self.t_estado / GUIA_T_ANIM_S)
            if invalida:
                # algo la ha invalidado durante la vuelta: se quita el verde en el acto
                self.color, self.mensaje = C_AVISO, "No contara: " + invalida
            elif self.t_estado < GUIA_T_VERDE_S:
                self.color, self.mensaje = C_OK, "Muy bien!"   # el verde se mantiene un momento
            else:
                self.color, self.mensaje = C_GUIA, "Ahora vuelve despacio"
            if self.progreso <= 0.0:
                self._cambiar(self.ESPERA_REP)
        elif self.estado == self.ESPERA_REP:
            self.progreso = 0.0
            if invalida:
                self.color, self.mensaje = C_AVISO, "No contara: " + invalida
            else:
                self.color = C_GUIA
                self.mensaje = "Vuelve a la posicion inicial" if tipo == "abduccion" else "Ponte de pie del todo"
            if terminada:
                self._cerrar_rep(motor, t, cuenta > self.cuenta_ini, motivo_reciente)

        # confirmacion: el motor acaba de sumar la repeticion -> verde mientras la sombra espera en reposo
        if t - self.t_contada < GUIA_T_VERDE_S and self.estado in (self.PREPARACION, self.IR) \
                and self.progreso <= 0.0 and not avisos:
            self.color, self.mensaje = C_OK, "Contada! +1"

    def _cerrar_rep(self, motor, t, valida, motivo_reciente):
        """Cierra la repeticion con lo que ha decidido el motor (valida = su contador ha subido)."""
        if valida:
            nota = self._nota()
            self.notas.append(nota)
            self.puntos += nota
            self.ultima_nota = nota
            self.rep += 1
            self.t_ultima_nota = self.t_contada = t
        else:
            self._no_valida(t, motivo_reciente)   # se repite la misma repeticion
        if self.rep >= self.ejercicio[2]:
            self.paso, self.rep = self.paso + 1, 0
            if self.paso < len(self.rutina):
                self._cambiar(self.PREPARACION)
            else:
                self._cambiar(self.FIN)
        else:
            self._empezar_rep(motor, t)

    # ------------------------------------------------------------ dibujo
    def dibujar(self, img):
        with self.lock:
            self._dibujar(img)

    def _dibujar(self, img):
        h, w = img.shape[:2]
        esc = max(0.45, h / 720)
        if self.estado != self.FIN and self.visible:
            P, tipo = self._pose_actual()
            if tipo == "abduccion":
                Q = {k: v.astype(int) for k, v in P.items()}
            else:
                # sentadilla: la figura objetivo apoyada en los tobillos reales de este fotograma
                if self.tobillos_c is not None:
                    ancla = self.tobillos_c
                else:
                    ancla = self.hombros_c - (P["hombroI"] + P["hombroD"]) / 2
                Q = {k: (v + ancla).astype(int) for k, v in P.items()}
            verde = self.color == C_OK
            grosor = max(4, int(self._p("tronco") * 0.09))
            huesos = [("hombroI", "hombroD"), ("caderaI", "caderaD"), ("hombroI", "caderaI"),
                      ("hombroD", "caderaD")]
            for l in ("I", "D"):
                huesos += [("hombro" + l, "codo" + l), ("codo" + l, "muneca" + l),
                           ("cadera" + l, "rodilla" + l), ("rodilla" + l, "tobillo" + l)]
            r_cabeza = int(self._p("cabeza") * 0.45)
            # rectangulo que contiene la figura: solo se copia y se mezcla esa zona (antes toda la imagen)
            pts = np.array(list(Q.values()))
            margen = r_cabeza + grosor * 3
            x0, y0 = np.maximum(pts.min(axis=0) - margen, 0)
            x1, y1 = np.minimum(pts.max(axis=0) + margen, [w - 1, h - 1])
            if x1 > x0 and y1 > y0:
                zona = img[y0:y1 + 1, x0:x1 + 1]
                capa = zona.copy()
                o = np.array([x0, y0])
                R = {k: tuple(v - o) for k, v in Q.items()}

                def figura(lienzo, color, g):
                    for a_, b_ in huesos:
                        if a_ in R and b_ in R:
                            cv2.line(lienzo, R[a_], R[b_], color, g, cv2.LINE_AA)
                    for k_, q in R.items():
                        if k_ != "cabeza":
                            cv2.circle(lienzo, q, int(g * 0.8), color, -1, cv2.LINE_AA)
                    if "cabeza" in R:
                        cuello = tuple(((np.array(R["hombroI"]) + np.array(R["hombroD"])) // 2).tolist())
                        cv2.line(lienzo, cuello, R["cabeza"], color, g, cv2.LINE_AA)
                        cv2.circle(lienzo, R["cabeza"], r_cabeza + (g - grosor) // 2, color, -1, cv2.LINE_AA)

                if verde:
                    # resaltado en verde: contorno oscuro grueso + figura mas gruesa (sin GaussianBlur)
                    figura(capa, (0, 90, 0), int(grosor * 2.2))
                    figura(capa, self.color, int(grosor * 1.4))
                else:
                    figura(capa, self.color, grosor)
                alfa = GUIA_ALFA_OK if verde else GUIA_ALFA
                cv2.addWeighted(capa, alfa, zona, 1 - alfa, 0, zona)

        # marcador superior
        tipo, lado, n = self.ejercicio
        if self.estado == self.FIN:
            media = sum(self.notas) / len(self.notas) if self.notas else 0
            estrellas = 3 if media >= 8 else 2 if media >= 5 else 1
            self._caja(img, f"SESION COMPLETADA  {'*' * estrellas}", (w // 2, int(h * 0.40)), 1.2 * esc, C_OK, centro=True)
            self._caja(img, f"Puntos {self.puntos}   Nota media {media:.1f}/10   Repetidas {self.anuladas}",
                       (w // 2, int(h * 0.52)), 0.8 * esc, (255, 255, 255), centro=True)
            self._caja(img, _ascii(self.mensaje), (w // 2, h - int(30 * esc)), 0.8 * esc, C_GUIA, centro=True)
            return
        titulo = f"{tipo.upper()}{' ' + lado.upper() if lado else ''}  {min(self.rep + 1, n)}/{n}"
        self._caja(img, titulo, (int(12 * esc), int(34 * esc)), 0.8 * esc, (255, 255, 255))
        self._caja(img, f"Puntos {self.puntos}", (w - int(190 * esc), int(34 * esc)), 0.8 * esc, C_AMARILLO)
        # nota de la ultima repeticion (se ve 2 s); X = el motor no la conto y se repite
        if self.ultima_nota is not None and self.t_prev is not None and self.t_prev - self.t_ultima_nota < 2.0:
            n_ = self.ultima_nota
            if n_ == "X":
                col = C_ROJO
            else:
                col = C_OK if n_ >= 8 else C_AMARILLO if n_ >= 5 else C_AVISO if n_ >= 3 else C_ROJO
            centro = (w - int(70 * esc), int(110 * esc))          # bajo el marcador, sin tapar a la persona
            cv2.circle(img, centro, int(42 * esc), col, -1, cv2.LINE_AA)
            self._texto(img, str(n_), (centro[0], centro[1] + int(14 * esc)), 1.3 * esc, (0, 0, 0), centro=True)
        # mensaje inferior
        if self.mensaje:
            self._caja(img, _ascii(self.mensaje), (w // 2, h - int(30 * esc)), 0.9 * esc, self.color, centro=True)

    @staticmethod
    def _texto(img, texto, pos, escala, color, centro=False):
        grosor = max(1, int(2 * escala))
        (tw, th), _ = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, escala, grosor)
        x, y = (pos[0] - tw // 2, pos[1]) if centro else pos
        cv2.putText(img, texto, (x, y), cv2.FONT_HERSHEY_SIMPLEX, escala, color, grosor, cv2.LINE_AA)

    @staticmethod
    def _caja(img, texto, pos, escala, color, centro=False):
        grosor = max(1, int(2 * escala))
        (tw, th), base = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, escala, grosor)
        x, y = (pos[0] - tw // 2, pos[1]) if centro else pos
        pad = int(8 * escala) + 2
        x0, y0 = max(0, x - pad), max(0, y - th - pad)
        sub = img[y0:y + base + pad, x0:x + tw + pad]
        if sub.size:
            sub[:] = (sub * 0.35).astype(sub.dtype)      # fondo oscuro translucido
        cv2.putText(img, texto, (x, y), cv2.FONT_HERSHEY_SIMPLEX, escala, color, grosor, cv2.LINE_AA)


# modo juego: la sombra guia la rutina (False = solo el motor, como antes)
MODO_GUIA = True
# resolucion pedida a la camara: None = la del navegador (lo mas ligero). Si se ve borroso y el
# ordenador va sobrado, probar (640, 480). Mas resolucion = menos FPS = el motor ve "a saltos".
CAMARA_RESOLUCION = None


if not isinstance(st.session_state.get("rehab_motor"), RehabEngine):
    st.session_state["rehab_motor"] = RehabEngine()
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
    if MODO_GUIA:
        alto, ancho = img.shape[:2]
        guia.actualizar(motor, landmarks, t_ms / 1000.0, ancho, alto)
        guia.dibujar(img)
    if MOSTRAR_DEPURACION:
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
    # volver a empezar: el hilo del video lo detecta como sesion nueva en el siguiente fotograma
    # (reinicia motor, guia y reloj el mismo, sin pisarse con el procesado)
    if st.button("Reiniciar sesión", key="reiniciar_sesion",
                 help="Pone a cero los contadores y vuelve a empezar la rutina"):
        with tiempo_ctx["lock"]:
            tiempo_ctx["ultimo_reloj"] = None
            tiempo_ctx["ultimo_frame_s"] = None
            tiempo_ctx["acumulado_ms"] = 0.0
        if not ctx.state.playing:
            motor.reiniciar()
            guia.reiniciar()


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