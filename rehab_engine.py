"""Motor de rehabilitacion: logica de procesamiento sin ninguna dependencia de Streamlit.

La interfaz (app.py) y el evaluador automatico (eval.py) usan este mismo motor.
Todo el tiempo se calcula a partir de los timestamps de los fotogramas (milisegundos),
nunca contando fotogramas, de modo que el resultado no depende de los FPS del video
ni de la velocidad a la que se procese.

Fase 1 (abduccion de hombro), ejercicio 2D en el plano frontal:
  * solo usa de cadera hacia arriba (vale de pie y sentado); si las caderas no se
    ven, el eje del tronco se estima perpendicular a la linea de hombros;
  * angulo del brazo (hombro-codo) contra el eje del tronco, por producto escalar,
    en el plano XY de la imagen con la relacion de aspecto corregida. No se usa la
    z de MediaPipe: con una sola camara es poco fiable;
  * rotacion del torso: con la anchura aparente de hombros y caderas respecto a la
    altura del tronco se estima si el paciente esta de frente; si esta girado no cuenta;
  * brazo en el plano del torso: si el brazo se ve acortado (escorzo) apunta hacia
    delante o atras y no cuenta; si en la subida paso por delante, tampoco;
  * se tiene en cuenta la muneca: el brazo entero (hombro-muneca) debe llegar al angulo
    y el codo debe estar estirado (una abduccion se hace con el brazo extendido);
  * cuenta al alcanzar 90 grados en la subida (con margen permisivo), en ambos brazos;
  * posiciones filtradas con One Euro y tiempos con dt reales (nunca fotogramas).
"""
import math
import threading
import time
from collections import deque

import cv2
import mediapipe as mp
import numpy as np

from utils import calcular_angulo

# indices de MediaPipe Pose
OREJA_I, OREJA_D = 7, 8
HOMBRO_I, HOMBRO_D = 11, 12
CODO_I, CODO_D = 13, 14
MUNECA_I, MUNECA_D = 15, 16
CADERA_I, CADERA_D = 23, 24
RODILLA_I, RODILLA_D = 25, 26
TOBILLO_I, TOBILLO_D = 27, 28
N_PUNTOS = 33

VISIBILIDAD_MIN = 0.5

# filtro One Euro sobre la posicion de los puntos
# (en reposo suaviza mucho para quitar el jitter; en movimiento rapido apenas, para no meter retardo)
FILTRO_MIN_CUTOFF = 1.2   # Hz, frecuencia de corte en reposo (menor = mas suave)
FILTRO_BETA = 1.5         # cuanto sube la frecuencia de corte con la velocidad del punto
FILTRO_D_CUTOFF = 1.0     # Hz, suavizado de la estimacion de velocidad
FILTRO_HUECO_S = 0.5      # si la pose se pierde mas de esto, el filtro se reinicia

# fase 1: abduccion de hombro (brazo contra el eje del tronco, plano frontal)
ABD_OBJETIVO = 90.0             # la abduccion pide llegar a 90 grados...
ABD_MARGEN = 10.0               # ...con este margen permisivo
ABD_ARRIBA = ABD_OBJETIVO - ABD_MARGEN   # se cuenta al superar este angulo en la subida
ABD_ABAJO = 40.0                # hay que volver por debajo para poder contar otra (rearme)
ABD_T_ARMADO_S = 0.4            # el brazo debe verse abajo este tiempo antes de poder contar
ABD_T_MIN_ARRIBA_S = 0.10       # permanencia minima sobre el umbral (descarta picos de ruido)
ABD_T_MIN_ABAJO_S = 0.15        # permanencia minima abajo para rearmar
ABD_HUECO_REARME_S = 1.0        # si el brazo deja de verse este tiempo, hay que volver a armar
ABD_FUSION_BILATERAL_S = 0.8    # dos brazos que llegan arriba casi a la vez cuentan como una
CODO_EXTENDIDO_MIN = 140.0      # angulo hombro-codo-muneca minimo: brazo estirado (180 = recto)
VIS_CADERA_MIN = 0.3            # sentado las caderas se ven peor: se aceptan con menos confianza
# escorzo del brazo: |hombro-codo| / escala del tronco, comparado con su referencia
ABD_ESCORZO_MIN = 0.65          # por debajo, el brazo apunta hacia la camara: fuera del plano
ABD_BRAZO_REF_MIN = 0.45        # referencia minima anatomica (brazo/tronco ~0.6): evita aprender un escorzo
# rotacion del torso: anchura de hombros / altura del tronco, comparada con su referencia
TORSO_REF_MIN = 0.60            # referencia minima anatomica (hombros/tronco ~0.7)
TORSO_ROTACION_MAX = 40.0       # grados de giro maximos para considerar que esta de frente
TRONCO_POR_HOMBROS = 1.4        # altura del tronco ~ 1.4 x anchura de hombros (si no se ven caderas)
REF_TAU_SUBE_S = 0.5            # la referencia del brazo se adapta rapido hacia arriba
TORSO_REF_TAU_SUBE_S = 3.0      # la del torso mas despacio (los picos de ruido no la inflan)
REF_TAU_BAJA_S = 20.0           # y muy despacio hacia abajo

# fase 2: sentadilla (angulo cadera-rodilla-tobillo)
SENT_ABAJO = 110.0
SENT_ARRIBA = 160.0
# fase 3: compensacion por inclinacion lateral del tronco
INCLINACION_MAX = 15.0       # grados respecto a la vertical
INCLINACION_PERSISTE_S = 0.3  # segundos seguidos para considerarlo error
ERROR_SEPARACION_S = 1.0      # separacion minima entre errores registrados

TAU_SUAVIZADO_S = 0.08  # constante de tiempo del filtro exponencial (fase 2)


# ==================================================================== filtrado
def _alfa(cutoff, dt):
    """Factor de suavizado de una EMA equivalente a un paso bajo de frecuencia `cutoff`."""
    tau = 1.0 / (2.0 * math.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


class FiltroOneEuro:
    """Filtro One Euro vectorizado para N puntos (Casiez et al., 2012).

    Es una EMA cuya frecuencia de corte crece con la velocidad de cada punto:
    cutoff = min_cutoff + beta * |velocidad|. Usa dt real, no numero de fotogramas.
    """

    def __init__(self, min_cutoff=FILTRO_MIN_CUTOFF, beta=FILTRO_BETA, d_cutoff=FILTRO_D_CUTOFF):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.reiniciar()

    def reiniciar(self):
        self.x = None
        self.dx = None

    def __call__(self, x, dt):
        x = np.asarray(x, dtype=float)
        if self.x is None or self.x.shape != x.shape:
            self.x = x.copy()
            self.dx = np.zeros_like(x)
            return self.x
        if dt <= 0:
            return self.x
        dx = (x - self.x) / dt
        self.dx = self.dx + _alfa(self.d_cutoff, dt) * (dx - self.dx)
        velocidad = np.linalg.norm(self.dx, axis=-1, keepdims=True)
        cutoff = self.min_cutoff + self.beta * velocidad
        self.x = self.x + _alfa(cutoff, dt) * (x - self.x)
        return self.x


# ==================================================================== geometria
def angulo_vectores(u, v):
    """Angulo en grados entre dos vectores (producto escalar, ec. 1 del enunciado)."""
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    if nu < 1e-9 or nv < 1e-9:
        return None
    c = float(np.dot(u, v) / (nu * nv))
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def _ema_tiempo(previo, nuevo, dt, tau):
    if previo is None:
        return nuevo
    if dt <= 0:
        return previo
    return previo + (1.0 - math.exp(-dt / tau)) * (nuevo - previo)


# ==================================================================== maquina de estados
class FSMAbduccion:
    """FSM de un brazo:  SIN_ARMAR -> ABAJO -> ARRIBA (= 1 abduccion) -> ABAJO (rearme).

    * La abduccion es la SUBIDA: cuenta al alcanzar 90 grados. La bajada (aduccion)
      no cuenta; solo rearma el brazo para la siguiente.
    * SIN_ARMAR: al empezar (o tras perder el brazo) no se cuenta nada hasta ver el
      brazo en reposo un tiempo; asi un brazo que ya estaba levantado no suma.
    * Cada transicion exige histeresis de angulo y permanencia minima (suma de dt).
    * Recorrido: si durante la subida el brazo pasa por delante (escorzado), es una
      elevacion frontal y no cuenta aunque luego acabe por encima de la cabeza.
    """

    SIN_ARMAR, ABAJO, ARRIBA = "sin armar", "abajo", "arriba"

    def __init__(self):
        self.estado = self.SIN_ARMAR
        self.t_cond = 0.0        # tiempo acumulado cumpliendo la condicion de cambio
        self.t_sin_ver = 0.0     # tiempo seguido sin medida valida
        self.pico = 0.0          # angulo maximo de la repeticion en curso
        self.t_salida = None     # instante en que el brazo salio de abajo
        self.paso_delante = False  # en esta subida el brazo ha pasado por delante del cuerpo

    def sin_medida(self, dt):
        """El brazo no se ve en este fotograma."""
        self.t_sin_ver += dt
        self.t_cond = 0.0
        if self.t_sin_ver >= ABD_HUECO_REARME_S:
            self.estado = self.SIN_ARMAR

    def actualizar(self, ang, en_plano, dt, t, valido=True):
        """Devuelve True cuando se completa una abduccion (llegada a 90 grados).

        `valido` reune el resto de condiciones para poder contar la subida
        (brazo entero por encima del umbral y codo estirado).
        """
        self.t_sin_ver = 0.0
        abajo = ang < ABD_ABAJO and en_plano
        if abajo:
            self.paso_delante = False
        elif not en_plano:
            self.paso_delante = True
        arriba = ang > ABD_ARRIBA and en_plano and valido and not self.paso_delante

        if self.estado == self.SIN_ARMAR:
            self.t_cond = self.t_cond + dt if abajo else 0.0
            if self.t_cond >= ABD_T_ARMADO_S:
                self.estado, self.t_cond = self.ABAJO, 0.0
            return False

        if self.estado == self.ABAJO:
            if abajo:
                self.t_salida = t        # ultimo instante en reposo
                self.t_cond, self.pico = 0.0, 0.0
            elif arriba:
                self.t_cond += dt
                self.pico = max(self.pico, ang)
                if self.t_cond >= ABD_T_MIN_ARRIBA_S:
                    self.estado, self.t_cond = self.ARRIBA, 0.0
                    return True          # abduccion completada: ha llegado a 90 grados
            else:
                self.t_cond = 0.0
            return False

        # ARRIBA
        if en_plano:
            self.pico = max(self.pico, ang)
        if abajo:
            self.t_cond += dt
            if self.t_cond >= ABD_T_MIN_ABAJO_S:
                self.estado, self.t_cond = self.ABAJO, 0.0   # rearmado, no cuenta
        else:
            self.t_cond = 0.0
        return False


# ==================================================================== motor
class RehabEngine:
    def __init__(self):
        self.pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=1,  # modelo incluido en el paquete: funciona sin conexion
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        # la interfaz lee el estado desde otro hilo mientras el callback de video lo escribe
        self.lock = threading.RLock()
        self.reiniciar()

    def reiniciar(self):
        with self.lock:
            self.t_prev = None
            self.t_ultima_pose = None
            self.fps = 0.0
            self.latencia_ms = 0.0
            self.resolucion = "-"
            self.aspecto = 1.0
            self.pose_detectada = False
            self.visibles = 0
            self.lado = "-"
            self.vectores = {}
            self.rom_min = None
            self.rom_max = None
            self.log = deque(maxlen=30)
            self.filtro_img = FiltroOneEuro()
            # fase 1
            self.abd = None
            # por brazo: angulo, si esta en el plano frontal, escorzo y longitud de referencia
            self.torso_ref = None
            self.torso_rotacion = 0.0
            self.torso_caderas = False
            self.abd_info = {lado: {"ang": None, "en_plano": False, "escorzo": None, "ref": None,
                                    "codo": None, "ang_muneca": None, "valido": True}
                             for lado in ("izquierdo", "derecho")}
            self.fsm_abd = {"izquierdo": FSMAbduccion(), "derecho": FSMAbduccion()}
            self.ultima_abd_t = None
            self.repeticiones_abduccion = 0
            # fase 2
            self.sent = None
            self.sent_abajo = False
            self.repeticiones_sentadilla = 0
            # fase 3
            self.inclinado_desde = None
            self.error_activo = False
            self.timestamps_errores = []

    # ---------------------------------------------------------------- entrada
    def procesar_fotograma(self, frame_bgr, t_ms):
        """Detecta la pose en un fotograma BGR y actualiza el estado con su timestamp (ms).

        Devuelve los landmarks de MediaPipe (o None) para que la interfaz pueda dibujarlos.
        """
        inicio = time.perf_counter()
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res = self.pose.process(rgb)
        lm = res.pose_landmarks.landmark if res.pose_landmarks else None
        alto, ancho = frame_bgr.shape[:2]
        with self.lock:
            self.aspecto = ancho / alto if alto else 1.0
            self.actualizar(lm, t_ms)
            self.resolucion = f"{ancho}x{alto}"
            self.latencia_ms = (time.perf_counter() - inicio) * 1000.0
        return res.pose_landmarks

    def actualizar(self, lm, t_ms):
        """Actualiza la logica con 33 landmarks de imagen (o None) y el tiempo en ms."""
        with self.lock:
            t = t_ms / 1000.0
            if self.t_prev is None:
                self._anotar(t, "stream iniciado")
                dt = 0.0
            else:
                dt = max(0.0, t - self.t_prev)
            self.t_prev = t
            if dt > 0:
                self.fps = self._suavizar(self.fps or None, 1.0 / dt, dt)

            if lm is None:
                if self.pose_detectada:
                    self._anotar(t, "pose perdida")
                self.pose_detectada = False
                self.visibles = 0
                self.inclinado_desde = None
                for fsm in self.fsm_abd.values():
                    fsm.sin_medida(dt)
                return
            if not self.pose_detectada:
                self._anotar(t, "pose detectada")
            self.pose_detectada = True

            # tras un hueco largo sin pose el filtro arranca de cero (no arrastra posiciones viejas)
            if self.t_ultima_pose is None or t - self.t_ultima_pose > FILTRO_HUECO_S:
                self.filtro_img.reiniciar()
                dt_f = 0.0
            else:
                dt_f = t - self.t_ultima_pose
            self.t_ultima_pose = t

            vis = np.array([p.visibility for p in lm])
            self.visibles = int(np.sum(vis > VISIBILIDAD_MIN))
            img = self.filtro_img(np.array([[p.x, p.y] for p in lm]), dt_f)
            # plano frontal XY con la misma escala en los dos ejes
            # (x e y normalizados no miden lo mismo si el video no es cuadrado)
            P = img * np.array([self.aspecto, 1.0])

            self._fase_1(P, img, vis, dt_f, t)
            self._fase_2(P, vis, dt_f, t)
            self._fase_3(P, vis, t)

    def _anotar(self, t, texto):
        self.log.append(f"{t:6.1f}s  {texto}")

    # ---------------------------------------------------------------- utilidades
    @staticmethod
    def _visibles(vis, ids):
        return all(vis[i] >= VISIBILIDAD_MIN for i in ids)

    @staticmethod
    def _suavizar(previo, nuevo, dt):
        if previo is None:
            return nuevo
        alfa = 1.0 - math.exp(-dt / TAU_SUAVIZADO_S) if dt > 0 else 0.0
        return previo + alfa * (nuevo - previo)

    # ---------------------------------------------------------------- fases
    def _referencia(self, previo, valor, dt, minimo, tau_sube=REF_TAU_SUBE_S):
        """Referencia que sube rapido y baja muy despacio, nunca por debajo de un minimo anatomico."""
        tau = tau_sube if (previo is None or valor > previo) else REF_TAU_BAJA_S
        return max(minimo, _ema_tiempo(previo, valor, dt, tau))

    def _torso(self, P, vis, dt):
        """Eje del tronco, escala y rotacion del torso usando solo de cadera hacia arriba.

        Devuelve (abajo, escala, de_frente) o None si no se ven los hombros.
          abajo: vector unitario del centro de hombros hacia el centro de caderas
          escala: altura del tronco en unidades de imagen (para normalizar longitudes)
          de_frente: False si el torso esta girado respecto a la camara
        """
        if not self._visibles(vis, (HOMBRO_I, HOMBRO_D)):
            return None
        hombros = P[HOMBRO_I] - P[HOMBRO_D]
        ancho_h = float(np.linalg.norm(hombros))
        if ancho_h < 1e-6:
            return None
        centro_h = (P[HOMBRO_I] + P[HOMBRO_D]) / 2
        caderas_ok = vis[CADERA_I] >= VIS_CADERA_MIN and vis[CADERA_D] >= VIS_CADERA_MIN
        if caderas_ok:
            eje = (P[CADERA_I] + P[CADERA_D]) / 2 - centro_h
            escala = float(np.linalg.norm(eje))
            caderas_ok = escala > 1e-6
        if caderas_ok:
            abajo = eje / escala
        else:
            # sentado con las caderas tapadas: el tronco es perpendicular a la linea de hombros
            perp = np.array([-hombros[1], hombros[0]]) / ancho_h
            abajo = perp if perp[1] > 0 else -perp   # en la imagen "abajo" es y creciente
            escala = ancho_h * TRONCO_POR_HOMBROS

        # rotacion: al girar, los hombros se ven mas estrechos respecto a la altura del tronco
        rotacion = 0.0
        if caderas_ok:
            rel = ancho_h / escala
            self.torso_ref = self._referencia(self.torso_ref, rel, dt, TORSO_REF_MIN, TORSO_REF_TAU_SUBE_S)
            rotacion = math.degrees(math.acos(min(1.0, rel / self.torso_ref)))
        self.torso_rotacion = rotacion
        self.torso_caderas = caderas_ok
        return abajo, escala, rotacion <= TORSO_ROTACION_MAX

    def _fase_1(self, P, img, vis, dt, t):
        # abduccion: angulo del brazo contra el eje del tronco, en el plano frontal
        torso = self._torso(P, vis, dt)
        mejor = None
        for lado, (hombro, codo, muneca) in (
            ("izquierdo", (HOMBRO_I, CODO_I, MUNECA_I)),
            ("derecho", (HOMBRO_D, CODO_D, MUNECA_D)),
        ):
            info = self.abd_info[lado]
            fsm = self.fsm_abd[lado]
            ang = None
            if torso is not None and self._visibles(vis, (hombro, codo)):
                abajo, escala, de_frente = torso
                brazo = P[codo] - P[hombro]
                ang = angulo_vectores(abajo, brazo)
            if ang is None:
                info["ang"], info["en_plano"], info["escorzo"] = None, False, None
                fsm.sin_medida(dt)
                continue

            # escorzo: el brazo en el plano del torso se ve con toda su longitud
            rel = float(np.linalg.norm(brazo)) / escala
            info["ref"] = self._referencia(info["ref"], rel, dt, ABD_BRAZO_REF_MIN)
            escorzo = rel / info["ref"]
            en_plano = de_frente and escorzo >= ABD_ESCORZO_MIN
            info["ang"], info["en_plano"], info["escorzo"] = ang, en_plano, escorzo

            # muneca: brazo entero por encima del umbral y codo estirado
            info["codo"], info["ang_muneca"], valido = None, None, True
            if self._visibles(vis, (muneca,)):
                antebrazo = P[muneca] - P[codo]
                codo_ang = angulo_vectores(-brazo, antebrazo)
                ang_muneca = angulo_vectores(abajo, P[muneca] - P[hombro])
                info["codo"], info["ang_muneca"] = codo_ang, ang_muneca
                if codo_ang is None or ang_muneca is None:
                    valido = False
                else:
                    valido = codo_ang >= CODO_EXTENDIDO_MIN and ang_muneca > ABD_ARRIBA
            info["valido"] = valido

            if fsm.actualizar(ang, en_plano, dt, t, valido):
                self._cerrar_abduccion(lado, t)
            if en_plano and not fsm.paso_delante and (mejor is None or ang > mejor[0]):
                mejor = (ang, lado, hombro, codo, muneca)

        if mejor is None:
            return
        ang, lado, hombro, codo, muneca = mejor
        self.lado = lado
        self.vectores = {
            nombre: (float(img[i][0]), float(img[i][1]), float(vis[i]))
            for nombre, i in (("hombro", hombro), ("codo", codo), ("muneca", muneca))
        }
        self.abd = ang
        self.rom_min = ang if self.rom_min is None else min(self.rom_min, ang)
        self.rom_max = ang if self.rom_max is None else max(self.rom_max, ang)

    def _cerrar_abduccion(self, lado, t):
        # si los dos brazos terminan casi a la vez es una abduccion bilateral: cuenta una vez
        if self.ultima_abd_t is not None and t - self.ultima_abd_t < ABD_FUSION_BILATERAL_S:
            return
        self.ultima_abd_t = t
        self.repeticiones_abduccion += 1
        pico = self.fsm_abd[lado].pico
        self._anotar(t, f"abduccion {self.repeticiones_abduccion} ({lado[0].upper()}, {pico:.0f}°)")
        self.fsm_abd[lado].pico = 0.0

    def _fase_2(self, img, vis, dt, t):  # img: plano XY con aspecto corregido
        # sentadilla: la rodilla mas flexionada
        angulos = []
        for cadera, rodilla, tobillo in ((CADERA_I, RODILLA_I, TOBILLO_I), (CADERA_D, RODILLA_D, TOBILLO_D)):
            if self._visibles(vis, (cadera, rodilla, tobillo)):
                angulos.append(calcular_angulo(img[cadera], img[rodilla], img[tobillo]))
        if not angulos:
            return
        self.sent = self._suavizar(self.sent, min(angulos), dt)
        if not self.sent_abajo and self.sent < SENT_ABAJO:
            self.sent_abajo = True
        elif self.sent_abajo and self.sent > SENT_ARRIBA:
            self.sent_abajo = False
            self.repeticiones_sentadilla += 1
            self._anotar(t, f"sentadilla {self.repeticiones_sentadilla}")

    def _fase_3(self, img, vis, t):
        # compensacion: tronco inclinado lateralmente de forma sostenida
        if not self._visibles(vis, (HOMBRO_I, HOMBRO_D, CADERA_I, CADERA_D)):
            self.inclinado_desde = None
            return
        hx, hy = (img[HOMBRO_I] + img[HOMBRO_D]) / 2
        cx, cy = (img[CADERA_I] + img[CADERA_D]) / 2
        inclinacion = abs(math.degrees(math.atan2(hx - cx, cy - hy)))

        if inclinacion <= INCLINACION_MAX:
            self.inclinado_desde = None
            self.error_activo = False
            return
        if self.inclinado_desde is None:
            self.inclinado_desde = t
        if not self.error_activo and t - self.inclinado_desde >= INCLINACION_PERSISTE_S:
            self.error_activo = True
            if not self.timestamps_errores or t - self.timestamps_errores[-1] >= ERROR_SEPARACION_S:
                self.timestamps_errores.append(round(t, 2))
                self._anotar(t, "compensacion tronco")

    # ---------------------------------------------------------------- salida
    def resultados(self):
        """Diccionario con el esquema exacto exigido por el evaluador."""
        with self.lock:
            return {
                "fase_1": {"repeticiones_abduccion": int(self.repeticiones_abduccion)},
                "fase_2": {"repeticiones_sentadilla": int(self.repeticiones_sentadilla)},
                "fase_3": {
                    "compensaciones_detectadas": bool(self.timestamps_errores),
                    "timestamps_errores_segundos": list(self.timestamps_errores),
                },
            }

    def texto_depuracion(self):
        """Lineas cortas con lo que ve la fase 1 de cada brazo, para pintar sobre el video."""
        with self.lock:
            if self.torso_rotacion > TORSO_ROTACION_MAX:
                lineas = [(f"torso GIRADO {self.torso_rotacion:.0f}deg: ponte de frente", False)]
            else:
                caderas = "" if self.torso_caderas else "  (sin caderas)"
                lineas = [(f"torso de frente ({self.torso_rotacion:.0f}deg){caderas}", True)]
            for lado, letra in (("izquierdo", "I"), ("derecho", "D")):
                info, fsm = self.abd_info[lado], self.fsm_abd[lado]
                if info["ang"] is None:
                    lineas.append((f"{letra}: no visible", False))
                    continue
                plano = "en plano" if info["en_plano"] else "FUERA DE PLANO"
                codo = "codo ?" if info["codo"] is None else (
                    f"codo {info['codo']:.0f}" + ("" if info["codo"] >= CODO_EXTENDIDO_MIN else " DOBLADO"))
                lineas.append((f"{letra}: {info['ang']:3.0f}deg  {plano}  {codo}  {fsm.estado}",
                               info["en_plano"] and info["valido"]))
            lineas.append((f"abducciones: {self.repeticiones_abduccion}", True))
            return lineas

    def instantanea(self):
        """Estado actual para la interfaz en vivo (panel de metricas de app.py)."""
        with self.lock:
            rom = None if self.rom_min is None else round(self.rom_max - self.rom_min)
            if any(f.estado == FSMAbduccion.ARRIBA for f in self.fsm_abd.values()):
                fase = "abduccion"
            elif self.sent_abajo:
                fase = "sentadilla"
            else:
                fase = "reposo"
            return {
                "live": True,
                "pose": self.pose_detectada,
                "angulo": None if self.abd is None else round(self.abd),
                "rom": rom,
                "reps": f"{self.repeticiones_abduccion} / {self.repeticiones_sentadilla}",
                "fase": fase,
                "lado": self.lado,
                "fps": round(self.fps, 1),
                "lat": round(self.latencia_ms, 1),
                "res": self.resolucion,
                "vis": self.visibles,
                "vec": {k: [round(v[0], 2), round(v[1], 2), round(v[2], 2)] for k, v in self.vectores.items()},
                "log": list(self.log)[-7:],
            }


# ==================================================================== depuracion por consola
# No forma parte de la evaluacion (eval.py importa la clase y nunca ejecuta esto).
# Uso dentro del contenedor, con el video copiado en la carpeta app/:
#     docker compose run --rm web python rehab_engine.py mi_video.mp4
# Genera mi_video_depurado.mp4 (esqueleto + texto del motor) y mi_video_depurado.csv
# (lo que mide el motor en cada fotograma) e imprime el resultados.json que daria eval.py.
def _depurar_video(ruta):
    import csv
    import json
    import os

    from utils import obtener_estilos_dibujo

    cap = cv2.VideoCapture(ruta)
    if not cap.isOpened():
        print(f"No se pudo abrir el video: {ruta}")
        return 1
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    ancho, alto = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    base = os.path.splitext(ruta)[0]
    salida = cv2.VideoWriter(base + "_depurado.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (ancho, alto))
    f_csv = open(base + "_depurado.csv", "w", newline="", encoding="utf-8")
    w = csv.writer(f_csv)
    w.writerow(["t", "visibles", "torso_giro", "caderas",
                "I_ang", "I_muneca", "I_codo", "I_escorzo", "I_en_plano", "I_estado",
                "D_ang", "D_muneca", "D_codo", "D_escorzo", "D_en_plano", "D_estado",
                "abducciones", "sentadillas", "rodilla", "errores_tronco"])

    motor = RehabEngine()
    puntos, conexiones = obtener_estilos_dibujo()
    dibujo = mp.solutions.drawing_utils
    ultimo_ms = -1.0

    def r(v):
        return "" if v is None else round(float(v), 1)

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        if t_ms <= ultimo_ms:
            t_ms = ultimo_ms + 1000.0 / fps
        ultimo_ms = t_ms
        lm = motor.procesar_fotograma(frame, t_ms)
        if lm:
            dibujo.draw_landmarks(frame, lm, mp.solutions.pose.POSE_CONNECTIONS, puntos, conexiones)
        escala = max(0.5, alto / 720)
        y = int(30 * escala)
        for texto, valido in [(f"t = {t_ms / 1000:.2f} s", True)] + motor.texto_depuracion():
            color = (140, 255, 150) if valido else (40, 60, 255)
            cv2.putText(frame, texto, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6 * escala, (0, 0, 0),
                        int(4 * escala), cv2.LINE_AA)
            cv2.putText(frame, texto, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6 * escala, color,
                        max(1, int(1.5 * escala)), cv2.LINE_AA)
            y += int(26 * escala)
        salida.write(frame)

        fila = [round(t_ms / 1000, 3), motor.visibles, round(motor.torso_rotacion, 1), int(motor.torso_caderas)]
        for lado in ("izquierdo", "derecho"):
            i = motor.abd_info[lado]
            fila += [r(i["ang"]), r(i["ang_muneca"]), r(i["codo"]), r(i["escorzo"]), int(i["en_plano"]),
                     motor.fsm_abd[lado].estado]
        fila += [motor.repeticiones_abduccion, motor.repeticiones_sentadilla, r(motor.sent),
                 len(motor.timestamps_errores)]
        w.writerow(fila)

    cap.release()
    salida.release()
    f_csv.close()
    print(json.dumps(motor.resultados(), ensure_ascii=False, indent=2))
    print(f"-> {base}_depurado.mp4\n-> {base}_depurado.csv")
    return 0


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Uso: python rehab_engine.py <video.mp4>")
        sys.exit(2)
    sys.exit(_depurar_video(sys.argv[1]))