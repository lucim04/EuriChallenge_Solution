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

Fase 1 en 3D (modo por defecto si MediaPipe da pose_world_landmarks):
  * se construye un sistema de referencia anclado al TORSO con los puntos 3D en metros:
      v = eje vertical (caderas -> hombros), l = eje lateral (hombro der -> izq),
      n = l x v = eje antero-posterior (normal al plano frontal del paciente);
  * elevacion = angulo del brazo (hombro-codo) con el eje "abajo" del torso (ec. 1);
  * plano de elevacion = atan2(|a.n|, a.l) con la componente horizontal del brazo:
      0 grados = abduccion pura (lateral), 90 = flexion (hacia delante/atras),
      180 = cruzando por delante del cuerpo. Solo cuenta si el brazo sube en el plano
      frontal DEL TORSO (no de la camara), asi que un torso girado si cuenta;
  * si no hay puntos 3D se usa el metodo 2D de arriba como respaldo.
"""
import math
import threading
import time
from collections import deque

import cv2
import mediapipe as mp
import numpy as np

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

# fase 1 en 3D (sistema de referencia del torso con pose_world_landmarks)
PLANO_ABD_MAX = 40.0            # plano de elevacion maximo: 0 = lateral puro, 90 = hacia delante
PLANO_ELEV_MIN = 30.0           # por debajo de esta elevacion el plano no esta definido (brazo colgando)
PLANO_T_FUERA_S = 0.12          # permanencia minima fuera de plano para invalidar la subida (ruido)
TORSO_GIRO_MAX_3D = 65.0        # con giro mayor (casi de perfil) el brazo lejano se ve mal: no se mide

# fase 3: compensacion por inclinacion lateral del tronco
INCLINACION_MAX = 15.0       # grados respecto a la vertical
INCLINACION_PERSISTE_S = 0.3  # segundos seguidos para considerarlo error
ERROR_SEPARACION_S = 1.0      # separacion minima entre errores registrados

TAU_SUAVIZADO_S = 0.08  # constante de tiempo del filtro exponencial (fps)


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



# ==================================================================== fase 2: sentadilla
# Integrado tal cual desde fase2_sentadilla.py. Descripcion original:
# Fase 2: conteo de sentadillas con una maquina de estados (FSM) e histeresis temporal.
#
# Sin dependencias de Streamlit ni de MediaPipe (solo math y numpy), para poder probarla
# aislada. El motor (rehab_engine.py) le pasa en cada fotograma los puntos ya filtrados
# en el plano XY (con la relacion de aspecto corregida), su visibilidad y el dt real.
#
# Senal: angulo de flexion de rodilla (180 = pierna recta). Por cada pierna se estiman dos:
#   * 2D: angulo cadera-rodilla-tobillo en la imagen. Bueno de perfil, pero de frente
#     la pierna se sigue viendo recta aunque se doble (el muslo apunta a la camara);
#   * frontal: por el escorzo. Al flexionar, el muslo se inclina phi hacia la camara y la
#     tibia psi hacia delante; sus proyecciones verticales se acortan (L*cos). El cociente
#     r = (altura_muslo / altura_tibia) / referencia_de_pie = cos(phi) / cos(psi) no depende
#     de la distancia a la camara. Con la relacion biomecanica tipica de la sentadilla
#     psi ~ phi * TIBIA_FRACCION se despeja phi, y la flexion de rodilla es phi + psi.
# Se usa el menor de los dos (el que mas flexion ve): de perfil manda el 2D y de frente
# el de escorzo.
#
# FSM:
#     SIN_ARMAR --(de pie T_ARMADO)--> DE_PIE --(empieza a doblar)--> BAJANDO
#     BAJANDO --(pausa: quieto bajo SENT_ABAJO durante T_PAUSA)--> ABAJO
#     BAJANDO --(vuelve a estirar sin pausa abajo)--> DE_PIE              (no cuenta)
#     ABAJO --(rodilla > SENT_ABAJO + banda)--> SUBIENDO
#     SUBIENDO --(vuelve a doblar)--> ABAJO                               (rebote, no cuenta)
#     SUBIENDO --(rodilla > SENT_ARRIBA durante T_MIN_ARRIBA)--> DE_PIE   => +1 sentadilla
#
# Histeresis:
#   * de umbral: se entra en flexion por debajo de SENT_ABAJO y se sale por encima de
#     SENT_ABAJO + SENT_BANDA; la repeticion se cierra solo por encima de SENT_ARRIBA;
#   * temporal: cada transicion exige mantener la condicion un tiempo minimo, sumando los
#     dt reales de los timestamps (T_estado = sum(dt)), nunca contando fotogramas.
#   * pausa (t_min en flexion): no basta con pasar tiempo por debajo de SENT_ABAJO (una
#     sentadilla rapida ya pasa ahi ~0.3 s); en los ultimos T_PAUSA segundos el angulo
#     debe haber estado siempre abajo y casi quieto (variacion <= PAUSA_TOLERANCIA).
PIERNAS = (("I", CADERA_I, RODILLA_I, TOBILLO_I), ("D", CADERA_D, RODILLA_D, TOBILLO_D))

VIS_RODILLA_MIN = 0.5           # rodilla y tobillo deben verse con esta confianza
DESACUERDO_PIERNAS_MAX = 35.0   # si las dos piernas difieren mas, se usa la mas fiable
MUSLO_2D_MIN = 0.4              # muslo proyectado < 0.4 x tibia: apunta a la camara y el angulo
                                # 2D es ruido (cadera y rodilla casi coinciden): no se usa

# estimacion frontal por escorzo del muslo
TIBIA_FRACCION = 1.0 / 3.0      # la tibia se inclina ~1/3 de lo que se inclina el muslo
MUSLO_REF_MIN = 0.6             # minimo anatomico de (altura muslo / tibia) de pie (~1.0)
MUSLO_REF_TAU_SUBE_S = 0.5      # la referencia sube rapido hacia el valor de pie...
MUSLO_REF_TAU_BAJA_S = 20.0     # ...y baja muy despacio (no aprende la flexion como "de pie")
TAU_RATIO_S = 0.05              # suavizado ligero del cociente antes de compararlo

# umbrales de angulo de rodilla (grados, 180 = pierna recta)
SENT_ARRIBA = 160.0     # de pie: pierna estirada
SENT_INICIO = 150.0     # por debajo, ha empezado a bajar
SENT_ABAJO = 115.0      # flexion suficiente (objetivo ~90 grados, con margen permisivo)
SENT_BANDA = 10.0       # histeresis: para salir de abajo hay que superar SENT_ABAJO + banda

# tiempos (segundos, siempre suma de dt reales)
T_ARMADO_S = 0.3        # de pie este tiempo antes de empezar a contar
T_MIN_INICIO_S = 0.08   # flexion sostenida para considerar que baja (no un pico de ruido)
T_PAUSA_S = 0.5         # t_min: pausa abajo, quieto, para que la sentadilla sea valida
PAUSA_TOLERANCIA = 6.0  # grados que puede variar el angulo durante la pausa
T_MIN_SALIDA_S = 0.08   # sostenido por encima de la banda para dar por iniciada la subida
T_MIN_ARRIBA_S = 0.10   # sostenido de pie para cerrar la repeticion
T_MAX_ABAJO_S = 6.0     # mas tiempo abajo no es una sentadilla, es estar sentado
T_HUECO_S = 1.0         # sin ver las piernas este tiempo, la repeticion en curso se pierde

TAU_ANGULO_S = 0.06     # EMA ligera del angulo (los puntos ya vienen filtrados con One Euro)


def _ema(previo, nuevo, dt, tau):
    if previo is None:
        return nuevo
    if dt <= 0:
        return previo
    return previo + (1.0 - math.exp(-dt / tau)) * (nuevo - previo)


def _angulo(a, b, c):
    """Angulo en b (grados) entre los vectores b->a y b->c."""
    u, v = a - b, c - b
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    if nu < 1e-9 or nv < 1e-9:
        return None
    cos = float(np.dot(u, v) / (nu * nv))
    return math.degrees(math.acos(max(-1.0, min(1.0, cos))))


# ==================================================================== senal
# tabla para despejar phi de r = cos(phi) / cos(phi * TIBIA_FRACCION), monotona en [0, 90]
_PHI = np.radians(np.linspace(0.0, 90.0, 181))
_R = np.cos(_PHI) / np.cos(_PHI * TIBIA_FRACCION)


def flexion_por_escorzo(r):
    """Angulo de rodilla (180 = recta) a partir del cociente de escorzo normalizado r."""
    r = min(1.0, max(0.0, r))
    phi = float(np.interp(r, _R[::-1], _PHI[::-1]))
    return 180.0 - math.degrees(phi * (1.0 + TIBIA_FRACCION))


class EstimadorRodilla:
    """Angulo de rodilla combinando 2D y escorzo del muslo, para las dos piernas."""

    def __init__(self):
        self.reiniciar()

    def reiniciar(self):
        self.ratio = {"I": None, "D": None}   # altura muslo / tibia, suavizado
        self.ref = {"I": None, "D": None}     # su valor de pie (referencia)
        self.detalle = {}                     # por pierna: (ang_2d, ang_frontal), para depurar

    def medir(self, P, vis, dt):
        """Devuelve el angulo de rodilla combinado, o None si no se ve ninguna pierna.

        P: array (33, 2) en el plano XY con aspecto corregido (y hacia abajo). vis: (33,).
        """
        piernas = []
        self.detalle = {}
        for lado, cadera, rodilla, tobillo in PIERNAS:
            if vis[cadera] < VIS_CADERA_MIN or vis[rodilla] < VIS_RODILLA_MIN or vis[tobillo] < VIS_RODILLA_MIN:
                self.ratio[lado] = None
                continue
            ang_2d = _angulo(P[cadera], P[rodilla], P[tobillo])
            altura_tibia = float(P[tobillo][1] - P[rodilla][1])
            if ang_2d is None or altura_tibia < 1e-6:
                continue
            # escorzo: solo componentes verticales (de frente la pierna apenas se mueve en x)
            altura_muslo = max(0.0, float(P[rodilla][1] - P[cadera][1]))
            self.ratio[lado] = _ema(self.ratio[lado], altura_muslo / altura_tibia, dt, TAU_RATIO_S)
            ratio = self.ratio[lado]
            previo = self.ref[lado]
            tau = MUSLO_REF_TAU_SUBE_S if (previo is None or ratio > previo) else MUSLO_REF_TAU_BAJA_S
            self.ref[lado] = max(MUSLO_REF_MIN, _ema(previo, ratio, dt, tau))
            ang_frontal = flexion_por_escorzo(ratio / self.ref[lado])
            muslo_2d = float(np.linalg.norm(P[rodilla] - P[cadera]))
            tibia_2d = float(np.linalg.norm(P[tobillo] - P[rodilla]))
            if muslo_2d < MUSLO_2D_MIN * tibia_2d:
                ang_2d = ang_frontal
            self.detalle[lado] = (ang_2d, ang_frontal)
            confianza = float(min(vis[cadera], vis[rodilla], vis[tobillo]))
            piernas.append((min(ang_2d, ang_frontal), confianza))

        if not piernas:
            return None
        if len(piernas) == 2 and abs(piernas[0][0] - piernas[1][0]) > DESACUERDO_PIERNAS_MAX:
            # una de las dos esta mal estimada (oclusion de perfil): la de mas confianza
            return max(piernas, key=lambda p: p[1])[0]
        peso = sum(c for _, c in piernas)
        return sum(a * c for a, c in piernas) / peso


# ==================================================================== maquina de estados
class FSMSentadilla:
    SIN_ARMAR, DE_PIE, BAJANDO, ABAJO, SUBIENDO = "sin armar", "de pie", "bajando", "abajo", "subiendo"

    def __init__(self):
        self.reiniciar()

    def reiniciar(self):
        self.estado = self.SIN_ARMAR
        self.t = 0.0            # reloj interno: suma de dt
        self.t_cond = 0.0       # tiempo acumulado cumpliendo la condicion de la transicion
        self.t_abajo = 0.0      # tiempo total en flexion de la repeticion en curso
        self.t_sin_ver = 0.0
        self.angulo = None      # angulo suavizado
        self.minimo = None      # angulo minimo (flexion maxima) de la repeticion en curso
        self.ultimo_minimo = None
        self.ventana = deque()  # (t, angulo) de los ultimos T_PAUSA_S, para detectar la pausa
        self.pausa = 0.0        # cuanto lleva quieto abajo (para depurar)
        self.evento = None      # texto del ultimo evento, para el registro del motor

    def _cambiar(self, estado):
        self.estado, self.t_cond = estado, 0.0

    def sin_medida(self, dt):
        """Las piernas no se ven en este fotograma."""
        self.t += dt
        self.t_sin_ver += dt
        self.t_cond = 0.0
        if self.t_sin_ver >= T_HUECO_S and self.estado != self.SIN_ARMAR:
            self.evento = "sentadilla: piernas perdidas, hay que volver a ponerse de pie"
            self._cambiar(self.SIN_ARMAR)
            self.angulo = None

    def _en_pausa(self, a):
        """True si en los ultimos T_PAUSA_S el angulo ha estado abajo y casi quieto."""
        self.ventana.append((self.t, a))
        # se conserva una muestra de mas antigua que T_PAUSA_S para saber si la ventana esta completa
        while len(self.ventana) > 1 and self.t - self.ventana[1][0] >= T_PAUSA_S:
            self.ventana.popleft()
        angulos = [x for _, x in self.ventana]
        if max(angulos) >= SENT_ABAJO or max(angulos) - min(angulos) > PAUSA_TOLERANCIA:
            # buscar desde cuando esta quieto abajo (solo para mostrarlo)
            self.pausa = 0.0
            return False
        self.pausa = self.t - self.ventana[0][0]
        return self.pausa >= T_PAUSA_S

    def actualizar(self, angulo, dt):
        """Avanza la FSM con el angulo de rodilla del fotograma. Devuelve True al completar una."""
        self.evento = None
        self.t += dt
        self.t_sin_ver = 0.0
        self.angulo = _ema(self.angulo, angulo, dt, TAU_ANGULO_S)
        a = self.angulo

        if self.estado == self.SIN_ARMAR:
            self.t_cond = self.t_cond + dt if a > SENT_ARRIBA else 0.0
            if self.t_cond >= T_ARMADO_S:
                self._cambiar(self.DE_PIE)
            return False

        if self.estado == self.DE_PIE:
            self.t_cond = self.t_cond + dt if a < SENT_INICIO else 0.0
            if self.t_cond >= T_MIN_INICIO_S:
                self._cambiar(self.BAJANDO)
                self.minimo, self.t_abajo, self.pausa = a, 0.0, 0.0
                self.ventana.clear()
            return False

        self.minimo = a if self.minimo is None else min(self.minimo, a)

        if self.estado == self.BAJANDO:
            if self._en_pausa(a):
                self.t_abajo = self.pausa
                self._cambiar(self.ABAJO)
            elif a > SENT_ARRIBA:
                # se ha vuelto a estirar sin pausa abajo: no cuenta
                motivo = "sin pausa abajo" if self.minimo < SENT_ABAJO else "poco profunda"
                self.evento = f"sentadilla no valida: {motivo} ({self.minimo:.0f}°)"
                self._cambiar(self.DE_PIE)
            return False

        if self.estado == self.ABAJO:
            self.t_abajo += dt
            if self.t_abajo > T_MAX_ABAJO_S:
                # demasiado tiempo abajo: esta sentado. Hay que ponerse de pie de nuevo
                self.evento = "sentadilla descartada: demasiado tiempo abajo (sentado)"
                self._cambiar(self.SIN_ARMAR)
                return False
            self.t_cond = self.t_cond + dt if a > SENT_ABAJO + SENT_BANDA else 0.0
            if self.t_cond >= T_MIN_SALIDA_S:
                self._cambiar(self.SUBIENDO)
            return False

        # SUBIENDO
        if a < SENT_ABAJO:
            self._cambiar(self.ABAJO)   # rebote: vuelve a la flexion, sigue siendo la misma
            return False
        self.t_cond = self.t_cond + dt if a > SENT_ARRIBA else 0.0
        if self.t_cond >= T_MIN_ARRIBA_S:
            self.ultimo_minimo = self.minimo
            self.minimo = None
            self._cambiar(self.DE_PIE)
            return True
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
            self.filtro_mundo = FiltroOneEuro()
            self.modo_3d = False
            # fase 1
            self.abd = None
            # por brazo: angulo, si esta en el plano frontal, escorzo y longitud de referencia
            self.torso_ref = None
            self.torso_rotacion = 0.0
            self.torso_caderas = False
            self.abd_info = {lado: {"ang": None, "en_plano": False, "escorzo": None, "ref": None,
                                    "codo": None, "ang_muneca": None, "valido": True,
                                    "plano": None, "t_fuera": 0.0}
                             for lado in ("izquierdo", "derecho")}
            self.fsm_abd = {"izquierdo": FSMAbduccion(), "derecho": FSMAbduccion()}
            self.ultima_abd_t = None
            self.repeticiones_abduccion = 0
            # fase 2
            self.sent = None
            self.est_rodilla = EstimadorRodilla()
            self.fsm_sent = FSMSentadilla()
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
        mundo = res.pose_world_landmarks.landmark if res.pose_world_landmarks else None
        alto, ancho = frame_bgr.shape[:2]
        with self.lock:
            self.aspecto = ancho / alto if alto else 1.0
            self.actualizar(lm, t_ms, mundo)
            self.resolucion = f"{ancho}x{alto}"
            self.latencia_ms = (time.perf_counter() - inicio) * 1000.0
        return res.pose_landmarks

    def actualizar(self, lm, t_ms, mundo=None):
        """Actualiza la logica con 33 landmarks de imagen (o None) y el tiempo en ms.

        `mundo`: los 33 pose_world_landmarks de MediaPipe (metros, origen en la cadera), opcional.
        """
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
                self.fsm_sent.sin_medida(dt)
                if self.fsm_sent.evento:
                    self._anotar(t, self.fsm_sent.evento)
                return
            if not self.pose_detectada:
                self._anotar(t, "pose detectada")
            self.pose_detectada = True

            # tras un hueco largo sin pose el filtro arranca de cero (no arrastra posiciones viejas)
            if self.t_ultima_pose is None or t - self.t_ultima_pose > FILTRO_HUECO_S:
                self.filtro_img.reiniciar()
                self.filtro_mundo.reiniciar()
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

            if mundo is not None:
                W = self.filtro_mundo(np.array([[p.x, p.y, p.z] for p in mundo]), dt_f)
                self.modo_3d = True
                self._fase_1_3d(W, img, vis, dt_f, t)
            else:
                self.modo_3d = False
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

    @staticmethod
    def _ejes_torso(W):
        """Sistema de referencia del torso (vectores unitarios) a partir de los puntos 3D.

        Devuelve (v, l, n, giro): v hacia arriba, l lateral (hacia el hombro izquierdo),
        n normal al plano frontal del torso, giro = rotacion del torso respecto a la camara.
        """
        centro_h = (W[HOMBRO_I] + W[HOMBRO_D]) / 2
        centro_c = (W[CADERA_I] + W[CADERA_D]) / 2
        v = centro_h - centro_c
        nv = np.linalg.norm(v)
        if nv < 1e-6:
            return None
        v = v / nv
        l = W[HOMBRO_I] - W[HOMBRO_D]
        l = l - np.dot(l, v) * v          # ortogonal a v
        nl = np.linalg.norm(l)
        if nl < 1e-6:
            return None
        l = l / nl
        n = np.cross(l, v)
        # giro: angulo de la linea de hombros con el eje x de la camara, en el plano horizontal
        giro = math.degrees(math.atan2(abs(l[2]), abs(l[0])))
        return v, l, n, giro

    def _fase_1_3d(self, W, img, vis, dt, t):
        """Abduccion medida en el sistema de referencia del torso (invariante a la rotacion)."""
        ejes = None
        if self._visibles(vis, (HOMBRO_I, HOMBRO_D)) and \
                vis[CADERA_I] >= VIS_CADERA_MIN and vis[CADERA_D] >= VIS_CADERA_MIN:
            ejes = self._ejes_torso(W)
        self.torso_caderas = ejes is not None
        if ejes is not None:
            v, l, n, giro = ejes
            self.torso_rotacion = giro
        mejor = None
        for lado, signo, (hombro, codo, muneca) in (
            ("izquierdo", 1.0, (HOMBRO_I, CODO_I, MUNECA_I)),
            ("derecho", -1.0, (HOMBRO_D, CODO_D, MUNECA_D)),
        ):
            info = self.abd_info[lado]
            fsm = self.fsm_abd[lado]
            if ejes is None or giro > TORSO_GIRO_MAX_3D or not self._visibles(vis, (hombro, codo)):
                info["ang"], info["en_plano"], info["plano"], info["t_fuera"] = None, False, None, 0.0
                fsm.sin_medida(dt)
                continue

            brazo = W[codo] - W[hombro]
            ang = angulo_vectores(-v, brazo)          # elevacion respecto al eje del torso
            if ang is None:
                fsm.sin_medida(dt)
                continue

            # plano de elevacion: con la componente horizontal (perpendicular a v) del brazo
            plano = None
            en_plano_ahora = True
            if ang >= PLANO_ELEV_MIN:
                lateral = float(np.dot(brazo, l)) * signo   # > 0 = hacia fuera de su lado
                frontal = abs(float(np.dot(brazo, n)))
                plano = math.degrees(math.atan2(frontal, lateral))   # 0..180
                en_plano_ahora = plano <= PLANO_ABD_MAX
            # un solo fotograma fuera de plano puede ser ruido: exige permanencia (suma de dt)
            info["t_fuera"] = 0.0 if en_plano_ahora else info["t_fuera"] + dt
            en_plano = en_plano_ahora or info["t_fuera"] < PLANO_T_FUERA_S
            info["ang"], info["en_plano"], info["plano"], info["escorzo"] = ang, en_plano, plano, None

            # muneca: brazo entero por encima del umbral y codo estirado (en 3D)
            info["codo"], info["ang_muneca"], valido = None, None, True
            if self._visibles(vis, (muneca,)):
                antebrazo = W[muneca] - W[codo]
                codo_ang = angulo_vectores(-brazo, antebrazo)
                ang_muneca = angulo_vectores(-v, W[muneca] - W[hombro])
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

    def _fase_2(self, P, vis, dt, t):
        """Sentadilla: angulo de rodilla (2D + escorzo) y FSM con pausa minima abajo."""
        ang = self.est_rodilla.medir(P, vis, dt)
        if ang is None:
            self.fsm_sent.sin_medida(dt)
        elif self.fsm_sent.actualizar(ang, dt):
            self.repeticiones_sentadilla += 1
            self._anotar(t, f"sentadilla {self.repeticiones_sentadilla} ({self.fsm_sent.ultimo_minimo:.0f}°)")
        if self.fsm_sent.evento:
            self._anotar(t, self.fsm_sent.evento)
        self.sent = self.fsm_sent.angulo

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
            giro_max = TORSO_GIRO_MAX_3D if self.modo_3d else TORSO_ROTACION_MAX
            modo = "3D" if self.modo_3d else "2D"
            if self.torso_rotacion > giro_max:
                lineas = [(f"[{modo}] torso GIRADO {self.torso_rotacion:.0f}deg: no se mide", False)]
            else:
                caderas = "" if self.torso_caderas else "  (sin caderas)"
                lineas = [(f"[{modo}] torso giro {self.torso_rotacion:.0f}deg{caderas}", True)]
            for lado, letra in (("izquierdo", "I"), ("derecho", "D")):
                info, fsm = self.abd_info[lado], self.fsm_abd[lado]
                if info["ang"] is None:
                    lineas.append((f"{letra}: no visible", False))
                    continue
                plano = "en plano" if info["en_plano"] else "FUERA DE PLANO"
                if info.get("plano") is not None:
                    plano += f" ({info['plano']:.0f})"
                codo = "codo ?" if info["codo"] is None else (
                    f"codo {info['codo']:.0f}" + ("" if info["codo"] >= CODO_EXTENDIDO_MIN else " DOBLADO"))
                lineas.append((f"{letra}: {info['ang']:3.0f}deg  {plano}  {codo}  {fsm.estado}",
                               info["en_plano"] and info["valido"]))
            lineas.append((f"abducciones: {self.repeticiones_abduccion}", True))
            f2 = self.fsm_sent
            rodilla = "rodilla ?" if f2.angulo is None else f"rodilla {f2.angulo:.0f}deg"
            lineas.append((f"sentadilla: {f2.estado}  {rodilla}  pausa {f2.pausa:.1f}s  "
                           f"reps {self.repeticiones_sentadilla}", f2.angulo is not None))
            return lineas

    def instantanea(self):
        """Estado actual para la interfaz en vivo (panel de metricas de app.py)."""
        with self.lock:
            rom = None if self.rom_min is None else round(self.rom_max - self.rom_min)
            if any(f.estado == FSMAbduccion.ARRIBA for f in self.fsm_abd.values()):
                fase = "abduccion"
            elif self.fsm_sent.estado in (FSMSentadilla.BAJANDO, FSMSentadilla.ABAJO, FSMSentadilla.SUBIENDO):
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
                "I_ang", "I_muneca", "I_codo", "I_escorzo", "I_plano", "I_en_plano", "I_estado",
                "D_ang", "D_muneca", "D_codo", "D_escorzo", "D_plano", "D_en_plano", "D_estado",
                "abducciones", "sentadillas", "rodilla", "sent_estado", "sent_pausa", "errores_tronco"])

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
            fila += [r(i["ang"]), r(i["ang_muneca"]), r(i["codo"]), r(i["escorzo"]), r(i["plano"]), int(i["en_plano"]),
                     motor.fsm_abd[lado].estado]
        fila += [motor.repeticiones_abduccion, motor.repeticiones_sentadilla, r(motor.sent),
                 motor.fsm_sent.estado, round(motor.fsm_sent.pausa, 2), len(motor.timestamps_errores)]
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