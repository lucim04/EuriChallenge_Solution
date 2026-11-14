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
# De frente la abduccion ocurre en el plano de la imagen, donde el angulo 2D es casi exacto, mientras
# que la profundidad estimada por MediaPipe suele "bajar" el brazo 3D unos grados (a 90 reales leia <80).
# Con el torso girado como mucho esto y el brazo en plano, la elevacion es la mayor de las dos (2D y 3D).
# Que el brazo va hacia delante lo sigue decidiendo el plano de elevacion en 3D.
ABD_GIRO_2D_MAX = 35.0

# fase 3: compensacion por inclinacion lateral del tronco
INCLINACION_MAX = 15.0       # inclinacion LATERAL maxima del tronco (grados, plano frontal del cuerpo)
INCLINACION_PERSISTE_S = 0.3  # segundos seguidos (suma de dt) para considerarlo error (delta t)
ERROR_SEPARACION_S = 1.0      # dos errores a menos de esto son el mismo episodio
# que segundo se apunta de cada episodio: "inicio" (cuando empezo) o "confirmacion" (cuando
# llevaba INCLINACION_PERSISTE_S, que es lo que hacia la plantilla)
MODO_TIMESTAMP = "inicio"
TAU_FASE3_S = 0.10            # EMA de las senales de la fase 3 (los puntos ya van con One Euro)
FASE3_GIRO_MAX = 60.0         # mas girado (casi de perfil) la inclinacion lateral no se ve: no se mide
# asimetria escapular (encogimiento): distancia vertical oreja-hombro de cada lado, medida a lo
# largo del eje del tronco, comparada con su valor en reposo (brazo abajo)
HOMBRO_RATIO_MIN = 0.72       # por debajo, ese hombro se ha subido hacia la oreja
HOMBRO_RATIO_MIN_BRAZO_ARRIBA = 0.65  # con el brazo por encima de 90 grados el hombro sube un poco
                                      # de forma natural (rotacion de la escapula): mas permisivo
HOMBRO_ASIMETRIA_MIN = 0.12   # y ademas ese lado debe estar esto por debajo del otro (asimetria)
HOMBRO_PERSISTE_S = 0.3
HOMBRO_REF_TAU_SUBE_S = 0.5   # referencia de reposo: sube rapido...
HOMBRO_REF_TAU_BAJA_S = 10.0  # ...y baja despacio (un encogimiento no se aprende como reposo)
SENT_TRONCO_COMPENSACION = True  # la sentadilla con el tronco demasiado inclinado se apunta tambien
# Reparto de responsabilidades:
#   * la fase 3 DETECTA las compensaciones y apunta sus timestamps (no toca ningun contador);
#   * las fases 1 y 2 miran si hubo compensacion durante SU repeticion y, si la hubo, no la cuentan.
#     Abduccion: inclinacion lateral u hombro encogido desde que el brazo sube hasta que vuelve abajo
#     (por eso la abduccion se cuenta al TERMINARLA, cuando el brazo vuelve abajo).
#     Sentadilla: inclinacion lateral desde que empieza a bajar hasta que vuelve a estar de pie.
VALIDAR_COMPENSACIONES = True
COMPENSACION_HOMBRO_EN_SENTADILLA = False  # el hombro encogido solo invalida abducciones

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
# Integrado desde fase2_sentadilla.py con cambios (ver constantes USAR_RODILLA_3D, MODO_T_MIN,
# SENT_MIN_VALIDO, SENT_AMBAS_PIERNAS, SENT_PIE_MARGEN, SENT_ZANCADA_MAX, SENT_TRONCO_MAX y SENT_PIE_LEVANTADO_MAX). Descripcion original:
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

# ---- cambios v2 (sobre la version original de la fase 2) ----
# 1) Angulo de rodilla en 3D con pose_world_landmarks (igual de frente que de perfil).
#    El 2D y el escorzo quedan de respaldo si no hay puntos 3D.
USAR_RODILLA_3D = True
# 2) t_min en flexion: "acumulado" = basta con sumar T_MIN_ABAJO_S por debajo de SENT_ABAJO
#    (sentadillas fluidas, sin parar abajo); "pausa" = criterio original (quieta T_PAUSA_S).
MODO_T_MIN = "acumulado"
T_MIN_ABAJO_S = 0.25    # tiempo minimo acumulado (suma de dt) por debajo de SENT_ABAJO
# 3) Profundidad maxima: por debajo de este angulo de rodilla es una cuclilla, no cuenta.
SENT_MIN_VALIDO = 60.0
# 4) Las dos piernas deben doblarse: si las dos se ven bien, manda la MENOS flexionada
#    (asi apoyar el peso en una pierna o doblar solo una no cuenta como sentadilla).
SENT_AMBAS_PIERNAS = True
VIS_AMBAS_MIN = 0.65        # confianza minima de rodilla y tobillo de cada pierna para exigir las
                            # dos (la cadera no cuenta: de frente MediaPipe le da poca confianza).
                            # De perfil la pierna del fondo se ve peor: entonces manda la fiable.
# 5) "De pie" relativo a cada persona: el angulo de rodilla de pie no siempre se lee como
#    160-180 (de frente, MediaPipe suele dar 150-165 con la pierna recta). Se aprende el angulo
#    de pie de la persona (sube rapido, baja muy despacio) y los umbrales de pie se fijan respecto
#    a el, nunca por debajo de un minimo.
SENT_PIE_MARGEN = 12.0      # cierra la repeticion al volver a menos de esto del angulo de pie
SENT_ARRIBA_MIN = 145.0     # umbral de pie minimo (nunca se cierra una repeticion por debajo)
SENT_INICIO_MARGEN = 8.0    # empieza a bajar cuando cae esto por debajo del umbral de pie
SENT_PIE_TAU_SUBE_S = 0.3
SENT_PIE_TAU_BAJA_S = 30.0
# 6) Zancada: en una sentadilla los pies estan a la par; en una zancada uno va delante y otro
#    detras. Se mide la separacion delante-detras de los tobillos (eje antero-posterior del
#    cuerpo, en 3D) dividida por la longitud de la pierna. Si se mantiene por encima del
#    umbral durante la repeticion, no cuenta.
SENT_ZANCADA_MAX = 0.35     # pies separados delante-detras mas de ~1/3 de pierna = zancada
SENT_ZANCADA_T_S = 0.20     # mantenido este tiempo (suma de dt)
# 7) Tronco demasiado inclinado ("casi a gatas con las rodillas dobladas"): solo se comprueba
#    cuando la persona esta DE PERFIL (o casi), porque entonces la inclinacion se ve en el plano de
#    la imagen y se mide sin depender de la profundidad estimada. De frente el tronco se inclina
#    hacia la camara y MediaPipe no lo mide bien (daba falsos positivos), asi que no se comprueba.
#    Referencia: en una sentadilla correcta el tronco va ~paralelo a la tibia (30-45 grados).
SENT_TRONCO_MAX = 55.0      # inclinacion maxima del tronco respecto a la vertical (grados)
SENT_TRONCO_T_S = 0.20      # mantenida este tiempo (suma de dt)
SENT_PERFIL_GIRO_MIN = 50.0 # giro del cuerpo respecto a la camara para considerarlo de perfil
SENT_PERFIL_ANCHO_MAX = 0.35  # sin 3D: anchura aparente de caderas+hombros / altura del tronco
# 8) Tronco inclinado DE FRENTE: hacia la camara no se ve el angulo, pero el tronco se ve mas corto
#    (escorzo): altura vertical aparente del tronco / anchura de hombros = cos(inclinacion) x su valor de pie.
#    El valor de pie se aprende mientras la persona esta de pie. Para evitar falsos positivos solo
#    invalida si ademas la inclinacion 3D (pose_world_landmarks) lo confirma.
SENT_TRONCO_FRENTE = True
SENT_TRONCO_FRENTE_MAX = 60.0   # de frente un poco mas permisivo (la perspectiva exagera el escorzo)
SENT_TRONCO_3D_CONFIRMA = 40.0  # la inclinacion 3D debe pasar al menos de esto para confirmarlo
SENT_TRONCO_REF_TAU_S = 1.0     # aprendizaje del valor de pie (solo mientras esta de pie)
# 9) Pierna levantada: en una sentadilla los DOS pies estan en el suelo. Si un tobillo queda muy por
#    encima del otro (diferencia de altura / longitud de pierna), es levantar la pierna, una marcha
#    o un paso, no una sentadilla. Se mide en vertical, asi que vale de frente, de perfil y girado.
SENT_PIE_LEVANTADO_MAX = 0.20   # ~1/5 de pierna (unos 15-20 cm); talones algo levantados no llegan
SENT_PIE_LEVANTADO_T_S = 0.15   # mantenido este tiempo (suma de dt)

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
        self.detalle = {}                     # por pierna: (ang_2d, ang_frontal[, ang_3d]), para depurar
        self.usa_3d = False

    def medir(self, P, vis, dt, W=None):
        """Devuelve el angulo de rodilla combinado, o None si no se ve ninguna pierna.

        P: array (33, 2) en el plano XY con aspecto corregido (y hacia abajo). vis: (33,).
        W: array (33, 3) de pose_world_landmarks (metros), opcional. Si esta y USAR_RODILLA_3D,
           el angulo de cada pierna es el 3D cadera-rodilla-tobillo (el 2D y el escorzo se
           siguen calculando para depurar y para mantener caliente la referencia de pie).
        """
        self.usa_3d = False
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
            ang_pierna = min(ang_2d, ang_frontal)
            if W is not None and USAR_RODILLA_3D:
                ang_3d = _angulo(W[cadera], W[rodilla], W[tobillo])
                if ang_3d is not None:
                    ang_pierna = ang_3d
                    self.usa_3d = True
                    self.detalle[lado] = (ang_2d, ang_frontal, ang_3d)
            piernas.append((ang_pierna, confianza, float(min(vis[rodilla], vis[tobillo]))))

        if not piernas:
            return None
        if SENT_AMBAS_PIERNAS and len(piernas) == 2 and min(p[2] for p in piernas) >= VIS_AMBAS_MIN:
            return max(p[0] for p in piernas)   # la menos flexionada: tienen que doblarse las dos
        if len(piernas) == 2 and abs(piernas[0][0] - piernas[1][0]) > DESACUERDO_PIERNAS_MAX:
            # una de las dos esta mal estimada (oclusion de perfil): la de mas confianza
            return max(piernas, key=lambda p: p[1])[0]
        peso = sum(p[1] for p in piernas)
        return sum(p[0] * p[1] for p in piernas) / peso


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
        self.t_bajo = 0.0       # tiempo acumulado por debajo de SENT_ABAJO en la bajada (modo acumulado)
        self.evento = None      # texto del ultimo evento, para el registro del motor
        self.ref_pie = None     # angulo de rodilla de pie de esta persona (aprendido)
        self.arriba = SENT_ARRIBA
        self.inicio = SENT_INICIO

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
            self.ref_pie = None

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
        # angulo de pie de la persona y umbrales relativos a el
        tau = SENT_PIE_TAU_SUBE_S if (self.ref_pie is None or a > self.ref_pie) else SENT_PIE_TAU_BAJA_S
        self.ref_pie = _ema(self.ref_pie, a, dt, tau)
        self.arriba = min(SENT_ARRIBA, max(SENT_ARRIBA_MIN, self.ref_pie - SENT_PIE_MARGEN))
        self.inicio = min(SENT_INICIO, self.arriba - SENT_INICIO_MARGEN)

        if self.estado == self.SIN_ARMAR:
            self.t_cond = self.t_cond + dt if a > self.arriba else 0.0
            if self.t_cond >= T_ARMADO_S:
                self._cambiar(self.DE_PIE)
            return False

        if self.estado == self.DE_PIE:
            self.t_cond = self.t_cond + dt if a < self.inicio else 0.0
            if self.t_cond >= T_MIN_INICIO_S:
                self._cambiar(self.BAJANDO)
                self.minimo, self.t_abajo, self.pausa, self.t_bajo = a, 0.0, 0.0, 0.0
                self.ventana.clear()
            return False

        self.minimo = a if self.minimo is None else min(self.minimo, a)

        if self.estado == self.BAJANDO:
            if MODO_T_MIN == "acumulado":
                if a < SENT_ABAJO:
                    self.t_bajo += dt
                self.pausa = self.t_bajo        # se muestra como "pausa" en la depuracion
                listo = self.t_bajo >= T_MIN_ABAJO_S
            else:
                listo = self._en_pausa(a)
            if listo:
                self.t_abajo = self.pausa
                self._cambiar(self.ABAJO)
            elif a > self.arriba:
                # se ha vuelto a estirar sin pausa abajo: no cuenta
                motivo = ("sin pausa abajo" if MODO_T_MIN == "pausa" else "muy breve abajo") \
                    if self.minimo < SENT_ABAJO else "poco profunda"
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
        self.t_cond = self.t_cond + dt if a > self.arriba else 0.0
        if self.t_cond >= T_MIN_ARRIBA_S:
            self.ultimo_minimo = self.minimo
            self.minimo = None
            self._cambiar(self.DE_PIE)
            if self.ultimo_minimo is not None and self.ultimo_minimo < SENT_MIN_VALIDO:
                self.evento = f"sentadilla no valida: demasiado profunda, cuclillas ({self.ultimo_minimo:.0f}°)"
                return False
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
                                    "plano": None, "t_fuera": 0.0, "ang3d": None, "ang2d": None}
                             for lado in ("izquierdo", "derecho")}
            self.fsm_abd = {"izquierdo": FSMAbduccion(), "derecho": FSMAbduccion()}
            self.ultima_abd_t = None
            self.repeticiones_abduccion = 0
            # fase 2
            self.sent = None
            self.est_rodilla = EstimadorRodilla()
            self.fsm_sent = FSMSentadilla()
            self.sent_estado_prev = FSMSentadilla.SIN_ARMAR
            self.zancada = None          # separacion delante-detras de los pies / pierna (depuracion)
            self.zancada_t = 0.0
            self.zancada_max = 0.0
            self.zancada_mal = False
            self.tronco = None           # inclinacion del tronco (solo de perfil), depuracion
            self.de_perfil = False
            self.tronco_t = 0.0
            self.tronco_max = 0.0
            self.tronco_mal = False
            self.tronco_ref = None       # altura del tronco / anchura de hombros, de pie (de frente)
            self.tronco_modo = None      # "perfil" o "frente" (depuracion)
            self.pie_alto = None         # diferencia de altura de los tobillos / pierna (depuracion)
            self.pie_alto_t = 0.0
            self.pie_alto_max = 0.0
            self.pie_alto_mal = False
            self.repeticiones_sentadilla = 0
            # fase 3
            self.timestamps_errores = []
            self.motivos_errores = []        # motivo de cada timestamp (registro y depuracion)
            # episodios en curso: senal -> {"t": tiempo cumpliendo, "activo": ya registrado}
            self.episodios = {k: {"t": 0.0, "activo": False}
                              for k in ("lateral", "hombro_I", "hombro_D", "tronco_sent")}
            self.f3_lateral = None           # inclinacion lateral suavizada (grados)
            self.f3_hombro = {"I": None, "D": None}       # distancia oreja-hombro / reposo
            self.f3_hombro_ref = {"I": None, "D": None}   # distancia oreja-hombro en reposo
            self.f3_modo = None              # "3D" o "2D": si cambia, las referencias se reinician
            # abduccion: repeticion de cada brazo que ha llegado arriba y espera a cerrarse abajo,
            # y compensacion vista durante la repeticion en curso de ese brazo
            self.abd_pend = {"izquierdo": None, "derecho": None}
            self.abd_comp = {"izquierdo": "", "derecho": ""}
            self.sent_comp = ""              # motivo si hubo compensacion en la sentadilla en curso

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
                self._fase3_cortar()
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

            W = None
            if mundo is not None:
                W = self.filtro_mundo(np.array([[p.x, p.y, p.z] for p in mundo]), dt_f)
            self.modo_3d = W is not None
            # fase 3 primero: detecta compensaciones y apunta timestamps; las fases 1 y 2 las consultan
            self._fase_3(P, vis, dt_f, t, W)
            if W is not None:
                self._fase_1_3d(W, img, vis, dt_f, t)
            else:
                self._fase_1(P, img, vis, dt_f, t)
            self._abd_revisar(t)
            self._fase_2(P, vis, dt_f, t, W)

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

            # de frente: la elevacion 2D (exacta en el plano de la imagen) corrige al 3D si este se queda corto
            P2 = img * np.array([self.aspecto, 1.0])
            usar_2d = giro <= ABD_GIRO_2D_MAX and en_plano_ahora
            abajo_2d = (P2[CADERA_I] + P2[CADERA_D]) / 2 - (P2[HOMBRO_I] + P2[HOMBRO_D]) / 2
            ang_2d = angulo_vectores(abajo_2d, P2[codo] - P2[hombro]) if usar_2d else None
            info["ang3d"], info["ang2d"] = ang, ang_2d
            if ang_2d is not None and ang_2d > ang:
                ang = ang_2d
                info["ang"] = ang

            # muneca: brazo entero por encima del umbral y codo estirado (en 3D)
            info["codo"], info["ang_muneca"], valido = None, None, True
            if self._visibles(vis, (muneca,)):
                antebrazo = W[muneca] - W[codo]
                codo_ang = angulo_vectores(-brazo, antebrazo)
                ang_muneca = angulo_vectores(-v, W[muneca] - W[hombro])
                if usar_2d:
                    m2 = angulo_vectores(abajo_2d, P2[muneca] - P2[hombro])
                    if m2 is not None and ang_muneca is not None:
                        ang_muneca = max(ang_muneca, m2)
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
        """El brazo ha llegado a 90 grados: la repeticion queda pendiente hasta que vuelva abajo."""
        self.abd_pend[lado] = {"t": t, "pico": self.fsm_abd[lado].pico}

    def _abd_revisar(self, t):
        """Cierra las abducciones: cuentan al volver el brazo abajo si no hubo compensacion."""
        for lado in ("izquierdo", "derecho"):
            fsm, info = self.fsm_abd[lado], self.abd_info[lado]
            en_rep = fsm.estado == FSMAbduccion.ARRIBA or (info["ang"] is not None and info["ang"] > ABD_ABAJO)
            if en_rep and VALIDAR_COMPENSACIONES and not self.abd_comp[lado]:
                self.abd_comp[lado] = self._compensacion_activa(hombro=True)
            pend = self.abd_pend[lado]
            if pend is not None and fsm.estado == FSMAbduccion.ABAJO:        # rearmado: repeticion completa
                self._contar_abduccion(lado, pend, t)
                self.abd_pend[lado], self.abd_comp[lado] = None, ""
            elif pend is not None and fsm.estado == FSMAbduccion.SIN_ARMAR:  # brazo perdido mucho rato
                self._anotar(t, f"abduccion descartada: brazo {lado} perdido")
                self.abd_pend[lado], self.abd_comp[lado] = None, ""
            elif pend is None and not en_rep:
                self.abd_comp[lado] = ""

    def _contar_abduccion(self, lado, pend, t):
        # si los dos brazos terminan casi a la vez es una abduccion bilateral: cuenta una vez
        if self.ultima_abd_t is not None and t - self.ultima_abd_t < ABD_FUSION_BILATERAL_S:
            return
        self.ultima_abd_t = t
        self.fsm_abd[lado].pico = 0.0
        if self.abd_comp[lado]:
            self._anotar(t, f"abduccion no valida: compensacion ({self.abd_comp[lado]})")
            return
        self.repeticiones_abduccion += 1
        self._anotar(t, f"abduccion {self.repeticiones_abduccion} ({lado[0].upper()}, {pend['pico']:.0f}°)")

    def _abd_pendientes_validas(self):
        """Abducciones que llegaron arriba sin compensacion y aun no han bajado (fin del video)."""
        ts = sorted(p["t"] for l, p in self.abd_pend.items() if p is not None and not self.abd_comp[l])
        if not ts:
            return 0
        if self.ultima_abd_t is not None and abs(ts[0] - self.ultima_abd_t) < ABD_FUSION_BILATERAL_S:
            ts = ts[1:]
        return 0 if not ts else (1 if len(ts) == 1 or ts[-1] - ts[0] < ABD_FUSION_BILATERAL_S else 2)

    @staticmethod
    def _separacion_pies(P, vis, W):
        """Separacion delante-detras de los tobillos dividida por la longitud media de la pierna.

        En 3D usa el eje antero-posterior del cuerpo (normal a la linea de caderas, horizontal),
        asi que vale de frente, de perfil y girado. Sin 3D solo puede mirar la diferencia de
        altura de los tobillos en la imagen (de frente, el pie adelantado se ve mas abajo).
        """
        ids = (CADERA_I, CADERA_D, RODILLA_I, RODILLA_D, TOBILLO_I, TOBILLO_D)
        if any(vis[i] < VIS_CADERA_MIN for i in ids):
            return None
        Q = W if W is not None else P
        pierna = (np.linalg.norm(Q[CADERA_I] - Q[RODILLA_I]) + np.linalg.norm(Q[RODILLA_I] - Q[TOBILLO_I]) +
                  np.linalg.norm(Q[CADERA_D] - Q[RODILLA_D]) + np.linalg.norm(Q[RODILLA_D] - Q[TOBILLO_D])) / 2
        if pierna < 1e-6:
            return None
        if W is None:
            return abs(float(P[TOBILLO_I][1] - P[TOBILLO_D][1])) / pierna
        arriba = np.array([0.0, -1.0, 0.0])
        lat = W[CADERA_I] - W[CADERA_D]
        lat = lat - np.dot(lat, arriba) * arriba
        n = np.linalg.norm(lat)
        if n < 1e-6:
            return None
        delante = np.cross(lat / n, arriba)
        return abs(float(np.dot(W[TOBILLO_I] - W[TOBILLO_D], delante))) / pierna

    @staticmethod
    def _tronco_perfil(P, vis, W):
        """(de_perfil, inclinacion del tronco en grados respecto a la vertical) o (False, None).

        De perfil se mide en la imagen (plano XY con aspecto corregido), donde se ve tal cual.
        Para saber si esta de perfil: en 3D, giro de la linea de caderas+hombros respecto al eje x
        de la camara; sin 3D, lo estrechos que se ven caderas y hombros frente a la altura del tronco.
        """
        ids = (HOMBRO_I, HOMBRO_D, CADERA_I, CADERA_D)
        if any(vis[i] < VIS_CADERA_MIN for i in ids):
            return False, None
        h = (P[HOMBRO_I] + P[HOMBRO_D]) / 2
        c = (P[CADERA_I] + P[CADERA_D]) / 2
        tronco = h - c
        alto = float(np.linalg.norm(tronco))
        if alto < 1e-6:
            return False, None
        if W is not None:
            lat = (W[CADERA_I] - W[CADERA_D]) + (W[HOMBRO_I] - W[HOMBRO_D])
            giro = math.degrees(math.atan2(abs(float(lat[2])), abs(float(lat[0])) + 1e-9))
            perfil = giro >= SENT_PERFIL_GIRO_MIN
        else:
            ancho = (np.linalg.norm(P[CADERA_I] - P[CADERA_D]) + np.linalg.norm(P[HOMBRO_I] - P[HOMBRO_D])) / 2
            perfil = ancho / alto <= SENT_PERFIL_ANCHO_MAX
        if not perfil:
            return False, None
        return True, angulo_vectores(np.array([0.0, -1.0]), tronco)

    def _tronco_frente(self, P, vis, W, dt, de_pie):
        """Inclinacion del tronco de frente (grados) por escorzo, confirmada en 3D; o None.

        Mientras la persona esta de pie aprende la relacion altura del tronco / anchura de hombros.
        Devuelve el angulo por escorzo solo si la inclinacion 3D tambien supera SENT_TRONCO_3D_CONFIRMA
        (si no hay 3D, devuelve el de escorzo tal cual).
        """
        ids = (HOMBRO_I, HOMBRO_D, CADERA_I, CADERA_D)
        if any(vis[i] < VIS_CADERA_MIN for i in ids):
            return None
        ancho = float(np.linalg.norm(P[HOMBRO_I] - P[HOMBRO_D]))
        # solo la componente vertical: con el cuerpo algo girado parte de la inclinacion se ve de lado
        alto = abs(float(((P[HOMBRO_I] + P[HOMBRO_D]) / 2 - (P[CADERA_I] + P[CADERA_D]) / 2)[1]))
        if ancho < 1e-6:
            return None
        rel = alto / ancho
        if de_pie:
            self.tronco_ref = _ema(self.tronco_ref, rel, dt, SENT_TRONCO_REF_TAU_S)
            return None
        if self.tronco_ref is None:
            return None
        ang = math.degrees(math.acos(max(0.0, min(1.0, rel / self.tronco_ref))))
        if W is not None:
            t3 = (W[HOMBRO_I] + W[HOMBRO_D]) / 2 - (W[CADERA_I] + W[CADERA_D]) / 2
            ang3 = angulo_vectores(np.array([0.0, -1.0, 0.0]), t3)
            if ang3 is None or ang3 < SENT_TRONCO_3D_CONFIRMA:
                return min(ang, SENT_TRONCO_3D_CONFIRMA)   # no confirmado: no puede invalidar
        return ang

    @staticmethod
    def _pie_levantado(P, vis, W):
        """Diferencia de altura entre los dos tobillos dividida por la longitud media de la pierna.

        Usa la y de pose_world_landmarks (metros) si hay 3D, si no la de la imagen. Con un tobillo poco
        visible tambien mide: es justo lo que pasa de perfil con la pierna del fondo.
        """
        ids = (CADERA_I, CADERA_D, RODILLA_I, RODILLA_D, TOBILLO_I, TOBILLO_D)
        if any(vis[i] < VIS_CADERA_MIN for i in ids):
            return None
        Q = W if W is not None else P
        pierna = (np.linalg.norm(Q[CADERA_I] - Q[RODILLA_I]) + np.linalg.norm(Q[RODILLA_I] - Q[TOBILLO_I]) +
                  np.linalg.norm(Q[CADERA_D] - Q[RODILLA_D]) + np.linalg.norm(Q[RODILLA_D] - Q[TOBILLO_D])) / 2
        if pierna < 1e-6:
            return None
        return abs(float(Q[TOBILLO_I][1] - Q[TOBILLO_D][1])) / pierna

    def _fase_2(self, P, vis, dt, t, W=None):
        """Sentadilla: angulo de rodilla (3D, o 2D + escorzo), FSM con t_min abajo y zancadas."""
        ang = self.est_rodilla.medir(P, vis, dt, W)
        fsm = self.fsm_sent
        if ang is None:
            fsm.sin_medida(dt)
            completada = False
        else:
            completada = fsm.actualizar(ang, dt)

        # empieza una repeticion nueva: se olvida la anterior
        if fsm.estado == FSMSentadilla.BAJANDO and self.sent_estado_prev not in (
                FSMSentadilla.BAJANDO, FSMSentadilla.SUBIENDO):
            self.zancada_t, self.zancada_max, self.zancada_mal = 0.0, 0.0, False
            self.tronco_t, self.tronco_max, self.tronco_mal = 0.0, 0.0, False
            self.pie_alto_t, self.pie_alto_max, self.pie_alto_mal = 0.0, 0.0, False
            self.sent_comp = ""

        en_rep = fsm.estado in (FSMSentadilla.BAJANDO, FSMSentadilla.ABAJO, FSMSentadilla.SUBIENDO)
        if not en_rep and not completada and ang is not None and ang > fsm.arriba:
            self._tronco_frente(P, vis, W, dt, de_pie=True)   # de pie: aprende la referencia
        if (en_rep or completada) and VALIDAR_COMPENSACIONES and not self.sent_comp:
            self.sent_comp = self._compensacion_activa(hombro=COMPENSACION_HOMBRO_EN_SENTADILLA)
        if en_rep or completada:
            self.zancada = self._separacion_pies(P, vis, W)
            if self.zancada is not None:
                self.zancada_max = max(self.zancada_max, self.zancada)
                self.zancada_t = self.zancada_t + dt if self.zancada > SENT_ZANCADA_MAX else 0.0
                if self.zancada_t >= SENT_ZANCADA_T_S:
                    self.zancada_mal = True
            self.pie_alto = self._pie_levantado(P, vis, W)
            if self.pie_alto is not None:
                self.pie_alto_max = max(self.pie_alto_max, self.pie_alto)
                self.pie_alto_t = self.pie_alto_t + dt if self.pie_alto > SENT_PIE_LEVANTADO_MAX else 0.0
                if self.pie_alto_t >= SENT_PIE_LEVANTADO_T_S:
                    self.pie_alto_mal = True
            self.de_perfil, self.tronco = self._tronco_perfil(P, vis, W)
            limite = SENT_TRONCO_MAX
            self.tronco_modo = "perfil" if self.tronco is not None else None
            if self.tronco is None and SENT_TRONCO_FRENTE and not self.de_perfil:
                self.tronco = self._tronco_frente(P, vis, W, dt, de_pie=False)
                limite = SENT_TRONCO_FRENTE_MAX
                self.tronco_modo = "frente" if self.tronco is not None else None
            if self.tronco is not None:
                self.tronco_max = max(self.tronco_max, self.tronco)
                self.tronco_t = self.tronco_t + dt if self.tronco > limite else 0.0
                if self.tronco_t >= SENT_TRONCO_T_S:
                    self.tronco_mal = True
            else:
                self.tronco_t = 0.0

        if completada:
            if self.pie_alto_mal:
                self._anotar(t, f"no es sentadilla: una pierna levantada ({self.pie_alto_max:.2f})")
            elif self.zancada_mal:
                self._anotar(t, f"sentadilla no valida: zancada, un pie delante del otro ({self.zancada_max:.2f})")
            elif self.tronco_mal:
                self._anotar(t, f"sentadilla no valida: tronco demasiado inclinado ({self.tronco_max:.0f}°)")
            elif VALIDAR_COMPENSACIONES and self.sent_comp:
                self._anotar(t, f"sentadilla no valida: compensacion ({self.sent_comp})")
            else:
                self.repeticiones_sentadilla += 1
                self._anotar(t, f"sentadilla {self.repeticiones_sentadilla} ({fsm.ultimo_minimo:.0f}°)")
            self.zancada_t, self.zancada_max, self.zancada_mal = 0.0, 0.0, False
            self.tronco_t, self.tronco_max, self.tronco_mal = 0.0, 0.0, False
            self.pie_alto_t, self.pie_alto_max, self.pie_alto_mal = 0.0, 0.0, False
            self.sent_comp = ""
        if fsm.evento:
            self._anotar(t, fsm.evento)
        self.sent_estado_prev = fsm.estado
        self.sent = fsm.angulo

    # ---------------------------------------------------------------- fase 3
    def _registrar_compensacion(self, t_inicio, t_confirmacion, motivo):
        """Apunta un error (un timestamp por episodio). Dos a menos de ERROR_SEPARACION_S se juntan."""
        ts = t_inicio if MODO_TIMESTAMP == "inicio" else t_confirmacion
        ts = round(max(0.0, ts), 2)
        if any(abs(ts - otro) < ERROR_SEPARACION_S for otro in self.timestamps_errores):
            return
        self.timestamps_errores.append(ts)
        self.motivos_errores.append(motivo)
        orden = sorted(range(len(self.timestamps_errores)), key=lambda i: self.timestamps_errores[i])
        self.timestamps_errores = [self.timestamps_errores[i] for i in orden]
        self.motivos_errores = [self.motivos_errores[i] for i in orden]
        self._anotar(t_confirmacion, f"compensacion: {motivo} (t={ts:.2f}s)")

    def _episodio(self, clave, cumple, dt, t, motivo, persiste):
        """Condicion mantenida durante `persiste` segundos (suma de dt) = un episodio, apuntado una vez."""
        ep = self.episodios[clave]
        if not cumple:
            ep["t"], ep["activo"] = 0.0, False
            return
        ep["t"] += dt
        if not ep["activo"] and ep["t"] >= persiste:
            ep["activo"] = True
            self._registrar_compensacion(t - ep["t"], t, motivo)

    def _fase3_cortar(self):
        """Sin pose fiable: los episodios en curso se cortan (no se arrastran por encima del hueco)."""
        for ep in self.episodios.values():
            ep["t"], ep["activo"] = 0.0, False

    def _fase_3(self, P, vis, dt, t, W=None):
        """Compensaciones: inclinacion lateral del tronco y asimetria escapular (encogimiento).

        Siempre activa mientras la pose sea fiable. Las senales se suavizan con una EMA con dt real
        (encima del One Euro de los puntos) y solo cuentan si se mantienen INCLINACION_PERSISTE_S /
        HOMBRO_PERSISTE_S seguidos. Ademas, la fase 2 apunta aqui la sentadilla con el tronco
        demasiado inclinado hacia delante.
        """
        if not self._visibles(vis, (HOMBRO_I, HOMBRO_D)) or \
                vis[CADERA_I] < VIS_CADERA_MIN or vis[CADERA_D] < VIS_CADERA_MIN:
            self._fase3_cortar()
            return
        modo = "3D" if W is not None else "2D"
        if modo != self.f3_modo:
            self.f3_modo = modo
            self.f3_hombro_ref = {"I": None, "D": None}
            self.f3_lateral = None
        Q = W if W is not None else P
        arriba = np.array([0.0, -1.0, 0.0]) if W is not None else np.array([0.0, -1.0])
        centro_h = (Q[HOMBRO_I] + Q[HOMBRO_D]) / 2
        tronco = centro_h - (Q[CADERA_I] + Q[CADERA_D]) / 2
        alto = float(np.linalg.norm(tronco))
        if alto < 1e-6:
            self._fase3_cortar()
            return
        eje = tronco / alto

        # ---- 1) inclinacion lateral del tronco, en el plano frontal DEL CUERPO
        lat = (Q[CADERA_I] - Q[CADERA_D]) + (Q[HOMBRO_I] - Q[HOMBRO_D])
        lat = lat - np.dot(lat, arriba) * arriba          # horizontal
        nl = float(np.linalg.norm(lat))
        lateral = None
        if nl > 1e-6:
            lat = lat / nl
            if W is not None:
                giro = math.degrees(math.atan2(abs(float(lat[2])), abs(float(lat[0])) + 1e-9))
            else:
                ancho = (np.linalg.norm(P[CADERA_I] - P[CADERA_D]) + np.linalg.norm(P[HOMBRO_I] - P[HOMBRO_D])) / 2
                giro = 90.0 if ancho / alto <= SENT_PERFIL_ANCHO_MAX else 0.0
            if giro <= FASE3_GIRO_MAX:
                lateral = math.degrees(math.atan2(abs(float(np.dot(tronco, lat))), float(np.dot(tronco, arriba))))
        if lateral is None:
            self.f3_lateral = None
            self.episodios["lateral"]["t"], self.episodios["lateral"]["activo"] = 0.0, False
        else:
            self.f3_lateral = _ema(self.f3_lateral, lateral, dt, TAU_FASE3_S)
            self._episodio("lateral", self.f3_lateral > INCLINACION_MAX, dt, t,
                           f"inclinacion lateral del tronco ({self.f3_lateral:.0f}°)", INCLINACION_PERSISTE_S)

        # ---- 2) asimetria escapular: el hombro sube hacia la oreja
        en_sentadilla = self.fsm_sent.estado in (FSMSentadilla.BAJANDO, FSMSentadilla.ABAJO, FSMSentadilla.SUBIENDO)
        ratios = {}
        for lado, oreja, hombro, nombre in (("I", OREJA_I, HOMBRO_I, "izquierdo"), ("D", OREJA_D, HOMBRO_D, "derecho")):
            if vis[oreja] < VISIBILIDAD_MIN:
                self.f3_hombro[lado] = None
                continue
            d = float(np.dot(Q[oreja] - Q[hombro], eje))     # altura de la oreja sobre el hombro
            if d <= 1e-6:
                self.f3_hombro[lado] = None
                continue
            brazo = self.abd_info[nombre]["ang"]
            brazo_abajo = brazo is None or brazo < ABD_ABAJO
            ref = self.f3_hombro_ref[lado]
            if brazo_abajo and not en_sentadilla:
                tau = HOMBRO_REF_TAU_SUBE_S if (ref is None or d > ref) else HOMBRO_REF_TAU_BAJA_S
                self.f3_hombro_ref[lado] = ref = _ema(ref, d, dt, tau)
            if ref is None:
                self.f3_hombro[lado] = None
                continue
            self.f3_hombro[lado] = _ema(self.f3_hombro[lado], d / ref, dt, TAU_FASE3_S)
            ratios[lado] = (self.f3_hombro[lado], brazo)
        for lado, otro, nombre in (("I", "D", "izquierdo"), ("D", "I", "derecho")):
            clave = "hombro_" + lado
            if lado not in ratios or otro not in ratios:
                self.episodios[clave]["t"], self.episodios[clave]["activo"] = 0.0, False
                continue
            r, brazo = ratios[lado]
            limite = HOMBRO_RATIO_MIN_BRAZO_ARRIBA if (brazo is not None and brazo > 90.0) else HOMBRO_RATIO_MIN
            cumple = r < limite and r < ratios[otro][0] - HOMBRO_ASIMETRIA_MIN
            self._episodio(clave, cumple, dt, t, f"hombro {nombre} encogido ({r:.2f})", HOMBRO_PERSISTE_S)

        # ---- 3) tronco demasiado inclinado hacia delante en la sentadilla (lo mide la fase 2)
        en_sent = self.fsm_sent.estado in (FSMSentadilla.BAJANDO, FSMSentadilla.ABAJO, FSMSentadilla.SUBIENDO)
        lim = SENT_TRONCO_MAX if self.tronco_modo == "perfil" else SENT_TRONCO_FRENTE_MAX
        self._episodio("tronco_sent", SENT_TRONCO_COMPENSACION and en_sent and self.tronco is not None
                       and self.tronco > lim, dt, t, "tronco inclinado en sentadilla", SENT_TRONCO_T_S)

    def _compensacion_activa(self, hombro=True):
        """Motivo de la compensacion activa ahora (inclinacion lateral y, si se pide, hombro), o ""."""
        nombres = {"lateral": "inclinacion lateral del tronco",
                   "hombro_I": "hombro izquierdo encogido", "hombro_D": "hombro derecho encogido"}
        claves = ("lateral", "hombro_I", "hombro_D") if hombro else ("lateral",)
        return ", ".join(nombres[k] for k in claves if self.episodios[k]["activo"])

    # ---------------------------------------------------------------- salida
    def resultados(self):
        """Diccionario con el esquema exacto exigido por el evaluador."""
        with self.lock:
            return {
                "fase_1": {"repeticiones_abduccion": int(self.repeticiones_abduccion + self._abd_pendientes_validas())},
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
                fuente = ""
                if info.get("ang2d") is not None and info.get("ang3d") is not None:
                    fuente = f" (3D {info['ang3d']:.0f}/2D {info['ang2d']:.0f})"
                lineas.append((f"{letra}: {info['ang']:3.0f}deg{fuente}  {plano}  {codo}  {fsm.estado}",
                               info["en_plano"] and info["valido"]))
            lineas.append((f"abducciones: {self.repeticiones_abduccion}", True))
            lat = "lateral ?" if self.f3_lateral is None else f"lateral {self.f3_lateral:.0f}deg"
            hi = "?" if self.f3_hombro["I"] is None else f"{self.f3_hombro['I']:.2f}"
            hd = "?" if self.f3_hombro["D"] is None else f"{self.f3_hombro['D']:.2f}"
            mal = any(ep["activo"] for ep in self.episodios.values())
            lineas.append((f"fase 3: {lat}  hombros I {hi} D {hd}  errores {len(self.timestamps_errores)}", not mal))
            f2 = self.fsm_sent
            rodilla = "rodilla ?" if f2.angulo is None else \
                f"rodilla{' 3D' if self.est_rodilla.usa_3d else ''} {f2.angulo:.0f}deg"
            lineas.append((f"sentadilla: {f2.estado}  {rodilla}  {'abajo' if MODO_T_MIN == 'acumulado' else 'pausa'} {f2.pausa:.1f}s  "
                           f"pie>{f2.arriba:.0f}  reps {self.repeticiones_sentadilla}", f2.angulo is not None))
            if self.zancada is not None:
                lineas.append((f"pies delante-detras: {self.zancada:.2f}"
                               + ("  ZANCADA" if self.zancada > SENT_ZANCADA_MAX else ""),
                               self.zancada <= SENT_ZANCADA_MAX))
            if self.pie_alto is not None:
                lineas.append((f"pies altura: {self.pie_alto:.2f}"
                               + ("  PIERNA LEVANTADA" if self.pie_alto > SENT_PIE_LEVANTADO_MAX else ""),
                               self.pie_alto <= SENT_PIE_LEVANTADO_MAX))
            if self.tronco is not None:
                lim = SENT_TRONCO_MAX if self.tronco_modo == "perfil" else SENT_TRONCO_FRENTE_MAX
                lineas.append((f"tronco ({self.tronco_modo}): {self.tronco:.0f}deg"
                               + ("  DEMASIADO INCLINADO" if self.tronco > lim else ""),
                               self.tronco <= lim))
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
                "abducciones", "sentadillas", "rodilla", "sent_estado", "sent_pausa", "zancada", "tronco_perfil", "pie_levantado", "f3_lateral", "f3_hombro_I", "f3_hombro_D", "errores_fase3"])

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
                 motor.fsm_sent.estado, round(motor.fsm_sent.pausa, 2), r(motor.zancada), r(motor.tronco), r(motor.pie_alto),
                 r(motor.f3_lateral), r(motor.f3_hombro["I"]), r(motor.f3_hombro["D"]),
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