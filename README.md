# Rehab Monitor · Grupo Rosa
**EuriChallenge 2026 · Reto de Ingeniería Biomédica: "Gamificando la Fisioterapia con IA"**
EESTEC LC Madrid & Eurielec · ETSIT, Universidad Politécnica de Madrid

Monitor de rehabilitación física que funciona con una sola cámara web. Cuenta repeticiones de **abducción de hombro** y **sentadilla**, detecta **compensaciones clínicas** (inclinación del tronco y asimetría escapular) y guía al paciente con un **juego** que se controla desde un **Elgato Stream Deck**.

La idea de diseño ha sido **pensar como un fisioterapeuta**: solo cuentan las repeticiones bien hechas, y cada aviso es explicable (el sistema dice qué ha pasado y en qué segundo). Al mismo tiempo, es **permisivo con el paciente**: los umbrales son realistas y se adaptan a la postura de cada persona.

---

## Índice
1. [Cómo ejecutarlo](#1-cómo-ejecutarlo)
2. [Arquitectura](#2-arquitectura)
3. [Fase 1 · Abducción de hombro](#3-fase-1--abducción-de-hombro)
4. [Fase 2 · Sentadilla](#4-fase-2--sentadilla)
5. [Fase 3 · Compensaciones y filtrado de señal](#5-fase-3--compensaciones-y-filtrado-de-señal)
6. [Robustez en vídeos difíciles](#6-robustez-en-vídeos-difíciles)
7. [Gamificación y Stream Deck](#7-gamificación-y-stream-deck)
8. [Herramientas de depuración](#8-herramientas-de-depuración)
9. [Parámetros ajustables](#9-parámetros-ajustables)
10. [Limitaciones conocidas](#10-limitaciones-conocidas)
11. [Licencias](#11-licencias)

---

## 1. Cómo ejecutarlo

Requisitos: **Docker Desktop** (o Docker Engine + Compose) y un navegador moderno (Chrome recomendado).

### Interfaz con cámara (demo y juego)
```bash
docker compose up --build
```
Abre `http://localhost:8501` y pulsa **START**.

### Evaluación automática sobre un vídeo
`eval.py` procesa el vídeo completo, sin interfaz, y escribe `resultados.json` en la raíz del proyecto:
```bash
docker compose run --rm -v "$PWD":/proyecto -w /proyecto web python eval.py /proyecto/mi_video.mp4
```
Salida (esquema oficial del reto):
```json
{
  "fase_1": { "repeticiones_abduccion": 0 },
  "fase_2": { "repeticiones_sentadilla": 0 },
  "fase_3": { "compensaciones_detectadas": false, "timestamps_errores_segundos": [] }
}
```
Todos los tiempos se calculan con los timestamps de los fotogramas (`CAP_PROP_POS_MSEC`), **nunca contando fotogramas**, así que el resultado no depende de los FPS del vídeo.

---

## 2. Arquitectura

```
app/
├── rehab_engine.py   # motor: toda la lógica de las 3 fases, sin Streamlit
├── app.py            # interfaz Streamlit: vídeo, juego (guía fantasma), Stream Deck
├── utils.py
└── assets/           # estilos, panel lateral (intro.js), fuentes Inter, logos
eval.py               # evaluación por línea de comandos (usa solo el motor)
```

- **Motor desacoplado:** `RehabEngine` no depende de la interfaz. La interfaz y `eval.py` usan la misma API (`procesar_fotograma`, `instantanea`, `resultados`, `reiniciar`), compatible con la plantilla original.
- **Sin dependencias nuevas:** solo MediaPipe, OpenCV y NumPy, las del `requirements.txt` del reto.

Flujo de cada fotograma:
```
cámara / vídeo → (mejora de luz si está oscuro) → MediaPipe Pose (33 puntos 2D + 3D)
   → filtro One Euro sobre los puntos
   → fase 3 (compensaciones) → fase 1 (abducción) → fase 2 (sentadilla)
```

---

## 3. Fase 1 · Abducción de hombro

**Medida.** El ángulo sigue la ecuación del enunciado: vértice en el hombro H, entre el vector H → codo E y el vector H → cadera del mismo lado C. Se calcula en **3D, en el sistema de referencia del propio torso**, así que es el mismo con la persona de frente, en diagonal o de perfil. Si no se ve la cadera de ese lado, se usa el eje del tronco.

**Conteo.** Máquina de estados por brazo con histéresis: hay que subir por encima de **80°** (objetivo 90° con margen de 10°) y volver por debajo de **40°**, manteniendo cada posición un tiempo mínimo. La repetición se confirma **al bajar el brazo**, para poder anularla si hubo una compensación. Si el vídeo termina con el brazo arriba, la repetición se suma igualmente.

**Validación clínica:**
| Situación | Decisión |
|---|---|
| Brazo hacia delante (elevación frontal) | No cuenta: el **plano de elevación en 3D** debe estar a ≤ 40° del plano lateral. El plano escapular (~30°) sí cuenta. |
| Pasa por delante y luego se abre al lado | No cuenta |
| Codo doblado | No cuenta: codo ≥ 140° y muñeca también por encima del umbral |
| Brazo cruzando el cuerpo | No cuenta |
| Dos brazos a la vez | Una repetición (ventana de 0.8 s) |
| Persona casi de perfil | Se mide en modo permisivo, comprobando solo que el brazo no vaya hacia delante |
| Brazos estirados durante una sentadilla | No cuenta como abducción |
| Compensación durante la repetición | No cuenta (y se registra en la fase 3) |

---

## 4. Fase 2 · Sentadilla

**Medida.** Ángulo de rodilla (cadera-rodilla-tobillo) en **3D**: de frente, en la imagen plana no se ve la flexión, pero en 3D sí.

**Conteo.** Máquina de estados `de pie → bajando → abajo → subiendo → de pie`. Hay que doblar la rodilla por debajo de **130°** y volver a estar de pie. El umbral de "de pie" **se adapta a cada persona** (12° por debajo de su ángulo de rodilla de pie, mínimo 145°), así funciona igual con personas bajas o con hiperextensión.

**Validación clínica:**
| Situación | Decisión |
|---|---|
| De frente, solo se dobla una pierna (zancada) | No cuenta: deben doblarse **las dos** |
| Un pie delante del otro | No cuenta (solo se comprueba con las dos piernas bien visibles y no de perfil) |
| Una pierna levantada | No cuenta (solo con las dos piernas bien visibles) |
| Tronco muy inclinado ("casi a cuatro patas") | No cuenta y se registra como compensación |
| Cuclillas (rodilla < 60°) | No cuenta |
| Piernas fuera de la imagen | La sentadilla se reinicia en silencio, sin afectar a las abducciones |

---

## 5. Fase 3 · Compensaciones y filtrado de señal

### Qué se detecta
Las dos compensaciones del enunciado:

1. **Inclinación anómala del tronco,** en cualquier dirección, y el sistema indica cuál:
   | Dirección | De pie | En sentadilla |
   |---|---|---|
   | Hacia delante | > 15° | > 45° (algo de inclinación es natural al bajar) |
   | Hacia atrás | > 15° | > 15° |
   | Hacia un lado | > 10° | > 12° |
2. **Asimetría escapular (hombro encogido):** inclinación de la línea de hombros respecto al tronco, > 9° con el brazo abajo. El umbral **crece con la elevación del brazo** (~13° a 90°), porque al levantarlo el hombro sube de forma natural.

### Referencia: la vertical del cuerpo, no la de la cámara
Si la inclinación se mide respecto a la cámara, una cámara torcida o desde arriba hace que una persona recta parezca inclinada. Por eso:
- **Con las piernas a la vista,** la vertical es la línea **tobillos → caderas** de la propia persona. La compensación es que el tronco se separe de esa línea, sin importar cómo esté puesta la cámara.
- **Sin piernas a la vista,** la referencia es la **postura de reposo** de la persona (de pie, brazos abajo). Durante el ejercicio esa referencia se **congela** y se compara contra ella. Solo se actualiza con posturas casi rectas (< 4° de diferencia), así que una compensación nunca se aprende como "normal". Si el tronco se queda quieto en otro ángulo 3 s, se interpreta como un cambio de cámara.
- **La asimetría natural de hombros** se aprende en reposo, pero **limitada a ±4°**: un hombro encogido desde el principio se sigue detectando.

### Filtrado (el compromiso ruido / retraso del enunciado)
MediaPipe da puntos que tiemblan aunque la persona esté quieta. Filtrar mucho quita el ruido pero mete retraso; filtrar poco deja el ruido. Nuestra solución tiene dos capas:

1. **Filtro One Euro** (Casiez et al., 2012) sobre los puntos, para **todo el motor**. Es una EMA adaptativa: suaviza mucho cuando el punto está quieto (quita el temblor) y poco cuando se mueve rápido (no retrasa el movimiento real). Así las fases 1 y 2 cuentan bien sin perder repeticiones rápidas.
2. **EMA ligera** (τ = 0.1 s) sobre las señales de la fase 3. Estas señales se calculan restando puntos cercanos y amplifican el ruido que queda. La EMA puede ser muy suave porque los puntos ya llegan limpios.

Ambos filtros usan el **tiempo real entre fotogramas** (α = 1 − e^(−dt/τ)), así que se comportan igual a 30 que a 60 fps.

**Medida (datos simulados, 6 sentadillas con ruido):**
| | Temblor | Retraso | Sentadillas detectadas |
|---|---|---|---|
| Sin filtro | 9.25° | 0 ms | 6 / 6 |
| **Nuestro filtrado** | **0.70° (−92 %)** | **100 ms** | **6 / 6** |
| Filtrado excesivo | 0.05° | 833 ms | 1 / 6 |

### Δt sostenido y timestamps
- Una compensación solo cuenta si supera el umbral **de forma continuada** 0.3 s (tronco) o 0.5 s (hombro). El tiempo se acumula sumando el dt de cada fotograma.
- Se registra **un timestamp por episodio**, en el segundo en que **empezó**, con su motivo. Los episodios separados por menos de 1 s se fusionan.
- Las repeticiones con compensación **no cuentan** en las fases 1 y 2. Otros motivos de anulación (zancada, brazo por delante…) anulan la repetición pero **no** generan timestamp: en el JSON solo aparecen las compensaciones del enunciado.

---

## 6. Robustez en vídeos difíciles
- **Mala luz:** si el brillo medio es bajo, antes de MediaPipe se aplica **corrección gamma** (si es muy oscuro) y **CLAHE** sobre la luminancia (canal L de LAB): aumento de contraste por zonas y con límite, sin alterar los colores ni amplificar el ruido de las zonas lisas.
- **Nunca se deja de medir:** si no hay puntos 3D o no se ven las caderas, se pasa a 2D. De perfil, se mide en modo permisivo.
- **Confianza permisiva** en los puntos (≥ 0.3) para no descartar fotogramas con poca luz; los puntos ya van filtrados.
- **Persona girada o cámara inclinada:** ángulos en el sistema del torso y vertical del cuerpo (ver fase 3).

---

## 7. Gamificación y Stream Deck

### Guía fantasma
- Una **sombra con las proporciones del propio paciente** (medidas calibradas en reposo, mediana de varios fotogramas) muestra el movimiento ideal. Se coloca de frente o de perfil según hacia dónde mire la persona.
- **Sigue las reglas del motor:** solo se pone en verde si la repetición va a contar de verdad.
- **Nota del 1 al 10** en cada repetición, según los avisos y lo cerca del objetivo que se quede, y un **resumen final**.
- **Mensajes amables** para el paciente ("Espalda más recta", "No te ladees") en lugar de mensajes técnicos.
- **Diseño tipo Apple:** tipografía Inter, anillos de progreso, efecto cristal y animaciones al completar una repetición, dibujados en ~3 ms por fotograma.
- **Rutinas:** hombro derecho, hombro izquierdo, sentadillas o rutina completa.

### Control con Elgato Stream Deck
El fisioterapeuta controla la sesión sin tocar el ordenador. Cada tecla del Stream Deck envía un atajo de teclado (acción *Hotkey*) que la app convierte en la pulsación del botón correspondiente:

| Atajo (Mac) | Acción |
|---|---|
| Ctrl + Opción + 1 | Reiniciar sesión |
| Ctrl + Opción + 2 | START / STOP de la cámara |
| Ctrl + Opción + 3 | Modo juego (guía fantasma) |
| Ctrl + Opción + 4 | Depuración (lo que mide el motor sobre el vídeo) |
| Ctrl + Opción + 5 | Rutina: hombro derecho |
| Ctrl + Opción + 6 | Rutina: hombro izquierdo |
| Ctrl + Opción + 7 | Rutina: sentadillas |
| Ctrl + Opción + 8 | Rutina completa |

Los botones están en una **barra flotante** siempre visible, también con el vídeo a pantalla completa. En macOS hay que dar permiso de **Accesibilidad** a la app de Stream Deck.

---

## 8. Herramientas de depuración

**Depurador del motor:** procesa un vídeo y genera el vídeo con el esqueleto y las medidas dibujadas, un CSV fotograma a fotograma y la lista de cada compensación con su segundo y su motivo.
```bash
docker compose run --rm -v "$PWD":/proyecto -w /proyecto/app web python rehab_engine.py /proyecto/mi_video.mp4
```
Genera `mi_video_depurado.mp4` y `mi_video_depurado.csv` junto al vídeo.

**Modo depuración en la interfaz:** botón *Depuración* (o Ctrl + Opción + 4). Muestra sobre el vídeo los ángulos, el giro del torso y por qué una repetición es válida o no.

---

## 9. Parámetros ajustables
Todos los umbrales son constantes documentadas al principio de `app/rehab_engine.py`. Las principales:

| Constante | Valor | Qué controla |
|---|---|---|
| `ABD_OBJETIVO`, `ABD_MARGEN` | 90°, 10° | Altura necesaria en la abducción |
| `ABD_ABAJO` | 40° | Hay que bajar de aquí para la siguiente repetición |
| `PLANO_ABD_MAX` | 40° | Cuánto puede ir el brazo hacia delante |
| `CODO_EXTENDIDO_MIN` | 140° | Extensión mínima del codo |
| `SENT_ABAJO` | 130° | Profundidad mínima de la sentadilla |
| `SENT_MIN_VALIDO` | 60° | Por debajo, cuclillas |
| `TRONCO_MAX_DE_PIE` / `_SENTADILLA` / `_ATRAS` | 15° / 45° / 15° | Inclinación del tronco |
| `TRONCO_MAX_LADO` | 10° | Inclinación lateral |
| `ESCAPULA_MAX`, `ESCAPULA_POR_GRADO_BRAZO` | 9°, 0.06 | Asimetría de hombros |
| `TRONCO_PERSISTE_S`, `ESCAPULA_PERSISTE_S` | 0.3 s, 0.5 s | Δt sostenido |
| `FILTRO_MIN_CUTOFF`, `FILTRO_BETA` | 1.2, 1.5 | Filtro One Euro |
| `TAU_FASE3_S` | 0.1 s | EMA de la fase 3 |
| `METRICA_FASE3` | `"enunciado"` | `"lateral"` recupera la métrica anterior (solo inclinación lateral) |
| `USAR_MUNDO_3D` | `True` | `False` = abducción y compensaciones solo en 2D |

---

## 10. Limitaciones conocidas
- La **profundidad (z) de MediaPipe es estimada**: con mala luz o la cámara muy cerca puede ser imprecisa.
- **Puntos fuera de la imagen:** MediaPipe se los inventa. Para mejores resultados, la cámara debe estar a la altura del pecho, a 2–3 m, con todo el cuerpo dentro de la imagen y buena luz.
- **Sin piernas en la imagen,** el sistema no puede distinguir del todo "persona inclinada" de "cámara inclinada": si la persona se mantiene ladeada todo el rato, a los 3 s se toma como un cambio de cámara.
- **De perfil,** la abducción se mide en modo permisivo (no se comprueba el codo).

---

## 11. Licencias
- Código del reto y de este proyecto: ver [`LICENSE`](LICENSE).
- Tipografía **Inter** (`app/assets/fuente*.ttf`): SIL Open Font License, ver `app/assets/LICENCIA_INTER_OFL.txt`.
- **MediaPipe Pose** (Google), licencia Apache 2.0.
- Filtro One Euro: Casiez, Roussel y Vogel, *"1 € Filter: A Simple Speed-based Low-pass Filter for Noisy Input in Interactive Systems"*, CHI 2012.
