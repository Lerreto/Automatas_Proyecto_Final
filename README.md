# Reconstrucción de Imágenes Faciales con Autómatas Celulares Neurales

Proyecto final de **Autómatas y Lenguajes Formales** — Universidad Industrial de Santander.

Implementación de un **Neural Cellular Automaton (NCA)** para reconstrucción de imágenes (*image inpainting*): dado un rostro con una zona borrada, el autómata la reconstruye iterativamente usando únicamente el contexto de los píxeles vecinos visibles.

---

## Idea central

Un autómata celular clásico tiene células en una grilla que evolucionan aplicando la misma regla local. La pregunta del proyecto es:

> *¿Qué pasa si el alfabeto de estados es continuo (vectores reales en ℝ³²) y la función de transición δ no se define a mano, sino que se aprende por optimización?*

El resultado es un NCA que sigue cumpliendo las tres propiedades formales del modelo — **localidad**, **homogeneidad** y **paralelismo** — pero cuya regla δ es una red neuronal entrenada con gradiente descendente sobre ~8 000 rostros del dataset CelebA.

---

## Estructura del repositorio

```
.
├── README.md                       ← Este archivo (visión general)
├── Codigo/
│   ├── README.md                   ← Guía del código (entrenamiento + app)
│   ├── nca_entrenamiento.ipynb     ← Notebook de entrenamiento (Google Colab)
│   └── nca-web/                    ← Aplicación web interactiva
│       ├── backend/
│       │   ├── app.py              ← API FastAPI
│       │   ├── inference.py        ← Loop de reconstrucción y métricas
│       │   ├── model.py            ← Definición PyTorch del NCA
│       │   └── checkpoints/
│       │       └── latest.pth      ← Pesos entrenados
│       ├── frontend/
│       │   ├── index.html
│       │   ├── script.js
│       │   └── style.css
│       ├── README.md               ← Instrucciones de ejecución detalladas
│       ├── requirements.txt
│       ├── run.sh                  ← Arranque en Linux/Mac
│       └── run.bat                 ← Arranque en Windows
├── Contexto/
│   ├── README.md                   ← Índice de los documentos de contexto
│   ├── 01_de_que_trata.md          ← Descripción del problema y enfoque
│   ├── 02_arquitectura.md          ← Arquitectura del modelo y del sistema
│   └── 03_flujo_completo.md        ← Flujo de datos de extremo a extremo
├── Diapositivas/
│   ├── README.md                   ← Link a la presentación en Canva
│   └── Presentacion Proyecto Automatas.pdf
├── ImagenesPrueba/                 ← Fotos del equipo usadas en pruebas
│   ├── README.md
│   ├── CamiloFlorez.jpeg
│   ├── DavidSantiago.jpeg
│   ├── Mariana.jpeg
│   ├── Mauricio.jpeg
│   ├── Milo.jpeg
│   ├── Rueda1.jpeg
│   ├── Rueda2.jpeg
│   ├── SantiagoParedes.jpeg
│   └── Teban.jpeg
├── Informe/                        ← Informe escrito del proyecto
│   ├── README.md
│   └── SEGUIMIENTO PROYECTO AUTOMATAS CELULARES--PDF--.pdf
├── Poster/
│   ├── README.md
│   └── Poster.png                  ← Poster de presentación
└── Video/
    └── README.md                   ← Link al video en YouTube
```

---

## Arquitectura del modelo

```
Estado: (B, 32, 100, 100)   ← 32 canales por célula, grilla 100×100
  │
  ▼  PerceptionBlock
  │  Kernels depthwise fijos: Sobel X, Sobel Y, Laplaciano
  │  Salida: (B, 128, H, W)
  │
  ▼  UpdateNetwork  — 4 × Conv 1×1 con GELU
  │  128 → 256 → 256 → 128 → 32
  │  Salida: Δstate
  │
  ▼  Actualización estocástica
  │  state = state + Δstate · Bernoulli(0.5)
  │
  ▼  Condición de frontera Dirichlet
     Los píxeles visibles se "clavan" a su valor original en cada paso.
     Solo la zona del hueco evoluciona.

  × 120 pasos
```

---

## Ejecutar la aplicación web

Ver instrucciones detalladas en [`Codigo/nca-web/README.md`](Codigo/nca-web/README.md).

**Linux / Mac:**
```bash
cd Codigo/nca-web
chmod +x run.sh
./run.sh
```

**Windows:**
```
cd Codigo\nca-web
run.bat
```

Abre `http://localhost:8000` en el navegador, carga una imagen, pinta la zona a reconstruir y haz clic en **Reconstruct**.

**Requisitos:** Python 3.10+. GPU NVIDIA opcional (corre en CPU si no hay CUDA).

---

## Entrenamiento

El notebook [`Codigo/nca_entrenamiento.ipynb`](Codigo/nca_entrenamiento.ipynb) contiene el pipeline completo de entrenamiento: descarga de CelebA, definición del modelo, loop de optimización y guardado del checkpoint. Está pensado para ejecutarse en Google Colab con GPU.

---

## Demo en video

Video de demostración disponible en YouTube: [https://youtu.be/qtjdtpCSV40](https://youtu.be/qtjdtpCSV40)

---

## Recorrido de la presentación

La presentación (`Diapositivas/`) desarrolla el proyecto en siete bloques. Este es el resumen de cada uno:

1. **Intro — el problema del *in-painting*.** El *in-painting* es la técnica de rellenar áreas dañadas o borradas de una imagen. Se contrastan dos familias de soluciones: los algoritmos clásicos, que difunden información local pero carecen de "comprensión estructural", y los métodos de *deep learning*, que funcionan como "cajas negras" con una interpretación formal opaca. El proyecto propone el Autómata Celular Neuronal (NCA) como un punto intermedio: un modelo matemático formal cuya regla, aun siendo aprendida, conserva una definición rigurosa.

2. **Marco teórico — del autómata clásico al NCA.** Se repasan las tres propiedades formales que definen a un autómata celular: **localidad** (cada célula solo usa la información de su vecindad de Moore), **homogeneidad** (la misma regla de evolución se aplica a todas las células) y **paralelismo** (todas evolucionan simultáneamente, sin autoridad central). Con el *Game of Life* de Conway como ejemplo, se muestra cómo reglas simples generan auto-organización y comportamiento emergente. Los NCA superan al modelo clásico con dos saltos: **estados continuos** (cada célula almacena color y memoria interna en vectores reales) y **reglas aprendidas** (la transición se entrena con redes neuronales en lugar de programarse a mano).

3. **Formalización del modelo.** Se definen formalmente los componentes del autómata: la grilla, el alfabeto continuo (ℝ³²), la vecindad de Moore, el operador de percepción, la función de transición aprendida, la regla de evolución y la condición de frontera. Los 32 canales de cada célula se reparten así: los canales 0-2 guardan el color RGB visible; el canal 3 es la **máscara** (1 si el píxel está sano, 0 si está roto); los canales 4-6 almacenan el producto color×máscara (un *prior* estructural); y los canales 7-31 son la "memoria RAM" interna de la célula. La percepción no "ve" sino que aplica matemáticas: *kernels* de Sobel (derivadas espaciales para detectar contornos) y el operador Laplaciano (divergencia del gradiente para detectar curvaturas), concatenados sobre los 32 canales. La actualización es estocástica (50 % de probabilidad de aplicar la transición por célula y paso) y la **función de pérdida L** combina penalizaciones por error de color en la zona dañada, por alterar píxeles sanos, por incoherencia perceptual (comparada con la red VGG-16 de Google) y por ruido de alta frecuencia.

4. **Implementación.** El *stack* tecnológico es Python 3 con PyTorch, OpenCV, PIL, torchvision y VGG-16. El entrenamiento se realizó en una GPU Tesla T4 de 16 GB en Google Colab. El *dataset* fueron las primeras 8 000 imágenes de **CelebA**, redimensionadas a 100×100 px y normalizadas al rango [0, 1].

5. **Diseño experimental.** Se evalúa el modelo con métricas como **SSIM** (índice de similitud estructural), el **error absoluto medio (L1)**, el volumen de parámetros y el número de pasos T*. El protocolo garantiza rigor mediante tres reglas: fijación de semilla dinámica para resultados reproducibles, aislamiento de la zona de cómputo (las métricas se calculan solo sobre los píxeles ocluidos, no sobre la zona conocida) y agregación estadística de promedios sobre todos los pares (imagen, máscara) del conjunto de validación.

6. **Resultados y análisis.** Las curvas de pérdida muestran que el modelo aprende rápido al inicio y se estabiliza, con la pérdida del hueco (L_hole) convergiendo correctamente. No hay sobreajuste visible: el *train loss* se mantiene estable entre 0.163 y 0.170 y, aunque el *val loss* oscila más, no aparece una brecha creciente, lo que indica buena generalización. El mejor valor de validación fue **0.1527 en la época 91**.

7. **Discusión.** Según la **clasificación de Wolfram**, el NCA converge a un punto fijo estable, lo que lo sitúa en la **Clase II** —comportamiento estable, ideal para *in-painting*—, un rasgo que el propio entrenamiento forzó al penalizar oscilaciones; las clases III y IV (caóticas o complejas) no son alcanzables sin cambiar el esquema de entrenamiento. La **Turing-completitud** queda como pregunta abierta: no hay prueba formal ni estructuras de largo alcance que la confirmen. Entre las **limitaciones** están el sesgo del *dataset* (solo rostros frontales jóvenes), la resolución fija de 100×100, el fallo ante oclusiones mayores al 25 %, los artefactos en fondos con texturas fuertes y la lentitud (120 pasos de inferencia). El **trabajo futuro** plantea escalar a 256×256 o 512×512, probar nuevos dominios (paisajes, documentos, imágenes médicas) y caracterizar formalmente qué clases de Wolfram son alcanzables mediante descenso de gradiente.

---

## Documentación y materiales

Cada carpeta tiene su propio README con el detalle. Vista rápida:

| Carpeta | Contenido |
|---|---|
| [`Codigo/`](Codigo/README.md) | Notebook de entrenamiento + aplicación web (instrucciones Linux y Windows). |
| [`Contexto/`](Contexto/README.md) | Explicación teórica y técnica del proyecto. |
| [`Diapositivas/`](Diapositivas/README.md) | Presentación en Canva: [abrir ▸](https://canva.link/8uyik28td4ph7p7) |
| [`Informe/`](Informe/README.md) | Informe escrito (PDF). |
| [`Poster/`](Poster/README.md) | Póster de presentación. |
| [`ImagenesPrueba/`](ImagenesPrueba/README.md) | Fotos del equipo para probar la app. |
| [`Video/`](Video/README.md) | Video de demostración (YouTube). |

---

## Tecnologías

| Componente | Tecnología |
|---|---|
| Modelo | PyTorch |
| Backend | FastAPI + uvicorn |
| Frontend | HTML/CSS/JS puro, Chart.js |
| Dataset | CelebA (~8 000 imágenes de entrenamiento) |
| Entrenamiento | Google Colab (GPU) |
