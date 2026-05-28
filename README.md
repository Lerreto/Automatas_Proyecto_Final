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
