# Arquitectura del sistema

## Visión general

```
┌─────────────────────────────────────────────────────┐
│                   NAVEGADOR (Frontend)               │
│  index.html + style.css + script.js                  │
│                                                      │
│  Canvas imagen  ──┐                                  │
│  Canvas máscara ──┤──► extractForModel() ──► POST    │
│  Controles UI   ──┘         /api/reconstruct         │
│                                    │                 │
│  ◄─── frames[] + metrics ──────────┘                 │
│  Animación + Gráficas (Chart.js)                     │
└─────────────────────────────────────────────────────┘
                        │ HTTP JSON
┌─────────────────────────────────────────────────────┐
│                   SERVIDOR (Backend)                 │
│  FastAPI  ·  uvicorn  ·  Python                      │
│                                                      │
│  GET  /api/health        → estado del modelo         │
│  POST /api/reconstruct   → ejecuta el NCA            │
│                                                      │
│  backend/app.py          API + servidor estáticos    │
│  backend/inference.py    lógica de reconstrucción    │
│  backend/model.py        definición PyTorch del NCA  │
│  backend/checkpoints/    latest.pth (modelo .pth)    │
└─────────────────────────────────────────────────────┘
```

---

## Modelo NCA (`backend/model.py`)

El modelo tiene dos bloques apilados:

### 1. PerceptionBlock
Aplica tres kernels convolucionales fijos (no entrenables) a cada canal del estado de forma *depthwise* (independiente por canal):

| Kernel    | Qué detecta             |
|-----------|-------------------------|
| Sobel X   | Gradiente horizontal    |
| Sobel Y   | Gradiente vertical      |
| Laplaciano| Curvatura / bordes      |

La salida concatena el estado original + los tres gradientes → **4× canales**.

```
entrada: (B, 32, H, W)
salida:  (B, 128, H, W)   ← x, gx, gy, lap concatenados
```

### 2. UpdateNetwork
Red de 4 capas Conv 1×1 (sin padding, ve solo el píxel actual):

```
128 → 256 → 256 → 128 → 32    (con activación GELU entre capas)
```

Los pesos de la última capa se inicializan en cero → al inicio el modelo no cambia nada (entrenamiento estable).

### 3. Regla de actualización estocástica
En cada paso solo una fracción aleatoria de células se actualiza (máscara Bernoulli con p = 0.5):

```
s_{t+1} = s_t + Δs · m       donde m ~ Bernoulli(0.5)
```

Esto introduce asincronismo: simula que no todas las células "despiertan" al mismo tiempo.

### 4. Condición de frontera (Dirichlet)
Después de cada paso se reimponenen los píxeles visibles:

```python
state[:, :3] = state[:, :3] * (1 - mask) + corrupted * mask
```

Los píxeles que el usuario no pintó siempre vuelven a su valor original. Solo la zona del hueco evoluciona libremente.

---

## Estado del autómata

```
Canal 0-2  → RGB de la imagen (zona visible: fijos; hueco: evolucionan)
Canal 3    → máscara binaria (1 = visible, 0 = hueco)
Canal 4-6  → RGB × mask (imagen visible, hueco = 0)
Canales 7-31 → canales ocultos (estado interno latente)
```

Dimensiones en inferencia: `(1, 32, 100, 100)` — batch=1, 32 canales, 100×100 píxeles.

---

## Frontend (`frontend/`)

- **Dos canvas superpuestos**: uno muestra la imagen, el otro recibe los trazos del pincel (máscara).
- **Herramientas**: pincel, borrador, undo/redo (pila de hasta 25 estados), limpiar, atajos de teclado.
- **Reproducción**: los frames devueltos por el backend se animan con `requestAnimationFrame` a FPS configurable.
- **Métricas**: dos gráficas Chart.js (convergencia + calidad) y tarjetas con PSNR, SSIM y paso de convergencia.

---

## Dependencias clave

| Capa     | Tecnología                   |
|----------|------------------------------|
| Backend  | FastAPI, PyTorch, Pillow, NumPy |
| Servidor | uvicorn                      |
| Frontend | Vanilla JS, Chart.js (CDN)   |
| Modelo   | PyTorch checkpoint (.pth)    |
