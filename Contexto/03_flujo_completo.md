# Flujo completo — de clic a animación

## Paso a paso

```
Usuario                  Frontend (JS)              Backend (Python)
  │                          │                            │
  │── carga imagen ──────────►│                            │
  │                          │ center-crop → canvas       │
  │                          │                            │
  │── pinta zona ────────────►│                            │
  │   (pincel negro)         │ maskCanvas acumula         │
  │                          │ trazos del usuario         │
  │                          │                            │
  │── click "reconstruct" ───►│                            │
  │                          │ extractForModel()           │
  │                          │  ├─ imagen corrupta (negro  │
  │                          │  │  en zona pintada)        │
  │                          │  ├─ imagen original          │
  │                          │  └─ máscara binaria          │
  │                          │  todo a 100×100 → base64    │
  │                          │                            │
  │                          │── POST /api/reconstruct ───►│
  │                          │   {image, original, mask,  │
  │                          │    steps=120, every=2}      │
  │                          │                            │
  │                          │               decode b64   │
  │                          │               ↓            │
  │                          │          init_state()      │
  │                          │          state(1,32,100,100)│
  │                          │               ↓            │
  │                          │          loop 120 pasos:   │
  │                          │           Perceive(state)  │
  │                          │           → Update(→Δs)    │
  │                          │           → s += Δs·Bern   │
  │                          │           → Dirichlet      │
  │                          │           → métricas       │
  │                          │           → cada 2 pasos   │
  │                          │             guarda frame   │
  │                          │               ↓            │
  │                          │◄── {frames[], metrics} ────│
  │                          │    ~60 PNGs base64         │
  │                          │                            │
  │                          │ updateCharts(metrics)      │
  │                          │ startPlayback()            │
  │◄── animación + gráficas ─│                            │
```

---

## Inicialización del estado (`NCA.init_state`)

```python
state = zeros(1, 32, 100, 100)
state[:, 0:3] = imagen_rgb          # canales RGB
state[:, 3:4] = mascara             # 1=visible, 0=hueco
state[:, 4:7] = imagen_rgb * mascara  # RGB solo en zona visible
# canales 7-31 quedan en 0
```

Los canales ocultos (7–31) parten en cero y el NCA los usa libremente como memoria interna durante la reconstrucción.

---

## Un paso del autómata

```
state (1,32,H,W)
  │
  ▼ PerceptionBlock
  │  depthwise conv con kernels fijos (Sobel X, Sobel Y, Laplaciano)
  │  concatena: [state | grad_x | grad_y | laplaciano]
  │  → (1, 128, H, W)
  │
  ▼ UpdateNetwork
  │  Conv1×1(128→256) → GELU
  │  Conv1×1(256→256) → GELU
  │  Conv1×1(256→128) → GELU
  │  Conv1×1(128→32)         ← inicializado en cero
  │  → Δstate (1, 32, H, W)
  │
  ▼ Actualización estocástica
  │  mask_aleatoria ~ Bernoulli(p=0.5)
  │  state = state + Δstate * mask_aleatoria
  │
  ▼ Condición de frontera Dirichlet
     state[:, 0:3] = state[:, 0:3] * (1-mascara) + corrupted * mascara
     ← los píxeles visibles se "clavan" siempre a su valor original
```

---

## Métricas calculadas en cada paso

| Métrica        | Qué mide                                                  |
|----------------|-----------------------------------------------------------|
| `convergence`  | Cambio RGB medio entre pasos `||s_t − s_{t-1}||`          |
| `activity`     | % de células con cambio RGB > ε (5×10⁻⁴)                 |
| `psnr_hole`    | PSNR entre la zona reconstruida y el ground truth (dB)    |
| `ssim_hole`    | SSIM estructural en la zona del hueco (0–1)               |

El modelo converge cuando `convergence` cae por debajo de ε — empíricamente cerca del paso 80–120.

---

## Estructura de archivos de referencia

```
Codigo/nca-web/
├── backend/
│   ├── app.py          ← FastAPI: rutas, lifespan, decode imágenes
│   ├── inference.py    ← loop de reconstrucción, métricas, conversión a PNG
│   ├── model.py        ← PerceptionBlock, UpdateNetwork, NCA
│   └── checkpoints/
│       └── latest.pth  ← pesos entrenados (CHANNELS=32, IMG_SIZE=100)
├── frontend/
│   ├── index.html      ← layout: dos paneles + sección métricas
│   ├── script.js       ← toda la lógica UI, pintura, fetch, playback
│   └── style.css       ← estilos
├── requirements.txt
└── run.sh / run.bat    ← scripts de arranque
```
