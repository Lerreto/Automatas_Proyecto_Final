# NCA · Reconstrucción de imágenes con Autómata Celular Neuronal

Aplicación web para reconstruir zonas de imágenes usando un Neural Cellular Automata (NCA) entrenado sobre CelebA a resolución 100×100.

---

## Requisitos

- Python 3.10 o superior
- (Opcional) GPU NVIDIA con CUDA para inferencia rápida

---

## 1. Coloca el checkpoint

Copia tu archivo de modelo entrenado aquí:

```
nca-web/
└── backend/
    └── checkpoints/
        └── latest.pth     ← EL ARCHIVO VA AQUÍ
```

El backend también acepta `best.pth` si lo renombras a `latest.pth`, o si modificas `CHECKPOINT_PATH` en `backend/app.py`.

---

## 2. Ejecuta la aplicación

**Linux / Mac:**
```bash
chmod +x run.sh
./run.sh
```

**Windows:**
```
run.bat
```

**Manualmente (cualquier SO):**
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn backend.app:app --host 0.0.0.0 --port 8000
```

---

## 3. Abre el navegador

```
http://localhost:8000
```

---

## 4. Cómo usar

1. **Load image** — carga cualquier foto (se recorta automáticamente al cuadrado central).
2. **Pinta la zona a reconstruir** con el pincel negro sobre la imagen.
3. Ajusta los **pasos** del NCA (default 120). Más pasos = reconstrucción más detallada pero más lenta.
4. Click en **Reconstruct** — el backend ejecuta el autómata y devuelve los frames.
5. La animación se reproduce automáticamente. Puedes pausarla, hacer scrub con el slider, o ajustar el FPS.
6. **Save frame** guarda el frame actual como PNG.

---

## 5. Atajos de teclado

| Tecla | Acción |
|---|---|
| `B` | Pincel |
| `E` | Borrador |
| `Ctrl+Z` | Deshacer |
| `Ctrl+Shift+Z` / `Ctrl+Y` | Rehacer |
| `Del` | Limpiar máscara |
| `Space` | Play / Pause |
| `L` | Cargar imagen |

---

## 6. Troubleshooting

**Puerto 8000 ocupado:**
```bash
uvicorn backend.app:app --port 8001
# Luego abre http://localhost:8001
```

**No detecta CUDA:**
El modelo corre en CPU automáticamente. Es más lento (~5–15 s por reconstrucción) pero funciona igual.

**Error "Model checkpoint not found":**
Verifica que el archivo `latest.pth` esté en `backend/checkpoints/`.

**Error de arquitectura al cargar el modelo:**
Asegúrate de usar el `latest.pth` entrenado con CHANNELS=32 e IMG_SIZE=100. Si el modelo tiene parámetros distintos, ajusta las constantes en `backend/inference.py`.

---

## Arquitectura del modelo

```
Estado: (B, 32, 100, 100)
  ↓ PerceptionBlock (Sobel X/Y + Laplaciano, depthwise)
  ↓ UpdateNetwork (4× Conv 1×1 con GELU)
  ↓ + Δs · Bernoulli(0.5)
  ↓ Reimposición de frontera visible
  × 120 pasos
```

Las zonas pintadas se inicializan en negro (0) y el autómata las regenera iterativamente, guiado por el contexto de los píxeles vecinos visibles.
