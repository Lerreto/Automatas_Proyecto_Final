# Código

Esta carpeta contiene las dos piezas de software del proyecto:

| Carpeta / archivo | Qué es |
|---|---|
| `nca_entrenamiento.ipynb` | Notebook de **entrenamiento** del modelo NCA (Google Colab). |
| `nca-web/` | **Aplicación web interactiva** que usa el modelo ya entrenado. |

El flujo es: se entrena el modelo en el notebook → se obtiene un checkpoint `.pth` → ese checkpoint se coloca en la app web para reconstruir imágenes en tiempo real.

---

## 1. Notebook de entrenamiento — `nca_entrenamiento.ipynb`

Pipeline completo de entrenamiento, pensado para **Google Colab con GPU** (también corre en local si se tiene una GPU NVIDIA con CUDA).

**Qué hace:**
1. Descarga y descomprime el dataset **CelebA** (rostros) desde Google Drive.
2. Define el modelo NCA (las mismas clases que están en `nca-web/backend/model.py`).
3. Entrena con un esquema *warm-up + gradiente*: en cada muestra corre el autómata unos pasos sin gradiente y luego unos pasos con gradiente, para que aprenda a reconstruir de forma estable.
4. Guarda el checkpoint entrenado (`latest.pth` / `best.pth`).

**Hiperparámetros principales:**

| Parámetro | Valor |
|---|---|
| Canales de estado (`CHANNELS`) | 32 |
| Resolución | 100 × 100 |
| Épocas | 300 |
| Batch | 6 (× 2 de acumulación → efectivo 12) |
| Optimizador | AdamW, `lr = 2e-4`, `weight_decay = 1e-4` |
| Scheduler | OneCycleLR |
| Pasos NCA | 20 de *warm-up* (sin gradiente) + 8 con gradiente |
| Tasa de actualización | Bernoulli(0.5) |

### Cómo ejecutarlo

**Opción A — Google Colab (recomendada):**
1. Sube `nca_entrenamiento.ipynb` a [Google Colab](https://colab.research.google.com/).
2. Menú **Entorno de ejecución → Cambiar tipo de entorno → GPU**.
3. Ajusta las rutas de Drive (`DRIVE_DIR`) a tu cuenta y ejecuta las celdas en orden.

**Opción B — Jupyter local (requiere GPU para tiempos razonables):**

Linux / Mac:
```bash
python3 -m venv venv
source venv/bin/activate
pip install jupyter torch torchvision pillow numpy matplotlib
jupyter notebook nca_entrenamiento.ipynb
```

Windows (PowerShell o CMD):
```bat
python -m venv venv
venv\Scripts\activate
pip install jupyter torch torchvision pillow numpy matplotlib
jupyter notebook nca_entrenamiento.ipynb
```

> El checkpoint ya entrenado viene incluido en `nca-web/backend/checkpoints/latest.pth`, así que **no es necesario reentrenar** para probar la aplicación web.

---

## 2. Aplicación web — `nca-web/`

App de escritorio en el navegador para cargar una imagen, pintar la zona a borrar y verla reconstruir paso a paso. Backend en **FastAPI + PyTorch**, frontend en **HTML/CSS/JS** puro.

### Requisitos
- **Python 3.10 o superior**
- GPU NVIDIA con CUDA *(opcional)* — sin ella corre en CPU, solo un poco más lento.

### Ejecutar en Linux / Mac
```bash
cd nca-web
chmod +x run.sh      # solo la primera vez
./run.sh
```

### Ejecutar en Windows
```bat
cd nca-web
run.bat
```
(o haz doble clic en `run.bat` desde el explorador de archivos)

### Ejecutar manualmente (cualquier sistema operativo)

Linux / Mac:
```bash
cd nca-web
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn backend.app:app --host 0.0.0.0 --port 8000
```

Windows:
```bat
cd nca-web
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
uvicorn backend.app:app --host 0.0.0.0 --port 8000
```

Luego abre **http://localhost:8000** en el navegador.

> Los scripts `run.sh` y `run.bat` crean el entorno virtual, instalan dependencias y arrancan el servidor automáticamente. Detén el servidor con `Ctrl+C`.

Instrucciones completas de uso, atajos de teclado y solución de problemas en **[`nca-web/README.md`](nca-web/README.md)**.

---

## Estructura

```
Codigo/
├── nca_entrenamiento.ipynb     ← Entrenamiento (Colab)
└── nca-web/                    ← Aplicación web
    ├── backend/
    │   ├── app.py              ← API FastAPI (rutas /api/*)
    │   ├── inference.py        ← Loop de reconstrucción + métricas (PSNR, SSIM)
    │   ├── model.py            ← Modelo PyTorch (PerceptionBlock, UpdateNetwork, NCA)
    │   └── checkpoints/
    │       └── latest.pth      ← Pesos entrenados (CHANNELS=32, 100×100)
    ├── frontend/
    │   ├── index.html
    │   ├── script.js
    │   └── style.css
    ├── requirements.txt
    ├── run.sh                  ← Arranque Linux/Mac
    └── run.bat                 ← Arranque Windows
```

> El modelo en `backend/model.py` se define con `channels=96` por defecto, pero tanto el entrenamiento como la inferencia lo instancian con **`CHANNELS = 32`**, que es la configuración del checkpoint incluido.
