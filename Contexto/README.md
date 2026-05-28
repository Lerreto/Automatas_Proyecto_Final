# Contexto

Documentación teórica y técnica del proyecto. Estos documentos explican *qué* se hizo, *por qué* y *cómo*, sin necesidad de leer el código. Son la base conceptual de la presentación y del informe.

Lee los documentos en orden:

| # | Documento | De qué trata |
|---|---|---|
| 01 | [`01_de_que_trata.md`](01_de_que_trata.md) | El problema (*in-painting*), la idea central y la conexión con la teoría de autómatas y la jerarquía de Chomsky. |
| 02 | [`02_arquitectura.md`](02_arquitectura.md) | Arquitectura del sistema: modelo NCA (PerceptionBlock + UpdateNetwork), estado de 32 canales, frontera de Dirichlet y el frontend/backend. |
| 03 | [`03_flujo_completo.md`](03_flujo_completo.md) | Flujo de datos de extremo a extremo: del clic del usuario a la animación, con un paso del autómata y las métricas (PSNR, SSIM). |

---

## Resumen en una línea

Un **Autómata Celular Neural (NCA)** reconstruye zonas borradas de rostros: sigue siendo un autómata celular formal (localidad, homogeneidad, paralelismo), pero su regla de transición δ es una red neuronal entrenada sobre el dataset CelebA.

---

## Para profundizar

- Presentación: [`../Diapositivas/`](../Diapositivas/)
- Informe escrito: [`../Informe/`](../Informe/)
- Implementación: [`../Codigo/`](../Codigo/)
