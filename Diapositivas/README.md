# Diapositivas

Presentación de apoyo del proyecto **Reconstrucción de Imágenes Faciales con Autómatas Celulares Neurales** — Autómatas y Lenguajes Formales, Universidad Industrial de Santander.

---

## Presentación en Canva (versión online)

**[Abrir la presentación en Canva ▸](https://canva.link/8uyik28td4ph7p7)**

> https://canva.link/8uyik28td4ph7p7

Es la versión interactiva y siempre actualizada de las diapositivas. Se abre en el navegador, no requiere instalar nada y permite presentar en pantalla completa desde Canva.

---

## Archivo local

`Presentacion Proyecto Automatas.pdf` — exportación en PDF de la presentación, por si se necesita mostrar sin conexión a internet.

> El PDF pesa ~75 MB porque incluye las imágenes y capturas en alta resolución. Para la exposición en vivo se recomienda el enlace de Canva; el PDF queda como respaldo offline.

---

## Contenido de la presentación

La presentación recorre el proyecto de principio a fin:

1. **Problema** — reconstrucción de zonas borradas en imágenes (*in-painting*).
2. **Marco teórico** — de la jerarquía de Chomsky y la función de transición δ a los autómatas celulares y su generalización neuronal.
3. **Modelo NCA** — alfabeto continuo (ℝ³²), regla aprendida, y las tres propiedades formales: localidad, homogeneidad y paralelismo.
4. **Arquitectura** — PerceptionBlock (Sobel + Laplaciano) + UpdateNetwork, actualización estocástica y frontera de Dirichlet.
5. **Entrenamiento** — dataset CelebA, ~300 épocas en Google Colab.
6. **Demo** — aplicación web interactiva y resultados sobre fotos del equipo.

---

## Recursos relacionados

- Explicación teórica detallada: [`../Contexto/`](../Contexto/)
- Informe escrito: [`../Informe/`](../Informe/)
- Video de demostración: [`../Video/`](../Video/)
- Código y app web: [`../Codigo/`](../Codigo/)
