# De qué trata el proyecto

**Materia:** Autómatas y Lenguajes Formales — Universidad Industrial de Santander  
**Título:** *Reconstrucción de Imágenes Faciales mediante Autómatas Celulares Neurales*  
**Tipo:** Informe Final de Proyecto

---

## El problema que resuelve

Cuando una imagen tiene una zona dañada, borrada o tapada, se necesita *reconstruir* esa región a partir del contexto visible. Este proceso se llama **in-painting**. El proyecto lo resuelve con un enfoque proveniente de los autómatas celulares.

---

## La idea central

Un **autómata celular** es un modelo de cómputo donde muchas células en una grilla evolucionan en paralelo aplicando la misma regla local. El ejemplo más conocido es el *Juego de la Vida de Conway*: reglas binarias simples que producen comportamientos complejos.

La pregunta que articula este trabajo es:

> *¿Qué pasa si el alfabeto de estados es continuo (vectores reales) y la función de transición no se define a mano, sino que se **aprende por optimización**?*

El resultado es un **Autómata Celular Neural (NCA)**: sigue siendo un autómata celular en sentido formal (tiene localidad, homogeneidad y paralelismo), pero su regla de transición δ es una pequeña red neuronal entrenada con gradiente descendente.

---

## Conexión con la teoría

El proyecto parte de la jerarquía de Chomsky (AFD → AP → LBA → MT) y observa que el ingrediente común es siempre la función de transición δ local. Los autómatas celulares generalizan eso a una grilla 2D, y los NCA generalizan el alfabeto a ℝⁿ y aprenden δ por optimización. El modelo sigue cumpliendo las tres propiedades fundamentales:

1. **Localidad** — cada célula solo ve su vecindad de Moore (8 vecinos).
2. **Homogeneidad** — la misma red se aplica en todas las células.
3. **Paralelismo** — todas las células se actualizan en cada paso.

---

## En la práctica

El autómata opera sobre una grilla de **100 × 100 células**, cada una con **32 canales de estado** (los 3 primeros son RGB). Se entrenó sobre ~8 000 imágenes del dataset **CelebA** (rostros humanos). Dada una imagen con una zona pintada (hueco), el autómata itera paso a paso propagando la información de los píxeles visibles hacia el interior del hueco hasta reconstruirlo.
