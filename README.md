# GMM vs. K-Means: Estudio por Simulación

**Trabajo Práctico Grupal — Cálculo Numérico y Simulación**  
Lic. en Ciencia de Datos · Universidad Austral

---

## Integrantes

| Nombre |
|--------|
| Marcos Ziadi |
| Juan Martín Leoni |
| Facundo Rubiolo |
| Máximo Verdondoni |

---

## Descripción

Este proyecto lleva adelante un estudio por simulación completo para responder la siguiente pregunta de investigación:

> ¿Cómo se degradan la precisión de partición geométrica y la estabilidad paramétrica de los centroides estimados por **K-Means** en comparación con un **Modelo de Mezclas Gaussianas (GMM)** cuando se violan de forma progresiva y controlada los supuestos de esfericidad, homogeneidad de varianza e independencia en los clústeres reales?

---

## Estructura del Repositorio

```
.
├── Estudio.ipynb          # Notebook principal con metodología, experimentos y análisis
├── README.md
└── (scripts auxiliares .py si aplica)
```

---

## Metodología

El estudio recorre las etapas canónicas de una simulación:

1. **Formulación del problema** — Pregunta de investigación, objetivos e hipótesis previas.
2. **Modelización conceptual** — Factores fijos y variables, supuestos de cada algoritmo.
3. **Mecanismo Generador de Datos (MGD)** — Mixtura de gaussianas multivariadas en $\mathbb{R}^2$ con $N = 500$ observaciones y $K = 3$ clústeres equiprobables.
4. **Diseño de escenarios** — Cuatro configuraciones que violan progresivamente los supuestos de K-Means.
5. **Implementación y registro de métricas** — Dos flujos de evaluación independientes.
6. **Experimentación** — Ejecución de los experimentos con suficientes réplicas para resultados estables.
7. **Análisis de resultados y conclusiones** — Tablas, boxplots y scatter plots de centroides.

---

## Escenarios

| # | Nombre | Descripción |
|---|--------|-------------|
| 1 | **Control** | Clústeres esféricos, varianza homogénea, bien separados |
| 2 | **Heterocedasticidad** | Varianzas fuertemente desiguales entre clústeres ($\sigma^2 \in \{0.5, 2.0, 5.0\}$) |
| 3 | **Anisotropía** | Covarianzas no nulas: clústeres elípticos con distintas orientaciones |
| 4 | **Anisotropía + Superposición** | Estructura elíptica del Escenario 3 con centroides más cercanos |

---

## Algoritmos Evaluados

- **K-Means** (`KMeans(n_clusters=3)`) — asignación dura, fronteras de Voronoi, asume esfericidad e igual varianza.
- **GMM** (`GaussianMixture(n_components=3, covariance_type='full')`) — asignación probabilística, estima covarianza completa por componente.

---

## Métricas

### Flujo A — Simulación de Montecarlo ($M = 200$ réplicas por escenario)
- **Adjusted Rand Index (ARI)**: mide la coincidencia entre etiquetas predichas y reales, corregida por azar (rango $[-1, 1]$; 1 = partición perfecta).

### Flujo B — Bootstrap no paramétrico ($B = 500$ iteraciones por escenario)
- **Coordenadas de centroides estimados** $(X_1, X_2)$: permiten cuantificar la varianza e intervalos de confianza empíricos de la localización de cada centroide.

---

## Hipótesis

Se postula que bajo el **Escenario 1 (Control)** ambos algoritmos exhibirán rendimiento equivalente. A medida que se introduce heterocedasticidad y anisotropía, **K-Means** sufrirá una degradación severa en ARI y alta varianza en la ubicación de sus centroides (Bootstrap), mientras que **GMM** mantendrá robustez paramétrica por su capacidad de modelar covarianzas completas.

---

## Requisitos

```bash
pip install numpy pandas scikit-learn matplotlib seaborn jupyter
```

---

## Ejecución

```bash
jupyter notebook Estudio.ipynb
```

---

## Materia y Docentes

**Cálculo Numérico y Simulación**  
Docentes: Marcos Prunello · Julián L'Heureux  
Licenciatura en Ciencia de Datos — Universidad Austral
