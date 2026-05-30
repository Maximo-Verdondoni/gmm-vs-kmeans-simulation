# 🚀 GMM vs. K-Means: Cuando la Geometría Rompe los Algoritmos

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine_Learning-orange.svg)
![Simulación](https://img.shields.io/badge/Métodos-Montecarlo_%7C_Bootstrap-success.svg)

K-Means es el algoritmo de clustering más utilizado en la industria por su velocidad y simplicidad, pero tiene un talón de Aquiles estructural: **asume que el mundo está hecho de esferas perfectas y simétricas**. 

¿Qué pasa cuando los datos reales presentan clústeres elípticos, densidades desiguales o alta superposición? En este proyecto demostramos empíricamente por qué los algoritmos de fronteras rígidas (Hard Clustering) colapsan frente a la complejidad espacial, y cómo los **Modelos de Mezclas Gaussianas (GMM)** logran descifrar la estructura latente mediante asignación probabilística (Soft Clustering).

---

## 🧠 Destacados del Proyecto

Este repositorio no es solo una comparativa de modelos, es un entorno de experimentación estadística completo. Incluye:

- ⚙️ **Mecanismo Generador de Datos (MGD):** Un motor de simulación de mixturas gaussianas en $\mathbb{R}^2$ para crear "pruebas de estrés" geométricas controladas.
- 🎲 **Simulación de Montecarlo:** Ejecución de cientos de réplicas para evaluar la esperanza matemática de métricas extrínsecas (ARI, Homogeneidad, Completitud).
- 🔄 **Resolución de Label Switching:** Implementación del *Algoritmo Húngaro* (asignación lineal) para solucionar la permutación aleatoria de etiquetas en la inestabilidad paramétrica.
- 📊 **Visualizaciones Avanzadas:** Renderizado de funciones de densidad de probabilidad (PDF) en 3D con proyección de autovectores, mapeo de fronteras difusas (alpha mapping) y elipses de confianza paramétricas.

---

## 📂 Arquitectura del Repositorio

El código fuente está estructurado siguiendo buenas prácticas de MLOps, separando la lógica matemática de la visualización y el análisis.

```text
.
├── notebooks/              # Notebooks
│   ├── Estudio.ipynb       # Notebook principal con la narrativa, análisis y conclusiones
├── src/                    # Módulos Python reutilizables
│   ├── data_generation.py  # Lógica del MGD y perturbación espacial
│   ├── simulation.py       # Pipeline de Montecarlo y recolección de métricas
│   ├── bootstrap.py        # Remuestreo no paramétrico y anclaje de centroides
│   └── utils.py            # Caja de herramientas visuales (Seaborn / Matplotlib 3D)
├── requirements.txt
└── README.md
```

---

## 🧪 Pruebas de Estrés: Los 5 Escenarios

El espacio latente fue deformado progresivamente para evaluar el límite de ruptura de ambos algoritmos:

| # | Escenario | Geometría | El Desafío Algorítmico |
|---|-----------|-----------|------------------------|
| 1 | **Control** | Esferas perfectas y separadas | Línea base. Empate técnico esperado. |
| 2 | **Heterocedasticidad** | Varianzas fuertemente desiguales | K-Means es propenso a "robar" puntos de clústeres dispersos. |
| 3 | **Anisotropía** | Elipses con distintas orientaciones | K-Means divide las elipses a la mitad. GMM ajusta matrices de covarianza. |
| 4 | **Superposición** | Elipses cercanas | Prueba de fuego para evaluar la incertidumbre en las fronteras. |
| 5 | **Colapso Total** | Ruido y superposición extrema | Evaluación del *Overfitting* y varianza paramétrica. |

---

## 📈 Hallazgos Principales (Bias-Variance Trade-off)

1. **El Sesgo de K-Means:** Ante la anisotropía (Escenario 3), K-Means presenta un alto sesgo geométrico, desplazando sus centroides lejos del parámetro poblacional real ($\mu$) al forzar celdas de Voronoi donde no corresponden.
2. **La Robustez de GMM:** GMM se mantiene insesgado frente a varianzas y covarianzas desiguales, logrando una calidad de partición (ARI) superior.
3. **La Trampa de la Complejidad:** En escenarios de ruido puro (Escenario 5), la flexibilidad de GMM le juega en contra (alta varianza paramétrica), sobreajustándose al ruido en cada iteración de Bootstrap, mientras que K-Means mantiene estimaciones erróneas pero estables.

---

## ⚙️ Reproducción del Estudio

Para correr las simulaciones y generar los gráficos 3D localmente, se requieren las siguientes dependencias:

```bash
# Clonar el repositorio
git clone [https://github.com/tu-usuario/gmm-vs-kmeans-simulation.git](https://github.com/tu-usuario/gmm-vs-kmeans-simulation.git)
cd gmm-vs-kmeans-simulation

# Instalar dependencias
pip install -r requirements.txt

# Iniciar el entorno interactivo
jupyter notebook Estudio.ipynb
```

---

## 👥 Sobre los Autores

Este proyecto fue desarrollado como estudio final para la asignatura **Cálculo Numérico y Simulación**, correspondiente a la **Licenciatura en Ciencia de Datos** de la **Universidad Austral**.

**Equipo de Data Science:**
* Marcos Ziadi
* Juan Martín Leoni
* Facundo Rubiolo
* Máximo Verdondoni

*Agradecimientos especiales al equipo docente (Marcos Prunello y Julián L'Heureux) por la guía académica durante el desarrollo metodológico.*
