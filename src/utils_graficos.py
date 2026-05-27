import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns
import numpy as np

# Configuración global para que los gráficos se vean unificados y profesionales
sns.set_theme(style="whitegrid", palette="muted")

def plot_datos_reales(X, y, titulo="Distribución Real de los Datos"):
    """
    Grafica la dispersión espacial de los datos sintéticos crudos, 
    coloreados por su etiqueta latente real (el clúster generador).
    """
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y, palette="deep", s=50, alpha=0.8, edgecolor="k")
    plt.title(titulo, fontsize=14, pad=15)
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.legend(title="Clúster Real")
    plt.tight_layout()
    plt.show()

def plot_comparacion_modelos(X, y_km, y_gmm, c_km, c_gmm, titulo="K-Means vs GMM"):
    """
    Crea un panel lado a lado comparando las asignaciones (hard/soft) 
    y la ubicación de los centroides de ambos modelos.
    
    Parámetros:
    X: Coordenadas de los datos.
    y_km, y_gmm: Etiquetas predichas por cada modelo.
    c_km, c_gmm: Coordenadas de los centroides/medias ajustados.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharex=True, sharey=True)
    
    # Panel 1: K-Means
    sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y_km, palette="Set1", ax=axes[0], s=40, alpha=0.6, legend=False)
    axes[0].scatter(c_km[:, 0], c_km[:, 1], c='black', marker='X', s=200, label='Centroides')
    axes[0].set_title("Predicción K-Means (Hard Clustering)", fontsize=13)
    axes[0].legend()

    # Panel 2: GMM
    sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y_gmm, palette="Set1", ax=axes[1], s=40, alpha=0.6, legend=False)
    axes[1].scatter(c_gmm[:, 0], c_gmm[:, 1], c='black', marker='X', s=200, label='Medias')
    axes[1].set_title("Predicción GMM (Soft Clustering)", fontsize=13)
    axes[1].legend()

    fig.suptitle(titulo, fontsize=16, fontweight='bold', y=1.05)
    plt.tight_layout()
    plt.show()

def plot_montecarlo_boxplots(df_resultados):
    """
    Grafica los boxplots del ARI comparando ambos modelos por escenario.
    (Versión mejorada con estética para el reporte).
    """
    plt.figure(figsize=(10, 6))
    
    # Usamos un palette contrastante para diferenciar rápidamente los algoritmos
    sns.boxplot(data=df_resultados, x='Escenario', y='ARI', hue='Modelo', palette=['#e74c3c', '#3498db'])
    
    plt.title('Calidad de Partición (ARI) mediante Simulación Montecarlo', fontsize=14, pad=15)
    plt.ylabel('Adjusted Rand Index (ARI)')
    plt.xlabel('Escenario (Aumento de Complejidad Geométrica)')
    plt.ylim(-0.1, 1.1)
    plt.axhline(1.0, color='gray', linestyle='--', alpha=0.5) # Línea de perfección
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(title='Algoritmo', loc='lower left')
    plt.tight_layout()
    plt.show()

def plot_bootstrap_centroids(X, centroides_km, centroides_gmm, titulo="Inestabilidad Paramétrica (Bootstrap)"):
    """
    Grafica las nubes de puntos de los centroides calculados en las B iteraciones
    de Bootstrap, superpuestas sobre los datos originales atenuados.
    
    Parámetros:
    centroides_km: array de forma (B_iteraciones, K_clusters, 2)
    centroides_gmm: array de forma (B_iteraciones, K_clusters, 2)
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharex=True, sharey=True)
    
    # Para graficar correctamente, aplanamos las dimensiones de los centroides
    # B_iteraciones * K_clusters = Total de puntos a graficar
    c_km_flat = centroides_km.reshape(-1, 2)
    c_gmm_flat = centroides_gmm.reshape(-1, 2)

    # Panel 1: K-Means Bootstrap
    axes[0].scatter(X[:, 0], X[:, 1], c='lightgray', s=10, alpha=0.3, label='Datos Originales')
    axes[0].scatter(c_km_flat[:, 0], c_km_flat[:, 1], c='red', s=15, alpha=0.1, edgecolors='none')
    axes[0].set_title("Dispersión de Centroides K-Means", fontsize=13)
    axes[0].legend()

    # Panel 2: GMM Bootstrap
    axes[1].scatter(X[:, 0], X[:, 1], c='lightgray', s=10, alpha=0.3, label='Datos Originales')
    axes[1].scatter(c_gmm_flat[:, 0], c_gmm_flat[:, 1], c='blue', s=15, alpha=0.1, edgecolors='none')
    axes[1].set_title("Dispersión de Medias GMM", fontsize=13)
    axes[1].legend()

    fig.suptitle(titulo, fontsize=16, fontweight='bold', y=1.05)
    plt.tight_layout()
    plt.show()

def plot_gmm_background_certainty(X, y_real, gmm, titulo="Regiones de Decisión y Certeza de Fondo en GMM"):
    """
    Grafica los puntos reales de forma sólida y colorea el fondo del gráfico 
    según el clúster dominante en cada región espacial, disipando el color 
    (aumentando la transparencia) en las zonas de incertidumbre fronteriza.
    
    Parámetros:
    -----------
    X : np.array de forma (N, 2)
        Coordenadas de los puntos del dataset.
    y_real : np.array de forma (N,)
        Etiquetas reales o predichas para los puntos (para colorear los puntos sólidos).
    gmm : GaussianMixture
        El modelo GMM ya entrenado (.fit() ejecutado previamente).
    titulo : str
        Título del gráfico.
    """
    plt.figure(figsize=(10, 8))
    
    # 1. Definir los límites del gráfico basados en tus datos reales
    x_min, x_max = X[:, 0].min() - 1.5, X[:, 0].max() + 1.5
    y_min, y_max = X[:, 1].min() - 1.5, X[:, 1].max() + 1.5
    
    # 2. Crear la malla (Meshgrid) para evaluar el espacio de fondo
    # 300x300 puntos es un excelente balance entre resolución visual y velocidad computacional
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                         np.linspace(y_min, y_max, 300))
    
    # Aplanar la malla para pasársela al GMM como un vector de coordenadas (N_malla, 2)
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    
    # 3. Obtener las probabilidades blandas de la grilla de fondo
    grid_probs = gmm.predict_proba(grid_points)
    grid_pred = np.argmax(grid_probs, axis=1) # Clúster ganador en cada pixel
    grid_certeza = np.max(grid_probs, axis=1) # Nivel de certeza [0.33 a 1.0]
    
    # 4. Normalizar la certeza de fondo para el canal Alpha
    K = gmm.n_components
    prob_min = 1.0 / K
    alpha_background = (grid_certeza - prob_min) / (1.0 - prob_min)
    alpha_background = np.clip(alpha_background, 0.0, 1.0) # Forzar rango estricto [0, 1]
    
    # 5. Mapear colores de fondo (Matriz RGBA de píxeles)
    unique_labels = np.arange(K)
    base_palette = sns.color_palette("deep", K)
    
    rgba_background = np.zeros((grid_points.shape[0], 4))
    for i, label in enumerate(grid_pred):
        rgba_background[i, :3] = base_palette[label]     # Color del clúster dominante
        rgba_background[i, 3] = alpha_background[i] * 0.4 # Multiplicamos por 0.4 para que sea un fondo sutil y no tape los puntos
        
    # Reestructurar la matriz plana RGBA a las dimensiones de la imagen (300, 300, 4)
    rgba_image = rgba_background.reshape(xx.shape + (4,))
    
    # 6. Pintar el fondo usando imshow
    plt.imshow(rgba_image, extent=(x_min, x_max, y_min, y_max), origin='lower', aspect='auto')
    
    # 7. Graficar los puntos reales ARRIBA del fondo (completamente sólidos y delineados)
    # Usamos la misma paleta para mantener la correlación
    scatter = sns.scatterplot(
        x=X[:, 0], y=X[:, 1], hue=y_real, 
        palette="deep", s=55, alpha=1.0, edgecolor="black", linewidth=0.8
    )
    
    # 8. Estética final y Leyendas
    plt.title(titulo, fontsize=14, pad=15, fontweight='bold')
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    
    # Leyenda manual para evitar duplicados feos en matplotlib
    legend_elements = [Patch(facecolor=base_palette[k], edgecolor='k', label=f'Clúster {k}') 
                       for k in unique_labels]
    plt.legend(handles=legend_elements, title="Regiones de Clúster", loc="upper right")
    
    plt.tight_layout()
    plt.show()

# Ejemplo de como ejecutarlo en un notebook:
    # from sklearn.mixture import GaussianMixture

    # # 1. Entrenas tu GMM con el set de datos del escenario actual
    # gmm_modelo = GaussianMixture(n_components=3, covariance_type='full', random_state=42)
    # gmm_modelo.fit(X)

    # # 2. Invocas la nueva función de fondo disipado
    # plot_gmm_background_certainty(
    #     X=X, 
    #     y_real=y,  # Tu columna de "verdad absoluta" generada originalmente
    #     gmm=gmm_modelo, 
    #     titulo="Escenario 4: Análisis Espacial de Fronteras con Fondo de Incertidumbre"
    # )