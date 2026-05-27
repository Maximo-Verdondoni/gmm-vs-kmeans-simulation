import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Ellipse
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

def plot_montecarlo_violin(df_resultados):
    """
    Grafica la distribución del ARI comparando ambos modelos por escenario.
    Utiliza un violin plot dividido para mostrar la densidad real y los cuartiles de 
    las simulaciones Montecarlo.
    """
    plt.figure(figsize=(10, 6))
    
    # Violinplot con split=True funde ambas distribuciones en una sola figura simétrica por escenario
    sns.violinplot(
        data=df_resultados, 
        x='Escenario', 
        y='ARI', 
        hue='Modelo',
        gap=0.1,
        dodge=True, 
        inner='quart',     # Dibuja las líneas de los cuartiles dentro del violín
        palette=['#e74c3c', '#3498db']
    )
    
    plt.title('Distribución de Calidad de Partición (ARI) mediante Simulación Montecarlo', fontsize=14, pad=15)
    plt.ylabel('Adjusted Rand Index (ARI)')
    plt.xlabel('Escenario (Aumento de Complejidad Geométrica)')
    plt.ylim(-0.1, 1.1)
    plt.axhline(1.0, color='gray', linestyle='--', alpha=0.5) # Línea de perfección
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Ajuste de leyenda en caso de que split=True la duplique o altere
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

def plot_elipse_confianza(ax, datos_xy, color, n_std=2.0):
    """
    Dibuja la elipse de confianza empírica (por defecto 95% con n_std=2) 
    para una nube de puntos 2D.
    """
    # Matriz de covarianza de la muestra empírica
    cov = np.cov(datos_xy[:, 0], datos_xy[:, 1])
    
    # Autovalores y autovectores para definir la forma y ángulo de la elipse
    vals, vecs = np.linalg.eigh(cov)
    
    # Ordenar de mayor a menor
    orden = vals.argsort()[::-1]
    vals, vecs = vals[orden], vecs[:, orden]
    
    # Calcular el ángulo de rotación de la elipse
    angulo = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    
    # El ancho y alto están dados por la raíz cuadrada de los autovalores escalados
    ancho, alto = 2 * n_std * np.sqrt(vals)
    
    # Crear y añadir el parche de la elipse al eje
    elipse = Ellipse(xy=(datos_xy[:, 0].mean(), datos_xy[:, 1].mean()),
                     width=ancho, height=alto, angle=angulo,
                     edgecolor=color, facecolor='none', linewidth=2, linestyle='--')
    ax.add_patch(elipse)

def plot_bootstrap_centroids_bootstrap(X, centroides_km, centroides_gmm, titulo="Estabilidad de Centroides (Bootstrap)"):
    """
    Grafica la dispersión de los centroides obtenidos por Bootstrap para K-Means y GMM,
    superponiendo elipses de confianza del 95%.
    """
    COLORES = ['#e74c3c', '#2ecc71', '#3498db']  # uno por clúster
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharex=True, sharey=True)

    for ax, c_boot, nombre in zip(axes, [centroides_km, centroides_gmm], ['K-Means', 'GMM']):
        # Datos de fondo
        ax.scatter(X[:, 0], X[:, 1], c='lightgray', s=10, alpha=0.3, label='Datos')
        
        for k in range(3):
            # Extraemos las coordenadas (X, Y) de todos los bootstraps para el clúster k
            datos_cluster = c_boot[:, k, :]
            
            # 1. Puntos semi-transparentes de fondo (muy tenues)
            ax.scatter(datos_cluster[:, 0], datos_cluster[:, 1],
                       c=COLORES[k], s=15, alpha=0.05, edgecolors='none')
            
            # 2. Elipse de confianza del 95% (n_std=2.0)
            plot_elipse_confianza(ax, datos_cluster, color=COLORES[k], n_std=2.0)
            
            # Dummy scatter invisible solo para que la leyenda se vea limpia
            ax.scatter([], [], c=COLORES[k], label=f'Clúster {k}')

        ax.set_title(f"Bootstrap {nombre}", fontsize=13)
        ax.legend(markerscale=1.5, loc='best')

    fig.suptitle(titulo, fontsize=16, fontweight='bold')
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
    sns.scatterplot(
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

def plot_silhouette_comparacion(df_resultados):
    plt.figure(figsize=(10, 6))
    sns.violinplot(
        data=df_resultados, x='Escenario', y='Silhouette', hue='Modelo',
        gap=0.1, dodge=True , inner='quart', palette=['#e74c3c', '#3498db']
    )
    plt.title('Silhouette Score por Escenario y Modelo', fontsize=14, pad=15)
    plt.ylabel('Silhouette Score')
    plt.xlabel('Escenario')
    plt.axhline(0, color='gray', linestyle='--', alpha=0.5)
    plt.legend(title='Algoritmo', loc='lower left')
    plt.tight_layout()
    plt.show()

def plot_heatmap_varianza_bootstrap(df_stats):
    for coord in ['X1', 'X2']:
        pivot = (df_stats[df_stats['Coordenada'] == coord]
                 .pivot_table(index=['Modelo', 'Clúster'], columns='Escenario', values='Varianza'))
        plt.figure(figsize=(8, 4))
        sns.heatmap(pivot, annot=True, fmt='.4f', cmap='YlOrRd', linewidths=0.5)
        plt.title(f'Varianza Bootstrap de Centroides — Coordenada {coord}')
        plt.tight_layout()
        plt.show()

def plot_intervalos_centroides(df_stats):
    """
    Grafica la evolución de las medias de los centroides y su IC del 95%
    a lo largo de los escenarios, separando por Coordenada y Clúster.
    """
    # Configuramos una grilla de 2 filas (X1, X2) x 3 columnas (Clústeres 0, 1, 2)
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 8), sharex=True)
    coords = ['X1', 'X2']
    modelos = ['K-Means', 'GMM']
    colores = {'K-Means': '#e74c3c', 'GMM': '#3498db'}

    # Obtenemos la lista de escenarios únicos para el eje X
    escenarios_unicos = df_stats['Escenario'].unique()
    x_pos = np.arange(len(escenarios_unicos))

    for i, coord in enumerate(coords):
        for k in range(3): # Clústeres 0, 1, 2
            ax = axes[i, k]
            
            for modelo in modelos:
                # Filtrar el subset específico
                mask = (df_stats['Coordenada'] == coord) & (df_stats['Clúster'] == k) & (df_stats['Modelo'] == modelo)
                subset = df_stats[mask]
                
                if subset.empty:
                    continue
                
                # Jitter: Desplazamos un poco K-Means a la izq y GMM a la der para que no se superpongan las barras
                desplazamiento = -0.15 if modelo == 'K-Means' else 0.15
                x_vals = x_pos + desplazamiento
                
                # Para plt.errorbar, necesitamos la distancia desde la media hasta los límites
                yerr_lower = subset['Media'] - subset['IC_95_lower']
                yerr_upper = subset['IC_95_upper'] - subset['Media']
                
                # Graficamos el punto de la media y la barra del IC
                ax.errorbar(x_vals, subset['Media'], yerr=[yerr_lower, yerr_upper], 
                            fmt='o', color=colores[modelo], label=modelo,
                            capsize=4, markersize=6, elinewidth=2, alpha=0.8)
            
            # Títulos y etiquetas para mantener el gráfico limpio
            if i == 0:
                ax.set_title(f'Clúster {k}', fontsize=12, fontweight='bold')
            if k == 0:
                ax.set_ylabel(f'Coordenada {coord}', fontsize=11)
            if i == 1:
                ax.set_xticks(x_pos)
                ax.set_xticklabels(escenarios_unicos, rotation=45)
                ax.set_xlabel('Escenario')
                
            ax.grid(True, linestyle='--', alpha=0.5)

    # Leyenda única y título general
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=2, title='Algoritmo', bbox_to_anchor=(0.5, -0.05))
    
    plt.suptitle('Evolución de los Centroides e Incertidumbre (IC 95%) por Escenario', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()