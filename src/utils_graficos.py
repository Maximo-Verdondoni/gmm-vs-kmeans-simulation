import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Ellipse
from scipy.stats import multivariate_normal
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import numpy as np

#Librerias para arreglar label switching
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment

# Configuración global para que los gráficos se vean unificados y profesionales
sns.set_theme(style="whitegrid", palette="muted")

def alinear_etiquetas(y_real, y_pred):
    """
    Usa el Algoritmo Húngaro para reasignar los números de los clústeres predichos 
    de modo que los colores coincidan visualmente con las etiquetas reales.
    """
    # 1. Calculamos la matriz de confusión
    cm = confusion_matrix(y_real, y_pred)
    
    # 2. El algoritmo húngaro minimiza costos, así que le pasamos la matriz 
    # en negativo para que busque MAXIMIZAR las coincidencias.
    row_ind, col_ind = linear_sum_assignment(-cm)
    
    # 3. Armamos un diccionario que dice "La etiqueta predicha X ahora es la etiqueta real Y"
    mapeo = {col_ind[i]: i for i in range(len(col_ind))}
    
    # 4. Reemplazamos los valores en el array original
    y_pred_alineado = np.vectorize(mapeo.get)(y_pred)
    
    return y_pred_alineado

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
    y_km, y_gmm: Etiquetas predichas y alineadas a las etiquetas reales
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

def plot_bootstrap_centroids_bootstrap(X, centroides_km, centroides_gmm, escenario_id, titulo="Estabilidad de Centroides (Bootstrap)"):
    """
    Grafica la dispersión de los centroides obtenidos por Bootstrap para K-Means y GMM,
    superponiendo elipses de confianza del 95% e indicando el centroide real (μ) de fondo.
    """
    COLORES = ["#1359b6", "#db9d15", "#0e7905"]  # Uno por clúster
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharex=True, sharey=True)

    # Definimos los parámetros poblacionales reales (mu) correspondientes a cada escenario
    mu_reales = {
        1: {0: [0, 0], 1: [5, 5], 2: [10, 0]},
        2: {0: [0, 0], 1: [5, 5], 2: [10, 0]},
        3: {0: [0, 0], 1: [5, 5], 2: [10, 0]},
        4: {0: [-2, 0], 1: [2, 2], 2: [5, -1]},
        5: {0: [0, 0], 1: [1, 1], 2: [3, 0]}
    }

    for ax, c_boot, nombre in zip(axes, [centroides_km, centroides_gmm], ['K-Means', 'GMM']):
        # Datos de fondo atenuados
        ax.scatter(X[:, 0], X[:, 1], c='lightgray', s=10, alpha=0.3, label='Datos')
        
        for k in range(3):
            # Extraemos las coordenadas (X, Y) de todos los bootstraps para el clúster k
            datos_cluster = c_boot[:, k, :]
            
            # 1. Puntos semi-transparentes de fondo (muy tenues)
            ax.scatter(datos_cluster[:, 0], datos_cluster[:, 1],
                       c=COLORES[k], s=15, alpha=0.12, edgecolors='none')
            
            # 2. Elipse de confianza del 95% (n_std=2.0)
            plot_elipse_confianza(ax, datos_cluster, color=COLORES[k], n_std=2.0)
            
            # Graficar el centroide real (μ)
            mu_val = mu_reales[int(escenario_id)][k]
            # Usamos una estrella grande destacada
            ax.scatter(mu_val[0], mu_val[1], color='black', marker='.', s=100, 
                       edgecolor='white', linewidth=1.2, zorder=10, 
                       label='μ Real' if k == 0 else "") # Agrega una sola etiqueta a la leyenda
            
            # Dummy scatter invisible solo para que la leyenda se vea limpia
            ax.scatter([], [], c=COLORES[k], label=f'Clúster {k}')

        ax.set_title(f"Bootstrap {nombre}", fontsize=13)
        ax.legend(markerscale=1.2, loc='best')

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

def plot_gmm_3d_mountains(X, gmm, titulo="Densidad 3D y Ejes de Componentes Principales"):
    """
    Grafica la Función de Densidad de Probabilidad (PDF) de cada clúster 
    como una superficie 3D, y proyecta en la base (Z=0) las medias (mu) 
    junto con los vectores de varianza.
    """
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    # 1. Definir los límites y construir la malla 2D
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    
    x, y = np.mgrid[x_min:x_max:.05, y_min:y_max:.05]
    pos = np.empty(x.shape + (2,))
    pos[:, :, 0] = x
    pos[:, :, 1] = y
    
    K = gmm.n_components
    base_palette = sns.color_palette("deep", K)
    
    max_z = 0 
    
    # 2. Iterar sobre cada clúster
    for k in range(K):
        mu = gmm.means_[k]
        cov = gmm.covariances_[k]
        weight = gmm.weights_[k]
        
        # Calcular PDF
        rv = multivariate_normal(mu, cov)
        z = weight * rv.pdf(pos)
        
        if z.max() > max_z:
            max_z = z.max()
            
        # =========================================================
        # 3. LA SOLUCIÓN AL PLANO VERDE (Máscara NaN)
        # =========================================================
        z_surface = z.copy()
        # Todo valor que sea "piso" (densidad casi nula) lo volvemos NaN 
        # para que el motor 3D de Matplotlib sea incapaz de dibujarlo.
        z_surface[z_surface < 1e-4] = np.nan 
        
        color_rgb = base_palette[k]
        colors = [(color_rgb[0], color_rgb[1], color_rgb[2], 0.2), # Un poco más transparente en la base
                  (color_rgb[0], color_rgb[1], color_rgb[2], 0.9)]
        cmap_custom = LinearSegmentedColormap.from_list(f'custom_cmap_{k}', colors)
        
        # Quitamos el alpha=0.8 de acá para que el colormap haga su trabajo
        ax.plot_surface(x, y, z_surface, cmap=cmap_custom, linewidth=0, antialiased=True)
        
        # El contorno lo seguimos haciendo con el 'z' original porque no sufre este problema
        ax.contour(x, y, z, zdir='z', offset=0, colors=[color_rgb], alpha=0.5, linewidths=2)

        # =========================================================
        # 4. Proyección en el piso (Z=0) de Mu y Vectores
        # =========================================================
        ax.scatter(mu[0], mu[1], 0, color=color_rgb, marker='.', s=120, edgecolor='black', zorder=10, linewidth=2)
        
        autovalores, autovectores = np.linalg.eigh(cov)
        
        for i in range(2):
            vector_desviacion = autovectores[:, i] * np.sqrt(autovalores[i]) * 2.0
            x_line = [mu[0], mu[0] + vector_desviacion[0]]
            y_line = [mu[1], mu[1] + vector_desviacion[1]]
            z_line = [0, 0]
            ax.plot(x_line, y_line, z_line, color='black', linewidth=2, linestyle='-', zorder=11)

    # 5. Configuración final
    ax.set_title(titulo, fontsize=14, pad=20, fontweight='bold')
    ax.set_xlabel('Eje X1', labelpad=10)
    ax.set_ylabel('Eje X2', labelpad=10)
    ax.set_zlabel('Densidad de Probabilidad $p(x)$', labelpad=10)
    
    ax.set_zlim(0, max_z * 1.1)
    ax.view_init(elev=35, azim=-45)
    
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
    a lo largo de los escenarios, indicando el parámetro poblacional real
    para visualizar el sesgo (bias).
    """
    #Ordenamos el DF globalmente para evitar desfasajes
    df_stats = df_stats.sort_values('Escenario').reset_index(drop=True)
    
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(16, 9), sharex=True)
    coords = ['X1', 'X2']
    modelos = ['K-Means', 'GMM']
    colores = {'K-Means': '#e74c3c', 'GMM': '#3498db'}

    # 1. Definimos los parámetros poblacionales (mu reales) directamente en la función
    mu_reales = {
        1: {0: [0, 0], 1: [5, 5], 2: [10, 0]},
        2: {0: [0, 0], 1: [5, 5], 2: [10, 0]},
        3: {0: [0, 0], 1: [5, 5], 2: [10, 0]},
        4: {0: [-2, 0], 1: [2, 2], 2: [5, -1]},
        5: {0: [0, 0], 1: [1, 1], 2: [3, 0]}
    }

    escenarios_unicos = df_stats['Escenario'].unique()
    x_pos = np.arange(len(escenarios_unicos))
    
    leyenda_mu_agregada = False # Bandera para no repetir la leyenda de la estrella

    for i, coord in enumerate(coords):
        for k in range(3): # Clústeres 0, 1, 2
            ax = axes[i, k]
            
            # Graficar los estimadores (K-Means y GMM)
            for modelo in modelos:
                mask = (df_stats['Coordenada'] == coord) & (df_stats['Clúster'] == k) & (df_stats['Modelo'] == modelo)
                subset = df_stats[mask]
                
                if subset.empty:
                    continue
                
                # Jitter: Desplazamos un poco para que no se superpongan
                desplazamiento = -0.15 if modelo == 'K-Means' else 0.15
                x_vals = x_pos + desplazamiento
                
                yerr_lower = subset['Media'] - subset['IC_95_lower']
                yerr_upper = subset['IC_95_upper'] - subset['Media']
                
                # Etiqueta para la leyenda global (solo la agregamos en el primer subplot para no duplicar)
                etiqueta_modelo = modelo if (i == 0 and k == 0) else ""
                
                ax.errorbar(x_vals, subset['Media'], yerr=[yerr_lower, yerr_upper], 
                            fmt='o', color=colores[modelo], label=etiqueta_modelo,
                            capsize=4, markersize=6, elinewidth=2, alpha=0.8)

            #Graficar la estrella del mu real
            for j, esc in enumerate(escenarios_unicos):
                # Extraemos el valor real (0 para X1, 1 para X2)
                coord_idx = 0 if coord == 'X1' else 1
                val_real = mu_reales[int(esc)][k][coord_idx]
                
                etiqueta_real = 'μ Poblacional (Valor Real)' if not leyenda_mu_agregada else ""
                ax.scatter(j, val_real, color='black', marker='.', s=180, zorder=5, label=etiqueta_real)
                leyenda_mu_agregada = True # Ya la agregamos una vez, no la repetimos

            # Títulos y estética
            if i == 0:
                ax.set_title(f'Clúster {k}', fontsize=13, fontweight='bold')
            if k == 0:
                ax.set_ylabel(f'Coordenada {coord}', fontsize=12)
            if i == 1:
                ax.set_xticks(x_pos)
                ax.set_xticklabels([f"Esc. {int(e)}" for e in escenarios_unicos], fontsize=11)
            
            ax.grid(True, linestyle='--', alpha=0.5)

    # Configurar una única leyenda global arriba del gráfico
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=3, fontsize=12, frameon=False)
    
    plt.tight_layout()
    plt.show()