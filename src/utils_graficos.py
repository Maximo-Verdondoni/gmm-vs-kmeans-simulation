import matplotlib.pyplot as plt
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

def plot_gmm_soft_boundaries(X, y_pred, probs, titulo="Fronteras Difusas y Certeza en GMM"):
    """
    Grafica los puntos coloreados por el clúster asignado por GMM, aplicando
    una opacidad (alpha) proporcional a la certeza/probabilidad de la asignación.
    
    Parámetros:
    -----------
    X : np.array de forma (N, 2)
        Coordenadas de los puntos.
    y_pred : np.array de forma (N,)
        Etiquetas predichas por el modelo (el clúster con mayor probabilidad).
    probs : np.array de forma (N, K)
        Matriz de probabilidades obtenida con gmm.predict_proba(X).
    """
    plt.figure(figsize=(9, 7))
    
    # 1. Extraer la probabilidad del clúster asignado para cada punto (la certeza)
    certeza = np.max(probs, axis=1)
    
    # 2. Normalizar la certeza para el canal alfa (opacidad)
    # Si K=3, la probabilidad mínima de ganar es 1/3 (0.33). 
    # Al normalizar de 0 a 1, exageramos visualmente la pérdida de certeza en las fronteras.
    K = probs.shape[1]
    prob_min = 1.0 / K
    # Evitamos división por cero si la certeza fuera menor al mínimo teórico
    alpha_scaled = (certeza - prob_min) / (1.0 - prob_min)
    alpha_scaled = np.clip(alpha_scaled, 0.05, 1.0) # Asegurar rango [0.05, 1.0] para que no desaparezcan del todo
    
    # 3. Mapear colores de seaborn manualmente para poder aplicar alphas individuales
    unique_labels = np.unique(y_pred)
    base_palette = sns.color_palette("deep", len(unique_labels))
    
    # 4. Graficar punto por punto (o en bucle por nivel de opacidad) para aplicar alphas individuales
    # Usamos un truco eficiente con matplotlib: pasar un array de colores RGBA
    rgba_colors = np.zeros((X.shape[0], 4))
    for i, label in enumerate(y_pred):
        # Color base asignado al clúster (R, G, B)
        rgba_colors[i, :3] = base_palette[label]
        # Opacidad según su certeza calculada (A)
        rgba_colors[i, 3] = alpha_scaled[i]
        
    # Graficar usando la matriz RGBA personalizada
    scatter = plt.scatter(x=X[:, 0], y=X[:, 1], c=rgba_colors, s=50, edgecolor="none")
    
    # 5. Estética y detalles del reporte
    plt.title(titulo, fontsize=14, pad=15, fontweight='bold')
    plt.xlabel("X1")
    plt.ylabel("X2")
    
    # Crear una leyenda manual elegante para los clústeres
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=base_palette[k], edgecolor='none', label=f'Clúster {k}') 
                       for k in unique_labels]
    plt.legend(handles=legend_elements, title="Clúster Asignado", loc="upper right")
    
    # Añadir barra de color lateral que indique qué significa la opacidad
    sm = plt.cm.ScalarMappable(cmap=plt.cm.Greys, norm=plt.Normalize(vmin=prob_min, vmax=1.0))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=plt.gca(), fraction=0.046, pad=0.04)
    cbar.set_label('Nivel de Certeza P(Clúster | X)', fontsize=11, labelpad=10)
    
    plt.tight_layout()
    plt.show()

# Ejemplo de como ejecutarlo en un notebook:
    # from sklearn.mixture import GaussianMixture

    # # 1. Ajustar el modelo GMM
    # gmm = GaussianMixture(n_components=3, covariance_type='full', random_state=42)
    # gmm.fit(X)

    # # 2. Predecir etiquetas duras y obtener probabilidades blandas
    # y_pred_gmm = gmm.predict(X)
    # probs_gmm = gmm.predict_proba(X) # <--- AQUÍ está el secreto

    # # 3. Invocar tu nueva función visual profesional
    # plot_gmm_soft_boundaries(
    #     X=X, 
    #     y_pred=y_pred_gmm, 
    #     probs=probs_gmm, 
    #     titulo="Escenario 4: Degradación de Certeza en Zonas de Superposición Difusa"
    # )