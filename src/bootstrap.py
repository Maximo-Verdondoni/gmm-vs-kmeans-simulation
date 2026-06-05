import numpy as np
import pandas as pd
from sklearn.utils import resample
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

def alinear_centroides(c_ref, c_nuevo):
    """
    Resuelve el problema de 'Label Switching' (intercambio de etiquetas) emparejando 
    los centroides de un modelo nuevo con los de un modelo de referencia.
    
    Parámetros:
    - c_ref: Array con las coordenadas de los centroides de referencia.
    - c_nuevo: Array con las coordenadas de los centroides recién calculados.
    
    Retorna:
    - c_nuevo ordenado para que sus filas coincidan lógicamente con c_ref.
    """
    # 1. Calcula la distancia Euclidiana entre todos los centroides nuevos y de referencia
    dist_matrix = cdist(c_ref, c_nuevo)
    
    # 2. Usa el algoritmo húngaro para encontrar la asignación óptima 1 a 1 
    # que minimice la distancia total entre las parejas de centroides.
    _, col_ind = linear_sum_assignment(dist_matrix)
    
    # 3. Devuelve los centroides nuevos reordenados según el índice óptimo
    return c_nuevo[col_ind]

def ejecutar_bootstrap(X, B_iteraciones=500):
    """
    Aplica el método Bootstrap para estimar la inestabilidad espacial de los 
    centroides de K-Means y GMM ante pequeñas variaciones en los datos.
    
    Parámetros:
    - X: Conjunto de datos original.
    - B_iteraciones: Número de remuestreos (por defecto 500).
    
    Retorna:
    - Dos arrays 3D (K-Means y GMM) con forma (B_iteraciones, K_clusters, 2_coordenadas)
    """
    centroides_km = []
    centroides_gmm = []
    
    # --- FASE 1: CREAR EL MODELO DE REFERENCIA ABSOLUTO ---
    # Ajustamos los modelos una vez con los datos originales reales
    km_ref = KMeans(n_clusters=3, n_init=10, random_state=0).fit(X)
    gmm_ref = GaussianMixture(n_components=3, covariance_type='full', 
                              random_state=0, reg_covar=1e-6, max_iter=200).fit(X)
    
    c_ref_km_raw = km_ref.cluster_centers_
    c_ref_gmm_raw = gmm_ref.means_

    # Anclaje espacial: Ordenamos la referencia basándonos en la coordenada X (columna 0).
    # Esto fuerza a que el 'Clúster 0' siempre sea el que está más a la izquierda en el gráfico.
    c_ref_km = c_ref_km_raw[c_ref_km_raw[:, 0].argsort()]
    c_ref_gmm = c_ref_gmm_raw[c_ref_gmm_raw[:, 0].argsort()]
    
    # --- FASE 2: BUCLE BOOTSTRAP ---
    for b in range(B_iteraciones):
        # 1. Remuestreo con reemplazo: Simula variaciones creando una nueva muestra del mismo tamaño
        X_boot = resample(X, random_state=b)
        
        # 2. Ajustar Modelos sobre la nueva muestra alterada
        km = KMeans(n_clusters=3, n_init=10, random_state=b).fit(X_boot)
        gmm = GaussianMixture(n_components=3, covariance_type='full', 
                              random_state=b, reg_covar=1e-6, max_iter=200).fit(X_boot)
        
        # 3. Extraer centroides crudos (que probablemente sufran de label switching)
        c_km_raw = km.cluster_centers_
        c_gmm_raw = gmm.means_
        
        # 4. Alinear usando el algoritmo húngaro y nuestra referencia ordenada
        c_km = alinear_centroides(c_ref_km, c_km_raw)
        c_gmm = alinear_centroides(c_ref_gmm, c_gmm_raw)
        
        # Guardar el historial de esta iteración
        centroides_km.append(c_km)
        centroides_gmm.append(c_gmm)
        
    return np.array(centroides_km), np.array(centroides_gmm)

def calcular_estadisticas_bootstrap(centroides_boot, nombre_modelo):
    """
    Toma el historial histórico de los centroides de todas las iteraciones Bootstrap 
    y calcula sus métricas de dispersión e incertidumbre.
    
    Parámetros:
    - centroides_boot: Array 3D con forma (B_iteraciones, K_clusters, 2).
    - nombre_modelo: String ('K-Means' o 'GMM') para etiquetar el resultado.
    
    Retorna:
    - Un DataFrame de Pandas listo para imprimir o graficar.
    """
    filas = []
    # Desempaquetamos las dimensiones: B (ej. 500), K (ej. 3 clústeres), _ (2 coordenadas)
    B, K, _ = centroides_boot.shape
    
    # Iteramos por cada clúster y por cada dimensión (X1, X2)
    for k in range(K):
        for coord_idx, coord_nombre in enumerate(['X1', 'X2']):
            
            # Extraemos los 500 valores que tomó esa coordenada específica en el proceso
            valores = centroides_boot[:, k, coord_idx]
            
            # Calculamos y registramos las estadísticas clave
            filas.append({
                'Modelo': nombre_modelo,
                'Clúster': k,
                'Coordenada': coord_nombre,
                'Media': np.mean(valores), # Posición promedio
                'Varianza': np.var(valores), # Magnitud de la oscilación
                'IC_95_lower': np.percentile(valores, 2.5),  # Límite inferior del Intervalo de Confianza 95%
                'IC_95_upper': np.percentile(valores, 97.5), # Límite superior del Intervalo de Confianza 95%
            })
            
    return pd.DataFrame(filas)