import numpy as np
import pandas as pd
from sklearn.utils import resample
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

def alinear_centroides(c_ref, c_nuevo):
    """
    Reordena las filas de c_nuevo para minimizar la distancia total
    con c_ref usando el algoritmo húngaro.
    """
    dist_matrix = cdist(c_ref, c_nuevo)
    _, col_ind = linear_sum_assignment(dist_matrix)
    return c_nuevo[col_ind]

def ejecutar_bootstrap(X, B_iteraciones=500):
    """
    Aplica Bootstrap para estimar la inestabilidad de los centroides.
    """
    centroides_km = []
    centroides_gmm = []
    # Calcular referencia una sola vez
    km_ref = KMeans(n_clusters=3, n_init=10, random_state=0).fit(X)
    gmm_ref = GaussianMixture(n_components=3, covariance_type='full', random_state=0, reg_covar=1e-6, max_iter=200).fit(X)
    c_ref_km_raw = km_ref.cluster_centers_
    c_ref_gmm_raw = gmm_ref.means_

    # ---------------------------------------------------------
    # LA SOLUCIÓN: Alinear las referencias entre sí ordenando por X
    # De esta forma, el Cluster 0 SIEMPRE es el de la izquierda para ambos.
    # ---------------------------------------------------------
    c_ref_km = c_ref_km_raw[c_ref_km_raw[:, 0].argsort()]
    c_ref_gmm = c_ref_gmm_raw[c_ref_gmm_raw[:, 0].argsort()]
    
    for b in range(B_iteraciones):
        # 1. Remuestreo con reemplazo
        X_boot = resample(X, random_state=b)
        
        # 2. Ajustar Modelos
        km = KMeans(n_clusters=3, n_init=10, random_state=b).fit(X_boot)
        gmm = GaussianMixture(n_components=3, covariance_type='full', random_state=b, reg_covar=1e-6, max_iter=200).fit(X_boot)
        
        # 3. Extraer centroides crudos
        c_km_raw = km.cluster_centers_
        c_gmm_raw = gmm.means_
        
        # 4. Solución al Label Switching (algoritmo húngaro)
        c_km = alinear_centroides(c_ref_km, c_km_raw)
        c_gmm = alinear_centroides(c_ref_gmm, c_gmm_raw)
        
        centroides_km.append(c_km)
        centroides_gmm.append(c_gmm)
        
    return np.array(centroides_km), np.array(centroides_gmm)

def calcular_estadisticas_bootstrap(centroides_boot, nombre_modelo):
    """
    centroides_boot: array (B, K, 2)
    Devuelve un DataFrame con varianza e IC 95% por clúster y coordenada.
    """
    filas = []
    B, K, _ = centroides_boot.shape
    for k in range(K):
        for coord_idx, coord_nombre in enumerate(['X1', 'X2']):
            valores = centroides_boot[:, k, coord_idx]
            filas.append({
                'Modelo': nombre_modelo,
                'Clúster': k,
                'Coordenada': coord_nombre,
                'Media': np.mean(valores),
                'Varianza': np.var(valores),
                'IC_95_lower': np.percentile(valores, 2.5),
                'IC_95_upper': np.percentile(valores, 97.5),
            })
    return pd.DataFrame(filas)