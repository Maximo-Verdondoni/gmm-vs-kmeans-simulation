import numpy as np
from sklearn.utils import resample
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

def ejecutar_bootstrap(X, B_iteraciones=500):
    """
    Aplica Bootstrap para estimar la inestabilidad de los centroides.
    """
    centroides_km = []
    centroides_gmm = []
    
    for b in range(B_iteraciones):
        # 1. Remuestreo con reemplazo
        X_boot = resample(X, random_state=b)
        
        # 2. Ajustar Modelos
        km = KMeans(n_clusters=3, n_init=10, random_state=b).fit(X_boot)
        gmm = GaussianMixture(n_components=3, covariance_type='full', random_state=b).fit(X_boot)
        
        # 3. Extraer centroides crudos
        c_km_raw = km.cluster_centers_
        c_gmm_raw = gmm.means_
        
        # 4. Solución al Label Switching (Ordenar manteniendo el par X,Y)
        # argosort() devuelve los índices que ordenarían la columna 0 (eje X).
        # Luego usamos esos índices para reordenar las filas completas.
        c_km = c_km_raw[c_km_raw[:, 0].argsort()]
        c_gmm = c_gmm_raw[c_gmm_raw[:, 0].argsort()]
        
        centroides_km.append(c_km)
        centroides_gmm.append(c_gmm)
        
    return np.array(centroides_km), np.array(centroides_gmm)