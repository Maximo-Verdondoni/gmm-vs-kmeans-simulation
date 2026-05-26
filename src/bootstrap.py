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
        
        # 3. Extraer centroides 
        # NOTA TÉCNICA: Se ordenan por el eje X para evitar el "Label Switching"
        # y que el cluster 0 siempre sea el de la izquierda.
        c_km = np.sort(km.cluster_centers_, axis=0) 
        c_gmm = np.sort(gmm.means_, axis=0)
        
        centroides_km.append(c_km)
        centroides_gmm.append(c_gmm)
        
    return np.array(centroides_km), np.array(centroides_gmm)