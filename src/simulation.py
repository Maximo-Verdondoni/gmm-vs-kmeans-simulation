import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score, homogeneity_score, completeness_score
from src.data_generation import generar_datos_escenario
from sklearn.metrics import silhouette_score

def ejecutar_montecarlo(M_replicas=200):
    """
    Ejecuta M réplicas de Montecarlo sobre los 5 escenarios y calcula el ARI.
    """
    resultados = []
    
    for escenario in [1, 2, 3, 4, 5]:
        for m in range(M_replicas):
            # 1. Generar nueva muestra
            X, y_real = generar_datos_escenario(escenario_id=escenario, random_state=escenario * 1000 + m)
            
            # 2. Ajustar K-Means
            kmeans = KMeans(n_clusters=3, random_state=m, n_init=10)
            y_pred_km = kmeans.fit_predict(X)
            ari_km = adjusted_rand_score(y_real, y_pred_km)
            
            # 3. Ajustar GMM
            gmm = GaussianMixture(n_components=3, covariance_type='full', 
                random_state=m, reg_covar=1e-6, max_iter=200)
            y_pred_gmm = gmm.fit_predict(X)
            ari_gmm = adjusted_rand_score(y_real, y_pred_gmm)
            
            # 4. Guardar registro
            resultados.append({
                'Escenario': escenario,
                'Replica': m,
                'Modelo': 'K-Means',
                'ARI': ari_km,
                'Silhouette': silhouette_score(X, y_pred_km),
                'Homogeneidad': homogeneity_score(y_real, y_pred_km),
                'Completitud': completeness_score(y_real, y_pred_km)
            })
            resultados.append({
                'Escenario': escenario,
                'Replica': m,
                'Modelo': 'GMM',
                'ARI': ari_gmm,
                'Silhouette': silhouette_score(X, y_pred_gmm),
                'Homogeneidad': homogeneity_score(y_real, y_pred_gmm),
                'Completitud': completeness_score(y_real, y_pred_gmm)
            })
            
    return pd.DataFrame(resultados)