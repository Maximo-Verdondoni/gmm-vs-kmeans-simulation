import numpy as np

def generar_datos_escenario(escenario_id, N=500, random_state=None):
    """
    Genera datos sintéticos bidimensionales basados en mezclas gaussianas 
    para evaluar la robustez de algoritmos de clustering.
    
    Parámetros:
    -----------
    escenario_id : int
        Identificador del escenario a simular (1 al 4).
    N : int, opcional (default=500)
        Número total de observaciones a generar.
    random_state : int, opcional (default=None)
        Semilla para garantizar la reproducibilidad de los datos generados.
        
    Retorna:
    --------
    X : numpy.ndarray de forma (N, 2)
        Matriz de características continuas (coordenadas espaciales).
    y : numpy.ndarray de forma (N,)
        Vector de etiquetas reales (clúster latente de origen).
    """
    if random_state is not None:
        np.random.seed(random_state)
        
    # 1. Definición de la distribución latente
    # Probabilidades a priori (pi) para K=3 clústeres.
    pi = [0.33, 0.33, 0.34]
    
    # Sorteo de la pertenencia de las N observaciones a los clústeres latentes.
    y = np.random.choice([0, 1, 2], size=N, p=pi)
    
    # Inicialización de la matriz de características X en el espacio R^2
    X = np.zeros((N, 2))
    
    # 2. Configuración de parámetros poblacionales (mu y cov) según el escenario
    
    if escenario_id == 1:
        # ESCENARIO 1: Estructura Isotrópica Perfecta (Línea de Base)
        # - Medias: Bien separadas.
        # - Covarianzas: Esféricas y de varianza idéntica.
        mu = {0: [0, 0], 1: [5, 5], 2: [10, 0]}
        cov = {
            0: [[1.0, 0.0], [0.0, 1.0]],
            1: [[1.0, 0.0], [0.0, 1.0]],
            2: [[1.0, 0.0], [0.0, 1.0]]
        }
        
    elif escenario_id == 2:
        # ESCENARIO 2: Varianzas Altamente Desiguales (Heterocedasticidad)
        # - Medias: Bien separadas (igual que el escenario 1).
        # - Covarianzas: Esféricas, pero con escalares de dispersión muy distintos.
        mu = {0: [0, 0], 1: [5, 5], 2: [10, 0]}
        cov = {
            0: [[0.5, 0.0], [0.0, 0.5]], # Clúster muy denso/compacto
            1: [[2.0, 0.0], [0.0, 2.0]], # Clúster de dispersión media
            2: [[6.0, 0.0], [0.0, 6.0]]  # Clúster muy disperso
        }
        
    elif escenario_id == 3:
        # ESCENARIO 3: Geometría Anisotrópica (Estiramiento y Orientación)
        # - Medias: Bien separadas (igual que el escenario 1).
        # - Covarianzas: No diagonales. Se introducen correlaciones entre X1 y X2
        #   para crear elipses con diferentes ángulos.
        mu = {0: [0, 0], 1: [5, 5], 2: [10, 0]}
        cov = {
            # Correlación positiva (elipse inclinada hacia la derecha)
            0: [[2.0, 1.5], [1.5, 2.0]], 
            # Correlación negativa (elipse inclinada hacia la izquierda)
            1: [[2.0, -1.5], [-1.5, 2.0]], 
            # Varianza desigual en los ejes (elipse estirada horizontalmente sin rotación)
            2: [[3.0, 0.0], [0.0, 0.5]]  
        }
        
    elif escenario_id == 4:
        # ESCENARIO 4: Geometría Anisotrópica + Superposición Poblacional
        # - Medias: Acercadas drásticamente para forzar el solapamiento.
        # - Covarianzas: Las mismas elipses del escenario 3.
        mu = {
            0: [-2, 0], 
            1: [2, 2], # Acercamos el centroide 1 al origen
            2: [5, -1]  # Acercamos el centroide 2 al origen
        }
        cov = {
            0: [[2.0, 1.5], [1.5, 2.0]],
            1: [[2.0, -1.5], [-1.5, 2.0]],
            2: [[3.0, 0.0], [0.0, 0.5]]
        }

    elif escenario_id == 5:
        # ESCENARIO 4: Geometría Anisotrópica + Superposición Poblacional extrema
        # - Medias: Acercadas drásticamente para forzar el solapamiento.
        # - Covarianzas: Las mismas elipses del escenario 3.
        mu = {
            0: [0, 0], 
            1: [1, 1], # Acercamos el centroide 1 al origen
            2: [3, 0]  # Acercamos el centroide 2 al origen
        }
        cov = {
            0: [[2.0, 1.5], [1.5, 2.0]],
            1: [[2.0, -1.5], [-1.5, 2.0]],
            2: [[3.0, 0.0], [0.0, 0.5]]
        }
        
    else:
        raise ValueError(f"Error: El escenario {escenario_id} no está definido. Elija del 1 al 5.")

    # 3. Generación de las coordenadas espaciales
    # Muestreamos iterativamente de la distribución Gaussiana multivariada 
    # correspondiente al clúster asignado previamente.
    for i in range(N):
        k = y[i]
        X[i] = np.random.multivariate_normal(mean=mu[k], cov=cov[k])
        
    return X, y