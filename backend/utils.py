import numpy as np
import pandas as pd
import json
#from sklearn.cluster import KMeans
#from sklearn.preprocessing import StandardScaler

def calculate_inertia(eigenvalues):
    return np.sum(eigenvalues)

def calculate_quality(eigenvalues, n_components=None):
    total_inertia = calculate_inertia(eigenvalues)
    
    if n_components is None:
        return np.ones(len(eigenvalues)) * 100
    
    cumulative_quality = np.cumsum(eigenvalues) / total_inertia * 100
    return cumulative_quality[:n_components]

def pca(data, pca_type='normalized'):
  
    # Convertir en numpy array si nécessaire
    if isinstance(data, pd.DataFrame):
        variables = data.columns.tolist()
        individuals = data.index.tolist()
        X = data.values
    else:
        X = np.array(data)
        variables = [f'V{i+1}' for i in range(X.shape[1])]
        individuals = [f'P{i+1}' for i in range(X.shape[0])]
    
    # Centrer les données
    X_centered = X - np.mean(X, axis=0)
    
    # Calculer la matrice de variance/covariance
    n = X.shape[0]
    variance_matrix = (1 / n) * (X_centered.T @ X_centered)
    
    # La  métrique 
    if pca_type in ['normalized', 'normalized_kaiser']:
        # PCA normée: Utilise la matrice diagonale des inverses des variances
        std = np.std(X_centered, axis=0, ddof=1)
        metric = np.diag(1 / std**2)
    elif pca_type in ['homogeneous', 'homogeneous_kaiser']:
        # PCA non normée homogène: Utilise la matrice identité
        metric = np.eye(X.shape[1])
    elif pca_type in ['heterogeneous', 'heterogeneous_kaiser']:
        # PCA non normée hétérogène: Utilise une autre pondération
        std = np.std(X_centered, axis=0, ddof=1)
        metric = np.diag(1 / std**2)
    else:
        raise ValueError("Type de PCA non reconnu. Utilisez 'normalized', 'normalized_kaiser', 'homogeneous', 'homogeneous_kaiser', 'heterogeneous' ou 'heterogeneous_kaiser'")
    
    # Calcul des valeurs propres et vecteurs propres
    eigenvalues, eigenvectors = np.linalg.eigh(metric @ variance_matrix)
    
    # Tri par ordre décroissant
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Calcul de la qualité et du nombre d'axes à retenir
    total_inertia = calculate_inertia(eigenvalues)
    cumulative_quality = np.cumsum(eigenvalues) / total_inertia * 100
    
    # 80% ou règle de Kaiser (valeurs propres > 1)
    if pca_type in ['normalized_kaiser', 'homogeneous_kaiser', 'heterogeneous_kaiser']:
        n_components = np.sum(eigenvalues > 1)
        n_components = max(1, n_components) if n_components == 0 else n_components
    else:
        n_components = np.argmax(cumulative_quality >= 80) + 1 if np.any(cumulative_quality >= 80) else len(eigenvalues)
    
    # Calculer les CK
    principal_components = X_centered @ eigenvectors
    
    # Calcul les statistiques des CK
    pc_means = np.mean(principal_components, axis=0)
    pc_variance = np.var(principal_components, axis=0)
    pc_covariance = np.cov(principal_components, rowvar=False)
    
    # Corrélations entre variables et composantes
    correlations = np.zeros((X.shape[1], len(eigenvalues)))
    for i in range(X.shape[1]):
        for j in range(len(eigenvalues)):
            sigma_x = np.std(X_centered[:, i])
            correlations[i, j] = np.sum(X_centered[:, i] * principal_components[:, j]) / (n * sigma_x * np.sqrt(eigenvalues[j]))
    
    # Contributions des variables aux axes
    contributions_var = np.zeros((X.shape[1], len(eigenvalues)))
    for i in range(X.shape[1]):
        for j in range(len(eigenvalues)):
            contributions_var[i, j] = (eigenvectors[i, j] ** 2) * eigenvalues[j] / total_inertia * 100
    
    # Contributions des individus aux axes
    contributions_ind = np.zeros((X.shape[0], len(eigenvalues)))
    for i in range(X.shape[0]):
        for j in range(len(eigenvalues)):
            contributions_ind[i, j] = (principal_components[i, j] ** 2) / (eigenvalues[j])
    
    # Qualité de représentation des individus
    cos2_ind = np.zeros((X.shape[0], len(eigenvalues)))
    for i in range(X.shape[0]):
        for j in range(len(eigenvalues)):
            cos2_ind[i, j] = (principal_components[i, j] ** 2) / np.sum(X_centered[i] ** 2)
    
    # Classification des variables
    variable_classifications = {}
    for i, var in enumerate(variables):
        # Une variable est bien représentée si sa corrélation au carré (cos²) est > 0.5
        correlations_squared = correlations[i, :] ** 2
        significant_axes = [j+1 for j, corr in enumerate(correlations_squared) if corr >= 0.5]
        
        # Interprétation du sens physique
        physical_meaning = []
        for axis in significant_axes:
            correlation = correlations[i, axis-1]
            contribution = contributions_var[i, axis-1]
            
            interpretation = {
                'axis': axis,
                'correlation': float(correlation),
                'contribution': float(contribution),
                'meaning': 'positif' if correlation > 0 else 'négatif',
                'quality': 'forte' if abs(correlation) > 0.7 else 'moyenne'
            }
            physical_meaning.append(interpretation)
        
        variable_classifications[var] = {
            'significant_axes': significant_axes,
            'physical_meaning': physical_meaning
        }

    # résultats
    results = {
        'pca_type': pca_type,
        'n_individuals': X.shape[0],
        'n_variables': X.shape[1],
        'variables': variables,
        'individuals': individuals,
        'data_original': X.tolist(),
        'data_centered': X_centered.tolist(),
        'variance_matrix': variance_matrix.tolist(),
        'metric': metric.tolist(),
        'eigenvalues': eigenvalues.tolist(),
        'eigenvectors': eigenvectors.tolist(),
        'inertia': {
            'total': float(total_inertia),
            'explained_by_axis': eigenvalues.tolist(),
            'explained_percent_by_axis': (eigenvalues / total_inertia * 100).tolist(),
            'cumulative_percent': cumulative_quality.tolist()
        },
        'n_significant_components': int(n_components),
        'principal_components': principal_components.tolist(),
        'pc_statistics': {
            'means': pc_means.tolist(),
            'variance': pc_variance.tolist(),
            'covariance': pc_covariance.tolist()
        },
        'correlations': correlations.tolist(),
        'contributions': {
            'variables': contributions_var.tolist(),
            'individuals': contributions_ind.tolist()
        },
        'cos2': {
            'individuals': cos2_ind.tolist()
        },
        'variable_classifications': variable_classifications

    }
    """
    CLUSTERING WITH K-MEANS (mba3d hh)
    """
    
    """
    pc_for_clustering = principal_components[:, :n_components]
    
    max_clusters = min(10, X.shape[0]-1)  # Maximum de 10 clusters ou n-1 si moins d'observations
    inertias = []
    for k in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(pc_for_clustering)
        inertias.append(kmeans.inertia_)
    
    inertia_diffs = np.diff(inertias)
    inertia_diffs_2 = np.diff(inertia_diffs)
    optimal_n_clusters = np.argmin(inertia_diffs_2) + 2  # +2 car on commence à 1 et on a fait deux diff
    
    final_kmeans = KMeans(n_clusters=optimal_n_clusters, random_state=42)
    clusters = final_kmeans.fit_predict(pc_for_clustering)
    
    cluster_centers = final_kmeans.cluster_centers_
    
    results['clustering'] = {
        'n_clusters': int(optimal_n_clusters),
        'cluster_labels': clusters.tolist(),
        'cluster_centers': cluster_centers.tolist(),
        'inertia_curve': inertias,
        'cluster_sizes': [int(np.sum(clusters == i)) for i in range(optimal_n_clusters)],
        'silhouette_scores': None 
    }
    """
    return results


def prepare_response(results):
    for key, value in results.items():
        if isinstance(value, np.ndarray):
            results[key] = value.tolist()
        elif isinstance(value, dict):
            results[key] = prepare_response(value)
    
    return results