import numpy as np
import pandas as pd
import json

def calculate_inertia(eigenvalues):
    """Calcule l'inertie totale à partir des valeurs propres"""
    return np.sum(eigenvalues)

def calculate_quality(eigenvalues, n_components=None):
    """Calcule la qualité de représentation pour un nombre donné de composantes"""
    total_inertia = calculate_inertia(eigenvalues)
    
    if n_components is None:
        return np.ones(len(eigenvalues)) * 100  # 100% avec toutes les composantes
    
    cumulative_quality = np.cumsum(eigenvalues) / total_inertia * 100
    return cumulative_quality[:n_components]

def pca(data, pca_type='normalized'):
    """
    Args:
        data: DataFrame ou numpy array contenant les données
        pca_type: Type de PCA ('normalized', 'homogeneous', 'heterogeneous')
    
    Returns:
        dict: Dictionnaire contenant tous les résultats de la PCA
    """
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
    
    # Définir la matrice métrique selon le type de PCA
    if pca_type == 'normalized':
        # PCA normée: Utilise la matrice diagonale des inverses des variances
        std = np.std(X_centered, axis=0, ddof=1)
        metric = np.diag(1 / std**2)
    elif pca_type == 'homogeneous':
        # PCA non normée homogène: Utilise la matrice identité
        metric = np.eye(X.shape[1])
    elif pca_type == 'heterogeneous':
        # PCA non normée hétérogène: Utilise une autre pondération
        std = np.std(X_centered, axis=0, ddof=1)
        metric = np.diag(1 / std**2)
    else:
        raise ValueError("Type de PCA non reconnu. Utilisez 'normalized', 'homogeneous' ou 'heterogeneous'")
    
    # Calcul des valeurs propres et vecteurs propres
    eigenvalues, eigenvectors = np.linalg.eigh(metric @ variance_matrix)
    
    # Tri par ordre décroissant
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Calcul de la qualité et du nombre d'axes à retenir (critère 80%)
    total_inertia = calculate_inertia(eigenvalues)
    cumulative_quality = np.cumsum(eigenvalues) / total_inertia * 100
    n_components = np.argmax(cumulative_quality >= 80) + 1 if np.any(cumulative_quality >= 80) else len(eigenvalues)
    
    # Calculer les composantes principales
    principal_components = X_centered @ eigenvectors
    
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
    
    # Préparer les résultats
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
        'correlations': correlations.tolist(),
        'contributions': {
            'variables': contributions_var.tolist(),
            'individuals': contributions_ind.tolist()
        },
        'cos2': {
            'individuals': cos2_ind.tolist()
        }
    }
    
    return results

def load_csv_data(file_content, sep=','):
    """Charge les données depuis un contenu CSV"""
    try:
        # Vérifier si le contenu est binaire ou déjà une chaîne
        if isinstance(file_content, bytes):
            # Essayer différents encodages
            encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
            content_str = None
            
            for encoding in encodings:
                try:
                    content_str = file_content.decode(encoding)
                    break
                except UnicodeDecodeError:
                    continue
            
            if content_str is None:
                # Si tous les encodages échouent, utiliser latin-1 qui est le plus permissif
                content_str = file_content.decode('latin-1', errors='replace')
        else:
            content_str = file_content
        
        # Nettoyer les données si nécessaire (BOM, etc.)
        if content_str.startswith('\ufeff'):
            content_str = content_str[1:]  # Supprimer le BOM
            
        # Essayer différents séparateurs si celui fourni ne fonctionne pas
        try:
            df = pd.read_csv(pd.io.common.StringIO(content_str), sep=sep)
        except Exception as e1:
            # Si le séparateur spécifié échoue, essayer d'autres séparateurs courants
            for alt_sep in [',', ';', '\t', '|']:
                if alt_sep != sep:
                    try:
                        df = pd.read_csv(pd.io.common.StringIO(content_str), sep=alt_sep)
                        # Si ça fonctionne, utiliser ce séparateur
                        return df
                    except:
                        continue
            # Si aucun séparateur ne fonctionne, relancer l'exception originale
            raise e1
            
        return df
    except Exception as e:
        raise ValueError(f"Erreur lors du chargement du fichier CSV: {str(e)}")

def prepare_response(results):
    """Convertit les résultats en format JSON compatible pour l'API"""
    # Convertir les arrays numpy en listes
    for key, value in results.items():
        if isinstance(value, np.ndarray):
            results[key] = value.tolist()
        elif isinstance(value, dict):
            results[key] = prepare_response(value)
    
    return results

def verify_properties(eigenvectors, metric, principal_components):
    """Vérifie les propriétés mentionnées dans la remarque"""
    # Base orthonormée
    for i in range(eigenvectors.shape[1]):
        for j in range(i+1, eigenvectors.shape[1]):
            assert np.abs(eigenvectors[:, i].T @ metric @ eigenvectors[:, j]) < 1e-10
            assert np.abs(eigenvectors[:, i].T @ metric @ eigenvectors[:, i] - 1) < 1e-10

    # Moyenne nulle des composantes
    assert np.all(np.abs(np.mean(principal_components, axis=0)) < 1e-10)

    # Variance des composantes = valeurs propres
    assert np.all(np.abs(np.var(principal_components, axis=0) - eigenvalues) < 1e-10)

    # Covariance nulle entre composantes
    for i in range(principal_components.shape[1]):
        for j in range(i+1, principal_components.shape[1]):
            assert np.abs(np.cov(principal_components[:, i], principal_components[:, j])[0, 1]) < 1e-10
