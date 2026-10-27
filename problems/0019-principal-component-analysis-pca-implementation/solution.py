import numpy as np

def pca(data: np.ndarray, k: int) -> np.ndarray:
    """
    Perform PCA and return the top k principal components.
    
    Args:
        data: Input array of shape (n_samples, n_features)
        k: Number of principal components to return
    
    Returns:
        Principal components of shape (n_features, k), rounded to 4 decimals.
        Each eigenvector's sign is fixed so its first non-zero element is positive.
    """
    # Your code here
    X=(data-data.mean(axis=0))/data.std(axis=0)
    C=X.T@X/(X.shape[0]-1)
    eigenvalues,eigenvectors=np.linalg.eigh(C)
    idx=np.argsort(eigenvalues)[::-1]
    components=eigenvectors[:,idx[:k]]
    for j in range(k):
        for i in range(components.shape[0]):
            if abs(components[i,j])>1e-10:
                if components[i,j]<0:
                    components[:,j]=-components[:,j]
                break
    return np.round(components,4)


    
