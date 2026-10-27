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
    L=list(zip(eigenvalues,eigenvectors.T))
    L.sort(key=lambda x:x[0],reverse=True)
    L=[L[i][1] for i in range(k)]
    L=np.array(L).T
    for j in range(k):
        for i in range(L.shape[0]):
            if abs(L[i,j])>1e-10:
                if L[i,j]<0:
                    L[:,j]=-L[:,j]
                break
    return np.round(L,4)


    
