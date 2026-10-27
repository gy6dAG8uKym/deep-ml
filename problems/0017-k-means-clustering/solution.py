import numpy as np
def k_means_clustering(points: list[tuple[float, ...]], k: int, initial_centroids: list[tuple[float, ...]], max_iterations: int) -> list[tuple[float, ...]]:
	# Your code here
	X=np.array(points,dtype=float)
	centroids=np.array(initial_centroids,dtype=float)
	for it in range(max_iterations):
		distances=((X[:,None,:]-centroids[None,:,:])**2).sum(axis=2)
		labels=np.argmin(distances,axis=1)
		for j in range(k):
			centroids[j]=X[labels==j].mean(axis=0)
	final_centroids=[tuple(row) for row in centroids]
	return final_centroids