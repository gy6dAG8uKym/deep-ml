import numpy as np

def train_logreg(X: np.ndarray, y: np.ndarray, learning_rate: float, iterations: int) -> tuple[list[float], ...]:
	"""
	Gradient-descent training algorithm for logistic regression, optimizing parameters with Binary Cross Entropy loss.
	"""
	# Your code here
	X=np.c_[np.ones(X.shape[0]),X]
	theta=np.zeros(X.shape[1],dtype=float)
	predicted_y=1/(1+np.exp(-X@theta))
	#i use np.clip ??
	eps=1e-15
	predicted_y=np.clip(predicted_y,eps,1-eps)
	loss=-np.sum(y*np.log(predicted_y)+(1-y)*np.log(1-predicted_y))
	history=[0]*iterations
	for it in range(iterations):
		predicted_y=1/(1+np.exp(-X@theta))
		predicted_y=np.clip(predicted_y,eps,1-eps)
		loss=-np.sum(y*np.log(predicted_y)+(1-y)*np.log(1-predicted_y))
		history[it]=loss
		theta=theta-learning_rate*X.T@(predicted_y-y)
	
	return (theta,history)

