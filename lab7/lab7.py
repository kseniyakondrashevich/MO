mat = loadmat('data/ex7data1.mat')
X = mat['X']
X.shape

plt.scatter(X[:,0], X[:,1])

def cov_matrix(X):
    return np.dot(X.T, X) / X.shape[0]

def normalize_features(X):
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    
    X_norm = (X - mu)/sigma
    
    return X_norm, mu , sigma

from numpy.linalg import svd

def pca(X):
    sigma = cov_matrix(X)
    return svd(sigma)

X_norm, mu, std = normalize_features(X)
U, S, V = pca(X_norm)
U

mu = X.mean(axis=0)
projected_data = np.dot(X, U)
sigma = projected_data.std(axis=0).mean()

fig, ax = plt.subplots()
ax.plot(X[:, 0], X[:, 1], marker='o', linestyle="None", markersize=3)
for ind, axis in enumerate(U):
    start, end = mu, mu + (S[ind] + sigma) * axis
    ax.annotate(
        '', xy=end, xycoords='data',
        xytext=start, textcoords='data',
        arrowprops=dict(facecolor='red', width=2.0))
ax.set_aspect('equal')
ax.set_xlabel('X1')
ax.set_ylabel('X2')
plt.show()

U[:,0]

def make_projection(X, U, K):
    U_reduce = U[:, :K]
    return np.dot(X, U_reduce)

Z = make_projection(X_norm, U, 1)
Z.shape

def recover(Z, U, K=None):
    U_reduce = U[:, :K]
    return np.dot(Z, U_reduce.T)

X_rec  = recover(Z, U, 1)

plt.scatter(X_norm[:,0],X_norm[:,1],marker="o",label="Original",facecolors="none",edgecolors="b",s=15)
plt.scatter(X_rec[:,0],X_rec[:,1],marker="o",label="Approximation",facecolors="none",edgecolors="r",s=15)
plt.title("The Normalized and Projected Data after PCA")
plt.legend()