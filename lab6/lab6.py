from scipy.io import loadmat

mat = loadmat('data/ex6data1.mat')
x = mat['X']

def init_centroids(X, K):
    m,n = X.shape[0], X.shape[1]
    centroids = np.zeros((K,n))
    
    for i in range(K):
        centroids[i] = X[np.random.randint(0 ,m+1),:]

    return centroids

def find_closes_clusters(X, centroids):
    K = centroids.shape[0]
    clusters = np.zeros(len(X), dtype=int)
    temp = np.zeros((centroids.shape[0],1))

    for i in range(X.shape[0]):
        for j in range(K):
            dist = X[i,:] - centroids[j,:]
            length = np.sum(dist**2)
            temp[j] = length

        clusters[i] = np.argmin(temp)

    return clusters

def update_centroids(X, clusters, K):
    new_centroids = np.zeros((K, X.shape[1]))
    for k in range(K):
        if len(X[clusters == k]) == 0:
            continue
        new_centroids[k] = X[clusters == k].mean(axis=0)

    return new_centroids

def k_means(X, centroids, clusters, K, num_iters):
    logs = [centroids]
    for i in range(num_iters):
        centroids = update_centroids(X, clusters, K)
        logs.append(centroids)
        clusters = find_closes_clusters(X, centroids)

    return clusters, logs


def plot_k_means(X, clusters, cenroids_moving_logs):
    """
    plots the data points with colors assigned to each centroid
    """
    m, n = X.shape[0], X.shape[1]
    plt.scatter(X[:, 0], X[:, 1], c=clusters)

    clusters_centroids = [[], [], [], [], [], []]
    for centroids in cenroids_moving_logs:
        for idx, cl_centroid in enumerate(centroids):
            clusters_centroids[idx * 2].append(cl_centroid[0])
            clusters_centroids[idx * 2 + 1].append(cl_centroid[1])
    
    plt.plot(*clusters_centroids, marker='x')
    plt.show()

K = 3
initial_centroids = init_centroids(x, K)
clusters = find_closes_clusters(x, initial_centroids)
result_clusters, cenroids_moving_logs = k_means(x, initial_centroids, clusters, 3, 50)
cenroids_moving_logs
plot_k_means(x, result_clusters, cenroids_moving_logs)