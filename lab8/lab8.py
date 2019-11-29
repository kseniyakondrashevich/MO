mat = loadmat('data/ex8data1.mat')
X = mat['X']
X_val = mat['Xval']
y_val = mat['yval']
y_val = y_val.reshape(y_val.shape[0])

X.shape

X_val.shape

plt.scatter(X[:,0],X[:,1], s=3)
plt.xlabel("Latency (ms)")
plt.ylabel("Throughput (mb/s)")

x1, x2 = X[:, 0], X[:, 1]

fig, axs = plt.subplots(1, 2, figsize=(20, 5))
axs[0].hist(x1, bins=200)
axs[0].set_xlabel("Latency (ms)")

axs[1].hist(x2, bins=200)
axs[1].set_xlabel("Throughput (mb/s)")

plt.show()

def estimate_gaussian(X):
    return X.mean(axis=0), X.std(axis=0)

mu, sigma = estimate_gaussian(X)

import scipy.stats as stats

def p(X):
    axis = int(len(X.shape) > 1)
    mu, sigma = estimate_gaussian(X)
    return stats.norm.pdf(X, mu, sigma).prod(axis=axis)

x, y = X[:, 0], X[:, 1]

h = 1.8
u = np.linspace(x.min() - h, x.max() + h, 50)
v = np.linspace(y.min() - h, y.max() + h, 50)
u_grid, v_grid = np.meshgrid(u, v)
Xnew = np.column_stack((u_grid.flatten(), v_grid.flatten()))
z = p(Xnew).reshape((len(u), len(v)))

fig, ax = plt.subplots(figsize=(7, 7))
ax.contour(u, v, z)
ax.scatter(x, y, s=6)

plt.xlabel("Latency (ms)")
plt.ylabel("Throughput (mb/s)")
plt.show()

def predict_anomalies(X, mu, sigma, eps):
    axis = int(len(X.shape) > 1)
    p = stats.norm.pdf(X, mu, sigma).prod(axis=axis)
    res = p < eps
    return res.astype(int) if axis else int(res)

def calc_eps(y_val, p_y_val, X_val, mu, sigma):
    best_eps = 0
    best_F1 = 0
    
    stepsize = (max(p_y_val) - min(p_y_val))/1000
    eps_range = np.arange(p_y_val.min(), p_y_val.max(), stepsize)
    for eps in eps_range:
        predictions = predict_anomalies(X_val, mu, sigma, eps)
        tp = np.sum(predictions[y_val==1]==1)
        fp = np.sum(predictions[y_val==0]==1)
        fn = np.sum(predictions[y_val==1]==0)

        # compute precision, recall and F1
        prec = tp/(tp+fp)
        rec = tp/(tp+fn)

        F1 = (2*prec*rec)/(prec+rec)

        if F1 > best_F1:
            best_F1 = F1
            best_eps = eps

    return best_eps, best_F1

p_y_val = p(X_val)
mu, sigma = estimate_gaussian(X_val)
eps, f1_score = calc_eps(y_val, p_y_val, X_val, mu, sigma)

eps