from scipy.io import loadmat

mat = loadmat('data/ex3data1.mat')
mat_y = mat['y']
x_train, y_train = mat['X'], mat_y.reshape(len(mat_y))

yval = mat['yval']
x_val, y_val = mat['Xval'], yval.reshape(len(yval))

ytest = mat['ytest']
x_test, y_test = mat['Xtest'], ytest.reshape(len(ytest))

plt.plot(x_train, y_train, 'o')
plt.xlabel('water change value')
plt.ylabel('water volume')
plt.show()


def h(x, theta):
    if len(x.shape) > 1 and x.shape[1] < theta.shape[0]:
        x = np.column_stack((np.ones(x.shape[0]), x))

    return x.dot(theta)

def cost_func(x, y, theta, l2_penalty_value=0.1):
    err = (h(x, theta) - y) ** 2

    theta_ = theta[1:]
    total_cost = err.sum() + l2_penalty_value * np.dot(theta_.T, theta_)
    return total_cost / 2 / x.shape[0]

THRESHOLD = 1e-7

def gradient_descent(x, y, max_iters_count=300000, a=0.001, l2_penalty_value=0.1, cost_func=cost_func):
    theta = np.zeros(x.shape[1])
    last_loss = cost_func(x, y, theta)
    logs = []

    for i in range(max_iters_count):
        diff = h(x, theta) - y
        gradient_first = np.dot(x.T[:1], diff)
        gradient_full = np.dot(x.T[1:], diff) + l2_penalty_value * theta[1:]
        gradient = np.insert(gradient_full, 0, gradient_first)
        gradient /= x.shape[0]
        gradient *= a
        theta -= gradient 

        curr_los = cost_func(x, y, theta)
        logs.append(curr_los)
        if abs(curr_los - last_loss) < THRESHOLD:
            break

        last_loss = curr_los

    return theta, logs

def normalize_features(x):
    N = x.shape[1]
    copy_x = x.copy()
    for i in range(N):
        feature = x[:, i]
        mean = np.mean(feature)
        delta = np.max(feature) - np.min(feature)            
        copy_x[:, i] -= mean
        copy_x[:, i] /= delta
    return copy_x

def fit(x, y, normalize=False, **kwargs):
    x = x.astype('float64') 
    y = y.astype('float64')

    if normalize:
        x = normalize_features(X)

    x = np.column_stack((np.ones(x.shape[0]), x))

    return gradient_descent(x, y, **kwargs)

def predict(x, theta):
    x_extended = np.insert(x, 0, 1)        
    return h(x_extended, theta)

theta, logs = fit(x_train, y_train, a=0.001, max_iters_count=1000000, l2_penalty_value=0)

xi = list(range(-50, 50))
line = [predict(np.array(i), theta) for i in xi]
plt.plot(x_train, y_train, 'o', xi, line, markersize=4)
plt.xlabel('water level change')
plt.ylabel('water volume')
plt.show()

def learning_curves(cost_func, x_train, y_train, x_val, y_val, max_axis=100, l2_penalty_value=0, **kwargs):
    N = len(y_train)
    train_err = np.zeros(N)
    val_err = np.zeros(N)

    for i in range(1, N):
        theta, logs = fit(x_train[0:i + 1, :], y_train[0:i + 1], l2_penalty_value=l2_penalty_value, **kwargs)
        train_err[i] = cost_func(x_train[0:i + 1, :], y_train[0:i + 1], theta, l2_penalty_value=l2_penalty_value)
        val_err[i] = cost_func(x_val, y_val, theta, l2_penalty_value=l2_penalty_value)

    plt.plot(range(2, N + 1), train_err[1:], c="r", linewidth=2)
    plt.plot(range(2, N + 1), val_err[1:], linewidth=2)
    plt.xlabel("number of training examples")
    plt.ylabel("error")
    plt.legend(["training", "validation"], loc="best")
    plt.axis([2, N, 0, max_axis])
    plt.show()

learning_curves(cost_func, x_train, y_train, X_val, y_val)

def make_polynom_of_features(x, degree):
    x = x.reshape(x.shape[0])
    x_res = np.array(x)

    for i in range(2, degree + 1):
        x_res = np.column_stack((x_res, x ** i))

    return x_res


x_train_poly = normalize_features(make_polynom_of_features(x_train, 8))
theta, _ = fit(x_train_poly, y_train, a=0.3, max_iters_count=500000, l2_penalty_value=0)