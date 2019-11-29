mat = loadmat('data/ex4data1.mat')
x_train, y_train = mat['X'], mat['y']
y_train = y_train.reshape(y_train.shape[0])

# replace all 10 to 0
y_train = np.where(y_train == 10, 0, y_train)

weights = loadmat('data/ex4weights.mat')
theta1 = weights['Theta1']
theta2 = weights['Theta2']

s_L = [400, 25, 10]

def sigmoid(z):
    return 1 / (1 + np.e ** (-z))

def insert_ones(x):
    if len(x.shape) == 1:
        return np.insert(x, 0, 1)
    return np.column_stack((np.ones(x.shape[0]), x))

def forward_propagation(x, thetas, cache=False):
    cur_activation = x.copy()
    activations = [cur_activation]

    for theta_i in thetas:
        temp_a = insert_ones(cur_activation)
        z_i = theta_i.dot(temp_a.T).T
        cur_activation = sigmoid(z_i)
        if cache:
            activations.append(cur_activation)

    return activations if cache else cur_activation

def unroll(weights):
    result = np.array([])

    for theta in weights:
        result = np.concatenate((result, theta.flatten()))

    return result

def roll(weights):
    weights = np.array(weights)
    thetas = []
    left = 0

    for i in range(len(s_L) - 1):
        x, y = s_L[i + 1], s_L[i] + 1
        right = x*y
        thetas.append(weights[left:left + right].reshape(x, y))
        left = right

    return thetas

def accuracy(predicted, y):
    correct_result_count = np.count_nonzero(predicted.argmax(axis=1) - y)
    return 1 - correct_result_count / y.shape[0]

weights = [theta1, theta2]
predicted = forward_propagation(x_train, weights)
accuracy(predicted, y_train)

# |class|    |class1|class2|
# |1    | => |1     |0     |
# |2    |    |0     |1     |
In [47]:
def one_hot(y, classes_count=10):
    y_extended = np.zeros((len(y), classes_count))

    for i, y_i in enumerate(y):
        y_extended[i][y_i] = 1

    return y_extended
In [48]:
y_one_hot = one_hot(y_train)

ONE = 1.0 + 1e-15

def cost_func(X, y, weights):
    total_cost = 0
    K = y.shape[1]
    hyp = forward_propagation(X, weights)
    for k in range(K):
        y_k, hyp_k = y[:, k], hyp[:, k]
        cost_trues =  y_k * np.log(hyp_k)
        cost_falses = (1 - y_k) * np.log(ONE - hyp_k)
        cost = cost_trues + cost_falses
        total_cost += cost
    return -total_cost.sum() / y.shape[0]

def cost_func_regularized(X, y, weights, reg_L=1):
    weights = roll(weights)
    reg = 0
    cost = cost_func(X, y, weights)

    for theta in weights:
        theta_R = theta[:, 1:]
        reg += (theta_R ** 2).sum()

    return cost + (reg_L / 2 / y.shape[0]) * reg

def activation_der(act):
    return act * (1 - act)