import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

df = pd.read_csv('data/ex2data1.txt', header=None, names=['first_exam', 'second_exam', 'accepted'])
x_train, y_train = df.filter(['first_exam', 'second_exam']), df['accepted']

df_accepted = df[df['accepted'] == 1]
df_not_accepted = df[df['accepted'] == 0]
fig, ax = plt.subplots()
ax.scatter(df_accepted['first_exam'], df_accepted['second_exam'], marker='o', label='Accepted', s=20)
ax.scatter(df_not_accepted['first_exam'], df_not_accepted['second_exam'], marker='o', c='r', label='Not accepted', s=20)
ax.set_xlabel('1st exam')
ax.set_ylabel('2nd exam')
plt.show()

THRESHOLD = 1e-8

def sigmoid(z):
    return 1 / (1 + np.e ** (-z))

def h(x, theta):
    return sigmoid(x.dot(theta))
    
def cost_func_vectorized(x, y, theta, **kwargs):
    h_theta = h(x, theta)
    cost_1 = y * np.log(h_theta)
    cost_0 = (1 - y) * np.log(1 - h_theta)
    return -np.mean(cost_1 + cost_0)

def cost_func_deriative(x, y, theta, **kwargs):
    h_v = h(x, theta)
    gradient = np.dot(x.T, h(x, theta) - y)
    gradient /= x.shape[0]
    return gradient

def gradient_descent_vectorized(x, y, theta, a=1, max_iter_count=30000,
                                cost_func=cost_func_vectorized, cost_func_der=cost_func_deriative,
                                **kwargs):
    logs = []
    last_loss = cost_func(x, y, theta)
    for iter_num in range(max_iter_count):
        gradient = cost_func_der(x, y, theta, **kwargs)
        theta -= gradient * a
        curr_loss = cost_func(x, y, theta, **kwargs)

        logs.append([iter_num, curr_loss])

        if abs(curr_loss - last_loss) < THRESHOLD:
            break

        last_loss = curr_loss

    return theta, logs

def fit(x_train, y_train, minimization_func=gradient_descent_vectorized, regularized=False,
        cost_func=cost_func_vectorized, cost_func_der=cost_func_deriative, **kwargs):

    x = getattr(x_train, 'values', x_train).astype('float64')
    y = getattr(y_train, 'values', y_train).astype('float64')

    if not regularized:
        x = np.column_stack((np.ones(x.shape[0]), x))

    theta = np.zeros(x.shape[1])
    return minimization_func(x, y, theta, cost_func=cost_func,
                             cost_func_der=cost_func_der, **kwargs)

    # Nelder-Mead method
def nelder_mead_algo(x, y, theta, cost_func=cost_func_vectorized, **kwargs):
    from scipy.optimize import fmin

    res_theta = fmin(lambda theta: cost_func(x, y, theta),
                        theta, xtol=THRESHOLD, maxfun=150000)
    return res_theta, []

# B-F-G-S method
def bfgs_algo(x, y, theta, cost_func=cost_func_vectorized, cost_func_der=cost_func_deriative, **kwargs):
    from scipy.optimize import fmin_bfgs

    res_theta = fmin_bfgs(lambda theta: cost_func(x, y, theta),
                          theta, fprime=lambda theta: cost_func_der(x, y, theta),
                          gtol=THRESHOLD)
    return res_theta, []

fit(x_train, y_train, minimization_func=bfgs_algo)
fit(x_train, y_train, minimization_func=nelder_mead_algo)


def predict(x, theta, regularized=False):
    x = np.array(x)
    if not regularized:
        x = np.insert(x, 0, 1)
    h_value = h(x, theta)
    return 1 if h_value >= 0.5 else 0

theta, logs = fit(x_train, y_train)

def x2_func(x1, theta):
    return -(theta[0] + theta[1] * x1) / theta[2]

df_accepted = df[df['accepted'] == 1]
df_not_accepted = df[df['accepted'] == 0]
fig, ax = plt.subplots()
ax.scatter(df_accepted['first_exam'], df_accepted['second_exam'], marker='o', label='Accepted', s=20)
ax.scatter(df_not_accepted['first_exam'], df_not_accepted['second_exam'], marker='o', c='r', label='Not accepted', s=20)
ax.plot(df_not_accepted['first_exam'],
        [x2_func(i, theta) for i in df_not_accepted['first_exam']],
        c='b', label='boundary')
ax.set_xlabel('1st exam')
ax.set_ylabel('2nd exam')
plt.show()
print(f'Кол-во итераций: {len(logs)}')

df = pd.read_csv('data/ex2data2.txt', header=None, names=['first_test', 'second_test', 'passed'])
x_train, y_train = df.filter(['first_test', 'second_test']), df['passed']
df

df_accepted = df[df['passed'] == 1]
df_not_accepted = df[df['passed'] == 0]
fig, ax = plt.subplots()
ax.scatter(df_accepted['first_test'], df_accepted['second_test'], marker='o', label='Passed', s=20)
ax.scatter(df_not_accepted['first_test'], df_not_accepted['second_test'], marker='o', c='r', label='Not passed', s=20)
ax.set_xlabel('1st test')
ax.set_ylabel('2nd test')
plt.show()


def build_feature_for_pair(x1, x2, degree):
    res = []
    for i in range(degree + 1):
            for j in range(i, degree + 1):
                res.append(x1**(j - i) * x2**i)
    return res

def build_features(x1, x2, degree):
    return [build_feature_for_pair(x1[idx], x2[idx], degree) for idx in range(len(x1))]

extended_x_train = build_features(x_train['first_test'], x_train['second_test'], 6)
extended_x_train = pd.DataFrame(extended_x_train)

def cost_func_vectorized_reg_l2(x, y, theta, penalty_term=0.1):
    cost = cost_func_vectorized(x, y, theta)
    theta_sliced = theta[1:]
    penalty_cost = (penalty_term / 2 / x.shape[0]) * np.dot(theta_sliced.T, theta_sliced)
    return cost + penalty_cost

def cost_func_deriative_reg_l2(x, y, theta, penalty_term=0.1):
    err = h(x, theta) - y
    grad = np.dot(x.T[:1], err)
    grad_with_reg = np.dot(x.T[1:], err) + penalty_term * theta[1:]
    grad = np.insert(grad_with_reg, 0, grad)
    grad /= x.shape[0]
    return grad

theta, logs = fit(extended_x_train, y_train, minimization_func=gradient_descent_vectorized,
               regularized=True, cost_func=cost_func_vectorized_reg_l2, cost_func_der=cost_func_deriative_reg_l2)
theta, logs[-1][1]

theta1, logs1 = fit(extended_x_train, y_train, minimization_func=nelder_mead_algo,
                  regularized=True, cost_func=cost_func_vectorized_reg_l2)
theta2, logs2 = fit(extended_x_train, y_train, minimization_func=bfgs_algo,
                  regularized=True, cost_func=cost_func_vectorized_reg_l2, cost_func_der=cost_func_deriative_reg_l2)
theta1, theta2
