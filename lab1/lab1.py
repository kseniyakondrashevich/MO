import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

ex1data1 = np.loadtxt('ex1data1.txt', delimiter=',')
x, y = np.expand_dims(ex1data1[:, 0], axis=1), ex1data1[:, 1]

plt.scatter(x, y)
plt.xlabel('Population', size=16)
plt.ylabel('Income', size=16)
plt.show()


def MSECost(y, theta):
    def predictedValue(x):
        return theta[0]*x + theta[1]

    def lossValue(existent, predicted):
        return (existent - predicted)

    hypothesis = map(predictedValue, y)
    lossValue = map(lossValue, y, hypothesis)
    costValue = sum(map(lambda x: x**2, lossValue)) / (2*len(y))
    return hypothesis, lossValue, costValue

theta = [0.8, 0.8]
_, _, cost = MSECost(y, theta)
print(cost)

def cost_func(x, y, theta):
    j = 0
    count = len(y)
    for i in range(count):
        h = theta[0] + theta[1] * x[i]
        j += (h - y[i])**2
    return j / (2 * count)

def f(x, y, theta):
    return theta[0] + x * theta[1] - y

def gradient(x, y, theta, a, iter_count):
    training_logs = []
    for i in range(iter_count):
        dj_dt0, dj_dt1 =0, 0
        count = len(x)
        for j in range(len(x)):
            h = f(x[j], y[j], theta)
            dj_dt0 += h
            dj_dt1 += x[j] * h
        dj_dt0 /= count
        dj_dt1 /= count
        theta[0] -= a* dj_dt0
        theta[1] -=a * dj_dt1

        curr_cost = cost_func(x, y, theta)
        training_logs.append([i, curr_cost, theta[0], theta[1]])
    return theta, training_logs

theta, logs = gradient(
    x, y, theta = [0, 0], a = 0.012, iter_count = 1000
)

plt.scatter(x, y)
plt.xlabel('Population', size=16)
plt.ylabel('Income', size=16)
x = np.linspace(5, 25, 3)
plt.plot(x, theta[0] + theta[1] * x, 'g')
plt.show()

logs_df = pd.DataFrame(logs, columns=['iter', 'loss', 'theta0', 'theta1'])
data = logs_df[logs_df['theta1'] > 1]
X, Y = np.meshgrid(data['theta0'], data['theta1'])
Z = np.zeros((data['theta0'].size, data['theta1'].size))

for i, theta0 in enumerate(data['theta0']):
    for j, theta1 in enumerate(data['theta1']):
        Z[i, j] = cost_func(x, y, [theta0, theta1])

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

fig = plt.figure()
ax = fig.gca(projection = '3d')
ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0)
ax.set_xlabel('Theta 0')
ax.set_ylabel('Theta 1')
ax.set_zlabel('Loss function')
plt.show()

fig, ax = plt.subplots()
CS = ax.contour(X, Y, Z, cmap='viridis')
ax.clabel(CS, inline=1, fontsize=10)
ax.set_xlabel('Theta 0')
ax.set_ylabel('Theta 1')
plt.show()


