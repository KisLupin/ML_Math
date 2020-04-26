import numpy as np
import matplotlib.pyplot as plt

np.random.seed(2)

X = np.array([[0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 1.75, 2.00, 2.25, 2.50,
               2.75, 3.00, 3.25, 3.50, 4.00, 4.25, 4.50, 4.75, 5.00, 5.50]])  # input
y = np.array([0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1])  # brand

# extended data
X = np.concatenate((np.ones((1, X.shape[1])), X), axis=0)  # data

def sigmoid(s):  # sigmoid func
    return 1 / (1 + np.exp(-s))


def logistic_sigmoid_regression(X, y, w_init, eta, tol=1e-4, max_count=10000):   # logistic func
    w = [w_init]  # khoi tao mang vector trong so
    N = X.shape[1]  # kich thuoc matran theo cot
    d = X.shape[0]  # kich thuoc ma tran theo hang
    count = 0
    check_w_after = 20
    while count < max_count:
        # mix data
        mix_id = np.random.permutation(N)  # hoan vi data
        for i in mix_id:
            xi = X[:, i].reshape(d, 1)  # chuyen vi ma tran input
            yi = y[i]  # lay ra brand cua phan tu tuong ung
            zi = sigmoid(np.dot(w[-1].T, xi))  # tinh loss func theo sigmoid func
            w_new = w[-1] + eta * (yi - zi) * xi  # tinh vector trong so theo SGD (optimize)
            count += 1
            # stopping criteria
            if count % check_w_after == 0:
                if np.linalg.norm(w_new - w[-check_w_after]) < tol:  # kiem tra dao ham de lay ra sai so du nho
                    return w
            w.append(w_new)
    return w, count


eta = .05
d = X.shape[0]
w_init = np.random.randn(d, 1)
w, count = logistic_sigmoid_regression(X, y, w_init, eta)

# X0 = X[1, np.where(y == 0)][0]
# y0 = y[np.where(y == 0)]
# X1 = X[1, np.where(y == 1)][0]
# y1 = y[np.where(y == 1)]
#
# plt.plot(X0, y0, 'ro', markersize = 8)
# plt.plot(X1, y1, 'bs', markersize = 8)
#
# xx = np.linspace(0, 6, 1000)
# w0 = w[-1][0][0]
# w1 = w[-1][1][0]
# threshold = -w0/w1
# yy = sigmoid(w0 + w1*xx)
# plt.axis([-2, 8, -1, 2])
# plt.plot(xx, yy, 'g-', linewidth = 2)
# plt.plot(threshold, .5, 'y^', markersize = 8)
# plt.xlabel('studying hours')
# plt.ylabel('predicted probability of pass')
# plt.show()
