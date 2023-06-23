import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from tqdm import tqdm

np.random.seed(0)

# Plot functions


def plot_dataset(suptitle, features, labels):
    # prepare the plot
    fig, ax = plt.subplots(1, 1)
    # pylab.subplots_adjust(bottom=0.2, wspace=0.4)
    fig.suptitle(suptitle, fontsize=16)
    ax.set_xlabel('$x_i[0]$ -- (feature 1)')
    ax.set_ylabel('$x_i[1]$ -- (feature 2)')

    colors = ['r' if l else 'b' for l in labels]
    ax.scatter(features[:, 0], features[:, 1], marker='o', c=colors, s=100, alpha=0.5)
    plt.show()


def plot_decision_boundary(train_x, train_y, net, fig, ax):
    draw_colorbar = True
    # remove previous plot
    while ax.collections:
        ax.collections.pop()
        draw_colorbar = False

    # generate countour grid
    x_min, x_max = train_x[:, 0].min() - 1, train_x[:, 0].max() + 1
    y_min, y_max = train_x[:, 1].min() - 1, train_x[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    grid_points = np.c_[xx.ravel().astype('float32'), yy.ravel().astype('float32')]
    n_classes = max(train_y) + 1
    while train_x.shape[1] > grid_points.shape[1]:
        # pad dimensions (plot only the first two)
        grid_points = np.c_[grid_points,
        np.empty(len(xx.ravel())).astype('float64')]  # Fix overflow
        grid_points[:, -1].fill(train_x[:, grid_points.shape[1] - 1].mean())

    # evaluate predictions
    prediction = np.array(net.forward(grid_points))
    # for two classes: prediction difference
    if n_classes == 2:
        Z = np.array([0.5 + (p[0] - p[1]) / 2.0 for p in prediction]).reshape(xx.shape)
    else:
        Z = np.array([p.argsort()[-1] / float(n_classes - 1) for p in prediction]).reshape(xx.shape)

    # draw contour
    levels = np.linspace(0, 1, 40)
    cs = ax.contourf(xx, yy, Z, alpha=0.4, levels=levels)
    if draw_colorbar:
        fig.colorbar(cs, ax=ax, ticks=[0, 0.5, 1])
    c_map = [cm.jet(x) for x in np.linspace(0.0, 1.0, n_classes)]
    colors = [c_map[l] for l in train_y]
    ax.scatter(train_x[:, 0], train_x[:, 1], marker='o', c=colors, s=60, alpha=0.5)


def plot_training_progress(x, y_data, fig, ax):
    styles = ['k--', 'g-']
    # remove previous plot
    while ax.lines:
        ax.lines.pop()
    # draw updated lines
    for i in range(len(y_data)):
        ax.plot(x, y_data[i], styles[i])
    ax.legend(ax.lines, ['training accuracy', 'validation accuracy'],
              loc='upper center', ncol=2)


# Perceptron


class Linear:

    def __init__(self, nin, nout):
        self.W = np.random.normal(0, 1.0 / np.sqrt(nin), (nout, nin))
        self.b = np.zeros((1, nout))
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        self.v_W = self.dW
        self.v_b = self.db
        self.G_W = self.dW
        self.G_b = self.db

    def forward(self, x):
        self.x = x
        return np.dot(x, self.W.T) + self.b

    def backward(self, dz):
        dx = np.dot(dz, self.W)
        dW = np.dot(dz.T, self.x)
        db = dz.sum(axis=0)
        self.dW = dW
        self.db = db
        return dx

    def sgd(self, lr):
        self.W -= lr * self.dW
        self.b -= lr * self.db
        
    def momentum(self, lr):
        self.v_W = 0.9*self.v_W + 0.1*lr*self.dW
        self.v_b = 0.9*self.v_b + 0.1*lr*self.db
        self.W -= self.v_W
        self.b -= self.v_b
    
    def rmsp(self, lr):
        self.G_W = 0.9*self.G_W + 0.1*np.square(self.dW)
        self.G_b = 0.9*self.G_b + 0.1*np.square(self.db)
        self.G_W[:] += 1e-8
        self.G_b[:] += 1e-8
        self.W -= lr*np.divide(self.dW, np.sqrt(self.G_W))
        self.b -= lr*np.divide(self.db, np.sqrt(self.G_b))


# Activation functions


class Softmax:

    def forward(self, z):
        self.z = z
        zmax = z.max(axis=1, keepdims=True)
        expz = np.exp(z - zmax)
        Z = expz.sum(axis=1, keepdims=True)
        return expz / Z

    def backward(self, dp):
        p = self.forward(self.z)
        pdp = p * dp
        return pdp - p * pdp.sum(axis=1, keepdims=True)


class Tanh:

    def forward(self, x):
        y = np.tanh(x)
        self.y = y
        return y

    def backward(self, dy):
        return (1.0 - self.y ** 2) * dy


class ReLU:

    def forward(self, input):
        self.input = input
        self.output = np.maximum(input, 0)
        return self.output

    def backward(self, dy):
        mask = self.input > 0
        return np.multiply(dy, mask)


# Loss functions

# loss functions for classification
def zero_one(d):
    if d < 0:
        return 1
    return 0


def logistic_loss(fx):
    # assumes y == 1
    y = 1
    return 1 / np.log(2) * np.log(1 + np.exp(-y * fx))


zero_one_v = np.vectorize(zero_one)

# loss functions for regression
x = np.linspace(-2, 2, 101)
loss_funcs_regression = [np.abs(x), np.power(x, 2)]


class CrossEntropyLoss:

    def forward(self, p, y):
        self.p = p
        self.y = y
        p_of_y = p[np.arange(len(y)), y]
        log_prob = np.log(p_of_y)
        return -log_prob.mean()

    def backward(self, loss):
        dlog_softmax = np.zeros_like(self.p)
        dlog_softmax[np.arange(len(self.y)), self.y] -= 1.0 / len(self.y)
        return dlog_softmax / (np.clip(self.p, 1e-8, 1 - 1e-8))  # Fix ZeroDivisionError


# Model


class Net:
    def __init__(self):
        self.layers = []

    def add(self, l):
        self.layers.append(l)

    def forward(self, x):
        for l in self.layers:
            x = l.forward(x)
        return x

    def backward(self, z):
        for l in self.layers[::-1]:
            z = l.backward(z)
        return z

    def update(self, lr, optimizer):
        for l in self.layers:
            if optimizer in l.__dir__():
                if optimizer == "sgd":
                    l.sgd(lr)
                elif optimizer == "momentum":
                    l.momentum(lr)
                elif optimizer == "rmsp":
                    l.rmsp(lr)


def get_loss_acc(x, y, net, loss=CrossEntropyLoss()):
    p = net.forward(x)
    l = loss.forward(p, y)
    pred = np.argmax(p, axis=1)
    acc = (pred == y).mean()
    return l, acc


def train_epoch(net, train_x, train_labels, loss=CrossEntropyLoss(), optimizer="sgd", batch_size=4, lr=0.1):
    for i in range(0, len(train_x), batch_size):
        xb = train_x[i:i + batch_size]
        yb = train_labels[i:i + batch_size]

        p = net.forward(xb)
        l = loss.forward(p, yb)
        dp = loss.backward(l)
        dx = net.backward(dp)
        net.update(lr, optimizer)


def train_and_plot(train_x, train_y, test_x, test_y, n_epoch, net: Net, loss=CrossEntropyLoss(), optimizer="sgd",
                   batch_size=4, lr=0.1):
    fig, ax = plt.subplots(2, 1)
    ax[0].set_xlim(0, n_epoch + 1)
    ax[0].set_ylim(0, 1)

    train_acc = np.empty((n_epoch, 3))
    train_acc[:] = np.NAN
    valid_acc = np.empty((n_epoch, 3))
    valid_acc[:] = np.NAN

    for epoch in tqdm(range(1, n_epoch + 1)):
        train_epoch(net, train_x, train_y, loss, optimizer, batch_size, lr)
        tloss, taccuracy = get_loss_acc(train_x, train_y, net, loss)
        train_acc[epoch - 1, :] = [epoch, tloss, taccuracy]
        vloss, vaccuracy = get_loss_acc(test_x, test_y, net, loss)
        valid_acc[epoch - 1, :] = [epoch, vloss, vaccuracy]

        ax[0].set_ylim(0, max(max(train_acc[:, 2]), max(valid_acc[:, 2])) * 1.1)

        plot_training_progress(train_acc[:, 0], (train_acc[:, 2],
                                                 valid_acc[:, 2]), fig, ax[0])
        plot_decision_boundary(train_x, train_y, net, fig, ax[1])
        fig.canvas.draw()
        fig.canvas.flush_events()

    plt.show()
    return train_acc, valid_acc


class Learner:

    def __init__(self, *args):
        self.net = Net()
        for layer in args:
            self.net.add(layer)

    def train_and_plot(self, train_x, train_y, test_x, test_y, n_epoch, loss=CrossEntropyLoss(), optimizer="sgd",
                       batch_size=4, lr=0.1):
        res = train_and_plot(train_x, train_y, test_x, test_y, n_epoch, self.net, loss, optimizer, batch_size, lr)
        print(f'Train loss: {res[0][-1][1]}, accuracy = {res[0][-1][2]}')
        print(f'Test loss: {res[1][-1][1]}, accuracy = {res[1][-1][2]}')
        return res

    def predict(self, data):
        y_pred = self.net.forward(data)
        print('Prediction:', np.argmax(y_pred, axis=1)[0])
