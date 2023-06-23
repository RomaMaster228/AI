from neuro_learner.perceptron import Learner, Linear, Tanh, Softmax, CrossEntropyLoss
from neuro_learner.datasets import ClassDataset

learner = Learner(Linear(2, 5), Tanh(), Linear(5, 2), Softmax())
loss = CrossEntropyLoss()
train_x, train_y, test_x, test_y = ClassDataset().train_test_split()

# print("Initial loss={}, accuracy={}: ".format(*get_loss_acc(train_x, train_labels)))
res = learner.train_and_plot(train_x, train_y, test_x, test_y, n_epoch=30, loss=loss, lr=0.01)
