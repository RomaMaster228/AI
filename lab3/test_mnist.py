import pylab
from neuro_learner.perceptron import Learner, Linear, ReLU, Softmax, CrossEntropyLoss
from neuro_learner.datasets import Mnist

learner = Learner(Linear(784, 128), ReLU(), Linear(128, 32), ReLU(), Linear(32, 10), Softmax())
loss = CrossEntropyLoss()
train_x, test_x, train_y, test_y = Mnist().train_test_split()

res = learner.train_and_plot(train_x, train_y, test_x, test_y, n_epoch=50, loss=loss, batch_size=16, lr=0.00001)

# Prediction of model
picture = test_x[228]
pylab.imshow(picture.reshape(28, 28))
learner.predict(picture)
pylab.show()
