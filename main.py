import NNet
import mnist_loader

training_data, validation_data, test_data = \
    mnist_loader.load_data_wrapper();

layers = [28*28,30,10];

n = NNet.NNet(layers,0.1)
n.train_w_stochastic_grad_descent(training_data,10,1);
print n.evaluate(test_data);
