from __future__ import division
import numpy as np
import random

class NNet(object):

  def __init__(self,layers,reg_constant):

    self.layers = layers;
    self.num_layers = len(layers);

    self.bias = [np.random.randn(y,1) for y in layers[1:]];
    self.theta = [np.random.randn(y,x) for x,y in zip(layers[:-1],layers[1:])];

    self.reg_constant = reg_constant;

  def feedforward_for_test(self,test_input):
    x = test_input;
    for b,t in zip(self.bias, self.theta):
      x = self.logistic_regression(np.dot(t,x)+b)
    return x;

  def feedforward_for_training(self, training_input):
    x = [training_input];
    z = []
    for b,t in zip(self.bias, self.theta):
      temp_z = np.dot(t,x[-1])+b
      z.append(temp_z)
      temp_x = self.logistic_regression(temp_z);
      x.append(temp_x);
    return x,z;

  def logistic_regression(self, z):
    return 1.0/(1.0+np.exp(-z))

  def logistic_regression_prime(self, z):
    return self.logistic_regression(z)*(1-self.logistic_regression(z))

  def train_w_stochastic_grad_descent(self, training_data, batch_size, training_rate):
    random.shuffle(training_data);
    num_data = len(training_data);
    batches = [];
    for x in xrange(0, num_data, batch_size):
      batches.append(training_data[x:x+batch_size])

    for batch in batches:
      self.update_w_batch(batch, training_rate)

  def update_w_batch(self, training_data, training_rate):
    total_delta_theta = [ np.zeros(t.shape) for t in self.theta ];
    total_delta_bias = [ np.zeros(b.shape) for b in self.bias ];
    for x,y in training_data:
      set_layers, z = self.feedforward_for_training(x);
      delta_theta, delta_bias = self.backprop(set_layers, z, y);

      total_delta_theta = [total_dt+dt for total_dt, dt in zip(total_delta_theta, delta_theta)];
      total_delta_bias = [total_db+db for total_db, db in zip(total_delta_bias, delta_bias)];

    for x in xrange(0, len(total_delta_theta)):
      sidecalc = np.multiply(training_rate/len(training_data), total_delta_theta[x]);
      self.theta[x] -= sidecalc;

    for x in xrange(0, len(total_delta_bias)):
      sidecalc = np.multiply(training_rate/len(training_data), total_delta_bias[x]);
      self.bias[x] -= sidecalc;

  def backprop(self, layers, z , y):
    delta_bias = [np.zeros(b.shape) for b in self.bias];
    delta_theta = [np.zeros(t.shape) for t in self.theta];

    small_delta = (layers[-1]-y)* self.logistic_regression_prime(z[-1])

    delta_bias[-1] = small_delta;
    delta_theta[-1] = np.dot(small_delta, layers[-2].T)

    for l in xrange(2, self.num_layers):
      temp_z = z[-l]
      sidecalc1 = self.logistic_regression_prime(temp_z);
      sidecalc2 = np.dot(self.theta[-l+1].T, small_delta)
      small_delta = sidecalc2 * sidecalc1;

      delta_bias[-l] = small_delta;
      delta_theta[-l]= np.dot(small_delta, layers[-l-1].T)

    return delta_theta, delta_bias;

  def evaluate(self, test_data):
    test_results = [(np.argmax(self.feedforward_for_test(x)),y)
        for (x,y) in test_data];
    return sum(int(x==y) for (x,y) in test_results)


