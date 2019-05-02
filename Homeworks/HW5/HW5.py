from random import shuffle
import math
import numpy as np

def data_im(name):
    data = []

    with open(name) as f:
        for line in f:
            data.append(line.strip().split(','))

    return data

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))

def gamma_func(r, t): 
    return r / (1 + ((r/0.001) * t))    

def MAP(S, T, variance, gamma, d):
    w = [0.0 for _ in range(len(S[0]) - 1)]
    m = float(len(S))

    for t in range(T):
        shuffle(S)
        for s in S:
            gradient = []
            for i in range(len(w)):
                dp = np.dot(w, s[:-1])
                e = math.exp(dp * -s[-1])
                gradient.append(-(s[-1] * s[i] * m * e)/(1 + e) + (s[i]/(2 * variance)))

            w = w + s[-1] * gamma * np.array(gradient)

        gamma = gamma / (1 + (gamma/d) * t)

    return w

def ML(S, T, variance, gamma, d):
    w = [0.0 for _ in range(len(S[0]) - 1)]
    m = float(len(S))

    for t in range(T):
        shuffle(S)
        for s in S:
            gradient = []
            for i in range(len(w)):
                dp = np.dot(w, s[:-1])
                e = math.exp(dp * -s[-1])
                gradient.append(-(s[-1] * s[i] * m * e)/(1 + e))

            w = w + s[-1] * gamma * np.array(gradient)

        gamma = gamma / (1 + (gamma/d) * t)

    return w

def Error_Gene(S, w):
    error = 0
    for s in S:
        guess = np.dot(s[:-1], w)
        
        if guess > 0 and s[-1] == 1:
            pass
        elif guess < 0 and s[-1] == -1:
            pass
        else:
            error += 1

    return error/float(len(S))

class Network:

    def __init__(self):
        self.layers = []

    
    def add_layer(self, layer):
        self.layers.append(layer)

    def backpropagate(self, y_star, use_bias=True):
        layer_count = len(self.layers)
        output = []
        cached_gradient = []
        last_layer = None

        for i in range(layer_count):
            current_layer = self.layers[layer_count - i - 1]
            weight_gradients = np.zeros((current_layer.weights.shape[1], current_layer.weights.shape[0]))

            if i == 0:
                cached_gradient = current_layer.output - y_star
            elif use_bias:
                cached_gradient = np.transpose(cached_gradient) * last_layer.gradient * last_layer.weights[1:]
            else:
                cached_gradient = np.transpose(cached_gradient) * last_layer.gradient * last_layer.weights

            gs = np.transpose(cached_gradient) * current_layer.gradient

            for j in range(gs.shape[0]):
                weight_gradients += np.outer(gs[j], current_layer.input)

            output.append(weight_gradients)
            last_layer = current_layer

        return output

    def apply_input(self, input_vector, use_bias=True):

        for layer in self.layers:

            if use_bias:
                input_vector = np.append([1.0], input_vector)

            input_vector = layer.apply_input(input_vector)

        return input_vector 

    def random_init(self):
        for layer in self.layers:
            layer.random_init()

    def score(self, data, prob):
        count = 0.0

        for example in data:
            x = example[:-1]
            y = example[-1]
            output = self.apply_input(x)[0]
        
            if (output >= prob) != y:
                count += 1.0

        return count / data.shape[0]

    def train_sgd(self, data, r0, r_func, T):
        data = np.array(data)

        for t in range(T):
            np.random.shuffle(data)
            x_data = data[:,:-1]
            y_data = data[:,-1]

            for i in range(y_data.shape[0]):
                x = x_data[i]
                y = y_data[i]

                self.apply_input(x)

                gradients = self.backpropagate(y)

                for j in range(len(gradients)):
                    self.layers[len(gradients) - j - 1].weights -= np.transpose(gradients[j]) * r_func(r0, t)

class Layer:

    def __init__(self, input_shape, output_shape, activation_func, derivative_func):

        self.weights = np.zeros((input_shape, output_shape), dtype='float64')
        self.gradient = np.zeros((input_shape, output_shape), dtype='float64')
        self.input = np.zeros((input_shape), dtype='float64')
        self.output = np.zeros((output_shape), dtype='float64')

        self.func = activation_func
        self.func_d = derivative_func

    def apply_weight(self, node, weight):
        if weight.shape[0] != self.weights.shape[1]:
            print("Error! Invalid weight vector.")
            return None

        self.weights[node] = weight

    def apply_input(self, input_vector):

        self.input = input_vector
        self.output = self.func(np.einsum('ij, i->j', self.weights, input_vector))
        self.gradient = self.func_d(np.einsum('ij, i->j', self.weights, input_vector))

        return self.output

    def random_init(self):
        self.weights = np.random.normal(size=self.weights.shape)


f = open("bank-note/train.csv")
train = []
for line in f:
    attrs = line.strip().split(',')
    example = [float(s) for s in attrs[:-1]]
    if attrs[-1] == '1':
        example.append(1)
    else:
        example.append(-1)
    train.append(example)

f = open("bank-note/test.csv")
test = []
for line in f:
    attrs = line.strip().split(',')
    example = [float(s) for s in attrs[:-1]]
    if attrs[-1] == '1':
        example.append(1)
    else:
        example.append(-1)
    test.append(example)

test = np.array(test)

print "Start generating the result...\n\n"

#Problem 2
print "2.(a) starts generating...\n\n"

f = open("data_result/2a_result.txt", "w")

variances = [0.01, 0.1, 0.5, 1, 3, 5, 10, 100]
f.write("var\tTraining Error\tTesting Error\n")
for v in variances:
    train_temp = np.array(train)
    w = MAP(train_temp, 100, v, 0.0000001, 0.000001)
    train_error = Error_Gene(train, w)
    test_error = Error_Gene(test, w)
    f.write(str(v) + "\t" + str(train_error) + "\t" + str(test_error) + "\n")

print "2.(a) Complete~\n\n"

print "2.(b) starts generating...\n\n"

f = open("data_result/2b_result.txt", "w")

variances = [0.01, 0.1, 0.5, 1, 3, 5, 10, 100]
f.write("var\tTraining Error\tTesting Error\n")
for v in variances:
    train_temp = np.array(train)
    w = ML(train_temp, 100, v, 0.0000001, 0.000001)
    train_error = Error_Gene(train, w)
    test_error = Error_Gene(test, w)
    f.write(str(v) + "\t" + str(train_error) + "\t" + str(test_error) + "\n")

print "2.(b) Complete~\n\n"

train = np.array(data_im("bank-note/train.csv"), dtype='float64')
test = np.array(data_im("bank-note/test.csv"), dtype='float64')

#Problem 3
print "3.(a) starts generating...\n\n"

f = open("data_result/3a_result.txt", "w")

layer0 = Layer(3, 2, sigmoid, sigmoid_prime)
layer0.weights = np.array([[-1.0, 1.0], [-2.0, 2.0], [-3.0, 3.0]])
layer1 = Layer(3, 2, sigmoid, sigmoid_prime)
layer1.weights = np.array([[-1.0, 1.0], [-2.0, 2.0], [-3.0, 3.0]])
layer2 = Layer(3, 1, lambda x : x, lambda x : np.ones(x.shape))
layer2.weights = np.array([[-1.0], [2.0], [-1.5]])

net = Network()
net.add_layer(layer0)
net.add_layer(layer1)
net.add_layer(layer2)

net.apply_input([1, 1])
gradients = net.backpropagate([1])

f.write("Layer 0 gradient: \n" + str(np.transpose(gradients[0])) + "\n")
f.write("Layer 1 gradient: \n" + str(np.transpose(gradients[1])) + "\n")
f.write("Layer 2 gradient: \n" + str(np.transpose(gradients[2])) + "\n")

print "3.(a) Complete~\n\n"

print "3.(b) starts generating...\n\n"

f = open("data_result/3b_result.txt", "w")

widths = [5, 10, 25, 50, 100]

feature_count = train[:,:-1].shape[1]
epochs = 100

f.write("w\tTraining\tTesting\n")

for w in widths:

    layer0 = Layer(feature_count + 1, w, sigmoid, sigmoid_prime)
    layer1 = Layer(w + 1, w, sigmoid, sigmoid_prime)
    layer2 = Layer(w + 1, 1, lambda x : x, lambda x : np.ones(x.shape))

    net = Network()
    net.add_layer(layer0)
    net.add_layer(layer1)
    net.add_layer(layer2)
    net.random_init()

    net.train_sgd(train, 0.1, gamma_func, epochs)
    train_error = net.score(train, 0.5) * 100
    test_error = net.score(test, 0.5) * 100
    f.write(str(w) + "\t" + str(train_error) + "\t" + str(test_error) + "\n")

print "3.(b) Complete~\n\n"

print "3.(c) starts generating...\n\n"

f = open("data_result/3c_result.txt", "w")
f.write("w\tTraining\tTesting\n")

for w in widths:

    layer0 = Layer(feature_count + 1, w, sigmoid, sigmoid_prime)
    layer1 = Layer(w + 1, w, sigmoid, sigmoid_prime) 
    layer2 = Layer(w + 1, 1, sigmoid, sigmoid_prime)

    net = Network()
    net.add_layer(layer0)
    net.add_layer(layer1)
    net.add_layer(layer2)

    net.train_sgd(train, 0.1, gamma_func, epochs)
    train_error = net.score(train, 0.5) * 100
    test_error = net.score(test, 0.5) * 100
    f.write(str(w) + "\t" + str(train_error) + "\t" + str(test_error) + "\n")

print "3.(c) Complete~\n\n"
print "All Complete~\n\n"
