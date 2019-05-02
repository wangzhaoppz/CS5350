import tensorflow as tf
import numpy as np
from tensorflow import keras

train_data = []
train_labels = []
test_data = []
test_labels = []

f = open("bank-note/train.csv")
for line in f:
    attrs = line.strip().split(',')
    train_data.append([float(s) for s in attrs[:-1]])
    if attrs[-1] == '1':
        train_labels.append(1)
    else:
        train_labels.append(0)

f = open("bank-note/test.csv")
for line in f:
    attrs = line.strip().split(',')
    test_data.append([float(s) for s in attrs[:-1]])
    if attrs[-1] == '1':
        test_labels.append(1)
    else:
        test_labels.append(0)

train_data = np.array(train_data)
train_labels = np.array(train_labels)
test_data = np.array(test_data)
test_labels = np.array(test_labels)

print("3.(e) starts generating...\n")

f = open("data_result/3e_result.txt", "w")

widths = [5, 10, 25, 50, 100]
depths = [3, 5, 9]
acts = [keras.activations.tanh, tf.nn.relu]
inits = [tf.contrib.layers.xavier_initializer(), tf.initializers.he_normal()]
for i in range(2):
    f.write("Result " + str(i) + "\n")
    f.write("w/d\tTraining\tTesting\n")
    for w in widths:
        for d in depths:
            layers = [keras.layers.Dense(units=w, activation=acts[i], kernel_initializer=inits[i], bias_initializer=inits[i]) for _ in range(d - 1)]
            layers.append(keras.layers.Dense(units=1, activation=acts[i], kernel_initializer=inits[i], bias_initializer=inits[i]))

            model = keras.Sequential(layers)

            model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

            model.fit(train_data, train_labels, epochs=10)

            _, test_acc = model.evaluate(test_data, test_labels)
            _, train_acc = model.evaluate(train_data, train_labels)

            f.write(str(w) + "/" + str(d) + "\t" + str(train_acc) + "\t" + str(test_acc) + "\n")

    f.write("\n")        

print("3.(e) complete~\n")