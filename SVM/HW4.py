from random import *
from scipy.optimize import *
import numpy as np
import math

#SVM methods
def Primal_Train(S, Attributes, C, ite, g, d): 
    w = [0 for _ in range(len(Attributes))]

    for t in range(0, ite):
        shuffle(S)
        for s in S:
            prod = 0
            for i in range(len(w)):
                prod += s[Attributes[i]] * w[i]
            prod *= s['Label']

            if prod <= 1:
                w = [(1 - g) * w[i] + (g * C * len(S) * s['Label'] * s[Attributes[i]]) for i in range(len(w))]
            else:
                w = [(1 - g) * w[i] for i in range(len(w))]

        g = g / (1 + g/d*t)
    return w

def Primal_Train_SA(S, Attributes, C, ite, g, d): 
    w = [0 for _ in range(len(Attributes))]

    for t in range(0, ite):
        shuffle(S)
        for s in S:
            prod = 0
            for i in range(len(w)):
                prod += s[Attributes[i]] * w[i]
            prod *= s['Label']

            if prod <= 1:
                w = [(1 - g) * w[i] + (g * C * len(S) * s['Label'] * s[Attributes[i]]) for i in range(len(w))]
            else:
                w = [(1 - g) * w[i] for i in range(len(w))]

        g = g / (1 + g/d*t)
    return w

def Primal_Train_SB(S, Attributes, C, ite, g, d): 
    w = [0 for _ in range(len(Attributes))]

    for t in range(0, ite):
        shuffle(S)
        for s in S:
            prod = 0
            for i in range(len(w)):
                prod += s[Attributes[i]] * w[i]
            prod *= s['Label']

            if prod <= 1:
                w = [(1 - g) * w[i] + (g * C * len(S) * s['Label'] * s[Attributes[i]]) for i in range(len(w))]
            else:
                w = [(1 - g) * w[i] for i in range(len(w))]

        g = g / (1 + t)
    return w

def Primal_Test(W, S, Attributes):
    wrong = 0
    for s in S:
        guess = 0.0
        for i in range(0, len(W)):
            guess += float(s[Attributes[i]]) * W[i]
        
        if guess > 0 and s["Label"] == 1:
            pass
        elif guess < 0 and s["Label"] == -1:
            pass
        else:
            wrong += 1

    return wrong/float(len(S))

def Dual_Train(S, C):
    def main(x):
        ret_val = 0
        for i in xrange(len(x)):
            for j in xrange(len(x)):
                ret_val += x[i] * x[j] * np.dot(S[i], S[j])
        return (ret_val / 2) - sum(x)

    def const(x):
        ret_val = 0
        for i in xrange(len(x)):
            ret_val += x[i] * S[i][-1]
        return ret_val

    bounds = [(0, C) for _ in range(len(S))]
    w = [0 for _ in range(len(S))]
    constraints = ({'type':'eq', 'fun':const})

    result = minimize(main, w, method='SLSQP', bounds=bounds, constraints=constraints)
    A = result.x

    w = [0 for _ in range(len(S[0]) - 1)]
    for i in range(len(S)):
        for j in range(len(S[0]) - 1):
            w[j] += A[i] * S[i][-1] * S[i][j]

    count = 0
    for i in range(len(A)):
        if A[i] > 0 and A[i] < C:
            b = 0
            for j in range(len(w)):
                b += w[j] * S[i][j]
            count += 1

    return w, b/count

def Dual_Test(w, b, S, Attributes):
    wrong = 0
    for s in S:
        guess = b
        for i in range(len(w)):
            guess += float(s[Attributes[i]]) * w[i]
        
        if guess > 0 and s["Label"] == 1:
            pass
        elif guess < 0 and s["Label"] == -1:
            pass
        else:
            wrong += 1

    return wrong/float(len(S))

def Kernel_Train(S, C, g):
    def main(x):
        ret_val = 0
        for i in xrange(len(x)):
            for j in xrange(len(x)):
                ret_val += x[i] * x[j] * S[i][-1] * S[j][-1] * kernel(S[i][:-1], S[j][:-1], g)
        return (ret_val / 2) - sum(x)

    def const(x):
        ret_val = 0
        for i in xrange(len(x)):
            ret_val += x[i] * S[i][-1]
        return ret_val

    bounds = [(0, C) for _ in range(len(S))]
    w = [0 for _ in range(len(S))]
    constraints = ({'type':'eq', 'fun':const})

    result = minimize(main, w, method='SLSQP', bounds=bounds, constraints=constraints)
    
    return result.x

def Kernel_Test(a, S, S_train, g):
    error = 0
    for s in S:
        guess = 0
        for i in range(len(a)):
            guess += a[i] * S_train[i][-1] * kernel(S_train[i][:-1], s[:-1], g)
        
        if guess > 0 and s[-1] == 1:
            pass
        elif guess < 0 and s[-1] == -1:
            pass
        else:
            error += 1

    return error/float(len(S))

def kernel(x1, x2, g):
        norm = np.linalg.norm(x1-x2)
        norm = -(norm**2)
        norm /= g
        norm = math.exp(norm)
        return norm

def Count(a):
    return sum([1 for alpha in a if alpha > 0])

#Result Generating
f = open("bank-note/train.csv")

Attributes = ["Variance", "Skewness", "Curtosis", "Entropy", "Label"]

S_train = []
for line in f:
    attrs = line.strip().split(',')
    example = {}
    for i in range(0, len(Attributes)):
        if Attributes[i] == "Label":
            attrs[i] = 1 if attrs[i] == '1' else -1
        example[Attributes[i]] = float(attrs[i])
    S_train.append(example)

f = open("bank-note/test.csv")
S_test = []
for line in f:
    attrs = line.strip().split(',')
    example = {}
    for i in range(0, len(Attributes)):
        if Attributes[i] == "Label":
            attrs[i] = 1 if attrs[i] == '1' else -1
        example[Attributes[i]] = float(attrs[i])
    S_test.append(example)

Attributes.remove("Label")

test = []
train = []

print "Start generating the result...\n"

#Problem 2
print "2.(a) starts generating...\n"

f = open("data_result/2a_result.txt", "w")

C = [1, 10, 50, 100, 300, 500, 700]
for c in C:
    w = Primal_Train_SA(S_train, Attributes, C = float(c)/873, ite = 100, g = 0.01, d = 0.01) 
    err_train = Primal_Test(w, S_train, Attributes)
    err_test = Primal_Test(w, S_test, Attributes)
    train.append(round(err_train, 3))
    test.append(err_test)

strr = str(C).replace('[', '')
strr = str(strr).replace(']', '')
f.write("C\t" + str(strr).replace(', ', ' \t ') + "\n")
strr = str(train).replace('[', '')
strr = str(strr).replace(']', '')
f.write("Training Error\t" + str(strr).replace(', ', ' \t ') + "\n")
strr = str(test).replace('[', '')
strr = str(strr).replace(']', '')
f.write("Testing Error\t" + str(strr).replace(', ', ' \t ') + "\n")

print "2.(a) Complete~\n"

test = []
train = []

print "2.(b) starts generating...\n"

f = open("data_result/2b_result.txt", "w")

for c in C:
    w = Primal_Train_SB(S_train, Attributes, C = float(c)/873, ite = 100, g = 0.01, d = 0.01) 
    err_train = Primal_Test(w, S_train, Attributes)
    err_test = Primal_Test(w, S_test, Attributes)
    train.append(round(err_train, 3))
    test.append(err_test)

strr = str(C).replace('[', '')
strr = str(strr).replace(']', '')
f.write("C\t" + str(strr).replace(', ', ' \t ') + "\n")
strr = str(train).replace('[', '')
strr = str(strr).replace(']', '')
f.write("Training Error\t" + str(strr).replace(', ', ' \t ') + "\n")
strr = str(test).replace('[', '')
strr = str(strr).replace(']', '')
f.write("Testing Error\t" + str(strr).replace(', ', ' \t ') + "\n")

print "2.(b) Complete~\n"

print "2.(c) starts generating...\n"

print "Schedule A start...\n"

f = open("data_result/2c_a_result.txt", "w")

f.write("Schedule A\n")

for c in C:
    w = Primal_Train_SA(S_train, Attributes, C = float(c)/873, ite = 100, g = 0.01, d = 0.01) 
    f.write(str([round(wi, 3) for wi in w]) + "\n")
    
print "Schedule A complete...\n"

print "Schedule B starts...\n"

f = open("data_result/2c_b_result.txt", "w")

f.write("Schedule B\n")

for c in C:
    w = Primal_Train_SB(S_train, Attributes, C = float(c)/873, ite = 100, g = 0.01, d = 0.01) 
    f.write(str([round(wi, 3) for wi in w]) + "\n")

print "Schedule B complete...\n"

print "2.(c) Complete~\n"

#Problem 3
f = open("bank-note/train.csv")
S_train = []
for line in f:
    attrs = line.strip().split(',')
    example = [float(s) for s in attrs[:-1]]
    if attrs[-1] == '1':
        example.append(1)
    else:
        example.append(-1)
    S_train.append(example)

f = open("bank-note/test.csv")
S_test = []
for line in f:
    attrs = line.strip().split(',')
    example = [float(s) for s in attrs[:-1]]
    if attrs[-1] == '1':
        example.append(1)
    else:
        example.append(-1)
    S_test.append(example)

S_test_np = np.array(S_test)
S_train_np1 = np.array(S_train)
# for easy test
S_train_np2 = np.array(S_train[0:100])
S_train_np3 = np.array(S_train[0:200])

print "3.(a) starts generating...\n"

f = open("data_result/3a_result.txt", "w")
f.write("C\tb\tw\n")

C = [100, 500, 700]
for c in C:
    w, b = Dual_Train(S_train_np1, C = float(c)/873)
    f.write(str(c) + "/873\t" + str(b) + "\t" + str(w) + "\n")

print "3.(a) Complete~\n"

print "3.(b) starts generating...\n"

f = open("data_result/3b_result.txt", "w")

C = [100, 500, 700]
gamma = [0.01, 0.1, 0.5, 1, 2, 5, 10, 100]

strr = str(gamma).replace('[', '')
strr = str(strr).replace(']', '')

f.write("Training\n")
f.write("C\t" + str(strr).replace(', ', ' \t ') + "\n")

for c in C:
    f.write(str(c) + "/873")
    for g in gamma:
        a = Kernel_Train(S_train_np2, C = float(c)/873, g = g)
        train = Kernel_Test(a, S_train_np2, S_train_np2, g)
        f.write("\t" + str(train))
    f.write("\n")

f.write("\nTesting\n")
f.write("C\t" + str(strr).replace(', ', ' \t ') + "\n")

for c in C:
    f.write(str(c) + "/873")
    for g in gamma:
        a = Kernel_Train(S_train_np2, C = float(c)/873, g = g)
        test = Kernel_Test(a, S_test_np, S_train_np2, g)
        f.write("\t" + str(test))
    f.write("\n")

print "3.(b) Complete~\n"

print "3.(c) starts generating...\n"

f = open("data_result/3c_result.txt", "w")

gamma = [0.01, 0.1, 0.5, 1, 2, 5, 10, 100]

strr = str(gamma).replace('[', '')
strr = str(strr).replace(']', '')

f.write("gamma\t" + str(strr).replace(', ', ' \t ') + "\n")
f.write("count")
temp_a = []
for g in gamma:
    a = Kernel_Train(S_train_np1, C = 500.0/873, g = g)
    count = 0
    for i in range(len(temp_a)):
        if a[i] > 0 and temp_a[i] > 0:
            count += 1
    f.write("\t" + str(float(count)/len(a)))
    temp_a = a

print "3.(c) Complete~\n"

print "All Complete!"