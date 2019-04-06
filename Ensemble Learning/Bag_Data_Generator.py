## ZW
from Functions_bagging import *
import random
import numpy

def GetMedian(arr):
    n = len(arr)
    if n < 1:
            return None
    if n % 2 == 1:
            return sorted(arr)[n//2]
    else:
            return sum(sorted(arr)[n//2-1:n//2+1])/2.0

def reset_weights(S):
    weight = 1/float(len(S))
    for s in S:
        s["Weight"] = weight

def AdaBoost_Error(S_train, S_test, Attributes, T):
    trees = []
    alphas = []
    f = open("Adaboost_data_error.txt", "w")
    for _ in range(0, T):
        tree = ID3_weight(S_train, Attributes, 1, 0)
        trees.append(tree)

        norm = 0.0        
        epsilon = ID3_weight_err(tree, S_train)
        f.write(str(_) + "\t" + str(epsilon) + "\t" + str(ID3_weight_err(tree, S_test)) + "\n")
        alpha = .5 * math.log((1 - epsilon)/epsilon)
        alphas.append(alpha)
        for s in S_train:
            label = GetLabel(s, tree)
            if label != s["Label"]:
                newWeight = s["Weight"] * math.exp(alpha)
                s["Weight"] = newWeight
            else:
                newWeight = s["Weight"] * math.exp(-alpha)
                s["Weight"] = newWeight
            norm += newWeight

        for s in S_train:
            s["Weight"] /= norm

    return (trees, alphas)

attrFile = open("bank" + "/data-desc.txt")
attrFile.readline()
attrFile.readline()
labels = "".join(attrFile.readline().split()).split(',')

attrFile.readline()
attrFile.readline()
attrFile.readline()

Attributes = {}
attrList = []

line = attrFile.readline()
while line != "\n":
    splitLine = line.split(':')
    attr = splitLine[0]
    attrList.append(attr)
    attrVals = "".join(splitLine[1].split()).split(',')
    Attributes[attr] = attrVals
    line = attrFile.readline()

attrList.append("Label")

numericList = [A for A in attrList if A in Attributes and Attributes[A][0] == "(numeric)"]

S_train = []
numericalLists = {}
with open("bank" + "/train.csv") as f:
    for line in f:
        i = 0
        example = {}
        for attr in line.strip().split(','):
            attrName = attrList[i]
            if attrName in numericList:
                if attrName not in numericalLists:
                    numericalLists[attrName] = []
                numericalLists[attrName].append(float(attr))
            example[attrList[i]] = attr
            i += 1
        S_train.append(example)

medianList = {}
for name, arr in numericalLists.items():
    medianList[name] = GetMedian(arr)

for s in S_train:
    for attr in numericList:
        if s[attr] >= numericalLists[attr]:
            s[attr] = "1"
        elif s[attr] < numericalLists[attr]:
            s[attr] = "-1"
    s["Weight"] = 1/float(len(S_train))

S_test = []
with open("bank" + "/test.csv") as f:
    for line in f:
        i = 0
        example = {}
        for attr in line.strip().split(','):
            name = attrList[i]
            if name in numericList:
                val = float(attr)
                if val >= numericalLists[name]:
                    attr = "1"
                elif val < numericalLists[name]:
                    attr = "-1"
            example[name] = attr
            example["Weight"] = 1/5000.0
            i += 1
        S_test.append(example)

for attr in numericList:
    Attributes[attr] = ["-1", "1"]

print "Problem 2 tests start..."

f = open("Adaboost_data.txt", "w")
f.write("Data for 2.(a)\n")
f.write("Iteration\tTrain\tTest\n")
for T in range(1, 1050, 100):
    hypothesis = AdaBoost(S_train, Attributes, T)
    err_train = AdaBoost_Test(hypothesis, S_train)
    err_test = AdaBoost_Test(hypothesis, S_test)
    f.write(str(T-1) + "\t" + str(err_train) + "\t" + str(err_test) + "\n")
    reset_weights(S_train)

hypothesis = AdaBoost_Error(S_train, S_test, Attributes, 1000)

print "2.(a) data generating complete!"

f = open("bagging_data.txt", "w")
f.write("Data for 2.(b)\n")
f.write("Iteration\tTrain\tTest\n")
for T in range(1, 1050, 100):
    hypothesis = Bagging_Train(S_train, Attributes, T)
    err_train = Bagging_Test(hypothesis, S_train)
    err_test = Bagging_Test(hypothesis, S_test)
    f.write(str(T-1) + "\t" + str(err_train) + "\t" + str(err_test) + "\n")
    reset_weights(S_train)

print "2.(b) data generating complete!"

f = open("Bias&Variance_data.txt", "w")
f.write("Data for 2.(c)\n")

predictors = []
for _ in range(0, 100):
    copy_S = list(S_train)
    new_S = []
    for i in range(0, 1000):
        rand = random.randint(0, len(copy_S) - 1)
        new_S.append(copy_S[rand])
        del copy_S[rand]
    predictor = Bagging_Train(new_S, Attributes, 1000)
    predictors.append(predictor)

total_bias = 0.0
total_variance = 0.0
for s in S_test:
    avg = 0.0
    predictions = []
    for p in predictors:
        label = GetLabel(s, p[0][0])
        val = 1 if label == "yes" else -1
        avg += val
        predictions.append(val)
    avg /= len(predictors)
    label_num = 1 if s["Label"] == "yes" else -1

    bias = pow(label_num - avg, 2)
    total_bias += bias

    variance = numpy.var(predictions)
    total_variance += variance

bias = total_bias/len(S_test)
variance = total_variance/len(S_test)

f.write("Single bias: " + str(bias) + "\n")
f.write("Single variance: " + str(variance) + "\n")

T = 0
for s in S_test:
    T += 1
    avg = 0.0
    predictions = []
    for p in predictors:
        val = GetBagLabel(p, s) / float(len(p[0]))
        avg += val
        predictions.append(val)
    avg /= len(predictors)
    label_num = 1 if s["Label"] == "yes" else -1

    bias = pow(label_num - avg, 2)
    total_bias += bias

    variance = numpy.var(predictions)
    total_variance += variance

bias = total_bias/len(S_test)
variance = total_variance/len(S_test)

f.write("Mass bias: " + str(bias) + "\n")
f.write("Mass variance: " + str(variance) + "\n")

print "2.(c) data generating complete!"

f = open("forest_data.txt", "w")
f.write("Data for 2.(d)" + "\n")
f.write("N=2\n")
f.write("Iteration\tTrain\tTest\n")
for T in range(1, 1050, 100):
    hypothesis = Forest_Train_R(S_train, Attributes, T, 2)
    err_train = Forest_Test_R(hypothesis, S_train)
    err_test = Forest_Test_R(hypothesis, S_test)
    f.write(str(T-1) + "\t" + str(err_train) + "\t" + str(err_test) + "\n")
    reset_weights(S_train)

f.write("\nN=4")
f.write("Iteration\tTrain\tTest\n")
for T in range(1, 1050, 100):
    hypothesis = Forest_Train_R(S_train, Attributes, T, 4)
    err_train = Forest_Test_R(hypothesis, S_train)
    err_test = Forest_Test_R(hypothesis, S_test)
    f.write(str(T-1) + "\t" + str(err_train) + "\t" + str(err_test) + "\n")
    reset_weights(S_train)

f.write("\nN=8")
f.write("Iteration\tTrain\tTest\n")
for T in range(1, 1050, 100):
    hypothesis = Forest_Train_R(S_train, Attributes, T, 8)
    err_train = Forest_Test_R(hypothesis, S_train)
    err_test = Forest_Test_R(hypothesis, S_test)
    f.write(str(T-1) + "\t" + str(err_train) + "\t" + str(err_test) + "\n")
    reset_weights(S_train)

print "2.(d) data generating complete!"

f = open("Bias&Variance_data.txt", "w")
f.write("Data for 2.(e)\n")

single_predictions = len(S_test)*[None]
mass_predictions = len(S_test)*[None]

for s in range(0, len(S_test)):
    single_predictions[s] = []
    mass_predictions[s] = []

copy_S = list(S_train)
for i in range(0, 100):
    random.shuffle(copy_S)
    new_S = copy_S[:1000]
    predictor = Forest_Train_R(new_S, Attributes, 1000, 4)

    for s in range(0, len(S_test)):
        label = GetLabel(S_test[s], predictor[0][0])
        val = 1 if label == "yes" else -1
        single_predictions[s].append(val)

    for s in range(0, len(S_test)):
        val = GetBagLabel(predictor, S_test[s])/float(len(predictor[0]))
        mass_predictions[s].append(val)

    print "predictor " + str(i) + " done"

for s in range(0, len(S_test)):
    label_num = 1 if S_test[s]["Label"] == "yes" else -1
    avg_prediction = sum(single_predictions[s])/len(single_predictions[s])
    bias = pow(label_num - avg_prediction, 2)
    total_bias += bias

    total_variance += numpy.var(single_predictions[s])

bias = total_bias/100
variance = total_variance/100

f.write("Single bias: " + str(bias) + "\n")
f.write("Single variance: " + str(variance) + "\n")

for s in range(0, len(S_test)):
    label_num = 1 if S_test[s]["Label"] == "yes" else -1
    avg_prediction = sum(mass_predictions[s])/len(mass_predictions[s])
    bias = pow(label_num - avg_prediction, 2)
    total_bias += bias

    total_variance += numpy.var(mass_predictions[s])

bias = total_bias/100
variance = total_variance/100

f.write("Mass bias: " + str(bias) + "\n")
f.write("Mass variance: " + str(variance) + "\n")

print "2.(e) data generating complete!"
print "All data generating complete!"