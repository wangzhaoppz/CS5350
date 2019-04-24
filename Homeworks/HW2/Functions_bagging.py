from ID3 import *
import math
import random

def avg_error(Hypothesis, S):
    total = 0.0
    for tree in Hypothesis[0]:
        temp = 0
        for s in S:
            label = GetLabel(s, tree)
            if s["Label"] != label:
                temp += 1
        total += temp/float(len(S))

    return total/float(len(Hypothesis[0]))

def GetBagLabel(Hypothesis, s):
    prediction = 0.0
    for tree, weight in zip(Hypothesis[0], Hypothesis[1]):
            label = GetLabel(s, tree)
            label = 1 if label == "yes" else -1
            prediction += label * weight

    return prediction

def Forest_Train_R(S, Attributes, T, num_features):
    M = len(S)/2
    predictions = []
    weights = []
    for _ in range(0, T):
        new_S = [random.choice(S) for __ in range(0, M)]
        tree = ID3_R(new_S, Attributes, num_features)
        predictions.append(tree)
        weights.append(1)

    return predictions, weights

def Forest_Test_R(Hypothesis, S):
    error = 0
    for s in S:
        prediction = 0
        for tree , weight in zip(Hypothesis[0], Hypothesis[1]):
            label = GetLabel(s, tree)
            label = 1 if label == "yes" else -1
            prediction += label * weight

        if s["Label"] == "yes" and prediction > 0:
            pass
        elif s["Label"] == "no" and prediction < 0:
            pass
        else:
            error += 1

    return error/float(len(S))

def Bagging_Train(S, Attributes, T):
    M = len(S)/2
    predictions = []
    weights = []
    for _ in range(0, T):
        new_S = [random.choice(S) for __ in range(0, M)]

        tree = ID3_weight(new_S, Attributes, None, 0)
        predictions.append(tree)
        weights.append(1)

    return predictions, weights

def Bagging_Test(Hypothesis, S):
    wrong = 0
    for s in S:
        prediction = 0
        for tree , weight in zip(Hypothesis[0], Hypothesis[1]):
            label = GetLabel(s, tree)
            label = 1 if label == "yes" else -1
            prediction += label * weight

        if s["Label"] == "yes" and prediction > 0:
            pass
        elif s["Label"] == "no" and prediction < 0:
            pass
        else:
            wrong += 1

    return wrong/float(len(S))

def AdaBoost(S, Attributes, T):
    trees = []
    alphas = []
    for _ in range(0, T):
        tree = ID3_weight(S, Attributes, 1, 0)
        trees.append(tree)

        norm = 0.0        
        epsilon = ID3_weight_err(tree, S)
        alpha = .5 * math.log((1 - epsilon)/epsilon)
        alphas.append(alpha)
        for s in S:
            label = GetLabel(s, tree)
            if label != s["Label"]:
                newWeight = s["Weight"] * math.exp(alpha)
                s["Weight"] = newWeight
            else:
                newWeight = s["Weight"] * math.exp(-alpha)
                s["Weight"] = newWeight
            norm += newWeight

        for s in S:
            s["Weight"] /= norm

    return (trees, alphas)

def AdaBoost_Test(Hypothesis, S):
    error = 0
    for s in S:
        prediction = 0
        for tree, weight in zip(Hypothesis[0], Hypothesis[1]):
            label = GetLabel(s, tree)
            label = 1 if label == "yes" else -1
            prediction += label * weight

        if s["Label"] == "yes" and prediction > 0:
            pass
        elif s["Label"] == "no" and prediction < 0:
            pass
        else:
            error += 1

    return error/float(len(S))

