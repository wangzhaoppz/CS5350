## ZW
import random

def GetCost(S, Attributes, w):
    cost = 0.0
    for s in S:
        vals = [s[attr] for attr in Attributes]
        prediction = 0.0
        for k in range(0, len(w)):
            prediction += w[k] * vals[k]
        label = s["Label"]
        error = label - prediction
        cost += pow(error, 2)

    return cost/2

def LMS_B(S, Attributes, test, R, Convergence):
    w = [0 for _ in range(0, len(Attributes))]
    f = open("batch_data.txt", "w")
    norm = 1000
    T = 0
    f.write("Data for 4.(a)\n")
    f.write("Iteration\tCost\n")
    while norm > Convergence:
        cost = 0.0
        J_grad = []
        for j in range(0, len(Attributes)):
            temp = 0.0
            for s in S:
                vals = [s[attr] for attr in Attributes]
                prediction = 0.0
                for k in range(0, len(w)):
                    prediction += w[k] * vals[k]
                label = s["Label"]
                error = label - prediction
                x_ij = s[Attributes[j]]
                temp += error * x_ij
            J_grad.append(-temp)

        cost = GetCost(S, Attributes, w)
        f.write(str(T) + "\t" + str(cost/2) + "\n")
        T += 10

        new_w = []
        for i in range(0, len(w)):
            new_w.append(w[i] - (R * J_grad[i]))

        norm = -1
        for i in range(0, len(w)):
            gap = abs(w[i] - new_w[i])
            if gap > norm:
                norm = gap

        w = new_w

    w = [round(x,8) for x in w]
    f.write("Final W: " + str(w) + "\n")
    testCost = GetCost(test, Attributes, w)
    f.write("Test cost of batch is: " + str(testCost) + ".\n")
    print "Batch test complete!"
    return
    

def LMS_S(S, Attributes, test, R, Convergence):
    w = [0 for _ in range(0, len(Attributes))]
    f = open("stch_data.txt", "w")
    f.write("Data for 4.(b)\n")
    f.write("Iteration\tCost\n")
    norm = 1
    T = 0
    while norm > Convergence:
        storedW = list(w)
        for i in range(0, len(S)):
            new_w = []
            for j in range(0, len(Attributes)):
                vals = [S[i][attr] for attr in Attributes]
                prediction = 0.0
                for k in range(0, len(w)):
                    prediction += w[k] * vals[k]
                label = S[i]["Label"]
                error = label - prediction
                x_ij = S[i][Attributes[j]]
                adjustment = error * x_ij * R
                new_w.append(w[j] + adjustment)
            w = new_w

        cost = GetCost(S, Attributes, w)
        f.write(str(T) + "\t" + str(cost/2) + "\n")
        T += 10

        norm = -1
        for i in range(0, len(w)):
            diff = abs(storedW[i] - w[i])
            if diff > norm:
                norm = diff

    w = [round(x,8) for x in w]
    f.write("Final W: " + str(w) + "\n")
    testCost = GetCost(test, Attributes, w)
    f.write("Test cost of Stoch is: " + str(testCost) + ".\n")
    print "Stoch test complete!"
    return

Attributes = ["Cement", "Slag", "Fly ash", "Water", "SP", "Coarse Aggr", "Fine Aggr", "Label"]

train = []
with open("concrete" + "/train.csv") as f:
    for line in f:
        i = 0
        example = {}
        for attr in line.strip().split(','):
            example[Attributes[i]] = float(attr)
            i += 1
        train.append(example)

test = []
with open("concrete" + "/test.csv") as f:
    for line in f:
        i = 0
        example = {}
        for attr in line.strip().split(','):
            example[Attributes[i]] = float(attr)
            i += 1
        test.append(example)

Attributes.remove("Label")

print "Problem 4 tests start..."

LMS_B(train, Attributes, test, 0.0145, 0.000001)
LMS_S(train, Attributes, test, 0.1, 0.000001)

print "All data generating complete!"
