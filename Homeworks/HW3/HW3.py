import random

def Average(S, Attributes, LR, MaxEpochs):
    w = [0 for _ in range(0, len(Attributes))]
    a = list(w)
    for _ in range(0, MaxEpochs):
        random.shuffle(S)

        for s in S:
            guess = 0.0
            for i in range(0, len(w)):
                guess += float(s[Attributes[i]]) * w[i]
            guess *= s["Label"]
            if guess <= 0:
                for i in range(0, len(w)):
                    w[i] += float(s[Attributes[i]]) * s["Label"] * LR
            for i in range(0, len(w)):
                    a[i] += w[i]
            
    return a

def Standard(S, Attributes, LR, MaxEpochs):
    w = [0 for _ in range(0, len(Attributes))]
    for _ in range(0, MaxEpochs):
        random.shuffle(S)

        for s in S:
            guess = 0.0
            for i in range(0, len(w)):
                guess += float(s[Attributes[i]]) * w[i]
            guess *= s["Label"]
            if guess <= 0:
                for i in range(0, len(w)):
                    w[i] += float(s[Attributes[i]]) * s["Label"] * LR
            
    return w

def Standard_Test(S, Attributes, w):
    error = 0
    for s in S:
        guess = 0.0
        for i in range(0, len(w)):
            guess += float(s[Attributes[i]]) * w[i]
        
        if guess > 0 and s["Label"] == 1:
            pass
        elif guess < 0 and s["Label"] == -1:
            pass
        else:
            error += 1

    return error/float(len(S))

def Voted(S, Attributes, LR, MaxEpochs):
    w = [0 for _ in range(0, len(Attributes))]
    C = []
    W = []
    c = 1
    for _ in range(0, MaxEpochs):
        random.shuffle(S)

        for s in S:
            guess = 0.0
            for i in range(0, len(w)):
                guess += float(s[Attributes[i]]) * w[i]
            guess *= s["Label"]
            if guess <= 0:
                if w[0] != 0 and w[-1] != 0:
                    W.append(list(w))
                    C.append(c)
                c = 1
                for i in range(0, len(w)):
                    w[i] += float(s[Attributes[i]]) * s["Label"] * LR
            else:
                c += 1
            
    return W, C

def Voted_Test(S, Attributes, wc):
    error = 0
    for s in S:
        guess = 0.0
        for i in range(0, len(wc[0])):
            w = wc[0][i]
            c = wc[1][i]
            tmp_guess = 0.0
            for j in range(0, len(w)):
                tmp_guess += w[j] * float(s[Attributes[j]])
            tmp_guess = 1 if tmp_guess > 0 else -1
            guess += tmp_guess * c                
        
        if guess > 0 and s["Label"] == 1:
            pass
        elif guess < 0 and s["Label"] == -1:
            pass
        else:
            error += 1

    return error/float(len(S))

f = open("bank-note/train.csv")

Attributes = ["Variance", "Skewness", "Curtosis", "Entropy", "Label"]

train = []
for line in f:
    attrs = line.strip().split(',')
    e = {}
    for i in range(0, len(Attributes)):
        if Attributes[i] == "Label":
            attrs[i] = 1 if attrs[i] == '1' else -1
        e[Attributes[i]] = attrs[i]
    train.append(e)

f = open("bank-note/test.csv")
test = []
for line in f:
    attrs = line.strip().split(',')
    e = {}
    for i in range(0, len(Attributes)):
        if Attributes[i] == "Label":
            attrs[i] = 1 if attrs[i] == '1' else -1
        e[Attributes[i]] = attrs[i]
    test.append(e)

Attributes.remove("Label")

learning_rate = 0.1
MaxEpochs = 10

print "Start generating the result data...\n"
f = open("data_result/HW3_result.txt", "w")

f.write("2.(a) Standard Perceptron\n")
w = Standard(train, Attributes, learning_rate, MaxEpochs)
train_error = Standard_Test(train, Attributes, w)
test_error = Standard_Test(test, Attributes, w)
f.write("\nLearning rate: " + str(learning_rate))
f.write("\nLearned Weight Vector: " + str(w))
f.write("\nTesting Error: " + str(test_error))
f.write("\nTraining Error: " + str(train_error))

print "Standard result complete!"

f.write("\n\n2.(b) Voted Perceptron\n")
wc = Voted(train, Attributes, learning_rate, MaxEpochs)
train_error = Voted_Test(train, Attributes, wc)
test_error = Voted_Test(test, Attributes, wc)
f.write("\nLearning rate: " + str(learning_rate))
f.write("\n\nLearned Weight Vector:\n")
f.write("Votes\t\tWeight Vector\n")
for w, c in zip(wc[0], wc[1]):
    f.write("Votes = " + str(c) + ",\tw = " + str(w) + "\n")
f.write("\nTesting Error: " + str(test_error))
f.write("\nTraining Error: " + str(train_error))

print "Voted result complete!"

f.write("\n\n2.(c) Average Perceptron\n")
w = Average(train, Attributes, learning_rate, MaxEpochs)
train_error = Standard_Test(train, Attributes, w)
test_error = Standard_Test(test, Attributes, w)
f.write("\nLearning rate: " + str(learning_rate))
f.write("\nLearned Weight Vector: " + str(w))
f.write("\nTesting Error: " + str(test_error))
f.write("\nTraining Error: " + str(train_error))

print "Average result complete!"
print "All result generating complete!"