import math
import random

def GetLength(S):
    length = 0
    for s in S:
        length += s["Weight"]
    return length

def GetSv(S, A, v):
    return [s for s in S if s[A] == v]

def GetValues(Attributes, A):
    return Attributes[A]

def GetLabel(S, Tree):
    if Tree.prediction != "":
        return Tree.prediction
    
    newTree = None
    for node in Tree.children:
        if node.label == S[Tree.splitsOn]:
            newTree = node
            break

    return GetLabel(S, newTree)

def GetIG(S, Attributes, A, entropy):
    newEnt = 0.0
    for v in Attributes[A]:
        Sv = GetSv(S, A, v)
        ratio = GetLength(Sv)/float(GetLength(S))
        ent = GetEntropy(Sv)
        newEnt += ratio * ent

    return entropy - newEnt

def CommonLeaf(S, v = None):
    labels = {}
    for s in S:
        label = s["Label"]
        if label not in labels:
            labels[label] = 0.0
        labels[label] += s["Weight"]

    maxNum = 0
    maxLabel = ""
    for label, num in labels.items():
        if num > maxNum:
            maxNum = num
            maxLabel = label

    leaf = Node()
    leaf.prediction = maxLabel
    leaf.label = v
    return leaf

def GetEntropy(S):
    if len(S) == 0:
        return 0

    labels = {}
    for s in S:
        label = s["Label"]
        if label not in labels:
            labels[label] = 0.0
        labels[label] += s["Weight"]

    entropy = 0.0
    norm = GetLength(S)
    for (label, quant) in labels.items():
        ratio = quant/float(norm)
        entropy -= math.log((ratio), 2) * (ratio)

    return entropy

def ID3_weight(S, Attributes, MaxDepth, depth):
    if depth == MaxDepth:
        return CommonLeaf(S)

    labelCheck = S[0]["Label"]
    allSame = True
    for s in S:
        if s["Label"] != labelCheck:
            allSame = False
            break

    if allSame:
        leaf = Node()
        leaf.prediction = labelCheck
        return leaf

    if len(Attributes) == 0:
        return CommonLeaf(S)

    A = IG_N(S, Attributes)

    root = Node()
    root.splitsOn = A

    for v in Attributes[A]:
        Sv = GetSv(S, A, v)

        if len(Sv) == 0:
            leaf = CommonLeaf(S, v)
            root.children.append(leaf)
        else:
            tempAttr = dict(Attributes)
            tempAttr.pop(A)
            subtree = ID3_weight(Sv, tempAttr, MaxDepth, depth + 1)
            subtree.label = v
            root.children.append(subtree)

    return root

def ID3_weight_err(Tree, S):
    error = 0.0
    for s in S:
        label = GetLabel(s, Tree)
        if label != s["Label"]:
            error += s["Weight"]
    return error

def ID3_R(S, Attributes, NumFeatures):
    labelCheck = S[0]["Label"]
    allSame = True
    for s in S:
        if s["Label"] != labelCheck:
            allSame = False
            break

    if allSame:
        leaf = Node()
        leaf.prediction = labelCheck
        return leaf

    if len(Attributes) == 0:
        return CommonLeaf(S)

    A = IG_R(S, Attributes, NumFeatures)

    root = Node()
    root.splitsOn = A

    for v in Attributes[A]:
        Sv = GetSv(S, A, v)

        if len(Sv) == 0:
            leaf = CommonLeaf(S, v)
            root.children.append(leaf)
        else:
            tempAttr = dict(Attributes)
            tempAttr.pop(A)
            subtree = ID3_R(Sv, tempAttr, NumFeatures)
            subtree.label = v
            root.children.append(subtree)

    return root

def IG_N(S, Attributes):
    entropy = GetEntropy(S)

    maxInfo = -1
    maxAttr = ""
    for A in Attributes:
        IG = GetIG(S, Attributes, A, entropy)
        if IG > maxInfo:
            maxInfo = IG
            maxAttr = A

    return maxAttr

def IG_R(S, Attributes, NumFeatures):
    entropy = GetEntropy(S)

    newAttrs = []
    cpyAttrs = list(Attributes)
    for _ in range(0, NumFeatures):
        if len(cpyAttrs) == 0:
            break
        rand = random.randint(0, len(cpyAttrs)-1)
        newAttrs.append(cpyAttrs[rand])
        del cpyAttrs[rand]

    maxInfo = -1
    maxAttr = ""
    for A in newAttrs:
        IG = GetIG(S, Attributes, A, entropy)
        if IG > maxInfo:
            maxInfo = IG
            maxAttr = A

    return maxAttr

class Node:
    children = list()
    label = ""
    splitsOn = ""
    prediction = ""

    def __init__(self):
        self.children = list()
        self.label = ""
        self.splitsOn = ""
        self.prediction = ""