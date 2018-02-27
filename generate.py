from utilities import Preprocessor, charsToVec
import pickle
import numpy as np


def getStartingPoint():
    s = "name:\n"
    n = input("Enter recipe name!\n").strip().lower()
    return s+n+'\n\ningredients:\n\n'



def predictNext(txts, probs, prep, depth, maxDepth, beamSize):
    size = beamSize*prep.maxSize
    X1 = np.zeros((size, prep.maxLen))
    X2 = np.zeros((size, prep.maxLenHist))
    newtxts = []
    for n, txt in enumerate(txts):
        n2 = n*len(prep.charDict)
        for char in prep.charDict.keys():
            txt2 = txt+char
            x1 = charsToVec(txt2, 0, len(txt2), prep.maxLen, prep.charDict)
            x2 = charsToVec(txt2, 0, len(txt2), prep.maxLenHist,
                            prep.charDict)
            X1[n2] = x1
            X2[n2] = x2
            n2 += 1
            newtxts.append(txt2)