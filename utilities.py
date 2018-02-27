from keras.preprocessing.sequence import pad_sequences
import numpy as np


def charsToVec(recipe, start, end, l, charDict):
    s = [[charDict[i] for i in recipe[start:end]]]
    return pad_sequences(s, maxlen=l)[0]


def decode_prediction(inpString, prep, model):
    x1 = charsToVec(inpString, 0, len(inpString),
                    prep.maxLen, prep.charDict)
    x2 = charsToVec(inpString, 0, len(inpString),
                    prep.maxLenHist, prep.charDict)
    x = [np.array([x1]), np.array([x2])]
    p = model.predict(x)[0]
    if len(p.shape) > 1:
        p = p[0]
    probs = [(prep.charRev[k], p[k]) for k in prep.charRev.keys()]
    print(sorted(probs, key=lambda x: x[1], reverse=True))


def addOneChar(prep, x, char):
    """
    Add a single character to the matrix representation of a string
    """
    val = prep.charDict[char]
    return np.append(x[1:], val)

class Preprocessor():
    
    def __init__(self, maxLen=30, maxLenHist=600, predDepth=3):
        self.maxLen = maxLen
        self.maxLenHist = maxLenHist
        self.charDict = {'': 0}
        self.predDepth = predDepth
        self.startString = "\nname:\n{}\n\ningredients:\n\n"

    def sample(self, a, temperature=1.0):
        # helper function to sample an index from a probability array
        a = np.log(a) / temperature
        a = np.exp(a) / np.sum(np.exp(a))
        a = a.astype(float) -0.00001
        #print(np.sum(a[:-1]))
        return np.argmax(np.random.multinomial(1, a, 1))

    def get_recipe(self, name, model, method="topN", **kwargs):
        s = self.startString.format(name)
        if method == "topN":
            return self.get_recipe_top_N(s, model, kwargs)

    def get_recipe_top_N(self, rec, model, kwargs):
        N = kwargs["N"]
        T = kwargs["temperature"]
        while len(rec) < 2000 and rec[-1] != '$':
            x1 = charsToVec(rec, 0, len(rec),
                    self.maxLen, self.charDict)
            x2 = charsToVec(rec, 0, len(rec),
                    self.maxLenHist, self.charDict)
            p = model.predict([np.array([x1]),
                               np.array([x2])])[0][0]
            part = np.argpartition(-p, N)[:N]
            probs = [p[i] for i in part]
            print(part, probs)
            ind = self.sample(probs, temperature=T)
            nextInd = part[ind]
            nextChar = self.charRev[nextInd]
            print(nextInd, nextChar)
            rec += nextChar
            print(rec)

def get_recipe_beam(prep, rec, model, kwargs):
    beamWidths = kwargs["beamWidth"]
    seqLength = kwargs["seqLength"]
    if type(beamWidths) != list:
        beamWidths = [beamWidths]*seqLength
    x2 = charsToVec(rec, 0, len(rec),
                    prep.maxLenHist, prep.charDict)
    x1 = x2[-prep.maxLen:]
    while len(rec) < 2000 and rec[-1] != '$':
        currentx1 = x1
        currentx2 = x2
        for r in range(0, seqLength):
            # init input (beamWidth*num_candidates)
            fullx1 = np.zeroes(len(currentx1)*len(prep.charDict),
                               x1.shape[0])
            fullx2 = np.zeroes(len(currentx1)*len(prep.charDict),
                               x2.shape[0])
            
            # create examples for each candidate
            for n in range(0, len(currentx1)):
                for n2, char in enumerate(prep.charDict.keys()):
                    x1 = currentx1[n]
                    x2 = currentx2[n]
                    ind = n*len(prep.charDict) + n2
                    x1New = addOneChar(prep, x1, char)
                    x2New = addOneChar(prep, x2, char)
                    fullx1[ind, :] = x1New
                    fullx2[ind, :] = x2New

            # get probabilities of each candidate
            p = model.predict([fullx1, fullx2])[0][0]
            # partition, get top indices of top N
            topInds = np.partition()
            # slice the x1, x2 arrays with the top indices
            currentx1 = []
            currentx2 = []