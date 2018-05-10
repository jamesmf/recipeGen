from keras.preprocessing.sequence import pad_sequences
import numpy as np
import re
import time


def charsToVec(recipe, start, end, l, charDict):
    """
    Convenience function for converting a chunk of a recipe string into an
    example x1 or x2 row.
    Args:
        recipe (str): recipe string
        start (int): starting ind (often 0)
        end (int): ending ind (often char_to_predict-1)
        l (int): maxLength or maxLengthHist from Preprocessor object
        charDict: Preprocessor's charDict mapping from char: int
    Returns:
        (ndarray): single sequence of character ints, left-padded
    """
    s = [[charDict[i] for i in recipe[start:end]]]
    return pad_sequences(s, maxlen=l)[0]


def _decode_prediction(inpString, prep, model):
    """
    Convenience function for decoding a prediction
    """
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
    Args:
        prep (Preprocessor): Preprocessor object used for its charDict
        x (array): current array; sequence of integers representing chars
        char (str): character to append to the sequence
    Returns:
        (array): new sequence
    """
    val = prep.charDict[char]
    return np.append(x[1:], val)

class Preprocessor():

    nameRegex = re.compile("name:\n(.+)?\n")
    
    def __init__(self, maxLen=30, maxLenHist=600, predDepth=5):
        self.maxLen = maxLen
        self.maxLenHist = maxLenHist
        self.charDict = {'': 0}
        self.predDepth = predDepth
        self.startString = "\nname:\n{}\n\ningredients:\n\n"

    def sample(self, a, temperature=1.0):
        """
        helper function to sample an index from a probability array
        """
        a = np.log(a) / temperature
        a = np.exp(a) / np.sum(np.exp(a))
        a = a.astype(float) -0.00001
        #print(np.sum(a[:-1]))
        return np.argmax(np.random.multinomial(1, a, 1))

    def get_recipe(self, name, model, method="topN", **kwargs):
        """
        Wrapper around the various methods of generating recipes by sampling
        from the generator's distribution
        """
        s = self.startString.format(name)
        if method == "topN":
            return self.get_recipe_top_N(s, model, kwargs)

    def get_recipe_top_N(self, rec, func, kwargs):
        """
        Method for generating a recipe. Gets the prediction for char_0,
        restricts to the top N most likely characters, then samples with a
        given temperature and appends the sampled character. Does so until
        the recipe is >800 characters long or the STOPTOKEN '$' is reached
        """
        N = kwargs["N"]
        T = kwargs["temperature"]
        x2 = None
        while len(rec) < 800 and rec[-1] != '$':
            if x2 is None:
                x2 = charsToVec(rec, 0, len(rec),
                                self.maxLenHist, self.charDict)
            x1 = x2[-self.maxLen:]
            
            p = func([np.array([x1]), np.array([x2])])[0][0]
#            p = np.random.rand(len(self.charDict))
            part = np.argpartition(-p, N)[:N]
            probs = [p[i] for i in part]
            ind = self.sample(probs, temperature=T)
            nextInd = part[ind]
            nextChar = self.charRev[nextInd]
#            print(nextInd, nextChar)
            rec += nextChar
            x2 = np.append(x2, [nextInd])[1:]
#            print(rec)
        return rec

    def get_many_recipes(self, recs, func, N=10, temp=1, chunksize=64):
        """
        Takes an input set of recipe stubs and generates full recipes out of
        them. Uses the logic of get_recipe_top_N. Doing this in large chunks
        increases efficiency
        """
        # if chunksize > len(recs), we only do one 'chunk'
        if chunksize > len(recs):
            chunksize = len(recs)
        # we want the recipes to come out in order so we maintain an ordering:
        # the keys are index in current array, the values are order in inp/out
        recMap = {i: i for i in range(0, chunksize)}
        recsCurr = recs[:chunksize]
        recsOut = ['' for r in recs]
        doneInd = 0
        stopnum = len(recs)
        x2 = np.zeros((chunksize, self.maxLenHist))
        x1 = np.zeros((chunksize, self.maxLen))
        t1 = time.time()
        for i in range(0, chunksize):
            x2[i, :] = charsToVec(recs[i], 0, len(recs[i]), self.maxLenHist,
                                  self.charDict)
            x1[i, :] = x2[i, -self.maxLen:]
        while doneInd < stopnum:
            # send the recipes to the model
            p = func([x1, x2])

            # i indexes us into the recsCurr list, which contains the recipes
            # we are currently working on
            for i in range(0, chunksize):
                # get the top N probability chars for recipe i in the array
                part = np.argpartition(-p[i], N)[:N]
                probs = [p[i][j] for j in part]
                # sample from those top N characters, append to recipe i
                ind = self.sample(probs, temperature=temp)
                nextChar = self.charRev[part[ind]]
                recsCurr[i] += nextChar
                x2[i, :] = addOneChar(self, x2[i, :], nextChar)
                x1[i, :] = addOneChar(self, x1[i, :], nextChar)
                if len(recsCurr[i]) > 800 or nextChar == '$':
                    # if i not in recMap, it represents a finished recipe
                    # this will only occur when we're out of recipes to replace
                    if i in recMap:
                        outnum = recMap[i]
                        # terminate: add to recsOut, replace in matrix and recsCurr
                        recsOut[outnum] = recsCurr[i]+'$'
                        doneInd += 1
                    if doneInd+chunksize <= stopnum:
                        # if we have another recipe to start on, insert it
                        recsCurr[i] = recs[doneInd+chunksize-1]
                        # update the map
                        recMap[i] = doneInd+chunksize-1
                        x2[i, :] = charsToVec(recsCurr[i], 0, len(recsCurr[i]),
                                              self.maxLenHist, self.charDict)
                        x1[i, :] = x2[i, -self.maxLen]
                    else:
                        # if this recipe is done, we remove it from recMap
                        if i in recMap:
                            del recMap[i]
                        recsCurr[i] = ' '
        print("time to generate {} recipes".format(stopnum))
        print(time.time() - t1)
        return recsOut

    def name_from_text(self, txt):
        """
        Parse name out of a recipe string
        """
        name = self.nameRegex.search(txt)
        name = name.group(1)
        return name


def get_recipe_beam(prep, rec, model, kwargs):
    """
    STUB: unfinished
    """
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