# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 23:30:26 2018

@author: jmf
"""
import pickle
import numpy as np
from keras.layers import Input, Dense, Conv1D, Embedding, Flatten
from keras.layers import MaxPooling1D, Add, Dropout
from keras.models import Model, load_model
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
from utilities import Preprocessor, charsToVec
from sklearn.utils import shuffle
from keras import backend as K
import time

def readInRecipes(fname):
    with open(fname, 'r') as f:
        recipes = [i+'$$$$' for i in f.read().strip().split('$$$$')]
    prep = Preprocessor()
    np.random.shuffle(recipes)
    numTrain = int(0.8*len(recipes))
    recipesTrain = recipes[:numTrain]
    recipesVal = recipes[numTrain:]
    chars = set([c for r in recipes for c in r])
    for c in chars:
        prep.charDict[c] = len(prep.charDict)
    prep.charRev = {v: k for k,v in prep.charDict.items()}
    return recipesTrain, recipesVal, prep

    

def getRecipeBatch(recs, prep, numPer, genOrNot):
    shape1 = len(recs)*numPer

    # we want prep.predDepth outputs, depending on how many chars we pred
    ys = []
    for i in range(0, prep.predDepth):
        ys.append(np.zeros((shape1, len(prep.charRev))))
    # we also want binary outputs for seenBefore and genOrNot
    ys.append(np.zeros((shape1, 1)))
    ys.append(np.zeros((shape1, 1)))

    # initialize the X matrices to save memory
    X1 = np.zeros((shape1, prep.maxLen))
    X2 = np.zeros((shape1, prep.maxLenHist))
    for n, recipe in enumerate(recs):
        for i in range(0, numPer):
            n2 = n*numPer + i
            endInd = np.random.randint(0, len(recipe))
            startInd1 = max(0, endInd-prep.maxLen)
            startInd2 = max(0, endInd-prep.maxLenHist)
            x1 = charsToVec(recipe, startInd1, endInd, prep.maxLen,
                            prep.charDict)
            x2 = charsToVec(recipe, startInd2, endInd, prep.maxLenHist,
                            prep.charDict)
            for i in range(0, prep.predDepth):
                if (endInd+i) < len(recipe):
                    nextChar = prep.charDict[recipe[endInd+i]]
                else:
                    nextChar = prep.charDict['$']
                ys[i][n2, nextChar] = 1
            ys[prep.predDepth][n2] = getSeenBefore(recipe, endInd)
            ys[prep.predDepth+1][n2] = genOrNot[n]
            X1[n2, :] = x1
            X2[n2, :] = x2
    return [X1, X2], ys


def getSeenBefore(rec, endInd):
    if (endInd >= len(rec)) or (endInd == 0):
        return 0
    elif rec[endInd].strip() != rec[endInd]:
        return 1
    else:
        # is the last letter a strippable character?
        if len(rec[endInd-1].strip()) == 0:
            word = ""
        else:
            word = rec[:endInd].split()[-1]
        word += rec[endInd:].split()[0]
        c = rec[:endInd].count(word)
        return (c > 0)*1

def recipeGen(recs, prep, batchSize, genOrNot, numPer=10):
    while True:
        ind = 0
        X, y = getRecipeBatch(recs, prep, numPer, genOrNot)
        X = shuffle(*X, random_state=0)
        y = shuffle(*y, random_state=0)
        X1, X2 = X
        while ind < X1.shape[0]:
            ind2 = ind + batchSize
            yslice = [i[ind:ind2] for i in y]
            yield ([X1[ind:ind2], X2[ind:ind2]], yslice)
            ind += batchSize


def generateRecipes(recipesTrain, model, prep, num=11):
    newRecipes = []
    if num < len(recipesTrain):
        recs = np.random.choice(recipesTrain, num)
    else:
        recs = recipesTrain
    t1 = time.time()
    for n, recipe in enumerate(recs):
        name = prep.name_from_text(recipe)
        rec = prep.get_recipe(name, model, method="topN", N=10,
                              temperature=0.5)
        newRecipes.append(rec)
        if n % 10 == 0:
            t2 = time.time()
            print("{} recipes in {} s".format(str(n), str(t2-t1)))
    print(newRecipes[0])
    return newRecipes

def defineModel(prep):
    sharedSize = 128
    embeddingSize = 10
    outSize = len(prep.charDict)
    charInp = Input(shape=(prep.maxLen,))
    histInp = Input(shape=(prep.maxLenHist,))
    
    # character embedding
    emb = Embedding(len(prep.charDict), embeddingSize)
    
    charEmb = emb(charInp)
    histEmb = emb(histInp)

    conv2 = Conv1D(sharedSize, 3, padding="same", dilation_rate=1,
                   activation='relu')
    conv3 = Conv1D(sharedSize, 3, padding="same", dilation_rate=2,
                   activation='relu')
    conv4 = Conv1D(sharedSize, 3, padding="same", dilation_rate=4,
                   activation='relu')
    conv5_h = Conv1D(sharedSize, 3, padding="same", dilation_rate=8,
                     activation='relu')
    conv5_c = Conv1D(sharedSize, 3, padding="same", dilation_rate=1,
                     activation='relu')
    conv6_h = Conv1D(sharedSize, 3, padding="same", dilation_rate=1,
                     activation='relu')
    conv6_c = Conv1D(sharedSize, 3, padding="same", dilation_rate=1,
                     activation='relu')
    conv7_h = Conv1D(sharedSize, 3, padding="same", dilation_rate=1,
                     activation='relu')
    conv8_h = Conv1D(sharedSize, 3, padding="same", dilation_rate=1,
                     activation='relu')
    mp = MaxPooling1D(pool_size=4, strides=2)

    char2 = conv2(charEmb)
    hist2 = conv2(histEmb)

    char = conv3(char2)
    hist = conv3(hist2)

    char = conv4(char)
    hist = conv4(hist)
    hist = Add()([hist2, hist])
    hist = mp(hist)

    char = conv5_c(char)
    hist = conv5_h(hist)

    char = conv6_c(char)
    hist = conv6_h(hist)
    hist = mp(hist)
    
    hist = conv7_h(hist)
    hist = conv8_h(hist)
    hist = mp(hist)
    
    char = Flatten()(char)
    hist = Flatten()(hist)

    char = Dense(sharedSize, activation='relu')(char)
    char = Dense(sharedSize, activation='relu')(char)
    
    hist = Dense(sharedSize, activation='relu')(hist)

    # merge
    added = Add()([char, hist])
    
    outputs = []
    losses = {}
    lossWeights = {}
    for i in range(0, prep.predDepth):
        layerName = "char_"+str(i)
        outputs.append(Dense(outSize, name=layerName,
                             activation='softmax')(added))
        lossWeights[layerName] = 0.8**i
        losses[layerName] = 'categorical_crossentropy'
    outputs.append(Dense(1, name="seen_before",
                         activation='sigmoid')(added))
    outputs.append(Dense(1, name="generated_or_real",
                         activation="sigmoid")(hist))
    losses['seen_before'] = 'binary_crossentropy'
    lossWeights['seen_before'] = 2
    losses['generated_or_real'] = 'binary_crossentropy'
    lossWeights['generated_or_real'] = 2

    model = Model([charInp, histInp], outputs)
    model.compile('adam', loss=losses,
                  loss_weights=lossWeights)
    return model

np.random.seed(0)
batchSize = 64
numPer = 4
metaEpochs = 1
recipesTrain, recipesVal, prep = readInRecipes("data/allrecipes.txt")

recipesTrain = recipesTrain[:1000]

Xval, yval = getRecipeBatch(recipesVal, prep, numPer,
                            np.zeros(len(recipesVal)))
model = defineModel(prep)
with open("models/prep.pkl", 'wb') as f:
    pickle.dump(prep, f)


for metaEpoch in range(0, metaEpochs):

    # set up early stopping each time we train a new model
    callbacks = [
        EarlyStopping(patience=16, monitor='val_loss'),
        ModelCheckpoint(filepath='models/charLevel_'+str(metaEpoch)+'.cnn',
                        verbose=1, save_best_only=True,
                        monitor='val_loss'),
        TensorBoard() #  not all of the options work w/ TB+keras
    ]
    # the first time we are just training the generator, no discrim
    if metaEpoch == -1:
        generatedRecipes = []
        genOrNot = np.ones(len(recipesTrain))
    else:
        l = model.get_layer("char_0")
        func = K.function(model.input, [l.output])
        generatedRecipes = generateRecipes(recipesTrain, func, prep)
        genOrNot = np.zeros(len(recipesTrain)).tolist()
        genOrNot += np.ones(len(generatedRecipes)).tolist()

#    recipesAll = recipesTrain+generatedRecipes
#    model.fit_generator(recipeGen(recipesAll, prep, batchSize, genOrNot,
#                                  numPer=numPer),
#                        steps_per_epoch=numPer*int(len(recipesAll)/batchSize)+1,
#                        epochs=10,
#                        callbacks=callbacks,
#                        validation_data=(Xval, yval))
#    
#    model = load_model('models/charLevel_'+str(metaEpoch)+'.cnn')
