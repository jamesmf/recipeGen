# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 23:30:26 2018

@author: jmf
"""
import pickle
import numpy as np
from keras.layers import Input, Dense, Conv1D, Embedding, Flatten, GlobalMaxPooling1D
from keras.layers import MaxPooling1D, Add, Dropout, BatchNormalization
from keras.models import Model, load_model
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
from utilities import Preprocessor, charsToVec
from sklearn.utils import shuffle
from keras import backend as K
import time

def readInRecipes(fname):
    """
    Reads in a file with recipes formatted as specified in the README. Splits
    them into Train/Dev (Validation) set and initializes a Preprocessor.
    Args:
        fname (str): name of the file containing recipes
    Returns:
        recipesTrain (list): list of recipe strings in the training set
        recipesVal (list): list of recipe strings in the dev/val set
        prep (Preprocessor): Preprocessor object 'fit' to the recipe chars
    """
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

    

def getRecipeBatch(recs, prep, numPer, genOrNot=None):
    """
    Turns a list of strings (recipes) into a batch for training. When genOrNot
    is not None, the function returns the entire recipe and genOrNot
    (discriminator mode).
    Args:
        recs (list): list of recipe strings. Should have '\nname:\n' prepended
        prep (Preprocessor): fit Preprocessor object
        numPer (int): number of examples to generate per recipe string
        genOrNot (list or None): in discriminator mode, each entry indicates if recipe i is real or syntheticall generated
    Returns:
        Xs (list): list of [X1, X2]
        ys (list): list of [y1...yn]
    """
    shape1 = len(recs)*numPer
    X1 = np.zeros((shape1, prep.maxLen))
    X2 = np.zeros((shape1, prep.maxLenHist))
    ys = []
    # if genOrNot is not None, we're in discriminator mode, which means we are
    # predicting whether a recipe is real or synthetically generated
    if genOrNot is not None:
        # only one output: a discriminator
        y = []
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
                X1[n2, :] = x1
                X2[n2, :] = x2
                y.append(genOrNot[n][0])
        y = np.array(y)
        ys.append(y)
        return [X1, X2], ys

    # if genOrNot is None, we're fitting the generator, and as such we have
    # multiple 'y' matrices; one per char plus whether or not we've seen the
    # word before
    for i in range(0, prep.predDepth):
        ys.append(np.zeros((shape1, len(prep.charRev))))
    # we also want a binary output for seenBefore
    ys.append(np.zeros((shape1, 1)))

    # initialize the X matrices to save memory
    X1 = np.zeros((shape1, prep.maxLen))
    X2 = np.zeros((shape1, prep.maxLenHist))

    # iterate over recipes
    for n, recipe in enumerate(recs):
        # get numPer samples from each recipe
        for i in range(0, numPer):
            n2 = n*numPer + i
            # pick a character to predict; calculate the starting index
            endInd = np.random.randint(0, len(recipe))
            startInd1 = max(0, endInd-prep.maxLen)
            startInd2 = max(0, endInd-prep.maxLenHist)
            x1 = charsToVec(recipe, startInd1, endInd, prep.maxLen,
                            prep.charDict)
            x2 = charsToVec(recipe, startInd2, endInd, prep.maxLenHist,
                            prep.charDict)
            for i in range(0, prep.predDepth):
                # if it's > the len(recipe), we predict the STOPTOKEN, '$'
                if (endInd+i) < len(recipe):
                    nextChar = prep.charDict[recipe[endInd+i]]
                else:
                    nextChar = prep.charDict['$']
                ys[i][n2, nextChar] = 1
            ys[prep.predDepth][n2] = getSeenBefore(recipe, endInd)
            X1[n2, :] = x1
            X2[n2, :] = x2
    return [X1, X2], ys


def getSeenBefore(rec, endInd):
    """
    Given the index of the character we want to predict while the model is
    being fit in generator mode, return whether or not the 'current word' has
    been seen before in the recipe. Ideally adding this signal will serve to
    remind the model it needs to specify ingredients it has already mentioned.
    Args:
        rec (str): recipe string
        endInd (int): index of the character we are predicting
    Returns:
        (bool): whether the current word has been seen before in the recipe
    """
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

def recipeGen(recs, func, prep, batchSize, numGen=10000, numPer=10,
              headType="char"):
    """
    Generator to be used with keras's fit_generator. Works with either the
    discriminator mode or generator mode. Returns shuffled X/y slices.
    Args:
        recs (list): recipe strings
        func (keras.backend.Function): function mapping from input to output. used for discriminator mode only.
        batchSize (int): number of examples in a mini-batch
        numGen (int): if discriminator mode, how many recipes to create
        numPer (int): number of training samples per recipe
    Yields:
        X (list of numpy arrays)
        y (list of numpy arrays)
    """
    while True:
        ind = 0
        genOrNot = None
        gen = []
        if headType == "discrim":
            gen = generateRecipes(recs, func, prep, num=numGen)
            genOrNot = np.zeros((len(recs),1)).tolist()+np.ones((len(gen),1)).tolist()
        X, y = getRecipeBatch(recs+gen, prep, numPer, genOrNot=genOrNot)
#        print(X, y)
        X = shuffle(*X, random_state=0)
        y = shuffle(*y, random_state=0)
        if type(y) != list:
            # if this is the case, y had only one array
            y = [y]
        X1, X2 = X
        while ind < X1.shape[0]:
            ind2 = ind + batchSize
            yslice = [i[ind:ind2] for i in y]
            yield ([X1[ind:ind2], X2[ind:ind2]], yslice)
            ind += batchSize


def generateRecipes(recipesTrain, func, prep, num=10000):
    """
    Simple function that creates some number of recipes starting from the
    names of some existing recipes.
    Args:
        recipesTrain (list): recipe strings whose names will seed the generator
        func (keras.backend.Function): function mapping from input to next-char prediction
        prep (Preprocessor): Preprocessor object fit to recipes
        num (int): cap on how many recipes to generate. Won't generate more than len(recipesTrain)
    Returns:
        newRecipes (list): list of recipe strings
    """
    newRecipes = []
    if num < len(recipesTrain):
        recs = np.random.choice(recipesTrain, num)
    else:
        recs = recipesTrain
    t1 = time.time()
    names = [prep.name_from_text(i) for i in recs]
    newRecipes = prep.get_many_recipes(names, func)
    return newRecipes


def defineModel(prep):
    """
    Function to define keras/tf model. Currently has two inputs: one captures
    local history (last handful of characters) and one global history. The two
    model inputs share lower layers, prompting a shared understanding of word-
    level semantics. Model has dilation to increase receptive field without
    extreme depth. There are a handful of skip connections, BatchNorm layers,
    and Dropout. The history branch is globally pooled, wheras the local
    history is Flattened. The outputs are the next several (default 5)
    characters in the sequence and whether the current word has been seen
    before. The loss decays such that char_0 is more important than char_4.
    Args:
        prep (Preprocessor): Preprocessor object contains most of the relevant info.
    Returns:
        model (Model): keras Model compiled
    """
    sharedSize = 128
    embeddingSize = 64
    outSize = len(prep.charDict)
    charInp = Input(shape=(prep.maxLen,))
    histInp = Input(shape=(prep.maxLenHist,), name='hist')
    
    # character embedding
    emb = Embedding(len(prep.charDict), embeddingSize)
    
    charEmb = emb(charInp)
    histEmb = emb(histInp)

    conv2 = Conv1D(sharedSize, 3, padding="same", dilation_rate=1,
                   activation='relu', name='shared_conv1')
    conv3 = Conv1D(sharedSize, 3, padding="same", dilation_rate=2,
                   activation='relu', name='shared_conv2')
    conv4 = Conv1D(sharedSize, 3, padding="same", dilation_rate=4,
                   activation='relu', name='shared_conv3')
    conv5_h = Conv1D(sharedSize, 3, padding="same", dilation_rate=8,
                     activation='relu', name='conv_hist4')
    conv5_c = Conv1D(sharedSize, 3, padding="same", dilation_rate=1,
                     activation='relu', name='conv_char4')
    conv6_h = Conv1D(sharedSize, 3, padding="same", dilation_rate=1,
                     activation='relu', name='conv_hist5')
    conv6_c = Conv1D(sharedSize, 3, padding="same", dilation_rate=1,
                     activation='relu', name='conv_char5')
    conv7_h = Conv1D(sharedSize, 3, padding="same", dilation_rate=1,
                     activation='relu', name='conv_hist6')
    conv8_h = Conv1D(sharedSize, 3, padding="same", dilation_rate=1,
                     activation='relu', name='conv_hist7')
    mp = MaxPooling1D(pool_size=4, strides=2)

    char2 = conv2(charEmb)
    hist2 = conv2(histEmb)

    char = conv3(char2)
    hist = conv3(hist2)

    char = Dropout(0.25)(conv4(char))
    char = BatchNormalization()(char)
    hist = Dropout(0.25)(conv4(hist))
    hist = BatchNormalization()(hist)
    hist = Add(name='shortcut_hist')([hist2, hist])
    hist = mp(hist)

    char = conv5_c(char)
    hist = conv5_h(hist)

    char = Dropout(0.25)(conv6_c(char))
    char = BatchNormalization()(char)
    hist = Dropout(0.25)(conv6_h(hist))
    hist = BatchNormalization()(hist)
    hist = mp(hist)
    
    hist = conv7_h(hist)
    hist = BatchNormalization()(hist)
    hist = conv8_h(hist)
    # final global max pooling layer for the hist side of things
    hist = GlobalMaxPooling1D()(hist)
    
    char = Flatten()(char)

    char = Dense(sharedSize, activation='relu')(char)
    char = Dense(sharedSize, activation='relu')(char)
    
    hist = Dense(sharedSize, activation='relu')(hist)

    # merge
    added = Add(name='added')([char, hist])
    
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
    losses['seen_before'] = 'binary_crossentropy'
    lossWeights['seen_before'] = 2

    model = Model([charInp, histInp], outputs)
    model.compile('adam', loss=losses,
                  loss_weights=lossWeights)
    return model


def discriminator_mode(model, prep, mode="discrim"):
    """
    Function to 'flip' the model from generator to discriminator mode. Does so
    by redefining the outputs and losses, then recompiling. Should be factored
    out such that the generator logic is shared w/ define_model()
    Args:
        model (Model): model whose state should be flipped
        prep (Preprocessor): Preprocessor object contains relevant info
        mode (str): "discrim" or "generator" depending on which mode to switch to
    Returns:
        model (Model): model flipped to the new mode.
    """
    final_shared = model.get_layer("added").output
    if mode == "discrim":
        outputs = [Dense(1, name='gen_or_not',
                         activation='sigmoid')(final_shared)]
        lossWeights = [1]
        losses = ['binary_crossentropy']
    else:
        outputs = []
        losses = {}
        lossWeights = {}
        for i in range(0, prep.predDepth):
            layerName = "char_"+str(i)
            outputs.append(Dense(len(prep.charDict), name=layerName,
                                 activation='softmax')(final_shared))
            lossWeights[layerName] = 0.9**i
            losses[layerName] = 'categorical_crossentropy'
        outputs.append(Dense(1, name="seen_before",
                             activation='sigmoid')(final_shared))
        losses['seen_before'] = 'binary_crossentropy'
        lossWeights['seen_before'] = 1

    model = Model(model.input, outputs)
    model.compile('adam', loss=losses,
                  loss_weights=lossWeights)
    return model

np.random.seed(0)
batchSize = 64
numPer = 4
epochs = 20
recipesTrain, recipesVal, prep = readInRecipes("data/allrecipes.txt")

#recipesTrain = recipesTrain[:1000]
#recipesVal = recipesVal[:1000]

Xval, yval = getRecipeBatch(recipesVal, prep, numPer)
model = defineModel(prep)
with open("models/prep.pkl", 'wb') as f:
    pickle.dump(prep, f)

for metaEpoch in range(0, epochs):

    # set up early stopping each time we train a new model
    callbacks = [
        EarlyStopping(patience=15, monitor='val_loss'),
        ModelCheckpoint(filepath='models/charLevel_'+str(metaEpoch)+'.cnn',
                        verbose=1, save_best_only=True,
                        monitor='val_char_0_loss'),
        TensorBoard(log_dir='logs/{}_char_{}'.format(logID, metaEpoch))
    ]
    func = None
    # train the model to predict next characters until validation loss plateaus
    model.fit_generator(recipeGen(recipesTrain, func, prep, batchSize,
                                  numPer=numPer),
                        steps_per_epoch=numPer*int(len(recipesTrain)/batchSize)+1,
                        epochs=150,
                        callbacks=callbacks,
                        validation_data=(Xval, yval))

    # RUN THE DISCRIMINATOR PORTION    
    model = load_model('models/charLevel_'+str(metaEpoch)+'.cnn')
    mod = Model(model.input, [model.get_layer("char_0").output])
    func = mod.predict

    # get some validation data
    genVal = generateRecipes(recipesVal, func, prep, num=10000)
    genOrNotVal = np.zeros((len(recipesVal),1)).tolist()
    genOrNotVal += np.ones((len(genVal),1)).tolist()
    Xval_disc, yval_disc = getRecipeBatch(recipesVal+genVal, prep, numPer,
                                          genOrNot=genOrNotVal)
    genTrain = generateRecipes(recipesTrain, func, prep, num=10000)
    genOrNotTrain = np.zeros((len(recipesTrain),1)).tolist()
    genOrNotTrain += np.ones((len(genTrain),1)).tolist()
    Xtrain_disc, ytrain_disc = getRecipeBatch(recipesTrain+genTrain, prep,
                                              numPer, genOrNot=genOrNotTrain)
    
    model = discriminator_mode(model, prep, mode='discrim')
    # set up early stopping each time we train a new model
    callbacks = [
        EarlyStopping(patience=5, monitor='val_loss'),
        ModelCheckpoint(filepath='models/discrim_'+str(metaEpoch)+'.cnn',
                        verbose=1, save_best_only=True,
                        monitor='val_loss')
    ]
    # train the model to discriminate between real and fake recipes
    model.fit(Xtrain_disc, ytrain_disc, epochs=10, callbacks=callbacks,
              validation_data=(Xval_disc, yval_disc))
    model = discriminator_mode(model, prep, mode='normal')