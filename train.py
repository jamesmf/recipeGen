# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 23:30:26 2018

@author: jmf
"""
import pickle
import numpy as np
import os
import json
from keras.models import Model, load_model
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
import utilities as util
from utilities import readInRecipes, getRecipeBatch, discriminator_mode
from utilities import recipeGen, generateRecipes, defineModel


np.random.seed(0)
batchSize = 128
numPer = 10
epochs = 2
logID = util.get_log_id("logs")
util.set_up_logging()
model_dir = "models/model_{}/".format(logID)

recipesTrain, recipesVal, prep = readInRecipes()

#recipesTrain = recipesTrain[:1000]
#recipesVal = recipesVal[:1000]

Xval, yval = getRecipeBatch(recipesVal, prep, numPer)
model = defineModel(prep)
os.makedirs(model_dir)
with open("{}prep.pkl".format(model_dir), 'wb') as f:
    pickle.dump(prep, f)
# save json with the reqs for sampling
prepjson = {}
with open("{}prep.json".format(model_dir), 'wb') as f:
    prepjson["charRev"] = prep.charRev
    prepjson["charMap"] = prep.charDict
    prepjson["x1_len"] = prep.maxLen
    prepjson["x2_len"] = prep.maxLenHist
    f.write(json.dumps(prepjson))
    
for metaEpoch in range(0, epochs):

    # set up early stopping each time we train a new model
    callbacks = [
        EarlyStopping(patience=15, monitor='val_loss'),
        ModelCheckpoint(filepath=model_dir+'charLevel_'+str(metaEpoch)+'.cnn',
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
    model = load_model(model_dir+'charLevel_'+str(metaEpoch)+'.cnn')
    mod = Model(model.input, [model.get_layer("char_0").output])
    mod.save('{}only_char_0_{}.cnn'.format(model_dir, str(metaEpoch)))
    func = mod.predict

    # get some validation data
    genVal = generateRecipes(recipesVal, func, prep, num=10000)
    genOrNotVal = np.zeros((len(recipesVal),1)).tolist()
    genOrNotVal += np.ones((len(genVal),1)).tolist()
    Xval_disc, yval_disc = getRecipeBatch(recipesVal+genVal, prep, numPer,
                                          genOrNot=genOrNotVal)
    genTrain = generateRecipes(recipesTrain, func, prep, num=30000)
    genOrNotTrain = np.zeros((len(recipesTrain),1)).tolist()
    genOrNotTrain += np.ones((len(genTrain),1)).tolist()
    Xtrain_disc, ytrain_disc = getRecipeBatch(recipesTrain+genTrain, prep,
                                              numPer, genOrNot=genOrNotTrain)
    
    model = discriminator_mode(model, prep, mode='discrim')

    # train the model to discriminate between real and fake recipes
    model.fit(Xtrain_disc, ytrain_disc, epochs=1,
              validation_data=(Xval_disc, yval_disc))
    model = discriminator_mode(model, prep, mode='normal')