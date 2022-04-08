import tensorflow as tf
import numpy as np
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.layers import Input, Embedding, Reshape, concatenate, Dense, Bidirectional, GRU, MultiHeadAttention, LayerNormalization
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, CSVLogger
import logging
import json
from MatrixLinkGenerator import generateEventFile, synthesizeEvents, synthesizeEventsFromEventFile
from Utils import nzBCE, nzMSE, nzPrecision, nzRecall, nzHaversine, nzTime, trainingResults

@tf.autograph.experimental.do_not_convert
def MatrixLink(params):
    logging.getLogger("tensorflow").setLevel(logging.ERROR)
    def buildModel(modelArch):
        outputs = []
        inputs = []
        numericalInputs = Input(shape=(None,4), name='numerical_features')
        outputs.append(numericalInputs)
        inputs.append(numericalInputs)
        categoricalInputs = Input(shape=(None,1), name='phase')
        embed = Embedding(5, 2, trainable=True, embeddings_initializer=RandomNormal())(categoricalInputs)
        embed = Reshape(target_shape=(-1,2))(embed)
        outputs.append(embed)
        inputs.append(categoricalInputs)
        outputs = concatenate(outputs)

        def TransformerBlock(inputs, embed_dim, ff_dim, num_heads=2, rate=0.1, eps=1e-6):
            attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)(inputs, inputs)
#             attn_output = Dropout(rate)(attn_output)
            out1 = LayerNormalization(epsilon=eps)(inputs + attn_output)
            ffn_output = Dense(ff_dim, activation="relu")(out1)
            ffn_output = Dense(embed_dim)(ffn_output)
#             ffn_output = Dropout(rate)(ffn_output)
            return LayerNormalization(epsilon=eps)(out1 + ffn_output) 

        for d1Units in modelArch['dense']:
            outputs = Dense(units=d1Units, activation=tf.nn.relu)(outputs)
        transformerOutputs = outputs
        gruOutputs = outputs

        for tUnits in modelArch['transformers']:
            transformerOutputs = TransformerBlock(transformerOutputs, d1Units, tUnits, modelArch['heads'])
        for gUnits in modelArch['grus']:
            gruOutputs = Bidirectional(GRU(gUnits, return_sequences=True))(gruOutputs)

        outputs = concatenate([transformerOutputs, gruOutputs], axis=2)
        for tUnits in modelArch['transformers']:
            outputs = TransformerBlock(outputs, d1Units+gUnits*2, tUnits, modelArch['heads'])

        association = Dense(units=params['maxArrivals'], activation=tf.nn.sigmoid, name='association')(outputs)
        location = Dense(units=2, name='location')(outputs)
        noise = Dense(units=1, activation=tf.nn.sigmoid, name='noise')(outputs)
        time = Dense(units=1, name='time')(outputs)
        
        model = Model(inputs=inputs, outputs=[association, location, noise, time])
        losses = { 'association': nzBCE, 'location': nzMSE, 'noise': nzBCE, 'time': nzMSE }
        weights = { 'association': 1.0, 'location': 1.0, 'noise': 0.25, 'time': 0.5 }
        metrics = { 'association': [nzPrecision, nzRecall],
                    'location': nzHaversine,
                    'noise': [nzPrecision, nzRecall],
                    'time': nzTime
                  }
        model.compile(optimizer=Adam(clipnorm=0.00001), loss=losses, loss_weights=weights, metrics=metrics)
        return model

    model = buildModel(params['modelArch'])
    try:
        model.load_weights(params['model'])
        print("Loaded previous weights.")
    except Exception as e:
        print(e)
        print("No previous weights loaded.")
    print(model.summary())
    return model

class saveCb(Callback):
    def on_train_begin(self, logs=None):
        self.best = 100.
    def on_epoch_end(self, epoch, logs=None):
        if logs['loss'] < self.best:
            self.best = logs['loss']
            print('Saving best model with loss', self.best)
            modelName = 'E%03d L%.4f AL%.4f LL%.4f NL%.4f TL%.4f AP%.4f AR%.4f NP%.4f NR%.4f HL%.1f TL%.3f.h5' %\
                (epoch, logs['loss'], logs['association_loss'], logs['location_loss'], logs['noise_loss'], logs['time_loss'], logs['association_nzPrecision'], logs['association_nzRecall'], logs['noise_nzPrecision'], logs['noise_nzRecall'], logs['location_nzHaversine'], logs['time_nzTime'])
            model.save("./Training/Models/"+modelName)

if __name__ == "__main__":
    with open("Parameters.json", "r") as f:
        params = json.load(f)
    # trainingEvents, trainingEventList = generateEventFile(params, trainingSet=True)
    # validationEvents, validationEventList = generateEventFile(params)
    # generator = synthesizeEventsFromEventFile(params, trainingEvents, trainingEventList, trainingSet=True)
    generator = synthesizeEvents(params)
    # vgen = synthesizeEventsFromEventFile(params, validationEvents, validationEventList)
    # vgen = synthesizeEvents(params)
    
    model = MatrixLink(params)
    history = model.fit(generator,
    #                  validation_data=vgen,
                     steps_per_epoch= params['samplesPerEpoch']/params['batchSize'],
    #                  validation_steps = params['validationSamplesPerEpoch']/params['batchSize'],
                     epochs=params['epochs'],
                     callbacks=[saveCb(), EarlyStopping(monitor='loss', patience=40), CSVLogger('./Training/Models/logs.csv', append = True)],
                     verbose=1)
    trainingResults(np.genfromtxt('./Training/Models/logs.csv', delimiter=',', names=True))