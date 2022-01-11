import tensorflow as tf
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.layers import Input, Embedding, Reshape, concatenate, Dense, Bidirectional, GRU, MultiHeadAttention, LayerNormalization
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.callbacks import EarlyStopping, CSVLogger
import logging
import json
from MatrixLinkGenerator import generateEventFile, synthesizeEvents, synthesizeEventsFromEventFile
from Utils import nzBCE, nzMSE2, nzHaversine, nzAccuracy, trainingResults

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
        embed = Embedding(4, 2, trainable=True, embeddings_initializer=RandomNormal())(categoricalInputs)
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
        # depth = Dense(units=1, name='depth')(outputs)
        # time = Dense(units=1, name='time')(outputs)
        
        # model = Model(inputs=inputs, outputs=[association, location, depth, time])
        model = Model(inputs=inputs, outputs=[association, location])
        # losses = { 'association': nzBCE, 'location': nzMSE2, 'depth': nzMSE1, 'time': nzMSE1 }
        losses = { 'association': nzBCE, 'location': nzMSE2 }
        # weights = { 'association': 1.0, 'location': 6.0, 'depth': 0.075, 'time': 1.0 }
        weights = { 'association': 1.0, 'location': 10.0 }
        metrics = { 'association': [nzAccuracy, Precision(name='precision'), Recall(name='recall')],
                    'location': nzHaversine }
                    # 'depth': nzDepth,
                    # 'time': nzTime }
        model.compile(optimizer=Adam(clipnorn=1.0), loss=losses, loss_weights=weights, metrics=metrics)
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
            # modelName = 'E%03d L%.4f A%.4f P%.4f R%.4f HL%.1f DL%.3f TL%.3f VL%.4f VA%.4f VP%.4f VR%.4f VHL%.1f VDL%.3f VTL%.3f.h5' %\
            #     (epoch, logs['loss'], logs['association_nzAccuracy'], logs['association_precision'], logs['association_recall'], logs['location_nzHaversine'], logs['depth_nzDepth'], logs['time_nzTime'],
            #      logs['val_loss'], logs['val_association_nzAccuracy'], logs['val_association_precision'], logs['val_association_recall'], logs['val_location_nzHaversine'], logs['val_depth_nzDepth'], logs['val_time_nzTime'])
            modelName = 'E%03d L%.4f A%.4f P%.4f R%.4f HL%.1f VL%.4f VA%.4f VP%.4f VR%.4f VHL%.1f.h5' %\
                (epoch, logs['loss'], logs['association_nzAccuracy'], logs['association_precision'], logs['association_recall'], logs['location_nzHaversine'], 
                 logs['val_loss'], logs['val_association_nzAccuracy'], logs['val_association_precision'], logs['val_association_recall'], logs['val_location_nzHaversine'])
            model.save("./Training/Models/"+modelName)

if __name__ == "__main__":
    with open("Parameters.json", "r") as f:
        params = json.load(f)
    # trainingEvents, trainingEventList = generateEventFile(params, trainingSet=True)
    validationEvents, validationEventList = generateEventFile(params)
    
    # generator = synthesizeEventsFromEventFile(params, trainingEvents, trainingEventList, trainingSet=True)
    generator = synthesizeEvents(params)
    vgen = synthesizeEventsFromEventFile(params, validationEvents, validationEventList)
    # vgen = synthesizeEvents(params)

    model = MatrixLink(params)
    history = model.fit(generator,
                     validation_data=vgen,
                     steps_per_epoch= params['samplesPerEpoch']/params['batchSize'],
                     validation_steps = params['validationSamplesPerEpoch']/params['batchSize'],
                     epochs=params['epochs'],
                     callbacks=[saveCb(), EarlyStopping(monitor='loss', patience=30), CSVLogger('./Training/Models/logs.csv', append = True)],
                     verbose=1)
    trainingResults(history)