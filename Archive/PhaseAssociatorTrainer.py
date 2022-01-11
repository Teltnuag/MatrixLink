import tensorflow as tf
# physical_devices = tf.config.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.layers import Input, Embedding, Reshape, concatenate, Dense, Bidirectional, GRU, Dropout, Flatten, MultiHeadAttention, LayerNormalization
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras import backend as K
import logging
import json
from PhaseAssociatorGenerator import generateEventFile, synthesizeEventsFromEventFile, synthesizeEvents, synthesizeLocatorEventsFromEventFile
from Utils import trainingResults

@tf.autograph.experimental.do_not_convert
def PhaseAssociator(epochs=10, weight=1.0):
    logging.getLogger("tensorflow").setLevel(logging.ERROR)
    class saveCb(Callback):
        def on_epoch_end(self, epoch, logs=None):
            # print('\n' + str(model.layers[1].get_weights()[0])) # weights of embedded layer
            modelName = '%03d - L%.4f - A%.4f - P%.4f - R%.4f' % (epoch, logs['val_loss'], logs['val_acc'], logs['val_precision'], logs['val_recall'])
            # modelName = '%03d - L%.4f - A%.4f - P%.4f - R%.4f' % (epoch, logs['loss'], logs['acc'], logs['precision'], logs['recall'])
            model.save("./Training/Models/Associator1/"+modelName)
    
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
            
        for units in modelArch['dense']:
            outputs = Dense(units=units, activation=tf.nn.relu)(outputs)
        for units in modelArch['grus']:
            outputs = Bidirectional(GRU(units, return_sequences=True))(outputs)

        outputs = Dense(units=1, activation=tf.nn.sigmoid)(outputs)
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[Precision(name='precision'), Recall(name='recall')], weighted_metrics=['acc'])
        return model
    
    model = buildModel(params['modelArch']['associator'])
    try:
        model.load_weights(params['models']['trainingAssociator'])
        print("Loaded previous weights.")
    except Exception as e:
        print(e)
        print("No previous weights loaded.")

    # generator = synthesizeEventsFromEventFile(params, trainingEvents, trainingEventList)
    generator = synthesizeEvents(params)
    vgen = synthesizeEventsFromEventFile(params, validationEvents, validationEventList)
    # vgen = synthesizeEvents(params)
    return model.fit(generator,
                     validation_data=vgen,
                     steps_per_epoch= params['samplesPerEpoch']/params['batchSize'],
                     validation_steps = params['validationSamplesPerEpoch']/params['batchSize'],
                     epochs=epochs,
                     callbacks=[saveCb()],
                     class_weight={0: 1.0, 1: weight},
                     verbose=1)

@tf.autograph.experimental.do_not_convert
def Locator(epochs=10):
    logging.getLogger("tensorflow").setLevel(logging.ERROR)
    class saveCb(Callback):
        def on_epoch_end(self, epoch, logs=None):
            # print('\n' + str(model.layers[1].get_weights()[0])) # weights of embedded layer
#             modelName = '%03d - L%.4f - A%.4f - P%.4f - R%.4f' % (epoch, logs['val_loss'], logs['val_acc'], logs['val_precision'], logs['val_recall'])
            modelName = '%03d - L%.4f' % (epoch, logs['val_loss_haversine'])
            model.save("./Training/Models/Locator/"+modelName)
    
    def degrees_to_radians(deg):
        return deg * 0.017453292519943295

    def loss_haversine(ytrue, ypred):
        observation = ytrue[:,0:2]
        prediction = ypred[:,0:2]
        obv_rad = tf.map_fn(degrees_to_radians, observation)
        prev_rad = tf.map_fn(degrees_to_radians, prediction)

        dlon_dlat = obv_rad - prev_rad 
        v = dlon_dlat / 2
        v = tf.sin(v)
        v = v**2

        a = v[:,1] + tf.cos(obv_rad[:,1]) * tf.cos(prev_rad[:,1]) * v[:,0] 

        c = tf.sqrt(a)
        c = 2* tf.math.asin(c)
        c = c*6378.1
        final = tf.reduce_sum(c)

        #if you're interested in having MAE with the haversine distance in KM
        #uncomment the following line
        final = final/tf.dtypes.cast(tf.shape(observation)[0], dtype= tf.float32)

        return final
    
    def loss_depth(ytrue, ypred):
        observation = ytrue[:,2]
        prediction = ypred[:,2]
        diffs = abs(observation - prediction)
        diffs = tf.reduce_sum(diffs)
        return diffs/tf.dtypes.cast(tf.shape(observation)[0], dtype= tf.float32)
    
    def loss_eventTime(ytrue, ypred):
        observation = ytrue[:,3]
        prediction = ypred[:,3]
        diffs = abs(observation - prediction)
        diffs = tf.reduce_sum(diffs)
        return diffs/tf.dtypes.cast(tf.shape(observation)[0], dtype= tf.float32)
    
    def nonzero_mse(y_true, y_pred):
        y_pred = y_pred * tf.cast(y_true != 0, tf.float32)
        return K.mean(K.square(y_pred-y_true))

    def TransformerBlock(inputs, embed_dim, ff_dim, num_heads=2, rate=0.1, eps=1e-6):
        attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)(inputs, inputs)
        attn_output = Dropout(rate)(attn_output)

        out1 = LayerNormalization(epsilon=eps)(inputs + attn_output)

        ffn_output = Dense(ff_dim, activation="relu")(out1)
        ffn_output = Dense(embed_dim)(ffn_output)
        ffn_output = Dropout(rate)(ffn_output)

        return LayerNormalization(epsilon=eps)(out1 + ffn_output)    

    def buildModel(modelArch):
        outputs = []
        inputs = []
        numericalInputs = Input(shape=(params['maxArrivals'],4), name='numerical_features') # station lat, station lon, arrival time, padding
        outputs.append(numericalInputs)
        inputs.append(numericalInputs)
        categoricalInputs = Input(shape=(params['maxArrivals'],1), name='phase')
        embed = Embedding(4, 2, trainable=True, embeddings_initializer=RandomNormal())(categoricalInputs)
        embed = Reshape(target_shape=(-1,2))(embed)
        outputs.append(embed)
        inputs.append(categoricalInputs)
        outputs = concatenate(outputs)
            
        for dUnits in modelArch['dense1']:
            outputs = Dense(units=dUnits, activation=tf.nn.relu)(outputs)
        for tUnits in modelArch['transformers']:
            outputs = TransformerBlock(outputs, dUnits, tUnits, modelArch['heads'])
        for units in modelArch['grus']:
            outputs = Bidirectional(GRU(units, return_sequences=True))(outputs)
        # for dUnits in modelArch['dense2']:
        #     outputs = Dense(units=dUnits, activation=tf.nn.relu)(outputs)
        # outputs = Dense(units=dUnits)(outputs)
        outputs = Flatten()(outputs)
        outputs = Dense(units=4, activation='linear')(outputs)
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer='adam',
                      loss=[nonzero_mse, loss_haversine],
                      metrics=[loss_haversine, loss_depth, loss_eventTime])
        return model
    
    model = buildModel(params['modelArch']['locator'])
    try:
        model.load_weights(params['models']['trainingLocator'])
        print("Loaded previous weights.")
    except Exception as e:
        print(e)
        print("No previous weights loaded.")

    # generator = synthesizeLocatorEventsFromEventFile(params, trainingEvents, trainingEventList)
    generator = synthesizeEvents(params, locator=True)
    vgen = synthesizeLocatorEventsFromEventFile(params, validationEvents, validationEventList, trainingSet=False)
    return model.fit(generator,
                     validation_data=vgen,
                     steps_per_epoch= params['samplesPerEpoch']/params['batchSize'],
                     validation_steps = params['validationSamplesPerEpoch']/params['batchSize'],
                     epochs=epochs,
                     callbacks=[saveCb()],
                     verbose=1)

if __name__ == "__main__":
    with open("Parameters.json", "r") as f:
        params = json.load(f)
    # trainingEvents, trainingEventList = generateEventFile(params, trainingSet=True)
    validationEvents, validationEventList = generateEventFile(params, trainingSet=False)
    if params['training_mode'] == 'Associator':
        history = PhaseAssociator(epochs=params['epochs'], weight=params['training_weight'])
    else:
        history = Locator(epochs=params['epochs'])
    trainingResults(history, params['training_mode'])