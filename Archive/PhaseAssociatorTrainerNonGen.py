import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.layers import Input, Embedding, Reshape, concatenate, Dense, Bidirectional, GRU, Dropout
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.metrics import Precision, Recall
import json
import numpy as np
import matplotlib.pyplot as plt
from PhaseAssociatorGenerator import generateEventFile, synthesizeEventsFromEventFile

synth_events_X = "./Training/Train X.npy"
synth_events_Y = "./Training/Train Y.npy"

def PhaseAssociator(X, Y, epochs=100, weight=1.0):
    batch_size = 1000
    class saveCb(Callback):
        def on_epoch_end(self, epoch, logs={}):
            # print(model.layers[1].get_weights()[0]) # weights of embedded layer
            modelName = '%03d - L%.4f - A%.4f - P%.4f - R%.4f' % (epoch, logs['val_loss'], logs['val_acc'], logs['val_precision'], logs['val_recall'])
            model.save("./Training/Models/Associator/"+modelName)
    callbacks = saveCb()
    
    def buildModel(modelUnits):
        models = []
        inputs = []
        numericalInputs = Input(shape=(None,4), name='numerical_features')
        models.append(numericalInputs)
        inputs.append(numericalInputs)
        categoricalInputs = Input(shape=(None,1), name='phase')
        embed = Embedding(4, 2, trainable=True, embeddings_initializer=RandomNormal())(categoricalInputs)
        embed = Reshape(target_shape=(-1,2))(embed)
        models.append(embed)
        inputs.append(categoricalInputs)
        models = concatenate(models)
        
        for units in modelUnits['dense']:
            models = Dense(units=units, activation=tf.nn.relu)(models)
        for units in modelUnits['grus']:
            models = Bidirectional(GRU(units, return_sequences=True))(models)

        models = Dense(units=1, activation=tf.nn.sigmoid)(models)
        model = Model(inputs=inputs, outputs=models)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[Precision(name='precision'), Recall(name='recall')], weighted_metrics=['acc'])
        return model
    
    model = buildModel(params['modelArch']['associator'])
    try:
        model.load_weights(params['models']['trainingAssociator'])
    except Exception as e:
        print(e)
        print("No previous model loaded.")
    X_dict = {"phase": X[:,:,3], "numerical_features": X[:,:,[0,1,2,4]]}
    # V_dict = ({"phase": validation[0][:,:,3], "numerical_features": validation[0][:,:,[0,1,2,4]]}, validation[1])
    vgen = synthesizeEventsFromEventFile(params, validationEvents, validationEventList)
    return model.fit(X_dict, Y,
                     validation_data=vgen,
                     validation_steps = params['validation_samples_per_epoch']/batch_size,
                     epochs=epochs,
                     batch_size=batch_size,
                     callbacks=[callbacks],
                     class_weight={0: 1.0, 1: weight},
                     verbose=1)

if __name__ == "__main__":
    def results(history):
        acc = history.history['acc']
        val_acc = history.history['val_acc']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        val_precision = history.history['val_precision']
        val_recall = history.history['val_recall']
        epochs = range(len(acc))
        
        plt.plot(epochs, acc, 'r')
        plt.plot(epochs, val_acc, 'b')
        plt.title('Training and Validation Accuracy')
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend(["Accuracy", "Validation Accuracy"])
        plt.figure()
        
        plt.plot(epochs, loss, 'r')
        plt.plot(epochs, val_loss, 'b')
        plt.title('Training and Validation Loss')
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend(["Loss", "Validation Loss"])
        plt.figure()
        
        plt.plot(epochs, val_acc, 'y')
        plt.plot(epochs, val_precision, 'r')
        plt.plot(epochs, val_recall, 'b')
        plt.title('Validation Precision and Recall')
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend(["Accuracy", "Precision", "Recall"])
        plt.figure()
    with open("Parameters.json", "r") as f:
        params = json.load(f)
    X = np.load(synth_events_X)
    Y = np.load(synth_events_Y)
    print(X.shape, Y.shape)
    print(np.where(Y==1)[0].size, "1 labels")
    print(np.where(Y==0)[0].size, "0 labels")
    if params['training_weight'] > 0.0:
        weight = params['training_weight']
    else:
        weight = (np.where(Y==0)[0].size - np.where(X[:,:,4] == 0)[0].size) / np.where(Y==1)[0].size

    # Split into train/test
    # indices = list(range(len(X)))
    # n_test = int(0.01*X.shape[0])
    # validation_idx = np.random.choice(indices, size=n_test, replace=False)
    # train_idx = list(set(indices) - set(validation_idx))
    # validation = (X[validation_idx], Y[validation_idx])
    # X = X[train_idx]
    # Y = Y[train_idx]
    validationEvents, validationEventList = generateEventFile(params, trainingSet=False)

    history = PhaseAssociator(X=X, Y=Y, epochs=params['epochs'], weight=weight)
    results(history)