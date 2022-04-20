import numpy as np
import tensorflow as tf
import json
from tensorflow.keras import backend as K
from tensorflow.keras.losses import binary_crossentropy as BCE
from tensorflow.keras.metrics import binary_accuracy

with open("Parameters.json", "r") as f:
    params = json.load(f)
maxArrivals = params['maxArrivals']
matrixSize = maxArrivals**2
extents = np.array(list(params['extents'][params['location']].values())+[params['maxDepth'],params['maxStationElevation']])
latRange = abs(extents[1] - extents[0])
lonRange = abs(extents[3] - extents[2])
timeNormalize = params['timeNormalize']
    
def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])        
    dlat = lat2 - lat1
    dlon = lon2 - lon1    
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2    
    c = 2 * np.arcsin(np.sqrt(a))
    return 6378.1 * c

def nzHaversine(y_true, y_pred):
    y_pred = y_pred * tf.cast(y_true != 99, tf.float32)
    y_true = y_true * tf.cast(y_true != 99, tf.float32)
    observation = tf.stack([y_true[:,:,0]*latRange + extents[0], y_true[:,:,1]*lonRange + extents[2]],axis=2)*0.017453292519943295
    prediction = tf.stack([y_pred[:,:,0]*latRange + extents[0], y_pred[:,:,1]*lonRange + extents[2]],axis=2)*0.017453292519943295
    used = tf.reduce_sum(tf.cast(tf.greater(tf.reduce_sum(y_true, axis=2),0), dtype=tf.float32), axis=1)
    used = tf.where(tf.equal(used, 0.), 1., used)
    dlat_dlon = (observation - prediction) / 2
    a = tf.sin(dlat_dlon[:,:,0])**2 + tf.cos(observation[:,:,0]) * tf.cos(prediction[:,:,0]) * tf.sin(dlat_dlon[:,:,1])**2
    c = 2*tf.asin(tf.sqrt(a))*6378.1
    final = tf.reduce_sum((tf.reduce_sum(c, axis=1))/used) / tf.dtypes.cast(tf.shape(observation)[0], dtype= tf.float32)
    return final

# def nzDepth(ytrue, ypred):
#     used = maxArrivals - tf.reduce_sum(tf.cast(tf.equal(ytrue,0), dtype=tf.float32), axis=1)
#     used = tf.where(tf.equal(used, 0.), 1., used)
#     diffs = abs(tf.squeeze(ypred)-ytrue)*extents[4]
#     diffs = tf.reduce_sum(tf.reduce_sum(diffs, axis=1)/used)
#     return diffs/tf.dtypes.cast(tf.shape(ytrue)[0], dtype= tf.float32)

# def nzTime(ytrue, ypred):
#     used = maxArrivals - tf.reduce_sum(tf.cast(tf.equal(ytrue,0), dtype=tf.float32), axis=1)
#     used = tf.where(tf.equal(used, 0.), 1., used)
#     diffs = abs(tf.squeeze(ypred)-ytrue)*timeNormalize
#     diffs = tf.reduce_sum(tf.reduce_sum(diffs, axis=1)/used)
#     return diffs/tf.dtypes.cast(tf.shape(ytrue)[0], dtype= tf.float32)

def nzTime(y_true, y_pred):
    y_pred = y_pred * tf.cast(y_true != 99, tf.float32)
    y_true = y_true * tf.cast(y_true != 99, tf.float32)
    used = maxArrivals - tf.reduce_sum(tf.cast(tf.equal(y_true,0), dtype=tf.float32), axis=1)
    used = tf.where(tf.equal(used, 0.), 1., used)
    diffs = tf.math.abs(tf.squeeze(y_pred)-y_true)*timeNormalize
#     diffs = (tf.squeeze(y_pred)-y_true)*timeNormalize
    diffs = tf.reduce_sum(tf.reduce_sum(diffs, axis=1)/used)
    return diffs/tf.dtypes.cast(tf.shape(y_true)[0], dtype= tf.float32)
    
def nzMSE1(ytrue, ypred):
    ypred = ypred * tf.cast(ytrue != 99, tf.float32)
    ytrue = ytrue * tf.cast(ytrue != 99, tf.float32)
    used = maxArrivals - tf.reduce_sum(tf.cast(tf.equal(ytrue,0), dtype=tf.float32), axis=1)
    used = tf.where(tf.equal(used, 0.), 1., used)
    return K.mean(tf.reduce_sum(K.square(tf.squeeze(ypred)-ytrue),axis=1)/used)

def nzMSE2(ytrue, ypred):
    ypred = ypred * tf.cast(ytrue != 99, tf.float32)
    ytrue = ytrue * tf.cast(ytrue != 99, tf.float32)
    used = tf.reduce_sum(tf.cast(tf.greater(tf.reduce_sum(ytrue, axis=-1),0), dtype=tf.float32), axis=1)
    used = tf.where(tf.equal(used, 0.), 1., used)
    return K.mean(tf.reduce_sum(K.square(ypred-ytrue),axis=[1,2])/used)

def nzMSE(y_true, y_pred):
    y_pred = y_pred * tf.cast(y_true != 99, tf.float32)
    y_true = y_true * tf.cast(y_true != 99, tf.float32)
    return K.mean(K.square(y_pred-y_true))

# def nzBCE2(ytrue, ypred):
#     ypred = ypred * tf.cast(ytrue != 99, tf.float32)
#     ytrue = ytrue * tf.cast(ytrue != 99, tf.float32)
#     used = tf.reduce_sum(tf.cast(tf.greater(tf.reduce_sum(ytrue, axis=1),0), dtype=tf.float32), axis=1)
#     used = tf.where(tf.equal(used, 0.), 1., used)
#     return K.mean(tf.reduce_sum(BCE(ytrue, ypred),axis=1)/used)

def nzBCE(ytrue, ypred):
    ypred = ypred * tf.cast(ytrue != 99, tf.float32)
    ytrue = ytrue * tf.cast(ytrue != 99, tf.float32)
    used = maxArrivals - tf.reduce_sum(tf.cast(tf.equal(ytrue,0), dtype=tf.float32), axis=1)
    used = tf.where(tf.equal(used, 0.), 1., used)
    return K.mean(BCE(ytrue, ypred)/used)

# def nzBCE(y_true, y_pred):
#     y_pred = y_pred * tf.cast(y_true != 99, tf.float32)
#     y_true = y_true * tf.cast(y_true != 99, tf.float32)
#     return K.mean(BCE(y_true, y_pred))

def nzAccuracy(ytrue, ypred):
    used = matrixSize/(tf.reduce_sum(tf.cast(tf.greater(tf.reduce_sum(ytrue, axis=1),0), dtype=tf.float32), axis=1)**2)
    used = tf.where(tf.equal(used, 0.), 1., used)
    acc = tf.reduce_sum(tf.cast(ytrue==tf.round(ypred), dtype=tf.float32),axis=(1,2))/matrixSize
    return K.mean(acc*used - used + 1)

# def nzAccuracy(y_true, y_pred):
#     y_pred = tf.squeeze(y_pred) * tf.cast(y_true != 99, tf.float32)
#     y_true = y_true * tf.cast(y_true != 99, tf.float32)
#     return K.mean(binary_accuracy(y_true, y_pred))

def nzRecall(y_true, y_pred):
    y_pred = y_pred * tf.cast(y_true != 99, tf.float32)
    y_true = y_true * tf.cast(y_true != 99, tf.float32)
#     y_true = K.ones_like(y_true)
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    all_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    
    recall = true_positives / (all_positives + K.epsilon())
    return recall

def nzPrecision(y_true, y_pred):
    y_pred = y_pred * tf.cast(y_true != 99, tf.float32)
    y_true = y_true * tf.cast(y_true != 99, tf.float32)
#     y_true = K.ones_like(y_true)
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision
