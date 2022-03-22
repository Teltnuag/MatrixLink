import pandas as pd
import numpy as np
import tensorflow as tf
import json
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
import cartopy.feature as cfeature
from matplotlib.colors import Normalize
from collections import deque
from tensorflow.keras import backend as K
from geographiclib.geodesic import Geodesic
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

def trainingResults2(logs):
    loss = logs['loss']
    val_loss = logs['val_loss']
    association_loss = logs['association_loss']
    val_association_loss = logs['val_association_loss']
    location_loss = logs['location_loss']
    val_location_loss = logs['val_location_loss']
    location_loss_haversine = logs['location_nzHaversine']
    val_location_loss_haversine = logs['val_location_nzHaversine']
    val_association_nzAccuracy = logs['val_association_nzAccuracy']
    association_precision = logs['association_precision']
    val_association_precision = logs['val_association_precision']
    association_recall = logs['association_recall']
    val_association_recall = logs['val_association_recall']
    epochs = range(len(loss))

    plt.plot(epochs, loss, 'tab:red')
    plt.plot(epochs, val_loss, 'tab:blue')
    plt.plot(epochs, association_loss, 'tab:orange')
    plt.plot(epochs, val_association_loss, 'tab:cyan')
    plt.plot(epochs, location_loss, 'coral')
    plt.plot(epochs, val_location_loss, 'aquamarine')
    plt.title('Losses')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend(["Training Total", "Validation Total", "Training Association", "Validation Association", "Training Location", "Validation Location"])
    plt.figure()

    plt.plot(epochs, location_loss_haversine, 'tab:red')
    plt.plot(epochs, val_location_loss_haversine, 'tab:blue')
    plt.title('Haversine Loss')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend(["Training", "Validation"])
    plt.figure()

    plt.plot(epochs, association_precision, 'tab:olive')
    plt.plot(epochs, association_recall, 'tab:blue')
    plt.title('Training Precision and Recall')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend(["Precision", "Recall"])
    plt.figure()

    plt.plot(epochs, val_association_nzAccuracy, 'tab:red')
    plt.plot(epochs, val_association_precision, 'tab:olive')
    plt.plot(epochs, val_association_recall, 'tab:blue')
    plt.title('Validation Precision and Recall')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend(["Accuracy", "Precision", "Recall"])
    plt.figure()
    
    # print(logs)
    
def trainingResults(logs):
    loss = logs['loss']
    association_loss = logs['association_loss']
    try:
        noise_loss = logs['noise_loss']
        noise_precision = logs['noise_nzPrecision']
        noise_recall = logs['noise_nzRecall']
    except:
        pass
    location_loss = logs['location_loss']
#     location_loss_haversine = logs['location_nzHaversine']
    try:
        time_loss = logs['time_loss']
        time_loss_nz = logs['time_nzTime']
    except:
        pass
    association_precision = logs['association_nzPrecision']
    association_recall = logs['association_nzRecall']
    epochs = range(len(loss))

    legend = ["Total", "Association"]
    plt.plot(epochs, loss, 'tab:red')
    plt.plot(epochs, association_loss, 'tab:orange')
#     plt.plot(epochs, location_loss, 'coral')
    try:
        plt.plot(epochs, noise_loss, 'tab:blue')
        legend.append("Noise")
    except:
        pass
    try:
        plt.plot(epochs, time_loss, 'tab:olive')
        legend.append("Time")
    except:
        pass
    plt.legend(legend)
    plt.title('Losses')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.figure()

    plt.plot(epochs, location_loss, 'tab:red')
    try:
        plt.plot(epochs, time_loss_nz, 'tab:olive')
        plt.title('Haversine/Time Loss')
    except:
        plt.title('Haversine Loss')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.figure()

    plt.plot(epochs, association_precision, 'tab:olive')
    plt.plot(epochs, association_recall, 'tab:blue')
    plt.title('Association Precision and Recall')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend(["Precision", "Recall"])
    plt.figure()

#     plt.plot(epochs, noise_precision, 'tab:olive')
#     plt.plot(epochs, noise_recall, 'tab:blue')
#     plt.title('Noise Precision and Recall')
#     plt.xlabel("Epochs")
#     plt.ylabel("Accuracy")
#     plt.legend(["Precision", "Recall"])
#     plt.figure()
    
    # print(logs)

def evaluate(params, inputs, outputs, verbose=False):
    # inputs = inputs.groupby('ORID').filter(lambda x: len(x) >= params['minArrivals'])
    eventMatches = {}
    ious = {}
    eventCount = outputs.EVID.nunique()
    actualEventCount = inputs[inputs.ORID != -1].groupby('ORID').filter(lambda x: len(x) >= params['minArrivals']).ORID.nunique()
    current = 0
    for eid, arrivals in outputs.groupby('EVID'):
        current += 1
        print("\rMatching event " + str(current) + ' / ' + str(eventCount), end='        ')
        maxInt = 0
        union = 0
        for oeid, oarrivals in inputs[(inputs.ORID != -1) & (inputs.ORID.isin(arrivals.ORID.unique()))].groupby('ORID'):
            intersection = len(arrivals.index.intersection(oarrivals.index))
            if intersection > maxInt:
                maxInt = intersection
                eventMatches[eid] = oeid
                union = len(arrivals.index.union(oarrivals.index))
            if maxInt == len(arrivals):
                break
        if maxInt == 0:
            eventMatches[eid] = -2
            union = len(arrivals)
        ious[eid] = [maxInt, union]
    print()

    overall = list(map(sum, zip(*ious.values())))
    fakeArrivalsUsed = len(outputs[outputs.ORID == -1.0])
    numFakeArrivals = len(inputs[inputs.EVID == -1])
    fakeArrivalRatio = 1.0 - fakeArrivalsUsed/numFakeArrivals if numFakeArrivals > 0 else 1.0
    fakeArrivalRatio = fakeArrivalRatio if fakeArrivalRatio <= 1.0 else 0.0
    createdEventRatio = eventCount/actualEventCount
    createdEventRatio = createdEventRatio if createdEventRatio <= 1.0 else actualEventCount/eventCount
    if verbose == True:
        for eid, oeid in eventMatches.items():
            print("Matched EIDs:", str(eid) + ' : ' + str(oeid), "- IoU:", ious[eid][0], '/', ious[eid][1], '=',  (ious[eid][0]/ious[eid][1]))
            print('Actual:', inputs[inputs.EVID == eid].ARID.values, '\nCreated:', outputs[outputs.EVID == oeid].ARID.values)
            print('Fake:', (outputs[outputs.EVID == oeid].ORID == 0).sum(), '\n')
    precision, recall, f1, splitEvents, mergedEvents, fakeEvents, missedEvents, evaledEvents = prlEvaluate(params, inputs, outputs, eventMatches)
    mergedRatio = (actualEventCount - mergedEvents) / actualEventCount
    missedRatio = (actualEventCount - missedEvents) / actualEventCount
    fakeEventRatio = (eventCount - fakeEvents) / eventCount
    ahm = (overall[0]/overall[1])*fakeArrivalRatio*mergedRatio*createdEventRatio*missedRatio*fakeEventRatio*f1*100
    print("-----------------------------")
    print("IoU evaluation for events with at least", params['minArrivals'], "arrivals for model", params['model'])
    print("Parameters - evalInFile: {}\t maxArrivals: {}\t associationWindow: {}\t clusterStrength: {}".format(params['evalInFile'], params['maxArrivals'], params['associationWindow'], params['clusterStrength']))
    print("Totals/Average IoU:", overall[0], '/', overall[1], '=', (overall[0]/overall[1]))
    print("Precision:", str(precision))
    print("Recall:", str(recall))
    print("F1:", str(f1))
    print("Fake Arrivals:", fakeArrivalsUsed, '/', numFakeArrivals)
    print("Created / Actual Events:", eventCount, '/', actualEventCount)
    print("Missed Events:", missedEvents)
    # print("Split Events:", splitEvents)
    print("Merged Events:", mergedEvents)
    print("Fake Events:", fakeEvents)
    print("Arbitrary Health Measure:", ahm)
    
    locationErrors = evaledEvents.groupby('EVID').LOCATION_ERROR.min()
    locationStats = locationErrors[locationErrors != -1].describe()

    print("\nLocation Errors Summary")
    print("Mean:{:8.2f}".format(locationStats[1]))
    print("STD:{:9.2f}".format(locationStats[2]))
    print("Min:{:9.2f}".format(locationStats[3]))
    print("25%:{:9.2f}".format(locationStats[4]))
    print("50%:{:9.2f}".format(locationStats[5]))
    print("75%:{:9.2f}".format(locationStats[6]))
    print("Max:{:9.2f}".format(locationStats[7]))
    
    try:
        timeErrors = evaledEvents.groupby('EVID').TIME_ERROR.min()
        timeStats = timeErrors[timeErrors != -1].describe()
        
        print("\nTime Errors Summary")
        print("Mean:{:8.2f}".format(timeStats[1]))
        print("STD:{:9.2f}".format(timeStats[2]))
        print("Min:{:9.2f}".format(timeStats[3]))
        print("25%:{:9.2f}".format(timeStats[4]))
        print("50%:{:9.2f}".format(timeStats[5]))
        print("75%:{:9.2f}".format(timeStats[6]))
        print("Max:{:9.2f}".format(timeStats[7]))
    except:
        pass
    
    locations = outputs[outputs.ORID != -1].groupby('EVID').first().reset_index()[['EV_LAT', 'EV_LON', 'LAT', 'LON']].values
    stations = np.unique(outputs[['ST_LAT', 'ST_LON']].values, axis=0)
    receivingStations = outputs[outputs.ORID != -1].groupby('EVID').head().reset_index()[['LAT', 'LON', 'ST_LAT', 'ST_LON']].values
    predsMap(locations, stations, receivingStations=None, outfile='map.png')
    print("-----------------------------")
    return [ahm, locationStats[1]]

def prlEvaluate(params, inputs, outputs, eventMatches):
    def azimuther(row):
        return Geodesic.WGS84.Inverse(row.ST_LAT, row.ST_LON, row.EV_LAT, row.EV_LON)['s12']/1000

    window = params['evalWindow']
    eventCount = len(eventMatches)
    current = 0
    missedEvents = inputs[inputs.ORID.isin(np.setdiff1d(inputs[inputs.EVID != -1].groupby('EVID').filter(lambda x: len(x) >= params['minArrivals']).ORID.unique(), outputs.ORID.unique()))]
    splitEvents = -1
    mergedEvents = 0
    fakeEvents = 0
    events = deque()
    for evid, orid in eventMatches.items():
        current += 1
        print("\rEvaluating event " + str(current) + ' / ' + str(eventCount), end='        ')
        inArrivals = inputs[inputs.ORID == orid]
        outArrivals = outputs[outputs.EVID == evid]
        start = inArrivals.TIME.min()
        start = start - window if start <= outArrivals.TIME.min() else outArrivals.TIME.min() - window
        end = inArrivals.TIME.max()
        end = end + window if end >= outArrivals.TIME.max() else outArrivals.TIME.max() + window
        windowArrivals = inputs[(inputs.TIME >= start) & (inputs.TIME <= end)]
        inIds = inArrivals.ARID.values
        outIds = outArrivals.ARID.values
        realOutArrivals = outArrivals[outArrivals.ORID != -1]
        if len(realOutArrivals) > 0:
            locationError = haversine(realOutArrivals.EV_LAT.mode().min(), realOutArrivals.EV_LON.mode().min(), realOutArrivals.LAT.iloc[0], realOutArrivals.LON.iloc[0])
            if realOutArrivals.ORID.nunique() > 1:
                mergedEvents += 1
        else:
            locationError = -1
            fakeEvents += 1
        for _, arrival in windowArrivals.iterrows():
            if arrival.ARID in inIds:
                ruling = 'TP' if arrival.ARID in outIds else 'FN'
            else:
                ruling = 'FP' if arrival.ARID in outIds else 'TN'
            events.append({'EVID': evid,
                                    'ORID': orid,
                                    'ARID': arrival.ARID,
                                    'TIME': arrival.TIME,
                                    'PHASE': arrival.PHASE,
                                    'DISTANCE': azimuther(arrival),
                                    'LOCATION_ERROR': locationError,
                                    # 'DEPTH_ERROR': abs(inArrivals.EV_DEPTH.max() - outArrivals.DEPTH.max()),
                                    'TIME_ERROR': abs(inArrivals.EV_TIME.mode().min() - outArrivals.ETIME.max()),
                                    'FAKE': arrival.ARID < 0,
                                    'RULING': ruling})
    for _, arrival in missedEvents.iterrows():
        events.append({'EVID': arrival.EVID,
                                'ORID': arrival.ORID,
                                'ARID': arrival.ARID,
                                'TIME': arrival.TIME,
                                'PHASE': arrival.PHASE,
                                'DISTANCE': azimuther(arrival),
                                'LOCATION_ERROR': -1,
                                # 'DEPTH_ERROR': -1,
                                'TIME_ERROR': -1,
                                'FAKE': False,
                                'RULING': 'FN'})
    events = pd.DataFrame(events)
    missedEvents = missedEvents.ORID.nunique()
    events.sort_values(by=['EVID', 'TIME']).to_pickle(params['prlEvalOutFile'])
    values = {'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0}
    values.update(events.RULING.value_counts().to_dict())
    try:
        precision = values['TP'] / (values['TP'] + values['FP'])
    except:
        precision = 0.0 # because there were 0 true and false positives
    try:
        recall = values['TP'] / (values['TP'] + values['FN'])
    except:
        recall = 0.0 # because there were 0 true and false negatives
    f1 = 2*precision*recall/(precision+recall)
    print()
    return precision, recall, f1, splitEvents, mergedEvents, fakeEvents, missedEvents, events

# Generate a map of observed and predicted locations
# locations - array of:
#    [[obsLat1, obsLon1, predLat1, predLon1],
#     [obsLat2, obsLon2, predLat2, predLon2],
#     ...,
#     [obsLat99, obsLon99, predLat99, predLon99]]
# stations - array of:
#    [[staLat1, staLon1],
#     [staLat2, staLon2],
#     ...,
#     [staLat99, staLon99]]
# outfile - file to save map to
def predsMap(locations, stations, receivingStations=None, outfile=None):
    extend = 1
    lat_min = extents[0] - extend
    lat_max = extents[1] + extend
    lon_min = extents[2] - extend
    lon_max = extents[3] + extend
#     lat_min = locations[:,[0,2]].min() - extend
#     lat_max = locations[:,[0,2]].max() + extend
#     lon_min = locations[:,[1,3]].min() - extend
#     lon_max = locations[:,[1,3]].max() + extend
    
    locationsR = locations*0.017453292519943295
    dlat_dlon = (locationsR[:,[0,1]] - locationsR[:,[2,3]]) / 2
    a = tf.sin(dlat_dlon[:,0])**2 + tf.cos(locationsR[:,0]) * tf.cos(locationsR[:,2]) * tf.sin(dlat_dlon[:,1])**2
    diff = 2*tf.asin(tf.sqrt(a))*6378.1

    cmap = plt.cm.rainbow
    norm = Normalize(vmin=np.quantile(diff, 0.05), vmax=np.quantile(diff, 0.95))
    colors = cmap(norm(diff))
    transform = ccrs.Geodetic()
    projection = ccrs.Robinson()
    
    fig, ax = plt.subplots(figsize=(25,25), subplot_kw={'projection': projection}, sharex=True, sharey=True)
    states_provinces = cfeature.NaturalEarthFeature(
        category='cultural',
        name='admin_1_states_provinces_lines',
        scale='50m',
        facecolor='none')
    ax.add_image(cimgt.Stamen('terrain-background'), 4)
    ax.add_feature(states_provinces, edgecolor='gray')
    ax.coastlines('110m')
    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.RIVERS)
    ax.add_feature(cfeature.LAKES)
    
    ax.plot(stations[:,1], stations[:,0], 'o', markersize=7, c='k', transform=transform)
    ax.plot(locations[:,1], locations[:,0], 'o', markersize=3, c='g', transform=transform)
    ax.plot(locations[:,3], locations[:,2], 'o', markersize=3, c='r', alpha=0.7, transform=transform)
    for e in range(len(locations)):
        ax.plot([locations[e,1], locations[e,3]], [locations[e,0], locations[e,2]], color=colors[e], alpha=0.5, transform=transform)
    if receivingStations is not None:
        for p in range(len(receivingStations)):
            pred = receivingStations[p][0:2]
            stas = receivingStations[p][2:4]
            ax.plot([pred[1], stas[1]], [pred[0], stas[0]], color='black', alpha=0.4, transform=transform)
    
#     ax.set_extent([lon_min, lon_max, lat_min, lat_max])
#     ax.legend(['Stations','Observed','Predicted'])

    fig.canvas.draw()
    fig.tight_layout(pad=0, w_pad=1, h_pad=0)
    if outfile is not None:
        fig.savefig(outfile)
    plt.show()

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

# def nzMSE(ytrue, ypred):
#     if tf.equal(tf.shape(ypred)[-1],1):
#         m = maxArrivals/(tf.reduce_sum(tf.cast(tf.greater(ytrue,0), dtype=tf.float32), axis=1))
#         m = tf.where(tf.math.is_inf(m), 1., m)
#         return K.mean(tf.reduce_sum(K.square(tf.squeeze(ypred)-ytrue),axis=1)*m)
#     else:
#         m = maxArrivals/(tf.reduce_sum(tf.cast(tf.greater(tf.reduce_sum(ytrue, axis=-1),0), dtype=tf.float32), axis=1))
#         m = tf.where(tf.math.is_inf(m), 1., m)
#         return K.mean(tf.reduce_sum(K.square(ypred-ytrue),axis=[1,2])*m)

# def nzMSE(ytrue, ypred):
#     return tf.cond(tf.equal(ypred.shape[-1],2), lambda: nzMSEtwo(ytrue, ypred), lambda: nzMSEone(ytrue, ypred))
    
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

# def nzAccuracy(ytrue, ypred):
#     used = matrixSize/(tf.reduce_sum(tf.cast(tf.greater(tf.reduce_sum(ytrue, axis=1),0), dtype=tf.float32), axis=1)**2)
#     used = tf.where(tf.equal(used, 0.), 1., used)
#     acc = tf.reduce_sum(tf.cast(ytrue==tf.round(ypred), dtype=tf.float32),axis=(1,2))/matrixSize
#     return K.mean(acc*used - used + 1)

def nzAccuracy(y_true, y_pred):
    y_pred = tf.squeeze(y_pred) * tf.cast(y_true != 99, tf.float32)
    y_true = y_true * tf.cast(y_true != 99, tf.float32)
    return K.mean(binary_accuracy(y_true, y_pred))

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