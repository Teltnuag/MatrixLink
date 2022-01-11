import numpy as np
import pandas as pd
import json
from copy import deepcopy
# physical_devices = tf.config.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False
from tensorflow.keras.models import load_model
from obspy import UTCDateTime
from Utils import loss_haversine, loss_depth, loss_eventTime, nonzero_mse, evaluate
batch_size = 1024
minNucl = 3
minMerge = 1

def permute_seq(X, timeNormalize, associationWindow, maxArrivals):
    X0 = np.zeros((X.shape[0], maxArrivals, X.shape[1]))
    for i in range(X.shape[0]):
        i_start = i
        i_end = i + maxArrivals
        if i_end > X.shape[0]:
            i_end = X.shape[0]
        # Map picks for slice into new array
        X0[i_start,:(i_end-i_start),:] = X[i_start:i_end,:]
        # Set initial pick to t=0
        idx = np.where(X0[i,:,2] > 0)[0]
        X0[i,idx,2] -= X0[i,0,2]
        # Remove all times with t > event_window
        idx = np.where(X0[i,:,2] > associationWindow)[0]
        X0[i,idx,:] = 0
        # Normalize time values
        X0[i,:,2] /= timeNormalize
    return X0

def link_phases(Y, min_nucl, min_merge):
    clusters = []
    for i in range(Y.shape[0]):
        idx = np.where(Y[i,:] == 1)[0]
        if idx.size < min_nucl:
            continue
        idx += i
        idx = idx[np.where(idx < Y.shape[0])[0]]
        idx_set = set(idx)
        if len(clusters) == 0:
            clusters.append(idx_set)
            continue
        n_common = np.zeros(len(clusters))
        for j, cluster in enumerate(clusters):
            n_common[j] = len(cluster.intersection(idx_set))
        best = np.argmax(n_common)
        if n_common[best] < min_merge:
            clusters.append(idx_set)
        else:
            clusters[best].update(idx_set)
    return np.array(clusters)

def run_phaselink(X, labels, params):
    # Permute arrival matrix for all lags 
    print("Permuting sequence for all lags... ", end='')
    X_perm = permute_seq(X, params['timeNormalize'], params['associationWindow'], params['maxArrivals'])

    # Predict association labels for all windows
    Y_pred = np.zeros((X_perm.shape[0], X_perm.shape[1], 1))
    print("\rPredicting phase associations...  ", end='')
    for i in range(0, Y_pred.shape[0], batch_size):
        i_start = i
        i_stop = i + batch_size
        if i_stop > Y_pred.shape[0]:
            i_stop = Y_pred.shape[0]
        X_test = {"phase": X_perm[i_start:i_stop,:,3], "numerical_features": X_perm[i_start:i_stop,:,[0,1,2,4]]}
        Y_pred[i_start:i_stop] = associatorModel.predict(X_test)
    Y0 = np.round(Y_pred)
    print("\rLinking phases...               ", end='')
    clusters = link_phases(Y0, minNucl, minMerge)
    print("\r%d events detected initially.       " % len(clusters))

    # Remove duplicate phases and events below threshold
    print("Removing duplicate phases... ", end='')
    for i, cluster in enumerate(clusters):
        phases = {}
        for idx in cluster:
            if idx >= len(labels):
                continue
            phase = labels[idx]
            phase = (phase.ST_LAT, phase.ST_LON, phase.IPHASE)
            if phase not in phases:
                phases[phase] = [idx]
            else:
                phases[phase].append(idx)
        for key in phases:
            if len(phases[key]) > 1:
                sorted(phases[key])
                phases[key] = [phases[key][-1]]
        clusters[i] = [phases[key][0] for key in phases]

    clusters = [x for x in clusters if len(x) >= params['minArrivals']]
    print("{} events left after duplicate removal and applying threshold.".format(len(clusters)))

    # Pull arrivals from predicted events to predict locations for said events
    constructedEvents = np.zeros((len(clusters),params['maxLocatorArrivals'],5))
    for event in range(len(clusters)):
        constructedEvents[event,:len(clusters[event])] = [X[arrival] for arrival in clusters[event]][:params['maxLocatorArrivals']]
        constructedEvents[event,:,2] -= constructedEvents[event,0,2]
    constructedEvents[:,:,2] /= params['timeNormalize']
    constructedEvents = {"phase": constructedEvents[:,:,3], "numerical_features": constructedEvents[:,:,[0,1,2,4]]}
    
    # Predict event locations and times
    locations = locatorModel.predict(constructedEvents)

    # Write output file
    EID = 0
    for i, cluster in enumerate(clusters):
        idx = np.array(list(cluster))
        for j in idx:
            thisArrival = deepcopy(labels[j])
            thisArrival.EVID = EID
            lat,lon,dep,time = locations[i]
            thisArrival['LAT'] = lat
            thisArrival['LON'] = lon
            thisArrival['DEPTH'] = dep
            thisArrival['EV_TIME'] = time
            outputs.append(thisArrival)
        EID += 1
        print("\rWriting output event " + str(EID) + ' / ' + str(len(clusters)), end='')
    print()
    return len(clusters)

def processInput(params):
    print("Reading input file")
    phases = params['phases']
    extents = np.array(list(params['extents'][params['location']].values()))
    latRange = abs(extents[1] - extents[0])
    lonRange = abs(extents[3] - extents[2])
    X = []
    labels = []
    for i, r in inputs.iterrows(): # I can do this better
        phase = r.IPHASE
        time = UTCDateTime(r.TIME)
        lat = abs((r.ST_LAT - extents[0]) / latRange)
        lon = abs((r.ST_LON - extents[2]) / lonRange)
        otime = time - UTCDateTime(0)
        try:
            arrival = [lat, lon, otime, phases[phase], 1]
            X.append(arrival)
            labels.append(r)
        except Exception as e:
            print(e)
    X = np.array(X)
    idx = np.argsort(X[:,2])
    X = X[idx,:]
    X[:,2] -= X[0,2]
    labels = [labels[i] for i in idx]
    print("Finished processing input file: %d arrivals found" % len(labels))
    return X, labels

if __name__ == "__main__":
    with open("Parameters.json", "r") as f:
        params = json.load(f)

    associatorModel = load_model(params['models']['associator'], compile=False)
    locatorModel = load_model(params['models']['locator'], custom_objects={'loss_haversine':loss_haversine, 'loss_depth':loss_depth, 'loss_eventTime':loss_eventTime, 'nonzero_mse':nonzero_mse})

    global outputs
    inFiles = ['./Inputs/S1 1.0.gz', './Inputs/S1 0.5.gz', './Inputs/S1 0.25.gz']
    for i in range(len(inFiles)):
        inputs = pd.read_pickle(inFiles[i])
        params['evalInFile'] = inFiles[i]
        inputs = inputs[100000:105000]
        X, labels = processInput(params)
        outputs = []
        detections = run_phaselink(X, labels, params)
        outputs = pd.DataFrame(outputs)
        outputs.to_pickle(params['evalOutFile'])
        print("{} detections total".format(detections))
        evaluate(params, inputs, outputs, locatorModel, verbose=False)
    
    params['evalInFile'] = './Inputs/S1 TEST.gz'
    inputs = pd.read_pickle(params['evalInFile']).sort_values(by=['TIME'])
    X, labels = processInput(params)
    outputs = []
    detections = run_phaselink(X, labels, params)
    outputs = pd.DataFrame(outputs)
    outputs.to_pickle(params['evalOutFile'])
    print("{} detections total".format(detections))
    evaluate(params, inputs, outputs, locatorModel, verbose=False)