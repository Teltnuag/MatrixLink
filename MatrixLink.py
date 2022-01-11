import tensorflow as tf
# physical_devices = tf.config.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

import numpy as np
import pandas as pd
import json
from collections import deque
from math import ceil
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False
from tensorflow.keras.models import load_model
from obspy import UTCDateTime
from scipy.cluster.hierarchy import ward, fcluster
from scipy.spatial.distance import squareform
from Utils import nzBCE, nzAccuracy, nzHaversine, nzDepth, nzTime, nzMSE1, nzMSE2, evaluate

# Build permutation lists and matrices to predict on
def permute():
    outerWindow = params['associationWindow']
    minArrivals = params['minArrivals']
    maxArrivals = params['maxArrivals']
    edgeWindow = outerWindow/5
    numWindows = ceil((X[:,2].max() + edgeWindow*2) / edgeWindow)
    innerWindows = deque()
    X_perm = deque()
    start = -edgeWindow
    for window in range(numWindows):
        end = start+outerWindow
        windowArrivals = np.where((X[:,2] >= start) & (X[:,2] < end))[0]
        start += edgeWindow
        if len(windowArrivals) >= minArrivals:
            X_perm.append(windowArrivals[:maxArrivals])
            innerWindows.append(start)
            
            #Experimental
#             if len(windowArrivals) > maxArrivals:
#                 print(len(windowArrivals), end=' ')
#                 extras = len(windowArrivals) - maxArrivals
#                 for s in [s+1 for s in range(extras)]:
#                     X_perm.append(windowArrivals[s:s+maxArrivals])
#                     innerWindows.append(start)


    X_test = np.zeros((len(X_perm),maxArrivals,5))
    for i in range(len(X_perm)):
        X_test[i,:len(X_perm[i])] = X[X_perm[i]]
        X_test[i,:len(X_perm[i]),2] -= X_test[i,0,2]
    X_test[:,:,2] /= params['timeNormalize']
    return X_perm, X_test, innerWindows

def buildEvents(X_perm, X_test, Y_pred, innerWindows):
    # Get clusters for predicted matrix at index i
    def cluster(i):
        valids = np.where(X_test[i][:,4])[0]
        validPreds = Y_pred[0][i][valids,:len(valids)]
        L = 1-((validPreds.T + validPreds)/2)
        np.fill_diagonal(L,0)
        return fcluster(ward(squareform(L)), params['clusterStrength'], criterion='distance')

    innerWindow = params['associationWindow'] * (3/5)
    minArrivals = params['minArrivals']
    timeNormalize = params['timeNormalize']
    catalogue = pd.DataFrame(columns=labels.columns)
    events = deque()
    evid = 1
    for window in range(len(X_perm)):
        clusters = cluster(window)
        for c in np.unique(clusters):
            pseudoEventIdx = np.where(clusters == c)[0]
            pseudoEvent = X_perm[window][pseudoEventIdx]
            if len(pseudoEvent) >= minArrivals: #TODO remove this?
                event = X[pseudoEvent]
                # check for containment within inner window
                contained = (event[0,2] >= innerWindows[window]) & (event[-1,2] <= (innerWindows[window]+innerWindow))
                if contained:
                    candidate = labels.iloc[pseudoEvent].copy()
                    candidate['LAT'] = np.median(Y_pred[1][window][pseudoEventIdx][:,0])*latRange+extents[0]
                    candidate['LON'] = np.median(Y_pred[1][window][pseudoEventIdx][:,1])*lonRange+extents[2]
                    # candidate['DEPTH'] = np.median(Y_pred[2][window][pseudoEventIdx])*extents[4]
                    candidate['ETIME'] = candidate.TIME.iloc[0]+(np.median(Y_pred[2][window][pseudoEventIdx])*timeNormalize)
                    # check for existence in catalogue
                    overlap = candidate.ARID.isin(catalogue.ARID).sum()
                    if overlap == 0:
                        print("\rPromoting event " + str(evid), end='')
                        events.append(pseudoEvent)
                        candidate.EVID = evid
                        catalogue = catalogue.append(candidate)
                        evid += 1
                    elif len(pseudoEvent) > overlap:
                        catalogue.drop(catalogue[catalogue.ARID.isin(candidate.ARID)].index, inplace=True)
                        candidate.EVID = evid
                        catalogue = catalogue.append(candidate)
                        evid += 1
    catalogue = catalogue.groupby('EVID').filter(lambda x: len(x) >= minArrivals)
    print()
    return events, catalogue

def matrixLink(X, labels):
    print("Creating permutations... ", end='')
    X_perm, X_test, innerWindows = permute()
    print("predicting... ", end='')
    Y_pred = model.predict({"phase": X_test[:,:,3], "numerical_features": X_test[:,:,[0,1,2,4]]})
    print("clustering and building events...")
    events, catalogue = buildEvents(X_perm, X_test, Y_pred, innerWindows)
    
    return catalogue

def processInput():
    print("Reading input file... ", end='')
    X = []
    labels = []
    for i, r in inputs.iterrows(): # I can do this better
        phase = r.PHASE
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
    labels = pd.DataFrame([labels[i] for i in idx])
    print("%d arrivals found" % len(labels))
    return X, labels

if __name__ == "__main__":
    pd.options.display.float_format = "{:.2f}".format
    with open("Parameters.json", "r") as f:
        params = json.load(f)
    phases = params['phases']
    extents = np.array(list(params['extents'][params['location']].values())+[params['maxDepth'],params['maxStationElevation']])
    latRange = abs(extents[1] - extents[0])
    lonRange = abs(extents[3] - extents[2])
    model = load_model(params['model'], custom_objects={'nzBCE':nzBCE, 'nzMSE1':nzMSE1, 'nzMSE2':nzMSE2, 'nzAccuracy':nzAccuracy, 'nzHaversine':nzHaversine, 'nzDepth':nzDepth, 'nzTime':nzTime}, compile=True)

    inFiles = ['./Inputs/S1 50.gz', './Inputs/S1 25.gz', './Inputs/S1 15.gz', './Inputs/S1 00.gz']
    evals = {file:[] for file in inFiles}
    for i in range(len(inFiles)):
        inputs = pd.read_pickle(inFiles[i]).sort_values(by=['TIME'])
        params['evalInFile'] = inFiles[i]
        inputs = inputs[100000:105000]

        X, labels = processInput()
        outputs = matrixLink(X, labels)
        outputs.to_pickle(params['evalOutFile'])
        evals[inFiles[i]] = evaluate(params, inputs, outputs, verbose=False)

    print("Consolidated summary for:", params['model'])
#     print('File\tAHM\t Loc\t  Dep\t  Time')
#     for file in evals.keys():
#         print(file[-5:-3], "{:8.2f}".format(evals[file][0]), "{:8.2f}".format(evals[file][3]), "{:8.2f}".format(evals[file][1]), "{:8.2f}".format(evals[file][2]))
    print('File\tAHM\t Time\t  Location')
    for file in evals.keys():
        print(file[-5:-3], "{:8.2f}".format(evals[file][0]), "{:8.2f}".format(evals[file][1]), "{:8.2f}".format(evals[file][2]))