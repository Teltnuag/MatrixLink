#MatrixLink
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
from Utils import nzBCE, nzMSE1, nzMSE2, nzHaversine, nzAccuracy, nzPrecision, nzRecall, nzTime, evaluate

# Build permutation lists and matrices to predict on
def permute(X):
    outerWindow = params['associationWindow']
    minArrivals = params['minArrivals']
    maxArrivals = params['maxArrivals']
    edgeWindow = outerWindow/5
    numWindows = ceil((X[:,2].max() + edgeWindow*2) / edgeWindow)
    start = -edgeWindow

    innerWindows = deque()
    X_perm = deque()
    for window in range(numWindows):
        print('\rCreating permutations... ' + str(window) + ' / ' + str(numWindows), end='')
        end = start+outerWindow
        windowArrivals = np.where((X[:,2] >= start) & (X[:,2] < end))[0]
        start += edgeWindow
        if len(windowArrivals) >= minArrivals:
            X_perm.append(windowArrivals[:maxArrivals])
            innerWindows.append(start)
    X_test = np.zeros((len(X_perm),maxArrivals,5))
    for i in range(len(X_perm)):
        X_test[i,:len(X_perm[i])] = X[X_perm[i]]
        X_test[i,:len(X_perm[i]),2] -= X_test[i,0,2]
    X_test[:,:,2] /= params['timeNormalize']
    return X_perm, X_test, innerWindows

def buildEvents(X, labels, X_perm, X_test, Y_pred, innerWindows):
    # Get clusters for predicted matrix at index i
    def cluster(i):
        valids = np.where(X_test[i][:,-1])[0]
        validPreds = Y_pred[0][i][valids,:len(valids)]
        L = 1-((validPreds.T + validPreds)/2)
        np.fill_diagonal(L,0)
        return fcluster(ward(squareform(L)), params['clusterStrength'], criterion='distance')

    innerWindow = params['associationWindow'] * (3/5)
    minArrivals = params['minArrivals']
    catalogue = pd.DataFrame(columns=labels.columns)
    evid = 1
    created = 1
    for window in range(len(X_perm)):
        clusters = cluster(window)
        for c in np.unique(clusters):
            pseudoEventIdx = np.where(clusters == c)[0]
            pseudoEvent = X_perm[window][pseudoEventIdx]
            if len(pseudoEvent) >= minArrivals:
                event = X[pseudoEvent]
                # check for containment within inner window
                contained = (event[0,2] >= innerWindows[window]) & (event[-1,2] <= (innerWindows[window]+innerWindow))
                if contained:
                    candidate = labels.iloc[pseudoEvent].copy()
                    try:
                        candidate['ETIME'] = candidate.TIME.min() + np.median(Y_pred[3][window][pseudoEventIdx][:]*params['timeNormalize'])
                    except:
                        candidate['ETIME'] = -1
                    candidate['PLAT'] = Y_pred[1][window][pseudoEventIdx][:,0]*latRange+extents[0]
                    candidate['PLON'] = Y_pred[1][window][pseudoEventIdx][:,1]*lonRange+extents[2]
#                     candidate['LAT'] = np.median(Y_pred[1][window][pseudoEventIdx][:,0])*latRange+extents[0]
#                     candidate['LON'] = np.median(Y_pred[1][window][pseudoEventIdx][:,1])*lonRange+extents[2]
                    candidate['LAT'] = np.median(candidate.PLAT)
                    candidate['LON'] = np.median(candidate.PLON)
                    # check for existence in catalogue
                    overlap = candidate.ARID.isin(catalogue.ARID).sum()
                    if overlap == 0:
                        print("\rPromoting event " + str(created), end='')
                        candidate.EVID = evid
                        catalogue = catalogue.append(candidate)
                        evid += 1
                        created += 1
                    # elif len(pseudoEvent) > overlap:
                    #     catalogue.drop(catalogue[catalogue.ARID.isin(candidate.ARID)].index, inplace=True)
                    #     candidate.EVID = evid
                    #     catalogue = catalogue.append(candidate)
                    #     evid += 1
    catalogue = catalogue.groupby('EVID').filter(lambda x: len(x) >= minArrivals)
    print()
    return catalogue

def matrixLink(X, labels, denoise=False):
    print("Creating permutations... ", end='')
    X_perm, X_test, innerWindows = permute(X)
    print("\nMaking initial predictions... ", end='')
    Y_pred = model.predict({"phase": X_test[:,:,3], "numerical_features": X_test[:,:,[0,1,2,4]]})
    if denoise:
        print("\nEliminating noise and predicting again... ", end='')
        for _ in range(3):
            valids = deque()
            for i in range(len(X_perm)):
                noise = np.where(Y_pred[2][i] > 0.008)[0]
                valids.append(np.delete(X_perm[i], noise[noise < len(X_perm[i])]))
            valids = np.array(list(set(np.concatenate(valids))))
            X = X[valids]
            labels = labels.iloc[valids]

            X_perm, X_test, innerWindows = permute(X)
            Y_pred = model.predict({"phase": X_test[:,:,3], "numerical_features": X_test[:,:,[0,1,2,4]]})
    print("clustering and building events...")
    catalogue = buildEvents(X, labels, X_perm, X_test, Y_pred, innerWindows)
    return catalogue

def processInput(oneHot = False):
    print("Reading input file... ", end='')
    X = []
    labels = []
    for i, r in inputs.iterrows(): # I can do this better
        phase = phases[r.PHASE]
        time = UTCDateTime(r.TIME)
        lat = abs((r.ST_LAT - extents[0]) / latRange)
        lon = abs((r.ST_LON - extents[2]) / lonRange)
        otime = time - UTCDateTime(0)
        try:
            if oneHot:
                arrival = [lat, lon, otime, 0, 0, 0, 0, 1]
                arrival[3+phase] = 1
            else:
                arrival = [lat, lon, otime, phase, 1]
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
    model = load_model(params['model'], custom_objects={'nzBCE':nzBCE, 'nzMSE':nzMSE2, 'nzMSE1':nzMSE1, 'nzMSE2':nzMSE2, 'nzHaversine':nzHaversine, 'nzPrecision':nzPrecision, 'nzRecall':nzRecall, 'nzAccuracy':nzAccuracy, 'nzTime':nzTime}, compile=True)
    denoise = False
    inputs = pd.read_pickle(params['evalInFile']).sort_values(by=['TIME']).reset_index(drop=True)[:3000]
    X, labels = processInput()
    outputs = matrixLink(X, labels, denoise)
    outputs.to_pickle(params['evalOutFile'])
    evaluate(params, inputs, outputs, verbose=False)