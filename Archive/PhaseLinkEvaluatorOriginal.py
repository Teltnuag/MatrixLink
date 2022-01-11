import sys
import numpy as np
import pandas as pd
import json
import copy
import torch
import torch.utils.data
from obspy import UTCDateTime
from geopy.distance import geodesic

batch_size = 1024

class StackedGRU(torch.nn.Module):
    def __init__(self):
        super(StackedGRU, self).__init__()
        self.hidden_size = 128
        self.fc1 = torch.nn.Linear(5, 32)
        self.fc2 = torch.nn.Linear(32, 32)
        self.fc3 = torch.nn.Linear(32, 32)
        self.fc4 = torch.nn.Linear(32, 32)
        self.fc5 = torch.nn.Linear(32, 32)
        self.fc6 = torch.nn.Linear(2*self.hidden_size, 1)
        self.gru1 = torch.nn.GRU(32, self.hidden_size, \
            batch_first=True, bidirectional=True)
        self.gru2 = torch.nn.GRU(self.hidden_size*2, self.hidden_size, \
            batch_first=True, bidirectional=True)

    def forward(self, inp):
        out = self.fc1(inp)
        out = torch.nn.functional.relu(out)
        out = self.fc2(out)
        out = torch.nn.functional.relu(out)
        out = self.fc3(out)
        out = torch.nn.functional.relu(out)
        out = self.fc4(out)
        out = torch.nn.functional.relu(out)
        out = self.fc5(out)
        out = torch.nn.functional.relu(out)
        out = self.gru1(out)
        h_t = out[0]
        out = self.gru2(h_t)
        h_t = out[0]
        out = self.fc6(h_t)
        out = torch.sigmoid(out)
        return out

class Arrival():
    def __init__(self, net, sta, time, phase):
        self.net = net
        self.sta = sta
        self.time = time
        self.phase = phase

class Event():
    def __init__(self, arrivals = None):
        if arrivals is not None:
            self.arrivals = arrivals
        else:
            self.arrivals = []

def permute_seq(X, t_win, max_picks):
    X0 = np.zeros((X.shape[0], max_picks, X.shape[1]))
    for i in range(X.shape[0]):
        i_start = i
        i_end = i + max_picks
        if i_end > X.shape[0]:
            i_end = X.shape[0]

        # Map picks for slice into new array
        X0[i_start,:(i_end-i_start),:] = X[i_start:i_end,:]

        # Set initial pick to t=0
        idx = np.where(X0[i,:,2] > 0)[0]
        X0[i,idx,2] -= X0[i,0,2]

        # Remove all times with t > t_win
        idx = np.where(X0[i,:,2] > t_win)[0]
        X0[i,idx,:] = 0

        # Normalize time values
        X0[i,:,2] /= t_win
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

def repick_event(cluster, X, params):
    idx = np.array(list(cluster))
    idx = idx[np.where(idx < X.shape[0])[0]]
    t_start = np.min(X[idx,2]) - params['t_repick']/2.0
    t_stop = np.max(X[idx,2]) + params['t_repick']/2.0
    return np.where((X[:,2] >= t_start) & (X[:,2] < t_stop))[0].astype(np.int32)

from numba import jit
@jit(nopython=True)
def back_project(cluster, X, indices, tt_p, tt_s, max_pick_dist, phases, min_sep):
    best_cluster0 = np.zeros(cluster.size, dtype=np.int32)
    best_cluster = np.zeros(cluster.size, dtype=np.int32)
    arrival_times = np.zeros(cluster.size, dtype=np.float64)
    weights = np.zeros(cluster.size, dtype=np.int32)
    n_best = 0
    for i in range(tt_p.shape[1]):
        for j in range(tt_p.shape[2]):
            for k in range(tt_p.shape[3]):

                for l in range(cluster.size):
                    if phases[l] >= 1:
                        arrival_times[l] = tt_s[indices[l],i,j,k]
                        if arrival_times[l] <= max_pick_dist / 3.5:
                            weights[l] = 1
                        else:
                            weights[l] = 0
                    else:
                        arrival_times[l] = tt_p[indices[l],i,j,k]
                        if arrival_times[l] <= max_pick_dist / 6.0:
                            weights[l] = 1
                        else:
                            weights[l] = 0

                tt_diff = X[cluster,2] - arrival_times

                n_best2 = 0
                for l in range(len(cluster)):
                    start = tt_diff[l]
                    stop = start + min_sep
                    idx = np.where(
                        np.logical_and(tt_diff >= start, tt_diff < stop)
                    )[0]
                    if np.sum(weights[idx]) > n_best2:
                        best_cluster0[:] = 0
                        best_cluster0[idx] = 1
                        best_cluster0 *= weights
                        n_best2 = np.sum(best_cluster0)

                if n_best2 > n_best:
                    n_best = n_best2
                    best_cluster[:] = best_cluster0

    return cluster[np.where(best_cluster==1)[0]]

class tt_interp:
    def __init__(self, ttfile, datum):
        with open(ttfile, 'r') as f:
            count = 0
            for line in f:
                if count == 0:
                    count += 1
                    continue
                elif count == 1:
                    n_dist, n_depth = line.split()
                    n_dist = int(n_dist)
                    n_depth = int(n_depth)
                    dists = np.zeros(n_dist)
                    tt = np.zeros((n_depth, n_dist))
                    count += 1
                    continue
                elif count == 2:
                    depths = line.split()
                    depths = np.array([float(x) for x in depths])
                    count += 1
                    continue
                else:
                    temp = line.split()
                    temp = np.array([float(x) for x in temp])
                    dists[count-3] = temp[0]
                    tt[:, count-3] = temp[1:]
                    count += 1
        self.tt = tt
        self.dists = dists
        self.depths = depths
        self.datum = datum

        from scipy.interpolate import RectBivariateSpline
        self.interp_ = RectBivariateSpline(self.depths, self.dists, self.tt)

    def interp(self, dist, depth):
        return self.interp_.ev(depth + self.datum, dist)

def build_tt_grid(params):
    NX = params['n_x_nodes']
    NY = params['n_y_nodes']
    NZ = params['n_z_nodes']
    x = np.linspace(params['lon_min'], params['lon_max'], NX)
    y = np.linspace(params['lat_min'], params['lat_max'], NY)
    z = np.linspace(params['z_min'], params['z_max'], NZ)

    # Pwaves
    pTT = tt_interp(params['tt_table']['P'], params['datum'])
    print('Read pTT')
    print('(dep,dist) = (0,0), (10,0), (0,10), (10,10):')
    print('             {:.3f}   {:.3f}   {:.3f}   {:.3f}'.format(
        pTT.interp(0,0).item(), pTT.interp(10,0).item(),pTT.interp(0,10).item(),
        pTT.interp(10,10).item()))

    #Swaves
    sTT = tt_interp(params['tt_table']['S'], params['datum'])
    print('Read sTT')
    print('(dep,dist) = (0,0), (10,0), (0,10), (10,10):')
    print('             {:.3f}   {:.3f}   {:.3f}   {:.3f}'.format(
        sTT.interp(0,0).item(), sTT.interp(10,0).item(),sTT.interp(0,10).item(),
        sTT.interp(10,10).item()))

    n_stations = 0
    with open(params['station_file']) as f:
        for line in f:
            n_stations += 1

    tt_p = np.zeros((n_stations, NY, NX, NZ), dtype=np.float32)
    tt_s = np.zeros((n_stations, NY, NX, NZ), dtype=np.float32)
    station_index_map = {}

    n_stations = 0
    with open(params['station_file']) as f:
        for line in f:
            try:
                net, sta, lat, lon, elev = line.split()
            except:
                net, sta, lat, lon = line.split()
            station_index_map[(net, sta)] = n_stations
            print("\rBuilding travel time grid for station " + str(n_stations+1), end='')
            for i in range(NX-1):
                for j in range(NY-1):
                    dist = geodesic((y[j], x[i]), (lat, lon)).km
                    for k in range(NZ-1):
                        tt_p[n_stations, j, i, k] = pTT.interp(dist, z[k])
                        tt_s[n_stations, j, i, k] = sTT.interp(dist, z[k])
            n_stations += 1
    print()
    np.save(params["tt_p_file"], tt_p)
    np.save(params["tt_s_file"], tt_s)
    np.save(params["station_index_map_file"], station_index_map)
    return tt_p, tt_s, station_index_map

def build_idx_maps(labels, new_clust, station_index_map):
    phase_idx = {'Pg': 0, 'Pn': 1, 'P': 0, 'Pb': 0, 'Sg': 2, 'S': 2, 'Sn': 3, 'Sb': 2, 'Lg': 2}
    indices = []
    phases = []
    for idx in new_clust:
        net, sta, phase, _ = labels[idx].split()
        indices.append(station_index_map[(net, sta)])
        phases.append(phase_idx[phase])
    phases = np.array(phases, dtype=np.int32)
    indices = np.array(indices, dtype=np.int32)
    return phases, indices

def run_phaselink(X, labels, params, tt_p, tt_s, station_index_map):
    # Permute pick matrix for all lags 
    print("Permuting sequence for all lags... ", end='')
    X_perm = permute_seq(X, params['t_win'], params['max_picks'])
    X_perm = torch.from_numpy(X_perm).float().to(device)
    print("\rFinished permuting sequence.     ", end='')

    # Predict association labels for all windows
    Y_pred = torch.zeros((X_perm.size(0), X_perm.size(1), 1)).float()
    Y_pred = Y_pred.to(device)
    print("\rPredicting labels for all phases...", end='')
    for i in range(0, Y_pred.shape[0], batch_size):
        i_start = i
        i_stop = i + batch_size
        if i_stop > Y_pred.shape[0]:
            i_stop = Y_pred.shape[0]
        X_test = X_perm[i_start:i_stop]
        with torch.no_grad():
            Y_pred[i_start:i_stop] = model(X_test)
    Y_pred = Y_pred.view(Y_pred.size(0), Y_pred.size(1))
    print("\rFinished label prediction.         ", end='')
    Y0 = torch.round(Y_pred).cpu().numpy()
    print("\rLinking phases...         ", end='')
    clusters = link_phases(Y0, params['min_nucl'], params['min_merge'])
    print("\r%d events detected initially.       " % len(clusters))

    # Remove events below threshold
    print("Removing duplicates... ", end='')
    for i, cluster in enumerate(clusters):
        phases = {}
        for idx in cluster:
            if idx >= len(labels):
                continue
            net, sta, phase, time = labels[idx].split()
            if (net, sta, phase) not in phases:
                phases[(net, sta, phase)] = [idx]
            else:
                phases[(net, sta, phase)].append(idx)
        for key in phases:
            if len(phases[key]) > 1:
                sorted(phases[key])
                phases[key] = [phases[key][-1]]
        clusters[i] = [phases[key][0] for key in phases]
    # clusters = [x for x in clusters if len(x) >= params['min_det']]
    print("\r%d events detected after duplicate removal." % len(clusters))

    if params['back_project'] == "True":
        # Repick and back-project to clean up
        for i, cluster in enumerate(clusters):
            new_clust = repick_event(cluster, X, params)
            phases, indices = build_idx_maps(labels, new_clust, station_index_map)
            clusters[i] = back_project(
                new_clust, X, indices, tt_p, tt_s,
                params['max_pick_dist'], phases, params['min_sep']
            )
            print("\rBackproject {}: {} -> {} -> {} picks   ".format(i, len(cluster), len(new_clust), len(clusters[i])), end='')

    # Remove events below threshold
    clusters = [x for x in clusters if len(x) >= params['min_det']]
    print("\n{} events left after applying min_det threshold.".format(len(clusters)))

    # Write out solutions
    EID = 0
    for i, cluster in enumerate(clusters):
        idx = np.array(list(cluster))
        for j in idx:
            net, sta, phase, time = labels[j].split()
            otime = int(UTCDateTime(time) - UTCDateTime(0))
            thisPick = copy.deepcopy(inputPicks[(inputPicks.STA == sta) & (inputPicks.IPHASE == phase) & (inputPicks.TIME == otime)])
            thisPick.EVID = EID
            outputPicks.append(thisPick)
        EID += 1
        print("\rWriting output event " + str(EID) + ' / ' + str(len(clusters)), end='')
    print()
    return len(clusters)

def detect_events(X, Y, model, params):
    global outputPicks
    outputPicks = []
    n_cumul_dets = 0
    X[:,2] -= X[0,2]

    if params['back_project']:
        try:
            tt_p = np.load(params["tt_p_file"], allow_pickle=True)
            tt_s = np.load(params["tt_s_file"], allow_pickle=True)
            station_index_map = np.load(params["station_index_map_file"], allow_pickle=True).item()
            print("Travel time grids loaded.")
        except:
            print("Could not load travel time grids. Building from scratch.")
            tt_p, tt_s, station_index_map = build_tt_grid(params)
    else:
        tt_p, tt_s, station_index_map = None, None, None
        
    n_cumul_dets = run_phaselink(X, labels, params, tt_p, tt_s, station_index_map)
    
    outputPicks = pd.concat(outputPicks)
    outputPicks.to_pickle(params['eval_out_file'])
    print("{} detections total".format(n_cumul_dets))
    return n_cumul_dets

def processInput(params):
    print("Reading input file")
    stations = np.load(params["station_map_file"],allow_pickle=True)
    networkLookup = {k[1]: k[0] for k,v in stations.items()}
    phase_idx = {'Pg': 0, 'Pn': 1, 'P': 0, 'Pb': 0, 'Sg': 2, 'S': 2, 'Sn': 3, 'Sb': 2, 'Lg': 2}
    X = []
    labels = []
    missing_sta_list = set()
    for i, r in inputPicks.iterrows():
        net = networkLookup[r.STA]
        sta = r.STA
        phase = r.IPHASE
        time = UTCDateTime(r.TIME)
        try:
            sta_X, sta_Y = stations[(net, sta)]
        except:
            if (net, sta) not in missing_sta_list:
                print("%s %s missing from station list" % (net, sta))
                missing_sta_list.add((net, sta))
            continue
        otime = time - UTCDateTime(0)
        pick = [sta_X, sta_Y, otime, phase_idx[phase], 1]
        X.append(pick)
        labels.append("%s %s %s %s" % (net, sta, phase, time))
    X = np.array(X)
    idx = np.argsort(X[:,2])
    X = X[idx,:]
    labels = [labels[i] for i in idx]
    print("Finished reading input file: %d picks found." % len(labels))
    return X, labels

def evaluate(detections, verbose=False, biou=False):
    inputPicksGrouped = (inputPicks[inputPicks.EVID != -1].groupby('EVID').filter(lambda x: len(x) >= params['min_det'])).groupby('EVID')
    outputPicksGrouped = outputPicks.groupby('EVID')
    eventMatch1 = {}
    eventMatch2 = {}
    ious1 = {}
    ious2 = {}
    actualEventCount = len(inputPicksGrouped)
    current = 0
    for eid, picks in inputPicksGrouped:
        current += 1
        print("\rEvaluating actual event " + str(current) + ' / ' + str(actualEventCount), end='        ')
        maxInt = 0
        union = 0
        for oeid, opicks in outputPicksGrouped:
            intersection = len(picks.index.intersection(opicks.index))
            if intersection != 0 and intersection > maxInt:
                maxInt = intersection
                eventMatch1[eid] = oeid
                union = len(picks.index.union(opicks.index))
            if maxInt == 1.0:
                break
        if maxInt == 0:
            eventMatch1[eid] = -2
            union = len(picks)
        ious1[eid] = [maxInt, union]
    print()
    
    if biou:
        current = 0
        for eid, picks in outputPicksGrouped:
            current += 1
            print("\rEvaluating picked event " + str(current) + ' / ' + str(detections), end='        ')
            maxInt = 0
            union = 0
            for oeid, opicks in inputPicksGrouped:
                intersection = len(picks.index.intersection(opicks.index))
                if intersection != 0 and intersection > maxInt:
                    maxInt = intersection
                    eventMatch2[eid] = oeid
                    union = len(picks.index.union(opicks.index))
                if maxInt == 1.0:
                    break
            if maxInt == 0:
                eventMatch2[eid] = -2
                union = len(picks)
            ious2[eid] = [maxInt, union]
        print()

    overall1 = list(map(sum, zip(*ious1.values())))
    overall2 = list(map(sum, zip(*ious2.values()))) if biou else [1.0, 1.0]
    fakePicksMade = len(outputPicks[outputPicks.ORID == 0.0])
    numFakePicks = len(inputPicks[inputPicks.EVID == -1])
    fakePickRatio = 1.0 - fakePicksMade/numFakePicks if numFakePicks > 0 else 1.0
    fakePickRatio = fakePickRatio if fakePickRatio <= 1.0 else 0.0
    pickedRatio = detections/actualEventCount
    pickedRatio = pickedRatio if pickedRatio <= 1.0 else actualEventCount/detections
    if(verbose == "True"):
        for eid, oeid in eventMatch1.items():
            print("Matched EIDs:", str(eid) + ' : ' + str(oeid), "- IoU:", ious1[eid][0], '/', ious1[eid][1], '=',  (ious1[eid][0]/ious1[eid][1]))
            print('Actual:', inputPicks[inputPicks.EVID == eid].ARID.values, '\nPicked:', outputPicks[outputPicks.EVID == oeid].ARID.values)
            print('Fake:', (outputPicks[outputPicks.EVID == oeid].ORID == 0).sum(), '\n')
        for eid, oeid in eventMatch2.items():
            print("Matched EIDs:", str(eid) + ' : ' + str(oeid), "- IoU:", ious2[eid][0], '/', ious2[eid][1], '=',  (ious2[eid][0]/ious2[eid][1]))
            print('Actual:', inputPicks[inputPicks.EVID == oeid].ARID.values, '\nPicked:', outputPicks[outputPicks.EVID == eid].ARID.values)
            print('Fake:', (outputPicks[outputPicks.EVID == eid].ORID == 0).sum(), '\n')
    print("\nIoU evaluation for events with at least", params['min_det'], "picks for model", params['model_file'])
    print("Parameters - eval_in_file: {}\tmax_pick_dist: {}\tmin_sep: {}\tmax_picks: {}\tmin_nucl: {}\tmin_merge: {}\tt_repick: {}\tt_win: {}".format(params['eval_in_file'], params['max_pick_dist'], params['min_sep'], params['max_picks'], params['min_nucl'], params['min_merge'], params['t_repick'], params['t_win']))
    print("Totals/Average IoU Actual-Picked:", overall1[0], '/', overall1[1], '=', (overall1[0]/overall1[1]))
    precision, recall, f1 = prEvaluate(eventMatch1, params['eval_window'])
    if biou:
        print("Totals/Average IoU Picked-Actual:", overall2[0], '/', overall2[1], '=', (overall2[0]/overall2[1]))
        precision, recall = prEvaluate(eventMatch2, params['eval_window'])
    print("Precision:", str(precision))
    print("Recall:", str(recall))
    print("F1:", str(f1))
    print("Number of fake picks:", fakePicksMade, '/', numFakePicks)
    print("Picked / Actual Events:", detections, '/', actualEventCount)
    print("Arbitrary Health Measure:", (overall1[0]/overall1[1])*(overall2[0]/overall2[1])*pickedRatio*fakePickRatio*f1*100)

def prEvaluate(eventMatches, window = 10):
    events = pd.DataFrame(columns = ['EVID', 'OEVID', 'ARID', 'TIME', 'PHASE', 'FAKE', 'RULING'])
    for eid, oeid in eventMatches.items():
        inPicks = inputPicks[inputPicks.EVID == eid]
        outPicks = outputPicks[outputPicks.EVID == oeid]
        start = inPicks.TIME.min()
        start = start - window if start <= outPicks.TIME.min() else outPicks.TIME.min() - window
        end = inPicks.TIME.max()
        end = end + window if end >= outPicks.TIME.max() else outPicks.TIME.max() + window
        windowPicks = inputPicks[(inputPicks.TIME >= start) & (inputPicks.TIME <= end)]
        inIds = inPicks.ARID.values
        outIds = outPicks.ARID.values
        for i, pick in windowPicks.iterrows():
            if pick.ARID in inIds:
                ruling = 'TP' if pick.ARID in outIds else 'FN'
            else:
                ruling = 'FP' if pick.ARID in outIds else 'TN'
            events = events.append({'EVID': eid, 'OEVID': oeid, 'ARID': pick.ARID, 'TIME': pick.TIME, 'PHASE': pick.IPHASE, 'FAKE': pick.ARID < 0, 'RULING': ruling}, ignore_index = True)
    events.sort_values(by=['EVID', 'TIME']).to_pickle(params['pr_eval_out_file'])
    values = events.RULING.value_counts()
    precision = values['TP'] / (values['TP'] + values['FP'])
    recall = values['TP'] / (values['TP'] + values['FN'])
    f1 = 2*precision*recall/(precision+recall)
    return precision, recall, f1

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("PhaseLinkEvaluator config_json")
        sys.exit()

    with open(sys.argv[1], "r") as f:
        params = json.load(f)

    device = torch.device("cuda")
    model = StackedGRU().cuda(device)
    checkpoint = torch.load(params['model_file'], map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    inputPicks = pd.read_pickle(params['eval_in_file'])
    # limit = 5000
    # inputPicks = inputPicks[:limit]
    inputPicks = inputPicks[100000:105000]
    X, labels = processInput(params)
    detections = detect_events(X, labels, model, params)
    evaluate(detections, verbose=params['verbose'], biou=False)