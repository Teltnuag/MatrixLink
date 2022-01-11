import json
import numpy as np
import random

samples = 1000000
synth_events_X = "./Training/Train X.npy"
synth_events_Y = "./Training/Train Y.npy"

def synthesizeEvents(params):
    #Setup - load dictonary of events to their picks
    phases = params['phases']
    events = np.load(params['training_events_file'], allow_pickle=True)['events'].flatten()[0]
    eventList = list(events.keys())
    print("Events loaded.")

    X = []
    Y = []
    duration = params['t_win']
    max_picks = params['max_picks']
    minTimeShift = params['time_shifts']['min']
    maxTimeShift = params['time_shifts']['max']
    maxEventsToAdd = params['events_per_example']['max']
    dropFactor = params['drop_factor']
    trainingSample = 1
    while trainingSample <= samples:
        print('\rGenerating sample %d / %d, %.2f%%' % (trainingSample, samples, trainingSample/samples*100), end='')
        
        #Setup - choose a random event as the primary
        tempEventList = eventList.copy()
        primary = random.choice(tempEventList)
        tempEventList.remove(primary)
        sequence = events[primary]
        
        #Randomly drop some picks from the primary event
        drops = sequence[:,6]
        drops = drops + dropFactor*(1-drops) if dropFactor > 0 else drops*(1+dropFactor)
        drops = np.random.binomial(1,drops)
        idx = np.where(drops==1)[0]
        sequence = sequence[idx,:]
        primaryLength = len(sequence)
        
        #Choose a random number of events to add in
        numAddedEvents = random.randrange(0, maxEventsToAdd)
        timeShifts = np.random.uniform(minTimeShift, maxTimeShift, size=numAddedEvents)
        for i in range(0, numAddedEvents):
            currentLength = len(sequence)
            secondary = random.choice(tempEventList)
            tempEventList.remove(secondary)
            secondary = events[secondary]
            
            #Randomly drop some picks from the new event
            drops = secondary[:,6]
            drops = drops + dropFactor*(1-drops) if dropFactor > 0 else drops*(1+dropFactor)
            drops = np.random.binomial(1,drops)
            idx = np.where(drops==1)[0]
            secondary = secondary[idx,:]
            
            #Add the picks from the chosen event
            sequence = np.append(sequence, secondary, axis=0)
            
            #Shift the starting time of the chosen event
            sequence[currentLength:currentLength+len(secondary),2] += timeShifts[i]

        #Add random arrival time errors, except for the first pick of the primary event
        timeShifts = np.random.uniform(-sequence[1:,5]/duration, sequence[1:,5]/duration)
        sequence[1:,2] += timeShifts

        #Drop the unneeded paramaters
        sequence = sequence[:,0:5]

        #Make label array, set the primary event to 1s
        labels = np.zeros([len(sequence)])
        labels[0:primaryLength] = 1

        #Sort by arrival time, drop picks with negative arrival times
        idx = np.argsort(sequence[:,2])
        remove = len(np.where((sequence[:,2] < 0))[0])
        idx = idx[remove:]
        sequence = sequence[idx,:]
        labels = labels[idx]
        
        #Reset primary event times
        ones = np.where(labels==1)
        if len(ones[0]) == 0:
            #We lost all the valid picks, so scrap this training sample
            continue
        sequence[ones,2] -= sequence[ones,2][0][0]
        idx = np.argsort(sequence[:,2])
        
        #Truncate picks over maximum allowed
        idx = idx[:max_picks]
        sequence = sequence[idx,:]
        labels = labels[idx]
        
        #Pad the end if not enough picks were selected
        padding = max_picks - len(sequence)
        if padding > 0:
            labels = np.pad(labels, (0,max_picks-len(labels)))
            sequence_ = np.zeros((max_picks, 5))
            sequence_[sequence.shape[0]:,2] = 0.0
            sequence_[:sequence.shape[0], :] = sequence
            sequence = sequence_

        #Add this training sample to the list
        X.append(sequence)
        Y.append(labels.astype(np.int32))
        trainingSample += 1

    ones = np.sum(Y)
    zeros = np.size(Y) - ones
    total = ones + zeros
    X = np.array(X)
    nonPadding = X[np.where(X[:,:,4] == 1)]
    print("\nOnes:", ones, 100*ones/total)
    print("Zeros:", zeros, 100*zeros/total)
    print("Padding:", 100-100*len(nonPadding)/total)
    print('Recommended weight:', (zeros - np.where(X[:,:,4] == 0)[0].size) / ones)
    print('P/Pg/PcP/Pb', np.sum(nonPadding[:,3] == phases['P']) / len(nonPadding))
    print('Pn', np.sum(nonPadding[:,3] == phases['Pn']) / len(nonPadding))
    print('S/Sg/ScP/Lg/Sb', np.sum(nonPadding[:,3] == phases['S']) / len(nonPadding))
    print('Sn', np.sum(nonPadding[:,3] == phases['Sn']) / len(nonPadding))
    
    #Save the final training set
    np.save(synth_events_X, X)
    np.save(synth_events_Y, Y)

if __name__ == "__main__":
    with open("Parameters.json", "r") as f:
        params = json.load(f)
    
    synthesizeEvents(params)