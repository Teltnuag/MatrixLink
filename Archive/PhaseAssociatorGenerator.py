import numpy as np
import pandas as pd
from obspy import UTCDateTime
import random
import rstt
from copy import deepcopy
from collections import deque
from itertools import chain, repeat
import math
modelPath = "./Training/RSTT Model/pdu202009Du.geotess"
phases = ['Pg','Pn','Lg','Sn']

def generateEventFile(params, trainingSet = True):
    if trainingSet:
        eventsFile = params['trainingEventsFile']
        generatorFile = params['trainingGeneratorSourceFile']
    else:
        eventsFile = params['validationEventsFile']
        generatorFile = params['validationGeneratorSourceFile']
    try:
        events = np.load(eventsFile, allow_pickle=True)['events'].flatten()[0]
        print("Training events loaded.") if trainingSet else print("Validation events loaded.")
    except:
        print("Events not loaded. Building from scratch.")
        extents = np.array(list(params['extents'][params['location']].values()))
        latRange = abs(extents[1] - extents[0])
        lonRange = abs(extents[3] - extents[2])
        phases = params['phases']
        events = {}
        inputArrivals = pd.read_pickle(generatorFile)
        groupedEvents = (inputArrivals.groupby('EVID').filter(lambda x: len(x) >= params['minArrivals'])).groupby('EVID')
        count = 0
        for eid, arrivals in groupedEvents:
            count += 1
            print("\rBuilding event list: " + str(count) + ' / ' + str(len(groupedEvents)), end='')
            eventArrivals = []
            first = UTCDateTime(arrivals.ARRTT.min())
            for i, arrival in arrivals.iterrows():
                lat = abs((arrival.STALAT - extents[0]) / latRange)
                lon = abs((arrival.STALON - extents[2]) / lonRange)
                thisArrival = [lat, lon, ((UTCDateTime(arrival.ARRTT) - first)/params['timeNormalize']), phases[arrival.PHASE], 1, arrival.ARRUNCERT, arrival.DROPOUT, arrival.EV_LAT, arrival.EV_LON, arrival.EV_DEPTH, arrival.EV_TIME]
                eventArrivals.append(thisArrival)
            events[eid] = np.array(eventArrivals)
        np.savez_compressed(eventsFile, events=events)
        print()
    eventList = list(events.keys())
    return events, eventList

def synthesizeLocatorEventsFromEventFile(params, events, eventList, trainingSet = True):
    duration = params['timeNormalize']
    maxArrivals = params['maxArrivals']
    dropFactor = params['dropFactor']
    batchSize = params['batchSize']

    while True:
        X = np.zeros((batchSize, maxArrivals, 11))
        #Setup - choose random events
        chosenEvents = random.sample(eventList, batchSize)
        for i in range(batchSize):
            thisEvent = events[chosenEvents[i]]
            #Randomly drop some arrivals from the event
            if trainingSet:
                drops = thisEvent[:,6]
                drops = drops + dropFactor*(1-drops) if dropFactor > 0 else drops*(1+dropFactor)
                drops = np.random.binomial(1,drops)
                idx = np.where(drops==1)[0]
                thisEvent = thisEvent[idx,:]
                #Add random arrival time errors, except for the first arrival of the primary event
                thisEvent[1:,2] += np.random.uniform(-thisEvent[1:,5]/duration, thisEvent[1:,5]/duration)
    
            #Sort by arrival time, drop arrivals with negative arrival times
            idx = np.argsort(thisEvent[:,2])
            remove = len(np.where((thisEvent[:,2] < 0))[0])
            idx = idx[remove:]
            thisEvent = thisEvent[idx]
            
            if len(thisEvent) == 0:
                #We lost all the valid arrivals, so scrap this training sample
                continue
            
            #Reset arrival times to start at 0
            thisEvent[:,2] -= thisEvent[0,2]
            
            #Truncate arrival over maximum allowed
            thisEvent = thisEvent[:maxArrivals]
            
            X[i,:len(thisEvent)] = thisEvent

        #Yield these training examples
        Y = X[:,0,7:]
        X = {"phase": X[:,:,3], "numerical_features": X[:,:,[0,1,2,4]]}
        yield X, Y

def synthesizeEventsFromEventFile(params, events, eventList, trainingSet = True):
    duration = params['timeNormalize']
    maxArrivals = params['maxArrivals']
    minTimeShift = params['timeShifts']['min']
    maxTimeShift = params['timeShifts']['max']
    minEvents = params['eventsPerExample']['min']
    maxEvents = params['eventsPerExample']['max']+1 # because using in np.random.randint
    dropFactor = params['dropFactor']

    while True:
        X = []
        Y = []
        for example in range(params['batchSize']):
            #Setup - choose random events, with the first being the primary event
            numEvents = np.random.randint(minEvents, maxEvents)
            chosenEvents = random.sample(eventList, numEvents)
            timeShifts = np.random.uniform(minTimeShift, maxTimeShift, size=numEvents-1)
            for i in range(0, len(chosenEvents)):
                thisEvent = events[chosenEvents[i]]
                #Randomly drop some arrivals from the event
                if trainingSet:
                    drops = thisEvent[:,6]
                    drops = drops + dropFactor*(1-drops) if dropFactor > 0 else drops*(1+dropFactor)
                    drops = np.random.binomial(1,drops)
                    idx = np.where(drops==1)[0]
                    thisEvent = thisEvent[idx,:]
                if i > 0:
                    #Add the arrivals from this event
                    sequence = np.append(sequence, thisEvent, axis=0)
                    #Shift the starting time of this event
                    currentLength = len(sequence)
                    sequence[currentLength:currentLength+len(thisEvent),2] += timeShifts[i-1]
                else:
                    sequence = thisEvent
                    primaryLength = len(sequence)
            
            #Add random arrival time errors, except for the first arrival of the primary event
            if trainingSet:
                timeShifts = np.random.uniform(-sequence[1:,5]/duration, sequence[1:,5]/duration)
                sequence[1:,2] += timeShifts
            
            #Drop the unneeded paramaters
            sequence = sequence[:,0:5]
    
            #Make label array, set the primary event to 1s
            labels = np.zeros([len(sequence)])
            labels[0:primaryLength] = 1
    
            #Sort by arrival time, drop arrivals with negative arrival times
            idx = np.argsort(sequence[:,2])
            remove = len(np.where((sequence[:,2] < 0))[0])
            idx = idx[remove:]
            sequence = sequence[idx,:]
            labels = labels[idx]
            
            #Reset primary event times
            ones = np.where(labels==1)
            if len(ones[0]) == 0:
                #We lost all the valid arrivals, so scrap this training sample
                continue
            sequence[ones,2] -= sequence[ones,2][0][0]
            idx = np.argsort(sequence[:,2])
            
            #Truncate arrivals over maximum allowed
            idx = idx[:maxArrivals]
            sequence = sequence[idx,:]
            labels = labels[idx]
            
            #Pad the end if not enough arrivals were selected
            padding = maxArrivals - len(sequence)
            if padding > 0:
                labels = np.pad(labels, (0,maxArrivals-len(labels)))
                sequence_ = np.zeros((maxArrivals, 5))
                sequence_[sequence.shape[0]:,2] = 0.0
                sequence_[:sequence.shape[0], :] = sequence
                sequence = sequence_
            
            X.append(sequence)
            Y.append(labels.astype(np.int32))

        #Yield these training examples
        X = np.array(X)
        Y = np.array(Y)
        X = {"phase": X[:,:,3], "numerical_features": X[:,:,[0,1,2,4]]}
        yield X, Y

def synthesizeEvents(params, locator=False):
    def get_TT(srcLatDeg, srcLonDeg, srcDepKm, rcvLatDeg, rcvLonDeg, rcvDepKm, phase, slbm):
        phase = phases[phase]
        # create a great circle from source to the receiver
        slbm.createGreatCircle(phase,
            rstt.deg2rad(srcLatDeg),
            rstt.deg2rad(srcLonDeg),
            srcDepKm,
            rstt.deg2rad(rcvLatDeg),
            rstt.deg2rad(rcvLonDeg),
            rcvDepKm)
    
        # get the distance and travel time from source --> receiver
        travelTimeSec = slbm.getTravelTime()   # compute travel time (sec)
        # get the travel time uncertainty
        travelTimeUncertSec   = slbm.getTravelTimeUncertainty()
    
        return travelTimeSec, travelTimeUncertSec

    def extentsCheck(extents):
        latRange = abs(extents[1] - extents[0])
        lonRange = abs(extents[3] - extents[2])
        a = (math.sin(math.radians(latRange)/2)**2) + (math.cos(math.radians(extents[0])) * math.cos(math.radians(extents[1])) * (math.sin(math.radians(lonRange/2))**2))
        maxSep = math.degrees(2*math.asin(min(1,math.sqrt(a))))
        print('Max event to receiver separation in degrees:', maxSep)
        
    def haversine(lat1, lon1, lat2, lon2):
        """
        Calculate the great circle distance between every combination of two points
        on the earth (specified in decimal degrees)
        """
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        lat1 = lat1[:, np.newaxis]
        lon1 = lon1[:, np.newaxis]
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
    
        a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    
        c = 2 * np.arcsin(np.sqrt(a))
        km = 6368.1 * c
        bucket = (km/100).astype(int)
        bucket = np.clip(bucket, 0, 11)
        return bucket
    
    # instantiate an RSTT object
    slbm = rstt.SlbmInterface()
    # load the velocity model
    slbm.loadVelocityModel(modelPath)
    # load the arrival probabilities
    arrivalProbs = np.load(params['arrivalProbsFile']) # Phase (Pg, Pn, Lg, Sn), Distance (0-11), Magnitude (0-5)
    
    # read in all the parameters first to avoid redundant lookups. Really just more readable
    timeWindow = params['timeNormalize']
    batchSize = params['batchSize']
    maxArrivals = params['maxArrivals']
    minTimeShift = params['timeShifts']['min']
    maxTimeShift = params['timeShifts']['max']
    minEvents = params['eventsPerExample']['min']
    maxEvents = params['eventsPerExample']['max']+1 # because using in np.random.randint
    minStations = params['stationsPerBatch']['min']
    maxStations = params['stationsPerBatch']['max']+1 # because using in np.random.randint
    extents = np.array(list(params['extents'][params['location']].values())+[params['maxDepth'],params['maxStationElevation']])
    latRange = abs(extents[1] - extents[0])
    lonRange = abs(extents[3] - extents[2])
    arrivalProbs = np.load(params['arrivalProbsFile'])/100 # Phase (Pg, Pn, Lg, Sn), Distance (0-11), Magnitude (0-5)
    arrivalProbs[0] = np.clip(arrivalProbs[0]*params['arrivalProbMods']['Pg'],0,1)
    arrivalProbs[1] = np.clip(arrivalProbs[1]*params['arrivalProbMods']['Pn'],0,1)
    arrivalProbs[2] = np.clip(arrivalProbs[2]*params['arrivalProbMods']['Sg'],0,1)
    arrivalProbs[3] = np.clip(arrivalProbs[3]*params['arrivalProbMods']['Sn'],0,1)
    cycleGrid = (params['cycleGrid'] == 'True')
    if cycleGrid:
        latMod = 0.
        lonMod = 0.
        gridDiv = 0.5
        gridStep = 0.01
        partial = True

    def processEventsLocator(events):
        eventsCycle = chain.from_iterable(repeat(events))
        examples = [deepcopy(next(eventsCycle)) for _ in range(batchSize)]

        X = np.zeros((batchSize,maxArrivals,9))
        for i in range(batchSize):
            examples[i][1:,2] += examples[i][1:,4]*np.random.uniform(-1,1,len(examples[i][1:]))
            examples[i][:,4] = 1.
            idx = np.argsort(examples[i][:,2])
            start = np.argmax(idx == 0)
            idx = idx[start:start+maxArrivals]
            examples[i] = examples[i][idx]
            X[i][:len(examples[i])] = examples[i]
        X = np.array(X)
        Y = X[:,0,5:]
        X = {"phase": X[:,:,3], "numerical_features": X[:,:,[0,1,2,4]]}
        return X, Y

    def processEventsAssociator(events):
        examplesList = [random.sample(range(len(events)), numEvents) for _ in range(batchSize)]
        examples = [[deepcopy(events[e]) for e in examplesList[ex]] for ex in range(len(examplesList))]

        # generate all the amounts to shift events and phases around
        timeMods = np.random.uniform(minTimeShift,maxTimeShift,batchSize*(numEvents-1))
        timeModsUsed = 0

        Y = np.zeros((batchSize, maxArrivals))
        X = np.zeros((batchSize, maxArrivals, 5))
        for i in range(batchSize):
            example = examples[i]
            for event in example[1:]:
                event[:,2] += timeMods[timeModsUsed]
                timeModsUsed += 1
            examples[i] = np.concatenate(example)
            examples[i][1:,2] += examples[i][1:,4]*np.random.uniform(-1,1,len(examples[i][1:]))
            examples[i][:,4] = 1.
            idx = np.argsort(examples[i][:,2])
            start = np.argmax(idx == 0)
            idx = idx[start:start+maxArrivals]
            Y[i,np.where(idx < len(example[0]))] = 1
            examples[i] = examples[i][idx][:,:5]
            X[i][:len(examples[i])] = examples[i]

        X = np.array(X)
        Y = np.array(Y)
        X = {"phase": X[:,:,3], "numerical_features": X[:,:,[0,1,2,4]]}
        return X, Y

    def processEventsAssociator2(events):
        initial = [deque(random.sample(range(len(events)), numEvents)) for e in range(int(batchSize/numEvents))]
        examplesList = []
        for seq in range(numEvents-1):
            rotatedExamples = deepcopy(initial)
            for eventList in rotatedExamples:
                eventList.rotate(seq+1)
            examplesList += rotatedExamples
        examplesList = initial + examplesList
        examples = [[deepcopy(events[e]) for e in examplesList[i]] for i in range(len(examplesList))]

        # generate all the amounts to shift events and phases around
        timeMods = np.random.uniform(minTimeShift,maxTimeShift,batchSize*(numEvents-1))
        timeModsUsed = 0

        Y = np.zeros((len(examples),maxArrivals))
        X = np.zeros((len(examples),maxArrivals,5))
        for i in range(len(examples)):
            example = examples[i]
            for event in example[1:]:
                event[:,2] += timeMods[timeModsUsed]
                timeModsUsed += 1
            examples[i] = np.concatenate(example)
            examples[i][1:,2] += examples[i][1:,4]*np.random.uniform(-1,1,len(examples[i][1:]))
            examples[i][:,4] = 1.
            idx = np.argsort(examples[i][:,2])
            start = np.argmax(idx == 0)
            idx = idx[start:start+maxArrivals]
            Y[i,np.where(idx < len(example[0]))] = 1
            examples[i] = examples[i][idx][:,:5]
            X[i][:len(examples[i])] = examples[i]

        X = np.array(X)
        Y = np.array(Y)
        X = {"phase": X[:,:,3], "numerical_features": X[:,:,[0,1,2,4]]}
        return X, Y

    while(True):
        # Create a random numbers of events per training example...
        # ...and a random numbers of receiving stations to be used for each training example
        numEvents = np.random.randint(minEvents, maxEvents)
        numStations = np.random.randint(minStations, maxStations)
        totalEvents = int(batchSize/numEvents)
        genEvents = np.zeros((totalEvents, 6))
        stations = np.zeros((numStations, 6))
        if cycleGrid and partial:
            genEvents[:,0] = np.random.uniform(0+latMod, gridDiv+latMod, totalEvents) # evLat
            genEvents[:,1] = np.random.uniform(0+lonMod, gridDiv+lonMod, totalEvents) # evLon
            genEvents[:,2] = np.random.uniform(0, 1, totalEvents) # evDepth
            stations[:,0] = np.random.uniform(0+latMod, gridDiv+latMod, numStations) # stLat
            stations[:,1] = np.random.uniform(0+lonMod, gridDiv+lonMod, numStations) # stLon
            stations[:,2] = np.random.uniform(0, 1, numStations) # stDepth
            
            # Update the sliding box in which to generate the next set of events
            latMod = (latMod+gridStep)%(1-gridDiv)
            if latMod == 0:
                lonMod = (lonMod+gridStep)%(1-gridDiv)
        else:
            genEvents[:,0:3] = np.random.rand(totalEvents, 3) # [evLat, evLon, evDepth]
            stations[:,0:3] = np.random.rand(numStations, 3) # [stLat, stLon, stDepth]
            # partial = not partial
            

        # Calculate the denormalized event and station latitudes
        genEvents[:,3] = genEvents[:,0]*latRange + extents[0]
        stations[:,3] = stations[:,0]*latRange + extents[0]
        # Calculate the denormalized event and station longitudes
        genEvents[:,4] = genEvents[:,1]*lonRange + extents[2]
        stations[:,4] = stations[:,1]*lonRange + extents[2]
        # Calculate the denormalized event and stations depths
        genEvents[:,5] = genEvents[:,2]*extents[4]
        stations[:,5] = -stations[:,2]*extents[5] # because it's elevation, I think

        # Generate random magnitudes (as buckets for looking up in the arrival probability table)
        # Calculate the distances between stations and events and get the buckets for looking up in the probability table
        mags = np.random.randint(0, 6, totalEvents)
        dists = haversine(genEvents[:,3],genEvents[:,4],stations[:,3],stations[:,4]) # distance buckets as dists[event, station]

        # Lookup retention values for each training example, phase, event, station
        arrivals = arrivalProbs[:,dists,mags[:,np.newaxis]]
        arrivals = np.random.binomial(1,arrivals) # arrivals[phase][event][station]
        arrivals = np.argwhere(arrivals==1)

        # Build the events list
        events = np.full((len(arrivals),10),np.nan) # normal station lat, normal station lon, travel time, phase, travel time uncertainty, event lat, event lon, event depth, event time, event id
        currentPhase = 0
        for arrival in range(len(arrivals)):
            ph, ev, st = arrivals[arrival]
            try:
                tt, uncert = get_TT(genEvents[ev,3],genEvents[ev,4],genEvents[ev,5],stations[st,3],stations[st,4],stations[st,5],ph,slbm)
                events[currentPhase] = [stations[st,0], stations[st,1], tt, ph, uncert, genEvents[ev,3], genEvents[ev,4], genEvents[ev,5], 0, ev]
                currentPhase += 1
            except Exception as e:
        #                 print(e)
                pass
        events = events[~np.any(np.isnan(events), axis=1)] # Remove NaNs (from failed TT calculations)
        events = events[np.lexsort((events[:,2], events[:,-1]))] # Sort by event id, then by travel time
        evids, counts = np.unique(events[:,-1], return_counts=True) # Ensure there are at least a few arrivals for each event
        evids = evids[np.where(counts >= params['minArrivals'])]
        events = events[np.isin(events[:,-1],evids)]
        events[:,[2,4]] /= timeWindow # Normalize time values

        events = np.split(events[:,:-1], np.unique(events[:,-1], return_index=True)[1][1:]) # arrivals as an array of events
        for event in events:
            event[:,[2,8]] -= event[0,2] # convert travel time to arrival time, retain event time TODO: can this be done vectorized?
            event[:,8] *= timeWindow
        numEvents = len(events) if len(events) < numEvents else numEvents
        yield processEventsLocator(events) if locator else processEventsAssociator(events)