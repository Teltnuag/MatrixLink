import numpy as np
import sys
import json
import pickle
from obspy.geodetics.base import gps2dist_azimuth

def get_network_centroid(params):
    # stlo = []
    # stla = []
    # with open(params['station_file'], 'r') as f:
    #     for line in f:
    #         try:
    #             net, sta, lat, lon, elev = line.split()
    #         except:
    #             net, sta, lat, lon = line.split()
    #         stlo.append(float(lon))
    #         stla.append(float(lat))

    # lat0 = (np.max(stla) + np.min(stla))*0.5
    # lon0 = (np.max(stlo) + np.min(stlo))*0.5
    # return lat0, lon0
    return 30.973, 47.4961755

def build_station_map(params):
    stations = {}
    sncl_map = {}
    count = 0
    with open(params['station_file'], 'r') as f:
        for line in f:
            try:
                net, sta, lat, lon, elev = line.split()
            except:
                net, sta, lat, lon = line.split()
            stla = float(lat)
            stlo = float(lon)
            X0 = gps2dist_azimuth(lat0, stlo, lat0, lon0)[0]/1000.
            if stlo < lon0:
                X0 *= -1
            Y0 = gps2dist_azimuth(stla, lon0, lat0, lon0)[0]/1000.
            if stla < lat0:
                Y0 *= -1
            # X0 = stlo / 100
            # Y0 = stla / 100
            if (net, sta) not in sncl_map:
                sncl_map[(net, sta)] = count
                stations[(net, sta)] = (X0, Y0)
                count += 1
    stlo = np.array([stations[x][0] for x in sncl_map])
    stla = np.array([stations[x][1] for x in sncl_map])
    sncl_idx = np.array([sncl_map[x] for x in sncl_map])

    return stlo, stla, sncl_idx, stations, sncl_map


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("PhaseLinkStationMapper config_json")
        sys.exit()

    with open(sys.argv[1], "r") as f:
        params = json.load(f)

    lat0, lon0 = get_network_centroid(params)
    stlo, stla, sncl_idx, stations, sncl_map = build_station_map(params)

    # x_min = np.min(stlo)
    # x_max = np.max(stlo)
    # y_min = np.min(stla)
    # y_max = np.max(stla)
    
    x_min = -1420.0
    x_max = 1420.0
    y_min = -1050.0
    y_max = 1050.0
    
    print(x_min, x_max, y_min, y_max)

    for key in sncl_map:
        X0, Y0 = stations[key]
        X0 = (X0 - x_min) / (x_max - x_min)
        Y0 = (Y0 - y_min) / (y_max - y_min)
        stations[key] = (X0, Y0)

    # Save station maps for detect mode
    pickle.dump(stations, open(params['station_map_file'], 'wb'))