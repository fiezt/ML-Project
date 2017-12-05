import pandas as pd
import numpy as np
import os
import json
import matplotlib.path as mplPath
import pickle
from collections import defaultdict 
import warnings
warnings.filterwarnings('ignore')


def load_data(curr_dir, driver_ids=[2010001271, 2010002704, 2010007579,
                                    2010007519, 2010007770, 2010003240,
                                    2010002920],
              aggregate=False, num_samples=None, num_areas=10):
    """Load the NY taxi data, clean it, and add in useful statistics.

    :param curr_dir: directory to be used to find the data.
    :param driver_ids: list of NY taxi drivers hack licenses to load data for.
    :param aggregate: Boolean indicating whether to sample and aggregate a set 
    of drivers.
    :param num_samples: The number of drivers to sample.

    :return taxi_data: Dataframe containing the cleaned data with useful stats.
    :return driver_areas: Dictionary of dictionaries where the outer dictionary
    has keys which are the driver ids and values of a dictionary, where the
    inner dictionary contains keys associated with a neighborhood and the
    corresponding values are the proportion of all rides starting in that
    neighborhood.
    :return neighborhoods: list of numpy arrays of the polygon of coordinates
    that define a neighborhood. Each array is a certain dimension N x 2
    where N varies. The first column contains the latitude points and the
    second column contains the longitude points.
    """

    # Loading the data of the neighborhood boundaries in new york.
    with open(os.path.join(curr_dir + '/data/taxi', 'community.json')) as f:
        bounds = json.load(f)

    # To hold the coordinates of each neighborhood for plotting purposes.
    neighborhoods = {}

    # To hold the path polygons of neighborhoods to determine which trips are contained by it.
    polygons = {}           

    i = 0
    for feature in bounds['features']:
        try:
            polygon = np.vstack((feature['geometry']['coordinates']))
            neighborhoods[i] = np.hstack((polygon[:, 1, None], polygon[:, 0, None]))
            polygons[i] = mplPath.Path(neighborhoods[i])
            i += 1
        except:
            polygon = feature['geometry']['coordinates']
            for poly in polygon:
                poly = np.vstack((poly))
                neighborhoods[i] = np.hstack((poly[:, 1, None], poly[:, 0, None]))
                polygons[i] = mplPath.Path(neighborhoods[i])
                i += 1

    trips = []
    fares = []

    cols = [' hack_license', ' pickup_datetime', ' dropoff_datetime', ' pickup_longitude',
            ' pickup_latitude', ' dropoff_longitude', ' dropoff_latitude']

    # Loading trip and fare data.
    for month in ['4', '5', '6']:
        trip_data = pd.read_csv(os.path.join(curr_dir + '/data/taxi', 'trip_data_' + month + '.csv'))
        trip_data = trip_data[cols]
        trip_data.columns = ['hack_license', 'pickup_datetime', 'dropoff_datetime', 
                             'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 
                             'dropoff_latitude']
        fare_data = pd.read_csv(os.path.join(curr_dir + '/data/taxi', 'trip_fare_' + month + '.csv'))

        # Keeping only the data that is from the set of selected drivers.
        if driver_ids is None and num_samples is None:
            pass
        elif driver_ids is not None and num_samples is None:
            trip_data = trip_data.loc[trip_data['hack_license'].isin(driver_ids)]
            fare_data = fare_data.loc[fare_data[' hack_license'].isin(driver_ids)]

        trips.append(trip_data)
        fares.append(fare_data)

    # Merging trip and fare data into one DataFrame.
    trips = pd.concat(trips)
    fares = pd.concat(fares)

    fares = fares[[' tolls_amount', ' total_amount']]
    fares.columns = ['tolls_amount', 'total_amount']

    taxi_data = pd.concat([trips, fares], axis=1)

    # Selecting a random subset of drivers to load data for.
    if driver_ids is None and num_samples is not None:
        driver_ids = np.random.choice(taxi_data['hack_license'].unique(), 
                                      size=num_samples, replace=False).tolist()
        taxi_data = taxi_data.loc[taxi_data['hack_license'].isin(driver_ids)]

    taxi_data['pickup_datetime'] = pd.to_datetime(taxi_data['pickup_datetime'])
    taxi_data['dropoff_datetime'] = pd.to_datetime(taxi_data['dropoff_datetime'])

    taxi_data['date'] = taxi_data['pickup_datetime'].dt.date
    taxi_data['hour'] = taxi_data['pickup_datetime'].dt.hour

    # Shifting the transaction peak to 12pm.
    for license in taxi_data['hack_license'].unique():
        max_hour = taxi_data.loc[taxi_data['hack_license'] == license].groupby('hour')['hour'].sum().argmax()
        if max_hour >= 12:
            shift_hour = max_hour - 12
            taxi_data.loc[taxi_data['hack_license'] == license, 'pickup_datetime'] -= pd.to_timedelta(shift_hour, unit='h')
            taxi_data.loc[taxi_data['hack_license'] == license, 'dropoff_datetime'] -= pd.to_timedelta(shift_hour, unit='h')
        else:
            shift_hour = 12 - max_hour
            taxi_data.loc[taxi_data['hack_license'] == license, 'pickup_datetime'] += pd.to_timedelta(shift_hour, unit='h')
            taxi_data.loc[taxi_data['hack_license'] == license, 'dropoff_datetime'] += pd.to_timedelta(shift_hour, unit='h')  

    # Resetting the date and time to the adjusted transaction times.
    taxi_data['date'] = taxi_data['pickup_datetime'].dt.date
    taxi_data['hour'] = taxi_data['pickup_datetime'].dt.hour        

    # Recalculating price with toll subtracted so it represents earnings.
    taxi_data['profit'] = taxi_data['total_amount'] - taxi_data['tolls_amount']

    # Calculating trip time based off start and end time to avoid data errors.
    taxi_data['trip_time'] = taxi_data['dropoff_datetime'] - taxi_data['pickup_datetime']
    taxi_data['trip_time'] = taxi_data['trip_time'].apply(lambda x: (x/np.timedelta64(1, 's'))/60.0)

    # Getting rid of bad location examples.
    taxi_data = taxi_data.loc[(taxi_data['pickup_latitude'] != 0) & (taxi_data['dropoff_latitude'] != 0)
                         & (taxi_data['pickup_longitude'] != 0) & (taxi_data['dropoff_longitude'] != 0)]

    # Columns to hold the start and end trip areas corresponding to the neighborhoods.
    taxi_data.reset_index(inplace=True, drop=True)
    taxi_data['start_trip_area'] = pd.Series([None for row in xrange(len(taxi_data))], index=taxi_data.index)
    taxi_data['end_trip_area'] = pd.Series([None for row in xrange(len(taxi_data))], index=taxi_data.index)

    # Creating arrays containing GPS coords for pickup and drop-off locations. 
    pickup_lats = taxi_data['pickup_latitude'].values.reshape((len(taxi_data), 1))
    pickup_longs = taxi_data['pickup_longitude'].values.reshape((len(taxi_data), 1))
    dropoff_lats = taxi_data['dropoff_latitude'].values.reshape((len(taxi_data), 1))
    dropoff_longs = taxi_data['dropoff_longitude'].values.reshape((len(taxi_data), 1))

    pickup_coords = np.hstack((pickup_lats, pickup_longs))
    dropoff_coords = np.hstack((dropoff_lats, dropoff_longs))

    # Finding which grid each trip is contained in for start and end location.
    for path in polygons:
        pickup_contained = polygons[path].contains_points(pickup_coords)
        contained_index = np.where(pickup_contained == True)[0].tolist()
        taxi_data.loc[contained_index, 'start_trip_area'] = path

        dropoff_contained = polygons[path].contains_points(dropoff_coords)
        contained_index = np.where(dropoff_contained == True)[0].tolist()
        taxi_data.loc[contained_index, 'end_trip_area'] = path
        
    # Dropping bad data where the coordinates were not in the boundaries.
    taxi_data.dropna(subset=['start_trip_area', 'end_trip_area'], inplace=True)

    # Dropping more bad trips.
    taxi_data = taxi_data.loc[taxi_data['trip_time'] != 0]

    # Calculating the earning rate in terms of dollars per minute.
    taxi_data['earn_rate'] = taxi_data.apply(lambda row: row['profit']/float(row['trip_time']), axis=1)

    # Getting rid of trips with incorrect earn rates.
    taxi_data = taxi_data.loc[(taxi_data['earn_rate'] >= 0.35) & (taxi_data['earn_rate'] <= 5)]

    # Getting the inverse of the earn rate for reward calculations.
    taxi_data['inverse_earn_rate'] = taxi_data['earn_rate']**(-1)

    # Adding the previous trip areas to each row for a new trip.
    taxi_data['prev_trip_area'] = taxi_data.groupby('hack_license')['end_trip_area'].shift(1)
    taxi_data['prev_trip_area'] = taxi_data.apply(lambda x: x['start_trip_area'] 
                                                  if pd.isnull(x['prev_trip_area']) 
                                                  else x['prev_trip_area'], axis=1)

    # Getting time between rides.
    taxi_data['seek_time'] = taxi_data.groupby('hack_license')['dropoff_datetime'].shift(1)
    taxi_data['seek_time'] = taxi_data['pickup_datetime'] - taxi_data['seek_time']
    taxi_data['seek_time'] = taxi_data['seek_time'].apply(lambda x: (x/np.timedelta64(1, 's'))/60.0)
    taxi_data['seek_time'] = taxi_data.apply(lambda x: 0.0 if pd.isnull(x['seek_time']) 
                                                           else x['seek_time'], axis=1)

    taxi_data['day_start'] = pd.Series([False for row in xrange(len(taxi_data))], index=taxi_data.index)

    # Setting the seek time to 0 for the first trip of the day.
    # and setting areas for the mdp problem.
    for license in taxi_data['hack_license'].unique():
        data = taxi_data.loc[taxi_data['hack_license'] == license]

        # Getting the first transactions in each day.
        day_starts = data.groupby('date')['pickup_datetime'].min().values.tolist()

        # Setting the first transaction of a days seek time to 0.
        taxi_data.loc[(taxi_data['hack_license'] == license) & 
                      (taxi_data['pickup_datetime'].isin(day_starts)), 'seek_time'] = 0.0

        # Setting flag for the first transaction of the day.
        taxi_data.loc[(taxi_data['hack_license'] == license) &
                      (taxi_data['pickup_datetime'].isin(day_starts)), 'day_start'] = True

    if not aggregate:
        change_pairs = defaultdict(list)

        for license in taxi_data['hack_license'].unique():
            data = taxi_data.loc[taxi_data['hack_license'] == license]

            start_area_index = data.groupby('start_trip_area')['start_trip_area'].sum().index.tolist()
            end_area_index = data.groupby('end_trip_area')['end_trip_area'].sum().index.tolist()

            start_area_values = data.groupby('start_trip_area')['start_trip_area'].sum().values.tolist()

            pairs = [(start_area_index[i], start_area_values[i]) for i in range(len(start_area_index))]
            sorted_pairs = sorted(pairs, key=lambda x: x[1], reverse=True)

            main_list = sorted_pairs[:num_areas]
            main_index = [pair[0] for pair in main_list]

            index_to_change = list(set(start_area_index).union(set(end_area_index)) - set(main_index))

            # Getting the centroid of all areas in which some trips start.
            centroids = {}
            for i in main_index:
                latitude = neighborhoods[i][:, 0]
                longitude = neighborhoods[i][:, 1]

                centroid = (sum(latitude)/len(neighborhoods[i]),
                            sum(longitude)/len(neighborhoods[i]))
                centroids[i] = np.array(centroid)

            # List of tuples where the first value is the 
            # previous area and the second the new value.
            area_pairs = []

            # Replacing ending areas that are not in starting locations with the
            # the key of the nearest starting area.
            for area in index_to_change:
                latitude = neighborhoods[area][:, 0]
                longitude = neighborhoods[area][:, 1]

                centroid = (sum(latitude)/len(neighborhoods[area]),
                            sum(longitude)/len(neighborhoods[area]))
                centroid = np.array(centroid)

                # Euclidean distance from the centroid to the start areas.
                dists = [(key, np.linalg.norm(centroids[key] - centroid)) for
                         key in centroids]

                # Choosing the area that minimizes the euclidean distance.
                new_area = sorted(dists, key=lambda x: x[1])[0][0]

                area_pairs.append((area, new_area))

                taxi_data.loc[(taxi_data['start_trip_area'] == area) &
                              (taxi_data['hack_license'] == license), 'start_trip_area'] = new_area

                taxi_data.loc[(taxi_data['end_trip_area'] == area) &
                              (taxi_data['hack_license'] == license), 'end_trip_area'] = new_area

                taxi_data.loc[(taxi_data['prev_trip_area'] == area) &
                              (taxi_data['hack_license'] == license), 'prev_trip_area'] = new_area

            change_pairs[license] = area_pairs
    else:
        change_pairs = defaultdict(list)

        start_area_index = taxi_data.groupby('start_trip_area')['start_trip_area'].sum().index.tolist()
        end_area_index = taxi_data.groupby('end_trip_area')['end_trip_area'].sum().index.tolist()

        start_area_values = taxi_data.groupby('start_trip_area')['start_trip_area'].sum().values.tolist()

        pairs = [(start_area_index[i], start_area_values[i]) for i in range(len(start_area_index))]
        sorted_pairs = sorted(pairs, key=lambda x: x[1], reverse=True)

        main_list = sorted_pairs[:num_areas]
        main_index = [pair[0] for pair in main_list]

        index_to_change = list(set(start_area_index).union(set(end_area_index)) - set(main_index))

        # Getting the centroid of all areas in which some trips start.
        centroids = {}
        for i in main_index:
            latitude = neighborhoods[i][:, 0]
            longitude = neighborhoods[i][:, 1]

            centroid = (sum(latitude)/len(neighborhoods[i]),
                        sum(longitude)/len(neighborhoods[i]))
            centroids[i] = np.array(centroid)

        # List of tuples where the first value is the 
        # previous area and the second the new value.
        area_pairs = []

        # Replacing ending areas that are not in starting locations with the
        # the key of the nearest starting area.
        for area in index_to_change:
            latitude = neighborhoods[area][:, 0]
            longitude = neighborhoods[area][:, 1]

            centroid = (sum(latitude)/len(neighborhoods[area]),
                        sum(longitude)/len(neighborhoods[area]))
            centroid = np.array(centroid)

            # Euclidean distance from the centroid to the start areas.
            dists = [(key, np.linalg.norm(centroids[key] - centroid)) for
                     key in centroids]

            # Choosing the area that minimizes the euclidean distance.
            new_area = sorted(dists, key=lambda x: x[1])[0][0]

            area_pairs.append((area, new_area))
            
            taxi_data.loc[(taxi_data['start_trip_area'] == area), 'start_trip_area'] = new_area
            taxi_data.loc[(taxi_data['end_trip_area'] == area), 'end_trip_area'] = new_area
            taxi_data.loc[(taxi_data['prev_trip_area'] == area), 'prev_trip_area'] = new_area

    if aggregate:
        driver_areas = {}
        ride_count = {}

        for path in polygons:
            ride_count[path] = len(taxi_data.loc[taxi_data['start_trip_area'] == path])
        
        total = float(len(taxi_data))
        ride_count = {k: (v/total)*100 for k, v in ride_count.items()}
        ride_count = sorted(ride_count.items(), key=lambda item: item[1], reverse=True)

        ride_count = {item[0]: item[1] for item in ride_count if item[1] > 0}
        driver_areas[tuple(driver_ids)] = ride_count
    else:
        driver_areas = {}

        for driver_id in driver_ids:
            driver = taxi_data.loc[taxi_data['hack_license'] == driver_id]
            ride_count = {}

            for path in polygons:
                ride_count[path] = len(driver.loc[driver['start_trip_area'] == path])

            total = float(len(driver))
            ride_count = {k: (v/total)*100 for k, v in ride_count.items()}
            ride_count = sorted(ride_count.items(), key=lambda item: item[1], reverse=True)

            ride_count = {item[0]: item[1] for item in ride_count if item[1] > 0}
            driver_areas[driver_id] = ride_count

    return taxi_data, driver_areas, neighborhoods, change_pairs


def main():
    dropbox_dir = '/Users/tfiez/Dropbox/riskTaxi'
    data_dir = os.path.join(os.getcwd(), '..', 'Data')
    taxi_data, driver_areas, neighborhoods, change_pairs = load_data(dropbox_dir)

    pickle.dump(neighborhoods, open(os.path.join(data_dir, "neighorhoods.p"), "wb"))
    pickle.dump(driver_areas, open(os.path.join(data_dir, "driver_areas.p"), "wb"))
    pickle.dump(change_pairs, open(os.path.join(data_dir, "change_pairs.p"), "wb"))
    taxi_data.to_csv(os.path.join(data_dir, 'taxi_data.csv'), sep=',')


if __name__ == "__main__":
    main()