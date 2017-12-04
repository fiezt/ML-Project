import pandas as pd
import numpy as np
import os
import json
import matplotlib.path as mplPath
import pickle
import sys
import itertools
import gmplot
import matplotlib.colors as colors
import matplotlib.cm as cmx
import matplotlib.pyplot as plt
from collections import defaultdict 
from geojson import MultiPolygon
from shapely.geometry import Polygon
from plotly.graph_objs import Scattermapbox, Marker, Layout, Data

# import warnings
# warnings.filterwarnings('ignore')


class MDP(object):

    def __init__(self, driver_id, taxi_data, ride_areas, neighborhoods, 
                 interval_size, change_pairs, aggregate=False):
        """Initializing the MDP to have needed statistics.
        
        :param driver_id: Hack license from the data for the specified driver 
        or list of hack licenses for a group of drivers.
        :param ride_areas: list of integer ride keys that correspond to a 
        neighborhood in New York.
        :param neighborhoods: list of numpy arrays of the polygon of coordinates
        that define a neighborhood. Each array is a certain dimension N x 2
        where N varies. The first column contains the latitude points and the
        second column contains the longitude points.
        :param interval_size: Size of the reward intervals in dollars.
        :param aggregate: Boolean variable to indicate whether to aggregate a 
        group of drivers if this field is true.
        """

        self.driver_id = driver_id
        self.neighborhoods = neighborhoods
        self.interval_size = interval_size
        self.change_pairs = change_pairs
        self.ride_areas = ride_areas

        # Dictionary to hold the mapping from area key to matrix index.
        self.mapping = {self.ride_areas[v]: v for v in range(len(self.ride_areas))}
        
        # Dictionary to hold the mapping from matrix index to area key.
        self.inverse_mapping = {v:k for k,v in self.mapping.items()}
                
        self.nodes = self.inverse_mapping.keys()
        
        if aggregate:
            self.data = taxi_data.loc[taxi_data['hack_license'].isin(list(driver_id))]
        else:
            self.data = taxi_data.loc[taxi_data['hack_license'] == driver_id]
            
      #  self.data_ = self.data.copy()
            
        # Changing the areas to correspond to self.nodes.
        self.data['start_trip_area'] = self.data['start_trip_area'].apply(lambda x: self.mapping[x])
        self.data['end_trip_area'] = self.data['end_trip_area'].apply(lambda x: self.mapping[x])
        self.data['prev_trip_area'] = self.data['prev_trip_area'].apply(lambda x: self.mapping[x])
        
        # Matrix to hold the probability of starting a trip in each area.
        self.demand = np.zeros((len(self.nodes), 1))

        self.trans_prob = np.zeros((len(self.nodes), len(self.nodes)))   

        self.search_time = np.zeros((len(self.nodes), 1))
        self.node_earn_rate = np.zeros((len(self.nodes), 1))
        self.search_rewards = np.zeros((len(self.nodes), 1))

        self.drive_time_avg = np.zeros((len(self.nodes), len(self.nodes)))
        self.drive_time_std = np.zeros((len(self.nodes), len(self.nodes)))

        self.fare_avg = np.zeros((len(self.nodes), len(self.nodes)))
        self.fare_std = np.zeros((len(self.nodes), len(self.nodes)))

        self.earn_rate_avg = np.zeros((len(self.nodes), len(self.nodes)))
        self.earn_rate_std = np.zeros((len(self.nodes), len(self.nodes)))

        self.full_reward_avg = np.zeros((len(self.nodes), len(self.nodes)))
        self.full_reward_std = np.zeros((len(self.nodes), len(self.nodes)))

        self.empty_drive_reward = np.zeros((len(self.nodes), len(self.nodes)))
        self.empty_reward_avg = np.zeros((len(self.nodes), len(self.nodes)))

        self.empty_policy = []

        # Median daily earnings value for driver or set of drivers.
        self.ref = None

        self.traj=None
        self.T=30
        self.state=None
        self.N=None

        self.states = []
        self.final = []
        self.state2num = {}
        self.num2state = {}
        self.X = []

        self.actions = []
        self.U = []
        self.m = None
        
        self.reward_intervals = []

        self.transitions = {}

        self.rewards = {}

        self.create_action_space()

        self.create_state_space(aggregate)
        
        self.calculate_params()

        # The initial state will always be the node with highest demand, meaning
        # the most rides beginning in it, the taxi being empty, and the reward in interval [0,20).
        self.initial = (max(enumerate(self.demand.tolist()), key=lambda x: x[1])[0], 
                              'e', self.reward_intervals[0])
        
        self.P = np.zeros((self.n, self.m, self.n))
        self.R = np.zeros((self.n, self.m, self.n))
        
        self.get_trans_and_rewards()
                        
        self.check_probabilities()

        del self.data
        

    def step(self,a):

        self.traj.append([self.state,a])
        self.N[self.state,a]+=1
        self.state=int(np.random.choice(self.X,1,False,list(self.P[self.state,a,:])))

        return int(self.state)


    def initialize(self,state=0):

        self.state=0
        self.traj=[]
        self.N=np.zeros((self.n,self.m))


    def create_action_space(self):
        """Creating the complete action space for the MDP. 
        
        The action space contains transitions from a location i to location j.
        
        :return: Nothing is returned but the class attribute self.actions is updated.
        """
        
        self.actions = [i for i in self.nodes]
        
        self.U = self.actions
        
        self.m = len(self.U)
        
        
    def create_state_space(self, aggregate):
        """Creating the full state space for the MDP.
        
        The complete state space is X = {N x S x R}\X_na where N is the index
        set of the zones or nodes in the city with N nodes, S = {e, f} are the
        states indicating if the taxi is empty or full, and R is the discretized
        cumulative fare value space. The states that are not allowed are 
        X_na = {(i, f, r)|r in R_terminal, i in N}.
        
        :param aggregate: Boolean variable for whether the MDP is over a set of 
        drivers or to aggregate for.

        :return: Nothing is returned but the class attributes self.ref, 
        self.reward_intervals, and self.states are updated.
        """
        
        self.get_ref(aggregate)
        self.create_reward_intervals()
        
        i = 0
        for state in itertools.product(self.nodes, ['e', 'f'], self.reward_intervals):
            
            # States that are not allowed.
            if state[1] == 'f' and float('inf') in state[2]:
                pass
            else:
                self.states.append(state)
    
                self.num2state[i] = state
                self.state2num[state] = i
                self.X.append(i)

                if state[1] == 'e' and float('inf') in state[2]:
                    self.final.append(i)

                i += 1

        self.n = len(self.X)

        
    def get_ref(self, aggregate):  
        """Calculating the reference point by finding the median daily earnings.

        :param aggregate: Boolean variable for whether the MDP is over a set of 
        drivers or to aggregate for.

        :return: Nothing is returned but the class attribute is updated.
        """

        self.data['pickup_date'] = self.data['pickup_datetime'].apply(lambda x: x.date())
        
        dates = sorted(self.data['pickup_date'].unique())
        
        daily_earnings = []

        if aggregate:
            for date in dates:
                driver_day = self.data.loc[self.data['pickup_date'] == date]
                active_drivers = driver_day['hack_license'].unique()
                daily_earnings.append(sum(driver_day['profit'])/float((len(active_drivers))))
        else:
            for date in dates:
                driver_day = self.data.loc[self.data['pickup_date'] == date]
                daily_earnings.append(sum(driver_day['profit']))

        self.ref = np.median(np.array(daily_earnings)) 
        
        
    def create_reward_intervals(self):
        """Creating rewards intervals until the reference point.
        
        :return: Nothing is returned but self.reward_intervals is set.
        """
        
        reward = self.interval_size

        while reward < self.ref:
            self.reward_intervals.append((reward - self.interval_size, reward))
            reward += self.interval_size

        self.reward_intervals.append((reward - self.interval_size, self.ref))
        
        # This reward is the terminal reward state.
        self.reward_intervals.append((self.ref, float('inf')))

        
    def calculate_params(self):
        """Getting parameters for the MDP.

        :return: Nothing is returned but self.search_time.avg,
        self.trans_prob, self.drive_time_avg,
        self.drive_time.std, self.fare_avg, self.fare_std, self.earn_rate_avg,
        self.earn_rate_std, self.full_reward_avg, self.full_reward_std, 
        and self.empty_reward_avg are all set.
        """


        for start in sorted(self.nodes):
            area_start = self.data.loc[self.data['start_trip_area'] == start]

            self.demand[start] = len(area_start)/float(len(self.data))

            t_search_trips = self.data.loc[(self.data['prev_trip_area'] == start)
                                           & (self.data['start_trip_area'] == start)]
            t_search_trips = t_search_trips.loc[t_search_trips['day_start'] == False]
            t_search_trips = t_search_trips.loc[t_search_trips['seek_time'] <= 20]
            
            for end in sorted(self.nodes):
                area_end = area_start.loc[area_start['end_trip_area'] == end]

                if len(area_end) == 0:
                    trans_prob = 0.0
                    drive_avg = None
                    drive_std = None
                    fare_avg = 0.0
                    fare_std = 0.0
                    earn_rate_avg = None
                    earn_rate_std = None
                else:
                    trans_prob = len(area_end)/float(len(area_start))
                    drive_avg = area_end['trip_time'].mean()
                    drive_std = area_end['trip_time'].std()
                    fare_avg = area_end['profit'].mean()
                    fare_std = area_end['profit'].std()
                    earn_rate_avg = area_end['earn_rate'].mean()
                    earn_rate_std = area_end['earn_rate'].std()

                self.trans_prob[start, end] = trans_prob
                
                self.drive_time_avg[start, end] = drive_avg
                self.drive_time_std[start, end] = drive_std
                
                self.fare_avg[start, end] = fare_avg
                self.fare_std[start, end] = fare_std
                
                self.earn_rate_avg[start, end] = earn_rate_avg
                self.earn_rate_std[start, end] = earn_rate_std
                
                self.full_reward_avg[start, end] = fare_avg
                self.full_reward_std[start, end] = fare_std

                if earn_rate_avg is not None:
                    self.empty_drive_reward[start, end] = -drive_avg/float(earn_rate_avg**-1)
                else:
                    self.empty_drive_reward[start, end] = 0.0

            if len(t_search_trips) == 0 or np.isnan(self.earn_rate_avg[start, start]) \
                or self.earn_rate_avg[start,start] == 0:

                self.search_rewards[start] = 0
            else:
                self.search_rewards[start] = -t_search_trips['seek_time'].mean()/float(self.earn_rate_avg[start, start]**-1)


        self.empty_search = np.where(self.search_rewards == 0)[0].tolist()
        self.search_rewards[self.search_rewards == 0] = np.min(self.search_rewards)
        
        empty_drive = np.where(self.empty_drive_reward == 0.0)
        empty_row_index = empty_drive[0]
        empty_col_index = empty_drive[1]

        self.empty_drive = [(empty_row_index[i], empty_col_index[i]) for i in range(len(empty_row_index))]
        self.empty_drive_reward[self.empty_drive_reward == 0] = np.min(self.empty_drive_reward)

        self.empty_reward_avg = self.empty_drive_reward + self.search_rewards.T

        for node in sorted(self.nodes):
            # In the case of the i to i transition, it is only the search time.
            # This line is removing the empty_drive time reward.
            self.empty_reward_avg[node, node] -= self.empty_drive_reward[node, node]
    
        
    def get_trans_and_rewards(self):
        """Finding the transition probabilities and rewards for the MDP.

        :return: Nothing is returned but self.transitions, self.P, self.rewards, 
        and self.R are updated.
        """
        
        for transition in itertools.product(self.states, self.actions, self.states):
            state1 = transition[0]
            action = transition[1]
            state2 = transition[2]
            
            state_num1 = self.state2num[state1]
            state_num2 = self.state2num[state2]

            if state1[1] == 'e' and state2[1] == 'e':

                # If in final state, you are guaranteed to stay there.
                if float('inf') in state1[2] and float('inf') in state2[2] \
                                             and state1[0] == state2[0]:

                    self.transitions[transition] = 1
                    self.P[state_num1, action, state_num2] = 1
                else:
                    self.transitions[transition] = 0
                    self.P[state_num1, action, state_num2] = 0
                
                # The reward is always 0 in this case.
                self.rewards[transition] = 0
                self.R[state_num1, action, state_num2] = 0

            elif state1[1] == 'e' and state2[1] == 'f':

                # This is case of driver picking someone up after a trip.
                if state1[2] == state2[2] and float('inf') not in state1[2] and action == state2[0]:

                    self.transitions[transition] = 1
                    self.P[state_num1, action, state_num2] = 1

                    self.rewards[transition] = self.empty_reward_avg[state1[0], state2[0]] 
                    self.R[state_num1, action, state_num2] = self.empty_reward_avg[state1[0], state2[0]]

                else:
                    self.transitions[transition] = 0
                    self.P[state_num1, action, state_num2] = 0
                    
                    self.rewards[transition] = 0
                    self.R[state_num1, action, state_num2] = 0

            elif state1[1] == 'f' and state2[1] == 'f':
                
                # Never transition from full to full.
                self.transitions[transition] = 0
                self.P[state_num1, action, state_num2] = 0
                
                self.rewards[transition] = 0
                self.R[state_num1, action, state_num2] = 0

            elif state1[1] == 'f' and state2[1] == 'e':

                # This is the case some fare is gained from the trip.
                if state2[2][0] >= state1[2][0]:

                    """
                    This piece of code is finding the probability: 
                    P((i,f,r), u, (j,f,r')) = P_dest(i,j)P(a_l - E[F(i,j)] <= r <= b_l - E[F(i,j)])
                    The following code uses that r is assumed to be uniformly 
                    distributed on the interval of the reward state.
                    """

                    trans_reward = state2[2]

                    # Lower bound on reward being transitioned to.
                    a_l = trans_reward[0]

                    # Upper bound on reward being transitioned to.
                    b_l = trans_reward[1]

                    curr_reward = state1[2]

                    # Lower bound on current reward.
                    a_i = curr_reward[0]

                    # Upper bound on current reward.
                    b_i = curr_reward[1]

                    start = state1[0] 
                    end = state2[0]

                    p_dest = self.trans_prob[start, end]
                    e_fare = self.fare_avg[start, end]

                    x_1 = b_l - e_fare
                    x_2 = a_l - e_fare

                    # CDF of the upper bound.
                    if x_1 < a_i:
                        F_1 = 0
                    elif x_1 < b_i:
                        F_1 = (x_1 - a_i)/float(b_i - a_i)
                    else:
                        F_1 = 1

                    # CDF of the lower bound.
                    if x_2 < a_i:
                        F_2 = 0
                    elif x_2 < b_i:
                        F_2 = (x_2 - a_i)/float(b_i - a_i)
                    else:
                        F_2 = 1

                    p_reward = F_1 - F_2

                    self.transitions[transition] = p_dest * p_reward
                    self.P[state_num1, action, state_num2] = p_dest * p_reward
                    
                    if float('inf') in state2[2]:
                        self.rewards[transition] = 1000
                        self.R[state_num1, action, state_num2] = 1000
                    else:
                        self.rewards[transition] = self.full_reward_avg[state1[0], state2[0]]
                        self.R[state_num1, action, state_num2] = self.full_reward_avg[state1[0], state2[0]]

                else:
                    self.transitions[transition] = 0   
                    self.P[state_num1, action, state_num2] = 0
                    
                    self.rewards[transition] = 0
                    self.R[state_num1, action, state_num2] = 0


    def check_probabilities(self):
        """This function is to ensure that all probability functions are valid.
        
        To ensure the MDP is correct this function contains checks that each of
        the probability functions are valid, for transitions and policies.
        
        :return: Nothing but assert error if probabilities are incorrect.
        """
        
        # Checking valid density function for the transition probability.
        for state in xrange(self.n):
            for action in xrange(self.m):
                assert abs(sum(self.P[state, action, :]) - 1) < 1e-3, 'Transitions do not sum to 1'

        # Checking that there is no None values of any of the MDP.
        assert True not in pd.isnull(self.P), 'None value in transitions'
        assert True not in pd.isnull(self.R), 'None value in rewards'
        assert True not in pd.isnull(self.U), 'None value in actions'


def load_data(curr_dir, driver_ids=[2010001271, 2010002704, 2010007579,
                                    2010007519, 2010007770, 2010003240,
                                    2010002920],
              aggregate=False, num_samples=None):
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
        taxi_data.ix[contained_index, 'start_trip_area'] = path

        dropoff_contained = polygons[path].contains_points(dropoff_coords)
        contained_index = np.where(dropoff_contained == True)[0].tolist()
        taxi_data.ix[contained_index, 'end_trip_area'] = path
        
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

            main_list = sorted_pairs[:25]
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

            change_pairs[driver_id] = area_pairs
    else:
        change_pairs = defaultdict(list)

        start_area_index = taxi_data.groupby('start_trip_area')['start_trip_area'].sum().index.tolist()
        end_area_index = taxi_data.groupby('end_trip_area')['end_trip_area'].sum().index.tolist()

        start_area_values = taxi_data.groupby('start_trip_area')['start_trip_area'].sum().values.tolist()

        pairs = [(start_area_index[i], start_area_values[i]) for i in range(len(start_area_index))]
        sorted_pairs = sorted(pairs, key=lambda x: x[1], reverse=True)

        main_list = sorted_pairs[:25]
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


def node_visual(mdp, filename='policy.html'):
    """Plotting the states bounds in unique colors.

    :param mdp: mdp object containing the needed variables.
    :param filename: Path to save the file to.

    :return color_map: Dictionary from color to node index.
    """

    gmap = gmplot.GoogleMapPlotter(40.723031, -73.931419, 12)

    colors = ['blue','green','red','cyan','magenta','yellow','black','white','brown']

    color_map = {}

    idx = 0

    for key in mdp.mapping:
        if key == -1:
            color_map[mdp.mapping[key]] = 'blank'
            continue
        
        color_map[mdp.mapping[key]] = colors[idx]

        node = mdp.mapping[key]
        
        label = [k for k in mdp.num2state if mdp.num2state[k][0] == node]
        label = 'Node is ' + str(node) + ', States are ' + str(label).strip('[]')
        
        points = mdp.neighborhoods[key]
        latitude = points[:,0]
        longitude = points[:,1]
        
        centroid = (sum(latitude) / len(points), sum(longitude) / len(points))
        
        gmap.polygon(latitude, longitude, face_color=colors[idx], face_alpha=0.99)
        
        idx += 1
        
        gmap.draw(filename)

    return color_map


def get_node_counts(mdp, driver, data):
    """Finding the policy for the agent for node movement.

    :param mdp: MDP object created for some set of taxi drivers.
    :param driver: Driver id.
    :param data: Data including the driver data.

    :return: N, the policy for the driver.
    """

    data = data.loc[data['hack_license'] == driver]

    data['pickup_datetime'] = pd.to_datetime(data['pickup_datetime'])
    data['dropoff_datetime'] = pd.to_datetime(data['dropoff_datetime'])

    data['date'] = data['pickup_datetime'].apply(lambda x: x.date())

    data['next_trip_area'] = data.groupby(['hack_license', 'date'])['start_trip_area'].shift(-1)

    data.dropna(inplace=True)

    # Starting and ending areas of policy decision following each transaction.
    data['start_choice'] = data['end_trip_area'].apply(lambda x: mdp.mapping[x])
    data['end_choice'] = data['next_trip_area'].apply(lambda x: mdp.mapping[x])

    N = np.zeros((mdp.m, mdp.m))

    counts = data.groupby(['start_choice', 'end_choice'])['hack_license'].count()
    counts = counts.reset_index(level=[0,1])

    for start_node in mdp.nodes:
        for end_node in mdp.nodes:
            value = counts.loc[(counts['start_choice'] == start_node) & (counts['end_choice'] == end_node)]['hack_license'].values
            if not value:
                N[start_node, end_node] = 0
            else:
                N[start_node, end_node] = value[0]

    return N


def get_policy(mdp, driver=None, data=None):
    """Finding the policy for the agent.

    :param mdp: MDP object created for some set of taxi drivers.
    :param driver: Driver id.
    :param data: Data including the driver data.

    :return: N, the policy for the driver.
    """

    # Case in which the mdp contains the data for all the driver ids.
    if data is None and driver is None:
        data = mdp.data_
        driver = mdp.driver_id

    # Case in which the mdp contains the data for the drivers being selected.
    elif data is None and driver is not None:
        data = mdp.data_

    # Case in which we want policy for drivers that were not used in making the mdp.
    elif data is not None and driver is not None:
        pass

    if isinstance(driver, tuple):
        data = data.loc[data['hack_license'].isin(driver)]
    else:
        data = data.loc[data['hack_license'] == driver]

    data['date'] = data['pickup_datetime'].apply(lambda x: x.date())

    data['cum_rewards'] = pd.Series([None for row in xrange(len(data))], index=data.index)

    # Tracking the daily cumulative rewards at each transaction.
    data['cum_rewards'] = data.groupby(['hack_license', 'date'])['profit'].cumsum()

    # Label indicating what reward interval earnings are at following a transaction.
    data['reward_interval'] = data['cum_rewards'].apply(lambda y: mdp.reward_intervals.index(filter(lambda x: x[0] <= y < x[1], 
                                                                                                     mdp.reward_intervals)[0]))

    data['next_trip_area'] = data.groupby(['hack_license', 'date'])['start_trip_area'].shift(-1)
    data.dropna(inplace=True)

    # Starting and ending areas of policy decision following each transaction.
    data['start_choice'] = data['end_trip_area'].apply(lambda x: mdp.mapping[x])
    data['end_choice'] = data['next_trip_area'].apply(lambda x: mdp.mapping[x])

    # Finding the policy for the data.
    N = np.zeros((mdp.n, mdp.m))

    for state in mdp.states:
        state_num = mdp.state2num[state]

        # Empty and not in final reward indicates a choice is being made.
        if state[1] == 'e' and state[2] != mdp.reward_intervals[-1]:

            state_num = mdp.state2num[state]

            start_choice = state[0]

            reward_interval = mdp.reward_intervals.index(state[2])

            final_reward = mdp.reward_intervals.index(mdp.reward_intervals[-1])

            for action in mdp.actions:
                N[state_num, action] = len(data.loc[(data['reward_interval'] == reward_interval) & 
                                                   (data['start_choice'] == start_choice) & 
                                                   (data['end_choice'] == action) & 
                                                   (data['reward_interval'] != final_reward)])
        else:
            N[state_num, :] = 1/float(len(mdp.actions))

    empty_rows = np.where(~N.any(axis=1))[0].tolist()

    if not empty_rows:
        pass
    else:
        for row in empty_rows:
            N[row] = 1

    return N


def get_color(row, scalarMap):
    """Get the color to use based off the proportion of rides starting there.

    :param row: pandas dataframe row.
    :param scalarMap: Matplotlib scalarMappable object.
    :return: Hex color to use for area.
    """

    intensity = row['ride_proportion']

    colorVal = scalarMap.to_rgba(intensity)
    colorVal = (int(colorVal[0] * 255), int(colorVal[1] * 255),
                int(colorVal[2] * 255))

    intensity = "#{:02X}".format(colorVal[0]) + "{:02X}".format(colorVal[1]) + \
                "{:02X}".format(colorVal[2])

    return intensity
       

def interactive_map(mdp):
    """Interactive MDP map with node number, earn rate, drive time.
    
    :param mdp: mdp object with the needed data.
    :return fig: plotly fig plot, to plot use plot(fig) from plotly.offline.
    """
    
    neighborhoods = mdp.neighborhoods

    # Getting the centroid of all areas that were kept as nodes.
    centroids = {}

    for i in mdp.mapping.keys():
        latitude = neighborhoods[i][:, 0]
        longitude = neighborhoods[i][:, 1]

        centroid = (sum(latitude)/len(neighborhoods[i]),
                    sum(longitude)/len(neighborhoods[i]))
        centroids[i] = np.array(centroid)


    # For areas not kept as nodes, finding the area that was kept that was closest.
    area_pairs = {}

    for j in range(len(neighborhoods)):
        if j in mdp.mapping.keys():
            continue

        ref_polygon = Polygon(neighborhoods[j])
        point = ref_polygon.representative_point().xy

        lat = point[0][0]
        longitude = point[1][0]
        centroid = np.array([lat, longitude])

        dists = [(key, np.linalg.norm(centroids[key] - centroid)) for
                 key in centroids]

        new_area = sorted(dists, key=lambda x: x[1])[0][0]

        area_pairs[j] = new_area

    area_counts = pd.DataFrame(data=range(len(neighborhoods)), columns=['start_trip_area'])

    # Node start location by neighborhood index.
    area_counts['start_trip_area'] = area_counts['start_trip_area'].apply(lambda x: area_pairs[x] if x in area_pairs else x)

    # Converting neighborhood index to mdp node area.
    area_counts['start_trip_area'] = area_counts['start_trip_area'].apply(lambda x: mdp.mapping[x])

    area_counts['ride_proportion'] = area_counts['start_trip_area'].apply(lambda x: mdp.demand[x,0])

    # Creating the polygon coordinates for each of the areas.
    polygons = []
    centroids = []

    for i in neighborhoods:

        ref_polygon = Polygon(neighborhoods[i])
        point = ref_polygon.representative_point().xy

        lat = point[0][0]
        longitude = point[1][0]

        centroid = [lat, longitude]
        centroids.append(centroid)

        multi_poly = MultiPolygon([[coord[1], coord[0]] for coord in neighborhoods[i].tolist()])
        multi_poly['coordinates'] = [[multi_poly['coordinates']]]

        full_geo = {}
        full_geo['geometry'] = multi_poly
        full_geo['properties'] = {}
        full_geo['type'] = 'Feature'

        polygons.append(full_geo)

    centroids = np.vstack((centroids))
    lats = centroids[:,0]
    longs = centroids[:,1]

    area_counts['lat_centroids'] = lats    
    area_counts['long_centroids'] = longs   
    area_counts['coordinates'] = polygons

    # Getting earn rates as strings for the plot.
    earn_rate_avg = mdp.earn_rate_avg.tolist()
    for i in range(len(earn_rate_avg)):
        earn_rate_str = [str(j) + ' = ' + str(round(earn_rate_avg[i][j], 1)) 
                         if j % 5 != 0 or j == 0 else '<br>' + str(j) + ' = ' 
                         + str(round(earn_rate_avg[i][j], 2)) 
                         for j in range(len(earn_rate_avg[i]))]

        earn_rate_avg[i] = ', '.join(earn_rate_str)


    # Getting drive time as strings for the plot.
    drive_time_avg = mdp.drive_time_avg.tolist()
    for i in range(len(drive_time_avg)):
        drive_time_str = [str(j) + ' = ' + str(round(drive_time_avg[i][j], 1)) 
                          if j % 5 != 0 or j == 0 else '<br>' + str(j) + ' = ' 
                          + str(round(drive_time_avg[i][j], 2)) 
                          for j in range(len(drive_time_avg[i]))]

        drive_time_avg[i] = ', '.join(drive_time_str)

    area_counts['earn_rate'] = area_counts['start_trip_area'].apply(lambda x: earn_rate_avg[x])
    area_counts['drive_time'] = area_counts['start_trip_area'].apply(lambda x: drive_time_avg[x])

    # Column to hold color to use for the area on the neighborhood map.
    area_counts['color'] = pd.Series([None for row in xrange(len(area_counts))], index=area_counts.index)

    # Create the color map for the plot using the max and min values of ride proportion.
    cm = plt.cm.hot_r
    cNorm = colors.Normalize(vmin=min(area_counts['ride_proportion'].values),
                             vmax=max(area_counts['ride_proportion'].values))
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
    scalarMap.get_clim()

    # Getting the color to use for area based off the ride proportion.
    area_counts['color'] = area_counts.apply(get_color, args=(scalarMap,), axis=1)  

    # Text to display on the plotly map.
    area_counts['text'] = '<br>' + 'Node: ' + area_counts['start_trip_area'].astype(str) + '<br><br>' \
                      + 'Earn Rates: ' +  area_counts['earn_rate'] + '<br><br>' \
                      + 'Drive Times: ' + area_counts['drive_time']

    mapbox_access_token = 'pk.eyJ1IjoiZmllenQiLCJhIjoiY2oxOGk5MXVqMDYyNjJ3b3ZiNHY1bTl0eSJ9.-6-DnmvlmyNtOQUsOhzdRg'

    # Creating layers of boundaries and colors for map layout.
    layers_ls = []
    for index, row in area_counts.iterrows():
        item_dict = dict(sourcetype = 'geojson', source = row['coordinates'],
                         type = 'fill', color = row['color'])
        layers_ls.append(item_dict)

    # Creating the full layout for the heatmap.
    layout = Layout(title = 'New York Taxi Data: Portion of Rides per Area',
                               height=1200, width=1200, autosize=False, hovermode='closest',
                               mapbox=dict(layers=layers_ls, accesstoken=mapbox_access_token,
                               bearing=0, center=dict(lat=40.704280,lon=-73.958805), pitch=0,
                               zoom=10.3,style='light'),)

    # Creating the color scale, list of lists containing ride proportion and color, low to high. 
    cscl_ = sorted(area_counts.color.unique(), reverse=True)
    cscl = []
    for col in cscl_:
        cscl.append([cNorm(min(area_counts.loc[area_counts['color'] == col]['ride_proportion'].unique())), col])


    # Adding the colorscale and all else for map.
    data = Data([Scattermapbox(lat = area_counts['lat_centroids'], 
                            lon = area_counts['long_centroids'], hoverinfo='lat+lon+text',
                            marker=Marker(cmax=max(area_counts['ride_proportion'].values),
                            cmin=min(area_counts['ride_proportion'].values), colorscale=cscl,
                            showscale = True, autocolorscale=False, size=0), text=area_counts['text'], 
                            mode = 'markers', textfont=dict(family='Courier New, monospace', size=18, color='#7f7f7f')),])

    fig = dict(data=data, layout=layout)
    
    return fig


def main():
    args = sys.argv
    argc = len(args)

    # Insert path to dropbox location for riskTaxi.
    tanner_dir = '/Users/tfiez/Dropbox/riskTaxi'

    if args[1].lower() == 'tanner':
        dropbox_dir = tanner_dir

    if args[2] == 'default':
        taxi_data, driver_areas, neighborhoods, change_pairs = load_data(dropbox_dir)

        for interval_size in [100]:
            for driver in driver_areas.keys():
                driver_mdp = MDP(driver, taxi_data, driver_areas[driver].keys(), 
                                 neighborhoods, change_pairs, interval_size)

                with open(os.path.join(dropbox_dir + '/data/MDP_SIM_NEW/', 'r' + str(interval_size) + 
                                       '_driver_' + str(driver) + '.pkl'), 'wb') as f:

                    pickle.dump(driver_mdp, f)


    elif args[2] == 'sample':

        interval_size = 100

        for samples in [100, 250, 500, 1000, 2500, 5000, 10000]:
            taxi_data, driver_areas, neighborhoods, change_pairs = load_data(dropbox_dir, driver_ids=None,
                                                                             aggregate=True, num_samples=samples)

            taxi_data.to_csv(os.path.join(dropbox_dir + '/data/TAXI_DATA/', str(samples) + '_samples_taxi_data' + '.csv'), index=False)

            driver = driver_areas.keys()[0]

            driver_mdp = MDP(driver, taxi_data, driver_areas[driver].keys(), neighborhoods, change_pairs,
                             interval_size, aggregate=True)

            N = get_policy(mdp=driver_mdp, driver=tuple(driver_mdp.driver_id), data=taxi_data.copy())

            np.savetxt(os.path.join(dropbox_dir + '/data/MDP_SIM_POLICY_V4/', 'r' + 
                       str(interval_size) + '_sample_' + str(samples) + '.csv'), N, delimiter=',')

            with open(os.path.join(dropbox_dir + '/data/MDP_SIM_NEW_V4/', 'r' + 
                      str(interval_size) + '_sample_' + str(samples) + '.pkl'), 'wb') as f:
                
                pickle.dump(driver_mdp, f)

    else:
        print('Bad Input')
        exit()


if __name__ == "__main__":
    main()
