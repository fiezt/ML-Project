import numpy as np
import pandas as pd
import itertools
import warnings
warnings.filterwarnings('ignore')


class TaxiMDP(object):

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
        :param change_pairs: Mapping of aggregation of nodes.
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

        self.states = []
        self.final = []
        self.state2num = {}
        self.num2state = {}
        self.X = []

        self.actions = []
        self.U = []
        self.m = None
        
        self.reward_intervals = []

        self.create_action_space()

        self.create_state_space(aggregate)
        
        self.calculate_params()
        
        self.P = np.zeros((self.n, self.m, self.n))
        self.R = np.zeros((self.n, self.m, self.n))
        
        self.get_trans_and_rewards()
                        
        self.check_probabilities()

        del self.data


    def create_action_space(self):
        """Creating the complete action space for the MDP. 
        
        The action space contains transitions from a location i to location j.
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
        """
        
        self.data['pickup_datetime'] = pd.to_datetime(self.data['pickup_datetime'])
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
        """Creating rewards intervals until the reference point."""

        reward = self.interval_size

        if reward is not None:
            while reward < self.ref:
                self.reward_intervals.append((reward - self.interval_size, reward))
                reward += self.interval_size

            self.reward_intervals.append((reward - self.interval_size, self.ref))
            
            # This reward is the terminal reward state.
            self.reward_intervals.append((self.ref, float('inf')))
        else:
            self.reward_intervals.append((0, self.ref))
            
            # This reward is the terminal reward state.
            self.reward_intervals.append((self.ref, float('inf')))

        
    def calculate_params(self):
        """Getting parameters for the MDP."""

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
        """Finding the transition probabilities and rewards for the MDP."""
        
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

                    self.P[state_num1, action, state_num2] = 1
                else:
                    self.P[state_num1, action, state_num2] = 0
                
                # The reward is always 0 in this case.
                self.R[state_num1, action, state_num2] = 0

            elif state1[1] == 'e' and state2[1] == 'f':

                # This is case of driver picking someone up after a trip.
                if state1[2] == state2[2] and float('inf') not in state1[2] and action == state2[0]:
                    self.P[state_num1, action, state_num2] = 1
                    self.R[state_num1, action, state_num2] = self.empty_reward_avg[state1[0], state2[0]]

                else:
                    self.P[state_num1, action, state_num2] = 0
                    self.R[state_num1, action, state_num2] = 0

            elif state1[1] == 'f' and state2[1] == 'f':
                
                # Never transition from full to full.
                self.P[state_num1, action, state_num2] = 0
                
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

                    self.P[state_num1, action, state_num2] = p_dest * p_reward
                    
                    if float('inf') in state2[2]:
                        self.R[state_num1, action, state_num2] = 1000
                    else:
                        self.R[state_num1, action, state_num2] = self.full_reward_avg[state1[0], state2[0]]

                else:
                    self.P[state_num1, action, state_num2] = 0
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