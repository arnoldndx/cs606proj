# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 00:56:39 2022

@author: stell
"""

class Harmony(State):
    
    def __init__(self, name, depot, customers, CSs, vehicle):
        '''Initialize the EVRP state
        Args:
            name::str
                name of the instance
            depot::Depot
                depot of the instance
            customers::[Customer]
                customers of the instance
            CSs::[ChargingStation]
                charging stations of the instance
            vehicle::Vehicle
                vehicle of the instance
        '''
        self.name = name
        self.depot = depot
        self.customers = customers
        self.CSs = CSs
        self.vehicle = vehicle
        # record the vehicle used
        self.vehicles = []
        # total travel time of the all the vehicle used
        self.travel_time = 0
        # total charge time of the all the vehicle used
        self.charging_time = 0
        # record the all the customers who have been visited by all the vehicles, eg. [Customer1, Customer2, ..., Customer7, Customer8]
        self.customer_visited = []
        # record the unvisited customers, eg. [Customer9, Customer10]
        self.customer_unvisited = []
        # the route visited by each vehicle, eg. [vehicle1.node_visited, vehicle2.node_visited, ..., vehicleN.node_visited]
        self.route = []
                    
    def random_initialize(self, seed=None):
        ''' Randomly initialize the state with split_route() (your construction heuristic)
        Args:
            seed::int
                random seed
        Returns:
            objective::float
                objective value of the state
        '''
        if seed is not None:
            random.seed(606)
        random_tour = copy.deepcopy(self.customers)
        random.shuffle(random_tour)
        self.split_route(random_tour)
        return self.objective()
    
    def copy(self):
        return copy.deepcopy(self)
    
    def split_route(self, tour):
        '''Generate the route given a tour visiting all the customers
        Args:
            tour::[Customer]
                a tour visiting all the customers
        
        # You should update the following variables for the EVRP
        EVRP.vehicles
        EVRP.travel_time
        EVRP.charging_time
        EVRP.customer_visited
        EVRP.customer_unvisited
        EVRP.route
        
        # You should update the following variables for each vehicle used
        Vehicle.travel_time
        Vehicle.charging_time
        Vehicle.battery_consumption
        Vehicle.node_visited
        Vehicle.battery_charged
        Vehicle.battery_charged_to
        '''
        # You should implement your own method to construct the route of EVRP from any tour visiting all the customers
        ...
        
       
    
    def objective(self):
        ''' Calculate the objective value of the state
        Return the total travel time and charging time of all vehicles used
        '''
        # or return sum([v.travel_time for v in self.vehicles]) + sum([v.charging_time for v in self.vehicles])
        return self.travel_time +  self.charging_time