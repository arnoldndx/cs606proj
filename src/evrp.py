'''
Description: EVRP State
Author: BO Jianyuan
Date: 2022-02-07 15:52:58
LastEditors: BO Jianyuan
LastEditTime: 2022-02-24 21:40:59
'''

import copy
import numpy as np
import xml.etree.ElementTree as ET
import random

from bisect import bisect_left
from itertools import accumulate
from scipy.spatial.distance import cdist, pdist
from pathlib import Path

import sys
sys.path.append('./ALNS')
from alns import ALNS, State

### Parser to parse instance xml file ###
# You should not change this class!
class Parser(object):
    
    def __init__(self, xml_file):
        '''initialize the parser
        Args:
            xml_file::str
                the path to the xml file
        '''
        self.xml_file = xml_file
        self.name = Path(xml_file).stem
        self.tree = ET.parse(self.xml_file)
        self.root = self.tree.getroot()
        self.depot = None
        self.customers = []
        self.CSs = []
        self.vehicle = None
        self.ini_nodes()
        
    def ini_nodes(self):
        for node in self.root.iter('node'):
            if int(node.attrib['type']) == 0:
                self.depot = Depot(int(node.attrib['id']), int(node.attrib['type']), 
                                   float(node.find('cx').text), float(node.find('cy').text))
            elif int(node.attrib['type']) == 1:
                self.customers.append(Customer(int(node.attrib['id']), int(node.attrib['type']), 
                                               float(node.find('cx').text), float(node.find('cy').text),
                                               None))
            else:
                self.CSs.append(ChargingStation(int(node.attrib['id']), int(node.attrib['type']), 
                                                float(node.find('cx').text), float(node.find('cy').text), 
                                                node.find('custom').find('cs_type').text,
                                                None))
        self.set_customers()
        self.set_CSs()
        self.set_vehicle()
    
    def set_customers(self):
        request = {}
        for r in self.root.iter('request'):
            request[int(r.attrib['id'])] = float(r.find('service_time').text)
        for c in self.customers:
            c.service_time = request[c.id]
    
    def set_CSs(self):
        function = {}
        for f in self.root.iter('function'):
            function[f.attrib['cs_type']] = [(float(b.find('battery_level').text), float(b.find('charging_time').text)) for b in f.findall('breakpoint')]
        for cs in self.CSs:
            cs.breakpoints = function[cs.charging_speed]
    
    def set_vehicle(self):
        for v in self.root.iter('vehicle_profile'):
            self.vehicle = Vehicle(0, self.depot, self.depot,
                                   float(v.find('max_travel_time').text), float(v.find('speed_factor').text), 
                                   float(v.find('custom').find('consumption_rate').text), float(v.find('custom').find('battery_capacity').text))

### Node class ###
# You should not change this class!
class Node(object):
    
    def __init__(self, id, type, x, y):
        '''Initialize a node
        Args:
            id::int
                id of the node
            type::int
                0 for depot, 1 for customer, 2 for charging station
            x::float
                x coordinate of the node
            y::float
                y coordinate of the node
        '''
        self.id = id
        self.type = type
        self.x = x
        self.y = y

    def get_nearest_node(self, nodes):
        '''Find the nearest node in the list of nodes
        Args:
            nodes::[Node]
                a list of nodes
        Returns:
            node::Node
                the nearest node found
        '''
        dis = [cdist([[self.x, self.y]], [[node.x, node.y]], 'euclidean') for node in nodes]
        idx = np.argmin(dis)
        return nodes[idx]
    
    def __eq__(self, other):
        if isinstance(other, Node):
            return self.id == other.id and self.type == other.type and self.x == other.x and self.y == other.y
        return False
        
    def __str__(self):
        return 'Node id: {}, type: {}, x: {}, y: {}'.format(self.id, self.type, self.x, self.y)

### Depot class ###
# You should not change this class!
class Depot(Node):
    
    def __init__(self, id, type, x, y):
        '''Initialize a depot
        Args:
            id::int
                id of the depot
            type::int
                0 for depot
            x::float
                x coordinate of the depot
            y::float
                y coordinate of the depot
        '''
        super(Depot, self).__init__(id, type, x, y)
        
### Customer class ###
# You should not change this class!
class Customer(Node):
    
    def __init__(self, id, type, x, y, service_time):
        '''Initialize a customer
        Args:
            id::int
                id of the customer
            type::int
                1 for customer
            x::float
                x coordinate of the customer
            y::float
                y coordinate of the customer
            service_time::float
                service time of the customer
        '''
        super(Customer, self).__init__(id, type, x, y)
        self.service_time = service_time
        
    def __str__(self):
        return 'Node id: {}, type: {}, x: {}, y: {}, service_time: {}'.format(self.id, self.type, self.x, self.y, self.service_time)
    
### ChargingStation class ###
# You can add your own helper functions to the class, and not required to use the functions defined
# But you should keep the rest untouched!
class ChargingStation(Node):
    
    def __init__(self, id, type, x, y, charging_speed, breakpoints):
        '''init the charging station
        Args:
            id::int
                id of the charging station
            type::int
                2 for charging station
            x::float
                x coordinate of the charging station
            y::float
                y coordinate of the charging station
            charging_speed::str
                'slow', 'normal', or 'fast'
            breakpoints::[(float, float)]
                a list of breakpoints for the charging function
        '''
        super(ChargingStation, self).__init__(id, type, x, y)
        self.charging_speed = charging_speed
        self.breakpoints = breakpoints
        
    def charge_to(self, vehicle, x):
        ''' Charge the vehicle to the target battery level and calculate the charging time required
        Args:
            vehicle::Vehicle
                vehicle to be charged
            x::float
                target battery level
        '''
        # ensure the target battery level is larger than the vehicle current battery level
        assert(x > vehicle.battery_level)
        
        # ensure the target battery level is not larger than the vehicle's battery capacity
        assert(x <= vehicle.battery_capacity)
        
        start_time = ChargingStation.reverse_charging_function(vehicle.battery_level, self.breakpoints)
        end_time = ChargingStation.reverse_charging_function(x, self.breakpoints)
        #print(start_time>=0, end_time<1)
        # the battery charged is recorded
        vehicle.battery_charged.append(x - vehicle.battery_level)
        # the battery level when leaving the CS is recorded
        vehicle.battery_charged_to.append(x)
        # the vehicle's battery level is updated after charging
        vehicle.battery_level = x
        # calculate the charging time required
        vehicle.charging_time += end_time - start_time
        vehicle.cs_visited.append(self)
        
    def charge_till(self, vehicle, t):
        ''' Charge the vehicle using time t and calculate the battery charged
        Args:
            vehicle::Vehicle
                vehicle to be charged
            t::float
                time to charge 
        '''
        start_time = ChargingStation.reverse_charging_function(vehicle.battery_level, self.breakpoints)
        end_time =  start_time + t
        # calculate the battery level of the vehicle after charging
        battery_level = charging_function(end_time, breakpoints)
        # the battery charged is recorded
        vehicle.battery_charged.append(battery_level - vehicle.battery_level)
        # the battery level when leaving the CS is recorded
        vehicle.battery_charged_to.append(battery_level)
        # the vehicle's battery level is updated after charging
        vehicle.battery_level = battery_level
        # update the charging time for the vehicle
        vehicle.charging_time += t
        vehicle.cs_visited.append(self)
        
    def __str__(self):
        return 'ChargingStation id: {}, type: {}, x: {}, y: {}, charging_speed: {}, breakpoints: {}'.format(self.id, self.type, self.x, self.y, self.charging_speed, self.breakpoints)
    
    @staticmethod
    def charging_function(t, breakpoints):
        ''' Charging function (x-axis is the time and y-axis is the battery level)
        Give the time t, find the corresponding battery level using the charging function
        Args:
            t::float
                time
            breakpoints::[(float, float)]
                a list of breakpoints for the charging function
        Returns:
            battery_level::float
                battery level corresponding to time t
        '''
        condlist = [(t > breakpoints[i][1]) & (t <= breakpoints[i + 1][1]) for i in range(len(breakpoints) - 1)]
        k = [(breakpoints[i + 1][0] - breakpoints[i][0])/(breakpoints[i + 1][1] - breakpoints[i][1]) for i in range(len(breakpoints) - 1)]
        funclist = [lambda t: k[0]*t + breakpoints[0][0], 
                    lambda t: breakpoints[1][0] + k[1]*(t - breakpoints[1][1]),
                    lambda t: breakpoints[2][0] + k[2]*(t - breakpoints[2][1])]
        return np.piecewise(float(t), condlist, funclist).reshape(1,)[0]
    
    @staticmethod
    def reverse_charging_function(x, breakpoints):
        ''' Reverse Charging function (x-axis is the battery level and y-axis is the time)
        Give the battery level x, find the corresponding time using the reverse charging function
        Args:
            x::float
                battery level
            breakpoints::[(float, float)]
                a list of breakpoints for the charging function
        Returns:
            time::float
                time corresponding to battery level x
        '''
        condlist = [(x > breakpoints[i][0]) & (x <= breakpoints[i + 1][0]) for i in range(len(breakpoints) - 1)]
        k = [(breakpoints[i + 1][1] - breakpoints[i][1])/(breakpoints[i + 1][0] - breakpoints[i][0]) for i in range(len(breakpoints) - 1)]
        funclist = [lambda x: k[0]*x + breakpoints[0][1], 
                    lambda x: breakpoints[1][1] + k[1]*(x-breakpoints[1][0]),
                    lambda x: breakpoints[2][1] + k[2]*(x - breakpoints[2][0])]
        return np.piecewise(float(x), condlist, funclist).reshape(1,)[0]
    
### Vehicle class ###
# Vehicle class. You could add your own helper functions freely to the class, and not required to use the functions defined
# But please keep the rest untouched!
class Vehicle(object):
    
    def __init__(self, id, start_node, end_node, max_travel_time, speed_factor, consumption_rate, battery_capacity):
        ''' Initialize the vehicle
        Args:
            id::int
                id of the vehicle
            start_node::Node
                starting node of the vehicle
            end_node::Node
                ending node of the vehicle
            max_travel_time::float
                maximum time allowed for the vehicle (including travel and charging time)
            speed_factor::float
                speed factor of the vehicle
            consumption_rate::float
                consumption rate of the vehicle
            battery_capacity::float
                battery capacity of the vehicle
        '''
        self.id = id
        self.start_node = start_node
        self.end_node = end_node
        self.max_travel_time = max_travel_time
        self.speed_factor = speed_factor
        self.consumption_rate = consumption_rate
        self.battery_capacity = battery_capacity
        self.battery_level = battery_capacity
        # travel time of the vehicle
        self.travel_time = 0
        # charging time of the vehicle
        self.charging_time = 0
        # total battery consumed by the vehicle
        self.battery_consumption = 0
        # all the nodes including depot, customers, or charging stations (if any) visited by the vehicle
        self.node_visited = [self.start_node] # start from depot
        # record the battery charged each time when visiting a charging station
        self.battery_charged = []
        # record the battery level of the vehicle each time when leaving the charging station 
        self.battery_charged_to = []
        self.cs_visited=[]
        
    def check_time(self):
        '''Check whether the vehicle's travel time plus charging time is over the maximum travel time or not
        Return True if it is not over the maximum travel time, False otherwise
        '''
        if self.travel_time + self.charging_time <= self.max_travel_time:
            return True
        return False
        
    def check_battery(self):
        ''' Check the total battery consumed by the vehicle
        Return True if the battery consumed is not larger than the summation of original battery capacity (when leaving the depot) plus the battery charged, False otherwise
        '''
        if len(self.battery_charged) == 0:
            return self.battery_consumption <= self.battery_capacity
        else:
            return self.battery_consumption <= sum(self.battery_charged) + self.battery_capacity
    
    def check_return(self):
        ''' Check whether the vehicle's return to the depot
        Return True if returned, False otherwise
        '''
        if len(self.node_visited) > 1:
            return self.node_visited[-1] == self.end_node
            
    def __str__(self):
        return 'Vehicle id: {}, start_node: {}, end_node: {}, max_travel_time: {}, speed_factor: {}, consumption_rate: {}, battery_capacity: {}'\
            .format(self.id, self.start_node, self.end_node, self.max_travel_time, self.speed_factor, self.consumption_rate, self.battery_capacity)
#*******************************************************************************      
  # Added by Stella HE CHEN  
    def copy(self):
        return copy.deepcopy(self)
        
    def check_battery_history(self):
        x = np.sum(np.array([(self.battery_charged_to[i]<self.battery_charged[i] or self.battery_charged[i]<0) for i in range(len(self.battery_charged_to))]))
        return x==0
#*******************************************************************************  
    def try_route(self,nodelist, chargeTo, optimize=False):
        feasible=True
        
        self.reset()
        cs_list= [nd for nd in nodelist if nd.type==2]
        #print("trying route", [x.id for x in nodelist]," with CS no", chargeTo,len(cs_list)==len(chargeTo) )
        assert (len(cs_list)==len(chargeTo) )
        iCS=0
        nodes=[self.start_node]+ nodelist+ [self.end_node]  # full route
    # node list exclusing depot 
        for i in range(len(nodes)-1):
            distance= ((nodes[i].x-nodes[i+1].x)**2 + (nodes[i].y-nodes[i+1].y)**2)**0.5
            self.battery_level-=distance *self.consumption_rate
            self.travel_time+=distance/self.speed_factor
            self.battery_consumption+=distance *self.consumption_rate
            self.node_visited .append(nodes[i+1])
            if self.battery_level<0:
                feasible=False
                return 100000
            if nodes[i+1].type==1: #next node is a customer
                self.travel_time+= nodes[i+1].service_time
                
            elif nodes[i+1].type==2: #node is a charging station
#print(self.battery_level, chargeTo[iCS]-self.battery_level)
                if chargeTo[iCS]>self.battery_level:
                    nodes[i+1].charge_to(self,chargeTo[iCS])
                else:
                    self.battery_charged.append( 0)  
                    self.battery_charged_to .append(self.battery_level)
                # self.charging_time+= nodes[i+1].reverse_charging_function(chargeTo(iCS))-nodes[i+1].reverse_charging_function(self.battery_level)
                # self.battery_charged.append( chargeTo(iCS)- self.battery_level  )  # check if positive
                # self.battery_charged_to .append(chargeTo[iCS])
                # self.battery_level==chargeTo[iCS]
                # self.cs_visited.append(nodes[i+1])
                iCS+=1
        if optimize and len(cs_list)>=1: # if extra battery left, try charge less in the last CS
            battery_left=self.battery_level
            if self.check_return and battery_left>0:
                               
                chargeToNew = chargeTo[:-1]+[chargeTo[-1]-battery_left]
                
                self.try_route(nodelist,chargeToNew)
                
        if self.check_time==False or self.check_battery_history==False:
            feasible=False
            return self.max_travel_time*100
        else:
            #print("travel time:",self.travel_time ,"charge time:",self.charging_time)
            return  self.travel_time + self.charging_time

#*******************************************************************************    
    def reset(self):
        self.battery_level = self.battery_capacity       
        self.travel_time = 0 
        self.charging_time = 0         
        self.battery_consumption = 0
        # all the nodes including depot, customers, or charging stations (if any) visited by the vehicle
        self.node_visited = [self.start_node] # start from depot
        self.battery_charged = []       
        self.battery_charged_to = []
        self.cs_visited=[]
    

### EVRP state class ###
# EVRP state class. You could and should add your own helper functions to the class
# But please keep the rest untouched!
class EVRP(State):
    
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
        
# a test
# =============================================================================
#         v=Vehicle(1, self.depot,self.depot,
#                    self.vehicle.max_travel_time,self.vehicle.speed_factor,
#                    self.vehicle.consumption_rate,self.vehicle.battery_capacity
#                    )
#         v.battery_level=0
#         self.CSs[0].charge_to(v,15000)
#         print("OK")
# 
# =============================================================================

            
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
        # depot is sa super charging station with zero charging time
        
#for each customer, geberated teh route involing one CS after servicing it
        def dist(node1, node2):
            return ( (node1.x - node2.x)**2 + (node1.y - node2.y)**2 )**0.5
            #return float(cdist([[node1.x, node1.y]], [[node2.x, node2.y]], 'euclidean'))

        l=len(self.customers)
        #[0:l]
        distances_to_depot=[ dist(cust, self.depot) for cust in self.customers] 
        #[0:l-1]
        distances_to_next=[ dist(self.customers[i], self.customers[i+1]) for i in range(l-1)]
        time_to_next2=[ dist(self.customers[i], self.customers[i+1])/self.vehicle.speed_factor + self.customers[i+1].service_time for i in range(l-1)]
        #time_to_next=[ dist(self.customers[i], self.customers[i+1])/self.vehicle.speed_factor for i in range(l-1)]       
        
        battery_to_next=[ dist(self.customers[i], self.customers[i+1])*self.vehicle.consumption_rate for i in range(l-1)]
        #loop
        
        #testcode___________________________________________________
        # v=Vehicle(1, self.depot,self.depot,
          # self.vehicle.max_travel_time,self.vehicle.speed_factor,
          # self.vehicle.consumption_rate,self.vehicle.battery_capacity
          # )
        # time=[]
        # b=[]
        # b_opti=[]
        # for cus in self.customers:
            # time=[]
            # time_opti=[]
            # #time.append(v.try_route([cus],[]))
            # #b.append(v.battery_level)
            # for cs in self.CSs:
                # v.reset()
                
                # time.append(v.try_route([cs, cus],[v.battery_capacity]))
                # b.append(v.battery_level)
                
                # time_opti.append(v.try_route([cs, cus],[v.battery_capacity], optimize=True))
                # print(v.try_route([cs, cus],[v.battery_capacity], optimize=True))                
                # b.append(v.battery_level)
            # #print(time)  
       
        # return 0
        #testcode___________________________________________________
        
        self.vehicles = []
        self.route=[]
        # total travel time of the all the vehicle used
        self.travel_time = 0
        # total charge time of the all the vehicle used
        self.charging_time = 0
        # record the all the customers who have been visited by all the vehicles, eg. [Customer1, Customer2, ..., Customer7, Customer8]
        self.customer_visited = []
        # record the unvisited customers, eg. [Customer9, Customer10]
        self.customer_unvisited = self.customers.copy()
        i=0
        while i<l:
            k=i
            timesum=distances_to_depot[k]/self.vehicle.speed_factor + tour[k].service_time
            bsum=self.vehicle.battery_capacity - distances_to_depot[k]*self.vehicle.consumption_rate
            blist=[self.vehicle.battery_capacity,bsum] # record battery level list
            
            while k<l-1 and bsum >=- 2*self.vehicle.battery_capacity:
                timesumNew=timesum + time_to_next2[k]
                if timesumNew<=self.vehicle.max_travel_time- distances_to_depot[k+1]/self.vehicle.speed_factor:
                    timesum=timesumNew
                    bsum-=battery_to_next[k]
                    blist.append(bsum)
                    k+=1
                else:
                    break
            #print("round",i, "try reach", k)        
            bsum-=  distances_to_depot[k]*self.vehicle.consumption_rate
            blist.append(bsum)
            while k>=i:
                    #need to insert CS
                       
                startind=np.argmax(np.array(blist)<0)
                startind=1
                # can travel furthest from customer i to k  
                
                candidate= tour[i:k+1]
                #print("checking possible tour:", len(candidate),i,k, startind)
                tour_candidates=[candidate]
                #candidate= [tour[p] for p in range(i:k+1)].insert( m, cs) for m in range(startind-1:)
                for cs in self.CSs:
                    for m in range(startind-1,k-i+2):
                        templ=tour[i:k+1]
                        templ.insert(m, cs) 
                        #templ=tour[i:k+1]+[cs]
                        #print([item.id for item in templ])
                        tour_candidates.append( templ )
                for cs1,cs2 in [(x,y) for x in self.CSs for y in self.CSs]:
                    for m1 in range(startind-1,k-i+2):
                        for m2 in range(startind-1,m1):
                            templ=tour[i:k+1]
                            templ.insert( m1, cs1)
                            templ.insert( m2, cs2) 
                            #print([item.id for item in templ])
                            tour_candidates.append( templ )
                #print("No of tours to check:", len(tour_candidates))
                besttime=self.vehicle.max_travel_time*l*10
                #bestroute=[]
                
                v1=Vehicle(i, self.depot,self.depot,
                          self.vehicle.max_travel_time,self.vehicle.speed_factor,
                          self.vehicle.consumption_rate,self.vehicle.battery_capacity
                          )
                # v0=Vehicle(i, self.depot,self.depot,
                          # self.vehicle.max_travel_time,self.vehicle.speed_factor,
                          # self.vehicle.consumption_rate,self.vehicle.battery_capacity
                          # )
                vNew=[]
                indicator=0
                for tr in tour_candidates:
                    #print ([item.id for item in tr])
                    place_of_Charge= [ p for p in tr if p.type==2]
                    no_charge=len(place_of_Charge)
                    
                    v1.reset()
                    #v0.reset()
                    thistime1=v1.try_route(tr, [v1.battery_capacity]*no_charge,optimize=True)  # record and return time and battery level
                    #thistime0=v0.try_route(tr, [v0.battery_capacity]*no_charge, optimize=False)  # record and return time and battery level
                    
                    if thistime1<besttime :
                        indicator=1
                        besttime= thistime1
                        vNew.append(v1.copy())

                if indicator==0 or vNew[-1].travel_time+vNew[-1].charging_time>self.vehicle.max_travel_time:#route not feasible even after insert CS
                    k-=1
                    continue
                else:  
                    self.vehicles.append(vNew[-1])
                    self.route.append(vNew[-1].node_visited)
                    self.customer_visited.extend([tour[p] for p in range(i,k+1)])
                    for p in range(i,k+1):
                        self.customer_unvisited.remove(tour[p])
                    i=k+1

        #print([x.id for x in self.customer_unvisited])
        #print([x.id for x in self.customer_visited])
        #print([x.id for x in self.customers])
        
        assert(len(self.customer_visited)==len(self.customers))
        assert(len(self.customer_unvisited)==0)
        self.travel_time=sum( [v.travel_time for v in self.vehicles] )
        self.charging_time=sum( [v.charging_time for v in self.vehicles] )          
           

    def objective(self):
        ''' Calculate the objective value of the state
        Return the total travel time and charging time of all vehicles used
        '''
        # or return sum([v.travel_time for v in self.vehicles]) + sum([v.charging_time for v in self.vehicles])
        return self.travel_time +  self.charging_time