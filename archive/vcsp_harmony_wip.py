# -*- coding: utf-8 -*-

# import os
# import pandas as pd
# from collections import defaultdict
from docplex.cp.model import CpoModel

key_shift=	0
melody_input=[  36, 36, 38, 40, 38, 36, 36]


tonality = "major" #or "minor"
N =len(melody_input)

chord_list=[[0, 3, 7], # i
            [0, 4, 7], # I
            [2, 5, 9], # ii
            [2, 6, 9], # II
            [4, 7, 11], # iii
            [4, 8, 11], # III
            [5, 8, 0], # iv
            [5, 9, 0], # IV
            [7, 10, 2], # v
            [7, 11, 2], # V
            [9, 0, 4], # vi
            [9, 1, 4], # VI
            [11, 2, 5], # vii_dim
            [11, 2, 6], # vii
            [11, 3, 6]] # VII

no_chord=len(chord_list)
m = CpoModel(name="Harmony")


# x[i,j]	Notes
# j belongs to {0, ..., N-1}, where n is the number of chords in the sequence
# i represents voices (1=alto, 2=tenor, 3=bass). 
# x[0,j] refers to the soprano voice, but this is provided and not a decision variable

arr=[(i,j) for i in range(4) for j in range(N)]
x= m.integer_var_dict(arr, 0, 50, "notes")  # !!!todo:  fix lower/upper bound by Key

# c[j]	Chords
c = m.integer_var_list(N, 0, len(chord_list)-1, "chords")


lb, ub=[19,12,5],[38,28,26]
for j  in range(N):  
    for i in range(1,4):
        m.add(m.greater(x[i,j],lb[i-1]-1)) 
        m.add(m.less(x[i,j],ub[i-1]+1))
# =============================================================================
# Hard constraints            

#notes must be in the same chord
for j in range(N):
    x[0,j]==melody_input[j]  # read input
    for i in range(1,4):
        for l in range(no_chord):
            m.add(m.if_then(l==c[j],m.allowed_assignments((x[i,j]-24)%12,chord_list[l])))
            
# #Tonality Requirements -first/last grade
m.add(m.if_then(tonality=="major",c[0]==1))
m.add(m.if_then(tonality=="minor",c[0]==0))
m.add(c[N-1] <2 )
m.add(m.if_then(tonality=="major",c[N-1]==1))
for j in range(N-1):
    if j%4==3:
        m.add(c[j]!=c[j+1])
           
# #forbidden to write voice overlap (strictly between soprano, alto and tenor, not strictly between tenor and bass):
for j in range(N):
    m.add(x[0,j]>x[1,j])
    m.add(x[1,j]>x[2,j])
    m.add(x[2,j]>=x[3,j])
    
# for j in range(N-1):  # !!!change to soft constraint?
#     m.add(x[0,j]>x[1,j+1])
#     m.add(x[1,j]>x[2,j+1])
#     m.add(x[2,j]>=x[3,j+1])
#  # No parallel fifths or octaves are allowed:  
for j in range(N-1):
    for i in range(4):
        for k in range(i+1,4):
            m.add(m.if_then((x[i,j]-x[k,j])%7==0,(x[i,j+1]-x[k,j+1])%7!=0))
            m.add(m.if_then((x[i,j]-x[k,j])%12==0,(x[i,j+1]-x[k,j+1])%12!=0))
# =============================================================================            
            
            
# =============================================================================
# Soft constraints          

#Grade transition constraints 
dic_chord_progression={(2,5) : 0,(5,1) : 0,(1,2) : 0,(4,5) : 0,(6,5) : 0,(6,2) : 0,(1,4) : 0,(5,6) : 0,(6,4) : 0,(5,1) : 0,(1,6) : 0,(1,1) : 0.3,(5,5) : 0.3,(6,1) : 0.3,(6,6) : 0.3,(4,2) : 0.4,(4,4) : 0.4,(2,2) : 0.4,(1,5) : 0.5,(4,1) : 1,(2,1) : 1,(2,6) : 1}      
def cost_chord_progression(c1,c2):
    return(  dic_chord_progression.get((chord_list[c1][0],chord_list[c2][0]),1000) )
    
# #cost= sum([cost_chord_progression(c[l],c[l+1]) for l in range(N-1)]) 
cost1= m.integer_var_list(N, 0,100000, "Progression cost")            
for j in range(N-1):
    for c1 in range(no_chord):
        for c2 in range(no_chord):
            m.add(m.if_then(m.logical_and(c1==c[j],c2==c[j+1]),cost1[j]==int(10*cost_chord_progression(c1,c2))) )
    

#Grade frequency constraints



#Voice range constraints 
    
chg=1   
lb, ub, slb, sub,stepl, stepu=[19,12,5],[38,28,26],[24,17, 10 ],[33,23 ,20 ], [0,1,1]  ,[4,3,1]      
cost3= m.integer_var_list(N, 0,40, "Voice range cost")                    
for j  in range(N):  
    for i in range(1,4):
        m.add(m.if_then(m.less(x[i,j],slb[i-1]),cost3[j]==chg*(slb[i-1]-x[i,j])+stepl[i-1])) 
        m.add(m.if_then(m.greater(x[i,j],sub[i-1]),cost3[j]==chg*(x[i,j]-sub[i-1])+stepu[i-1])) 


#Distance constraints  

cost4= m.integer_var_list(N, 0,40, "Distance cost") 



#Contrary motion constraints
cost5= m.integer_var_list(N, 0,40, "Contract motion cost")

for j  in range(N-1): 
    m.add(m.if_then(m.logical_and(x[0,j]<x[0,j+1],x[3,j]<x[3,j+1]),cost5[j]==10))
    m.add(m.if_then(m.logical_and(x[0,j]>x[0,j+1],x[3,j]>x[3,j+1]),cost5[j]==10 ))
    m.add(m.if_then(m.logical_and(x[0,j]==x[0,j+1],x[3,j]==x[3,j+1]),cost5[j]==10))
    m.add(m.if_then(m.logical_and(x[0,j]!=x[0,j+1],x[3,j]==x[3,j+1]),cost5[j]==4 ))
    
m.minimize(m.sum(cost3[j]+cost1[j]+cost5[j] for j in range(N)))        
#m.minimize(m.sum(cost5[j] for j in range(N)))          
sol = m.solve(log_output = True)

print(sol.get_objective_values())       
print(sol.print_solution())
