
import src.music_functions 
import pandas as pd
#%%
def evaluate_cost(list_x, list_c , tonality, meter=4, first_on_beat=0, mode="L") : 
    N=len(list_c)
    assert len(list_x)==4
    assert len(list_x[0])==N
    c=list_c.copy()
    x= {  (i,j): list_x[i][j] for i in range(4) for j in range(N)} 
    
    weight_df = pd.read_csv("../data/soft_constraint_weights.csv")
    soft_constraint_w_weights={}
    for _,name, w in weight_df.itertuples(): #name population is same as soft_constraint_options
        soft_constraint_w_weights[name]=float(w)
    file_progression_cost="chord_progression_major_v1.csv" if tonality=="major" else "chord_progression_minor_v1.csv"
    

    chord_progression_costs=src.music_functions.func_get_progression_costs(file_progression_cost) 
    # Importing Chord Vocabulary
    # if tonality=="major":
    #     chord_df = pd.read_csv("../data/chord_vocabulary_major.csv", index_col = 0)
    # else:
    #     chord_df = pd.read_csv("../data/chord_vocabulary_minor.csv", index_col = 0)
    # chord_vocab = []
    # for name, note_intervals in chord_df.itertuples():
    #     chord_vocab.append(Chord(name, set(int(x) for x in note_intervals.split(','))))
  
#***************************************************************************************************************  
        
        
    def hard_constraint_chord_bass_repetition(weight=1000):
        cost=   weight* sum((c[j] == c[j+1]) <= (x[3,j] != x[3,j+1]) for j in range(N-1)) 
        return cost
    # def hard_constraint_adjacent_bar_chords(weight=1000):
        # for j in range(1,N):
            # if j % musical_input.meter == musical_input.first_on_beat:
                # m.add_constraint(c[j] != c[j-1])
    def hard_constraint_adjacent_bar_chords(weight=1000):
        cost=0
        for j in range(1,N):
            if j % meter ==first_on_beat:
                cost+= (c[j] == c[j-1])
        return weight*cost
    def hard_constraint_voice_crossing(weight=1000):

        return weight*sum(x[i,j] >= x[i+1,j]  for i in range(3) for j in range(N))
    
    def hard_constraint_parallel_movement(weight=1000, disallowed_intervals = [7, 12]):
        cost=0
        for j in range(N-1):
            for i in range(3):
                for k in range(i+1, 4):
                    for interval in disallowed_intervals:
                        cost+=( (x[i,j]-x[k,j] ==interval)<= ( x[i,j+1]-x[k,j+1] !=interval) )
                        cost+=( (x[i,j]-x[k,j] ==interval+12)<= ( x[i,j+1]-x[k,j+1] !=interval+12) )
                        cost+=( (x[i,j]-x[k,j] ==interval+24)<= ( x[i,j+1]-x[k,j+1] !=interval+24) )
        return weight*cost
    def hard_constraint_chord_spacing( weight=1000, max_spacing = [12, 12, 16]):
        cost=0
        for j in range(N):
            for i in range(3):
                cost+=(x[i,j] - x[i+1,j] <= max_spacing[i-1])
        return weight*cost        

#***************************************************************************************************************       
    def soft_constraint_chord_progression( weight=1):

        cost= weight*sum(chord_progression_costs[(c[j],c[j+1])] for j in range(N-1))
        return cost
    def soft_constraint_leap_resolution(max_leap=10, weight=1): # leaps more than an interval of a major 6th should resolve in the opposite direction by stepwise motion(

        cost= weight*sum((1-(x[i,j] - x[i,j+1] <= max_leap) for i in range(1,4) for j in range(N-1))  )                
        return cost
    def soft_constraint_melodic_movement( weight=1):
        pass
    def soft_constraint_chord_repetition( weight=2):

        cost= weight*sum(c[j]==c[j+1] for j in range(N-1))
        return cost
    def soft_constraint_chord_bass_repetition( weight=1):    
        pass
    def soft_constraint_adjacent_bar_chords( weight=1):
        pass
    
    
    def soft_constraint_note_repetition( weight=2):
        cost= weight*sum(sum( (x[i,j]==x[i,j+1])*(x[i,j+1]==x[i,j+2]) 
                          for i in range(1,4))
                          for j in range(N-2))
        return cost                 

    def soft_constraint_parallel_movement( weight=1):# is a hard constraint
        pass
    
    def soft_constraint_voice_overlap(weight=1):

        return weight*sum((x[i,j+1] <= x[i+1,j]-1) for i in range(3) for j in range(N-1))
    def soft_constraint_chord_spacing( weight=1):
        pass
    
    def soft_constraint_distinct_notes(weight=1):#Chords with more distinct notes (i.e. max 3) are rewarded
                                 
        cost=weight*sum(sum( sum( (x[i,j]- x[k,j]==12) + (x[i,j]- x[k,j]==24) + (x[i,j]- x[k,j]==36) for i in range(3) ) for k in range(4) ) for j in range(N))
        return cost
    def soft_constraint_voice_crossing( weight=1): # is a hard constraint
        pass
    
    def soft_constraint_voice_range( slb = [24,17, 10], sub = [33,23 ,21],weight=1):
        return weight*sum( max(x[i,j] -sub[i-1],0) +  max(slb[i-1]-x[i,j],0) for i in range(1,4) for j in range(N))

    if mode=="L":
        return [hard_constraint_chord_bass_repetition(),
                hard_constraint_adjacent_bar_chords(),
                hard_constraint_voice_crossing(),
                hard_constraint_parallel_movement(),
                hard_constraint_chord_spacing(),
                soft_constraint_chord_progression(),
                soft_constraint_leap_resolution(),
                soft_constraint_chord_repetition(),
                soft_constraint_note_repetition(),
                soft_constraint_voice_overlap(),
                soft_constraint_distinct_notes(),
                soft_constraint_voice_range()
                ]
    else:
        return (hard_constraint_chord_bass_repetition()+
                hard_constraint_adjacent_bar_chords()+
                hard_constraint_voice_crossing()+
                hard_constraint_parallel_movement()+
                hard_constraint_chord_spacing()+
                soft_constraint_chord_progression()+
                soft_constraint_leap_resolution()+
                soft_constraint_chord_repetition()+
                soft_constraint_note_repetition()+
                soft_constraint_voice_overlap()+
                soft_constraint_distinct_notes()+
                soft_constraint_voice_range()
                )
#cost=evaluate_cost(solution[:-1],solution[-1] ,"major")                