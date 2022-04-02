**CS606: AI Planning & Decision Making Project (Group 3)**
# Auto-generation of four-part harmonies

This repository contains code to generating a four-part harmony in accordance with composition rules of Western music which was created for our group project for CS606 under SMU's MITB programme.

## Brief Introduction and Methods
The harmony generator applies four different optimisation techniques to generate potential solutions to four-part harmony given a fixed melody.

The techniques attempted are:
- Mixed-integer Linear Programming
- Constraint Programming
- Adaptive Large Neighbourhood Search
- Genetic Algorithm

Please refer to the rest of the readme for information on how to operate this harmony generator. We have included sample inputs under the data folder, although you are free to use your own inputs as well, provided they follow the correct format.

## Getting Started

Create environment from environment file.

`conda env create -f cs606_env.yaml`

Please note that the MILP and CP models require the installation of IBM's DOcplex solver.

## Usage

```
usage: run_harmony_gen.py [-h] [--method {'mp', 'cp', 'ga', 'alns'}] [--file FILE] [--weights {'defined', 'trained'}] 
                          [--weights_data FILEPATH] [--hard_constraints_choice FILEPATH] [--input_melody FILEPATH] [--music_index INT]
                          [--max_generation INT] [--population_size INT] [--mutation_probability [lower_bound, higher_bound]]

Optional arguments (all have defaults):
  -h, --help            Show this help message and exit
  --method  {'mp', 'cp', 'ga', 'alns'}    
                        Choice of harmony generation method, defaults to 'mp'.
  --file                Filename prefix. You should give a meaningful name for easy tracking.
  --weights {'defined', 'trained'}
                        Choice of whether to use train weights against a body of work or to use a pre-defined set in csv, defaults to defined.
  --weights_data        Filepath to weights data csv (if 'defined' weights selected) or folder of midi files (if 'trained' weights selected)
  --hard_constraints_choice    Filepath for hard constraint choices in csv.
  --time_limit          Integer expressing the time limit for the solver in seconds.
  --input_melody        Filepath of input melody
  --music_index         Integer expressing the index of the musical corpus to be used (for csv inputs with multiple melodies)
  --max_generation      Number of generations to iterate through for genetic algorithm
  --population_size     Population size for genetic algorithm
  --mutation_probability       Mutation probability for genetic algorithm = [lower_bound, higher_bound]
```

### Examples

Examples of run scripts in anaconda prompt

```
python run_harmony_gen_generalised.py --file 'test' --input_melody ../data/test_melody_hatikvah(israel)_4_1_minor_1.mid --method alns --time_limit 100
```
