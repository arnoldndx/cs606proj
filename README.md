# CS606: AI Planning & Decision Making Project (Group 3)
Generating Four-Part Harmony (Using Constraint Programming and Evolutionary Techniques)

## Getting Started

Create environment from environment file (to be created)

`conda env create -f cs606_env.yaml`

## Usage

```
usage: run_harmony_gen.py [-h] [--method {'mp', 'cp', 'ga', 'alns'}] [--file FILE] [--weights {'defined', 'trained'}] [--input FILEPATH]

Optional arguments (all have defaults):
  -h, --help            show this help message and exit
  --method {'mp', 'cp', 'ga', 'alns'}    
                        Choice of harmony generation method
  --file FILE           Filename prefix. You should give a meaningful name for easy tracking.
  --weights {'defined', 'trained'}
                        Choice of whether to use train weights against a body of work or to use a pre-defined set in csv
  --input FILEPATH      Filepath of input melody
```

### Examples

Examples of run scripts in anaconda prompt
