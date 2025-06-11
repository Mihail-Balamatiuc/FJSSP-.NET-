from typing import List # For type hints to clarify data types

from models import Job, Machine, Task
from scheduler import Scheduler
from utils import run_gannt_chart, save_chart_results



### Data Reading from the file and object creation ###

allJobs: List[Job] = []  # Will contain all the jobs with their tasks
jobsNr: int = 0          # Will contain the number of jobs
machinesNr: int = 0      # Will contain the number of machines
ind: int = 0             # Will keep the job Id nr

# Here is the read function in github data set format which is the following:
"""
First line: <number of jobs> <number of machines>
Then one line per job: <number of operations> and then, for each operation, <number of machines for this operation> and for each machine, a pair <machine> <processing time>.
Machine index starts at 0.
"""
with open('pythonService/dataset_github_compare.txt', 'r') as file:
    #read by lines
    first = True
    for line in file:
        allTasks = []# will keep all the tasks for a job before adding it to allJobs
        elements = line.split()
        if first: #we read the size of the problem JobsNr, MachinesNr
            jobsNr, machinesNr = int(elements[0]), int(elements[1])
            first = False
        else: #here we read the tasks for each job and create the jobs
            elements: List[int] = [int(e) for e in elements]        # Convert all to integers
            curr_index: int = 0                                     # The index of where we are in the number array
            num_ops: int = elements[curr_index]                     # Will contain the number of operations for the current job
            curr_index += 1                         
            operations: List[List[Task]] = []                       # List of list of tasks

            for op in range(num_ops):                               # Here we "iterate" through the operations, or until the range of operations
                num_machines: int = elements[curr_index]            # Here we'll keep the number of machines
                curr_index += 1
                task_list: List[Task] = []                          # A list of task for the current operation's tasks
                
                for m in range(num_machines):                       # Here we "iterate" through the machines and their durations, or until the range of machines
                    machine_id: int = elements[curr_index]          # First element of the pair is machine_id
                    curr_index += 1                         
                    processing_time: int = elements[curr_index]     # Second element of the pair is processing time
                    curr_index += 1
                    task: Task = Task(machine_id, processing_time)  # We create the task
                    task_list.append(task)                          # We append it to the current operaion's task list
                    
                operations.append(task_list)                        # We append the current operation with it's task to our operation list for the current job
            
            allJobs.append(Job(ind, operations))                    # We create the job with it's index and it's operaion list
            ind += 1                                                # We increase the job index


## Here is the read function for my format ##

# with open('pythonService/dataset2.txt', 'r') as file:
#     #read by lines
#     first = True
#     for line in file:
#         allTasks = []# will keep all the tasks for a job before adding it to allJobs
#         elements = line.split()
#         if first: #we read the size of the problem JobsNr, MachinesNr
#             jobsNr, machinesNr = int(elements[0]), int(elements[1])
#             first = False
#         else: #here we read the tasks for each job and create the jobs
#             isMachine = True
#             isDuration = False
#             add = []                                        #will keep the task list to be added
#             currTaskNumbers = []                            #will keep the current task's [machine, duration]
#             for element in elements:                        #we go through the elements of the job separated by space as strings
#                 if(isMachine):          
#                     if(element[0] == '['):                  #if it's the first one we remove the bracket
#                         currTaskNumbers.append(int(element[1:]))
#                     else:
#                         currTaskNumbers.append(int(element))
#                     #We swap the element's type processing
#                     isMachine = False
#                     isDuration = True
#                 else:
#                     if(element[len(element) - 1] == ']'):
#                         currTaskNumbers.append(int(element[:-1]))
#                         add.append(Task(currTaskNumbers[0], currTaskNumbers[1]))
#                         currTaskNumbers.clear()
#                         allTasks.append(copy.deepcopy(add))
#                         add.clear()
#                     else:
#                         currTaskNumbers.append(int(element))
#                         add.append(Task(currTaskNumbers[0], currTaskNumbers[1]))
#                         currTaskNumbers.clear()
#                     #We swap the element's type processing
#                     isMachine = True
#                     isDuration = False
            
#             allJobs.append(Job(ind, allTasks))
#             ind += 1

## Here the read function for my format ends ##

# Define machines
machines = []
for i in range(machinesNr):
    machines.append(Machine(i))
# We finally initialise the scheduler
scheduler = Scheduler(allJobs, machines)

# This is the dictionaire for the heuristic names
heuristic_names = {
    'SPT'   : 'Shortest Processing Time',
    'LPT'   : 'Longest Processing Time',
    'MWR'   : 'Most Work Remaining',
    'LWR'   : 'Least Work Remaining',
    'SA'    : 'Simulated Annealing',
    'HC'    : 'Hill Climber',
    'TS'    : 'Tabu Search',
    'GA'    : 'Genetic Algorithm',
    'ILS'   : 'Iterated Local Search'
}

### Main Execution and Visualization ###
scheduler = Scheduler(allJobs, machines)

heuristics: List[str] = []
with open('pythonService/compare_algorithms.txt', 'r') as file:
    #read by lines
    for line in file:
        heuristics = line.split()


for heuristic in heuristics:
    save_chart_results(heuristic, 5, 60, scheduler, heuristic_names)
