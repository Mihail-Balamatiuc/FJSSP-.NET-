# Scheduler class manages the entire scheduling process
import copy
import math
import random
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

from models import Job, Machine, Task
from config_loader import config


class Scheduler:
    def __init__(self, jobs: List[Job], machines: List[Machine]):
        self.jobs: List[Job] = copy.deepcopy(jobs)            # List of Job objects
        self.machines: List[Machine] = copy.deepcopy(machines)    # List of Machine objects
        self.global_max: int = 0
        self.work_remaining: Dict[int, List[int]] = {}      #will have the job id as a key and an array which will repesent the work remaining for each position
                                                            #ex: for [2 1 2 3 2 1] [0 3] [1 5] [1 6 1 3] will be [13 ,11 ,8, 4, 3, 0] (we take the min from task list) 

        #here we will compute the reverse pref sums for work remaining

        for job in self.jobs:
        #we initialise the keys first
            self.work_remaining[job.job_id] = [0] #this is goint to be the 0 at the end
            sum: int = 0
            for task_list in reversed(job.operations):
                mn = task_list[0].duration
                for task in task_list:
                    mn = min(mn, task.duration)
                sum += mn
                self.work_remaining[job.job_id].append(sum)
            
            #we reverse the pref array
            self.work_remaining[job.job_id].reverse()


    def shortest_processing_time(self) -> Optional[Tuple[Job, Task]]:
        next_task_touple: Optional[Tuple[Job, Task]] = None       
        for job in self.jobs:                           # We go throught jobs                 
            if (not job.is_complete()):                 # Check if it's completed (still has tasks)
                curr_task_list: Optional[List[Task]] = job.get_next_task_list()    # Get it's next task list 
                if (curr_task_list is not None):        # Check if it's not empty
                    if(next_task_touple == None):              
                        next_task_touple = (job, curr_task_list[0]) #format: (job, task)
                    for task in curr_task_list:         # Check the tasks from the current
                        if(task.duration < next_task_touple[1].duration):
                            next_task_touple = (job, task)
        
        return next_task_touple    # format: (job, task)
    
    def longest_processing_time(self) -> Optional[Tuple[Job, Task]]:
        next_task_touple = None       
        for job in self.jobs:                           # We go throught jobs                 
            if (not job.is_complete()):                 # Check if it's completed (still has tasks)
                curr_task_list: Optional[List[Task]] = job.get_next_task_list()    # Get it's next task list 
                if (curr_task_list is not None):        # Check if it's not empty
                    if(next_task_touple == None):              
                        next_task_touple = (job, curr_task_list[0]) #format: (job, task)
                    for task in curr_task_list:         # Check the tasks from the current
                        if(task.duration > next_task_touple[1].duration):
                            next_task_touple = (job, task)
        
        return next_task_touple    # format: (job, task)
    
    # Most work remaining with picking to execute the task on the machine with least execution time
    def most_work_remaining(self) -> Optional[Tuple[Job, Task]]:
        next_task_touple = None             # Will hold the best (job, task) pair
        next_task_list = None               # Will keep the task list from where we're gonna pick the task
        next_task = None                    # Will keep the next task that will be returned
        next_job = None                     # Will keep the job that will be returned
        mx = None                           # Tracks the current maximum remaining work
        for job in self.jobs:
            if(not job.is_complete()):
                work_remaining: int = self.work_remaining[job.job_id][job.current_operation_index]   # keeps the remaining work
                if(mx == None or work_remaining > mx):                                          # if mx is not initialized or we find a new max then we update
                    mx = self.work_remaining[job.job_id][job.current_operation_index]
                    next_task_list = job.get_next_task_list()
                    next_job = job

        # Here we pick the task with min duration from the task list
        next_task: Task = next_task_list[0]             # initialize the task with first task in the task list
        for task in next_task_list:
            if(task.duration < next_task.duration):
                next_task = task
        # Here we create the return tuple
        next_task_touple = (next_job, next_task)

        return next_task_touple
    
    # Least work remaining with picking to execute the task on the machine with least execution time
    def least_work_remaining(self):
        next_task_touple = None             # Will hold the best (job, task) pair
        next_task_list = None               # Will keep the task list from where we're gonna pick the task
        next_task = None                    # Will keep the next task that will be returned
        next_job = None                     # Will keep the job that will be returned
        mn = None                           # Tracks the current minimum remaining work
        for job in self.jobs:
            if(not job.is_complete()):
                work_remaining: int = self.work_remaining[job.job_id][job.current_operation_index]   # keeps the remaining work
                if(mn == None or work_remaining < mn):                                          # if mx is not initialized or we find a new min then we update
                    mn = self.work_remaining[job.job_id][job.current_operation_index]
                    next_task_list = job.get_next_task_list()
                    next_job = job

        # Here we pick the task with min duration from the task list
        next_task: Task = next_task_list[0]             # initialize the task with first task in the task list
        for task in next_task_list:
            if(task.duration < next_task.duration):
                next_task = task
        # Here we create the return tuple
        next_task_touple = (next_job, next_task)

        return next_task_touple
    

    # Will generate a starting solution for heuristics
    def generate_initial_solution(self) -> Tuple[List[int], List[List[int]]]:
        operation_sequence: List[int] = []  # Will keep the order of the operations for the jobs (ex after we shuffle: [0, 1, 2, 1, 0, 1])
        for job in self.jobs:
            for _ in job.operations:
                operation_sequence.append(job.job_id)   # Here we just created smth like [0, 0, 1, 1, 1, 2]

        random.shuffle(operation_sequence)              # And here we shuffle it for a random start sollution

        machine_assignment: List[List[int]] = []    # A 2D list where machine_assignment[job_id][operation_index] gives the machine ID for that operation.
                                                    # ex: [[0, 2], [1, 1, 0], [3]] Job 0: Operation 0 on Machine 0, Operation 1 on Machine 2 ...
        for job in self.jobs:
            job_assignment: List[int] = []          # Will create a list with the random picked machine id from the task list for every task and append it to the list
            for task_list in job.operations:             # We iterate through the task lists
                chosen_machine = random.choice(task_list)           # We pick a random possible task(we can call it subtask) with it's macihine that can solve our main task
                job_assignment.append(chosen_machine.machine_id)    # Append it's machine id in our list of machine id's for the job
            machine_assignment.append(job_assignment)               # Append the machine id's for task to our machine assignment list for every job

        return operation_sequence, machine_assignment               # Return both of them, the solution and associated machines
    
    # Will generate a starting solution based on a dispatching rule
    def generate_dispaching_inititial_solution(self, rule: str) -> Tuple[List[int], List[List[int]]]:
        operation_sequence: List[int] = []  # # Will keep the order of the operations for the jobs
        
        machine_assignment: List[List[int]] = []    # A 2D list where machine_assignment[job_id][operation_index] gives the machine ID for that operation.
                                                    # ex: [[0, 2], [1, 1, 0], [3]] Job 0: Operation 0 on Machine 0, Operation 1 on Machine 2 ...
        
        # We make sure to fulfil the list with the needed job size
        for i in range(len(self.jobs)):
            machine_assignment.append([])

        # Here is the code we use in the run() method, we just remember the operations and machines
        while any(not job.is_complete() for job in self.jobs): # do until all the jobs are completed
            # Get all available tasks sorted by the chosen heuristic
            if(rule == 'SPT'):
                next_task_tuple: Optional[Tuple[Job, Task]] = self.shortest_processing_time()   # format: (job, task)
            elif rule == 'LPT':
                next_task_tuple: Optional[Tuple[Job, Task]] = self.longest_processing_time()    # format: (job, task)
            elif rule == 'MWR':
                next_task_tuple: Optional[Tuple[Job, Task]] = self.most_work_remaining()        # format: (job, task)
            elif rule == 'LWR':
                next_task_tuple: Optional[Tuple[Job, Task]] = self.least_work_remaining()       # format: (job, task)
            else:
                raise ValueError(f"Unknown heuristic: {rule}")
            
            if next_task_tuple:
                job, task = next_task_tuple

                # Here we add the data from the dispatching rule in the needed format
                operation_sequence.append(job.job_id)
                machine_assignment[job.job_id].append(task.machine_id)

                machine: Machine = self.machines[task.machine_id]
                # Schedule the task at the earliest possible time for this machine
                start = machine.schedule[-1][2] if machine.schedule else 0  # Get the end time of the last machine's task or make it 0

                # Now we pick the max between the possible start time of the machine computed above and the possible start time of the job (the possible
                #   start time of the job is more or equal to when it's previous operation ended) because we need to satisfy both conditions
                start = max(start, job.last_ending_time)

                end = start + task.duration                                 # Calculate the end time
                machine.add_to_schedule(job.job_id, start, end)             # Add the task to needed machine's schedule
                task.start_time = start                                     # Update Task's start time
                task.end_time = end                                         # Update Task's end time
                self.global_max = max(self.global_max, end)                 # Update global makespan if needed
                job.last_ending_time = task.end_time                        # Update the job's last ending time
                job.complete_operation()                                    # Mark the task as done

        # Here we reset the scheduler and return the result
        self.reset_scheduler()
        return operation_sequence, machine_assignment
        

    # Will compute the makespan for the encoded operations array and machine assignments and it also updates the scheduler with the computed solution
    def compute_makespan(self, operation_sequence: List[int], machine_assignment: List[List[int]]) -> int:
        # Calculates the total completion time (makespan) for a given solution
        self.reset_scheduler() # Clear any existing schedule from previous computings
        job_last_times: List[int] = [0] * len(self.jobs)        # When each job’s last operation finished, job_last_times[job_id] will give it to us
        machine_times: List[int] = [0] * len(self.machines)     # When each machine becomes available, machine_times[machine_id] will give us the next available time
        scheduled_tasks: List[int] = [0] * len(self.jobs)       # Number of operations scheduled per job, basically scheduled_tasks[job_id] will give us the index 
                                                                # of the task of that needs to be executed for the current job

        # We iterate through the solution with job ids
        for job_id in operation_sequence:
            if (scheduled_tasks[job_id] < len(self.jobs[job_id].operations)):
                operation_index: int = scheduled_tasks[job_id]      # Next operation for this job
                machine_id: int = machine_assignment[job_id][operation_index]       # Assigned machine id for the current operation of the job
                operation_task_list: List[Task] = self.jobs[job_id].operations[operation_index]   # Possible tasks for this operation where we're gonna search the task with the machine_id

                for task in operation_task_list:                    # Iterate through the curr task_list of the job to find the task with the needed machine id
                    if (task.machine_id == machine_id):             # If we foun the task with the right machine
                        duration: int = task.duration
                        # Start when both the machine and job are ready
                        start_time: int = max(machine_times[machine_id], job_last_times[job_id])    # We pick the start time which is the max between the machine availability
                                                                                                    # and job last time, since we have to execute the previous tasks first
                        end_time: int = start_time + duration       # We change the end time after completing the current task
                        # Assign times to the task

                        task.start_time = start_time    # We update the start time of the task because it's initially none      
                        task.end_time = end_time        # We update the end time of the task because it's initially none
                        break                           # We break because it should be executed only for the needed task and then stop
                    
                # Update machine and job availability
                machine_times[machine_id] = end_time    # We update the time availability of the machine, we add the duration of the current task
                job_last_times[job_id] = end_time       # We will update the last time in the job when it finished
                self.machines[machine_id].add_to_schedule(job_id, start_time, end_time)     # We add the current task to machine's schedule
                scheduled_tasks[job_id] += 1  # Move to the next operation for this job     # We move to the next operation for the current job

        makespan: int = max(machine_times)  # Makespan is the latest machine finish time
        self.global_max = makespan     # Update scheduler’s makespan
        return makespan
    
    #This function will generate a neighbour solution based on the encoded provided one
    def generate_neighbor(self, solution: Tuple[List[int], List[List[int]]]) -> Tuple[List[int], List[List[int]]]:
        operation_sequence, machine_assignment = copy.deepcopy(solution)   # We separate the operation sequence and the maachines assigned
        # Here we will decide if we swap two operations in the sequence or if we pick a different machine and task for a opperation
        if random.random() < config.global_configs.operation_machine_ratio:
            # Option 1: Swap two operations in the sequence
            new_sequence: List[int] = operation_sequence.copy()    # We make a copy of the current sequence
            i, j = random.sample(range(len(new_sequence)), 2)   # Pick two random distinct positions from the new_sequence
            new_sequence[i], new_sequence[j] = new_sequence[j], new_sequence[i]     # Swap the 2 positions so that we get a neighbour solution
            return new_sequence, machine_assignment     # Return the solution tuple
        else:
            # Option 2: Change the machine for one operation
            new_machine_assignment: List[List[int]] = copy.deepcopy(machine_assignment)  # Deep copy of the machine_assignment array

            limit: int = 0  # Will be the limit for search, in case there are no operations with multiple machine options to be found
            for job in self.jobs:   # We are making the limit equal to the sum of number of operations of each job doubled (not a rule, but I decided to just double)
                limit += len(job.operations) * 2

            while(limit > 0):
                job_id = random.randint(0, len(self.jobs) - 1)      # Get a random job's index
                operation_index = random.randint(0, len(self.jobs[job_id].operations) - 1)  # Get a andom operation's id from the selected job
                task_list = self.jobs[job_id].operations[operation_index]     # Get the task list for the selected operation's index of the selecte job
                if len(task_list) > 1:  # We make sure there’s a choice of machines
                    current_machine = new_machine_assignment[job_id][operation_index]    # We get the selected machine operation from our copy of the machine assignments
                    # Get all other possible machines except the already selected one
                    possible_machines = [task.machine_id for task in task_list if task.machine_id != current_machine]
                    new_machine = random.choice(possible_machines)  # We pick a random machine from the other options we have
                    new_machine_assignment[job_id][operation_index] = new_machine    # We update the new machine for the operation in our new machine assignment
                    break
                limit -= 1

            return operation_sequence, new_machine_assignment   # Return the solution tuple
        
    def simulated_annealing(self,
                            initial_temperature: float = config.simulated_annealing.initial_temperature, 
                            cooling_rate: float = config.simulated_annealing.cooling_rate, 
                            min_temperature: float = config.simulated_annealing.min_temperature, 
                            max_iterations: int = config.simulated_annealing.max_iterations,
                            restarts: int = config.simulated_annealing.restarts
                            ) -> Tuple[Tuple[List[int], List[List[int]]], int]:

        best_solution: Tuple[List[int], List[List[int]]] = self.generate_initial_solution()     # Track the best solution found
        best_makespan: int = self.compute_makespan(*best_solution)                              # Track its makespan

        for i in range(restarts):
            current_solution: Tuple[List[int], List[List[int]]] = self.generate_initial_solution()
            current_makespan: int = self.compute_makespan(current_solution[0], current_solution[1])             # Evaluate it
            temperature: int = initial_temperature   # Start with a high temperature
            iteration: int = 0                       # Count iterations

            # We continue until temperature is low enough or max iterations reached
            while temperature > min_temperature and iteration < max_iterations:
                neighbor: Tuple[List[int], List[List[int]]] = self.generate_neighbor(current_solution)     # Create a new solution
                neighbor_makespan: int = self.compute_makespan(*neighbor)    # Evaluate it
                delta_E: int = neighbor_makespan - current_makespan          # Change in makespan
                # Accept if better (negative delta) or with probability if worse
                if delta_E < 0 or random.random() < math.exp(-delta_E / (temperature * current_makespan)):  
                    current_solution = neighbor
                    current_makespan = neighbor_makespan
                    # Update best solution if this one is better
                    if current_makespan < best_makespan:
                        best_solution = current_solution
                        best_makespan = current_makespan
                temperature *= cooling_rate  # Cool down the temperature    
                iteration += 1               # Increment iteration counter
        

        self.compute_makespan(*best_solution)  # Apply the best solution
        return best_solution, best_makespan    # Return the optimized solution and its makespan

    def hill_climbing(self, 
                      improvement_tries: int = config.hill_climbing.improvement_tries,
                      max_iterations: int = config.hill_climbing.max_iterations,
                      restarts: int = config.hill_climbing.restarts
                      ) -> Tuple[Tuple[List[int], List[List[int]]], int]:
        
        # Initialize the best solution and its makespan
        best_solution: Tuple[List[int], List[List[int]]] = self.generate_initial_solution()
        best_makespan: int = self.compute_makespan(*best_solution)

        # We'll do restart tries
        for i in range(restarts):
            current_solution: Tuple[List[int], List[List[int]]] = self.generate_initial_solution()
            current_makespan: int = self.compute_makespan(*current_solution)    # Compute the makespan of the current solution
            
            improvement_attempts: int = improvement_tries  # We give tries to find a better neighbour, if not found we consider the currens solution as local optimum
            # We iterate till the max or as long as we get improvements
            for j in range(max_iterations):
                if improvement_attempts == 0:
                    break
                # We generate some random list of neighbours and their makespans
                neighbors: List[Tuple[List[int], List[List[int]]]] = [self.generate_neighbor(current_solution) for _ in range(config.hill_climbing.neighbors_number)]
                neighbor_makespans = [self.compute_makespan(*neigh) for neigh in neighbors]
                # We determine the best solution
                neighbors_best_solution: Tuple[List[int], List[List[int]]] = neighbors[0]
                neighbors_best_makespan: int = neighbor_makespans[0]
                for index in range(len(neighbor_makespans)):
                    if(neighbor_makespans[index] < neighbors_best_makespan):
                        neighbors_best_makespan = neighbor_makespans[index]
                        neighbors_best_solution = neighbors[index]

                # Check if we got an improvement
                if neighbors_best_makespan < current_makespan:
                    current_solution = copy.deepcopy(neighbors_best_solution)
                    current_makespan = neighbors_best_makespan
                else:
                    improvement_attempts -= 1
            
            # We check if the found solution is better than our best
            if current_makespan < best_makespan:
                best_solution = current_solution
                best_makespan = current_makespan
                        
        # Apply the best solution to update the scheduler and return it
        self.compute_makespan(*best_solution)
        return best_solution, best_makespan
        
    # Tabu search which generates random solutions and the picks the better ones that are not in the tabu list (forbidden list)
    # In this function we check the moves only by the operation sequence, we don't check the difference for the machines, to consume less time
    def tabu_search(self,
                    tabu_tenure: int = config.tabu_search.tabu_tenure, 
                    max_iterations: int = config.tabu_search.max_iterations
                    ) -> Tuple[Tuple[List[int], List[List[int]]], int]:
        
        current_solution: Tuple[List[int], List[List[int]]] = self.generate_initial_solution()
        current_makespan: int = self.compute_makespan(*current_solution)                        # We calculate the current makespan
        best_solution: Tuple[List[int], List[List[int]]] = copy.deepcopy(current_solution)      # Keeping the best solution
        best_makespan: int = current_makespan                                                   # Keeping the best makespan for the best solution
        # Tabu list to store recent moves (tuples of sequences)
        tabu_list: Deque[Tuple[List[int], List[int]]] = deque(maxlen = tabu_tenure)             # Tabu list with fixed size (will automatically eliminate when exceeded)

        for iteration in range(max_iterations):
            # Generate 15 neighbors
            neighbors: List[Tuple[List[int], List[List[int]]]] = [self.generate_neighbor(current_solution) for _ in range(15)]

            # Find the best non-tabu neighbor
            best_neighbor: Optional[Tuple[List[int], List[List[int]]]] = None       # Will keep the best neighbour
            best_neighbor_makespan: float = float('inf')                            # Best makespan initially is the maximum value bc we need to compute the minimum
            best_move: Optional[Tuple[List[int], List[int]]] = None                 

            # Iterate through the neighbours
            for neighbor in neighbors:
                neighbor_makespan: int = self.compute_makespan(*neighbor)           # Calculate their makespan
                # Simplified move representation: tuple of operation sequences (before, after), e.g. ([0, 1, 0, 1], [1, 0, 0, 1])
                move: Tuple[List[int], List[int]] = (current_solution[0][:], neighbor[0][:])  # Only sequence for simplicity not machine list
                # Check if move is allowed (not tabu or meets aspiration criteria)
                if move not in tabu_list or neighbor_makespan < best_makespan:
                    if neighbor_makespan < best_neighbor_makespan:
                        best_neighbor = neighbor
                        best_neighbor_makespan = float(neighbor_makespan)  # Convert to float for consistency
                        best_move = move
            
            # If no valid move is found, terminate the loop
            if best_neighbor is None:
                break

            # Update current solution
            current_solution = copy.deepcopy(best_neighbor)
            current_makespan = int(best_neighbor_makespan)  # Convert back to int
            
            # Update best solution if improved
            if current_makespan < best_makespan:
                best_solution = copy.deepcopy(current_solution)
                best_makespan = current_makespan
            
            # Add move to tabu list
            tabu_list.append(best_move)

            # Increment iteration
            iteration += 1

        # Apply the best solution and return
        self.compute_makespan(*best_solution)
        return best_solution, best_makespan
    
    ### Todo: add the number of the best solutions carried to the next level ###
    # Here we implement the Genetic Algorithm heuristic
    def genetic_algorithm(self,
                          population_size: int = max(config.genetic_algorithm.population_size, 2), # We neet at leas 2 parents
                          num_generations: int = config.genetic_algorithm.num_generations,
                          crossover_rate: float = config.genetic_algorithm.crossover_rate,
                          mutation_rate: float = config.genetic_algorithm.mutation_rate,
                          tournament_size: int = config.genetic_algorithm.tournament_size
                          ) -> Tuple[Tuple[List[int], List[List[int]]], int]:
        
        
        # Creates a population of solutions and evolves them over generations
        # Initialize a population of random solutions initially
        population: List[Tuple[List[int], List[List[int]]]] = [self.generate_initial_solution() for _ in range(population_size)]
        
        # Evaluate fitness (makespan) for each solutions
        fitnesses: List[int] = [self.compute_makespan(*solution) for solution in population]
        # Initially best solution will be the first generated solution
        best_solution: Tuple[List[int], List[List[int]]] = copy.deepcopy(population[0])  
        best_makespan: int = fitnesses[0]
        
        # Compare and get the best solution from the existing ones
        for ind in range(population_size):
            if(fitnesses[ind] < best_makespan):
                # Keep the best solution from all
                best_solution = copy.deepcopy(population[ind])
                best_makespan: int = fitnesses[ind]

        # We repeat the proces num_generations times
        for generation in range(num_generations):
            # Initialize the new population
            new_population: List[Tuple[List[int], List[List[int]]]] = []

            # Elitism: Always carry over the best solution to the new population
            new_population.append(copy.deepcopy(best_solution))

            # Fill the new population with offspring (kids)
            while len(new_population) < population_size:
                # Tournament Selection: Select two parents
                parent1: Tuple[List[int], List[List[int]]] = self.tournament(population, fitnesses, tournament_size)
                parent2: Tuple[List[int], List[List[int]]] = self.tournament(population, fitnesses, tournament_size)

                # Crossover: Produce a offspring(kid) from the 2 parents with probability crossover_rate
                if(random.random() < crossover_rate):
                    offspring: Tuple[List[int], List[List[int]]] = self.crossover(parent1, parent2)
                else:
                    if(self.compute_makespan(*parent1) < self.compute_makespan(*parent2)):
                        offspring = parent1
                    else:
                        offspring = parent2

                # Mutation: Apply mutation to offspring based on mutation_rate
                if random.random() < mutation_rate:
                    offspring = self.generate_neighbor(offspring)

                # Add offspring to the new population
                new_population.append(offspring)
            
            # Replace the old population with the new one
            population = new_population
            fitnesses = [self.compute_makespan(*solution) for solution in population]

            # Check if there is a better solution, and if there is update the best solution
            for ind in range(population_size):
                if(fitnesses[ind] < best_makespan):
                    # Keep the best solution from all
                    best_solution = copy.deepcopy(population[ind])
                    best_makespan: int = fitnesses[ind]
        
        # Apply the best solution to the scheduler and return it
        self.compute_makespan(*best_solution)
        return best_solution, best_makespan
                    

    # Function for making a new solution out of 2 parent solution for the Genetic Algorithm
    def crossover(self, parent1: Tuple[List[int], List[List[int]]], 
                  parent2: Tuple[List[int], List[List[int]]]) -> Tuple[List[int], List[List[int]]]:

        operation_seq1, machine_assign1 = parent1
        operation_seq2, machine_assign2 = parent2
        # This dictionarry will keep count of the needed number of job indexes appearences in the operation sequence representation
        # (e.g. [2, 1, 0, 0, 1, 0, 2, 2]) will keep {0 : 3, 1 : 2, 2 : 3} 
        needed_count_dict: Dict = {}
        for job_id in operation_seq1:
            needed_count_dict[job_id] = needed_count_dict.get(job_id, 0) + 1

        # Crossover for operation sequence (using one-point crossover), we pick a point in the sequence excluding ends end get the left side from the first parent
        # and right part is build from the second parent
        crossover_point: int = random.randint(1, len(operation_seq1) - 1)
        # get the left from parent1
        offspring_seq: List[int] = operation_seq1[:crossover_point]

        offspring_count_dict: Dict = {}     # Will keep the current existing count
        for job_id in offspring_seq:
            offspring_count_dict[job_id] = offspring_count_dict.get(job_id, 0) + 1

        # Here we build the second part by adding the missing operations(job_id counts) in the order they appear in the second parent's sequence
        for job_id in operation_seq2:
            if offspring_count_dict.get(job_id, 0) < needed_count_dict.get(job_id):
                offspring_count_dict[job_id] = offspring_count_dict.get(job_id, 0) + 1
                offspring_seq.append(job_id)

        offspring_machine_assign: List[List[int]] = []
        # Now we're gonna do the crossover for the machines by picking the machines randomly either from the first or second parent
        for job_id in range(len(machine_assign1)):
            assign1 = machine_assign1[job_id]
            assign2 = machine_assign2[job_id]
            offspring_assign = []
            for operation_index in range(len(assign1)):
                # less we pick from first parent else from the second
                if random.random() < 0.5:
                    offspring_assign.append(assign1[operation_index])
                else:
                    offspring_assign.append(assign2[operation_index])
            offspring_machine_assign.append(offspring_assign)

        return (offspring_seq, offspring_machine_assign)


    # Tournament function for Genetic Algorithm, will pick randomly some contestants and return the best one as a parent
    def tournament(self, population: List[Tuple[List[int], List[List[int]]]], 
                   fitnesses: List[int], tournament_size: int) -> Tuple[List[int], List[List[int]]]:
        
        # Here we get tournaments_size amount of random indexes from the length of the population
        tournament_indices: List[int] = random.sample(range(len(population)), tournament_size)

        # Initially pick the first one
        best_solution_index = tournament_indices[0]

        # Determine the best solution (the one with the best fitness (makespan))
        for ind in tournament_indices:
            if (fitnesses[ind] < fitnesses[best_solution_index]):
                best_solution_index = ind

        # Return the best one as a parent
        return copy.deepcopy(population[best_solution_index])
    
    
    # Here we implement the Iterated Local Search
    def iterated_local_search(self,
                    max_iterations: int = config.iterated_local_search.max_iterations,
                    perturbation_strength: int = config.iterated_local_search.perturbation_strength
                    ) -> Tuple[Tuple[List[int], List[List[int]]], int]:
        
         # Start with a deep copy of the initial solution to avoid modifying the input
        current_solution: Tuple[List[int], List[List[int]]] = self.generate_dispaching_inititial_solution("MWR")
        current_makespan: int = self.compute_makespan(*current_solution)    # Compute the makespan of the current solution
        # Initialize the best solution and its makespan 
        best_solution: Tuple[List[int], List[List[int]]] = copy.deepcopy(current_solution)
        best_makespan: int = current_makespan
        
        
        for i in range(max_iterations):
            # Here we begin the local search
            improved = config.iterated_local_search.improvement_tries # We'll give more chances for improvement
            while improved > 0:
                neighbour = self.generate_neighbor(current_solution)
                neighbour_makespan = self.compute_makespan(*neighbour)
                if neighbour_makespan < current_makespan:
                    current_solution = copy.deepcopy(neighbour)
                    current_makespan = neighbour_makespan
                    if neighbour_makespan < best_makespan:
                        best_solution = copy.deepcopy(neighbour)
                        best_makespan = neighbour_makespan
                    improved = config.iterated_local_search.improvement_tries
                else:
                    improved -= 1

            pertubed_solution: Tuple[List[int], List[List[int]]] = copy.deepcopy(current_solution)
            # Perturbation step
            for i in range(perturbation_strength):
                pertubed_solution = self.generate_neighbor(pertubed_solution)
            pertubed_solution_makespan: int = self.compute_makespan(*pertubed_solution)
            # Check acceptance of perturbation
            if pertubed_solution_makespan < current_makespan:
                current_solution = copy.deepcopy(pertubed_solution)
                current_makespan = pertubed_solution_makespan

        # Apply the best solution and return
        self.compute_makespan(*best_solution)
        return best_solution, best_makespan
    


    def run(self, heuristic: str) -> None:
        # Executes the scheduling process based on the chosen heuristic
        if heuristic == "SA":
            # Use Simulated Annealing to optimize the schedule
            best_solution, best_makespan = self.simulated_annealing()
            return
        elif heuristic == "HC":
            # Generate an initial solution for Hill Climbing
            best_solution, best_makespan = self.hill_climbing()
            return
        elif heuristic == "TS":
            # Generate an initial solution for Tabu Search
            best_solution, best_makespan = self.tabu_search()
            return
        elif heuristic == "GA":
            best_solution, best_makespan = self.genetic_algorithm()
            return
        elif heuristic == "ILS":
            best_solution, best_makespan = self.iterated_local_search()
            return

        # Use dispatching rules for non heuristics
        # Main scheduling loop - continues until all jobs are complete
        while any(not job.is_complete() for job in self.jobs): # do until all the jobs are completed
            # Get all available tasks sorted by the chosen heuristic
            if(heuristic == 'SPT'):
                next_task_tuple: Optional[Tuple[Job, Task]] = self.shortest_processing_time()   # format: (job, task)
            elif heuristic == 'LPT':
                next_task_tuple: Optional[Tuple[Job, Task]] = self.longest_processing_time()    # format: (job, task)
            elif heuristic == 'MWR':
                next_task_tuple: Optional[Tuple[Job, Task]] = self.most_work_remaining()        # format: (job, task)
            elif heuristic == 'LWR':
                next_task_tuple: Optional[Tuple[Job, Task]] = self.least_work_remaining()       # format: (job, task)
            else:
                raise ValueError(f"Unknown heuristic: {heuristic}")
            
            if next_task_tuple:
                job, task = next_task_tuple
                machine: Machine = self.machines[task.machine_id]
                # Schedule the task at the earliest possible time for this machine
                start = machine.schedule[-1][2] if machine.schedule else 0  # Get the end time of the last machine's task or make it 0

                # Now we pick the max between the possible start time of the machine computed above and the possible start time of the job (the possible
                #   start time of the job is more or equal to when it's previous operation ended) because we need to satisfy both conditions
                start = max(start, job.last_ending_time)

                end = start + task.duration                                 # Calculate the end time
                machine.add_to_schedule(job.job_id, start, end)             # Add the task to needed machine's schedule
                task.start_time = start                                     # Update Task's start time
                task.end_time = end                                         # Update Task's end time
                self.global_max = max(self.global_max, end)                 # Update global makespan if needed
                job.last_ending_time = task.end_time                        # Update the job's last ending time
                job.complete_operation()                                    # Mark the task as done
            
    def print_machine_answer(self):
        for machine in self.machines:
            print(f'Machine id: {machine.machine_id}')
            for tuple in machine.schedule:
                print(f'Job: {tuple[0]}: {tuple[0]} -> {tuple[1]}')
            print()

    def print_job_answer(self):
        for job in self.jobs:
            print(f'Job id: {job.job_id}')
            for task_list in job.operations:
                for task in task_list:
                    if(task.start_time != None):
                        print(f'Machine {task.machine_id}:  {task.start_time} -> {task.end_time}')
            print()

    def reset_scheduler(self):
        #reset the timespan
        self.global_max = 0

        #reset machines
        for machine in self.machines:
            machine.schedule.clear()

        #reset the jobs
        for job in self.jobs:
            job.current_operation_index = 0
            job.last_ending_time = 0
            for task_list in job.operations:
                for task in task_list:
                    task.start_time = None
                    task.end_time = None

    def get_makespan(self) -> int:
        return self.global_max
    
    def display_work_remaining_arrays(self):
        for key, value in self.work_remaining.items():
            print(f'Job_Id: {key}')
            print(value)
            print()

    # should be used only after the scheluder completed the processing
    def get_machines_idle(self) -> List[int]:
        idles: List[int] = []
        for machine in self.machines:
            idle_time: int = 0
            # We check if we have tasks
            if len(machine.schedule):
                # Time before the start of the first operation
                idle_time += machine.schedule[0][1]
                # Time in between the operations
                for i in range(1, len(machine.schedule)):
                    idle_time += machine.schedule[i][1] - machine.schedule[i - 1][2]
                # Here we add the time between the end of the last operation and the makespan(end time)
                idle_time += self.get_makespan() - machine.schedule[-1][2]
            # In case there are no operations executed
            else:
                idle_time = self.get_makespan()
            # Add to the list
            idles.append(idle_time)
        return idles 