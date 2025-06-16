# Task class represents an individual operation that belongs to a job
from typing import List, Optional, Tuple


class Task:
    def __init__(self, machine_id: int, duration: int):
        self.machine_id: int = machine_id  # ID of the machine required to process this task
        self.duration: int = duration      # Time required to process this task
        self.start_time: int = None        # When the task starts (set during scheduling)
        self.end_time: int = None          # When the task ends (set during scheduling)

    def display_task(self) -> None:
        print(f'[{self.machine_id}, {self.duration}]', end = '')

# Machine class represents a single machine that can process tasks sequentially
class Machine:
    def __init__(self, machine_id: int):
        self.machine_id: int = machine_id                           # Unique ID for the machine
        self.schedule: List[Tuple[int, int, int]] = []              # List of tuples (job_id, start_time, end_time)

    def add_to_schedule(self,  job_id: int, start_time: int, end_time: int) -> None:
        # Adds a task to the machine's schedule
        self.schedule.append((job_id, start_time, end_time))
    

# Job class represents a sequence of tasks that must be completed in order
class Job:
    def __init__(self, job_id: int, operations: List[List[Task]]):
        self.job_id: int = job_id                       # Unique ID for the job
        self.operations: List[List[Task]] = operations            # List of Task objects for this job, we call it operations
        self.current_operation_index: int = 0           # Tracks progress of the job (which task is next)
        self.last_ending_time = 0                       # Will contain the last ending time of this job (ending time of the last executed operation)

    def get_next_task_list(self) -> Optional[List[Task]]:
        # Returns the next task to be scheduled, or None if the job is complete
        if self.current_operation_index < len(self.operations):
            return self.operations[self.current_operation_index] # returns a task_list with machines and durations
        return None

    def complete_operation(self) -> None:
        # Moves to the next operation in the job sequence
        self.current_operation_index += 1

    def is_complete(self) -> bool:
        # Checks if all tasks in the job have been completed
        return self.current_operation_index >= len(self.operations)