import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from typing import Dict, List

from scheduler import Scheduler


def run_gannt_chart(heuristic: str, scheduler: Scheduler, heuristic_names: Dict):
    scheduler.run(heuristic)
    print(f"\nResults for {heuristic}:")
    scheduler.print_job_answer()
    print(f'The total time is: {scheduler.get_makespan()}')

    ##### Here the plotting part starts #####

    # Initialize an empty list to collect task data, will contain a list of dictionaires
    task_data = []

    # Collect task data into the list of dictionaires
    for job in scheduler.jobs:
        for task_list in job.operations:
            for task in task_list:
                if task.start_time is not None:  # Check for valid start time
                    # Add a new dictionary into the list
                    task_data.append({
                        'Task': 'Job ' + str(job.job_id),
                        'Start': task.start_time,
                        'Finish': task.end_time,
                        'Machine': 'Machine' + str(task.machine_id)
                    })

    # Create DataFrame from the list of dictionaries, the pd DataFrame table
    df = pd.DataFrame(task_data)

    # Calculate duration, adds a new 'Duration' column computed from Finish - Start from every column 
    df['Duration'] = df['Finish'] - df['Start']

    # Extract machine IDs for sorting
    df['MachineID'] = df['Machine'].str.extract('(\d+)').astype(int)
    
    # Get ordered list of machine names (high to low)
    machine_order = [f"Machine{i}" for i in range(scheduler.machines[-1].machine_id, -1, -1)]
    
    # Create Gantt-like chart with numerical ranges, this is the configuration for horizontal bars
    fig = px.bar(df, y="Machine", x="Duration", base="Start", orientation='h', color="Task", 
                custom_data=['Task', 'Start', 'Finish', 'Duration'], 
                title=f"Gantt Chart for {heuristic_names[heuristic]}",
                category_orders={"Machine": machine_order})  # Set the category order
    # Above we use the custom data to ensure the data is taken directly from DataFrame to solve the Duration bug present in Plotly

    # Here we update the hover info type for the bars
    fig.update_traces(hovertemplate='Task: %{customdata[0]}<br>Machine: %{y}<br>Start: %{customdata[1]}<br>Finish: %{customdata[2]}<br>Duration: %{customdata[3]}')

    #print(df)

    # Customize layout
    fig.update_layout(
        xaxis_title="Time Units",
        yaxis_title="Machines",
        xaxis_range=[0, scheduler.get_makespan()],
        title = f'{heuristic_names[heuristic]}, makespan: {scheduler.get_makespan()}'
    )

    # Show the chart
    fig.show()

    ##### Here the plotting part ends #####
    
    # Here we print the idle(time when they were not working) times for machines
    machines_idle: List[int] = scheduler.get_machines_idle()
    print(f'\nDuring {heuristic}\n')
    for i in range(len(machines_idle)):
        print(f'Machine {i} was idle for {machines_idle[i]} units of time')
    print()

    # Here we reset the scheduler
    scheduler.reset_scheduler()


def save_chart_results(heuristic: str, nr_iterations: int, optimal_result: int, scheduler: Scheduler, heuristic_names: Dict):
    print(f'Plotting with {nr_iterations} iterations for {heuristic_names[heuristic]} running...')
    makespans: List[int] = []
    best_makespan: int = 1000000000 # We pick a billion as maximum bc there's no int limit in python
    for i in range(nr_iterations):
        scheduler.run(heuristic)
        makespans.append(scheduler.get_makespan())
        best_makespan = min(best_makespan, scheduler.get_makespan())
        scheduler.reset_scheduler()

    optimal_result_plot_line: List[int] = [optimal_result] * nr_iterations

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, nr_iterations + 1), makespans, label = f'{heuristic_names[heuristic]}, (Best makespan: {best_makespan})', marker='o')
    plt.plot(range(1, nr_iterations + 1), optimal_result_plot_line, label = f'Optimal value {optimal_result}', linestyle='--', color='red')
    plt.xlabel('Run nr.')
    plt.ylabel('Makespan')
    plt.title(f'Makespan over {nr_iterations} Runs for {heuristic_names[heuristic]}')
    plt.legend()
    plt.grid(True)

    # Save the plot to the results folder
    plt.savefig(f'pythonService/results/{heuristic}_makespan.png')
    plt.close()
    print(f'Plot with {nr_iterations} iterations for {heuristic_names[heuristic]} saved.')