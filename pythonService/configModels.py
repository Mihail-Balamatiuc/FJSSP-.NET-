# Define classes to provide type hints for the config structure
class GlobalConfigs:
    operation_machine_ratio: float

class SimulatedAnnealingConfig:
    initial_temperature: float
    cooling_rate: float
    min_temperature: float
    max_iterations: int
    restarts: int

class HillClimbingConfig:
    max_iterations: int
    improvement_tries: int
    restarts: int
    neighbors_number: int

class TabuSearchConfig:
    tabu_tenure: int
    max_iterations: int

class GeneticAlgorithmConfig:
    population_size: int
    num_generations: int
    crossover_rate: float
    mutation_rate: float
    tournament_size: int

class IteratedLocalSearch:
    max_iterations: int
    perturbation_strength: int
    improvement_tries: int

class Config:
    simulated_annealing: SimulatedAnnealingConfig
    hill_climbing: HillClimbingConfig
    tabu_search: TabuSearchConfig
    genetic_algorithm: GeneticAlgorithmConfig
    iterated_local_search: IteratedLocalSearch
    global_configs: GlobalConfigs