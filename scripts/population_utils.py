import os
import random
import string
import multiprocessing
import logging
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from scripts.individual_utils import create_individual_metadata
from scripts.file_utils import copy_individual, load_metadata
from scripts.evaluation_utils import run_make_model_with_file_output
from scripts.model_report_handler import read_model_report

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def generate_individual_id():
    return f"INDIVIDUAL_{''.join(random.choices(string.ascii_uppercase + string.digits, k=8))}"

def get_next_population_number():
    populations_dir = 'POPULATIONS'
    if not os.path.exists(populations_dir):
        os.makedirs(populations_dir)
        return 1
    existing_populations = [d for d in os.listdir(populations_dir) if d.startswith('POPULATION_')]
    if not existing_populations:
        return 1
    max_number = max(int(pop.split('_')[1]) for pop in existing_populations)
    return max_number + 1


def evolve_population(population_dir, all_individuals, n_best):
    logging.info(f"Evolving population with {len(all_individuals)} individuals")
    
    if not all_individuals:
        logging.warning("Empty population received in evolve_population")
        return [], [], []

    evaluations = []
    for individual in all_individuals:
        try:
            individual_dir = os.path.join(population_dir, individual)
            # Get objective value directly from model report
            report_data = read_model_report(individual_dir)
            if report_data and report_data.get('status') == 'SUCCESS':
                objective_value = report_data.get('objective_value')
                # Convert to float to handle numpy types
                objective_value = float(objective_value) if objective_value is not None else None
            else:
                objective_value = None
            evaluations.append((individual, objective_value))
        except Exception as e:
            logging.error(f"Error evaluating individual {individual}: {str(e)}")
            evaluations.append((individual, None))

    def sort_key(item):
        _, obj_value = item
        # Handle None values
        if obj_value is None:
            return (3, float('inf'))
        # Handle string values (like 'NA')
        if isinstance(obj_value, str):
            return (2, float('inf'))
        # Handle NaN values
        if isinstance(obj_value, float) and math.isnan(obj_value):
            return (2, float('inf'))
        # Valid numeric values
        return (1, float(obj_value))

    sorted_evaluations = sorted(evaluations, key=sort_key)
    best_individuals = []
    culled_individuals = []
    broken_individuals = []

    for individual, obj_value in sorted_evaluations:
        if obj_value is None or isinstance(obj_value, str) or (isinstance(obj_value, float) and math.isnan(obj_value)):
            broken_individuals.append(individual)
        elif len(best_individuals) < n_best:
            best_individuals.append(individual)
        else:
            culled_individuals.append(individual)

    logging.info(f"Evolution results: {len(best_individuals)} best, {len(culled_individuals)} culled, {len(broken_individuals)} broken")
    return best_individuals, culled_individuals, broken_individuals

def spawn_or_initialize_individual(args):
    population_dir, parent, project_topic, response_file, forcing_file, report_file, temperature, max_sub_iterations, llm_choice, train_test_split = args
    individual_id = generate_individual_id()
    
    try:
        if parent:
            # Spawn offspring
            copy_individual(population_dir, parent, population_dir, individual_id)
        
        create_individual_metadata(os.path.join(population_dir, individual_id), parent=parent)
        
        status, objective_value = run_make_model_with_file_output(
            population_dir, individual_id, project_topic, response_file, forcing_file, report_file, temperature, max_sub_iterations, llm_choice, train_test_split
        )
        
        logging.info(f"Created individual {individual_id} with parent {parent} and objective value {objective_value}")
        return individual_id
    except Exception as e:
        logging.error(f"Error in spawn_or_initialize_individual for {individual_id}: {str(e)}")
        return None

def create_new_generation(population_dir, best_individuals, n_individuals, project_topic, response_file, forcing_file, report_file, temperature, max_sub_iterations, llm_choice, train_test_split):
    logging.info(f"Creating new generation with {len(best_individuals)} best individuals and {n_individuals} total individuals")
    # Prepare arguments for parallel processing
    args_list = []
    
    # First, create one child for each parent
    for parent in best_individuals:
        args_list.append((population_dir, parent, project_topic, response_file, forcing_file, report_file, temperature, max_sub_iterations, llm_choice, train_test_split))
    
    # Then, fill the remaining slots with new random individuals
    for _ in range(n_individuals - len(best_individuals)):
        args_list.append((population_dir, None, project_topic, response_file, forcing_file, report_file, temperature, max_sub_iterations, llm_choice, train_test_split))
    
    # Spawn offspring or initialize new individuals in parallel
    new_individuals = []
    with ProcessPoolExecutor(max_workers=n_individuals, initializer=init_worker) as executor:
        future_to_args = {executor.submit(spawn_or_initialize_individual, args): args for args in args_list}
        for future in as_completed(future_to_args):
            args = future_to_args[future]
            try:
                result = future.result()
                if result:
                    new_individuals.append(result)
            except Exception as e:
                logging.error(f"Exception in create_new_generation for args {args}: {str(e)}")

    logging.info(f"Created {len(new_individuals)} new individuals")
    return new_individuals

def init_worker():
    """Initialize worker process with proper encoding settings"""
    os.environ['PYTHONIOENCODING'] = 'utf-8'
