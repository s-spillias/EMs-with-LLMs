import os
import logging
from contextlib import redirect_stdout, redirect_stderr
from scripts.model_report_handler import read_model_report
from scripts.individual_utils import update_individual_metadata, create_individual_metadata
# from dummy_run_model import dummy_process_individual
from scripts.make_model import process_individual
# Set up logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def evaluate_individual(population_dir, individual_id):
    individual_dir = os.path.join(population_dir, individual_id)
    if not os.path.exists(individual_dir):
        logging.warning(f"Individual directory {individual_dir} does not exist")
        return float('inf')
    
    report = read_model_report(individual_dir)
    if report:        
        if report.get('status') == 'SUCCESS':
            objective_value = report.get('objective_value')
            if objective_value is not None:
                objective_value = float(objective_value)
            else:
                objective_value = float('inf')
            logging.info(f"Individual {individual_id}: Objective value: {objective_value}")
            return objective_value
    
    logging.warning(f"No valid successful result found for {individual_id}")
    return float('inf')

def evaluate_population(population_dir, individuals):
    evaluations = []
    broken_individuals = []
    for individual in individuals:
        try:
            obj_value = evaluate_individual(population_dir, individual)
            if obj_value != float('inf'):
                evaluations.append((individual, obj_value))
                update_individual_metadata(os.path.join(population_dir, individual), obj_value)
            else:
                broken_individuals.append(individual)
            
            logging.info(f"Individual {individual} status: {'Evaluated' if obj_value != float('inf') else 'Broken'}")
        except Exception as e:
            logging.error(f"Error processing individual {individual}: {str(e)}")
            broken_individuals.append(individual)
    
    return evaluations, broken_individuals

def run_make_model_with_file_output(population_dir, individual_id, project_topic, response_file, forcing_file=None, report_file=None, template_file=None, temperature=0.1, max_sub_iterations=5, llm_choice='anthropic_sonnet', train_test_split=1.0, parent=None):
    individual_dir = os.path.join(population_dir, individual_id)
    os.makedirs(individual_dir, exist_ok=True)
    create_individual_metadata(individual_dir, parent=parent)
    output_file = os.path.join(individual_dir, 'make_model_output.txt')
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            with redirect_stdout(f), redirect_stderr(f):
                #                 # First, process the individual
                status, objective_value = process_individual(individual_dir, project_topic, response_file, forcing_file, report_file, template_file, temperature, max_sub_iterations, llm_choice, train_test_split)
                # # Use dummy process instead of real process
                # status, objective_value = dummy_process_individual(
                #     individual_dir, 
                #     project_topic, 
                #     data_file, 
                #     template_file, 
                #     temperature=temperature, 
                #     max_sub_iterations=max_sub_iterations, 
                #     parents=[parent] if parent else None
                # )
        
        logging.info(f"\033[93mModel processing completed for {individual_id}. Status: {status}, Objective value: {objective_value}\033[0m")
        return status, objective_value
    except Exception as e:
        logging.error(f"Error in run_make_model_with_file_output for {individual_id}: {str(e)}")
        return "ERROR", None

# You may want to add a function to clear any caches if you implement caching later
def clear_evaluation_cache():
    pass  # Implement if caching is added
