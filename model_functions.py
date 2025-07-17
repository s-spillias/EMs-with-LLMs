import os
import json
from make_model import process_individual

def write_metadata(individual_dir, parent_id, parameters):
    metadata = {
        "parent_id": parent_id,
        "parameters": parameters
    }
    with open(os.path.join(individual_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

def run_make_model(individual_id, project_topic, data_file, template_file, temperature, max_sub_iterations, parent_id=None):
    parent_directory = 'POPULATIONS'
    population_dir = os.path.join(parent_directory, f'POPULATION_{individual_id[:4]}')
    full_individual_dir = os.path.join(population_dir, individual_id)
    os.makedirs(full_individual_dir, exist_ok=True)
    
    parameters = {
        "project_topic": project_topic,
        "data_file": data_file,
        "template_file": template_file,
        "temperature": temperature,
        "max_sub_iterations": max_sub_iterations
    }
    write_metadata(full_individual_dir, parent_id, parameters)
    
    return process_individual(full_individual_dir, project_topic, data_file, template_file, temperature, max_sub_iterations)
