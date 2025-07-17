import json
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_latest_iteration(report_data):
    """
    Get the latest iteration number from the report data.

    Args:
    report_data (dict): The model report data containing iterations.

    Returns:
    int: The latest iteration number, or 0 if no iterations exist.
    """
    if not report_data or 'iterations' not in report_data:
        return 0
    iterations = [int(i) for i in report_data['iterations'].keys()]
    return max(iterations) if iterations else 0

def read_model_report(individual_dir, iteration=None):
    """
    Read the model report for an individual, optionally for a specific iteration.

    Args:
    individual_dir (str): Path to the individual's directory.
    iteration (int, optional): Specific iteration to read. If None, returns latest iteration.

    Returns:
    dict: The model report data for the specified iteration, or an empty dict if not found.
    """
    report_file = os.path.join(individual_dir, 'model_report.json')
    if os.path.exists(report_file):
        try:
            with open(report_file, 'r', encoding='utf-8') as f:
                report_data = json.load(f)
                
                # If no iterations exist yet, return empty dict
                if 'iterations' not in report_data or not report_data['iterations']:
                    return {}
                
                # If specific iteration requested
                if iteration is not None:
                    iteration_str = str(iteration)
                    return report_data['iterations'].get(iteration_str, {})
                
                # Return latest iteration data
                latest = str(get_latest_iteration(report_data))
                return report_data['iterations'].get(latest, {})
        except json.JSONDecodeError:
            logging.error(f"Error reading {report_file}. File may be empty or contain invalid JSON.")
        except Exception as e:
            logging.error(f"Unexpected error reading {report_file}: {str(e)}")
    else:
        logging.warning(f"Model report file not found: {report_file}")
    return {}

def update_model_report(individual_dir, updates):
    """
    Create a new iteration of the model report for an individual.

    Args:
    individual_dir (str): Path to the individual's directory.
    updates (dict): New data to be added or updated in the report.
    """
    report_file = os.path.join(individual_dir, 'model_report.json')
    
    try:
        # Read existing report or create new structure
        if os.path.exists(report_file):
            with open(report_file, 'r', encoding='utf-8') as f:
                report_data = json.load(f)
        else:
            report_data = {'iterations': {}}
        
        # Initialize iterations if not present
        if 'iterations' not in report_data:
            report_data['iterations'] = {}
        
        # Get next iteration number
        next_iteration = str(get_latest_iteration(report_data) + 1)
        
        # Add new iteration data
        report_data['iterations'][next_iteration] = updates
        
        # Write updated report
        os.makedirs(os.path.dirname(report_file), exist_ok=True)
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2)
        logging.info(f"Successfully created model report iteration {next_iteration}")
    except Exception as e:
        logging.error(f"Error updating model report {report_file}: {str(e)}")

def get_model_status(individual_dir):
    """
    Get the status of the model for an individual from the latest iteration.

    Args:
    individual_dir (str): Path to the individual's directory.

    Returns:
    str: The status of the model, or 'UNKNOWN' if not found.
    """
    report_file = os.path.join(individual_dir, 'model_report.json')
    try:
        with open(report_file, 'r', encoding='utf-8') as f:
            report_data = json.load(f)
            latest = str(get_latest_iteration(report_data))
            if latest and latest in report_data.get('iterations', {}):
                return report_data['iterations'][latest].get('status', 'UNKNOWN')
    except Exception as e:
        logging.error(f"Error reading status from {report_file}: {str(e)}")
    return 'UNKNOWN'

def get_objective_value(individual_dir):
    """
    Get the objective value of the model for an individual from the latest iteration.

    Args:
    individual_dir (str): Path to the individual's directory.

    Returns:
    float or None: The objective value, or None if not found.
    """
    report_file = os.path.join(individual_dir, 'model_report.json')
    try:
        with open(report_file, 'r', encoding='utf-8') as f:
            report_data = json.load(f)
            latest = str(get_latest_iteration(report_data))
            if latest and latest in report_data.get('iterations', {}):
                return report_data['iterations'][latest].get('objective_value')
    except Exception as e:
        logging.error(f"Error reading objective value from {report_file}: {str(e)}")
    return None

def get_lineage(individual_dir):
    """
    Get the lineage information for an individual.

    Args:
    individual_dir (str): Path to the individual's directory.

    Returns:
    list: The lineage of the individual, or an empty list if not found.
    """
    metadata_file = os.path.join(individual_dir, 'metadata.json')
    try:
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        return metadata.get('lineage', [])
    except Exception as e:
        logging.error(f"Error reading lineage from {metadata_file}: {str(e)}")
        return []

def get_parents(individual_dir):
    """
    Get the parent information for an individual.

    Args:
    individual_dir (str): Path to the individual's directory.

    Returns:
    list: The parents of the individual, or an empty list if not found.
    """
    metadata_file = os.path.join(individual_dir, 'metadata.json')
    try:
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        return metadata.get('parents', [])
    except Exception as e:
        logging.error(f"Error reading parents from {metadata_file}: {str(e)}")
        return []
