import os
import shutil
import json
import logging
import numpy as np

# Set up logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def copy_individual(src_population_dir, src_individual, dest_population_dir, dest_individual):
    src_path = os.path.join(src_population_dir, src_individual)
    dest_path = os.path.join(dest_population_dir, dest_individual)
    try:
        os.makedirs(dest_path, exist_ok=True)
        for file in ['metadata.json', 'model.cpp', 'parameters.json']:
            src_file = os.path.join(src_path, file)
            dest_file = os.path.join(dest_path, file)
            if os.path.exists(src_file):
                shutil.copy2(src_file, dest_file)
        
        # Update metadata for the new individual
        metadata = load_metadata(os.path.join(dest_path, 'metadata.json'))
        if metadata:
            metadata['parents'] = [src_individual]
            metadata['lineage'] = metadata.get('lineage', []) + [src_individual]
            save_metadata(os.path.join(dest_path, 'metadata.json'), metadata)
        
        logging.info(f"Successfully copied individual from {src_path} to {dest_path}")
    except Exception as e:
        logging.error(f"Error copying from {src_path} to {dest_path}: {str(e)}")

def move_individual(population_dir, individual, destination):
    src_path = os.path.join(population_dir, individual)
    dest_path = os.path.join(population_dir, destination, individual)
    try:
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        shutil.move(src_path, dest_path)
        logging.info(f"Successfully moved individual from {src_path} to {dest_path}")
    except Exception as e:
        logging.error(f"Error moving individual from {src_path} to {dest_path}: {str(e)}")

def convert_numpy_types(obj):
    """Convert numpy types to native Python types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    return obj

def save_metadata(filepath, metadata):
    try:
        # Convert any numpy types to native Python types
        converted_metadata = convert_numpy_types(metadata)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(converted_metadata, f, indent=2)
        logging.info(f"Successfully saved metadata to {filepath}")
    except Exception as e:
        logging.error(f"Error saving metadata to {filepath}: {str(e)}")

def load_metadata(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        logging.info(f"Successfully loaded metadata from {filepath}")
        return metadata
    except Exception as e:
        logging.error(f"Error loading metadata from {filepath}: {str(e)}")
        return None

def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        logging.info(f"Created directory: {directory}")

def list_individuals(population_dir, subdirectory=None):
    if subdirectory:
        target_dir = os.path.join(population_dir, subdirectory)
    else:
        target_dir = population_dir
    
    try:
        individuals = [d for d in os.listdir(target_dir) if os.path.isdir(os.path.join(target_dir, d)) and d.startswith("INDIVIDUAL_")]
        return individuals
    except Exception as e:
        logging.error(f"Error listing individuals in {target_dir}: {str(e)}")
        return []
