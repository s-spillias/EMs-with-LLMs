import os
import json
import numpy as np

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

def create_individual_metadata(individual_dir, parent=None):
    metadata_file = os.path.join(individual_dir, 'metadata.json')
    metadata = {}
    
    if os.path.exists(metadata_file):
        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            if parent:
                metadata["parent"] = parent
        except Exception as e:
            print(f"Error reading metadata for {individual_dir}: {str(e)}")
            metadata = {}
    
    if not metadata:
        metadata = {
            "parent": parent,
            "lineage": [],
            "objective_value": None
        }
    
    # Update lineage
    if parent:
        parent_metadata_file = os.path.join(os.path.dirname(individual_dir), parent, 'metadata.json')
        if os.path.exists(parent_metadata_file):
            try:
                with open(parent_metadata_file, 'r') as f:
                    parent_metadata = json.load(f)
                metadata["lineage"] = parent_metadata.get("lineage", []) + [parent]
            except Exception as e:
                print(f"Error reading parent metadata for {individual_dir}: {str(e)}")
    
    try:
        os.makedirs(individual_dir, exist_ok=True)
        with open(metadata_file, 'w') as f:
            json.dump(convert_numpy_types(metadata), f, indent=2)
        print(f"Updated metadata for {individual_dir} with parent: {parent}")
    except Exception as e:
        print(f"Error updating metadata for {individual_dir}: {str(e)}")

def update_individual_metadata(individual_dir, objective_value):
    metadata_file = os.path.join(individual_dir, 'metadata.json')
    metadata = {}
    
    if os.path.exists(metadata_file):
        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
        except Exception as e:
            print(f"Error reading metadata for {individual_dir}: {str(e)}")
    
    # Update the objective value
    metadata["objective_value"] = objective_value
    
    try:
        with open(metadata_file, 'w') as f:
            json.dump(convert_numpy_types(metadata), f, indent=2)
    except Exception as e:
        print(f"Error updating metadata for {individual_dir}: {str(e)}")
