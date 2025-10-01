#!/usr/bin/env python3
"""
Test script for the enhance_parameter_descriptions function
"""
import sys
import os
sys.path.append('scripts')

from make_model import enhance_parameter_descriptions

def test_enhance_function():
    # Use an existing individual directory
    individual_dir = "POPULATIONS/POPULATION_0050/INDIVIDUAL_8FJ6JTRF"
    project_topic = "Modeling episodic outbreaks of Crown of Thorns starfish on the Great Barrier Reef"
    
    print(f"Testing enhance_parameter_descriptions with:")
    print(f"  Individual dir: {individual_dir}")
    print(f"  Project topic: {project_topic}")
    print()
    
    # Check if the directory exists
    if not os.path.exists(individual_dir):
        print(f"Error: Directory {individual_dir} does not exist")
        return
    
    # Check if required files exist
    required_files = ['parameters.json', 'model.cpp']
    for file in required_files:
        file_path = os.path.join(individual_dir, file)
        if not os.path.exists(file_path):
            print(f"Error: Required file {file_path} does not exist")
            return
        print(f"✓ Found {file}")
    
    # Check population metadata
    population_dir = os.path.dirname(individual_dir)
    metadata_path = os.path.join(population_dir, 'population_metadata.json')
    if not os.path.exists(metadata_path):
        print(f"Error: Population metadata {metadata_path} does not exist")
        return
    print(f"✓ Found population metadata")
    
    # Read and display the rag_choice
    import json
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    rag_choice = metadata.get('rag_choice', 'Not found')
    print(f"✓ rag_choice: {rag_choice}")
    print()
    
    # Run the enhance function
    print("Running enhance_parameter_descriptions...")
    try:
        enhance_parameter_descriptions(individual_dir, project_topic)
        print("✓ Function completed successfully!")
    except Exception as e:
        print(f"✗ Error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_enhance_function()