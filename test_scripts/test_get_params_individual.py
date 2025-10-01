#!/usr/bin/env python3
"""
Test script to test the get_params function on a specific individual.
This will help verify that the literature search and citation functionality works correctly.
"""

import sys
import os
import json
import shutil
from pathlib import Path

# Add the current directory to the path so we can import our modules
sys.path.append('.')

from scripts.get_params import get_params

def backup_parameters_file(individual_dir):
    """Create a backup of the original parameters.json file."""
    params_file = os.path.join(individual_dir, 'parameters.json')
    backup_file = os.path.join(individual_dir, 'parameters_backup.json')
    
    if os.path.exists(params_file):
        shutil.copy2(params_file, backup_file)
        print(f"✓ Backed up parameters.json to parameters_backup.json")
        return True
    else:
        print(f"✗ No parameters.json file found in {individual_dir}")
        return False

def restore_parameters_file(individual_dir):
    """Restore the original parameters.json file from backup."""
    params_file = os.path.join(individual_dir, 'parameters.json')
    backup_file = os.path.join(individual_dir, 'parameters_backup.json')
    
    if os.path.exists(backup_file):
        shutil.copy2(backup_file, params_file)
        os.remove(backup_file)
        print(f"✓ Restored original parameters.json from backup")
        return True
    else:
        print(f"✗ No backup file found")
        return False

def modify_parameters_for_testing(individual_dir):
    """Modify some parameters to have 'literature' sources and mark them as unprocessed."""
    params_file = os.path.join(individual_dir, 'parameters.json')
    
    with open(params_file, 'r') as f:
        params_data = json.load(f)
    
    # Find parameters to modify for testing
    modified_count = 0
    for param in params_data['parameters']:
        # Modify a few parameters that currently have literature-related sources
        if param['parameter']:
            param['source'] = 'literature'  # Exact match to trigger search
            param['processed'] = False  # Mark as unprocessed
            # Remove any existing citations to test fresh
            if 'citations' in param:
                del param['citations']
            modified_count += 1
            print(f"✓ Modified {param['parameter']} to trigger literature search")
    
    # Save the modified parameters
    with open(params_file, 'w') as f:
        json.dump(params_data, f, indent=4)
    
    print(f"✓ Modified {modified_count} parameters for testing")
    return modified_count > 0

def analyze_results(individual_dir):
    """Analyze the results after running get_params."""
    params_file = os.path.join(individual_dir, 'parameters_metadata.json')
    
    with open(params_file, 'r') as f:
        params_data = json.load(f)
    
    print("\n" + "="*60)
    print("RESULTS ANALYSIS")
    print("="*60)
    
    literature_params = []
    params_with_citations = []
    params_with_found_values = []
    
    for param in params_data['parameters']:
        source = param.get('source', '')
        if 'literature' in source.lower():
            literature_params.append(param['parameter'])
            
            if param.get('citations'):
                params_with_citations.append(param['parameter'])
                print(f"\n📚 {param['parameter']}:")
                print(f"   Source: {source}")
                print(f"   Citations found: {len(param['citations'])}")
                for i, citation in enumerate(param['citations'][:3], 1):  # Show first 3
                    print(f"   {i}. {citation}")
                if len(param['citations']) > 3:
                    print(f"   ... and {len(param['citations']) - 3} more")
            
            if param.get('found_value') is not None:
                params_with_found_values.append(param['parameter'])
                print(f"   Found value: {param.get('found_value')}")
                if param.get('found_min') is not None:
                    print(f"   Found range: [{param.get('found_min')}, {param.get('found_max')}]")
    
    print(f"\n📊 SUMMARY:")
    print(f"   Parameters with literature sources: {len(literature_params)}")
    print(f"   Parameters with citations: {len(params_with_citations)}")
    print(f"   Parameters with found values: {len(params_with_found_values)}")
    
    if literature_params:
        print(f"   Literature parameters: {', '.join(literature_params)}")
    
    return len(params_with_citations) > 0

def test_get_params_on_individual(individual_dir):
    """Test the get_params function on a specific individual."""
    print(f"Testing get_params on individual: {individual_dir}")
    print("="*60)
    
    # Check if the individual directory exists
    if not os.path.exists(individual_dir):
        print(f"✗ Individual directory does not exist: {individual_dir}")
        return False
    
    # Check required files
    required_files = ['parameters.json', 'model.cpp']
    for file in required_files:
        file_path = os.path.join(individual_dir, file)
        if not os.path.exists(file_path):
            print(f"✗ Required file missing: {file}")
            return False
        print(f"✓ Found {file}")
    
    # Backup original parameters
    if not backup_parameters_file(individual_dir):
        return False
    
    try:
        # Modify parameters for testing
        if not modify_parameters_for_testing(individual_dir):
            print("✗ Failed to modify parameters for testing")
            return False
        
        print(f"\n🔍 Running get_params on {individual_dir}...")
        print("-" * 40)
        
        # Run get_params
        result = get_params(individual_dir)
        
        print("-" * 40)
        print("✓ get_params completed successfully")
        
        # Analyze results
        success = analyze_results(individual_dir)
        
        return success
        
    except Exception as e:
        print(f"✗ Error running get_params: {e}")
        import traceback
        traceback.print_exc()
        return False
    

def main():
    """Main function to run the test."""
    # Default individual to test
    default_individual = "POPULATIONS/POPULATION_0052/INDIVIDUAL_ES1AP0XP"
    
    # Allow command line argument to specify different individual
    if len(sys.argv) > 1:
        individual_dir = sys.argv[1]
    else:
        individual_dir = default_individual
    
    print("🧪 GET_PARAMS FUNCTION TEST")
    print("="*60)
    print(f"Target individual: {individual_dir}")
    print(f"This test will:")
    print(f"  1. Backup the original parameters.json")
    print(f"  2. Modify some parameters to trigger literature searches")
    print(f"  3. Run get_params to test citation collection")
    print(f"  4. Analyze the results")
    print(f"  5. Restore the original parameters.json")
    print()
    
    success = test_get_params_on_individual(individual_dir)
    
    print("\n" + "="*60)
    if success:
        print("🎉 TEST PASSED: Citations were successfully collected!")
    else:
        print("❌ TEST FAILED: No citations were collected or errors occurred")
    print("="*60)
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())