#!/usr/bin/env python3
"""
Test script to debug the get_params function and check citation handling.
"""

import sys
import os
import json
from scripts.get_params import get_params

def test_get_params():
    """Test the get_params function on a specific individual."""
    
    # Test directory
    test_dir = "POPULATIONS/POPULATION_0050/INDIVIDUAL_8FJ6JTRF"
    
    if not os.path.exists(test_dir):
        print(f"Error: Test directory {test_dir} does not exist")
        return False
    
    print(f"Testing get_params on: {test_dir}")
    print("=" * 50)
    
    # Check if parameters.json exists
    params_file = os.path.join(test_dir, "parameters.json")
    if not os.path.exists(params_file):
        print(f"Error: parameters.json not found in {test_dir}")
        return False
    
    # Read original parameters to see current state
    print("Reading original parameters.json...")
    with open(params_file, 'r') as f:
        original_params = json.load(f)
    
    # Count parameters by source and processing status
    literature_params = []
    processed_params = []
    unprocessed_literature = []
    
    for param in original_params.get("parameters", []):
        if param.get("source") == "literature":
            literature_params.append(param)
            if param.get("processed", False):
                processed_params.append(param)
            else:
                unprocessed_literature.append(param)
    
    print(f"Total parameters: {len(original_params.get('parameters', []))}")
    print(f"Literature parameters: {len(literature_params)}")
    print(f"Already processed: {len(processed_params)}")
    print(f"Unprocessed literature parameters: {len(unprocessed_literature)}")
    
    # Show citation status for literature parameters
    citations_empty = 0
    citations_populated = 0
    
    for param in literature_params:
        citations = param.get("citations", [])
        if not citations or citations == []:
            citations_empty += 1
        else:
            citations_populated += 1
    
    print(f"Literature parameters with empty citations: {citations_empty}")
    print(f"Literature parameters with citations: {citations_populated}")
    
    if unprocessed_literature:
        print(f"\nUnprocessed literature parameters:")
        for param in unprocessed_literature[:3]:  # Show first 3
            print(f"  - {param.get('parameter', 'Unknown')}: {param.get('description', 'No description')}")
        if len(unprocessed_literature) > 3:
            print(f"  ... and {len(unprocessed_literature) - 3} more")
    
    # Check population metadata
    population_dir = os.path.dirname(test_dir)
    metadata_file = os.path.join(population_dir, "population_metadata.json")
    
    if os.path.exists(metadata_file):
        print(f"\nReading population metadata...")
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        print(f"RAG choice: {metadata.get('rag_choice', 'Not set')}")
        print(f"Doc store dir: {metadata.get('doc_store_dir', 'Not set')}")
    else:
        print(f"\nWarning: No population_metadata.json found in {population_dir}")
    
    # Now run get_params if there are unprocessed literature parameters
    if unprocessed_literature:
        print(f"\n" + "=" * 50)
        print("Running get_params function...")
        print("=" * 50)
        
        try:
            result = get_params(test_dir)
            print(f"get_params completed successfully!")
            print(f"Processed {len(result)} parameters")
            
            # Check results
            literature_with_citations = 0
            for param in result:
                if param.get("source") == "literature":
                    citations = param.get("citations", [])
                    if citations and citations != []:
                        literature_with_citations += 1
                        print(f"Parameter '{param.get('parameter')}' has {len(citations)} citations")
            
            print(f"\nAfter processing:")
            print(f"Literature parameters with citations: {literature_with_citations}")
            
            return True
            
        except Exception as e:
            print(f"Error running get_params: {e}")
            import traceback
            traceback.print_exc()
            return False
    else:
        print(f"\nNo unprocessed literature parameters found. Skipping get_params execution.")
        return True

if __name__ == "__main__":
    success = test_get_params()
    if success:
        print("\nTest completed successfully!")
    else:
        print("\nTest failed!")
        sys.exit(1)