#!/usr/bin/env python3
"""
Test script to test a single parameter processing with detailed debugging.
"""

import sys
import os
import json
from scripts.get_params import process_parameter

def test_single_parameter():
    """Test processing a single parameter to debug citation extraction."""
    
    # Test parameters
    population_dir = "POPULATIONS/POPULATION_0050"
    test_dir = "POPULATIONS/POPULATION_0050/INDIVIDUAL_8FJ6JTRF"
    doc_store = "doc_store"
    
    # Read model.cpp for context
    cpp_file = os.path.join(test_dir, 'model.cpp')
    if not os.path.exists(cpp_file):
        print(f"Error: model.cpp not found in {test_dir}")
        return False
    
    with open(cpp_file, 'r') as f:
        cpp_content = f.read()
    
    # Test parameter details
    parameter = "log_A_Allee"
    description = "ln Allee half-saturation density for reproduction (ind m^-2)"
    enhanced_description = "Minimum population density for successful reproduction of Crown of Thorns"
    import_type = "PARAMETER"
    model_name = "claude-3-5-haiku-20241022"
    
    print(f"Testing single parameter processing:")
    print(f"  Parameter: {parameter}")
    print(f"  Description: {description}")
    print(f"  Enhanced description: {enhanced_description}")
    print(f"  Model: {model_name}")
    print("=" * 50)
    
    try:
        result = process_parameter(
            parameter=parameter,
            description=description,
            enhanced_semantic_description=enhanced_description,
            import_type=import_type,
            cpp_content=cpp_content,
            population_dir=population_dir,
            doc_store=doc_store,
            model_name=model_name,
            max_retries=1  # Only try once for testing
        )
        
        print("\nResult:")
        print(json.dumps(result, indent=2))
        
        citations = result.get("citations", [])
        print(f"\nFinal citations count: {len(citations)}")
        
        if citations:
            print("SUCCESS: Citations were found!")
            for i, citation in enumerate(citations):
                print(f"  Citation {i+1}: {citation}")
        else:
            print("PROBLEM: No citations found in final result")
            
        return len(citations) > 0
        
    except Exception as e:
        print(f"Error during parameter processing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_single_parameter()
    if success:
        print("\nTest completed successfully - citations found!")
    else:
        print("\nTest failed - no citations found!")
        sys.exit(1)