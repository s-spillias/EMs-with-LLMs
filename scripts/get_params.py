import json
import os
import logging
import pandas as pd
import time
from scripts.search import search_engine, rag_query
from scripts.ask_AI import ask_ai

# Set up logging
# logging.basicConfig(level=print, format='%(asctime)s - %(levelname)s - %(message)s')

def extract_values_from_text(search_results, parameter, description):
    """Use LLM to extract values from search results."""
    prompt = f"""
    Based on the following search results, extract numerical values for:
    {description}
    
    Search Results:
    {search_results}
    
    Please analyze these results and provide a JSON object with 'min', 'max', and 'value' for this parameter.
    If multiple values are found, 'value' should be the average.
    If no relevant values are found, respond with 'No relevant values found'.
    
    Note the units that the model expects. Please convert the values to be the appropriate units for the model. 

    Example response format:
    {{"min": 1.2, "max": 3.4, "value": 2.3}}
    """
    
    try:
        response = ask_ai(prompt,'claude')
        if "No relevant values found" in response:
            return None
            
        # Try to parse the JSON response
        try:
            # Find JSON-like string in the response
            import re
            json_match = re.search(r'\{[^}]+\}', response)
            if json_match:
                values = json.loads(json_match.group(0))
                if all(k in values for k in ['min', 'max', 'value']):
                    # Convert values to float
                    values = {k: float(v) for k, v in values.items()}
                    return values
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Error parsing values: {str(e)}")
            
        return None
    except Exception as e:
        print(f"Error extracting values: {str(e)}")
        return None

def flatten_list(nested_list):
    """Flatten a deeply nested list of strings."""
    flattened = []
    for item in nested_list:
        if isinstance(item, list):
            flattened.extend(flatten_list(item))  # Recursively flatten sublist
        else:
            flattened.append(item)
    return flattened

def process_parameter(parameter: str, description: str, enhanced_semantic_description: str, import_type: str, cpp_content: str, population_dir: str, doc_store: str, max_retries: int = 3):
    """Process a single parameter using RAG with Web Search."""
    
    for attempt in range(max_retries):
        # Use only the description for search to get broader results
        search_query = enhanced_semantic_description
        print(f"Attempt {attempt + 1} for {parameter}: {search_query}")
        all_search_results = []
        all_citations = []
        try:
            search_engines = ["semantic_scholar", "rag", "serper"]
            engine = search_engines[attempt % len(search_engines)]
            
            if engine == "rag":
                search_results, citations = rag_query(search_query, doc_store, population_dir)
            else:
                search_results, citations = search_engine(search_query, engine=engine, directory=doc_store)
            
            if not search_results:
                print(f"No search results found for {parameter} in attempt {attempt + 1}")
                continue
            
            print(f"Search results found for {parameter}")
            all_search_results.append(search_results)
            all_citations.append(citations)
        except Exception as e:
            print(f"Error processing {parameter} in attempt {attempt + 1}: {str(e)}")
            # Add delay if we hit rate limits
            if "429" in str(e):
                time.sleep(5 * (attempt + 1))
        
    # Use LLM to extract values from search results
    flat_search_results = flatten_list(all_search_results)
    flat_citations = flatten_list(all_citations)

    rag_info = "\n".join(flat_search_results)
    print("*"*50 + '\n\n' + rag_info + '\n\n' + "*"*50)
    values = extract_values_from_text(rag_info, parameter, description)
    
    output = {
        "parameter": parameter,
        "citations": flat_citations,
        "source": "literature",
        "processed": True
    }
    
    # Add found values if they exist with 'found_' prefix
    if values:
        output["found_value"] = values["value"]
        output["found_min"] = values["min"]
        output["found_max"] = values["max"]
        print(f"Found values for {parameter}: {values}")
    
    print(f"Successfully processed {parameter}")
    return output
    

def get_params(directory_path):
    """Read parameters directly from parameters.json and process literature parameters."""
    
    # Extract population_dir from directory_path
    population_dir = os.path.dirname(directory_path)
    
    # Set up doc_store directory
    doc_store = 'doc_store'
    
    # Read existing parameters.json
    params_file = os.path.join(directory_path, "parameters.json")
    with open(params_file, 'r') as file:
        params_data = json.load(file)
    
    # Read model.cpp for context
    with open(os.path.join(directory_path, 'model.cpp'), 'r') as file:
        cpp_str = file.read()
    
    # Process parameters from literature
    processed_outputs = []
    for param in params_data["parameters"]:
        # logging.info(f"Processing {param}")
        if param["source"] == "literature" and not param.get("processed", False):
            # logging.info(f"Processing {param} for real...")
            print(f"Processing literature parameter: {param['parameter']}")
            processed_result = process_parameter(
                param["parameter"],
                param["description"],
                param["enhanced_semantic_description"],
                param.get("import_type", "PARAMETER"),
                cpp_str,
                population_dir,
                doc_store
            )
            if processed_result:
                # Update with new fields while preserving original values
                param.update(processed_result)
        else:
            # If already processed or not from literature, mark as processed
            param["processed"] = True
        processed_outputs.append(param)
    
    # Save the updated parameters
    output_file = os.path.join(directory_path, "parameters.json")
    with open(output_file, 'w') as file:
        json.dump({"parameters": processed_outputs}, file, indent=4)
    
    print("Parameter processing completed.")
    return processed_outputs

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        directory_path = sys.argv[1]
    else:
        directory_path = "BRANCH_0/Version_0000"  # Default path
    get_params(directory_path)
