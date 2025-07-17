import os
import json
import re
from ask_AI import ask_ai

# Define the ecological characteristics scorecard based on NPZ model equations
ECOLOGICAL_CHARACTERISTICS = {
    "nutrient_equation_uptake": {
        "description": "In dN/dt: Nutrient uptake by phytoplankton with Michaelis-Menten kinetics (N/(e+N)) and self-shading (a/(b+c*P))",
        "weight": 1.0
    },
    "nutrient_equation_recycling": {
        "description": "In dN/dt: Nutrient recycling from zooplankton via predation (beta*lambda*P^2/(mu^2+P^2)*Z) and excretion (gamma*q*Z)",
        "weight": 1.0
    },
    "nutrient_equation_mixing": {
        "description": "In dN/dt: Environmental mixing term (k*(N0-N))",
        "weight": 1.0
    },
    "phytoplankton_equation_growth": {
        "description": "In dP/dt: Phytoplankton growth through nutrient uptake (N/(e+N))*(a/(b+c*P))*P",
        "weight": 1.0
    },
    "phytoplankton_equation_loss": {
        "description": "In dP/dt: Phytoplankton losses through mortality (r*P), predation (lambda*P^2/(mu^2+P^2)*Z), and mixing ((s+k)*P)",
        "weight": 1.0
    },
    "zooplankton_equation": {
        "description": "In dZ/dt: Zooplankton growth through predation (alpha*lambda*P^2/(mu^2+P^2)*Z) and mortality (q*Z)",
        "weight": 1.0
    }
}

def read_ground_truth():
    """Read the ground truth NPZ model file."""
    ground_truth_path = "Data/NPZ_example/NPZ_model.py"
    with open(ground_truth_path, 'r') as f:
        return f.read()

def evaluate_model(model_content, ground_truth_content):
    """
    Use LLM to evaluate ecological characteristics of a model against ground truth.
    Returns qualitative description and scores.
    """
    # Construct the prompt
    characteristics_text = "\n".join([
        f"- {name}: {details['description']} (weight: {details['weight']})"
        for name, details in ECOLOGICAL_CHARACTERISTICS.items()
    ])
    
    prompt = f"""Compare this C++ model against the following criteria that should be present in the NPZ model equation by equation.
The mathematical structure should be identical, even if variable names differ.

For each equation (dN/dt, dP/dt, dZ/dt), check these components:
{characteristics_text}


Model to Evaluate:
```cpp
{model_content}
```

For each characteristic:
1. Score 1.0 if the mathematical structure is equivalent, regardless of:
   - Variable names (e.g., 'N' vs 'nutrients' vs 'N[0]')
   - Programming syntax (e.g., 'pow(x,2)' vs 'x^2')
   - Code organization
2. Verify the mathematical operations are in the same order
3. Confirm all coefficients are present (even if named differently)

Provide your response in JSON format:
{{
    "qualitative_description": "Overall description of how well the model matches the criteria",
    "characteristic_scores": {{
        "characteristic_name": {{
            "score": 0.0 to 1.0,
            "explanation": "How it is implemented"
        }}
    }}
}}
"""
    
    # Get LLM response
    response = ask_ai(prompt, ai_model='claude')
    
    try:
        # Extract JSON content between triple backticks if present
        json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # If no backticks, try to find content between curly braces
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                print("Error: Could not find JSON content in response")
                return None
        
        # Clean up the string
        json_str = json_str.strip()
        
        # Remove line breaks and extra whitespace first
        json_str = re.sub(r'\s+', ' ', json_str)
        
        # Fix JSON formatting issues
        json_str = re.sub(r'},+', '},', json_str)  # Remove multiple commas
        json_str = re.sub(r'"},+', '"},', json_str)  # Remove multiple commas after quotes
        json_str = re.sub(r'(\d+|\d+\.\d+)\s*"', r'\1,"', json_str)  # Add comma after numbers
        json_str = re.sub(r'",\s*}', '"}', json_str)  # Remove trailing comma before closing brace
        json_str = re.sub(r'},\s*}', '}}', json_str)  # Remove trailing comma in nested object
        
        try:
            evaluation = json.loads(json_str)
            return evaluation
        except json.JSONDecodeError as e:
            print(f"Error parsing cleaned JSON: {str(e)}")
            print("Cleaned JSON string:")
            print(json_str)
            return None
    except (json.JSONDecodeError, AttributeError) as e:
        print(f"Error parsing JSON: {str(e)}")
        return None

def calculate_total_score(characteristic_scores):
    """Calculate weighted total score from individual characteristic scores."""
    total_score = 0.0
    for char_name, details in characteristic_scores.items():
        if char_name in ECOLOGICAL_CHARACTERISTICS:
            weight = ECOLOGICAL_CHARACTERISTICS[char_name]["weight"]
            score = details["score"]
            total_score += weight * score
    return total_score

def update_individual_metadata(individual_dir, evaluation):
    """Update the individual's metadata.json with ecological assessment."""
    metadata_path = os.path.join(individual_dir, "metadata.json")
    population_dir = os.path.dirname(individual_dir)
    individual_name = os.path.basename(individual_dir)
    
    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        metadata = {}
        
    # Get generation number and objective value from population metadata
    try:
        with open(os.path.join(population_dir, "population_metadata.json"), 'r') as f:
            pop_metadata = json.load(f)
            # Get generation number
            if "generations" in pop_metadata:
                generation = len(pop_metadata["generations"])
                metadata["generation"] = generation
            
            # Get objective value from current_best_performers
            if "current_best_performers" in pop_metadata:
                for performer in pop_metadata["current_best_performers"]:
                    if performer["individual"] == individual_name:
                        objective_value = performer["objective_value"]
                        break
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        generation = None
        objective_value = None
    
    # Add ecological assessment
    metadata["ecological_assessment"] = {
        "qualitative_description": evaluation["qualitative_description"],
        "characteristic_scores": evaluation["characteristic_scores"],
        "total_score": calculate_total_score(evaluation["characteristic_scores"]),
        "characteristics_present": [
            char_name for char_name, details in evaluation["characteristic_scores"].items()
            if details["score"] > 0
        ]
    }
    
    # Add objective value if available
    if objective_value is not None:
        metadata["ecological_assessment"]["objective_value"] = objective_value
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)

def update_population_metadata(population_dir, generation):
    """Update the population's metadata with ecological scores."""
    metadata_path = os.path.join(population_dir, "population_metadata.json")
    
    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        metadata = {}
    
    # Update ecological scores for current best performers
    if "current_best_performers" in metadata:
        updated = False
        for performer in metadata["current_best_performers"]:
            individual_dir = os.path.join(population_dir, performer["individual"])
            try:
                with open(os.path.join(individual_dir, "metadata.json"), 'r') as f:
                    ind_metadata = json.load(f)
                    if "ecological_assessment" in ind_metadata:
                        performer["ecological_score"] = ind_metadata["ecological_assessment"]["total_score"]
                        updated = True
            except (FileNotFoundError, json.JSONDecodeError):
                continue
        
        # Only write back if we actually updated any scores
        if updated:
            print(f"Updating ecological scores in population metadata for generation {generation}")
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)

def evaluate_individual(individual_dir):
    """Evaluate ecological characteristics of an individual model."""
    metadata_path = os.path.join(individual_dir, "metadata.json")
    
    # Check if individual has already been evaluated
    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            if "ecological_assessment" in metadata:
                print(f"Individual in {individual_dir} has already been evaluated, skipping...")
                return metadata["ecological_assessment"]
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    
    # If not evaluated, proceed with evaluation
    model_path = os.path.join(individual_dir, "model.cpp")
    try:
        with open(model_path, 'r') as f:
            model_content = f.read()
    except FileNotFoundError:
        print(f"Error: Could not find model.cpp in {individual_dir}")
        return
    
    # Read ground truth
    ground_truth_content = read_ground_truth()
    
    # Evaluate the model
    evaluation = evaluate_model(model_content, ground_truth_content)
    if evaluation:
        # Update metadata files
        update_individual_metadata(individual_dir, evaluation)
        
        # Extract generation number from path and update population metadata if in a population directory
        population_dir = os.path.dirname(individual_dir)
        try:
            if "POPULATION_" in os.path.basename(population_dir):
                generation = int(os.path.basename(population_dir).split('_')[1])
                update_population_metadata(population_dir, generation)
        except (ValueError, IndexError):
            print(f"Note: Not updating population metadata for {individual_dir} (not in a population directory)")
        
        return evaluation
    return None

if __name__ == "__main__":
    # Example usage
    individual_dir = "POPULATIONS/POPULATION_0001/INDIVIDUAL_ABC123"
    evaluation = evaluate_individual(individual_dir)
    if evaluation:
        print("Evaluation complete. Check metadata files for results.")
