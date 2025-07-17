import os
import json
import random
import numpy as np

def dummy_process_individual(individual_dir, project_topic, data_file, template_file, temperature=0.1, max_sub_iterations=5, parents=None):
    os.makedirs(individual_dir, exist_ok=True)

    model_file = os.path.join(individual_dir, 'model.cpp')
    parm_file = os.path.join(individual_dir, 'parameters.json')
    metadata_file = os.path.join(individual_dir, 'metadata.json')
    
    # Simulate model creation or improvement
    if not os.path.exists(model_file):
        # Simulate creating a new model
        with open(model_file, 'w') as f:
            f.write("// Dummy model content\n")
        
        # Simulate creating parameters
        parameters = {
            "parameters": [
                {
                    "parameter": "growth_rate",
                    "value": random.uniform(0.1, 1.0),
                    "description": "Dummy growth rate (year^-1)",
                    "source": "dummy_source",
                    "import_type": "PARAMETER",
                    "priority": 1
                }
            ]
        }
        with open(parm_file, 'w') as f:
            json.dump(parameters, f, indent=2)
        
        # Create metadata with lineage information
        metadata = {
            "parents": parents if parents else [],
            "lineage": [],
            "objective_values": [],
            "latest_objective_value": None
        }
        if parents:
            for parent in parents:
                parent_metadata_file = os.path.join(os.path.dirname(individual_dir), parent, 'metadata.json')
                if os.path.exists(parent_metadata_file):
                    with open(parent_metadata_file, 'r') as f:
                        parent_metadata = json.load(f)
                    metadata["lineage"].extend(parent_metadata.get("lineage", []))
                    metadata["lineage"].append(parent)
        
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Initialized new individual: {individual_dir}")
    else:
        # Simulate improving existing model
        with open(model_file, 'a') as f:
            f.write(f"// Improved model content {random.randint(1, 100)}\n")
        
        # Simulate updating parameters
        with open(parm_file, 'r+') as f:
            parameters = json.load(f)
            parameters['parameters'][0]['value'] = random.uniform(0.1, 1.0)
            f.seek(0)
            json.dump(parameters, f, indent=2)
            f.truncate()
        
        print(f"Improved existing model in individual: {individual_dir}")

    # Simulate run_model
    for sub_iteration in range(max_sub_iterations):
        run_status, error_info = dummy_run_model(individual_dir)
        
        if isinstance(run_status, (int, float)):
            objective_value = float(run_status)
            print(f"Model run successful. Objective value: {objective_value}")
            
            # Update metadata with new objective value
            with open(metadata_file, 'r+') as f:
                metadata = json.load(f)
                metadata["objective_values"].append(objective_value)
                metadata["latest_objective_value"] = objective_value
                f.seek(0)
                json.dump(metadata, f, indent=2)
                f.truncate()
            
            return "SUCCESS", objective_value
        
        print(f"Fixing broken model iteration: {sub_iteration}")
        if sub_iteration == max_sub_iterations - 1:
            break
        
        # Simulate model improvement attempts
        with open(model_file, 'a') as f:
            f.write(f"// Attempt to fix model: iteration {sub_iteration}\n")

    print(f"Maximum sub-iterations reached for {individual_dir}. The model could not be successfully run after {max_sub_iterations} attempts.")
    
    # Create final broken report
    report_file = os.path.join(individual_dir, 'model_report.json')
    if os.path.exists(report_file):
        with open(report_file, 'r') as f:
            model_report = json.load(f)
    else:
        model_report = {"iterations": {}}
    
    iteration = str(len(model_report["iterations"]) + 1)
    model_report["iterations"][iteration] = {
        "status": "ERROR",
        "message": "Maximum iterations reached without success",
        "stdout": "Model failed to converge after multiple attempts",
        "stderr": "Maximum iterations exceeded"
    }
    
    update_dummy_model_report(individual_dir, model_report)
    return "BROKEN", None

def dummy_run_model(individual_dir):
    # Initialize or load existing model report
    report_file = os.path.join(individual_dir, 'model_report.json')
    if os.path.exists(report_file):
        with open(report_file, 'r') as f:
            model_report = json.load(f)
    else:
        model_report = {"iterations": {}}

    # Get next iteration number
    iteration = str(len(model_report["iterations"]) + 1)
    
    # Simulate various model outcomes
    outcome = random.choices(
        ["compile_error", "runtime_error", "na_objective", "partial_success", "full_success", "missing_data"],
        weights=[20, 20, 15, 15, 20, 10]
    )[0]
    
    if outcome == "compile_error":
        model_report["iterations"][iteration] = {
            "status": "ERROR",
            "message": "Model failed to compile.",
            "stdout": "Compilation error in dummy model",
            "stderr": "g++ compilation error simulation"
        }
        update_dummy_model_report(individual_dir, model_report)
        return "FAILED", "Compilation error: Random failure in dummy model"
    
    elif outcome == "runtime_error":
        model_report["iterations"][iteration] = {
            "status": "ERROR",
            "message": "Model failed with numerical instability",
            "stdout": "Runtime error in dummy model",
            "stderr": "Numerical instability simulation"
        }
        update_dummy_model_report(individual_dir, model_report)
        return "COMPILED with Numerical Instability", "Numerical instability in dummy model execution"
    
    elif outcome == "na_objective":
        model_report["iterations"][iteration] = {
            "status": "SUCCESS",
            "objective_value": None,
            "model_summary": [
                "            Length Class  Mode     ",
                "par         10     -none- numeric  ",
                "objective    1     -none- numeric  ",
                "convergence  1     -none- numeric  ",
                "iterations   1     -none- numeric  ",
                "evaluations  2     -none- numeric  ",
                "message      1     -none- character"
            ],
            "model_report": {
                "cots_pred": None,
                "fast_pred": None,
                "slow_pred": None
            }
        }
        update_dummy_model_report(individual_dir, model_report)
        return None, "Model ran but produced NA objective value"
    
    elif outcome == "partial_success":
        # Generate a random objective value
        objective_value = random.uniform(30, 100)
        
        # Create partial data with some missing or problematic values
        years = list(range(1980, 2006))
        n_years = len(years)
        
        cots_pred = [None if random.random() < 0.1 else float(max(0.2, random.gauss(1, 0.5))) for _ in range(n_years)]
        fast_pred = [3.9787e-314 if random.random() < 0.1 else float(max(1, random.gauss(8, 2))) for _ in range(n_years)]
        slow_pred = [float('nan') if random.random() < 0.1 else float(max(2, random.gauss(15, 5))) for _ in range(n_years)]
        
        model_report["iterations"][iteration] = {
            "status": "SUCCESS",
            "objective_value": objective_value,
            "model_summary": [
                "            Length Class  Mode     ",
                "par         10     -none- numeric  ",
                "objective    1     -none- numeric  ",
                "convergence  1     -none- numeric  ",
                "iterations   1     -none- numeric  ",
                "evaluations  2     -none- numeric  ",
                "message      1     -none- character"
            ],
            "model_report": {
                "cots_pred": cots_pred,
                "fast_pred": fast_pred,
                "slow_pred": slow_pred
            }
        }
        update_dummy_model_report(individual_dir, model_report)
        return objective_value, None
    
    elif outcome == "missing_data":
        # Generate a random objective value
        objective_value = random.uniform(30, 100)
        
        model_report["iterations"][iteration] = {
            "status": "SUCCESS",
            "objective_value": objective_value,
            "model_summary": [
                "            Length Class  Mode     ",
                "par         10     -none- numeric  ",
                "objective    1     -none- numeric  ",
                "convergence  1     -none- numeric  ",
                "iterations   1     -none- numeric  ",
                "evaluations  2     -none- numeric  ",
                "message      1     -none- character"
            ]
            # Intentionally missing model_report and plot_data
        }
        update_dummy_model_report(individual_dir, model_report)
        return objective_value, None
    
    else:  # full_success
        # Generate a random objective value
        objective_value = random.uniform(30, 100)
    
    # Create dummy time series data (1980-2005)
    years = list(range(1980, 2006))
    n_years = len(years)
    
    # Generate realistic-looking predictions using native Python random
    cots_pred = [float(min(3, max(0.2, abs(random.gauss(1, 0.5))))) for _ in range(n_years)]
    fast_pred = [float(min(18, max(1, abs(random.gauss(8, 2))))) for _ in range(n_years)]
    slow_pred = [float(min(45, max(2, abs(random.gauss(15, 5))))) for _ in range(n_years)]
    
    # Create observed data with some correlation to predictions
    cots_obs = [float(max(0.2, p + random.gauss(0, 0.2))) for p in cots_pred]
    fast_obs = [float(max(1, p + random.gauss(0, 2))) for p in fast_pred]
    slow_obs = [float(max(2, p + random.gauss(0, 5))) for p in slow_pred]
    
    # Create plot data structure
    plot_data = {
        "cots_pred": {
            "Year": years,
            "Modeled": cots_pred,
            "Observed": cots_obs
        },
        "fast_pred": {
            "Year": years,
            "Modeled": fast_pred,
            "Observed": fast_obs
        },
        "slow_pred": {
            "Year": years,
            "Modeled": slow_pred,
            "Observed": slow_obs
        }
    }
    
    # Create iteration report
    model_report["iterations"][iteration] = {
        "status": "SUCCESS",
        "objective_value": objective_value,
        "model_summary": [
            "            Length Class  Mode     ",
            "par         10     -none- numeric  ",
            "objective    1     -none- numeric  ",
            "convergence  1     -none- numeric  ",
            "iterations   1     -none- numeric  ",
            "evaluations  2     -none- numeric  ",
            "message      1     -none- character"
        ],
        "model_report": {
            "cots_pred": cots_pred,
            "fast_pred": fast_pred,
            "slow_pred": slow_pred
        },
        "plot_data": plot_data
    }
    
    update_dummy_model_report(individual_dir, model_report)
    return objective_value, None

def update_dummy_model_report(individual_dir, model_report):
    report_file = os.path.join(individual_dir, 'model_report.json')
    with open(report_file, 'w') as f:
        json.dump(model_report, f, indent=2)

if __name__ == "__main__":
    # Test the dummy process_individual function
    test_dir = "test_individual"
    result, error = dummy_process_individual(test_dir, "Test project", "dummy_data.csv", "dummy_template.cpp", parents=["INDIVIDUAL_PARENT1", "INDIVIDUAL_PARENT2"])
    print(f"Test result: {result}")
    if error:
        print(f"Test error: {error}")
