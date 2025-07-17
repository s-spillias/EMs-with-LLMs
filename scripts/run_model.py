import os
import subprocess
import json
import matplotlib.pyplot as plt
import pandas as pd
import time
import sys
from scripts.model_report_handler import update_model_report, read_model_report

class TextColors:
    RESET = '\033[0m'
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    PURPLE = '\033[95m'
    CYAN = '\033[96m'

def log_message(message, color=TextColors.RESET):
    print(color + message + TextColors.RESET)

def remove_error_files(individual_dir):
    error_file_path = os.path.join(individual_dir, 'model_error.log')
    if os.path.exists(error_file_path):
        os.remove(error_file_path)
        log_message(f"Removed error file: {error_file_path}", TextColors.YELLOW)

def plot_model_vs_historical(individual_dir, plot_data):
    for var, data in plot_data.items():
        df = pd.DataFrame(data)
        plt.figure(figsize=(10, 6))
        time_col = df.columns[0]  # Get the first column name dynamically
        xlabel = time_col.split(' (')[0]  # Remove the parenthetical part
        plt.plot(df[time_col], df['Observed'], label='Historical', marker='o')
        plt.plot(df[time_col], df['Modeled'], label='Modeled', marker='x')
        plt.title(f'{var}: Modeled vs Historical')
        plt.xlabel(xlabel)
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(individual_dir, f'{var}_comparison.png'))
        plt.close()

def handle_error(individual_dir, message, stdout="", stderr=""):
    report_data = {
        "status": "ERROR",
        "message": message,
        "stdout": stdout,
        "stderr": stderr
    }
    update_model_report(individual_dir, report_data)
    log_message(message, TextColors.RED)
    return "FAILED", None

def run_model(individual_dir, test_args=None):
    log_message("Running model for individual...", TextColors.YELLOW)
    log_message(f"Individual directory: {individual_dir}")
    
    # Remove old model compilations
    for file in ['model.dll', 'model.o']:
        file_path = os.path.join(individual_dir, file)
        if os.path.exists(file_path):
            os.remove(file_path)
            log_message(f"Removed old compilation: {file}", TextColors.YELLOW)
    
    # Run R script
    individual_dir_fslash = individual_dir.replace('\\','//')
    timestamp = int(time.time())
    process = subprocess.run(f'Rscript --vanilla ./Code/ControlFile.R --model_location "{individual_dir_fslash}" --timestamp {timestamp}', capture_output=True, text=True, encoding='utf-8', shell=True)
    stdout = process.stdout.strip()
    stderr = process.stderr.strip()

    # Print stdout/stderr for debugging
    if stdout:
        log_message("R script stdout:", TextColors.YELLOW)
        log_message(stdout)
    if stderr:
        log_message("R script stderr:", TextColors.YELLOW)
        log_message(stderr)

    # Check if R script failed or compilation failed
    if process.returncode != 0 or 'Compilation failed' in stdout or 'Compilation failed' in stderr:
        error_msg = "Model failed to compile." if 'Compilation failed' in stdout or 'Compilation failed' in stderr else f"R script failed with return code {process.returncode}"
        return handle_error(individual_dir, error_msg, stdout, stderr)

    try:
        # Look for JSON report in stdout or stderr
        def extract_json(text):
            start = text.find('JSON_REPORT_START')
            if start == -1:
                return None
            
            # Find the actual start of JSON after the marker
            json_text_start = text.find('\n', start) + 1
            if json_text_start == 0:  # No newline found
                return None
            
            end = text.find('JSON_REPORT_END', json_text_start)
            if end == -1:
                return None
                
            return text[json_text_start:end].strip()
            
        # Try stdout first, then stderr
        json_str = extract_json(stdout)
        if json_str is None:
            json_str = extract_json(stderr)
            
        if json_str is None:
            return handle_error(individual_dir, "No JSON report found in output", stdout, stderr)
            
        try:
            report_data = json.loads(json_str)
        except json.JSONDecodeError as e:
            log_message(f"JSON parse error: {str(e)}", TextColors.RED)
            log_message("JSON string:", TextColors.YELLOW)
            log_message(json_str)
            return handle_error(individual_dir, f"Failed to parse JSON report: {str(e)}", stdout, stderr)
        
        # Add stdout/stderr to report
        report_data["stdout"] = stdout
        report_data["stderr"] = stderr
        
        # Create iteration with complete data
        update_model_report(individual_dir, report_data)
        
        if report_data.get('status') == 'ERROR':
            return handle_error(individual_dir, f"Model execution failed: {report_data.get('message')}")
        
        # Extract objective value
        objective_value = report_data.get('objective_value')
        # if objective_value is None:
        #     objective_value = report_data.get('model_report', {}).get('nll')
        
        if objective_value is None:
            raise ValueError("Objective function value is missing from the report")
        
        if isinstance(objective_value, list):
            if len(objective_value) == 0:
                raise ValueError("Objective function value list is empty")
            objective_value = objective_value[0]
        
        objective_value = float(objective_value)
        
        if objective_value == float('inf'):
            raise ValueError("Objective function value is infinity")
        
        log_message(f"Objective value: {objective_value}")
        
        # Remove any existing error files
        remove_error_files(individual_dir)
        
        # Create plots comparing modeled projections with historical data
        plot_data = report_data.get('plot_data', {})
        if isinstance(plot_data, list):
            plot_data = {str(i): data for i, data in enumerate(plot_data)}
        plot_model_vs_historical(individual_dir, plot_data)
        
        log_message(f"Objective Value: {objective_value}", TextColors.BLUE)
        log_message('Successful model run', TextColors.GREEN)
        log_message('--------------------------------------')
        return "SUCCESS", objective_value
    except Exception as e:
        return handle_error(individual_dir, f"Error processing model report: {str(e)}", stdout, stderr)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        log_message("Please provide the individual directory as an argument.", TextColors.RED)
        sys.exit(1)
    
    individual_dir = sys.argv[1]
    status, result = run_model(individual_dir)
    log_message(f"Final status: {status}")
    log_message(f"Final result: {result}")
    if status == "FAILED":
        log_message(f"Error information: {result}", TextColors.RED)
