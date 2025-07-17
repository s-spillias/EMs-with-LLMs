import re
import sys
from typing import List, Dict, Set, Tuple
from dataclasses import dataclass

@dataclass
class ValidationContext:
    filename: str
    content: str
    line_number: int = 0
    warnings: List[str] = None
    errors: List[str] = None

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []
        if self.errors is None:
            self.errors = []

    def add_warning(self, message: str):
        self.warnings.append(f"Line {self.line_number}: {message}")
        
    def add_error(self, message: str):
        self.errors.append(message)

def extract_data_vectors(content: str) -> Set[str]:
    """Extract all data vector declarations from TMB models."""
    # TMB-specific patterns for data vectors
    patterns = [
        r'DATA_VECTOR\((\w+)\)',  # Standard TMB DATA_VECTOR macro
        r'DATA_MATRIX\((\w+)\)',  # Matrix data inputs
        r'DATA_SCALAR\((\w+)\)',  # Scalar data inputs
        r'DATA_VECTOR_INDICATOR\((\w+)\)',  # Indicator vectors
    ]
    vectors = set()
    for pattern in patterns:
        vectors.update(re.findall(pattern, content))
    return vectors

def extract_prediction_vectors(content: str) -> Set[str]:
    """Extract all prediction vector declarations from TMB models."""
    vectors = set()
    
    # Single vector declarations
    single_patterns = [
        r'vector<Type>\s+(\w+)\s*\(\w+\.size\(\)\)',  # Size-based initialization
        r'vector<Type>\s+(\w+)\s*\([^;]+\)',  # Other parentheses initialization
        r'vector<Type>\s+(\w+)\s*\[\w+\]',  # Bracket initialization
        r'vector<Type>\s+(\w+);',  # Simple declaration
    ]
    for pattern in single_patterns:
        vectors.update(re.findall(pattern, content))
    
    # Comma-separated declarations
    comma_pattern = r'vector<Type>\s+\w+(?:\([^;]+\))?,\s*(\w+)(?:\([^;]+\))?,\s*(\w+)(?:\([^;]+\))?;'
    for match in re.finditer(comma_pattern, content):
        vectors.update(g for g in match.groups() if g is not None)
    
    return vectors

def extract_reported_vectors(content: str) -> Set[str]:
    """Extract all vectors that are reported using REPORT() statements."""
    pattern = r'REPORT\((\w+)\)'
    return set(re.findall(pattern, content))

def find_initial_conditions(content: str) -> List[Tuple[int, str]]:
    """Find lines with initial conditions assignments in TMB models."""
    lines = content.split('\n')
    initial_conditions = []
    
    # TMB-specific patterns for initial conditions
    patterns = [
        r'\w+\(0\)\s*=',  # Vector initialization at index 0
        r'\w+\[0\]\s*=',  # Array initialization at index 0
        r'Type\s+\w+\s*=\s*\w+\(0\)',  # Scalar initialization from vector
        r'vector<Type>\s+\w+\s*=\s*\w+\(0\)',  # Vector initialization from other vector
    ]
    
    for i, line in enumerate(lines, 1):
        line = line.strip()
        if any(re.search(pattern, line) for pattern in patterns):
            initial_conditions.append((i, line))
    return initial_conditions

def find_timestep_loop(content: str) -> Tuple[int, int, str]:
    """Find the main time-stepping loop in TMB models."""
    lines = content.split('\n')
    loop_start = -1
    loop_end = -1
    loop_content = ""
    
    # TMB-specific loop patterns
    loop_patterns = [
        r'for\s*\(\s*int\s+[it]\s*=\s*\d+\s*;',  # Standard integer counter
        r'for\s*\(\s*Type\s+[it]\s*=\s*\d+\s*;',  # Type counter
        r'for\s*\(\s*int\s+[it]\s*=\s*\w+\s*;',  # Variable start
    ]
    
    for i, line in enumerate(lines):
        if any(re.search(pattern, line) for pattern in loop_patterns):
            loop_start = i
            # Find matching closing brace
            brace_count = 0
            for j, next_line in enumerate(lines[i:], i):
                if '{' in next_line:
                    brace_count += 1
                if '}' in next_line:
                    brace_count -= 1
                    if brace_count == 0:
                        loop_end = j
                        loop_content = '\n'.join(lines[i:j+1])
                        break
            if loop_content:  # Only break if we found a complete loop
                break
    
    return loop_start + 1, loop_end + 1, loop_content

def get_response_variables_from_csv(filename: str) -> Set[str]:
    """Extract response variable names from the CSV file."""
    import pandas as pd
    try:
        df = pd.read_csv(filename)
        # Get column names that end with '_dat'
        return {col.split()[0] for col in df.columns if col.split()[0].endswith('_dat')}
    except Exception as e:
        print(f"Warning: Could not read response file {filename}: {e}")
        return set()

def check_data_usage_in_predictions(context: ValidationContext, data_vectors: Set[str], prediction_vectors: Set[str], loop_content: str):
    """Check for inappropriate data vector usage in prediction calculations."""
    import os
    import json
    import pandas as pd
    
    # Initialize predicted_vars
    predicted_vars = set()
    
    # Get response variables from population_metadata.json
    try:
        with open("population_metadata.json", 'r') as f:
            metadata = json.load(f)
            response_file = metadata.get("response_file")
            if response_file:
                response_vars = get_response_variables_from_csv(response_file)
                print(f"Found response variables from {response_file}: {response_vars}")
            else:
                response_vars = {vec for vec in data_vectors if vec.endswith('_dat')}
                print(f"No response file found, using data vectors: {response_vars}")
    except Exception as e:
        print(f"Warning: Could not read population_metadata.json: {e}")
        response_vars = {vec for vec in data_vectors if vec.endswith('_dat')}
        print(f"Using data vectors as fallback: {response_vars}")

    # Create mapping of response variables to their corresponding prediction vectors
    pred_map = {data_vec: f"{data_vec[:-4]}_pred" for data_vec in response_vars}
    
    # Check that all response variables have corresponding prediction vectors declared
    for data_vec, pred_vec in pred_map.items():
        if pred_vec not in prediction_vectors:
            context.add_warning(
                f"Missing prediction vector: {pred_vec} not found in model.\n"
                f"    Required for response variable: {data_vec}"
            )

    # Check each line in the loop
    lines = loop_content.split('\n')
    in_prediction = False
    in_likelihood = False
    current_prediction = None
    current_equation = []
    
    for i, line in enumerate(lines, context.line_number):
        context.line_number = i
        line = line.strip()
        
        # Skip comments and empty lines
        if not line or line.startswith('//'):
            continue

        # Check if this line is part of likelihood calculation
        if 'nll' in line or 'dnorm' in line or 'dpois' in line or 'dbinom' in line:
            in_likelihood = True
            in_prediction = False
            current_equation = []
            continue

        # Check if this line starts a prediction calculation
        if '=' in line and not in_prediction:
            left_side = line.split('=')[0].strip()
            # Find which prediction vector is being calculated
            for pred in prediction_vectors:
                # Look for vector assignments like "cots_pred(i) =" or "cots_pred[i] ="
                if pred in left_side and (
                    re.search(rf'{pred}\s*[\(\[]\s*[it]\s*[\)\]]', left_side) or  # (i) or [i]
                    re.search(rf'{pred}\s*[\(\[]\s*time\s*[\)\]]', left_side) or  # (time) or [time]
                    re.search(rf'{pred}\s*[\(\[]\s*step\s*[\)\]]', left_side)     # (step) or [step]
                ):
                    current_prediction = pred
                    in_prediction = True
                    in_likelihood = False
                    current_equation = [line]
                    break
        # If we're in a prediction calculation, add the line to the current equation
        elif in_prediction:
            # Remove continuation character if present
            if line.endswith('\\'):
                current_equation.append(line[:-1].strip())
            else:
                current_equation.append(line)
        
        # If we're in a prediction calculation (not likelihood), collect the equation
        if in_prediction and current_prediction and not in_likelihood:
            # If the line ends without continuation, process the complete equation
            if not line.endswith('\\') or line.rstrip('\\').strip().endswith(';'):
                # Join all lines of the equation
                full_equation = ' '.join(current_equation)
                
                # Check for data vector usage in the complete equation
                for data_vec, pred_vec in pred_map.items():
                    # Skip if this is not a response variable
                    if data_vec not in response_vars:
                        continue
                    
                    # Look for any usage of the data vector in the complete equation
                    if data_vec in full_equation:
                        # Data leakage can occur in two ways:
                        # 1. Using current value of a variable to predict itself
                        # 2. Using current value of any response variable in predictions
                        if pred_vec == current_prediction or data_vec in response_vars:
                            context.add_warning(
                                f"Data leakage detected: using {data_vec} in prediction calculation\n"
                                f"    in equation: {full_equation}"
                            )
                
                # Record that this variable was predicted
                for data_vec, pred_vec in pred_map.items():
                    if pred_vec == current_prediction:
                        predicted_vars.add(data_vec)
                
                # Reset prediction state
                in_prediction = False
                current_prediction = None
                current_equation = []
        
        # Reset likelihood flag if the line doesn't continue
        if in_likelihood and not line.endswith('\\'):
            in_likelihood = False

    # Verify all required prediction vectors exist and are used
    for data_vec, expected_pred in pred_map.items():
        if expected_pred not in prediction_vectors:
            context.add_warning(
                f"Missing prediction vector: {expected_pred} not found in model.\n"
                f"    Required for response variable: {data_vec}"
            )
        elif data_vec not in predicted_vars:
            context.add_warning(
                f"Missing prediction equation: {data_vec} has no corresponding prediction calculation.\n"
                f"    Expected to find equation for: {expected_pred}"
            )

def check_tmb_conventions(content: str) -> Dict[str, bool]:
    """Check if the file follows basic TMB conventions."""
    conventions = {
        "uses_tmb_header": "#include <TMB.hpp>" in content,
        "uses_data_macros": bool(re.search(r'DATA_\w+\(', content)),
        "uses_vector_type": bool(re.search(r'vector<Type>', content)),
        "uses_standard_report": bool(re.search(r'REPORT\(\w+\);', content)),
        "uses_objective_function": "objective_function" in content.lower(),
    }
    
    # Check for alternative patterns that indicate non-TMB style
    conventions["uses_std_vector"] = bool(re.search(r'std::vector<\w+>', content))
    conventions["uses_string_report"] = bool(re.search(r'REPORT\(\s*"[^"]+"\s*,', content))
    
    return conventions

def validate_tmb_model(filename: str) -> List[str]:
    """Validate a TMB model file for potential data leakage issues."""
    print(f"Opening file: {filename}")
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read()
    except UnicodeDecodeError:
        try:
            # Try with a different encoding if utf-8 fails
            with open(filename, 'r', encoding='latin-1') as f:
                content = f.read()
        except Exception as e:
            print(f"Error reading file: {e}")
            return [f"Could not read file due to encoding error: {e}"]
    print("File read successfully")

    context = ValidationContext(filename=filename, content=content)
    
    # Check if the file follows TMB conventions
    conventions = check_tmb_conventions(content)
    print(f"TMB convention check: {conventions}")
    
    # If the file doesn't follow TMB conventions, add an error and return early
    if not conventions["uses_data_macros"] or not conventions["uses_vector_type"] or conventions["uses_std_vector"]:
        context.add_error(
            "This file does not follow standard TMB conventions. Issues detected:\n" +
            (f"  - Uses std::vector<T> instead of vector<Type>\n" if conventions["uses_std_vector"] else "") +
            (f"  - Missing DATA_VECTOR() macros for data inputs\n" if not conventions["uses_data_macros"] else "") +
            (f"  - Missing vector<Type> declarations\n" if not conventions["uses_vector_type"] else "") +
            (f"  - Uses REPORT(\"name\", var) instead of REPORT(var)\n" if conventions["uses_string_report"] else "") +
            "\nThe model should be rewritten to follow TMB conventions."
        )
        # Return early without checking for data leakage
        return context.errors
    
    # Extract vectors
    print("Extracting vectors...")
    data_vectors = extract_data_vectors(content)
    prediction_vectors = extract_prediction_vectors(content)
    reported_vectors = extract_reported_vectors(content)
    print(f"Found data vectors: {data_vectors}")
    print(f"Found prediction vectors: {prediction_vectors}")
    print(f"Found reported vectors: {reported_vectors}")

    # Check if all prediction vectors are reported
    for pred_vec in prediction_vectors:
        if pred_vec.endswith('_pred') and pred_vec not in reported_vectors:
            context.add_warning(
                f"Missing REPORT statement: {pred_vec} is not reported.\n"
                f"    Add 'REPORT({pred_vec});' before the return statement."
            )
    
    # Find initial conditions
    print("Finding initial conditions...")
    initial_conditions = find_initial_conditions(content)
    print(f"Found {len(initial_conditions)} initial conditions")
    
    # Find main time-stepping loop
    print("Finding time-stepping loop...")
    loop_start, loop_end, loop_content = find_timestep_loop(content)
    if loop_content:
        print(f"Found loop from line {loop_start} to {loop_end}")
        context.line_number = loop_start
        check_data_usage_in_predictions(context, data_vectors, prediction_vectors, loop_content)
    else:
        print("No time-stepping loop found")
    
    return context.errors + context.warnings

def main():
    if len(sys.argv) != 2:
        print("Usage: python validate_tmb_model.py <path_to_model.cpp>")
        sys.exit(1)
    
    filename = sys.argv[1]
    messages = validate_tmb_model(filename)
    
    if messages:
        # Check if the first message contains "TMB conventions"
        if any("TMB conventions" in msg for msg in messages):
            print(f"\nStructural issues found in {filename}:")
            for message in messages:
                print(f"  {message}")
        else:
            print(f"\nData leakage issues found in {filename}:")
            for message in messages:
                print(f"  {message}")
    else:
        print(f"\nNo issues found in {filename}")

if __name__ == '__main__':
    main()
