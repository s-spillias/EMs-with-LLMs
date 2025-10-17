import os
import json
import pandas as pd
import re
import glob
import shutil
import random
import string
import sys
import io
import subprocess
import math
from scripts.ask_AI import ask_ai
from scripts.get_params import get_params,_resolve_model_name_from_rag_choice
from scripts.run_model import run_model
from scripts.model_report_handler import update_model_report, read_model_report
from scripts.validate_tmb_model import validate_tmb_model
from dotenv import load_dotenv
from aider.coders import Coder
from aider.models import Model
from aider.io import InputOutput
import functools as _functools
print = _functools.partial(print, flush=True)  # force-flush all prints in this module

from pathlib import Path



load_dotenv()
# Set PYTHONIOENCODING to utf-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
os.environ['PYTHONIOENCODING'] = 'utf-8'

def enhance_parameter_descriptions(individual_dir, project_topic):
    print("Enhancing parameter descriptions...")

    # Read the current parameters.json with error handling
    params_file = os.path.join(individual_dir, 'parameters.json')
    try:
        with open(params_file, 'r') as f:
            content = f.read()
        
        # Try to parse JSON directly first
        try:
            params_data = json.loads(content)
        except json.JSONDecodeError:
            # If that fails, strip comments and try again
            print("JSON parsing failed, removing comments...")
            # Remove // comments and /* */ comments
            content = re.sub(r'//.*$', '', content, flags=re.MULTILINE)
            content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)
            params_data = json.loads(content)
            print("Successfully parsed JSON after removing comments")
            
    except Exception as e:
        print(f"Error reading parameters.json: {e}")
        print("Skipping parameter enhancement due to JSON parsing issues")
        return

    # Read the model.cpp for additional context
    model_file = os.path.join(individual_dir, 'model.cpp')
    try:
        with open(model_file, 'r') as f:
            model_content = f.read()
    except Exception as e:
        print(f"Error reading model.cpp: {e}")
        print("Skipping parameter enhancement due to file reading issues")
        return

    # Create a list of current descriptions
    descriptions = []
    for param in params_data.get('parameters', []):
        # Skip if already enhanced
        if param.get('enhanced_semantic_description'):
            continue
        
        # Check if required keys exist
        if 'parameter' not in param:
            print(f"Warning: Parameter missing 'parameter' key, skipping: {param}")
            continue
            
        # Use description if available, otherwise use empty string
        description = param.get('description', '')
        
        descriptions.append({
            'parameter': param['parameter'],
            'description': description,
        })

    # If all parameters already have enhanced descriptions, return
    if not descriptions:
        return

    # Create prompt for the LLM
    prompt = f"""Given a mathematical model about {project_topic}, enhance the semantic descriptions of these parameters to be more detailed and searchable,
AND also propose biologically reasonable numeric bounds when applicable.

The model code shows these parameters are used in the following way:
{model_content}

For each parameter below:
- Create an 'enhanced_semantic_description' (≤ 10 words) suitable for RAG/semantic search.
- Provide 'lower_bound' and 'upper_bound' as numbers when applicable (e.g., rates ≥ 0, proportions in [0,1], variances > 0, half-saturation constants > 0, mortality in [0, ∞), etc.). Use null if not applicable or unknown.
- Ensure lower_bound < upper_bound when both are provided.

Current parameter descriptions:
{json.dumps(descriptions, indent=2)}

Return a JSON array of objects with fields:
- 'parameter'
- 'enhanced_semantic_description'
- 'lower_bound'  (number or null)
- 'upper_bound'  (number or null)

Example format:
[
  {{
    "parameter": "example_param",
    "enhanced_semantic_description": "Saturating resource uptake coefficient",
    "lower_bound": 0.0,
    "upper_bound": null
  }}
]
"""

    def extract_curly_braces(response):
        # Try strict JSON first
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            # Fallback: attempt to recover simple JSON objects via regex (best-effort)
            matches = re.findall(r'\{([^}]*)\}', response)
            json_objects = []
            for match in matches:
                fields = re.findall(r'"(.*?)"\s*:\s*"?(.*?)"?(?:,|$)', match)
                json_object = {key: value for key, value in fields}
                json_objects.append(json_object)
            return json_objects

    try:
        # Look for population_metadata.json in the parent directory to pick LLM
        parent_dir = os.path.dirname(individual_dir)
        with open(os.path.join(parent_dir, 'population_metadata.json'), 'r') as file:
            rag_choice = json.load(file)['rag_choice']
        
        llm_response = ask_ai(prompt, rag_choice)
        enhanced_descriptions = extract_curly_braces(llm_response)

        # Helper to coerce possible string -> float -> None
        def to_number_or_none(v):
            if v is None:
                return None
            if isinstance(v, (int, float)):
                return float(v)
            v_str = str(v).strip()
            if v_str.lower() in ("", "na", "n/a", "null", "none"):
                return None
            try:
                return float(v_str)
            except Exception:
                return None

        # Update parameters.json with enhanced descriptions and bounds
        for param in params_data['parameters']:
            for enhanced in enhanced_descriptions:
                if param['parameter'] == enhanced.get('parameter'):
                    if 'enhanced_semantic_description' in enhanced:
                        param['enhanced_semantic_description'] = enhanced['enhanced_semantic_description']

                    lb = to_number_or_none(enhanced.get('lower_bound', None))
                    ub = to_number_or_none(enhanced.get('upper_bound', None))

                    # Only write if at least one bound is provided
                    if lb is not None:
                        param['lower_bound'] = lb
                    if ub is not None:
                        param['upper_bound'] = ub

        # Write updated parameters.json
        with open(params_file, 'w') as f:
            json.dump(params_data, f, indent=4)

    except Exception as e:
        print(f"Error enhancing parameter descriptions: {e}")


class AutoRespondInputOutput(InputOutput):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def input(self, prompt):
        if "to the chat? (Y)es/(N)o/(D)on't ask again" in prompt:
            return 'N'
        return super().input(prompt)

# Define which LLMs can handle images
MULTIMODAL_LLMS = {
    "anthropic_sonnet": True,    # Claude 3 Sonnet
    "claude_3_7_sonnet": True,   # Claude 3.7 Sonnet
    "claude_4_sonnet":True,
    "gemini": True,              # Gemini
    "o3_mini": False,            # Other models
    "o3": False,            # Other models
    "o4_mini": False,            # Other models
    "o1_mini": False,
    "gpt_4.1":False,
    "anthropic_haiku": False,
    "groq": False,
    "bedrock": False,
    "gemini_2.0_flash": True,
    "gemini_2.5_pro": True,
    "ollama:DeepCoder_14B_Preview_GGUF": False,
    'gemini_2.5_pro_exp_03_25': True,
    'openrouter:openai/gpt-5-chat': False,
    # New Ollama models (all text-only)
    'ollama:deepseek-r1:latest': False,
    'ollama:gemma:latest': False,
    'ollama:devstral:latest': False,
    'ollama:qwen3:30b-a3b': False,
    'ollama:mistral:latest': False,
    'ollama:qwen3:4b': False
}

def setup_coder(filenames, read_files, temperature=0.1, llm_choice="anthropic_sonnet"):
    io = AutoRespondInputOutput(yes=True)
    if llm_choice == "anthropic_sonnet":
        model = Model('claude-3-5-sonnet-20241022')
    elif llm_choice == "o3_mini":
        model = Model('o3-mini')
    elif llm_choice == "o3":
        model = Model('o3')
    elif llm_choice == "o4_mini":
        model = Model('o3-mini')
    elif llm_choice == "o1_mini":
        model = Model('o1-mini')
    elif llm_choice == "gpt_4.1":
        model = Model('gpt-4.1')
    elif llm_choice == "anthropic_haiku":
        model = Model('claude-3-5-haiku-20241022')
    elif llm_choice == "claude_3_7_sonnet":
        model = Model('claude-3-7-sonnet-20250219')
    elif llm_choice == "claude_4_sonnet":
        model = Model('claude-sonnet-4-20250514')    
    elif llm_choice == "groq":
        model = Model('llama3-70b-8192')
    elif llm_choice == "bedrock":
        model = Model('amazon.titan-text-express-v1')
    elif llm_choice == "gemini":
        model = Model('models/gemini-1.5-flash')
    elif llm_choice == "gemini_2.0_flash":
        model = Model('gemini/gemini-2.0-flash')
    elif llm_choice == "gemini_2.5_pro":
        model = Model('gemini/gemini-2.5-pro')
    elif llm_choice == "gemini_2.5_pro_exp_03_25":
        model = Model('gemini/gemini-2.5-pro-exp-03-25')
    elif llm_choice == "gpt_4o":
        model = Model('gpt-4o')
    # --- in make_model.setup_coder() ---
    elif llm_choice.startswith('ollama:'):
        ollama_model_name = llm_choice.split(':', 1)[1]

        # Start 'ollama serve' if it's not already running (non-blocking)
        def _ollama_up():
            try:
                # Cheap check: see if the process is already listening on default port
                # You can replace this with a healthcheck call if your environment allows network.
                import socket
                with socket.create_connection(("127.0.0.1", 11434), timeout=0.5):
                    return True
            except Exception:
                return False

        if not _ollama_up():
            import subprocess, os
            subprocess.Popen(
                ['ollama', 'serve'],
                env={**os.environ, 'OLLAMA_CONTEXT_LENGTH': '8192'},
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            # Optional: small delay to let it come up
            import time; time.sleep(0.5)

        model = Model(f"ollama_chat/{ollama_model_name}")

    elif llm_choice.startswith('openrouter:'):
        openrouter_model_name = llm_choice.split(':', 1)[1]
        model = Model(f'openrouter/{openrouter_model_name}')
    else:
        raise ValueError(f"Unsupported LLM choice: {llm_choice}")
    
    coder = Coder.create(
        main_model=model,
        fnames=filenames,
        io=io
    )
    
    coder.repo_map = False
    coder.use_git = False
    coder.no_git = True
    coder.no_auto_commits = True
    if llm_choice == "o3_mini" or llm_choice == "o4_mini":
        coder.weak_model_name = 'gpt-4o-mini'
        coder.use_repo_map = False
        coder.use_temperature = False             
    else:
        coder.temperature = temperature
    coder.auto_commits = False
    # Convert read_files to absolute paths
    read_files = [str(Path(file).resolve()) for file in read_files]
    for file in read_files:
        coder.run(f"/read {file}")
    return coder

def make_script(filenames, read_files, prompt, temperature=0.1, llm_choice="anthropic_sonnet"):
    coder = setup_coder(filenames, read_files, temperature, llm_choice)
    coder.run(prompt)
    return coder

def improve_script(individual_dir, project_topic, temperature=0.05, llm_choice="anthropic_sonnet"):
    # Read metadata to get parent information
    metadata_file = os.path.join(individual_dir, 'metadata.json')
    if os.path.exists(metadata_file):
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        parent = metadata.get('parent')
    else:
        parent = None

    # Determine what files to use based on LLM capabilities
    read_files = []
    if MULTIMODAL_LLMS.get(llm_choice, False):
        # For multimodal LLMs, use PNG files from parent if available
        if parent:
            parent_dir = os.path.join(os.path.dirname(individual_dir), parent)
            read_files = glob.glob(os.path.join(parent_dir, '*.png'))
            print("Parent PNG files:", read_files)
    else:
        # For text-only LLMs, use residuals file if available
        residuals_file = os.path.join(individual_dir, 'model_residuals.json')
        if os.path.exists(residuals_file):
            read_files = [residuals_file]
            print("Using residuals file:", residuals_file)

    # Prepare filenames list
    filenames = [
        os.path.join(individual_dir, 'model.cpp'),
        os.path.join(individual_dir, 'parameters.json'),
        os.path.join(individual_dir, 'intention.txt')
    ]
    # Prepare the improvement prompt
    improve_prompt = (
        f"PROJECT CONTEXT: Mathematical model about {project_topic}\n\n"
        "First, assess the current model:\n"
        "1. Evaluate how well the model fits the data\n"
        "2. Analyze if the model effectively addresses the PROJECT CONTEXT\n"
        "3. Identify any key ecological processes that may be missing or oversimplified\n\n"
        "Additionally, review the current parameter values."
        "Be on the lookout for parameters that were initially placeholders but now have updated values "
        "from literature searches or other evidence. If these updated values suggest that the original "
        "equation structure is no longer appropriate (e.g., scaling, functional form, or interaction strength), "
        "propose modifications to the relevant equation components to maintain ecological realism.\n\n"
        "Based on your assessment, consider ONE meaningful ecological improvement, exploring these approaches:\n"
        "- Higher-order mathematical representations (e.g., polynomial terms, non-linear responses)\n"
        "- Resource limitation mechanisms (e.g., saturating uptake, competition effects)\n"
        "- Environmental modifiers of processes\n"
        "- Variable efficiency terms\n"
        "- Indirect pathways and feedback mechanisms\n\n"
        "Choose the approach that best captures the system dynamics for the PROJECT CONTEXT. "
        "Consider both simpler and more complex mathematical forms, but justify any added complexity "
        "with clear ecological reasoning.\n"
    )

    if MULTIMODAL_LLMS.get(llm_choice, False):
        improve_prompt += "Refer to the attached image files to see how the current model performs.\n"
    else:
        improve_prompt += "Refer to the residuals file to see how the current model performs. "
        improve_prompt += "The residuals show the difference between observed and predicted values.\n"

    improve_prompt += (
        "\nDocument your changes:\n"
        "1. Update intention.txt with your assessment and reasoning for the chosen improvement\n"
        "2. If adding or modifying parameters in parameters.json, include clear ecological justification and, where biologically appropriate, include numeric 'lower_bound' and 'upper_bound' suggestions (use null if not applicable)\n"
        "3. In model.cpp, ensure mathematical representations are properly implemented and reflect any structural changes required by updated parameter values\n\n"
        "IMPORTANT: Never use current time step values of response variables (variables ending in '_dat') "
        "in prediction calculations. Only use values from previous time steps to avoid data leakage.\n"
        "Do not attempt to add any more files to the chat or provide advice on compiling the model."
    )



    # Create a new coder object
    coder = setup_coder(filenames, read_files, temperature, llm_choice)
    try:
        # Run the improvement
        coder.run(improve_prompt)
    except UnicodeEncodeError as e:
        print(f"UnicodeEncodeError occurred: {e}")
        print("Attempting to encode the problematic string...")
        try:
            coder.run(improve_prompt)
        except Exception as e:
            print(f"Error occurred while trying to handle encoding: {e}")
    return coder

def generate_random_string(length=10):
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

def handle_successful_run(individual_dir, project_topic, objective_value):
    """Handle common tasks after a successful model run, then do a final run after get_params()."""
    print(f"Model run successful. Objective value: {objective_value}")
    update_model_report(individual_dir, {"status": "SUCCESS", "objective_value": objective_value})

    try:
        print("Model ran successful and returned meaningful objective value... enhancing parameter descriptions.")
        enhance_parameter_descriptions(individual_dir, project_topic)
        print("Parameter descriptions enhanced successfully.")

        print("Running parameter processing...")
        get_params(individual_dir)
        print("Parameter processing completed successfully.")

        # --- NEW: final model run after get_params() ---
        print("Re-running model after parameter processing...")
        final_status, final_objective_value = run_model(individual_dir)
        print("FINAL MODEL RUN FINISHED")

        try:
            if final_objective_value is None:
                raise ValueError("Final objective value is None")
            final_objective_value = float(final_objective_value)
            if math.isnan(final_objective_value):
                raise ValueError("Final objective value is NaN")

            # Update the report with the final objective and return the final value
            update_model_report(
                individual_dir,
                {"status": "SUCCESS", "objective_value": final_objective_value}
            )
            print(f"Final model run successful. Objective value: {final_objective_value}")
            objective_value = final_objective_value
        except Exception as e:
            # Keep initial objective if the final run doesn't yield a valid objective
            print(f"Final run after get_params failed or returned invalid objective value: {e}")
            update_model_report(
                individual_dir,
                {
                    "status": "SUCCESS_PARTIAL",
                    "message": "Final run after get_params failed; keeping initial objective value",
                    "objective_value": objective_value,
                },
            )

    except Exception as e:
        print(f"FATAL ERROR in post-processing: {e}")
        print("Marking individual as broken and returning large objective value...")
        import traceback
        traceback.print_exc()
        
        # Mark individual as broken instead of terminating the process
        error_message = f"Post-processing failed: {str(e)}\n{traceback.format_exc()}"
        update_model_report(
            individual_dir,
            {
                "status": "BROKEN",
                "message": error_message,
                "objective_value": float('inf')
            }
        )
        return "BROKEN", float('inf')

    return "SUCCESS", objective_value


def process_individual(individual_dir, project_topic, response_file, forcing_file, report_file, temperature=0.1, max_sub_iterations=5, llm_choice="anthropic_sonnet", train_test_split=1.0):
    os.makedirs(individual_dir, exist_ok=True)

    model_file = os.path.join(individual_dir, 'model.cpp')
    parm_file = os.path.join(individual_dir, 'parameters.json')
    intention_file = os.path.join(individual_dir, 'intention.txt')
    
    # Read response data file
    response_data = pd.read_csv(response_file)
    
    # Get the first column name from response data
    merge_column = response_data.columns[0]
    
    # Initialize data with response_data
    data = response_data.copy()
    
    # If forcing file is provided and not empty, merge with forcing data
    if forcing_file and forcing_file.strip():
        forcing_data = pd.read_csv(forcing_file)
        data = pd.merge(response_data, forcing_data, on=merge_column, how='outer')
    
    # Truncate data based on train_test_split
    if train_test_split < 1.0:
        n_rows = len(data)
        n_train = int(n_rows * train_test_split)
        data = data.iloc[:n_train]
    time_series = data.columns.tolist()

    if not os.path.exists(model_file):
        # Initialize the individual if it's new
        template_content = ""
        # if template_file and template_file.strip():
        #     with open(template_file, 'r') as file:
        #         template_content = file.read()
        random_string = generate_random_string()
        prompt_model = (
            "You are a leading expert in constructing dynamic ecosystem models. You always use robust ecological theory to construct your models, which will be used for predicting future ecosystem states given data on initial conditions. Please create a Template Model Builder model for the following topic:"
            f"{project_topic}. Start by writing intention.txt, in which you provide a concise summary of the ecological functioning of the model. In model.cpp, write your TMB model with the following important considerations:"
            "\n\n1. ECOLOGICAL PROCESSES:"
            "\n- Consider multiple forms of resource limitation (e.g., saturating functions, threshold effects)"
            "\n- Include process-specific efficiencies where biologically relevant"
            "\n- Think about how environmental conditions might modify rates"
            "\n- Consider indirect effects and feedback loops between components"
            "\n- Use functional responses that capture observed biological behaviors"
            "\n\n2. NUMERICAL STABILITY:"
            "\n- Always use small constants (e.g., Type(1e-8)) to prevent division by zero"
            "\n- Use smooth transitions instead of hard cutoffs in equations"
            "\n- Bound parameters within biologically meaningful ranges using smooth penalties rather than hard constraints"
            "\n  (and propose numeric lower/upper bounds per parameter when applicable; see parameters.json spec below)"
            "\n\n3. LIKELIHOOD CALCULATION:"
            "\n- Always include observations in the likelihood calculation, don't skip any based on conditions"
            "\n- Use fixed minimum standard deviations to prevent numerical issues when data values are small"
            "\n- Consider log-transforming data if it spans multiple orders of magnitude"
            "\n- Use appropriate error distributions (e.g., lognormal for strictly positive data)"
            "\n\n4. MODEL STRUCTURE:"
            "\n- Include comments after each line explaining the parameters (including their units and how to determine their values)"
            "\n- Provide a numbered list of descriptions for the equations"
            "\n- Ensure all _pred variables are included in the reporting section and called using REPORT()"
            "\n- Use '_pred' suffix for model predictions corresponding to '_dat' observations. Use the same _dat names as are found in the data file. Use the exact same time variable name as is provided in the first column of the datafile."
            "\n- IMPORTANT: Never use current time step values of response variables (variables ending in '_dat') in prediction calculations. Only use values from previous time steps to avoid data leakage."
            "\n- INITIAL CONDITIONS: Initialize your prediction vectors with the first data point using **name**_dat(0). For example: var1_pred(0) = var1_dat(0); var2_pred(0) = var2_dat(0); var3_pred(0) = var3_dat(0). This ensures initial conditions are drawn directly from the observed data rather than being optimization parameters."
            "\n\nFor the parameters.json file, please structure it as an array of parameter objects, where each parameter object must include the following fields:"
            "\n- parameter: The name of the parameter matching the model.cpp"
            "\n- value: The initial value for the parameter"
            "\n- description: A clear description of what the parameter represents, including units"
            "\n- source: Where the initial value comes from. IMPORTANT: If the source contains the word 'literature', this will automatically trigger downstream literature searches using Semantic Scholar and other academic databases to find citations and refine parameter values. Use 'literature' only when you want the system to search for academic papers. Use 'initial estimate' for parameters that are unlikely to have reported values in the literature."
            "\n- import_type: Should be 'PARAMETER' for model parameters, or 'DATA_VECTOR'/'DATA_SCALAR' for data inputs"
            "\n- priority: A number indicating the optimization priority (1 for highest priority parameters to optimize first)"
            "\n- lower_bound (optional): Suggested biological lower bound as a number, or null if not applicable"
            "\n- upper_bound (optional): Suggested biological upper bound as a number, or null if not applicable"
            "\n\nExample structure:"
            "\n{"
            "\n  \"parameters\": ["
            "\n    {"
            "\n      \"parameter\": \"growth_rate\","
            "\n      \"value\": 0.5,"
            "\n      \"units\": \"dimensionsless | year ^-1\","
            "\n      \"description\": \"Intrinsic growth rate (year^-1)\","
            "\n      \"source\": \"literature\","
            "\n      \"import_type\": \"PARAMETER\","
            "\n      \"priority\": 1,"
            "\n      \"lower_bound\": 0.0,"
            "\n      \"upper_bound\": null"
            "\n    }"
            "\n  ]"
            "\n}"
        )

        print(prompt_model)
        filenames = [model_file, parm_file, intention_file]
        data_dict = data.to_dict()
        read_files = [response_file]
        if forcing_file and forcing_file.strip():
            read_files.append(forcing_file)
        coder = make_script(filenames, read_files, prompt_model, temperature, llm_choice)
        print(f"Initialized new individual: {individual_dir}")
    else:
        # Improve the existing model
        coder = improve_script(individual_dir, project_topic, temperature, llm_choice)
        print(f"Improved existing model in individual: {individual_dir}")
        
    # After every call to the 'coder', validate the model first
    print("Validating model for data leakage...")
    
    warnings = validate_tmb_model(os.path.join(individual_dir, 'model.cpp'))
    if not warnings:
         # If validation passes, run the model
        run_status, objective_value = run_model(individual_dir)
        print("MODEL RUN FINISHED")
        try:
            if objective_value is None:
                raise ValueError("Objective value is None")
            objective_value = float(objective_value)
            if math.isnan(objective_value):
                raise ValueError("Objective value is NaN")
            return handle_successful_run(individual_dir, project_topic, objective_value)
        except ValueError as e:
            print(f"Error: {str(e)}")
            print("Initial run failed. Attempting to fix...")
    else:
        print("\nWarnings found in model:")
        for warning in warnings:
            print(f"  {warning}")
        print("\nModel contains data leakage issues - marking as broken")
        message = "Data leakage detected in model equations:\n" + "\n".join(f"  {warning}" for warning in warnings)
        update_model_report(individual_dir, {"status": "LEAKAGE", "message": message, "objective_value": "NA"})
        run_status = 'LEAKAGE'
   
    # Enter sub-iteration loop only if initial run failed
    for sub_iteration in range(max_sub_iterations):
        print(f"Fixing broken model iteration: {sub_iteration}")
        error_info = read_model_report(individual_dir)
        
        if run_status == 'FAILED':
            error_prompt = f"model.cpp failed to compile. Here's the error information:\n\n{error_info}\n\nDo not suggest how to compile the script"
        elif run_status == 'LEAKAGE':
            try:
                print("pre-error maybe")
                # Prepare forcing variables text based on whether forcing file exists
                forcing_vars_text = (
                    f"   - External forcing variables ({', '.join(col.split(' ')[0] for col in forcing_data.columns if col != merge_column)})" 
                    if forcing_file and 'forcing_data' in locals() 
                    else "and "
                )
                
                error_prompt = f"Issue with model set-up:" + message
            except:
                print("there is an error here.")
            print(error_prompt)
        elif run_status == 'SUCCESS':
            if sub_iteration >= 2:
                error_prompt = (
                    f"The model shows numerical instabilities. Here's the error information:\n\n{error_info}\n\n"
                    "Consider simplifying the ecological relationships:\n"
                    "1. Start with basic interactions between components\n"
                    "2. Use simpler functional responses\n"
                    "3. Remove secondary effects temporarily\n"
                    "4. Focus on dominant ecological processes\n"
                    "We can add complexity back gradually once the core dynamics are stable."
                )
            else:
                error_prompt = (
                    f"The model shows numerical instabilities. Here's the error information:\n\n{error_info}\n\n"
                    "Review the ecological relationships:\n"
                    "1. Check if process rates are biologically reasonable\n"
                    "2. Ensure interaction strengths are properly scaled\n"
                    "3. Consider if all included mechanisms are necessary\n"
                    "4. Verify resource limitation effects are appropriate"
                    "Do not suggest how to compile the script"
                )
        else:
            print(f"Unexpected run_status value: {run_status}")
            break

        try:
            coder.run(error_prompt)
        except UnicodeEncodeError as e:
            print(f"UnicodeEncodeError occurred: {e}")
            print("Attempting to encode the problematic string...")
            try:
                coder.run(error_prompt)
            except Exception as e:
                print(f"Error occurred while trying to handle encoding: {e}")
                break

        # Validate the fixed model
        print("Validating fixed model for data leakage...")
        warnings = validate_tmb_model(os.path.join(individual_dir, 'model.cpp'))
        if warnings:
            print("\nWarnings found in fixed model:")
            for warning in warnings:
                print(f"  {warning}")
            print("\nFixed model still contains data leakage issues")
            message = "Data leakage detected in model equations:\n" + "\n".join(f"  {warning}" for warning in warnings)
            update_model_report(individual_dir, {"status": "LEAKAGE", "message": message, "objective_value": "NA"})
            run_status = 'LEAKAGE'
            continue

        # If validation passes, run the model
        run_status, objective_value = run_model(individual_dir)
        print("MODEL RUN COMPLETED")
        try:
            if objective_value is None:
                raise ValueError("Objective value is None")
            objective_value = float(objective_value)
            if math.isnan(objective_value):
                raise ValueError("Objective value is NaN")
            return handle_successful_run(individual_dir, project_topic, objective_value)
        except ValueError:
            continue

    print(f"Maximum sub-iterations reached for {individual_dir}. The model could not be successfully run after {max_sub_iterations} attempts.")
    update_model_report(individual_dir, {"status": "BROKEN"})
    return "BROKEN", None
