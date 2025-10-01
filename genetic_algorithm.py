import os
import sys
import argparse
import time
import shutil
from typing import Dict, Any
from scripts.population_utils import get_next_population_number, generate_individual_id, evolve_population, create_new_generation
from scripts.individual_utils import update_individual_metadata
from scripts.evaluation_utils import run_make_model_with_file_output
from scripts.file_utils import save_metadata, load_metadata, move_individual
from scripts.model_report_handler import read_model_report
from scripts.rag_utils import rag_prepare 
from scripts.data_report import create_data_report
import multiprocessing
import logging
from dotenv import load_dotenv
import datetime

# NEW: file-free, shared rate limiter for Semantic Scholar
from scripts.search_utils import init_rate_limit_manager

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# NEW: pool worker initializer that wires the shared pacer into every worker
def _pool_worker_init(shared_mgr):
    """
    This is called once in each worker process when the Pool starts.
    It attaches the Manager-backed pacer so all workers on this host
    share a 1 req/sec budget to Semantic Scholar (no file locks).
    """
    try:
        # You can tweak via env: S2_MIN_INTERVAL, S2_MAX_CONCURRENT
        s2_min_interval = float(os.getenv("S2_MIN_INTERVAL", "1.0"))   # 1 rps by default
        s2_max_conc = os.getenv("S2_MAX_CONCURRENT")
        s2_max_concurrent = int(s2_max_conc) if s2_max_conc else 2     # allow up to 2 in-flight per process
        init_rate_limit_manager(shared_mgr, s2_min_interval=s2_min_interval, s2_max_concurrent=s2_max_concurrent)
    except Exception as e:
        print(f"WARNING: could not initialize S2 pacer in worker: {e}", flush=True)


def initialize_population(population_dir, n_individuals, n_parallel, project_topic, response_file,
                          forcing_file, report_file, temperature, max_sub_iterations, llm_choice,
                          train_test_split, shared_mgr=None):
    print("Initializing population")
    initial_individuals = [generate_individual_id() for _ in range(n_individuals)]
    print(f"Initial individuals: {initial_individuals}")

    # Check if we should run sequentially or in parallel
    if n_parallel <= 1:
        print(f"Running sequentially (n_parallel={n_parallel})")
        # Run sequentially
        for individual in initial_individuals:
            run_make_model_with_file_output(
                population_dir, individual, project_topic, response_file,
                forcing_file, report_file, temperature, max_sub_iterations,
                llm_choice, train_test_split
            )
    else:
        # Use the minimum of n_parallel and n_individuals for optimal resource usage
        actual_processes = min(n_parallel, n_individuals)
        print(f"Running in parallel with {actual_processes} processes")

    # Create a fork-based context explicitly (Unix)
    ctx = multiprocessing.get_context("fork")

    with ctx.Pool(
        processes=actual_processes,
        initializer=_pool_worker_init,
        initargs=(shared_mgr,),   # safe under 'fork'
    ) as pool:
        pool.starmap(
            run_make_model_with_file_output,
            [
                (
                    population_dir, individual, project_topic, response_file,
                    forcing_file, report_file, temperature, max_sub_iterations,
                    llm_choice, train_test_split
                )
                for individual in initial_individuals
            ]
        )

    return initial_individuals


def update_lineage(population_dir, child, parent):
    if parent:
        child_metadata = load_metadata(os.path.join(population_dir, child, 'metadata.json'))
        child_metadata['parent'] = parent
        parent_metadata = load_metadata(os.path.join(population_dir, parent, 'metadata.json'))
        child_metadata['lineage'] = parent_metadata.get('lineage', []) + [parent]
        save_metadata(os.path.join(population_dir, child, 'metadata.json'), child_metadata)


def get_individual_objective_value(population_dir, individual):
    report_data = read_model_report(os.path.join(population_dir, individual))
    objective_value = float(report_data.get('objective_value')) if report_data.get('objective_value') is not None else None
    # Update individual metadata with the objective value
    update_individual_metadata(os.path.join(population_dir, individual), objective_value)
    return objective_value


def load_config_from_file(config_file):
    """Load configuration from a JSON file."""
    import json
    with open(config_file, 'r') as f:
        config = json.load(f)
    return config


# Parameter validation specifications
PARAM_SPECS = {
    'project_topic': {'type': str},
    'response_file': {'type': str, 'is_path': True},
    'forcing_file': {'type': str, 'is_path': True, 'optional': True},
    'temperature': {'type': float, 'min': 0.0, 'max': 1.0},
    'max_sub_iterations': {'type': int, 'min': 1},
    'convergence_threshold': {'type': float, 'min': 0.0},
    'n_individuals': {'type': int, 'min': 1},
    'n_parallel': {'type': int, 'min': 1},
    'n_generations': {'type': int, 'min': 1},
    'llm_choice': {'type': str, 'choices': [
        'anthropic_sonnet', 'anthropic_haiku', "claude_4_sonnet", 'o3', 'o3_mini', 'gpt_4.1', 'o4_mini', 'o1_mini', 'groq', 'bedrock',
        'gemini', 'gemini_2.0_flash', 'gemini_2.5_pro', 'gemini_2.5_pro_exp_03_25', 'claude_3_7_sonnet', 'gpt_4o',
        'DeepCoder_14B_Preview_GGUF',
        'ollama:deepseek-r1:latest', 'ollama:gemma:latest', 'ollama:devstral:latest',
        'ollama:qwen3:30b-a3b', 'ollama:mistral:latest', 'ollama:qwen3:4b', 'ollama:gpt-oss:latest', 'ollama:gpt-oss:120b',
        'ollama:deepseek-r1:70b', 'ollama:qwen3:235b', 'openrouter:openai/gpt-5-chat', 'openrouter:openai/gpt-5'
    ]},
    'rag_choice': {'type': str, 'choices': [
        # Anthropic models
        'claude-3-5-sonnet-20241022', 'claude-3-5-haiku-20241022', 'claude-3-opus-20240229',
        'claude-sonnet-4-20250514', 'claude-3-7-sonnet-20250219',
        # AWS Bedrock models
        'anthropic.claude-3-5-sonnet-20240620-v1:0', 'anthropic.claude-3-haiku-20240307-v1:0',
        # Google models
        'gemini-2.5-pro', 'gemini-2.5-flash', 'gemini-2.0-flash', 'gemini-1.5-pro', 'gemini-1.5-flash',
        # Groq models
        'llama-3.3-70b-versatile', 'llama-3.1-8b-instant', 'mixtral-8x7b-32768', 'gemma2-9b-it',
        # OpenAI models
        'o1-mini', 'gpt-4o', 'gpt-4o-mini','gpt-5-mini','gpt-4.1-mini',
        # Ollama models (dynamic - examples)
        'ollama:gpt-oss:120b', 'ollama:gpt-oss:latest', 'ollama:deepseek-r1:70b', 'ollama:qwen3:235b',
        'ollama:deepseek-r1:latest', 'ollama:gemma:latest', 'ollama:devstral:latest', 'ollama:qwen3:30b-a3b',
        'ollama:mistral:latest', 'ollama:qwen3:4b'
    ]},
    'embed_choice': {'type': str, 'choices': ['azure', 'openai', 'ollama:mxbai-embed-large:latest']},
    'train_test_split': {'type': float, 'min': 0.0, 'max': 1.0, 'default': 1.0},
    'doc_store_dir': {'type': str, 'is_path': True, 'optional': True},
    'rag_search_engines': {'type': list, 'optional': True}
}


def validate_config(config: Dict[str, Any]) -> None:
    """Validate that all required parameters are present and have valid values."""
    # Check for missing required parameters
    missing_params = set(key for key, spec in PARAM_SPECS.items()
                         if not spec.get('optional', False)) - set(config.keys())
    if missing_params:
        raise ValueError(f"Missing required parameters in config file: {', '.join(missing_params)}")

    # Validate each parameter
    for param, value in config.items():
        if param not in PARAM_SPECS:
            raise ValueError(f"Unknown parameter in config file: {param}")
        spec = PARAM_SPECS[param]

        # Type check
        if not isinstance(value, spec['type']):
            raise ValueError(f"Parameter '{param}' must be of type {spec['type'].__name__}")

        # Path validation
        if spec.get('is_path', False) and value is not None and value != "":
            # Skip validation for optional parameters that are empty strings or None
            if spec.get('optional', False) and (value == "" or value is None):
                continue
            if not os.path.exists(value):
                if not spec.get('optional', False):
                    raise ValueError(f"File not found for parameter '{param}': {value}")

        # Range validation for numeric types
        if isinstance(value, (int, float)):
            if 'min' in spec and value < spec['min']:
                raise ValueError(f"Parameter '{param}' must be >= {spec['min']}")
            if 'max' in spec and value > spec['max']:
                raise ValueError(f"Parameter '{param}' must be <= {spec['max']}")

        # Choice validation
        if 'choices' in spec and value not in spec['choices']:
            raise ValueError(f"Parameter '{param}' must be one of: {', '.join(spec['choices'])}")


def main():
    parser = argparse.ArgumentParser(description='Run or resume genetic algorithm')
    parser.add_argument('--config', type=str, required=True, help='JSON configuration file containing all parameters')
    parser.add_argument('--resume', type=str, help='Population to resume (e.g., POPULATION_0033)')

    # Command line overrides (all optional)
    parser.add_argument('--project-topic', type=str, help='Project topic description')
    parser.add_argument('--response-file', type=str, help='Path to response data file.')
    parser.add_argument('--forcing-file', type=str, help='Path to forcing data file')
    parser.add_argument('--template-file', type=str, help='Template file for model generation')
    parser.add_argument('--temperature', type=float, help='Temperature parameter for model generation')
    parser.add_argument('--max-sub-iterations', type=int, help='Maximum number of sub-iterations')
    parser.add_argument('--convergence-threshold', type=float, help='Convergence threshold for stopping criterion')
    parser.add_argument('--n-individuals', type=int, help='Number of individuals in population')
    parser.add_argument('--n-generations', type=int, help='Number of generations to run')
    parser.add_argument('--llm-choice', type=str, help='LLM choice for model generation')
    parser.add_argument('--rag-choice', type=str, help='LLM choice for RAG operations')
    parser.add_argument('--embed-choice', type=str, help='Embedding model choice')
    parser.add_argument('--n-parallel', type=int, help='Number of parallel processes (>=1)')
    args = parser.parse_args()

    # Load and validate config file
    try:
        config = load_config_from_file(args.config)
        validate_config(config)
    except Exception as e:
        logging.error(f"Error in configuration: {e}")
        sys.exit(1)

    # Create a temporary config for validating command line overrides
    override_config = {}

    # Map command line args to config keys and validate each override
    arg_to_config = {
        'project_topic': 'project_topic',
        'response_file': 'response_file',
        'forcing_file': 'forcing_file',
        'temperature': 'temperature',
        'max_sub_iterations': 'max_sub_iterations',
        'convergence_threshold': 'convergence_threshold',
        'n_individuals': 'n_individuals',
        'n_parallel': 'n_parallel',
        'n_generations': 'n_generations',
        'llm_choice': 'llm_choice',
        'rag_choice': 'rag_choice',
        'embed_choice': 'embed_choice'
    }
    for arg_name, config_key in arg_to_config.items():
        value = getattr(args, arg_name.replace('-', '_'))
        if value is not None:
            # Validate this single override by merging over the base config
            try:
                validate_config({**config, **{config_key: value}})
                config[config_key] = value
            except Exception as e:
                logging.error(f"Invalid command line override for {arg_name}: {e}")
                sys.exit(1)

    # Set variables from config
    project_topic = config['project_topic']
    response_file = config['response_file']
    forcing_file = config['forcing_file']
    temperature = config['temperature']
    max_sub_iterations = config['max_sub_iterations']
    convergence_threshold = config['convergence_threshold']
    n_individuals = config['n_individuals']
    n_parallel = config['n_parallel']
    n_best = n_individuals  # n_best always equals n_individuals
    n_generations = config['n_generations']
    llm_choice = config['llm_choice']
    rag_choice = config['rag_choice']
    embed_choice = config['embed_choice']
    doc_store_dir = config.get('doc_store_dir')  # Can be None to disable RAG
    # Treat empty string as None
    if doc_store_dir == "":
        doc_store_dir = None
    train_test_split = config.get('train_test_split', 1.0)  # Default to 1.0 if not specified
    # Get RAG search engines with default of all three if not specified
    rag_search_engines = config.get('rag_search_engines', ["semantic_scholar", "serper", "rag"])
    resume_population = args.resume if args.resume else None

    # Create data report before initializing population
    base_name, _ = os.path.splitext(response_file)
    report_file = f"{base_name}_report.json"
    create_data_report(response_file)  # Note: may need to update data_report.py to handle both files

    # NEW: create a Manager once; pass to any Pools so workers share the pacer
    shared_mgr = multiprocessing.Manager()

    if resume_population:
        print(f"Resuming population {resume_population}")
        population_dir = os.path.join('POPULATIONS', resume_population)
        if not os.path.exists(population_dir):
            raise ValueError(f"Population {resume_population} does not exist.")
        population_metadata = load_metadata(os.path.join(population_dir, 'population_metadata.json'))

        # When resuming, start from the last generation number and allow n_generations more iterations
        last_generation = len(population_metadata['generations'])
        n_generations = last_generation + n_generations  # Extend the total generations
        start_generation = last_generation

        # Convert any numpy int64 objective values to float
        current_best_performers = []
        for performer in population_metadata.get('current_best_performers', []):
            objective_value = float(performer['objective_value']) if performer.get('objective_value') is not None else None
            current_best_performers.append({
                "individual": performer['individual'],
                "objective_value": objective_value
            })
        individuals = [ind['individual'] for ind in current_best_performers]
    else:
        population_number = get_next_population_number()
        population_dir = os.path.join('POPULATIONS', f'POPULATION_{population_number:04d}')
        os.makedirs(population_dir, exist_ok=True)
        os.makedirs(os.path.join(population_dir, 'BROKEN'), exist_ok=True)
        os.makedirs(os.path.join(population_dir, 'CULLED'), exist_ok=True)

        population_metadata = {
            "start_time": datetime.datetime.now().isoformat(),
            "n_individuals": n_individuals,
            "n_parallel": n_parallel,
            "n_generations": n_generations,
            "n_best": n_best,
            "project_topic": project_topic,
            "response_file": response_file,
            "forcing_file": forcing_file,
            "report_file": report_file,
            "temperature": temperature,
            "max_sub_iterations": max_sub_iterations,
            "convergence_threshold": convergence_threshold,
            "train_test_split": train_test_split,
            "llm_choice": llm_choice,
            "rag_choice": rag_choice,
            "embed_choice": embed_choice,
            "doc_store_dir": doc_store_dir,
            "rag_search_engines": rag_search_engines,
            "generations": [],
            "current_best_performers": []
        }
        start_generation = 0
        individuals = []
        current_best_performers = []

        # Save initial metadata
        save_metadata(os.path.join(population_dir, 'population_metadata.json'), population_metadata)


    if doc_store_dir:
        print("Preparing RAG index (parent process)...")
        rag_prepare(doc_store_dir, population_dir)
    else:
        print("RAG disabled - doc_store_dir is None")


    # Initialize population if there are no current best performers
    if not current_best_performers:
        print("Initializing population")
        individuals = initialize_population(
            population_dir, n_individuals, n_parallel, project_topic, response_file,
            forcing_file, report_file, temperature, max_sub_iterations, llm_choice,
            train_test_split, shared_mgr=shared_mgr 
        )

        best_individuals, culled_individuals, broken_individuals = evolve_population(population_dir, individuals, n_best)

        # Move culled and broken individuals
        for individual in culled_individuals:
            move_individual(population_dir, individual, 'CULLED')
        for individual in broken_individuals:
            move_individual(population_dir, individual, 'BROKEN')

        # Update metadata for the first generation
        current_best_performers = []
        for individual in best_individuals:
            report_data = read_model_report(os.path.join(population_dir, individual))
            objective_value = float(report_data.get('objective_value')) if report_data.get('objective_value') is not None else None
            current_best_performers.append({
                "individual": individual,
                "objective_value": objective_value
            })

        generation_data = {
            "generation_number": 1,
            "best_individuals": current_best_performers,
            "culled_individuals": culled_individuals,
            "broken_individuals": broken_individuals
        }
        population_metadata["generations"].append(generation_data)
        population_metadata["current_best_performers"] = current_best_performers
        save_metadata(os.path.join(population_dir, 'population_metadata.json'), population_metadata)

        # Check for convergence after first generation
        if current_best_performers:
            best_objective = current_best_performers[0].get('objective_value')
            if best_objective is not None:
                print(f"Initial best objective value: {best_objective}")
                if best_objective < convergence_threshold:
                    print(f"\033[92mConvergence achieved in first generation. Best objective value: {best_objective}\033[0m")
                    population_metadata["converged"] = True
                    population_metadata["convergence_generation"] = 1
                    population_metadata["end_time"] = datetime.datetime.now().isoformat()
                    population_metadata["total_runtime"] = (datetime.datetime.fromisoformat(population_metadata["end_time"]) -
                                                            datetime.datetime.fromisoformat(population_metadata["start_time"])).total_seconds()
                    save_metadata(os.path.join(population_dir, 'population_metadata.json'), population_metadata)
                    return
                else:
                    print(f"\033[93mConvergence NOT YET. Initial best objective value: {best_objective}\033[0m")

        start_generation = 1

    for generation in range(start_generation, n_generations):
        print(f"Starting Generation {generation + 1}")
        time.sleep(0.1)

        # Create new generation
        new_individuals = create_new_generation(
            population_dir, [ind['individual'] for ind in current_best_performers],
            n_individuals, project_topic, response_file, forcing_file, report_file,
            temperature, max_sub_iterations, llm_choice, train_test_split
        )
        print(f"\033[93mCreated {len(new_individuals)} new individuals\033[0m")
        if not new_individuals:
            logging.error("Failed to create new individuals. Stopping the algorithm.")
            break

        # Update lineage for new individuals
        for child in new_individuals:
            parent = next((ind['individual'] for ind in current_best_performers if ind['individual'] == child), None)
            update_lineage(population_dir, child, parent)

        # Evolve population (including current best performers)
        all_individuals = new_individuals + [ind['individual'] for ind in current_best_performers]
        best_individuals, culled_individuals, broken_individuals = evolve_population(population_dir, all_individuals, n_best)
        print(f"Evolution results: {len(best_individuals)} best, {len(culled_individuals)} culled, {len(broken_individuals)} broken")

        # Move broken and culled individuals
        for individual in broken_individuals:
            move_individual(population_dir, individual, 'BROKEN')
        for individual in culled_individuals:
            move_individual(population_dir, individual, 'CULLED')

        # Update current_best_performers with objective values
        current_best_performers = []
        for individual in best_individuals:
            report_data = read_model_report(os.path.join(population_dir, individual))
            objective_value = float(report_data.get('objective_value')) if report_data.get('objective_value') is not None else None
            current_best_performers.append({
                "individual": individual,
                "objective_value": objective_value
            })

        # Load latest metadata and update it
        population_metadata = load_metadata(os.path.join(population_dir, 'population_metadata.json'))
        generation_data = {
            "generation_number": generation + 1,  # +1 because generation starts from 0
            "best_individuals": current_best_performers,
            "culled_individuals": culled_individuals,
            "broken_individuals": broken_individuals
        }
        population_metadata["generations"].append(generation_data)
        population_metadata["current_best_performers"] = current_best_performers

        # Update population_metadata.json after each generation
        save_metadata(os.path.join(population_dir, 'population_metadata.json'), population_metadata)

        # Check for convergence
        if best_individuals:
            report_data = read_model_report(os.path.join(population_dir, best_individuals[0]))
            best_objective = float(report_data.get('objective_value')) if report_data.get('objective_value') is not None else None
            if best_objective is not None:
                print(f"Best objective value: {best_objective}")
                if best_objective < convergence_threshold:
                    print(f"\033[92mConvergence achieved. Best objective value: {best_objective}\033[0m")
                    print("\033[92mGenetic algorithm completed.\033[0m")
                    population_metadata["converged"] = True
                    population_metadata["convergence_generation"] = generation + 1  # +1 for the same reason as above
                    break
            else:
                print(f"\033[93mConvergence NOT YET. Best objective value: {best_objective}\033[0m")
        else:
            logging.warning("No valid objective value for the best individual")

    # Save final metadata
    if "converged" not in population_metadata:
        population_metadata["converged"] = False
        print("\033[91mMAXIMUM Generations Reached - Genetic algorithm completed.\033[0m")
    else:
        print("CONVERGENCE Achieved - Genetic algorithm completed.")
    population_metadata["end_time"] = datetime.datetime.now().isoformat()
    population_metadata["total_runtime"] = (datetime.datetime.fromisoformat(population_metadata["end_time"]) -
                                            datetime.datetime.fromisoformat(population_metadata["start_time"])).total_seconds()
    save_metadata(os.path.join(population_dir, 'population_metadata.json'), population_metadata)


if __name__ == "__main__":
    import multiprocessing as mp
    try:
        mp.set_start_method("spawn", force=True)  # safer across OS/backends
    except RuntimeError:
        pass
    main()
