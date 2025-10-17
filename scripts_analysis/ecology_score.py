import argparse
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from scripts_analysis.evaluate_ecological_characteristics import evaluate_individual

def load_metadata(file_path):
    """Load JSON metadata from file."""
    with open(file_path, 'r') as f:
        return json.load(f)

def get_all_models(population_metadata):
    """
    Extract all model IDs from population metadata.

    Returns a set of strings. Robust to:
      - current_best_performers entries that are not dicts
      - missing 'individual' keys
      - generations entries missing 'culled_individuals'
    """
    models = set()

    # Add current best performers (top-level directory)
    for performer in population_metadata.get('current_best_performers', []):
        if isinstance(performer, dict):
            indiv = performer.get('individual')
            if isinstance(indiv, str) and indiv:
                models.add(indiv)

    # Add all culled individuals
    for generation in population_metadata.get('generations', []):
        culled = generation.get('culled_individuals', [])
        for individual in culled:
            if isinstance(individual, str) and individual:
                models.add(individual)

    return models

def detect_model_type(population_dir: Path) -> str:
    """
    Determine model type by checking the 'report_file' field in population_metadata.json.
    Returns "NPZ" if "NPZ" is in the report file path, otherwise returns "COTS".
    """
    metadata_path = population_dir / "population_metadata.json"
    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        report_file = metadata.get("report_file", "")
        if "NPZ" in str(report_file).upper():
            return "NPZ"
        else:
            return "COTS"
    except Exception as e:
        print(f"[DEBUG] Error reading population metadata: {e}")
        # Default to NPZ if metadata can't be read
        return "NPZ"

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate ecological scores for NPZ populations under POPULATIONS/."
    )
    parser.add_argument(
        "-f", "--overwrite",
        action="store_true",
        help="Overwrite existing scores.json for individuals (recompute instead of skipping)."
    )
    parser.add_argument(
        "-j", "--jobs",
        type=int,
        default=1,
        help="Number of parallel processes for processing populations (default: 1 for sequential)."
    )
    return parser.parse_args()

def process_population(population_dir, overwrite):
    """
    Process a single population directory.
    Returns a dict with stats: models_count, success_count, error_count.
    """
    stats = {'models': 0, 'success': 0, 'errors': 0}
    
    print(f"\n=== Population: {population_dir.name} (NPZ) ===")
    metadata_path = population_dir / 'population_metadata.json'
    
    try:
        metadata = load_metadata(str(metadata_path))
    except Exception as e:
        print(f"Error loading metadata for {population_dir.name}: {e}")
        stats['errors'] += 1
        return stats

    try:
        models = get_all_models(metadata)
    except Exception as e:
        print(f"Error extracting models for {population_dir.name}: {e}")
        stats['errors'] += 1
        return stats

    if not models:
        print("No models found to evaluate in this population.")
        return stats

    print(f"Found {len(models)} models to evaluate in {population_dir.name}")
    stats['models'] = len(models)

    # Safely extract the list of best performers
    try:
        best_performers = [
            p.get('individual')
            for p in metadata.get('current_best_performers', [])
            if isinstance(p, dict) and isinstance(p.get('individual'), str)
        ]
    except Exception:
        best_performers = []

    # Evaluate each model
    for i, model_id in enumerate(sorted(models), 1):
        print(f"\nEvaluating model {i}/{len(models)}: {model_id}")
        try:
            # Decide model path based on whether it's a best performer (top-level) or culled
            if model_id in best_performers:
                model_path = population_dir / model_id
            else:
                model_path = population_dir / 'CULLED' / model_id

            result = evaluate_individual(str(model_path), overwrite=overwrite)
            if result is None:
                print(f"Error evaluating {model_id}: evaluation returned None")
                stats['errors'] += 1
            else:
                if overwrite:
                    print(f"Overwrote and evaluated {model_id}")
                else:
                    print(f"Successfully evaluated {model_id}")
                stats['success'] += 1
        except Exception as e:
            print(f"Error evaluating {model_id}: {str(e)}")
            stats['errors'] += 1
    
    return stats


def main():
    args = parse_args()

    populations_root = Path('POPULATIONS')
    if not populations_root.exists():
        print(f"POPULATIONS directory not found at: {populations_root.resolve()}")
        return

    # Discover all candidate population directories (must contain population_metadata.json)
    all_population_dirs = sorted([
        p for p in populations_root.iterdir()
        if p.is_dir() and (p / 'population_metadata.json').exists()
    ])
    if not all_population_dirs:
        print("No population directories with population_metadata.json were found.")
        return

    # Filter to NPZ populations
    npz_population_dirs = [p for p in all_population_dirs if detect_model_type(p) == "NPZ"]
    print(f"Discovered {len(all_population_dirs)} populations total; processing {len(npz_population_dirs)} NPZ populations.")
    
    if args.jobs > 1:
        print(f"Using {args.jobs} parallel processes for population processing.")
    else:
        print("Using sequential processing (1 process).")

    total_models = 0
    total_success = 0
    total_errors = 0

    # Process populations in parallel or sequentially based on args.jobs
    if args.jobs > 1:
        with ProcessPoolExecutor(max_workers=args.jobs) as executor:
            # Submit all populations for processing
            future_to_pop = {
                executor.submit(process_population, pop_dir, args.overwrite): pop_dir
                for pop_dir in npz_population_dirs
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_pop):
                pop_dir = future_to_pop[future]
                try:
                    stats = future.result()
                    total_models += stats['models']
                    total_success += stats['success']
                    total_errors += stats['errors']
                except Exception as e:
                    print(f"Error processing population {pop_dir.name}: {e}")
                    total_errors += 1
    else:
        # Sequential processing
        for pop_idx, population_dir in enumerate(npz_population_dirs, start=1):
            print(f"\n=== [{pop_idx}/{len(npz_population_dirs)}] Processing population ===")
            stats = process_population(population_dir, args.overwrite)
            total_models += stats['models']
            total_success += stats['success']
            total_errors += stats['errors']

    print("\n=== Summary ===")
    print(f"NPZ populations processed: {len(npz_population_dirs)}")
    print(f"Models attempted: {total_models}")
    print(f"Successful evaluations: {total_success}")
    print(f"Errors: {total_errors}")

if __name__ == "__main__":
    main()
