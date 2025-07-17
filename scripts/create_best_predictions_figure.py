import os
import json
import glob
from PIL import Image

def find_best_individuals_by_llm():
    """
    Search through all population_metadata.json files to find the best individual
    for each llm_choice.
    
    Returns:
        dict: {llm_choice: (best_individual_path, best_objective_value)}
    """
    best_by_llm = {}  # {llm_choice: (path, objective)}
    
    # Search for all population_metadata.json files
    for metadata_path in glob.glob('POPULATIONS/POPULATION_*/population_metadata.json'):
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Get the population directory and llm choice
            pop_dir = os.path.dirname(metadata_path)
            llm_choice = metadata.get('llm_choice')
            # print(pop_dir)
            # print(llm_choice)
            if not llm_choice:
                print(f"Warning: No llm_choice found in {metadata_path}")
                continue
            
            # Initialize best for this llm if not seen before
            if llm_choice not in best_by_llm:
                best_by_llm[llm_choice] = (None, float('inf'))
            
            # Check current best performers
            if 'current_best_performers' in metadata:
                for performer in metadata['current_best_performers']:
                    if performer['objective_value'] < best_by_llm[llm_choice][1]:
                        best_by_llm[llm_choice] = (
                            os.path.join(pop_dir, performer['individual']),
                            performer['objective_value']
                        )
        except Exception as e:
            print(f"Error processing {metadata_path}: {str(e)}")
            continue
    
    # Validate we found at least one individual
    if not best_by_llm:
        raise ValueError("No valid individuals found in any population")
    
    # Validate we found individuals for each llm
    llms_to_remove = []
    
    for llm, (path, *rest) in best_by_llm.items():
        try:
            if path is None:
                raise ValueError(f"No valid individual found for llm_choice: {llm}")
            
            # Your existing processing logic here
            # process_path(path, *rest)
            
        except ValueError as e:
            print(f"Warning: {e}. Removing {llm} from best_by_llm.")
            llms_to_remove.append(llm)
            continue  # Skip to next LLM
    
    # Remove problematic LLMs from the dictionary
    for llm in llms_to_remove:
        del best_by_llm[llm]
    
    return best_by_llm

def create_combined_figure(individual_path, llm_choice):
    """
    Load and combine the three prediction PNGs from the best individual's directory.
    
    Args:
        individual_path: Path to the best individual's directory
        
    Returns:
        str: Path to the created figure
    """
    # Ensure output directory exists
    os.makedirs('Figures', exist_ok=True)
    
    # Paths to the three PNG files
    png_files = [
        os.path.join(individual_path, 'cots_pred_comparison.png'),
        os.path.join(individual_path, 'fast_pred_comparison.png'),
        os.path.join(individual_path, 'slow_pred_comparison.png')
    ]
    
    # Check all files exist
    for png_file in png_files:
        if not os.path.exists(png_file):
            raise FileNotFoundError(f"Missing required PNG file: {png_file}")
    
    # Load all images
    images = [Image.open(f) for f in png_files]
    
    # Get dimensions
    widths, heights = zip(*(i.size for i in images))
    total_width = sum(widths)
    max_height = max(heights)
    
    # Create new image in landscape orientation
    combined = Image.new('RGB', (total_width, max_height), 'white')
    
    # Paste images side by side
    x_offset = 0
    for img in images:
        combined.paste(img, (x_offset, 0))
        x_offset += img.size[0]
    
    # Save combined figure
    output_path = f'Figures/combined_predictions_{llm_choice}.png'
    combined.save(output_path)
    
    # Close all images
    for img in images:
        img.close()
    
    return output_path


def main():
    try:
        # Find best individuals for each llm
        best_by_llm = find_best_individuals_by_llm()
        
        # Create combined figures for each llm
        for llm_choice, (best_path, best_objective) in best_by_llm.items():
            print(f"\nProcessing {llm_choice}:")
            print(f"Best individual at {best_path} with objective value {best_objective}")
            
            figure_path = create_combined_figure(best_path, llm_choice)
            print(f"Created combined figure at {figure_path}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main())
