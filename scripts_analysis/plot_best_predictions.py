import os
import json
import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.image as mpimg

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
            
            if not llm_choice:
                print(f"Warning: No llm_choice found in {metadata_path}")
                continue
            response_file = metadata.get('response_file')
            if 'NPZ' in response_file:
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
    
    # Filter out LLMs with no valid individuals
    valid_llms = {llm: data for llm, data in best_by_llm.items() if data[0] is not None}
    
    if not valid_llms:
        raise ValueError("No valid individuals found in any population")
    
    return valid_llms

def get_historical_data(model_data):
    """
    Extract historical data from model report.
    
    Args:
        model_data (dict): Model report data
    
    Returns:
        dict: Historical data for COTS and coral cover
    """
    # Get the last successful iteration
    success = None
    for iter_num in sorted(model_data['iterations'].keys(), key=int, reverse=True):
        iteration = model_data['iterations'][iter_num]
        if iteration['status'] == 'SUCCESS' and 'plot_data' in iteration:
            success = iteration
            break
    
    if success is None:
        raise ValueError("No successful iterations with plot data found")
    
    plot_data = success['plot_data']
    return {
        'cots_pred': plot_data['cots_pred']['Observed'],
        'fast_pred': plot_data['fast_pred']['Observed'],
        'slow_pred': plot_data['slow_pred']['Observed']
    }

def get_plot_data(model_path):
    """
    Get plot data from a model's report.
    
    Args:
        model_path (str): Path to model directory
    
    Returns:
        dict: Plot data containing modeled values
    """
    with open(os.path.join(model_path, 'model_report.json'), 'r') as f:
        model_data = json.load(f)
    
    # Get the last successful iteration
    success = None
    for iter_num in sorted(model_data['iterations'].keys(), key=int, reverse=True):
        iteration = model_data['iterations'][iter_num]
        if iteration['status'] == 'SUCCESS' and 'plot_data' in iteration:
            success = iteration
            break
    
    if success is None:
        raise ValueError(f"No successful iterations with plot data found in {model_path}")
    
    return success['plot_data']

def calculate_nmse(observed, predicted):
    """
    Calculate Normalized Mean Square Error (NMSE) for a single variable.
    
    Args:
        observed: Array of observed values
        predicted: Array of predicted values
    
    Returns:
        float: NMSE value
    """
    # Convert inputs to numpy arrays
    observed = np.array(observed)
    predicted = np.array(predicted)
    
    sigma = np.std(observed, ddof=1)
    
    if sigma != 0:
        return np.mean(((observed - predicted) / sigma) ** 2)
    else:
        return np.mean((observed - predicted) ** 2)

def create_comparison_plots(best_by_llm):
    """
    Create three subplots comparing predictions from each LLM.
    """
    # LLM name mapping
    llm_name_map = {
        'anthropic_sonnet': 'claude 3.6',
        'claude_3_7_sonnet': 'claude 3.7',
        'o3_mini':'o3-mini',
        'gemini_2.0_flash':'gemini-2.0-flash',
        'gemini_2.5_pro_exp_03_25':'gemini-2.5-pro',
        'openrouter:openai/gpt-5' : "GPT-5",
        'openrouter:anthropic/claude-sonnet-4.5' : "Sonnet-4.5",
        'openrouter:google/gemini-2.5-pro' : "Gemini-2.5"
    }
       
    # Read Jacob's Excel data
    jacob_data = pd.read_excel('FROM_JACOB/TIMESERIES/modelOutputs.xlsx')
    
    # Create human expert data structure to match the format expected by the plotting code
    human_plot_data = {
        'cots_pred': {'Modeled': jacob_data['CoTS_age2p'].tolist()},
        'fast_pred': {'Modeled': jacob_data['fastCoral'].tolist()},
        'slow_pred': {'Modeled': jacob_data['slowCoral'].tolist()}
    }
    
    # Calculate objective value for human expert model (mean NMSE across all variables)
    human_objective = 0  # This will be calculated later when we have historical data
    human_data = ('FROM_JACOB/TIMESERIES/modelOutputs.xlsx', human_objective)
    
    # Get historical data from first model
    first_path = next(iter(best_by_llm.values()))[0]
    with open(os.path.join(first_path, 'model_report.json'), 'r') as f:
        first_model_data = json.load(f)
    historical_data = get_historical_data(first_model_data)
    years = list(range(1980, 2006))
    
    # Plot data for each LLM and human expert model
    all_models = list(best_by_llm.items())
    all_models.append(('human expert', human_data))
    
    print("\nNMSE values for each model:")
    print("-" * 50)
    
        # Define colorblind-friendly color palette
    color_cycle = [
        '#CC79A7',  # Light pink
        '#0173B2',  # Blue
        '#DE8F05',  # Orange
        '#CC78BC',  # Pink
        '#56B4E9',  # Light blue
        '#029E73',  # Green
        '#009E73',  # Teal
        '#CC79A7'   # Light pink
    ]

    # Set figure size and DPI for better quality
    plt.figure(figsize=(12,16), dpi=300)

    # Create subplots with more space between them
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12,16))
    fig.subplots_adjust(hspace=0.4)

    

    for llm_choice, (path, objective) in all_models:
        # Map LLM name if it exists in mapping
        display_name = llm_name_map.get(llm_choice, llm_choice)
        
        if display_name == 'human expert':
            plot_data = human_plot_data
            # Calculate NMSE for human expert model
            cots_nmse = calculate_nmse(historical_data['cots_pred'], plot_data['cots_pred']['Modeled'])
            fast_nmse = calculate_nmse(historical_data['fast_pred'], plot_data['fast_pred']['Modeled'])
            slow_nmse = calculate_nmse(historical_data['slow_pred'], plot_data['slow_pred']['Modeled'])
            human_objective = np.mean([cots_nmse, fast_nmse, slow_nmse])
            
            print(f"\n{display_name}:")
            print(f" COTS NMSE: {cots_nmse:.4f}")
            print(f" Fast Coral NMSE: {fast_nmse:.4f}")
            print(f" Slow Coral NMSE: {slow_nmse:.4f}")
            print(f" Overall Mean NMSE: {human_objective:.4f}")
        else:
            plot_data = get_plot_data(path)
            # Calculate NMSE for each variable
            cots_nmse = calculate_nmse(historical_data['cots_pred'], plot_data['cots_pred']['Modeled'])
            fast_nmse = calculate_nmse(historical_data['fast_pred'], plot_data['fast_pred']['Modeled'])
            slow_nmse = calculate_nmse(historical_data['slow_pred'], plot_data['slow_pred']['Modeled'])
            mean_nmse = np.mean([cots_nmse, fast_nmse, slow_nmse])
            
            print(f"\n{display_name}:")
            print(f" COTS NMSE: {cots_nmse:.4f}")
            print(f" Fast Coral NMSE: {fast_nmse:.4f}")
            print(f" Slow Coral NMSE: {slow_nmse:.4f}")
            print(f" Overall Mean NMSE: {mean_nmse:.4f}")
        
        # Create labels with variable-specific NMSE values
        cots_label = f"{display_name} (NMSE: {cots_nmse:.4f})"
        fast_label = f"{display_name} (NMSE: {fast_nmse:.4f})"
        slow_label = f"{display_name} (NMSE: {slow_nmse:.4f})"
        
        # Get color from colorblind-friendly palette
        color = color_cycle[len(ax1.lines) % len(color_cycle)]
        
        # COTS predictions
        ax1.plot(years, plot_data['cots_pred']['Modeled'],
                label=cots_label, color=color, alpha=0.8, linewidth=2.5)
        if llm_choice == list(best_by_llm.keys())[0]:  # Only plot observed once
            ax1.scatter(years, historical_data['cots_pred'],
                    label='Observed', color='#333333', s=80, alpha=0.8, zorder=5)
        
        # Fast-growing coral predictions
        ax2.plot(years, plot_data['fast_pred']['Modeled'],
                label=fast_label, color=color, alpha=0.8, linewidth=2.5)
        if llm_choice == list(best_by_llm.keys())[0]:
            ax2.scatter(years, historical_data['fast_pred'],
                    label='Observed', color='#333333', s=80, alpha=0.8, zorder=5)
        
        # Slow-growing coral predictions
        ax3.plot(years, plot_data['slow_pred']['Modeled'],
                label=slow_label, color=color, alpha=0.8, linewidth=2.5)
        if llm_choice == list(best_by_llm.keys())[0]:
            ax3.scatter(years, historical_data['slow_pred'],
                    label='Observed', color='#333333', s=80, alpha=0.8, zorder=5)

    # Load PNG icons
    cots_icon = mpimg.imread('Figures/cots.png')
    fast_coral_icon = mpimg.imread('Figures/fast_coral.drawio.png')
    slow_coral_icon = mpimg.imread('Figures/slow_coral.drawio.png')
    
    # Customize plots with larger, more legible elements
    for ax, title, ylabel, icon in zip([ax1, ax2, ax3],
                                ['Crown-of-Thorns Starfish Abundance',
                                'Fast-Growing Coral Cover',
                                'Slow-Growing Coral Cover'],
                                ['Abundance (individuals/m2)', 'Cover (%)', 'Cover (%)'],
                                [cots_icon, fast_coral_icon, slow_coral_icon]):
        
        # Set titles and labels with larger font sizes
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.set_ylabel(ylabel, fontsize=14, fontweight='bold')
        
        # Add icon to top right corner (smaller size)
        imagebox = OffsetImage(icon, zoom=0.10)  # Reduced from 0.15 to 0.10
        ab = AnnotationBbox(imagebox, (0.95, 0.95),
                           xycoords='axes fraction',
                           frameon=False,
                           box_alignment=(1, 1))
        ax.add_artist(ab)
        
        # Customize legend
        ax.legend(fontsize=12, frameon=False, fancybox=False, shadow=False, 
                framealpha=0.9, loc='best')
        
        # Customize tick labels
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.tick_params(axis='both', which='minor', labelsize=10)
        
        # Set spine colors and width
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)
            spine.set_color('#333333')

    # Set x-axis label only for bottom subplot
    ax3.set_xlabel('Year', fontsize=14, fontweight='bold')
        # Reorder legend: Observed, Human Expert, then everything else alphabetically
    for ax in [ax1, ax2, ax3]:
        handles, labels = ax.get_legend_handles_labels()
        
        # Separate items into categories
        observed_items = []
        human_expert_items = []
        other_items = []
        
        for i, label in enumerate(labels):
            if 'Observed' in label:
                observed_items.append((handles[i], labels[i]))
            elif 'human expert' in label:
                human_expert_items.append((handles[i], labels[i]))
            else:
                other_items.append((handles[i], labels[i]))
        
        # Sort other items alphabetically by label
        other_items.sort(key=lambda x: x[1])
        
        # Combine in desired order
        ordered_handles = []
        ordered_labels = []
        
        # Add Observed first
        for handle, label in observed_items:
            ordered_handles.append(handle)
            ordered_labels.append(label)
        
        # Add Human Expert second  
        for handle, label in human_expert_items:
            ordered_handles.append(handle)
            ordered_labels.append(label)
        
        # Add everything else alphabetically
        for handle, label in other_items:
            ordered_handles.append(handle)
            ordered_labels.append(label)
        
        # Create new legend
        ax.legend(ordered_handles, ordered_labels, fontsize=12, frameon=False,
          fancybox=False, shadow=False, framealpha=0.9, 
          bbox_to_anchor=(1.05, 1), loc='upper left')
    # Adjust layout and save
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.35)  # Add more space between subplots
    os.makedirs('Figures', exist_ok=True)
    plt.savefig('Figures/llm_predictions_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig('Figures/llm_predictions_comparison.svg', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    try:
        # Find best individuals for each llm
        best_by_llm = find_best_individuals_by_llm()
        
        # Print summary
        print("\nBest performers by LLM:")
        for llm_choice, (path, objective) in best_by_llm.items():
            print(f"\n{llm_choice}:")
            print(f"Path: {path}")
            print(f"Objective value: {objective}")
        
        # Create comparison plots
        create_comparison_plots(best_by_llm)
        print("\nCreated comparison plot at: Figures/llm_predictions_comparison.png")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main())


