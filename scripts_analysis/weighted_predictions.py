import os
import json
import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def find_all_individuals():
    """
    Search through all individuals in the Populations directory.
    
    Returns:
        list: [(individual_path, objective_value, llm_choice)]
    """
    all_individuals = []
    
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
            
            # Look through all individuals in this population
            individual_paths = glob.glob(os.path.join(pop_dir, 'INDIVIDUAL_*'))
            for ind_path in individual_paths:
                # Check if model_report.json exists
                report_path = os.path.join(ind_path, 'model_report.json')
                if os.path.exists(report_path):
                    try:
                        with open(report_path, 'r') as f:
                            model_data = json.load(f)
                        
                        # Calculate weighted NMSE for this individual
                        plot_data = get_plot_data(ind_path)
                        historical = get_historical_data(model_data)
                        
                        cots_nmse = calculate_nmse(historical['cots_pred'], 
                                                 plot_data['cots_pred']['Modeled'])
                        fast_nmse = calculate_nmse(historical['fast_pred'], 
                                                 plot_data['fast_pred']['Modeled'])
                        slow_nmse = calculate_nmse(historical['slow_pred'], 
                                                 plot_data['slow_pred']['Modeled'])
                        
                        # Weight COTS double in the objective value
                        weighted_objective = (2 * cots_nmse + fast_nmse + slow_nmse) / 4
                        
                        all_individuals.append((ind_path, weighted_objective, llm_choice))
                    except Exception as e:
                        print(f"Error processing {report_path}: {str(e)}")
                        continue
                        
        except Exception as e:
            print(f"Error processing {metadata_path}: {str(e)}")
            continue
    
    # Validate we found at least one individual
    if not all_individuals:
        raise ValueError("No valid individuals found in any population")
    
    return all_individuals

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
    
    sigma = np.std(observed)
    
    if sigma != 0:
        return np.mean(((observed - predicted) / sigma) ** 2)
    else:
        return np.mean((observed - predicted) ** 2)

def create_comparison_plots(best_individuals):
    """
    Create three subplots comparing predictions from best individuals.
    """
    # Set up the figure
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))
    fig.suptitle('Model Predictions with Weighted NMSE (COTS Double Weight)')
    
    # Set up color cycle for the models
    color_cycle = plt.cm.tab10(np.linspace(0, 1, 10))
    
    # Get historical data from first model
    first_path = best_individuals[0][0]
    with open(os.path.join(first_path, 'model_report.json'), 'r') as f:
        first_model_data = json.load(f)
    historical_data = get_historical_data(first_model_data)
    years = list(range(1980, 2006))
    
    print("\nWeighted NMSE values for best individuals:")
    print("-" * 50)
    
    for path, objective, llm_choice in best_individuals:
        plot_data = get_plot_data(path)
        
        # Calculate NMSE for each variable
        cots_nmse = calculate_nmse(historical_data['cots_pred'], plot_data['cots_pred']['Modeled'])
        fast_nmse = calculate_nmse(historical_data['fast_pred'], plot_data['fast_pred']['Modeled'])
        slow_nmse = calculate_nmse(historical_data['slow_pred'], plot_data['slow_pred']['Modeled'])
        weighted_nmse = (2 * cots_nmse + fast_nmse + slow_nmse) / 4
        
        print(f"\n{llm_choice}:")
        print(f"  COTS NMSE:           {cots_nmse:.4f} (weighted x2)")
        print(f"  Fast Coral NMSE:     {fast_nmse:.4f}")
        print(f"  Slow Coral NMSE:     {slow_nmse:.4f}")
        print(f"  Weighted Mean NMSE:  {weighted_nmse:.4f}")
        
        # Create labels with variable-specific NMSE values
        cots_label = f"{llm_choice} (NMSE: {cots_nmse:.4f})"
        fast_label = f"{llm_choice} (NMSE: {fast_nmse:.4f})"
        slow_label = f"{llm_choice} (NMSE: {slow_nmse:.4f})"
        color = color_cycle[len(ax1.lines) % len(color_cycle)]
        
        # Plot predictions
        ax1.plot(years, plot_data['cots_pred']['Modeled'], 
                label=cots_label, color=color, alpha=0.7)
        if path == first_path:  # Only plot observed once
            ax1.scatter(years, historical_data['cots_pred'], 
                       label='Observed', color='grey', s=50)
        
        ax2.plot(years, plot_data['fast_pred']['Modeled'], 
                label=fast_label, color=color, alpha=0.7)
        if path == first_path:
            ax2.scatter(years, historical_data['fast_pred'], 
                       label='Observed', color='grey', s=50)
        
        ax3.plot(years, plot_data['slow_pred']['Modeled'], 
                label=slow_label, color=color, alpha=0.7)
        if path == first_path:
            ax3.scatter(years, historical_data['slow_pred'], 
                       label='Observed', color='grey', s=50)
    
    # Customize plots
    ax1.set_title('Crown-of-Thorns Starfish Abundance (Double Weight in NMSE)')
    ax1.set_ylabel('Abundance')
    ax1.legend()
    
    ax2.set_title('Fast-Growing Coral Cover')
    ax2.set_ylabel('Cover (%)')
    ax2.legend()
    
    ax3.set_title('Slow-Growing Coral Cover')
    ax3.set_xlabel('Year')
    ax3.set_ylabel('Cover (%)')
    ax3.legend()
    
    # Adjust layout and save
    plt.tight_layout()
    os.makedirs('Figures', exist_ok=True)
    plt.savefig('Figures/weighted_predictions_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    try:
        # Find all individuals and calculate their weighted NMSE
        all_individuals = find_all_individuals()
        
        # Sort by weighted objective value and get top 5
        best_individuals = sorted(all_individuals, key=lambda x: x[1])[:5]
        
        # Print summary of best performers
        print("\nTop 5 performers by weighted NMSE:")
        for path, objective, llm_choice in best_individuals:
            print(f"\n{llm_choice}:")
            print(f"Path: {path}")
            print(f"Weighted objective value: {objective:.4f}")
        
        # Create comparison plots
        create_comparison_plots(best_individuals)
        print("\nCreated comparison plot at: Figures/weighted_predictions_comparison.png")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main())
