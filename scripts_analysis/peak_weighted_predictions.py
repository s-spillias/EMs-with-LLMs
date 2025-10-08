import os
import json
import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

def calculate_peak_metrics(observed, predicted):
    """
    Calculate metrics focused on peak prediction accuracy.
    
    Args:
        observed: Array of observed values
        predicted: Array of predicted values
    
    Returns:
        dict: Dictionary containing peak-related metrics
    """
    # Convert to numpy arrays
    observed = np.array(observed)
    predicted = np.array(predicted)
    
    # Find peaks in observed data
    obs_peaks, _ = find_peaks(observed, height=np.mean(observed) + 0.5*np.std(observed))
    pred_peaks, _ = find_peaks(predicted, height=np.mean(predicted) + 0.5*np.std(predicted))
    
    # If no peaks found, return high error
    if len(obs_peaks) == 0:
        return {
            'peak_magnitude_error': float('inf'),
            'peak_timing_error': float('inf'),
            'peak_area_error': float('inf'),
            'combined_peak_score': float('inf')
        }
    
    # Calculate peak magnitude error
    obs_peak_heights = observed[obs_peaks]
    if len(pred_peaks) > 0:
        pred_peak_heights = predicted[pred_peaks]
        # Use the highest peaks for comparison
        magnitude_error = abs(np.max(obs_peak_heights) - np.max(pred_peak_heights))
    else:
        magnitude_error = np.max(obs_peak_heights)  # Penalty for no peaks
    
    # Calculate peak timing error
    if len(pred_peaks) > 0:
        # Find minimum distance between any observed and predicted peak
        timing_errors = []
        for obs_peak in obs_peaks:
            min_distance = min(abs(obs_peak - pred_peak) for pred_peak in pred_peaks)
            timing_errors.append(min_distance)
        timing_error = np.mean(timing_errors)
    else:
        timing_error = len(observed) / 2  # Large penalty for no peaks
    
    # Calculate area error during peak periods
    peak_windows = []
    window_size = 2  # Years before and after peak
    for peak in obs_peaks:
        start = max(0, peak - window_size)
        end = min(len(observed), peak + window_size + 1)
        peak_windows.extend(range(start, end))
    peak_windows = sorted(list(set(peak_windows)))  # Remove duplicates
    
    if peak_windows:
        obs_area = np.sum(observed[peak_windows])
        pred_area = np.sum(predicted[peak_windows])
        area_error = abs(obs_area - pred_area) / obs_area
    else:
        area_error = float('inf')
    
    # Combine metrics into a single score
    # Normalize each component
    norm_magnitude_error = magnitude_error / np.max(observed)
    norm_timing_error = timing_error / len(observed)
    
    combined_score = (
        0.4 * norm_magnitude_error +  # Weight peak magnitude errors
        0.3 * norm_timing_error +     # Weight peak timing errors
        0.3 * area_error             # Weight area under peak errors
    )
    
    return {
        'peak_magnitude_error': magnitude_error,
        'peak_timing_error': timing_error,
        'peak_area_error': area_error,
        'combined_peak_score': combined_score
    }

def find_all_individuals():
    """
    Search through all individuals in the Populations directory.
    
    Returns:
        list: [(individual_path, objective_value, peak_metrics, llm_choice)]
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
                        
                        # Get data for metrics calculation
                        plot_data = get_plot_data(ind_path)
                        historical = get_historical_data(model_data)
                        
                        # Calculate regular NMSE
                        cots_nmse = calculate_nmse(historical['cots_pred'], 
                                                 plot_data['cots_pred']['Modeled'])
                        fast_nmse = calculate_nmse(historical['fast_pred'], 
                                                 plot_data['fast_pred']['Modeled'])
                        slow_nmse = calculate_nmse(historical['slow_pred'], 
                                                 plot_data['slow_pred']['Modeled'])
                        
                        # Calculate peak metrics for COTS
                        peak_metrics = calculate_peak_metrics(
                            historical['cots_pred'],
                            plot_data['cots_pred']['Modeled']
                        )
                        
                        # Combined objective: 60% peak score, 40% weighted NMSE
                        weighted_nmse = (2 * cots_nmse + fast_nmse + slow_nmse) / 4
                        combined_objective = (
                            0.6 * peak_metrics['combined_peak_score'] +
                            0.4 * weighted_nmse
                        )
                        
                        all_individuals.append((
                            ind_path, 
                            combined_objective,
                            peak_metrics,
                            llm_choice
                        ))
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
    fig.suptitle('Model Predictions with Peak-Focused Metrics')
    
    # Set up color cycle for the models
    color_cycle = plt.cm.tab10(np.linspace(0, 1, 10))
    
    # Get historical data from first model
    first_path = best_individuals[0][0]
    with open(os.path.join(first_path, 'model_report.json'), 'r') as f:
        first_model_data = json.load(f)
    historical_data = get_historical_data(first_model_data)
    years = list(range(1980, 2006))
    
    print("\nMetrics for best individuals:")
    print("-" * 50)
    
    for path, objective, peak_metrics, llm_choice in best_individuals:
        plot_data = get_plot_data(path)
        
        # Calculate NMSE for each variable
        cots_nmse = calculate_nmse(historical_data['cots_pred'], plot_data['cots_pred']['Modeled'])
        fast_nmse = calculate_nmse(historical_data['fast_pred'], plot_data['fast_pred']['Modeled'])
        slow_nmse = calculate_nmse(historical_data['slow_pred'], plot_data['slow_pred']['Modeled'])
        
        print(f"\n{llm_choice}:")
        print(f"  COTS NMSE:              {cots_nmse:.4f}")
        print(f"  Peak Magnitude Error:    {peak_metrics['peak_magnitude_error']:.4f}")
        print(f"  Peak Timing Error:       {peak_metrics['peak_timing_error']:.4f}")
        print(f"  Peak Area Error:         {peak_metrics['peak_area_error']:.4f}")
        print(f"  Combined Score:          {objective:.4f}")
        
        # Create labels with metrics
        cots_label = f"{llm_choice}\nPeak Score: {peak_metrics['combined_peak_score']:.4f}"
        fast_label = f"{llm_choice}"
        slow_label = f"{llm_choice}"
        color = color_cycle[len(ax1.lines) % len(color_cycle)]
        
        # Plot predictions
        ax1.plot(years, plot_data['cots_pred']['Modeled'], 
                label=cots_label, color=color, alpha=0.7)
        if path == first_path:  # Only plot observed once
            ax1.scatter(years, historical_data['cots_pred'], 
                       label='Observed', color='grey', s=50)
            # Mark peaks in observed data
            peaks, _ = find_peaks(historical_data['cots_pred'], 
                                height=np.mean(historical_data['cots_pred']) + 
                                0.5*np.std(historical_data['cots_pred']))
            ax1.plot(np.array(years)[peaks], 
                    np.array(historical_data['cots_pred'])[peaks], 
                    'kx', markersize=10, label='Observed Peaks')
        
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
    ax1.set_title('Crown-of-Thorns Starfish Abundance (Peak-Focused Selection)')
    ax1.set_ylabel('Abundance')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    ax2.set_title('Fast-Growing Coral Cover')
    ax2.set_ylabel('Cover (%)')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    ax3.set_title('Slow-Growing Coral Cover')
    ax3.set_xlabel('Year')
    ax3.set_ylabel('Cover (%)')
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Adjust layout and save
    plt.tight_layout()
    os.makedirs('Figures', exist_ok=True)
    plt.savefig('Figures/peak_predictions_comparison.png', 
                dpi=300, bbox_inches='tight')
    plt.close()

def main():
    try:
        # Find all individuals and calculate their metrics
        all_individuals = find_all_individuals()
        
        # Sort by combined objective value and get top 5
        best_individuals = sorted(all_individuals, key=lambda x: x[1])[:5]
        
        # Print summary of best performers
        print("\nTop 5 performers by peak-focused metrics:")
        for path, objective, peak_metrics, llm_choice in best_individuals:
            print(f"\n{llm_choice}:")
            print(f"Path: {path}")
            print(f"Combined objective value: {objective:.4f}")
            print(f"Peak metrics:")
            for metric, value in peak_metrics.items():
                print(f"  {metric}: {value:.4f}")
        
        # Create comparison plots
        create_comparison_plots(best_individuals)
        print("\nCreated comparison plot at: Figures/peak_predictions_comparison.png")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main())
