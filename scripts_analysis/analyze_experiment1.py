import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
import logging
from pathlib import Path

class ConvergenceAnalyzer:
    def __init__(self, results_file):
        """Initialize analyzer with path to results file"""
        self.results_file = results_file
        self.results_dir = os.path.dirname(results_file)
        
        # Load experiment data
        with open(results_file, 'r') as f:
            self.experiment_data = json.load(f)
            
    def calculate_summary_statistics(self):
        """Calculate summary statistics from experiment results"""
        if not self.experiment_data["populations"]:
            return None
            
        metrics_df = pd.DataFrame([
            {k: v for k, v in p.items() if k != "predictions"}
            for p in self.experiment_data["populations"]
        ])
        
        return {
            "convergence_rate": metrics_df["converged"].mean(),
            "mean_generations": metrics_df["generations"].mean(),
            "std_generations": metrics_df["generations"].std(),
            "mean_objective": metrics_df["final_objective_value"].mean(),
            "std_objective": metrics_df["final_objective_value"].std(),
            "mean_culled": metrics_df["culled_count"].mean(),
            "mean_broken": metrics_df["broken_count"].mean()
        }
        
    def analyze_prediction_consistency(self):
        """Analyze consistency of predictions across populations"""
        predictions = [p["predictions"] for p in self.experiment_data["populations"]]
        
        # Convert predictions to numpy arrays for easier analysis
        cots_preds = np.array([p["cots"] for p in predictions])
        fast_coral_preds = np.array([p["fast_coral"] for p in predictions])
        slow_coral_preds = np.array([p["slow_coral"] for p in predictions])
        
        # Calculate mean and standard deviation at each time point
        consistency = {
            "cots": {
                "mean_trajectory": cots_preds.mean(axis=0).tolist(),
                "std_trajectory": cots_preds.std(axis=0).tolist(),
                "coefficient_of_variation": (cots_preds.std(axis=0) / cots_preds.mean(axis=0)).mean()
            },
            "fast_coral": {
                "mean_trajectory": fast_coral_preds.mean(axis=0).tolist(),
                "std_trajectory": fast_coral_preds.std(axis=0).tolist(),
                "coefficient_of_variation": (fast_coral_preds.std(axis=0) / fast_coral_preds.mean(axis=0)).mean()
            },
            "slow_coral": {
                "mean_trajectory": slow_coral_preds.mean(axis=0).tolist(),
                "std_trajectory": slow_coral_preds.std(axis=0).tolist(),
                "coefficient_of_variation": (slow_coral_preds.std(axis=0) / slow_coral_preds.mean(axis=0)).mean()
            }
        }
        
        return consistency
        
    def run_analysis(self):
        """Run complete analysis on experiment results"""
        analysis_results = {
            "experiment_settings": self.experiment_data["settings"],
            "start_time": self.experiment_data["start_time"],
            "summary_statistics": self.calculate_summary_statistics(),
            "prediction_consistency": self.analyze_prediction_consistency()
        }
        
        # Save analysis results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        analysis_file = os.path.join(self.results_dir, f"convergence_analysis_{timestamp}.json")
        
        with open(analysis_file, "w") as f:
            json.dump(analysis_results, f, indent=2)
            
        return analysis_file

def main():
    # Get latest results file from experiment_results directory
    results_dir = "scripts_analysis/experiment_results"
    results_files = sorted(
        [f for f in os.listdir(results_dir) if f.startswith("convergence_results_")],
        reverse=True
    )
    
    if not results_files:
        print("No experiment results found to analyze")
        return
        
    latest_results = os.path.join(results_dir, results_files[0])
    
    # Run analysis
    analyzer = ConvergenceAnalyzer(latest_results)
    analysis_file = analyzer.run_analysis()
    print(f"Analysis completed and saved to {analysis_file}")

if __name__ == "__main__":
    main()
