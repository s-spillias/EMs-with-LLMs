import os
import pandas as pd
import json

def create_data_report(file_path):
    """
    Reads a CSV or Excel file, analyzes its contents, and saves a summary report to a JSON file.
    
    Parameters:
        file_path (str): Path to the CSV or Excel file.
    """
    try:
        # Determine file type and load data
        if file_path.endswith(".csv"):
            df = pd.read_csv(file_path)
        elif file_path.endswith(".xlsx") or file_path.endswith(".xls"):
            df = pd.read_excel(file_path)
        else:
            raise ValueError("Unsupported file type. Please provide a CSV or Excel file.")
        
        # Generate the output filename
        base_name, _ = os.path.splitext(file_path)
        output_file = f"{base_name}_report.json"
        
        # Get numeric columns
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        
        # Calculate trends and patterns
        trends = {}
        for col in numeric_cols:
            series = df[col]
            trend = "increasing" if series.iloc[-1] > series.iloc[0] else "decreasing"
            if abs(series.iloc[-1] - series.iloc[0]) < 0.1 * series.mean():
                trend = "stable"
            
            # Calculate coefficient of variation to assess variability
            cv = series.std() / series.mean() if series.mean() != 0 else float('inf')
            variability = "high" if cv > 0.5 else "moderate" if cv > 0.1 else "low"
            
            trends[col] = {
                "pattern": trend,
                "variability": variability,
                "range": f"{series.min():.2f} to {series.max():.2f}"
            }
        
        # Identify strong correlations (|r| > 0.7)
        strong_correlations = []
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr()
            for i in range(len(numeric_cols)):
                for j in range(i+1, len(numeric_cols)):
                    if abs(corr_matrix.iloc[i,j]) > 0.7:
                        strong_correlations.append({
                            "variables": [numeric_cols[i], numeric_cols[j]],
                            "correlation": f"{corr_matrix.iloc[i,j]:.2f}"
                        })
        
        # Prepare simplified report
        report = {
            "file_name": os.path.basename(file_path),
            "overview": {
                "time_points": df.shape[0],
                "variables": df.shape[1],
                "time_range": f"From {df.index[0]} to {df.index[-1]}" if isinstance(df.index, pd.DatetimeIndex) else f"{df.shape[0]} time points"
            },
            "variables": {
                col: {
                    "type": str(df[col].dtype),
                    "summary": trends[col] if col in trends else "non-numeric"
                } for col in df.columns
            },
            "relationships": strong_correlations if strong_correlations else "No strong correlations found"
        }
        
        # Save the report to a JSON file
        with open(output_file, "w") as f:
            json.dump(report, f, indent=4)
        
        print(f"Data report successfully created and saved to {output_file}.")
    
    except Exception as e:
        print(f"An error occurred: {e}")

# Usage example
create_data_report("Data/timeseries_data.csv")  # Replace with your actual file path
