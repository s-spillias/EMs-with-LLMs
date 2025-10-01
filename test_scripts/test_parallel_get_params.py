#!/usr/bin/env python3
"""
Test script to run get_params in parallel processes to check for hanging issues.
This script creates multiple test directories with minimal model files,
then runs get_params on them in parallel to see if any processes hang.
"""

import os
import sys
import time
import json
import shutil
import random
import string
import multiprocessing
from multiprocessing import Pool, Manager
import argparse
from datetime import datetime
from scripts.search_utils import init_rate_limit_manager

# Import the get_params function
from scripts.get_params import get_params

# Set up logging with timestamps
def log(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    print(f"[{timestamp}] {message}", flush=True)

def create_test_directory(base_dir, index):
    """Create a test directory with minimal files needed for get_params to run."""
    # Create a unique directory name
    random_suffix = ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))
    individual_id = f"TEST_INDIVIDUAL_{random_suffix}"
    
    # Create population directory
    population_dir = os.path.join(base_dir, f"TEST_POPULATION_{index}")
    os.makedirs(population_dir, exist_ok=True)
    
    # Use the existing doc_store directory
    doc_store_dir = "doc_store"
    
    # Create population metadata with doc_store
    population_metadata = {
        "rag_choice": "claude-3-5-haiku-20241022",
        "doc_store_dir": doc_store_dir
    }
    with open(os.path.join(population_dir, "population_metadata.json"), "w") as f:
        json.dump(population_metadata, f, indent=2)
    
    # Create individual directory
    individual_dir = os.path.join(population_dir, individual_id)
    os.makedirs(individual_dir, exist_ok=True)
    
    # Create minimal model.cpp
    with open(os.path.join(individual_dir, "model.cpp"), "w") as f:
        f.write("""
#include <TMB.hpp>
template<class Type>
Type objective_function<Type>::operator() ()
{
  DATA_VECTOR(time_dat);
  DATA_VECTOR(N_dat);
  DATA_VECTOR(P_dat);
  DATA_VECTOR(Z_dat);
  
  PARAMETER(growth_rate);
  PARAMETER(carrying_capacity);
  PARAMETER(grazing_rate);
  
  int n = time_dat.size();
  
  vector<Type> N_pred(n);
  vector<Type> P_pred(n);
  vector<Type> Z_pred(n);
  
  N_pred(0) = N_dat(0);
  P_pred(0) = P_dat(0);
  Z_pred(0) = Z_dat(0);
  
  for(int i=1; i<n; i++) {
    N_pred(i) = N_pred(i-1) * (1 + growth_rate * (1 - N_pred(i-1)/carrying_capacity));
    P_pred(i) = P_pred(i-1) * (1 + 0.1);
    Z_pred(i) = Z_pred(i-1) * (1 + grazing_rate * P_pred(i-1));
  }
  
  Type nll = 0.0;
  for(int i=0; i<n; i++) {
    nll -= dnorm(N_dat(i), N_pred(i), 0.1, true);
    nll -= dnorm(P_dat(i), P_pred(i), 0.1, true);
    nll -= dnorm(Z_dat(i), Z_pred(i), 0.1, true);
  }
  
  REPORT(N_pred);
  REPORT(P_pred);
  REPORT(Z_pred);
  
  return nll;
}
""")
    
    # Create parameters.json with literature parameters
    with open(os.path.join(individual_dir, "parameters.json"), "w") as f:
        f.write("""
{
  "parameters": [
    {
      "parameter": "growth_rate",
      "value": 0.5,
      "description": "Intrinsic growth rate (year^-1)",
      "source": "literature",
      "import_type": "PARAMETER",
      "priority": 1,
      "lower_bound": 0.0,
      "upper_bound": 2.0
    },
    {
      "parameter": "carrying_capacity",
      "value": 100.0,
      "description": "Maximum population size (individuals)",
      "source": "literature",
      "import_type": "PARAMETER",
      "priority": 2,
      "lower_bound": 10.0,
      "upper_bound": 1000.0
    },
    {
      "parameter": "grazing_rate",
      "value": 0.3,
      "description": "Rate at which zooplankton graze on phytoplankton",
      "source": "literature",
      "import_type": "PARAMETER",
      "priority": 1,
      "lower_bound": 0.0,
      "upper_bound": 1.0
    }
  ]
}
""")
    
    return individual_dir

def _worker_init(shared_mgr):
    """Initialize rate limiting for worker processes."""
    # Enforce 1 rps host-wide; allow up to 2 in-flight per process
    init_rate_limit_manager(shared_mgr, s2_min_interval=1.0, s2_max_concurrent=2)

def worker_function(directory):
    """Worker function for the process pool that directly calls get_params."""
    log(f"Starting get_params for {directory}")
    start_time = time.time()
    
    try:
        # Directly call get_params without a timeout wrapper
        get_params(directory)
        elapsed = time.time() - start_time
        log(f"Completed get_params for {directory} in {elapsed:.2f} seconds")
        return {
            "directory": directory,
            "success": True,
            "elapsed": elapsed
        }
    except Exception as e:
        elapsed = time.time() - start_time
        log(f"ERROR: get_params for {directory} failed after {elapsed:.2f} seconds: {str(e)}")
        return {
            "directory": directory,
            "success": False,
            "elapsed": elapsed,
            "error": str(e)
        }

def main():
    parser = argparse.ArgumentParser(description="Test get_params in parallel processes")
    parser.add_argument("--processes", type=int, default=4, help="Number of parallel processes to run")
    args = parser.parse_args()
    
    # Create a temporary test directory
    base_dir = os.path.join(os.getcwd(), "test_get_params_parallel")
    if os.path.exists(base_dir):
        shutil.rmtree(base_dir)
    os.makedirs(base_dir)
    
    log(f"Creating {args.processes} test directories in {base_dir}")
    
    # Create test directories
    test_dirs = []
    for i in range(args.processes):
        test_dir = create_test_directory(base_dir, i)
        test_dirs.append(test_dir)
        log(f"Created test directory: {test_dir}")
    
    # Run get_params in parallel
    log(f"Running get_params in {args.processes} parallel processes")
    
    mgr = Manager()
    with Pool(processes=args.processes, initializer=_worker_init, initargs=(mgr,)) as pool:
        results = pool.map(worker_function, test_dirs)
    
    # Check results
    all_succeeded = all(result["success"] for result in results)
    
    if all_succeeded:
        log("SUCCESS: All processes completed successfully!")
    else:
        log("ERROR: Some processes failed!")
    
    # Print detailed results
    log("Detailed results:")
    for result in results:
        status = "SUCCESS" if result["success"] else "FAILURE"
        log(f"  {result['directory']}: {status} ({result['elapsed']:.2f} seconds)")
        if not result["success"] and "error" in result:
            log(f"    Error: {result['error']}")
    
    # Clean up
    if all_succeeded:
        log(f"Cleaning up test directory: {base_dir}")
        shutil.rmtree(base_dir)
    else:
        log(f"Test directory {base_dir} left for inspection")
    
    return 0 if all_succeeded else 1

if __name__ == "__main__":
    sys.exit(main())