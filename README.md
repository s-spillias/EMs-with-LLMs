# Data-Driven Discovery of Mechanistic Ecosystem Models with LLMs

## Overview

This project implements "AI for Models of Ecosystems" (AIME), a novel framework that integrates large language models (LLMs) with evolutionary optimization to automate the discovery of interpretable ecological models from time-series data. AIME addresses the inverse problem of inferring ecologically meaningful mechanistic models that explain observed data while maintaining biological plausibility.

The framework utilizes a genetic algorithm to evolve and optimize ecological models, with LLMs assisting in model creation, evaluation, and improvement. AIME produces interpretable models with meaningful parameters that capture real biological processes, facilitating scientific insight and potentially accelerating management applications.

This repository contains the code and resources associated with the paper: "Data-Driven Discovery of Mechanistic Ecosystem Models with LLMs", which demonstrates AIME's capabilities through two complementary marine case studies:

1. A nutrient-phytoplankton-zooplankton (NPZ) model, where AIME successfully recovered known ecological dynamics
2. A Crown-of-Thorns starfish (COTS) model, where AIME-generated models approached human expert models in capturing outbreak dynamics

## Prerequisites

- Python 3.7+
- R (for running certain scripts)
- C++ compiler (for compiling the TMB models)

## Dependencies

See requirements.txt

You will need to acquire various API keys for communing with an AI of your choice. Currently using New Claude-Sonnet 3.5.
Put these in .env

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/s-spillias/MEMs-with-LLMs.git
   cd MEMs-with-LLMs
   ```

2. Install the required Python packages:
   ```
   pip install -r requirements.txt
   ```

3. Ensure you have R and a C++ compiler installed on your system.

## Usage

To run the genetic algorithm:

1. Create a configuration JSON file with the required parameters (see example in config_NPZ.json or config_COTS.json).
2. Run the main script with the config file and optional overrides:
   ```
   python genetic_algorithm.py --config your_config.json [options]
   ```
   
   Required config parameters:
   - `project_topic`: Description of the ecological system being modeled
   - `response_file`: Path to the response data file
   - `template_file`: Template file for model generation
   - `temperature`: Temperature parameter for LLM (0.0-1.0)
   - `max_sub_iterations`: Maximum number of sub-iterations
   - `convergence_threshold`: Threshold for convergence
   - `n_individuals`: Number of individuals in population
   - `n_generations`: Number of generations to run
   - `llm_choice`: LLM choice for model generation
   - `rag_choice`: LLM choice for RAG operations
   - `embed_choice`: Embedding model choice
   
   Optional config parameters:
   - `forcing_file`: Path to forcing data file
   - `train_test_split`: Proportion of data to use for training (0.0-1.0, default: 1.0)
   
   Command-line options (override config file):
   - `--resume POPULATION_XXXX`: Resume from an existing population
   - `--project-topic`: Override project topic
   - `--response-file`: Override response data file path
   - `--forcing-file`: Override forcing data file path
   - `--template-file`: Override template file path
   - `--temperature`: Override temperature parameter
   - `--max-sub-iterations`: Override maximum sub-iterations
   - `--convergence-threshold`: Override convergence threshold
   - `--n-individuals`: Override number of individuals
   - `--n-generations`: Override number of generations
   - `--llm-choice`: Override LLM choice
   - `--rag-choice`: Override RAG LLM choice
   - `--embed-choice`: Override embedding model choice

4. To run experiments:
   ```
   python Experiments/experiment1_convergence.py [options]
   ```
   Available options:
   - Same as genetic_algorithm.py, plus:
   - `--iterations`: Number of times to run the experiment (default: 3)

## File Structure

- `genetic_algorithm.py`: Main script that runs the genetic algorithm.
- `make_model.py`: Handles the creation and improvement of individual models.
- `run_model.py`: Executes the compiled models and returns results.
- `ControlFile.R`: Does gradient descent to fit paramter values.
- `get_params.py`: Extracts parameters from the model files.
- `search.py`: Performs RAG on local files and web search to obtain parameter values.
- `model_functions.py`: Contains utility functions for model operations.
- `POPULATIONS/`: Directory where all generated populations and their branches are stored.
- `Experiments/`: Directory containing experimental scripts:
  - `experiment1_convergence.py`: Runs multiple iterations of the genetic algorithm to test convergence with different LLM models
  - `experiment2_prediction.py`: Tests model prediction capabilities
  - `experiment3_ecology.py`: Evaluates ecological implications
  - `experiment_results/`: Stores experiment results and analysis

## Key Components

1. **Genetic Algorithm**: Manages the overall process of evolving models across multiple generations.
2. **Individual**: Represents a single model in the population, identified by a unique ID.
3. **Population**: A collection of individuals for a single run of the algorithm, stored in a directory structure.
4. **Generation**: A complete cycle of evaluation, selection, and creation of new individuals.
5. **Lineage**: The evolutionary history of an individual, tracking its parent and ancestors.

## Detailed Script Descriptions

### genetic_algorithm.py

This is the main script that orchestrates the entire genetic algorithm process for the AIME (AI for Models of Ecosystems) framework. Here's what it does:

1. Processes command-line arguments and loads configuration from a JSON file with comprehensive validation.
2. Initializes a new population or resumes from an existing one:
   - For new populations: Creates necessary directories and initializes population metadata.
   - For resuming: Loads existing metadata and continues from the last generation.
3. Sets up RAG (Retrieval-Augmented Generation) index and LLM for context-aware model generation.
4. For each generation:
   - Initializes or creates new individuals using parallel processing.
   - Evaluates each individual's performance using objective values from model reports.
   - Evolves the population by selecting the best performers and culling underperforming models.
   - Updates lineage information to track model evolution.
   - Moves culled and broken individuals to separate directories.
5. Checks for convergence after each generation based on the best objective value.
6. Saves comprehensive metadata including generation history, runtime statistics, and convergence information.

The script uses multiprocessing to parallelize model creation and evaluation, significantly speeding up execution on multi-core systems. It also includes robust error handling, detailed logging, and support for early stopping when convergence criteria are met.

### make_model.py

This script is responsible for creating and improving individual model branches. Its key functions include:

1. `process_branch`: The main function that handles the creation or improvement of a branch.
   - For new branches, it generates a new model based on the provided template and project topic.
   - For existing branches, it attempts to improve the model.
2. `make_script`: Uses AI-assisted coding (via the `aider` library) to generate or modify the model code.
3. `improve_script`: Analyzes the current model's performance and suggests improvements.
4. `get_highest_version`: Keeps track of version numbers for each branch.

The script interacts with an AI model to generate and improve the ecological models, allowing for creative and potentially unexpected solutions to emerge.

### run_model.py

This script is responsible for executing the compiled models and evaluating their performance. Its main functions are:

1. Compiling the C++ model using the TMB (Template Model Builder) framework.
2. Running the compiled model with the current parameters.
3. Calculating the objective value (a measure of model fit).
4. Handling any runtime errors or numerical instabilities.

It acts as the bridge between the C++ implementation of the ecological model and the Python-based genetic algorithm.

### get_params.py

This utility script extracts and processes parameters from the model files. It performs the following tasks:

1. Reads the parameters from the `parameters.json` file.
2. Extracts parameter information from the C++ model file.
3. Combines and formats this information.
4. Updates the `parameters.json` file with any new information.

This script ensures that the parameter information is consistently maintained and updated throughout the evolutionary process.

### model_functions.py

This script contains various utility functions used across the project. Some key functions include:

1. `run_make_model`: A wrapper function that calls the `process_branch` function from `make_model.py`.
2. Error handling and logging functions.
3. Helper functions for file and directory management.

It serves as a central location for commonly used functions, promoting code reuse and maintaining consistency across the project.

### search.py

This script implements various search functionalities to support the ecological modeling process. Key features include:

1. Multiple search engines: DuckDuckGo, Serper, Semantic Scholar, and custom RAG (Retrieval-Augmented Generation) search.
2. RAG search: Implements a vector store-based search on local directories, using ChromaDB for efficient storage and retrieval.
3. Parameter search: A specialized RAG search for querying a master parameters file.
4. Web scraping: Asynchronous fetching and parsing of web pages for content extraction.
5. Integration with external APIs: Semantic Scholar API for academic paper searches.
6. Text processing: Functions for truncating and cleaning search results.

The script uses advanced NLP techniques, including embeddings and vector stores, to provide context-aware search capabilities across various data sources. This is particularly useful for finding relevant information during the model development and improvement process.

### Code/ControlFile.R

This R script serves as the main control file for running and evaluating the ecological models. Its primary functions include:

1. Model compilation: Uses the TMB (Template Model Builder) framework to compile the C++ model.
2. Parameter management: Loads and processes model parameters from a JSON file, supporting both old and new parameter formats.
3. Data preparation: Loads time series data and prepares it for model input.
4. Model fitting: Implements a multi-phase optimization process, allowing for prioritized parameter fitting.
5. Result extraction and visualization: Generates plots comparing modeled vs. observed data for each time series variable.
6. Error handling and reporting: Provides detailed error messages and writes them to a JSON file for easy integration with the broader system.
7. Performance evaluation: Calculates an objective function value based on the mean squared error of all available time series.
8. Output generation: Creates a comprehensive JSON report including model summary, fitted parameters, and plot data.

This script is crucial for the actual execution and evaluation of the ecological models generated by the genetic algorithm. It provides a flexible framework for model fitting, allowing for complex, multi-phase optimization processes and detailed result analysis.

## Experiments

### experiment1_convergence.py

This script is designed to test the convergence properties of the genetic algorithm with different LLM models. It:

1. Runs multiple iterations of the genetic algorithm with specified settings
2. Collects metrics for each run including:
   - Convergence status and generation
   - Final objective values
   - Number of culled and broken individuals
   - Best individual's predictions
3. Saves detailed results to JSON files in `Experiments/experiment_results/`
4. Supports command-line arguments for customizing:
   - LLM and RAG model choices
   - Number of iterations
   - Population size
   - Number of generations

The results can be used to analyze:
- Consistency of model convergence
- Impact of different LLM models on performance
- Relationship between generations and objective values
- Success rates of model creation

## Additional Script Descriptions

### dummy_run_model.py
This script provides a simulated environment for testing the genetic algorithm without running actual ecological models. It includes functions to:
- Simulate model creation and improvement
- Generate random objective values and plot data
- Create dummy model reports and plot files

### evaluation_utils.py
This utility script contains functions for evaluating individuals and populations in the genetic algorithm. Key features include:
- Evaluating individual models based on their objective values
- Evaluating entire populations
- Running the model creation process with file output logging

### file_utils.py
This script provides utility functions for file and directory operations, including:
- Copying and moving individual model directories
- Saving and loading metadata
- Listing individuals in a population directory

### individual_utils.py
This script contains functions for managing individual models, including:
- Creating and updating individual metadata
- Handling lineage and parent information

### model_report_handler.py
This script is responsible for reading, writing, and updating model reports. It includes functions to:
- Read and update model reports
- Get model status, objective value, lineage, and parent information

### population_utils.py
This script contains utilities for managing populations in the genetic algorithm, including:
- Generating unique individual IDs
- Evolving populations by selecting best individuals
- Creating new generations of individuals
- Parallel processing for spawning or initializing individuals

These additional scripts work together with the main scripts to provide a comprehensive framework for running and managing the genetic algorithm for ecological modeling.

## Output

The algorithm generates the following outputs:

1. **Population Directories**: Located in `POPULATIONS/POPULATION_XXXX/`, containing:
   - `population_metadata.json`: Comprehensive metadata for the population run, including:
     - Configuration parameters
     - Start and end times
     - Generation history
     - Convergence information
     - Best performers
   - Individual directories (identified by unique IDs)
   - `CULLED/`: Directory containing culled individuals
   - `BROKEN/`: Directory containing broken individuals (failed to compile or run)

2. **Individual Directories**: Named with unique IDs, containing:
   - `metadata.json`: Metadata specific to the individual, including:
     - Objective value
     - Parent information
     - Lineage history
   - `model.cpp`: The C++ implementation of the ecological model
   - `parameters.json`: Parameters used in the model
   - `model_report.json`: Performance report for the model
   - `make_model_output.txt`: Console output from the model creation process
   - Plot files and other outputs

## How It Works

1. **Configuration and Initialization**:
   - The algorithm loads parameters from a configuration file and validates them.
   - It either initializes a new population or resumes from an existing one.
   - A RAG (Retrieval-Augmented Generation) index is set up to provide context for model generation.

2. **Population Initialization** (for new populations):
   - Multiple individuals are created in parallel using multiprocessing.
   - Each individual represents a unique model generated by an LLM based on the project topic and data.
   - Initial models are evaluated and ranked based on their objective values.

3. **Evolutionary Process** (for each generation):
   - **Creation**: New individuals are created based on the best performers from the previous generation.
   - **Evaluation**: Each individual's model is compiled, run, and evaluated against the data.
   - **Selection**: The best-performing individuals are selected based on their objective values.
   - **Lineage Tracking**: Parent-child relationships are recorded to maintain evolutionary history.
   - **Culling**: Underperforming or broken models are moved to separate directories.

4. **Convergence Checking**:
   - After each generation, the best objective value is compared to the convergence threshold.
   - If the objective value is below the threshold, the algorithm terminates early.
   - Otherwise, it continues until the maximum number of generations is reached.

5. **Metadata Management**:
   - Comprehensive metadata is maintained at both population and individual levels.
   - This includes performance metrics, lineage information, and runtime statistics.
   - The metadata is updated after each generation and saved to JSON files.

The algorithm continues until it either achieves convergence (best objective value below threshold) or reaches the maximum number of generations specified in the configuration.

## Customization

To adapt this for different ecological systems:
1. Update the `project_topic` in `genetic_algorithm.py`.
2. Modify the data file to match your system's time series data.
3. Adjust the template file (`CoTSmodel_v3.cpp`) if necessary for your specific modeling needs.
4. Add relevant literature files to 'doc_store' to provide the AI with more context for identifying parameter values or equations.

## Troubleshooting

If you encounter issues:
1. Check the console output for error messages.
2. Verify that all dependencies are correctly installed.
3. Ensure your data file is in the correct format and location.
4. Check the `make_model_output.txt` files in branch directories for detailed error logs.

![Proposed Architecture](images/conceptual_diagram.png)
<!-- <img src="Figures/conceptual_diagram.png" alt="AIME Architecture" width="200"/> -->
