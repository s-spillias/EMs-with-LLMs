#!/usr/bin/env python3
"""
Test script to analyze NPZ and COTS populations and count parameters with citations.

This script:
1. Scans all POPULATIONS directories
2. Identifies NPZ vs COTS populations by checking response_file
3. For each population, counts parameters with non-empty citations fields
4. Creates visualization comparing citation percentages
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Set
import matplotlib.pyplot as plt
import numpy as np
import subprocess

def get_population_type(population_metadata: Dict) -> str:
    """
    Determine population type based on response_file.
    Returns 'NPZ' or 'COTS'.
    """
    response_file = population_metadata.get("response_file", "")
    if "NPZ_example/npz_model_response.csv" in response_file:
        return "NPZ"
    else:
        return "COTS"


def count_citations_in_parameters(parameters_metadata_path: Path) -> Tuple[int, int, int, int, List[str]]:
    """
    Count parameters with non-empty citations in a parameters_metadata.json file.
    Distinguishes between Semantic Scholar citations and local doc_store citations.
    
    Returns:
        Tuple of (total_params, params_with_citations, params_with_semantic, 
                  params_with_docstore, params_with_citations_names)
    """
    try:
        with open(parameters_metadata_path, 'r') as f:
            data = json.load(f)
        
        parameters = data.get("parameters", [])
        total_params = len(parameters)
        params_with_citations = 0
        params_with_semantic = 0
        params_with_docstore = 0
        params_with_citations_names = []
        
        for param in parameters:
            # Check citations field (this seems to be where citations are stored)
            citations = param.get("citations", [])
            
            # Check if citations field exists and is not empty
            if citations and len(citations) > 0:
                params_with_citations += 1
                params_with_citations_names.append(param.get("parameter", "unknown"))
                
                # Classify citations by type
                has_semantic = False
                has_docstore = False
                
                for citation in citations:
                    citation_str = str(citation).lower()
                    if 'semanticscholar.org' in citation_str or 'semantic' in citation_str:
                        has_semantic = True
                    if 'doc_store' in citation_str or citation_str.endswith('.pdf'):
                        has_docstore = True
                
                if has_semantic:
                    params_with_semantic += 1
                if has_docstore:
                    params_with_docstore += 1
        
        return total_params, params_with_citations, params_with_semantic, params_with_docstore, params_with_citations_names
    
    except Exception as e:
        print(f"  Error reading {parameters_metadata_path}: {e}")
        return 0, 0, 0, 0, []


def analyze_individual(individual_dir: Path) -> Dict:
    """Analyze an individual's parameters metadata."""
    parameters_metadata_path = individual_dir / "parameters_metadata.json"
    
    if not parameters_metadata_path.exists():
        return {
            "individual_id": individual_dir.name,
            "parameters_metadata_exists": False,
            "total_params": 0,
            "params_with_citations": 0,
            "params_with_semantic": 0,
            "params_with_docstore": 0,
            "citation_percentage": 0.0,
            "semantic_percentage": 0.0,
            "docstore_percentage": 0.0,
            "params_with_citations_names": []
        }
    
    total_params, params_with_citations, params_with_semantic, params_with_docstore, params_with_citations_names = count_citations_in_parameters(
        parameters_metadata_path
    )
    
    citation_percentage = (params_with_citations / total_params * 100) if total_params > 0 else 0.0
    semantic_percentage = (params_with_semantic / total_params * 100) if total_params > 0 else 0.0
    docstore_percentage = (params_with_docstore / total_params * 100) if total_params > 0 else 0.0
    
    return {
        "individual_id": individual_dir.name,
        "parameters_metadata_exists": True,
        "total_params": total_params,
        "params_with_citations": params_with_citations,
        "params_with_semantic": params_with_semantic,
        "params_with_docstore": params_with_docstore,
        "citation_percentage": citation_percentage,
        "semantic_percentage": semantic_percentage,
        "docstore_percentage": docstore_percentage,
        "params_with_citations_names": params_with_citations_names
    }


def load_best_performers() -> Dict[str, str]:
    """
    Load the best performing individuals from populations_analysis.json.
    Returns a dict mapping population_id -> best_individual_id
    """
    populations_analysis_path = Path("Results/populations_analysis.json")
    
    if not populations_analysis_path.exists():
        print(f"Warning: {populations_analysis_path} not found. Cannot identify best performers.")
        return {}
    
    with open(populations_analysis_path, 'r') as f:
        data = json.load(f)
    
    best_performers = {}
    
    # Get best performers from populations_overview
    if "populations_overview" in data:
        for pop_key, pop_data in data["populations_overview"].items():
            pop_num = pop_data["population"]
            pop_id = f"POPULATION_{pop_num:04d}"
            best_individual = pop_data["best_individual"]
            individual_id = f"INDIVIDUAL_{best_individual}"
            best_performers[pop_id] = individual_id
    
    return best_performers


def analyze_population(population_dir: Path, best_performers: Dict[str, str]) -> Dict:
    """Analyze a population directory and count citations."""
    population_metadata_path = population_dir / "population_metadata.json"
    
    if not population_metadata_path.exists():
        return None
    
    with open(population_metadata_path, 'r') as f:
        population_metadata = json.load(f)
    
    # Determine population type
    pop_type = get_population_type(population_metadata)
    
    # Get the best performer for this population
    best_performer_id = best_performers.get(population_dir.name, None)
    
    # Analyze all individuals in this population
    individuals_analysis = []
    
    # Get list of individual directories
    individual_dirs = [d for d in population_dir.iterdir() 
                      if d.is_dir() and d.name.startswith("INDIVIDUAL_")]
    
    for individual_dir in sorted(individual_dirs):
        individual_analysis = analyze_individual(individual_dir)
        # Mark if this is the best performer
        individual_analysis["is_best_performer"] = (individual_dir.name == best_performer_id)
        individuals_analysis.append(individual_analysis)
    
    # Calculate population-level statistics
    total_individuals = len(individuals_analysis)
    total_params_all_individuals = sum(ind["total_params"] for ind in individuals_analysis)
    total_params_with_citations = sum(ind["params_with_citations"] for ind in individuals_analysis)
    total_params_with_semantic = sum(ind["params_with_semantic"] for ind in individuals_analysis)
    total_params_with_docstore = sum(ind["params_with_docstore"] for ind in individuals_analysis)
    
    overall_citation_percentage = (
        total_params_with_citations / total_params_all_individuals * 100
        if total_params_all_individuals > 0 else 0.0
    )
    overall_semantic_percentage = (
        total_params_with_semantic / total_params_all_individuals * 100
        if total_params_all_individuals > 0 else 0.0
    )
    overall_docstore_percentage = (
        total_params_with_docstore / total_params_all_individuals * 100
        if total_params_all_individuals > 0 else 0.0
    )
    
    return {
        "population_id": population_dir.name,
        "population_type": pop_type,
        "response_file": population_metadata.get("response_file", ""),
        "total_individuals": total_individuals,
        "total_params_all_individuals": total_params_all_individuals,
        "total_params_with_citations": total_params_with_citations,
        "total_params_with_semantic": total_params_with_semantic,
        "total_params_with_docstore": total_params_with_docstore,
        "overall_citation_percentage": overall_citation_percentage,
        "overall_semantic_percentage": overall_semantic_percentage,
        "overall_docstore_percentage": overall_docstore_percentage,
        "individuals": individuals_analysis
    }


def create_boxplot(all_populations: List[Dict], output_path: Path):
    """Create visualization showing citation source breakdown (no citations, Semantic Scholar, doc_store)."""
    # Collect percentages for each category by population type
    npz_no_citations = []
    cots_no_citations = []
    npz_semantic = []
    cots_semantic = []
    npz_docstore = []
    cots_docstore = []
    
    # Best performers only (single value per model type)
    npz_no_citations_best = None
    cots_no_citations_best = None
    npz_semantic_best = None
    cots_semantic_best = None
    npz_docstore_best = None
    cots_docstore_best = None
    
    for pop in all_populations:
        for ind in pop['individuals']:
            if ind['parameters_metadata_exists'] and ind['total_params'] > 0:
                # Calculate no citations percentage
                no_cit_pct = 100.0 - ind['citation_percentage']
                
                is_best = ind.get('is_best_performer', False)
                
                if pop['population_type'] == 'NPZ':
                    npz_no_citations.append(no_cit_pct)
                    npz_semantic.append(ind['semantic_percentage'])
                    npz_docstore.append(ind['docstore_percentage'])
                    
                    if is_best:
                        npz_no_citations_best = no_cit_pct
                        npz_semantic_best = ind['semantic_percentage']
                        npz_docstore_best = ind['docstore_percentage']
                else:
                    cots_no_citations.append(no_cit_pct)
                    cots_semantic.append(ind['semantic_percentage'])
                    cots_docstore.append(ind['docstore_percentage'])
                    
                    if is_best:
                        cots_no_citations_best = no_cit_pct
                        cots_semantic_best = ind['semantic_percentage']
                        cots_docstore_best = ind['docstore_percentage']
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    tick_labels = ['COTS', 'NPZ']
    
    # Plot 1: No citations (top left)
    ax = axes[0]
    data_to_plot = [cots_no_citations, npz_no_citations]
    
    bp = ax.boxplot(data_to_plot, tick_labels=tick_labels, patch_artist=True,
                    showmeans=True, meanline=True)
    
    colors = ['#ffcccc', '#ccccff']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    # Add asterisks for best performers
    if cots_no_citations_best is not None:
        ax.scatter([1], [cots_no_citations_best], marker='*', s=300, c='gold', 
                  edgecolors='black', linewidths=1.5, zorder=5, label='Best Performer')
    if npz_no_citations_best is not None:
        ax.scatter([2], [npz_no_citations_best], marker='*', s=300, c='gold', 
                  edgecolors='black', linewidths=1.5, zorder=5)
    
    ax.yaxis.grid(True, linestyle='--', alpha=0.3)
    ax.set_ylabel('% of Parameters', fontsize=12)
    ax.set_xlabel('Model Type', fontsize=12)
    ax.set_title('No Citations', fontsize=14, fontweight='bold')
    if cots_no_citations_best is not None or npz_no_citations_best is not None:
        ax.legend(loc='best', fontsize=9)
    
    for i, (label, data) in enumerate(zip(tick_labels, data_to_plot)):
        ax.text(i + 1, ax.get_ylim()[1] * 0.95, f'n={len(data)}',
                ha='center', va='top', fontsize=9)
    
    # Plot 2: Semantic Scholar citations (top right)
    ax = axes[1]
    data_to_plot = [cots_semantic, npz_semantic]
    
    bp = ax.boxplot(data_to_plot, tick_labels=tick_labels, patch_artist=True,
                    showmeans=True, meanline=True)
    
    colors = ['#ffcccc', '#ccccff']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    # Add asterisks for best performers
    if cots_semantic_best is not None:
        ax.scatter([1], [cots_semantic_best], marker='*', s=300, c='gold', 
                  edgecolors='black', linewidths=1.5, zorder=5, label='Best Performer')
    if npz_semantic_best is not None:
        ax.scatter([2], [npz_semantic_best], marker='*', s=300, c='gold', 
                  edgecolors='black', linewidths=1.5, zorder=5)
    
    ax.yaxis.grid(True, linestyle='--', alpha=0.3)
    ax.set_ylabel('% of Parameters', fontsize=12)
    ax.set_xlabel('Model Type', fontsize=12)
    ax.set_title('Semantic Scholar Citations', fontsize=14, fontweight='bold')
    if cots_semantic_best is not None or npz_semantic_best is not None:
        ax.legend(loc='best', fontsize=9)
    
    for i, (label, data) in enumerate(zip(tick_labels, data_to_plot)):
        ax.text(i + 1, ax.get_ylim()[1] * 0.95, f'n={len(data)}',
                ha='center', va='top', fontsize=9)
    
    # Plot 3: doc_store citations (bottom left)
    ax = axes[2]
    data_to_plot = [cots_docstore, npz_docstore]
    
    bp = ax.boxplot(data_to_plot, tick_labels=tick_labels, patch_artist=True,
                    showmeans=True, meanline=True)
    
    colors = ['#ffcccc', '#ccccff']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    # Add asterisks for best performers
    if cots_docstore_best is not None:
        ax.scatter([1], [cots_docstore_best], marker='*', s=300, c='gold', 
                  edgecolors='black', linewidths=1.5, zorder=5, label='Best Performer')
    if npz_docstore_best is not None:
        ax.scatter([2], [npz_docstore_best], marker='*', s=300, c='gold', 
                  edgecolors='black', linewidths=1.5, zorder=5)
    
    ax.yaxis.grid(True, linestyle='--', alpha=0.3)
    ax.set_ylabel('% of Parameters', fontsize=12)
    ax.set_xlabel('Model Type', fontsize=12)
    ax.set_title('Local doc_store Citations', fontsize=14, fontweight='bold')
    if cots_docstore_best is not None or npz_docstore_best is not None:
        ax.legend(loc='best', fontsize=9)
    
    for i, (label, data) in enumerate(zip(tick_labels, data_to_plot)):
        ax.text(i + 1, ax.get_ylim()[1] * 0.95, f'n={len(data)}',
                ha='center', va='top', fontsize=9)
    
    # Plot 4: Stacked bar chart showing average breakdown (bottom right)
    ax = axes[3]
    
    # Calculate means for stacked bar
    cots_means = [np.mean(cots_no_citations), np.mean(cots_semantic), np.mean(cots_docstore)]
    npz_means = [np.mean(npz_no_citations), np.mean(npz_semantic), np.mean(npz_docstore)]
    
    x = np.arange(2)
    width = 0.6
    
    means_list = [cots_means, npz_means]
    
    # Plot stacked bars
    bottoms = [0, 0]
    for i, label in enumerate(['No Citations', 'Semantic Scholar', 'Local doc_store']):
        heights = [means[i] for means in means_list]
        colors_map = ['#ffcccc', '#ccccff']
        ax.bar(x, heights, width, bottom=bottoms, label=label, 
               color=colors_map, alpha=0.6)
        bottoms = [b + h for b, h in zip(bottoms, heights)]
    
    ax.set_ylabel('% of Parameters', fontsize=12)
    ax.set_xlabel('Model Type', fontsize=12)
    ax.set_title('Average Citation Source Breakdown', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['COTS', 'NPZ'], fontsize=10)
    ax.legend(loc='upper right', fontsize=9)
    ax.yaxis.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Citation breakdown figure saved to: {output_path}")
    plt.close()


def main():
    """Main function to analyze all populations."""
    populations_dir = Path("POPULATIONS")
    
    if not populations_dir.exists():
        print(f"Error: {populations_dir} directory not found")
        return
    
    print("=" * 80)
    print("POPULATION CITATION ANALYSIS (NPZ vs COTS)")
    print("Including Best Performers from populations_analysis.json")
    print("=" * 80)
    print()
    
    # Load best performers
    best_performers = load_best_performers()
    print(f"Loaded {len(best_performers)} best performers from populations_analysis.json\n")
    
    # Get all population directories
    population_dirs = [d for d in populations_dir.iterdir() 
                      if d.is_dir() and d.name.startswith("POPULATION_")]
    
    all_populations = []
    
    for population_dir in sorted(population_dirs):
        population_analysis = analyze_population(population_dir, best_performers)
        
        if population_analysis:
            all_populations.append(population_analysis)
    
    # Print results
    if not all_populations:
        print("No populations found.")
        return
    
    # Separate by type
    npz_populations = [p for p in all_populations if p['population_type'] == 'NPZ']
    cots_populations = [p for p in all_populations if p['population_type'] == 'COTS']
    
    print(f"Found {len(npz_populations)} NPZ population(s) and {len(cots_populations)} COTS population(s):\n")
    
    # Print NPZ populations
    if npz_populations:
        print("NPZ POPULATIONS:")
        print("-" * 80)
    for pop in npz_populations:
        print(f"Population: {pop['population_id']}")
        print(f"  Response file: {pop['response_file']}")
        print(f"  Total individuals: {pop['total_individuals']}")
        print(f"  Total parameters (all individuals): {pop['total_params_all_individuals']}")
        print(f"  Parameters with citations: {pop['total_params_with_citations']}")
        print(f"  Overall citation percentage: {pop['overall_citation_percentage']:.2f}%")
        print()
        
        print("  Individual breakdown:")
        for ind in pop['individuals']:
            print(f"    {ind['individual_id']}:")
            if ind['parameters_metadata_exists']:
                print(f"      Total params: {ind['total_params']}")
                print(f"      Params with citations: {ind['params_with_citations']}")
                print(f"      Citation percentage: {ind['citation_percentage']:.2f}%")
                if ind['params_with_citations_names']:
                    print(f"      Parameters with citations: {', '.join(ind['params_with_citations_names'])}")
            else:
                print(f"      No parameters_metadata.json found")
            print()
        
        print()
    
    # Print COTS populations
    if cots_populations:
        print("\nCOTS POPULATIONS:")
        print("-" * 80)
    for pop in cots_populations:
        print(f"Population: {pop['population_id']}")
        print(f"  Response file: {pop['response_file']}")
        print(f"  Total individuals: {pop['total_individuals']}")
        print(f"  Total parameters (all individuals): {pop['total_params_all_individuals']}")
        print(f"  Parameters with citations: {pop['total_params_with_citations']}")
        print(f"  Overall citation percentage: {pop['overall_citation_percentage']:.2f}%")
        print()
        
        print("  Individual breakdown:")
        for ind in pop['individuals']:
            print(f"    {ind['individual_id']}:")
            if ind['parameters_metadata_exists']:
                print(f"      Total params: {ind['total_params']}")
                print(f"      Params with citations: {ind['params_with_citations']}")
                print(f"      Citation percentage: {ind['citation_percentage']:.2f}%")
                if ind['params_with_citations_names']:
                    print(f"      Parameters with citations: {', '.join(ind['params_with_citations_names'])}")
            else:
                print(f"      No parameters_metadata.json found")
            print()
        print()
    
    # Summary statistics
    print("=" * 80)
    print("SUMMARY STATISTICS (% of total parameters)")
    print("=" * 80)
    
    # Calculate best performer summaries
    npz_best_params = 0
    npz_best_citations = 0
    npz_best_semantic = 0
    npz_best_docstore = 0
    cots_best_params = 0
    cots_best_citations = 0
    cots_best_semantic = 0
    cots_best_docstore = 0
    
    for pop in all_populations:
        for ind in pop['individuals']:
            if ind.get('is_best_performer', False) and ind['parameters_metadata_exists']:
                if pop['population_type'] == 'NPZ':
                    npz_best_params += ind['total_params']
                    npz_best_citations += ind['params_with_citations']
                    npz_best_semantic += ind['params_with_semantic']
                    npz_best_docstore += ind['params_with_docstore']
                else:
                    cots_best_params += ind['total_params']
                    cots_best_citations += ind['params_with_citations']
                    cots_best_semantic += ind['params_with_semantic']
                    cots_best_docstore += ind['params_with_docstore']
    
    # NPZ summary
    if npz_populations:
        npz_params = sum(pop['total_params_all_individuals'] for pop in npz_populations)
        npz_citations = sum(pop['total_params_with_citations'] for pop in npz_populations)
        npz_semantic = sum(pop['total_params_with_semantic'] for pop in npz_populations)
        npz_docstore = sum(pop['total_params_with_docstore'] for pop in npz_populations)
        npz_no_citations = npz_params - npz_citations
        
        npz_percentage = (npz_citations / npz_params * 100) if npz_params > 0 else 0.0
        npz_semantic_pct = (npz_semantic / npz_params * 100) if npz_params > 0 else 0.0
        npz_docstore_pct = (npz_docstore / npz_params * 100) if npz_params > 0 else 0.0
        npz_no_citations_pct = (npz_no_citations / npz_params * 100) if npz_params > 0 else 0.0
        
        print(f"\nNPZ Populations ({len(npz_populations)} total):")
        print(f"  All Individuals:")
        print(f"    Total parameters: {npz_params}")
        print(f"    Breakdown:")
        print(f"      - No citations: {npz_no_citations} ({npz_no_citations_pct:.2f}%)")
        print(f"      - Semantic Scholar: {npz_semantic} ({npz_semantic_pct:.2f}%)")
        print(f"      - Local doc_store: {npz_docstore} ({npz_docstore_pct:.2f}%)")
        print(f"    Total with ANY citations: {npz_citations} ({npz_percentage:.2f}%)")
        
        if npz_best_params > 0:
            npz_best_no_citations = npz_best_params - npz_best_citations
            npz_best_percentage = (npz_best_citations / npz_best_params * 100)
            npz_best_semantic_pct = (npz_best_semantic / npz_best_params * 100)
            npz_best_docstore_pct = (npz_best_docstore / npz_best_params * 100)
            npz_best_no_citations_pct = (npz_best_no_citations / npz_best_params * 100)
            
            print(f"  Best Performers Only:")
            print(f"    Total parameters: {npz_best_params}")
            print(f"    Breakdown:")
            print(f"      - No citations: {npz_best_no_citations} ({npz_best_no_citations_pct:.2f}%)")
            print(f"      - Semantic Scholar: {npz_best_semantic} ({npz_best_semantic_pct:.2f}%)")
            print(f"      - Local doc_store: {npz_best_docstore} ({npz_best_docstore_pct:.2f}%)")
            print(f"    Total with ANY citations: {npz_best_citations} ({npz_best_percentage:.2f}%)")
    
    # COTS summary
    if cots_populations:
        cots_params = sum(pop['total_params_all_individuals'] for pop in cots_populations)
        cots_citations = sum(pop['total_params_with_citations'] for pop in cots_populations)
        cots_semantic = sum(pop['total_params_with_semantic'] for pop in cots_populations)
        cots_docstore = sum(pop['total_params_with_docstore'] for pop in cots_populations)
        cots_no_citations = cots_params - cots_citations
        
        cots_percentage = (cots_citations / cots_params * 100) if cots_params > 0 else 0.0
        cots_semantic_pct = (cots_semantic / cots_params * 100) if cots_params > 0 else 0.0
        cots_docstore_pct = (cots_docstore / cots_params * 100) if cots_params > 0 else 0.0
        cots_no_citations_pct = (cots_no_citations / cots_params * 100) if cots_params > 0 else 0.0
        
        print(f"\nCOTS Populations ({len(cots_populations)} total):")
        print(f"  All Individuals:")
        print(f"    Total parameters: {cots_params}")
        print(f"    Breakdown:")
        print(f"      - No citations: {cots_no_citations} ({cots_no_citations_pct:.2f}%)")
        print(f"      - Semantic Scholar: {cots_semantic} ({cots_semantic_pct:.2f}%)")
        print(f"      - Local doc_store: {cots_docstore} ({cots_docstore_pct:.2f}%)")
        print(f"    Total with ANY citations: {cots_citations} ({cots_percentage:.2f}%)")
        
        if cots_best_params > 0:
            cots_best_no_citations = cots_best_params - cots_best_citations
            cots_best_percentage = (cots_best_citations / cots_best_params * 100)
            cots_best_semantic_pct = (cots_best_semantic / cots_best_params * 100)
            cots_best_docstore_pct = (cots_best_docstore / cots_best_params * 100)
            cots_best_no_citations_pct = (cots_best_no_citations / cots_best_params * 100)
            
            print(f"  Best Performers Only:")
            print(f"    Total parameters: {cots_best_params}")
            print(f"    Breakdown:")
            print(f"      - No citations: {cots_best_no_citations} ({cots_best_no_citations_pct:.2f}%)")
            print(f"      - Semantic Scholar: {cots_best_semantic} ({cots_best_semantic_pct:.2f}%)")
            print(f"      - Local doc_store: {cots_best_docstore} ({cots_best_docstore_pct:.2f}%)")
            print(f"    Total with ANY citations: {cots_best_citations} ({cots_best_percentage:.2f}%)")
    
    # Overall summary
    total_params = sum(pop['total_params_all_individuals'] for pop in all_populations)
    total_citations = sum(pop['total_params_with_citations'] for pop in all_populations)
    total_semantic = sum(pop['total_params_with_semantic'] for pop in all_populations)
    total_docstore = sum(pop['total_params_with_docstore'] for pop in all_populations)
    total_no_citations = total_params - total_citations
    
    overall_percentage = (total_citations / total_params * 100) if total_params > 0 else 0.0
    overall_semantic_pct = (total_semantic / total_params * 100) if total_params > 0 else 0.0
    overall_docstore_pct = (total_docstore / total_params * 100) if total_params > 0 else 0.0
    overall_no_citations_pct = (total_no_citations / total_params * 100) if total_params > 0 else 0.0
    
    print(f"\nOverall ({len(all_populations)} populations total):")
    print(f"  Total parameters: {total_params}")
    print(f"  Breakdown:")
    print(f"    - No citations: {total_no_citations} ({overall_no_citations_pct:.2f}%)")
    print(f"    - Semantic Scholar: {total_semantic} ({overall_semantic_pct:.2f}%)")
    print(f"    - Local doc_store: {total_docstore} ({overall_docstore_pct:.2f}%)")
    print(f"  Total with ANY citations: {total_citations} ({overall_percentage:.2f}%)")
    print()
    
    # Save results to JSON file
    output_file = Path("Results/citations_analysis.json")
    with open(output_file, 'w') as f:
        json.dump({
            "npz_populations": npz_populations,
            "cots_populations": cots_populations,
            "summary": {
                "total_populations": len(all_populations),
                "npz_populations": len(npz_populations),
                "cots_populations": len(cots_populations),
                "total_parameters": total_params,
                "total_parameters_with_citations": total_citations,
                "total_parameters_with_semantic": total_semantic,
                "total_parameters_with_docstore": total_docstore,
                "total_parameters_no_citations": total_no_citations,
                "overall_citation_percentage": overall_percentage,
                "overall_semantic_percentage": overall_semantic_pct,
                "overall_docstore_percentage": overall_docstore_pct,
                "overall_no_citations_percentage": overall_no_citations_pct,
                "npz_summary": {
                    "total_parameters": npz_params if npz_populations else 0,
                    "params_with_citations": npz_citations if npz_populations else 0,
                    "params_with_semantic": npz_semantic if npz_populations else 0,
                    "params_with_docstore": npz_docstore if npz_populations else 0,
                    "params_no_citations": npz_no_citations if npz_populations else 0,
                    "citation_percentage": npz_percentage if npz_populations else 0.0,
                    "semantic_percentage": npz_semantic_pct if npz_populations else 0.0,
                    "docstore_percentage": npz_docstore_pct if npz_populations else 0.0,
                    "no_citations_percentage": npz_no_citations_pct if npz_populations else 0.0
                },
                "cots_summary": {
                    "total_parameters": cots_params if cots_populations else 0,
                    "params_with_citations": cots_citations if cots_populations else 0,
                    "params_with_semantic": cots_semantic if cots_populations else 0,
                    "params_with_docstore": cots_docstore if cots_populations else 0,
                    "params_no_citations": cots_no_citations if cots_populations else 0,
                    "citation_percentage": cots_percentage if cots_populations else 0.0,
                    "semantic_percentage": cots_semantic_pct if cots_populations else 0.0,
                    "docstore_percentage": cots_docstore_pct if cots_populations else 0.0,
                    "no_citations_percentage": cots_no_citations_pct if cots_populations else 0.0
                },
                "npz_best_performers_summary": {
                    "total_parameters": npz_best_params,
                    "params_with_citations": npz_best_citations,
                    "params_with_semantic": npz_best_semantic,
                    "params_with_docstore": npz_best_docstore,
                    "params_no_citations": npz_best_params - npz_best_citations,
                    "citation_percentage": (npz_best_citations / npz_best_params * 100) if npz_best_params > 0 else 0.0,
                    "semantic_percentage": (npz_best_semantic / npz_best_params * 100) if npz_best_params > 0 else 0.0,
                    "docstore_percentage": (npz_best_docstore / npz_best_params * 100) if npz_best_params > 0 else 0.0,
                    "no_citations_percentage": ((npz_best_params - npz_best_citations) / npz_best_params * 100) if npz_best_params > 0 else 0.0
                },
                "cots_best_performers_summary": {
                    "total_parameters": cots_best_params,
                    "params_with_citations": cots_best_citations,
                    "params_with_semantic": cots_best_semantic,
                    "params_with_docstore": cots_best_docstore,
                    "params_no_citations": cots_best_params - cots_best_citations,
                    "citation_percentage": (cots_best_citations / cots_best_params * 100) if cots_best_params > 0 else 0.0,
                    "semantic_percentage": (cots_best_semantic / cots_best_params * 100) if cots_best_params > 0 else 0.0,
                    "docstore_percentage": (cots_best_docstore / cots_best_params * 100) if cots_best_params > 0 else 0.0,
                    "no_citations_percentage": ((cots_best_params - cots_best_citations) / cots_best_params * 100) if cots_best_params > 0 else 0.0
                }
            }
        }, f, indent=2)
    
    print(f"Results saved to: {output_file}")
    
    # Create boxplot visualization
    if all_populations:
        # After writing Results/citations_analysis.json:
        json_path = Path("Results/citations_analysis.json")
        out_path = Path("Figures/citations_boxplot.png")

        subprocess.run(
            ["Rscript", "scripts_analysis/plot_citations.R", str(json_path), str(out_path)],
            check=True
        )
        print(f"Citation breakdown figure saved to: {out_path}")

if __name__ == "__main__":
    main()
