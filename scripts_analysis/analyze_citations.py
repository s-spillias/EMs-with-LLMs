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
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np


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


def analyze_population(population_dir: Path) -> Dict:
    """Analyze a population directory and count citations."""
    population_metadata_path = population_dir / "population_metadata.json"
    
    if not population_metadata_path.exists():
        return None
    
    with open(population_metadata_path, 'r') as f:
        population_metadata = json.load(f)
    
    # Determine population type
    pop_type = get_population_type(population_metadata)
    
    # Analyze all individuals in this population
    individuals_analysis = []
    
    # Get list of individual directories
    individual_dirs = [d for d in population_dir.iterdir() 
                      if d.is_dir() and d.name.startswith("INDIVIDUAL_")]
    
    for individual_dir in sorted(individual_dirs):
        individual_analysis = analyze_individual(individual_dir)
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
    
    for pop in all_populations:
        for ind in pop['individuals']:
            if ind['parameters_metadata_exists'] and ind['total_params'] > 0:
                # Calculate no citations percentage
                no_cit_pct = 100.0 - ind['citation_percentage']
                
                if pop['population_type'] == 'NPZ':
                    npz_no_citations.append(no_cit_pct)
                    npz_semantic.append(ind['semantic_percentage'])
                    npz_docstore.append(ind['docstore_percentage'])
                else:
                    cots_no_citations.append(no_cit_pct)
                    cots_semantic.append(ind['semantic_percentage'])
                    cots_docstore.append(ind['docstore_percentage'])
    
    # Create figure with subplots - now 4 panels
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    tick_labels = ['COTS', 'NPZ']
    
    # Plot 1: No citations (top left)
    ax = axes[0]
    data_to_plot = [cots_no_citations, npz_no_citations]
    
    bp = ax.boxplot(data_to_plot, tick_labels=tick_labels, patch_artist=True,
                    showmeans=True, meanline=True)
    
    colors = ['#cccccc', '#999999']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.yaxis.grid(True, linestyle='--', alpha=0.3)
    ax.set_ylabel('% of Parameters', fontsize=12)
    ax.set_xlabel('Model Type', fontsize=12)
    ax.set_title('No Citations', fontsize=14, fontweight='bold')
    
    for i, (label, data) in enumerate(zip(tick_labels, data_to_plot)):
        ax.text(i + 1, ax.get_ylim()[1] * 0.95, f'n={len(data)}',
                ha='center', va='top', fontsize=10)
    
    # Plot 2: Semantic Scholar citations (top right)
    ax = axes[1]
    data_to_plot = [cots_semantic, npz_semantic]
    
    bp = ax.boxplot(data_to_plot, tick_labels=tick_labels, patch_artist=True,
                    showmeans=True, meanline=True)
    
    colors = ['#ffcc99', '#99ccff']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.yaxis.grid(True, linestyle='--', alpha=0.3)
    ax.set_ylabel('% of Parameters', fontsize=12)
    ax.set_xlabel('Model Type', fontsize=12)
    ax.set_title('Semantic Scholar Citations', fontsize=14, fontweight='bold')
    
    for i, (label, data) in enumerate(zip(tick_labels, data_to_plot)):
        ax.text(i + 1, ax.get_ylim()[1] * 0.95, f'n={len(data)}',
                ha='center', va='top', fontsize=10)
    
    # Plot 3: doc_store citations (bottom left)
    ax = axes[2]
    data_to_plot = [cots_docstore, npz_docstore]
    
    bp = ax.boxplot(data_to_plot, tick_labels=tick_labels, patch_artist=True,
                    showmeans=True, meanline=True)
    
    colors = ['#ff99cc', '#99ffcc']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.yaxis.grid(True, linestyle='--', alpha=0.3)
    ax.set_ylabel('% of Parameters', fontsize=12)
    ax.set_xlabel('Model Type', fontsize=12)
    ax.set_title('Local doc_store Citations', fontsize=14, fontweight='bold')
    
    for i, (label, data) in enumerate(zip(tick_labels, data_to_plot)):
        ax.text(i + 1, ax.get_ylim()[1] * 0.95, f'n={len(data)}',
                ha='center', va='top', fontsize=10)
    
    # Plot 4: Stacked bar chart showing average breakdown (bottom right)
    ax = axes[3]
    
    # Calculate means for stacked bar
    cots_means = [np.mean(cots_no_citations), np.mean(cots_semantic), np.mean(cots_docstore)]
    npz_means = [np.mean(npz_no_citations), np.mean(npz_semantic), np.mean(npz_docstore)]
    
    x = np.arange(2)
    width = 0.6
    
    p1 = ax.bar(x, [cots_means[0], npz_means[0]], width, label='No Citations', color='#cccccc')
    p2 = ax.bar(x, [cots_means[1], npz_means[1]], width, bottom=[cots_means[0], npz_means[0]], 
                label='Semantic Scholar', color='#99ccff')
    p3 = ax.bar(x, [cots_means[2], npz_means[2]], width, 
                bottom=[cots_means[0]+cots_means[1], npz_means[0]+npz_means[1]], 
                label='Local doc_store', color='#99ffcc')
    
    ax.set_ylabel('% of Parameters', fontsize=12)
    ax.set_xlabel('Model Type', fontsize=12)
    ax.set_title('Average Citation Source Breakdown', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels)
    ax.legend(loc='lower right')
    ax.yaxis.grid(True, linestyle='--', alpha=0.3)
    
    # Add percentage labels on bars
    for i, (c_vals, n_vals) in enumerate(zip([cots_means], [npz_means])):
        for j, (c, n) in enumerate(zip(c_vals, n_vals)):
            if c > 5:  # Only show label if segment is large enough
                y_pos = sum(cots_means[:j]) + c/2
                ax.text(0, y_pos, f'{c:.1f}%', ha='center', va='center', fontsize=10, fontweight='bold')
            if n > 5:
                y_pos = sum(npz_means[:j]) + n/2
                ax.text(1, y_pos, f'{n:.1f}%', ha='center', va='center', fontsize=10, fontweight='bold')
    
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
    print("=" * 80)
    print()
    
    # Get all population directories
    population_dirs = [d for d in populations_dir.iterdir() 
                      if d.is_dir() and d.name.startswith("POPULATION_")]
    
    all_populations = []
    
    for population_dir in sorted(population_dirs):
        population_analysis = analyze_population(population_dir)
        
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
        print(f"  Total parameters: {npz_params}")
        print(f"  Breakdown:")
        print(f"    - No citations: {npz_no_citations} ({npz_no_citations_pct:.2f}%)")
        print(f"    - Semantic Scholar: {npz_semantic} ({npz_semantic_pct:.2f}%)")
        print(f"    - Local doc_store: {npz_docstore} ({npz_docstore_pct:.2f}%)")
        print(f"  Total with ANY citations: {npz_citations} ({npz_percentage:.2f}%)")
    
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
        print(f"  Total parameters: {cots_params}")
        print(f"  Breakdown:")
        print(f"    - No citations: {cots_no_citations} ({cots_no_citations_pct:.2f}%)")
        print(f"    - Semantic Scholar: {cots_semantic} ({cots_semantic_pct:.2f}%)")
        print(f"    - Local doc_store: {cots_docstore} ({cots_docstore_pct:.2f}%)")
        print(f"  Total with ANY citations: {cots_citations} ({cots_percentage:.2f}%)")
    
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
                }
            }
        }, f, indent=2)
    
    print(f"Results saved to: {output_file}")
    
    # Create boxplot visualization
    if all_populations:
        boxplot_file = Path("Figures/citations_boxplot.png")
        create_boxplot(all_populations, boxplot_file)


if __name__ == "__main__":
    main()
