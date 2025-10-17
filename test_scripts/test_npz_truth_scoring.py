#!/usr/bin/env python3
"""
Test script to validate the ecological characteristics scoring system
by evaluating the ground truth NPZ model against itself.

Expected result: All characteristics should score 3 (TRUTH_MATCH)
with a normalized total score of 1.0.
"""

import sys
import os
import json

# Add parent directory to path to import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from scripts_analysis.evaluate_ecological_characteristics import (
    evaluate_model,
    read_ground_truth,
    calculate_total_score,
    ECOLOGICAL_CHARACTERISTICS
)

def test_truth_model_scoring():
    """
    Test that the ground truth NPZ model scores perfectly when evaluated against itself.
    """
    print("=" * 80)
    print("Testing NPZ Ground Truth Model Scoring")
    print("=" * 80)
    print()
    
    # Read the ground truth model
    try:
        truth_model = read_ground_truth()
        print(f"✓ Successfully loaded ground truth model from Data/NPZ_example/NPZ_model.py")
        print(f"  Model size: {len(truth_model)} characters")
        print()
    except Exception as e:
        print(f"✗ ERROR: Could not load ground truth model: {e}")
        return False
    
    # Evaluate the truth model against itself
    print("Evaluating ground truth model...")
    print("(This may take a moment as it calls the LLM)")
    print()
    
    try:
        evaluation = evaluate_model(truth_model)
        if not evaluation:
            print("✗ ERROR: Evaluation returned None")
            return False
        print("✓ Evaluation completed successfully")
        print()
    except Exception as e:
        print(f"✗ ERROR during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Check the results
    print("=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    print()
    
    # Display qualitative description
    print("Qualitative Description:")
    print("-" * 80)
    print(evaluation.get("qualitative_description", "N/A"))
    print()
    
    # Check characteristic scores
    char_scores = evaluation.get("characteristic_scores", {})
    if not char_scores:
        print("✗ ERROR: No characteristic scores found in evaluation")
        return False
    
    print("Characteristic Scores:")
    print("-" * 80)
    print(f"{'Characteristic':<40} {'Score':<8} {'Category':<25} {'Expected'}")
    print("-" * 80)
    
    all_perfect = True
    for char_name in ECOLOGICAL_CHARACTERISTICS.keys():
        if char_name in char_scores:
            details = char_scores[char_name]
            score = details.get("score", 0)
            category = details.get("category", "UNKNOWN")
            explanation = details.get("explanation", "No explanation")
            
            # Check if score is 3 (TRUTH_MATCH)
            is_perfect = (score == 3 and category == "TRUTH_MATCH")
            status = "✓" if is_perfect else "✗"
            
            print(f"{char_name:<40} {score:<8} {category:<25} {status}")
            
            if not is_perfect:
                print(f"  → Explanation: {explanation}")
                all_perfect = False
        else:
            print(f"{char_name:<40} {'MISSING':<8} {'NOT_EVALUATED':<25} ✗")
            all_perfect = False
    
    print()
    
    # Calculate aggregate scores
    if "aggregate_scores" in evaluation:
        agg = evaluation["aggregate_scores"]
        raw_total = agg.get("raw_total", 0)
        normalized_total = agg.get("normalized_total", 0)
        final_score = agg.get("final_score", normalized_total)
    else:
        # Calculate manually if not provided
        raw_total = calculate_total_score(char_scores)
        sum_weights = sum(d["weight"] for d in ECOLOGICAL_CHARACTERISTICS.values())
        normalized_total = raw_total / (sum_weights * 3.0) if sum_weights > 0 else 0.0
        final_score = normalized_total
    
    print("Aggregate Scores:")
    print("-" * 80)
    print(f"Raw Total Score:        {raw_total:.4f}")
    print(f"Normalized Total:       {normalized_total:.4f}")
    print(f"Final Score:            {final_score:.4f}")
    print(f"Expected (perfect):     1.0000")
    print()
    
    # Check for extra components
    if "extra_components_count" in evaluation:
        extra_count = evaluation["extra_components_count"]
        extra_desc = evaluation.get("extra_components_description", "")
        print("Extra Components:")
        print("-" * 80)
        print(f"Count: {extra_count}")
        if extra_desc:
            print(f"Description: {extra_desc}")
        print()
    
    # Final assessment
    print("=" * 80)
    print("TEST ASSESSMENT")
    print("=" * 80)
    
    is_perfect_score = abs(final_score - 1.0) < 0.01  # Allow small floating point error
    
    if all_perfect and is_perfect_score:
        print("✓ SUCCESS: All characteristics scored TRUTH_MATCH (3)")
        print("✓ SUCCESS: Normalized total score is 1.0 (perfect)")
        print()
        print("The scoring system correctly identifies the ground truth model!")
        return True
    else:
        print("✗ FAILURE: Scoring system did not achieve perfect scores")
        if not all_perfect:
            print("  → Not all characteristics received score of 3 (TRUTH_MATCH)")
        if not is_perfect_score:
            print(f"  → Normalized total score {final_score:.4f} != 1.0")
        print()
        print("This suggests the scoring system needs adjustment.")
        return False

def save_results(evaluation):
    """Save the evaluation results to a JSON file for inspection."""
    output_path = "test_scripts/test_npz_truth_scoring_results.json"
    try:
        with open(output_path, 'w') as f:
            json.dump(evaluation, f, indent=2)
        print(f"\n✓ Full evaluation results saved to: {output_path}")
    except Exception as e:
        print(f"\n✗ Could not save results: {e}")

if __name__ == "__main__":
    print()
    success = test_truth_model_scoring()
    print()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)
