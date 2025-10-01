#!/usr/bin/env python3
"""
Debug script to test RAG citation extraction.
"""

import os
import json
from scripts.rag_utils import rag_query, get_model_choices_from_metadata

def debug_rag_citations():
    """Debug the RAG citation extraction."""
    
    # Test parameters
    population_dir = "POPULATIONS/POPULATION_0050"
    doc_store = "doc_store"
    query = "Minimum population density for successful reproduction"
    
    print(f"Testing RAG query with:")
    print(f"  Population dir: {population_dir}")
    print(f"  Doc store: {doc_store}")
    print(f"  Query: {query}")
    print("=" * 50)
    
    try:
        # Test the rag_query function
        results, citations = rag_query(query, doc_store, population_dir)
        
        print(f"Results: {len(results)} items")
        for i, result in enumerate(results):
            print(f"  Result {i+1}: {result[:100]}...")
        
        print(f"\nCitations: {len(citations)} items")
        for i, citation in enumerate(citations):
            print(f"  Citation {i+1}: {citation}")
            
        if not citations:
            print("\nWARNING: No citations found!")
            print("This indicates the citation extraction is not working properly.")
            
        return results, citations
        
    except Exception as e:
        print(f"Error during RAG query: {e}")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    debug_rag_citations()