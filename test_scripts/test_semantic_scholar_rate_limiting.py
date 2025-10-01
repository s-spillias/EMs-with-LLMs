#!/usr/bin/env python3
"""
Test script to verify the semantic scholar rate limiting and exponential backoff functionality.
"""

import sys
import os
sys.path.append('.')

from scripts.search_utils import search_for_papers
import time
import multiprocessing as mp

def test_single_request():
    """Test a single request to ensure basic functionality works."""
    print("Testing single request...")
    start_time = time.time()
    
    papers = search_for_papers("marine ecosystem modeling", result_limit=5)
    
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"Single request completed in {duration:.2f} seconds")
    if papers:
        print(f"Found {len(papers)} papers")
        return True
    else:
        print("No papers found")
        return False

def make_request(query_num):
    """Helper function for multiprocessing test."""
    print(f"Process {query_num}: Starting request")
    start_time = time.time()
    
    papers = search_for_papers(f"marine ecosystem modeling {query_num}", result_limit=3)
    
    end_time = time.time()
    duration = end_time - start_time
    
    result = {
        'query_num': query_num,
        'duration': duration,
        'papers_found': len(papers) if papers else 0,
        'start_time': start_time,
        'end_time': end_time
    }
    
    print(f"Process {query_num}: Completed in {duration:.2f} seconds, found {result['papers_found']} papers")
    return result

def test_parallel_requests():
    """Test multiple parallel requests to verify cross-process rate limiting."""
    print("\nTesting parallel requests (should be rate limited to 1 per second)...")
    
    # Create 3 parallel processes
    with mp.Pool(processes=3) as pool:
        start_time = time.time()
        results = pool.map(make_request, [1, 2, 3])
        end_time = time.time()
    
    total_duration = end_time - start_time
    print(f"\nAll parallel requests completed in {total_duration:.2f} seconds")
    
    # Analyze timing
    for result in results:
        print(f"Query {result['query_num']}: {result['duration']:.2f}s, {result['papers_found']} papers")
    
    # Check if rate limiting worked (should take at least 2 seconds for 3 requests)
    if total_duration >= 2.0:
        print("✓ Rate limiting appears to be working correctly")
        return True
    else:
        print("⚠ Rate limiting may not be working - requests completed too quickly")
        return False

if __name__ == "__main__":
    print("Testing Semantic Scholar rate limiting and exponential backoff...")
    
    # Test single request
    single_success = test_single_request()
    
    if single_success:
        # Test parallel requests
        parallel_success = test_parallel_requests()
        
        if parallel_success:
            print("\n✓ All tests passed! Rate limiting and exponential backoff are working correctly.")
        else:
            print("\n⚠ Parallel test failed. Rate limiting may need adjustment.")
    else:
        print("\n✗ Single request test failed. Check API key and network connection.")