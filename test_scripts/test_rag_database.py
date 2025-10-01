#!/usr/bin/env python3
"""
Test script to verify the RAG database is working properly.
"""

import os
import json
from scripts.rag_utils import rag_query, get_model_choices_from_metadata, load_existing_index
import chromadb

def test_rag_database():
    """Test the RAG database setup and functionality."""
    
    print("Testing RAG Database Setup")
    print("=" * 50)
    
    # Test parameters
    population_dir = "POPULATIONS/POPULATION_0050"
    doc_store = "doc_store"
    
    # 1. Check if doc_store directory exists
    print(f"1. Checking doc_store directory: {doc_store}")
    if not os.path.exists(doc_store):
        print(f"   ERROR: doc_store directory '{doc_store}' does not exist!")
        return False
    else:
        print(f"   ✓ doc_store directory exists")
        
        # List files in doc_store
        files = os.listdir(doc_store)
        print(f"   Files in doc_store: {len(files)}")
        for file in files[:5]:  # Show first 5 files
            print(f"     - {file}")
        if len(files) > 5:
            print(f"     ... and {len(files) - 5} more files")
    
    # 2. Check population metadata
    print(f"\n2. Checking population metadata")
    metadata_file = os.path.join(population_dir, "population_metadata.json")
    if not os.path.exists(metadata_file):
        print(f"   ERROR: population_metadata.json not found in {population_dir}")
        return False
    
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    rag_choice = metadata.get('rag_choice')
    embed_choice = metadata.get('embed_choice')
    doc_store_dir = metadata.get('doc_store_dir')
    
    print(f"   ✓ RAG choice: {rag_choice}")
    print(f"   ✓ Embed choice: {embed_choice}")
    print(f"   ✓ Doc store dir: {doc_store_dir}")
    
    # 3. Check ChromaDB storage
    print(f"\n3. Checking ChromaDB storage")
    persist_dir = f'storage_chroma_{os.path.basename(doc_store)}'
    if not os.path.exists(persist_dir):
        print(f"   ERROR: ChromaDB storage directory '{persist_dir}' does not exist!")
        print(f"   This means the RAG index has not been created yet.")
        return False
    else:
        print(f"   ✓ ChromaDB storage directory exists: {persist_dir}")
        
        # List contents
        storage_files = os.listdir(persist_dir)
        print(f"   Storage files: {storage_files}")
    
    # 4. Test ChromaDB connection
    print(f"\n4. Testing ChromaDB connection")
    try:
        chroma_client = chromadb.PersistentClient()
        collections = chroma_client.list_collections()
        print(f"   ✓ ChromaDB client connected")
        print(f"   Collections: {[c.name for c in collections]}")
        
        # Check if our collection exists
        collection_name = os.path.basename(doc_store)
        try:
            collection = chroma_client.get_collection(collection_name)
            count = collection.count()
            print(f"   ✓ Collection '{collection_name}' exists with {count} documents")
        except Exception as e:
            print(f"   ERROR: Collection '{collection_name}' not found: {e}")
            return False
            
    except Exception as e:
        print(f"   ERROR: ChromaDB connection failed: {e}")
        return False
    
    # 5. Test loading the index
    print(f"\n5. Testing index loading")
    try:
        chroma_collection = chroma_client.get_collection(os.path.basename(doc_store))
        index = load_existing_index(doc_store, persist_dir, chroma_collection)
        print(f"   ✓ Index loaded successfully")
    except Exception as e:
        print(f"   ERROR: Failed to load index: {e}")
        return False
    
    # 6. Test a simple RAG query
    print(f"\n6. Testing RAG query")
    test_queries = [
        "coral growth rate",
        "population dynamics",
        "marine ecosystem",
        "Allee effect"
    ]
    
    for query in test_queries:
        print(f"   Testing query: '{query}'")
        try:
            results, citations = rag_query(query, doc_store, population_dir)
            print(f"     ✓ Query successful")
            print(f"     Results: {len(results)} items")
            print(f"     Citations: {len(citations)} items")
            
            if results and len(results) > 0:
                print(f"     First result preview: {results[0][:100]}...")
            
            if citations and len(citations) > 0:
                print(f"     Citations found:")
                for i, citation in enumerate(citations[:3]):  # Show first 3
                    print(f"       {i+1}. {citation}")
                if len(citations) > 3:
                    print(f"       ... and {len(citations) - 3} more")
            else:
                print(f"     WARNING: No citations found for this query")
                
        except Exception as e:
            print(f"     ERROR: Query failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    print(f"\n" + "=" * 50)
    print("RAG Database Test Summary:")
    print("✓ All basic tests passed")
    print("✓ RAG database appears to be working")
    
    return True

if __name__ == "__main__":
    success = test_rag_database()
    if success:
        print("\n🎉 RAG database is working correctly!")
    else:
        print("\n❌ RAG database has issues that need to be fixed.")