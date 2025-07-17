from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

import os
os.environ['TORCH_DEVICE'] = 'cuda'  # Explicitly set to use GPU

import json
from typing import List
from llama_index.core import (
    SimpleDirectoryReader, 
    StorageContext,
    Settings,
    load_index_from_storage,
    VectorStoreIndex,
    Document,
)
from llama_index.core.node_parser import MarkdownElementNodeParser
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import IndexNode
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.anthropic import Anthropic
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.llms.groq import Groq
from llama_index.llms.bedrock import Bedrock
from llama_index.llms.gemini import Gemini
import chromadb
# from marker.converters.pdf import PdfConverter
# from marker.models import create_model_dict
# from marker.output import text_from_rendered

# Imports for embedding models
from llama_index.embeddings.openai import OpenAIEmbedding

def get_model_choices_from_metadata(population_dir):
    metadata_path = os.path.join(population_dir, 'population_metadata.json')
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"population_metadata.json not found in {population_dir}")
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    rag_choice = metadata.get('rag_choice', 'anthropic_haiku')
    embed_choice = metadata.get('embed_choice', 'azure')
    
    return rag_choice, embed_choice

def get_llm(llm_choice):
    if llm_choice == "anthropic_sonnet":
        return Anthropic(
            api_key=os.environ.get("ANTHROPIC_API_KEY"),
            model="claude-3-5-sonnet-20240620"
        )
    elif llm_choice == "anthropic_haiku":
        return Anthropic(
            api_key=os.environ.get("ANTHROPIC_API_KEY"),
            model="claude-3-5-haiku-20240620"
        )
    elif llm_choice == "groq":
        return Groq(
            model="llama3-70b-8192",
            api_key=os.environ.get("GROQ_API_KEY")
        )
    elif llm_choice == "bedrock":
        return Bedrock(
            model="amazon.titan-text-express-v1",
            profile_name=os.environ.get("AWS_PROFILE")
        )
    elif llm_choice == "gemini":
        return Gemini(
            model="models/gemini-1.5-flash",
            api_key=os.environ.get("GOOGLE_API_KEY")
        )
    else:
        raise ValueError(f"Unsupported LLM (RAG) choice: {llm_choice}")

def get_embed_model(embed_choice):
    if embed_choice == "azure":
        return AzureOpenAIEmbedding(
            deployment_name=os.environ.get("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT"),
            api_version=os.environ.get("AZURE_OPENAI_VERSION"),
            api_key=os.environ.get("AZURE_OPENAI_KEY"),
            azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
            embed_batch_size=1,
        )
    elif embed_choice == "openai":
        return OpenAIEmbedding(embed_batch_size=10)
    else:
        raise ValueError(f"Unsupported embedding model choice: {embed_choice}")

def rag_setup(directory, population_dir):
    # Ensure the directory path is absolute
    directory = os.path.abspath(directory)
    
    if not os.path.exists(directory):
        raise ValueError(f"Directory does not exist: {directory}")
    
    # Get model choices from population_metadata.json
    rag_choice, embed_choice = get_model_choices_from_metadata(population_dir)
    
    
    chroma_client = chromadb.PersistentClient()
    chroma_collection = chroma_client.get_or_create_collection(os.path.basename(directory))
    
    # Set up LLM and embedding model
    llm = get_llm(rag_choice)
    embed_model = get_embed_model(embed_choice)

    Settings.llm = llm
    Settings.embed_model = embed_model
    
    store = directory
    persist_dir = os.path.join(os.path.dirname(directory), f'storage_chroma_{os.path.basename(directory)}')
    
    # Load existing index without parsing new documents
    index = load_existing_index(store, persist_dir, chroma_collection)
    
    return index, llm

def load_existing_index(store, persist_dir, chroma_collection):
    """Load an existing index without parsing new documents."""
    if not os.path.exists(persist_dir):
        raise ValueError(f"No existing index found at {persist_dir}")
    
    print(f"Loading existing index from {persist_dir}...")
    storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    index = VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context)
    print("Index Successfully Loaded")
    
    return index

def rag_query(query, doc_store, population_dir):
    # Get RAG LLM choice from population metadata
    rag_choice, _ = get_model_choices_from_metadata(population_dir)
    llm = get_llm(rag_choice)

    # Get or create index
    chroma_client = chromadb.PersistentClient()
    chroma_collection = chroma_client.get_or_create_collection(os.path.basename(doc_store))
    persist_dir = os.path.join(os.path.dirname(doc_store), f'storage_chroma_{os.path.basename(doc_store)}')
    
    # Load existing index without parsing new documents
    index = load_existing_index(doc_store, persist_dir, chroma_collection)

    # Query the index
    query_engine = index.as_query_engine(llm=llm)
    response = query_engine.query(query)
    out = response.response
    citations = list(set(entry['file_name'] for entry in response.metadata.values()))
    return [out], citations

def get_or_create_index(store, persist_dir, chroma_collection):
    """Create a new index by parsing documents."""
    index_files_path = os.path.join(persist_dir, 'indexed_files.json')
    
    if os.path.exists(persist_dir):
        print(f"Loading existing index from {persist_dir}...")
        storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        index = VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context)
        
        if os.path.exists(index_files_path):
            with open(index_files_path, 'r') as f:
                indexed_files = set(json.load(f))
        else:
            indexed_files = set()
        
        current_files = set(get_all_files(store))
        new_files = current_files - indexed_files
        
        if new_files:
            print(f"Found {len(new_files)} new files. Updating index...")
            new_documents = load_documents(list(new_files))
            if new_documents:
                index.insert_nodes(new_documents)
                
                indexed_files.update(new_files)
                with open(index_files_path, 'w') as f:
                    json.dump(list(indexed_files), f)
                print("Index Successfully Updated...")
            else:
                print('Documents failed to load.')
        print("Index Successfully Loaded")
    else:
        print(f"Creating new index for {store}...")
        documents = load_documents(store)
        
        if not documents:
            raise ValueError(f"No documents found in {store}")
        
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex(documents, storage_context=storage_context)
        index.storage_context.persist(persist_dir=persist_dir)
        
        indexed_files = get_all_files(store)
        with open(index_files_path, 'w') as f:
            json.dump(indexed_files, f)
        
        print(f"Index Successfully Created for {store}")
    
    return index

def get_all_files(directory):
    file_list = []
    for root, _, files in os.walk(directory):
        for file in files:
            file_list.append(os.path.join(root, file))
    return file_list

def load_documents(input_files):
    documents = []
    # Initialize marker with explicit GPU configuration and output settings
    converter = PdfConverter(
        artifact_dict=create_model_dict(),
        config={
            "TORCH_DEVICE": "cuda",  # Explicitly set to use GPU
            "output_format": "json",  # Set output format to JSON
            "output_dir": "parsings"  # Set output directory
        }
    )
    
    # Initialize text splitter for chunking
    text_splitter = SentenceSplitter(
        chunk_size=512,  # Adjust this value based on your needs
        chunk_overlap=50  # Some overlap to maintain context between chunks
    )
    
    try:
        if isinstance(input_files, str) and os.path.isdir(input_files):
            # Handle directory input
            for root, _, files in os.walk(input_files):
                for file in files:
                    if file.lower().endswith('.pdf'):
                        file_path = os.path.join(root, file)
                        try:
                            rendered = converter(file_path)
                            text, _, _ = text_from_rendered(rendered)
                            
                            # Split text into chunks
                            text_chunks = text_splitter.split_text(text)
                            
                            # Create Document objects for each chunk
                            for i, chunk in enumerate(text_chunks):
                                documents.append(Document(
                                    text=chunk,
                                    metadata={
                                        "file_name": file_path,
                                        "chunk_index": i
                                    }
                                ))
                        except Exception as e:
                            print(f"Error processing {file_path}: {e}")
        else:
            # Handle list of files input
            for file_path in input_files:
                if file_path.lower().endswith('.pdf'):
                    try:
                        rendered = converter(file_path)
                        text, _, _ = text_from_rendered(rendered)
                        
                        # Split text into chunks
                        text_chunks = text_splitter.split_text(text)
                        
                        # Create Document objects for each chunk
                        for i, chunk in enumerate(text_chunks):
                            documents.append(Document(
                                text=chunk,
                                metadata={
                                    "file_name": file_path,
                                    "chunk_index": i
                                }
                            ))
                    except Exception as e:
                        print(f"Error processing {file_path}: {e}")
        
        return documents
    except Exception as e:
        print(f"Error loading documents: {e}")
        print(f"Input files: {input_files}")
        return None

def rag_master_parameters(query):
    master_file = "master_parameters.json"
    
    if not os.path.exists(master_file):
        raise FileNotFoundError(f"Master parameters file not found: {master_file}")
    
    with open(master_file, 'r') as file:
        master_params = json.load(file)
    
    # Convert the JSON structure to a list of documents
    documents = []
    for branch, params in master_params.items():
        for param in params:
            doc = f"Branch: {branch}\n"
            doc += f"Parameter: {param['parameter']}\n"
            doc += f"Description: {param['description']}\n"
            doc += f"Value: {param['value']}\n"
            doc += f"Min: {param['min']}\n"
            doc += f"Max: {param['max']}\n"
            doc += f"Source: {param['source']}\n"
            documents.append(doc)
    
    # Create a VectorStoreIndex from the documents
    index = VectorStoreIndex.from_documents(documents)
    
    # Create a query engine
    query_engine = index.as_query_engine()
    
    # Perform the query
    response = query_engine.query(query)
    
    # Extract the response and relevant parameters
    out = response.response
    relevant_params = [doc for doc in documents if doc in out]
    
    return [out], relevant_params
