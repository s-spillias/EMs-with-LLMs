from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file


import os
import shutil
import subprocess
import time
import fcntl
import tempfile
from contextlib import contextmanager
os.environ['TORCH_DEVICE'] = 'cuda'  # Explicitly set to use GPU
import subprocess
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
# from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.embeddings.ollama import OllamaEmbedding
# from llama_index.llms.groq import Groq
# from llama_index.llms.bedrock import Bedrock
# from llama_index.llms.gemini import Gemini
from llama_index.llms.ollama import Ollama
import chromadb
# from marker.converters.pdf import PdfConverter
# from marker.models import create_model_dict
# from marker.output import text_from_rendered

# Imports for embedding models
from llama_index.embeddings.openai import OpenAIEmbedding

@contextmanager
def ollama_lock(timeout=300):
    """
    File-based lock to serialize access to Ollama server across multiple processes.
    Uses fcntl for POSIX systems.
    """
    lock_file_path = os.path.join(tempfile.gettempdir(), "ollama_rag_lock")
    
    try:
        # Create or open the lock file
        lock_file = open(lock_file_path, 'w')
        
        # Try to acquire the lock with timeout
        start_time = time.time()
        while True:
            try:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break  # Lock acquired
            except IOError:
                if time.time() - start_time > timeout:
                    raise TimeoutError(f"Could not acquire Ollama lock within {timeout} seconds")
                time.sleep(0.1)  # Wait a bit before retrying
        
        yield  # Execute the protected code
        
    finally:
        # Release the lock
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            lock_file.close()
        except:
            pass  # Ignore errors during cleanup

def get_model_choices_from_metadata(population_dir):
    metadata_path = os.path.join(population_dir, 'population_metadata.json')
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"population_metadata.json not found in {population_dir}")
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    rag_choice = metadata.get('rag_choice', 'anthropic_haiku')
    embed_choice = metadata.get('embed_choice', 'azure')
    
    return rag_choice, embed_choice



def _detect_ollama_bin(ollama_bin: str | None = None) -> str:
    """
    Decide which 'ollama' binary to use:
      1) explicit param
      2) OLLAMA_BIN env
      3) known HPC path (if it exists)
      4) 'ollama' from PATH
    """
    if ollama_bin and os.path.exists(ollama_bin):
        return ollama_bin

    env_bin = os.environ.get("OLLAMA_BIN")
    if env_bin and os.path.exists(env_bin):
        return env_bin

    # Known HPC install (optional shortcut)
    hpc_bin = "/scratch3/spi085/ollama/bin/ollama"
    if os.path.exists(hpc_bin):
        return hpc_bin

    path_bin = shutil.which("ollama")
    if path_bin:
        return path_bin

    raise FileNotFoundError(
        "Could not find the 'ollama' binary. Install Ollama or set OLLAMA_BIN to its path."
    )

def _ensure_ollama_server(
    base_url: str,
    start_local_ollama: bool = True,
    ollama_bin: str | None = None,
    ollama_models_dir: str | None = None,
    wait_seconds: float = 10.0,
):
    """
    Ensures there's an Ollama server reachable at base_url.
    If base_url points to localhost and start_local_ollama=True,
    attempt to start a local server if 'ollama list' fails.
    """
    is_local = base_url.startswith("http://127.0.0.1") or base_url.startswith("http://localhost")
    env = os.environ.copy()
    if ollama_models_dir:
        env["OLLAMA_MODELS"] = ollama_models_dir

    def _list_ok() -> bool:
        try:
            bin_path = _detect_ollama_bin(ollama_bin)
            subprocess.run(
                [bin_path, "list"],
                env=env,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    # If 'ollama list' works, a server is reachable; nothing to do.
    if _list_ok():
        return

    # If we're targeting localhost and allowed to manage the server, try to start it.
    if is_local and start_local_ollama:
        bin_path = _detect_ollama_bin(ollama_bin)
        subprocess.Popen(
            [bin_path, "serve"],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        # Wait a bit for the server to come up.
        deadline = time.time() + wait_seconds
        while time.time() < deadline:
            if _list_ok():
                return
            time.sleep(0.5)

        raise RuntimeError(
            f"Ollama server did not become ready within {wait_seconds}s at {base_url}."
        )
    else:
        # Non-local or not managing the server here; let the client try to connect.
        return

def _maybe_pull_ollama_model(
    model_name: str,
    base_url: str,
    ollama_bin: str | None = None,
    ollama_models_dir: str | None = None,
) -> None:
    """
    (Optional) For localhost only: if the model isn't present, try `ollama pull <model>`.
    Skips remote hosts to avoid requiring a local binary.
    """
    if not (base_url.startswith("http://127.0.0.1") or base_url.startswith("http://localhost")):
        return

    env = os.environ.copy()
    env["OLLAMA_HOST"] = base_url
    if ollama_models_dir:
        env["OLLAMA_MODELS"] = ollama_models_dir

    bin_path = _detect_ollama_bin(ollama_bin)

    # Check if model is available
    show_ok = subprocess.run(
        [bin_path, "show", model_name],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ).returncode == 0
    if show_ok:
        return

    # Pull if missing
    subprocess.run(
        [bin_path, "pull", model_name],
        env=env,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )


def get_rag_llm(
    llm_choice: str,
    *,
    # Optional overrides for Ollama behavior
    ollama_host: str | None = None,          # e.g., "http://localhost:11434" or "http://remote:11434"
    ollama_bin: str | None = None,           # path to 'ollama' binary if not on PATH
    ollama_models_dir: str | None = None,    # models dir; defaults to env or ~/.ollama (ollama default)
    start_local_ollama: bool = True,         # whether to attempt starting 'ollama serve' when localhost
    request_timeout: int | float = 120,      # applied to clients that support it
    auto_pull_ollama_model: bool = False,    # auto-pull the model if missing (localhost only)
):
    """
    Returns an LLM client for:
      - "anthropic_sonnet", "anthropic_haiku", "groq", "bedrock", "gemini"
      - "ollama:<model_name>"

    Notes for Ollama:
      * Set OLLAMA_HOST or pass `ollama_host` to target a remote server.
      * Set OLLAMA_BIN or pass `ollama_bin` if the binary isn't on PATH.
      * Set OLLAMA_MODELS or pass `ollama_models_dir` to control models dir.
      * We only start/inspect/pull via local binary for localhost targets.
    """
    llm_choice = (llm_choice or "").strip()

    if llm_choice == "anthropic_sonnet" or llm_choice == 'claude-3-5-haiku-20241022':
        return Anthropic(
            api_key=os.environ.get("ANTHROPIC_API_KEY"),
            model="claude-3-5-haiku-20241022"
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

    elif llm_choice.startswith("ollama:"):
        model_name = llm_choice.split(":", 1)[1].strip()
        if not model_name:
            raise ValueError("For Ollama, use 'ollama:<model_name>', e.g., 'ollama:llama3.1:8b'.")

        base_url = (
            ollama_host
            or os.environ.get("OLLAMA_HOST")
            or "http://127.0.0.1:11434"
        )

        resolved_models_dir = (
            ollama_models_dir
            or os.environ.get("OLLAMA_MODELS")
            or None
        )

        # Ensure server is running if local; no-ops for remote hosts
        _ensure_ollama_server(
            base_url=base_url,
            start_local_ollama=start_local_ollama,
            ollama_bin=ollama_bin,
            ollama_models_dir=resolved_models_dir,
        )

        # Optionally auto-pull model (localhost only)
        if auto_pull_ollama_model:
            _maybe_pull_ollama_model(
                model_name=model_name,
                base_url=base_url,
                ollama_bin=ollama_bin,
                ollama_models_dir=resolved_models_dir,
            )

        # Preserve your existing naming convention for the client
        client_model_name = f"ollama_chat/{model_name}"

        # If your Ollama class doesn't support base_url or request_timeout,
        # remove those kwargs and rely on OLLAMA_HOST env instead.
        return Ollama(
            model=client_model_name,
            base_url=base_url,
            request_timeout=request_timeout,
        )

    else:
        raise ValueError(f"Unsupported LLM (RAG) choice: {llm_choice}")


def get_embed_model(
    embed_choice: str,
    *,
    # Optional overrides for Ollama
    ollama_host: str | None = None,          # e.g., "http://localhost:11434" or "http://remote:11434"
    ollama_bin: str | None = None,           # path to 'ollama' binary if not on PATH
    ollama_models_dir: str | None = None,    # models dir; defaults to env or ~/.ollama
    start_local_ollama: bool = True,         # whether to attempt starting 'ollama serve' locally
    embed_batch_size: int = 10,              # batch size for embeddings
):
    """
    Returns an embedding model for 'azure', 'openai', or 'ollama:<model>'.
    Works on both HPC and non-HPC machines.

    Notes:
    - For Ollama:
        * Set OLLAMA_HOST or pass ollama_host to use a remote server.
        * Set OLLAMA_BIN or pass ollama_bin to choose the binary.
        * Set OLLAMA_MODELS or pass ollama_models_dir to choose models directory.
        * We only try to start 'ollama serve' for localhost targets when allowed.
    """
    if embed_choice == "azure":
        return AzureOpenAIEmbedding(
            deployment_name=os.environ.get("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT"),
            api_version=os.environ.get("AZURE_OPENAI_VERSION"),
            api_key=os.environ.get("AZURE_OPENAI_KEY"),
            azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
            embed_batch_size=1,
        )

    elif embed_choice == "openai":
        return OpenAIEmbedding(embed_batch_size=embed_batch_size)

    elif embed_choice.startswith("ollama:"):
        embed_model_name = embed_choice.split(":", 1)[1].strip()
        if not embed_model_name:
            raise ValueError("For Ollama, use 'ollama:<model_name>', e.g., 'ollama:nomic-embed-text'.")

        # Resolve base_url / host
        base_url = (
            ollama_host
            or os.environ.get("OLLAMA_HOST")
            or "http://127.0.0.1:11434"
        )

        # Resolve models dir (optional; do not force HPC path on non-HPC)
        resolved_models_dir = (
            ollama_models_dir
            or os.environ.get("OLLAMA_MODELS")
            or ("/scratch3/spi085/.ollama" if os.path.exists("/scratch3/spi085/.ollama") else None)
        )

        # Ensure server is up if local (no-ops for remote hosts)
        _ensure_ollama_server(
            base_url=base_url,
            start_local_ollama=start_local_ollama,
            ollama_bin=ollama_bin,
            ollama_models_dir=resolved_models_dir,
        )

        # Use the raw model name for embeddings (no ollama/ prefix)
        model_name_for_client = embed_model_name
        
        print(f"DEBUG: Creating OllamaEmbedding with:")
        print(f"  model_name: '{model_name_for_client}'")
        print(f"  base_url: '{base_url}'")
        print(f"  embed_batch_size: {embed_batch_size}")
        print(f"  current working directory: {os.getcwd()}")
        print(f"  OLLAMA_HOST env var: {os.environ.get('OLLAMA_HOST', 'Not set')}")
        print(f"  OLLAMA_MODELS env var: {os.environ.get('OLLAMA_MODELS', 'Not set')}")

        # Construct the embedding client. If your library doesn't support `base_url`,
        # remove it and rely on OLLAMA_HOST in the environment.
        return OllamaEmbedding(
            model_name=model_name_for_client,
            base_url=base_url,
            embed_batch_size=embed_batch_size,
        )

    else:
        raise ValueError(f"Unsupported embedding model choice: {embed_choice}")


def rag_setup(directory, population_dir):
    # Ensure the directory path is absolute
    directory = os.path.abspath(directory)
    
    if not os.path.exists(directory):
        raise ValueError(f"Directory does not exist: {directory}")
    
    # Get model choices from population_metadata.json
    rag_choice, embed_choice = get_model_choices_from_metadata(population_dir)
    
    # Use file lock when accessing Ollama-based models
    use_lock = (rag_choice and rag_choice.startswith("ollama:")) or (embed_choice and embed_choice.startswith("ollama:"))
    
    if use_lock:
        with ollama_lock():
            return _rag_setup_internal(directory, population_dir, rag_choice, embed_choice)
    else:
        return _rag_setup_internal(directory, population_dir, rag_choice, embed_choice)

def _rag_setup_internal(directory, population_dir, rag_choice, embed_choice):
    """Internal RAG setup function that does the actual work."""
    chroma_client = chromadb.PersistentClient()
    chroma_collection = chroma_client.get_or_create_collection(os.path.basename(directory))
    
    # Set up LLM and embedding model
    llm = get_rag_llm(rag_choice)
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
    rag_choice, embed_choice = get_model_choices_from_metadata(population_dir)
    
    # Use file lock when accessing Ollama-based models
    use_lock = (rag_choice and rag_choice.startswith("ollama:")) or (embed_choice and embed_choice.startswith("ollama:"))
    
    if use_lock:
        with ollama_lock():
            return _rag_query_internal(query, doc_store, population_dir, rag_choice)
    else:
        return _rag_query_internal(query, doc_store, population_dir, rag_choice)

def _rag_query_internal(query, doc_store, population_dir, rag_choice):
    """Internal RAG query function that does the actual work."""
    llm = get_rag_llm(rag_choice)

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
    
    # Extract citations more robustly
    citations = []
    try:
        # Try different ways to extract citations from the response
        if hasattr(response, 'source_nodes'):
            # Extract from source nodes
            for node in response.source_nodes:
                if hasattr(node, 'metadata') and 'file_name' in node.metadata:
                    citations.append(node.metadata['file_name'])
                elif hasattr(node, 'node') and hasattr(node.node, 'metadata') and 'file_name' in node.node.metadata:
                    citations.append(node.node.metadata['file_name'])
        
        # Fallback: try the original method
        if not citations and hasattr(response, 'metadata') and response.metadata:
            for entry in response.metadata.values():
                if isinstance(entry, dict) and 'file_name' in entry:
                    citations.append(entry['file_name'])
        
        # Remove duplicates and filter out empty citations
        citations = list(set(citation for citation in citations if citation))
        
    except Exception as e:
        print(f"Warning: Could not extract citations from RAG response: {e}")
        citations = []
    
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
