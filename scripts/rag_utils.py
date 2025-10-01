# scripts/rag_utils.py
"""
Fork-safe RAG utilities with automatic readiness detection.

What's new:
- If a prior index already exists (either at the legacy/default Chroma location or
  at the new {persist_dir}/chroma path, and a valid LlamaIndex storage folder),
  rag_prepare(...) will auto-write the .rag_ready sentinel and SKIP ingest.
- Set RAG_FORCE_REINDEX=1 to force a fresh ingest.
- Parent pre-fork: rag_prepare(doc_store_dir, population_dir)
- Workers post-fork: rag_query(query, doc_store_dir, population_dir)
- Back-compat: rag_setup(...) now calls rag_prepare(...) and returns (None, None).

Layout:
  doc_store_dir/                     # your documents (PDFs)
  ../storage_chroma_<name>/          # LlamaIndex storage
     chroma/                         # Chroma DB (new default path we use)
     ... (LI storage files)
  .rag_ready                         # sentinel in doc_store_dir
"""

from __future__ import annotations

import os
import json
import time
import shutil
import threading
from contextlib import contextmanager
from typing import List, Tuple, Optional
from chromadb.config import Settings as ChromaSettings
from dotenv import load_dotenv
load_dotenv()

# ----------------------------- LlamaIndex / Chroma -----------------------------
from llama_index.core import (
    StorageContext,
    Settings,
    VectorStoreIndex,
    Document,
    load_index_from_storage,
)
from llama_index.vector_stores.chroma import ChromaVectorStore

import chromadb

# ----------------------------- LLMs & Embeddings ------------------------------
# Azure OpenAI Embedding
try:
    from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding # type: ignore
    _HAS_AZURE_EMBED = True
except Exception:
    _HAS_AZURE_EMBED = False

# OpenAI Embedding
try:
    from llama_index.embeddings.openai import OpenAIEmbedding
    _HAS_OPENAI_EMBED = True
except Exception:
    _HAS_OPENAI_EMBED = False

# Ollama Embedding
try:
    from llama_index.embeddings.ollama import OllamaEmbedding
    _HAS_OLLAMA_EMBED = True
except Exception:
    _HAS_OLLAMA_EMBED = False

# Anthropic LLM
try:
    from llama_index.llms.anthropic import Anthropic
    _HAS_ANTHROPIC = True
except Exception:
    _HAS_ANTHROPIC = False

# Groq LLM
try:
    from llama_index.llms.groq import Groq
    _HAS_GROQ = True
except Exception:
    _HAS_GROQ = False

# Bedrock LLM
try:
    from llama_index.llms.bedrock import Bedrock
    _HAS_BEDROCK = True
except Exception:
    _HAS_BEDROCK = False

# Gemini LLM
try:
    from llama_index.llms.gemini import Gemini
    _HAS_GEMINI = True
except Exception:
    _HAS_GEMINI = False

# Ollama LLM
try:
    from llama_index.llms.ollama import Ollama
    _HAS_OLLAMA_LLM = True
except Exception:
    _HAS_OLLAMA_LLM = False

# ----------------------------- Optional PDF pipeline ---------------------------
try:
    from marker.converters.pdf import PdfConverter
    from marker.models import create_model_dict
    from marker.output import text_from_rendered
    _HAS_MARKER = True
except Exception:
    _HAS_MARKER = False
# ------------------------------------------------------------------------------
# Globals & process-local caching
# ------------------------------------------------------------------------------

_RAG_CLIENT = None
_RAG_CLIENT_PID = None
_RAG_LOCK = threading.Lock()



import os, json, time
from llama_index.core import StorageContext, load_index_from_storage

_READY_SENTINEL = ".rag_ready"

def _abs(s: str) -> str:
    return os.path.abspath(s) if s else s

def _paths(doc_store_dir: str):
    ds = _abs(doc_store_dir)
    name = os.path.basename(ds.rstrip(os.sep)) or "doc_store"
    persist_dir = os.path.join(os.path.dirname(ds), f"storage_chroma_{name}")
    chroma_path = os.path.join(persist_dir, "chroma")
    return ds, name, persist_dir, chroma_path

def _ready_path(doc_store_dir: str) -> str:
    return os.path.join(_abs(doc_store_dir), _READY_SENTINEL)

def rag_is_ready(doc_store_dir: str) -> bool:
    return bool(doc_store_dir) and os.path.exists(_ready_path(doc_store_dir))

def _write_ready_sentinel(doc_store_dir: str) -> None:
    os.makedirs(_abs(doc_store_dir), exist_ok=True)
    with open(_ready_path(doc_store_dir), "w") as f:
        json.dump({"ready": True, "ts": time.time()}, f)

def _li_storage_fs_ok(persist_dir: str) -> bool:
    """Filesystem-only check that LlamaIndex storage exists and looks valid."""
    if not os.path.isdir(persist_dir):
        return False
    # Look for common LlamaIndex files without loading anything:
    expected = {"docstore.json", "index_store.json"}
    present = set(os.listdir(persist_dir))
    if expected.intersection(present):
        return True
    # Fallback: any non-empty dir is a weak signal
    try:
        return any(os.scandir(persist_dir))
    except Exception:
        return False

def _chroma_fs_ok(chroma_path: str) -> bool:
    """Filesystem-only check for a Chroma SQLite/duckdb store."""
    if not os.path.isdir(chroma_path):
        return False
    names = set(os.listdir(chroma_path))
    # Chroma often creates sqlite files or subdirs; accept a few common markers:
    markers = {
        "chroma.sqlite3",     # sqlite backend
        "index", "index.pkl", # faiss/annoy side data (varies by version)
        "data", "blob_storage"
    }
    if names.intersection(markers):
        return True
    # If there are any files at all, assume it's likely initialized
    try:
        return any(entry.is_file() for entry in os.scandir(chroma_path))
    except Exception:
        return False

def _try_detect_existing_index_fs_only(doc_store_dir: str) -> bool:
    """Detect an existing index purely via filesystem heuristics (no clients)."""
    ds, coll_name, persist_dir, chroma_path = _paths(doc_store_dir)
    if _li_storage_fs_ok(persist_dir) and _chroma_fs_ok(chroma_path):
        _write_ready_sentinel(ds)
        return True
    # Legacy layouts: accept LI storage presence alone as “likely” ready
    if _li_storage_fs_ok(persist_dir):
        _write_ready_sentinel(ds)
        return True
    return False

def rag_prepare(doc_store_dir: str, population_dir: str) -> None:
    """
    Ensure persistent index exists and is ready:
      1) If .rag_ready exists -> done.
      2) Else try FS-only detection. If found -> write sentinel.
      3) Else (or if RAG_FORCE_REINDEX=1) -> call rag_ingest(...)
    NOTE: This function must not construct chromadb clients.
    """
    if not doc_store_dir:
        return

    force = os.getenv("RAG_FORCE_REINDEX", "0").strip().lower() in ("1", "true", "yes")
    if force:
        rag_ingest(doc_store_dir, population_dir)
        return

    if rag_is_ready(doc_store_dir):
        return
    if _try_detect_existing_index_fs_only(doc_store_dir):
        return

    # No convincing evidence of an existing index -> build it now
    rag_ingest(doc_store_dir, population_dir)


# ------------------------------------------------------------------------------
# Metadata
# ------------------------------------------------------------------------------
def get_model_choices_from_metadata(population_dir: str) -> Tuple[str, str]:
    """
    Read rag_choice and embed_choice from population_metadata.json
    """
    metadata_path = os.path.join(_abs(population_dir), "population_metadata.json")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"population_metadata.json not found in {population_dir}")
    with open(metadata_path, "r") as f:
        meta = json.load(f)
    rag_choice = meta.get("rag_choice", "anthropic_haiku")
    embed_choice = meta.get("embed_choice", "openai")
    return rag_choice, embed_choice

# ------------------------------------------------------------------------------
# Ollama helpers
# ------------------------------------------------------------------------------
def _detect_ollama_bin(ollama_bin: Optional[str] = None) -> str:
    if ollama_bin and os.path.exists(ollama_bin):
        return ollama_bin
    env_bin = os.environ.get("OLLAMA_BIN")
    if env_bin and os.path.exists(env_bin):
        return env_bin
    path_bin = shutil.which("ollama")
    if path_bin:
        return path_bin
    raise FileNotFoundError("Could not find the 'ollama' binary. Set OLLAMA_BIN or install Ollama.")

def _ensure_ollama_server(base_url: str,
                          start_local_ollama: bool = True,
                          ollama_bin: Optional[str] = None,
                          ollama_models_dir: Optional[str] = None,
                          wait_seconds: float = 10.0) -> None:
    import subprocess
    is_local = base_url.startswith("http://127.0.0.1") or base_url.startswith("http://localhost")
    env = os.environ.copy()
    if ollama_models_dir:
        env["OLLAMA_MODELS"] = ollama_models_dir

    def _list_ok() -> bool:
        try:
            bin_path = _detect_ollama_bin(ollama_bin)
            subprocess.run([bin_path, "list"], env=env, check=True,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True
        except Exception:
            return False

    if _list_ok():
        return
    if is_local and start_local_ollama:
        bin_path = _detect_ollama_bin(ollama_bin)
        subprocess.Popen([bin_path, "serve"], env=env,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        deadline = time.time() + wait_seconds
        while time.time() < deadline:
            if _list_ok():
                return
            time.sleep(0.5)
        raise RuntimeError(f"Ollama server did not become ready within {wait_seconds}s at {base_url}")

def _maybe_pull_ollama_model(model_name: str, base_url: str,
                             ollama_bin: Optional[str] = None,
                             ollama_models_dir: Optional[str] = None) -> None:
    if not (base_url.startswith("http://127.0.0.1") or base_url.startswith("http://localhost")):
        return
    import subprocess
    env = os.environ.copy()
    env["OLLAMA_HOST"] = base_url
    if ollama_models_dir:
        env["OLLAMA_MODELS"] = ollama_models_dir
    bin_path = _detect_ollama_bin(ollama_bin)
    show_ok = subprocess.run([bin_path, "show", model_name],
                             env=env, stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL).returncode == 0
    if not show_ok:
        subprocess.run([bin_path, "pull", model_name], env=env, check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

# ------------------------------------------------------------------------------
# Model factories
# ------------------------------------------------------------------------------
def get_rag_llm(llm_choice: str,
                *,
                ollama_host: Optional[str] = None,
                ollama_bin: Optional[str] = None,
                ollama_models_dir: Optional[str] = None,
                start_local_ollama: bool = True,
                request_timeout: float = 120.0,
                auto_pull_ollama_model: bool = False):
    llm_choice = (llm_choice or "").strip()
    if llm_choice in ("anthropic_sonnet", "claude-3-5-sonnet-20241022"):
        return Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"),
                         model="claude-3-5-sonnet-20241022")
    if llm_choice in ("anthropic_haiku", "claude-3-5-haiku-20241022"):
        return Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"),
                         model="claude-3-5-haiku-20241022")
    if llm_choice.startswith("ollama:"):
        model_name = llm_choice.split(":", 1)[1].strip()
        if not model_name:
            raise ValueError("For Ollama, use 'ollama:<model_name>'")
        base_url = (ollama_host or os.environ.get("OLLAMA_HOST") or "http://127.0.0.1:11434")
        resolved_models_dir = ollama_models_dir or os.environ.get("OLLAMA_MODELS") or None
        _ensure_ollama_server(base_url, start_local_ollama, ollama_bin, resolved_models_dir)
        if auto_pull_ollama_model:
            _maybe_pull_ollama_model(model_name, base_url, ollama_bin, resolved_models_dir)
        return Ollama(model=f"ollama_chat/{model_name}", base_url=base_url, request_timeout=request_timeout)
    raise ValueError(f"Unsupported LLM (RAG) choice: {llm_choice}")

def get_embed_model(embed_choice: str,
                    *,
                    ollama_host: Optional[str] = None,
                    ollama_bin: Optional[str] = None,
                    ollama_models_dir: Optional[str] = None,
                    start_local_ollama: bool = True,
                    embed_batch_size: int = 10):
    if embed_choice == "openai":
        return OpenAIEmbedding()
    if embed_choice == "azure":
        if not _HAS_AZURE_EMBED:
            raise RuntimeError("Azure embedding backend not installed (llama_index.embeddings.azure_openai).")
        return AzureOpenAIEmbedding(
            deployment_name=os.environ.get("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT"),
            api_version=os.environ.get("AZURE_OPENAI_VERSION"),
            api_key=os.environ.get("AZURE_OPENAI_KEY"),
            azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
            embed_batch_size=1,
        )
    if embed_choice.startswith("ollama:"):
        embed_model_name = embed_choice.split(":", 1)[1].strip()
        if not embed_model_name:
            raise ValueError("For Ollama, use 'ollama:<model_name>' (e.g., 'ollama:nomic-embed-text').")
        base_url = (ollama_host or os.environ.get("OLLAMA_HOST") or "http://127.0.0.1:11434")
        resolved_models_dir = ollama_models_dir or os.environ.get("OLLAMA_MODELS") or None
        _ensure_ollama_server(base_url, start_local_ollama, ollama_bin, resolved_models_dir)
        return OllamaEmbedding(model_name=embed_model_name, base_url=base_url, embed_batch_size=embed_batch_size)
    raise ValueError(f"Unsupported embedding model choice: {embed_choice}")

# ------------------------------------------------------------------------------
# PDF -> text chunking
# ------------------------------------------------------------------------------
def _require_marker():
    if not _HAS_MARKER:
        raise RuntimeError("PDF parsing 'marker' is not installed. Replace load_documents() or install it.")

def load_documents(input_path_or_files) -> List[Document]:
    _require_marker()
    converter = PdfConverter(
        artifact_dict=create_model_dict(),
        config={"TORCH_DEVICE": os.environ.get("TORCH_DEVICE", "cpu"),
                "output_format": "json",
                "output_dir": "parsings"}
    )

    from llama_index.core.node_parser import SentenceSplitter
    splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)

    docs: List[Document] = []

    def _process_pdf(path: str) -> List[Document]:
        rendered = converter(path)
        text, _, _ = text_from_rendered(rendered)
        chunks = splitter.split_text(text or "")
        return [Document(text=ch, metadata={"file_name": path, "chunk_index": i})
                for i, ch in enumerate(chunks) if ch.strip()]

    try:
        if isinstance(input_path_or_files, str) and os.path.isdir(input_path_or_files):
            for root, _, files in os.walk(input_path_or_files):
                for fn in files:
                    if fn.lower().endswith(".pdf"):
                        fpath = os.path.join(root, fn)
                        try:
                            docs.extend(_process_pdf(fpath))
                        except Exception as e:
                            print(f"Error processing {fpath}: {e}")
        else:
            for fpath in (input_path_or_files or []):
                p = str(fpath)
                if p.lower().endswith(".pdf"):
                    try:
                        docs.extend(_process_pdf(p))
                    except Exception as e:
                        print(f"Error processing {p}: {e}")
    except Exception as e:
        print(f"Error loading documents: {e}")
        return []
    return docs

# ------------------------------------------------------------------------------
# Ingest locking (avoid concurrent rebuilds)
# ------------------------------------------------------------------------------
@contextmanager
def _ingest_lock(doc_store_dir: str):
    lock_path = os.path.join(_abs(doc_store_dir), ".ingest.lock")
    os.makedirs(os.path.dirname(lock_path), exist_ok=True)
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        try:
            import fcntl  # POSIX
            fcntl.flock(fd, fcntl.LOCK_EX)
        except Exception:
            pass  # No-op on non-POSIX
        yield
    finally:
        try:
            import fcntl
            fcntl.flock(fd, fcntl.LOCK_UN)
        except Exception:
            pass
        os.close(fd)

# ------------------------------------------------------------------------------
# Readiness autodetection (handles legacy & new paths)
# ------------------------------------------------------------------------------
def _li_storage_looks_valid(persist_dir: str) -> bool:
    """
    Heuristic: LlamaIndex storage exists and is non-empty.
    """
    if not os.path.isdir(persist_dir):
        return False
    try:
        # Try loading the storage; cheap and reliable
        sc = StorageContext.from_defaults(persist_dir=persist_dir)
        _ = load_index_from_storage(sc)  # will raise if incompatible/empty
        return True
    except Exception:
        # Fallback: non-empty directory is a weak signal
        try:
            return any(os.scandir(persist_dir))
        except Exception:
            return False

def _chroma_collection_has_data(client: chromadb.PersistentClient, name: str) -> bool:
    try:
        coll = client.get_or_create_collection(name)
        # Chroma Collection has count() in recent versions; if not, try peek
        try:
            return coll.count() > 0
        except Exception:
            # As fallback, try a tiny query/peek
            res = coll.peek()
            # peek() returns dict with ids/texts if any; be permissive
            return bool(res and any(res.values()))
    except Exception:
        return False

# ------------------------------------------------------------------------------
# Readiness detection with client initialization
# ------------------------------------------------------------------------------
def _try_detect_existing_index(doc_store_dir: str) -> bool:
    """
    Detect an existing index by trying to initialize clients.
    More thorough than fs-only checks but may create temporary clients.
    """
    ds, coll_name, persist_dir, chroma_path = _paths(doc_store_dir)
    
    # First try filesystem-only checks (faster, no clients)
    if _try_detect_existing_index_fs_only(doc_store_dir):
        return True
        
    # Try loading LlamaIndex storage
    if _li_storage_looks_valid(persist_dir):
        _write_ready_sentinel(ds)
        return True
        
    # Try checking Chroma collection
    try:
        client = chromadb.PersistentClient(path=chroma_path)
        if _chroma_collection_has_data(client, coll_name):
            _write_ready_sentinel(ds)
            return True
    except Exception:
        pass
        
    return False

# ------------------------------------------------------------------------------
# INGEST (parent pre-fork)
# ------------------------------------------------------------------------------
def rag_ingest(doc_store_dir: str, population_dir: str) -> None:
    """
    Build/refresh the on-disk index from documents in doc_store_dir.
    Must return with NO long-lived threads/pools/handles.
    """
    if not doc_store_dir:
        return
    ds, coll_name, persist_dir, chroma_path = _paths(doc_store_dir)
    os.makedirs(ds, exist_ok=True)
    os.makedirs(persist_dir, exist_ok=True)

    with _ingest_lock(ds):
        # If some other process prepared it while we waited, bail out.
        if rag_is_ready(ds):
            return

        # Models
        _, embed_choice = get_model_choices_from_metadata(population_dir)
        Settings.embed_model = get_embed_model(embed_choice)

        # Corpus
        documents = load_documents(ds)
        if not documents:
            raise ValueError(f"No documents found to index in {ds}")

        # Vector store (new path)
        client = chromadb.PersistentClient(path=chroma_path)
        collection = client.get_or_create_collection(coll_name)
        vector_store = ChromaVectorStore(chroma_collection=collection)

        # Build index and persist
        storage_ctx = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex(documents, storage_context=storage_ctx)
        index.storage_context.persist(persist_dir=persist_dir)

        # Mark ready last
        _write_ready_sentinel(ds)



# ------------------------------------------------------------------------------
# READ-ONLY OPEN (worker post-fork): process-local cached client
# ------------------------------------------------------------------------------
class _ReadonlyRAGClient:
    def __init__(self, doc_store_dir: str, population_dir: str):
        self.doc_store_dir, self.collection_name, self.persist_dir, self.chroma_path = _paths(doc_store_dir)
        self.population_dir = population_dir
        # Embeddings for query-time similarity
        _, embed_choice = get_model_choices_from_metadata(population_dir)
        Settings.embed_model = get_embed_model(embed_choice)

        # Open vector store at NEW path first; fall back to legacy default path if empty
        
        self._client = chromadb.PersistentClient(
            path=self.chroma_path,
            settings=ChromaSettings(anonymized_telemetry=False)
        )

        coll = self._client.get_or_create_collection(self.collection_name)
        if not _chroma_collection_has_data(self._client, self.collection_name):
            # Try legacy/default client as a fallback (your older builds)
            legacy_client = chromadb.PersistentClient()
            if _chroma_collection_has_data(legacy_client, self.collection_name):
                self._client = legacy_client
                coll = self._client.get_or_create_collection(self.collection_name)

        vector_store = ChromaVectorStore(chroma_collection=coll)
        storage_ctx = StorageContext.from_defaults(persist_dir=self.persist_dir)
        self._index = VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_ctx)

    def retrieve(self, query: str, top_k: int = 5) -> Tuple[List[str], List[str]]:
        retriever = self._index.as_retriever(similarity_top_k=top_k)
        nodes = retriever.retrieve(query) or []
        texts, cites = [], []
        for n in nodes:
            try:
                node = n.node if hasattr(n, "node") else n
                txt = getattr(node, "text", None) or node.get_content()
                md = getattr(node, "metadata", {}) or {}
                src = md.get("file_name") or md.get("source") or ""
                if txt:
                    texts.append(txt)
                    cites.append(src)
            except Exception:
                continue
        return texts, cites

    def synthesize(self, query: str, llm_choice: str) -> Tuple[List[str], List[str]]:
        Settings.llm = get_rag_llm(llm_choice)
        qe = self._index.as_query_engine(llm=Settings.llm)
        resp = qe.query(query)
        out_text = getattr(resp, "response", str(resp))
        citations: List[str] = []
        try:
            if hasattr(resp, "source_nodes") and resp.source_nodes:
                for s in resp.source_nodes:
                    md = {}
                    if hasattr(s, "metadata"):
                        md = s.metadata or {}
                    elif hasattr(s, "node") and hasattr(s.node, "metadata"):
                        md = s.node.metadata or {}
                    fn = md.get("file_name")
                    if fn and fn not in citations:
                        citations.append(fn)
        except Exception:
            pass
        return [out_text], citations

def rag_open_readonly(doc_store_dir: str, population_dir: str):
    global _RAG_CLIENT, _RAG_CLIENT_PID
    pid = os.getpid()
    with _RAG_LOCK:
        if _RAG_CLIENT is None or _RAG_CLIENT_PID != pid:
            _RAG_CLIENT = _ReadonlyRAGClient(doc_store_dir, population_dir)
            _RAG_CLIENT_PID = pid
    return _RAG_CLIENT

# ------------------------------------------------------------------------------
# QUERY API (used by scripts.search.rag_query)
# ------------------------------------------------------------------------------
def rag_query(query: str, doc_store_dir: str, population_dir: str) -> Tuple[List[str], List[str]]:
    """
    Retrieval-first: returns (texts, citations). Set RAG_SYNTHESIZE=1 to enable LLM synthesis.
    """
    if not doc_store_dir:
        return [], []

    # If no sentinel yet, try to recognize existing storage on the fly to avoid surprises
    if not rag_is_ready(doc_store_dir):
        if not _try_detect_existing_index(doc_store_dir):
            # As a last resort, bail gracefully (workers shouldn't ingest)
            return [], []

    client = rag_open_readonly(doc_store_dir, population_dir)
    synth = os.getenv("RAG_SYNTHESIZE", "0").strip() in ("1", "true", "True", "yes")
    if synth:
        rag_choice, _ = get_model_choices_from_metadata(population_dir)
        return client.synthesize(query, rag_choice)
    return client.retrieve(query, top_k=int(os.getenv("RAG_TOP_K", "5")))

# ------------------------------------------------------------------------------
# Back-compat: rag_setup
# ------------------------------------------------------------------------------
def rag_setup(directory: str, population_dir: str):
    """
    Deprecated helper for older call sites.
    Now: just ensure readiness and return (None, None).
    """
    rag_prepare(directory, population_dir)
    return None, None

# ------------------------------------------------------------------------------
# Optional simple RAG for a master JSON (unchanged API)
# ------------------------------------------------------------------------------
def rag_master_parameters(query: str):
    master_file = "master_parameters.json"
    if not os.path.exists(master_file):
        raise FileNotFoundError(f"Master parameters file not found: {master_file}")
    with open(master_file, "r") as f:
        master_params = json.load(f)

    docs_text: List[str] = []
    for branch, params in (master_params or {}).items():
        for param in (params or []):
            doc = (
                f"Branch: {branch}\n"
                f"Parameter: {param.get('parameter')}\n"
                f"Description: {param.get('description')}\n"
                f"Value: {param.get('value')}\n"
                f"Min: {param.get('min')}\n"
                f"Max: {param.get('max')}\n"
                f"Source: {param.get('source')}\n"
            )
            docs_text.append(doc)

    docs = [Document(text=t) for t in docs_text]
    idx = VectorStoreIndex.from_documents(docs)
    qe = idx.as_query_engine()
    resp = qe.query(query)
    out = getattr(resp, "response", str(resp))
    relevant = [t for t in docs_text if t in out]
    return [out], relevant
