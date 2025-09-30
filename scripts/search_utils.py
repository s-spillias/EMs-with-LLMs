#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
search_utils.py

File-lock–free search & rate limiting utilities.

What this does:
- Enforces a hard *pace* for Semantic Scholar (S2): at most 1 request/second
  across all workers on a single host (using multiprocessing.Manager—no files).
- Bounds concurrent in-flight S2 requests per process (semaphore).
- Honors HTTP Retry-After and uses exponential backoff with jitter for 429/5xx.
- Keeps async DuckDuckGo fetch with robust timeouts.

Why this shape:
- Vendor-guided pacing: honor Retry-After / RateLimit headers when present
  (see IETF draft and common API guidance). If not present, use exponential
  backoff with jitter on transient errors.  References:
    - IETF RateLimit headers (RateLimit-Limit/Remaining/Reset): 
      https://www.ietf.org/archive/id/draft-polli-ratelimit-headers-02.html
    - AWS / Postman guidance on exponential backoff for 429/5xx:
      https://docs.aws.amazon.com/prescriptive-guidance/latest/cloud-design-patterns/retry-backoff.html
      https://www.postman.com/ping-identity/pingone/folder/2fuzaov/retries-best-practice-for-managing-transient-api-errors
- Semantic Scholar historically has not always exposed helpful rate-limit 
  headers; explicit pacing is safest when they ask for 1 req/sec:
  https://books.ropensci.org/fulltext/rate-limits.html
"""

from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

import os
import json
import time
import random
from typing import Dict, List, Union, Tuple

import requests
import asyncio
import aiohttp
from bs4 import BeautifulSoup

# Project deps
from scripts.rag_utils import rag_master_parameters  # used by 'rag_master' engine
from scripts.serper_search import enhanced_serper_search

# --- Config & logging --------------------------------------------------------

def _get_env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except Exception:
        return default

def _get_env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except Exception:
        return default

# S2 controls (per-host, file-less)
S2_MIN_INTERVAL   = _get_env_float("S2_MIN_INTERVAL", 1.0)   # **hard pace**: seconds between requests
S2_MAX_CONCURRENT = _get_env_int("S2_MAX_CONCURRENT", 2)     # in-flight per process (semaphore)
S2_TIMEOUT        = _get_env_float("S2_TIMEOUT", 30.0)       # per HTTP call
S2_MAX_RETRIES    = _get_env_int("S2_MAX_RETRIES", 6)        # backoff attempts
S2_BASE_DELAY     = _get_env_float("S2_BASE_DELAY", 0.6)     # base seconds for backoff

DDG_TIMEOUT       = _get_env_float("DDG_TIMEOUT", 30.0)      # per page fetch total timeout
DDG_MAX_RESULTS   = _get_env_int("DDG_MAX_RESULTS", 5)

# Force-flush prints to keep logs visible in parallel runs
import builtins as _bi
_print = _bi.print
def print(*args, **kwargs):
    kwargs.setdefault("flush", True)
    return _print(*args, **kwargs)

# --- Per-process concurrency (no files) -------------------------------------

import threading
from contextlib import contextmanager
# Per-process bounded concurrency (keeps in-flight calls small)
_S2_SEMA = threading.BoundedSemaphore(S2_MAX_CONCURRENT)

@contextmanager
def _acquire_s2_slot():
    _S2_SEMA.acquire()
    try:
        yield
    finally:
        _S2_SEMA.release()

# --- Cross-worker (same host) 1 rps pacer (Manager-based; no files) ---------

# Shared state proxies (set by init_rate_limit_manager); fall back to per-process
_S2_PACE_LOCK = None     # manager.Lock() or threading.Lock()
_S2_LAST_TS   = None     # manager.Value('d', 0.0) or process-local float
_S2_MIN_INT   = S2_MIN_INTERVAL  # can be overridden by init

# Process-local fallback
_proc_local_lock = threading.Lock()
_proc_local_last = 0.0

def init_rate_limit_manager(manager, s2_min_interval: float = None, s2_max_concurrent: int = None):
    """
    Initialize shared pacer + (optionally) adjust per-process concurrency.
    Call once in the main process and BEFORE you start Pool/Process workers.

    Example:
        from multiprocessing import Manager
        from scripts.search_utils import init_rate_limit_manager
        mgr = Manager()
        init_rate_limit_manager(mgr, s2_min_interval=1.0, s2_max_concurrent=2)
    """
    global _S2_PACE_LOCK, _S2_LAST_TS, _S2_MIN_INT, _S2_SEMA

    # Shared pacer (no files)
    _S2_PACE_LOCK = manager.Lock()
    _S2_LAST_TS   = manager.Value('d', 0.0)
    if s2_min_interval is not None:
        _S2_MIN_INT = float(s2_min_interval)

    # Optionally tune per-process semaphore
    if s2_max_concurrent is not None and s2_max_concurrent > 0:
        # Rebuild per-process semaphore to new size
        new_size = int(s2_max_concurrent)
        # NOTE: Python doesn't let us resize an existing BoundedSemaphore.
        # Create a new one and replace the global ref.
        globals()['_S2_SEMA'] = threading.BoundedSemaphore(new_size)

def _pace_s2_once(now: float) -> float:
    """
    Try to reserve the next allowed slot time. Returns the **reserved** timestamp.
    Uses a non-blocking loop: grabs lock, checks due time, if not due, releases,
    sleeps a short duration, and tries again—so we don't hold the lock while sleeping.
    """
    min_interval = _S2_MIN_INT

    # Choose shared vs local state
    use_shared = (_S2_PACE_LOCK is not None) and (_S2_LAST_TS is not None)
    if use_shared:
        lock = _S2_PACE_LOCK
        def get_last(): return _S2_LAST_TS.value
        def set_last(v): setattr(_S2_LAST_TS, "value", v)
    else:
        lock = _proc_local_lock
        def get_last(): return globals()['_proc_local_last']
        def set_last(v): globals()['_proc_local_last'] = v

    while True:
        with lock:
            last = get_last()
            due  = last + min_interval if last > 0.0 else 0.0
            now  = time.monotonic()
            if now >= due:
                # Reserve now
                set_last(now)
                return now
            wait = max(0.0, due - now)
        # Sleep outside the lock to avoid blocking other workers
        time.sleep(min(wait, 0.2))

def _pace_s2():
    """
    Pace S2 to enforce at most 1 request per second (S2_MIN_INTERVAL).
    Called immediately before making an S2 HTTP request.
    """
    _pace_s2_once(time.monotonic())

# --- Resilient HTTP (no file locks; honors Retry-After) ----------------------

def _sleep_retry_after(resp: requests.Response) -> bool:
    """Honor Retry-After if present (seconds only). Return True if slept."""
    ra = resp.headers.get("Retry-After")
    if not ra:
        return False
    try:
        secs = float(ra)
        time.sleep(max(0.0, min(secs, 90.0)))
        return True
    except Exception:
        return False

def http_get_with_resilience(
    url: str,
    *,
    headers: Dict[str, str] = None,
    params: Dict[str, Union[str, int, float]] = None,
    timeout: float = None,
    max_retries: int = None,
    base_delay: float = None,
) -> requests.Response:
    """
    GET with:
      - per-process concurrency bound (semaphore)
      - **host-wide 1 rps pacer** (Manager-based; no files)
      - exponential backoff + Retry-After honoring

    This shapes traffic proactively (1/sec) and reacts politely to server hints.
    """
    timeout     = timeout     or S2_TIMEOUT
    max_retries = max_retries or S2_MAX_RETRIES
    base_delay  = base_delay  or S2_BASE_DELAY

    attempt = 0
    with _acquire_s2_slot():
        while True:
            # Enforce 1 rps before each attempt
            _pace_s2()

            try:
                r = requests.get(url, headers=headers, params=params, timeout=timeout)
                if r.status_code == 200:
                    return r
                if r.status_code == 429:
                    # Prefer server-provided Retry-After
                    if not _sleep_retry_after(r):
                        delay = base_delay * (2 ** attempt) + random.uniform(0, 0.25)
                        time.sleep(min(delay, 30.0))
                elif 500 <= r.status_code < 600:
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 0.25)
                    time.sleep(min(delay, 30.0))
                else:
                    r.raise_for_status()
            except requests.exceptions.RequestException as e:
                delay = base_delay * (2 ** attempt) + random.uniform(0, 0.25)
                print(f"HTTP transient error: {e}. Retrying in {min(delay,30.0):.2f}s (attempt {attempt+1}/{max_retries})")
                time.sleep(min(delay, 30.0))

            attempt += 1
            if attempt > max_retries:
                raise RuntimeError(f"GET {url} failed after {max_retries} retries")

# --- Search entry point ------------------------------------------------------

def search_engine(query: str, engine: str = "ddg", directory: str = None, population_dir: str = None) -> Tuple[List[str], List[str]]:
    """
    Dispatch to different search providers.
    Returns (content_chunks, citations) where both are List[str].
    """
    if engine == "ddg":
        return ddg_search(query)
    elif engine == "serper":
        # Implementations of serper should have their own timeouts/backoff
        return enhanced_serper_search(query)
    elif engine == "semantic_scholar":
        return semantic_scholar_search(query)
    elif engine == "rag_master":
        return rag_master_parameters(query)
    else:
        raise ValueError("Unsupported search engine. Use 'ddg', 'serper', 'semantic_scholar', or 'rag_master'.")

# --- DuckDuckGo search (with robust timeouts fetching pages) -----------------

from duckduckgo_search import DDGS

def ddg_search(query: str) -> Tuple[List[str], List[str]]:
    results = DDGS().text(query, max_results=DDG_MAX_RESULTS)
    urls: List[str] = []
    for result in results:
        url = result.get('href')
        if url:
            urls.append(url)

    # Fetch pages concurrently (async) with per-request timeout
    docs = _run_async(get_pages(urls))
    content: List[str] = []
    citations: List[str] = []
    for doc, url in zip(docs, urls):
        page_text = "\n".join(doc)
        content.append(page_text)
        citations.append(url)
    return content, citations

async def _fetch(session: aiohttp.ClientSession, url: str) -> str:
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=DDG_TIMEOUT)) as response:
            return await response.text(encoding='utf-8', errors='ignore')
    except Exception as e:
        print(f"DDG fetch error for {url}: {e}")
        return ""

async def get_pages(urls: List[str]) -> List[List[str]]:
    headers = {"User-Agent": "ModelBuilderBot/1.0 (+https://example.org)"}  # polite UA
    async with aiohttp.ClientSession(headers=headers) as session:
        tasks = [_fetch(session, url) for url in urls]
        html_pages = await asyncio.gather(*tasks, return_exceptions=False)

    docs_transformed: List[List[str]] = []
    for html in html_pages:
        try:
            soup = BeautifulSoup(html, 'html.parser')
            for tag in soup.find_all('a'):
                tag.decompose()
            p_tags = soup.find_all('p')
            docs_transformed.append([tag.get_text() for tag in p_tags])
        except Exception:
            print("Can't access or parse that page.")
            docs_transformed.append([])
    return docs_transformed

def _run_async(coro):
    """
    Safe asyncio runner for environments that may already have a loop.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop and loop.is_running():
        # Run in a nested task
        return asyncio.run_coroutine_threadsafe(coro, loop).result()
    else:
        return asyncio.run(coro)

# --- Semantic Scholar search -------------------------------------------------

def search_for_papers(query: str, result_limit: int = 20) -> Union[None, List[Dict]]:
    """Search S2 with larger result set first to enable filtering; 1 rps paced."""
    if not query:
        return None
    try:
        rsp = http_get_with_resilience(
            "https://api.semanticscholar.org/graph/v1/paper/search",
            headers={"X-API-KEY": os.environ.get('S2_API_KEY')},
            params={
                "query": query,
                "limit": result_limit,
                "fieldsOfStudy": ["Biology", "Mathematics", "Environmental Science"],
                "fields": "title,abstract,venue,year,citationCount",
            },
            timeout=S2_TIMEOUT,
            max_retries=S2_MAX_RETRIES,
        )
        print(f"S2 Response Status Code: {rsp.status_code}")
        print(f"S2 Query: {query}")
        results = rsp.json()
        total = results.get("total", 0)
        print(f"S2 Total results: {total}")
        if not total:
            return None
        papers = results.get("data", [])
        return papers
    except Exception as e:
        print(f"Error searching for papers: {e}")
        return None

def semantic_scholar_search(query: str) -> Tuple[List[str], List[str]]:
    papers = search_for_papers(query, result_limit=20)
    if not papers:
        print('No papers found.')
        return [], []
    citations: List[str] = []
    content: List[str] = []
    # Take up to 10 relevant papers that have abstracts
    for paper in papers[:10]:
        if paper.get('paperId') and paper.get('abstract'):
            citations.append(f"https://www.semanticscholar.org/paper/{paper['paperId']}")
            content.append(paper['abstract'])
    return content, citations

# --- Utility ----------------------------------------------------------------

def truncate(text: str) -> str:
    words = text.split()
    truncated = " ".join(words[:400])
    return truncated
