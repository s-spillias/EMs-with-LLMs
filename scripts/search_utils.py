from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

import os
import json
from duckduckgo_search import DDGS
import aiohttp
import asyncio
from bs4 import BeautifulSoup
import requests
from typing import Dict, List, Union
import time
import random
import fcntl
import tempfile

from scripts.rag_utils import rag_setup, rag_query, rag_master_parameters
from scripts.serper_search import enhanced_serper_search

# File-based rate limiting for cross-process synchronization
RATE_LIMIT_FILE = os.path.join(tempfile.gettempdir(), 'semantic_scholar_rate_limit.txt')

def wait_for_rate_limit():
    """
    Ensure at least 1 second has passed since the last Semantic Scholar request
    across all processes using file-based synchronization.
    """
    try:
        # Create the rate limit file if it doesn't exist
        if not os.path.exists(RATE_LIMIT_FILE):
            with open(RATE_LIMIT_FILE, 'w') as f:
                f.write('0')
        
        with open(RATE_LIMIT_FILE, 'r+') as f:
            # Lock the file for exclusive access
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            
            try:
                last_request_time = float(f.read().strip() or '0')
            except ValueError:
                last_request_time = 0
            
            current_time = time.time()
            time_since_last = current_time - last_request_time
            
            if time_since_last < 1.0:
                sleep_time = 1.0 - time_since_last
                print(f"Rate limiting: sleeping for {sleep_time:.2f} seconds")
                time.sleep(sleep_time)
                current_time = time.time()
            
            # Update the last request time
            f.seek(0)
            f.write(str(current_time))
            f.truncate()
            
            # Release the lock
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            
    except Exception as e:
        print(f"Warning: Rate limiting failed: {e}. Falling back to basic sleep.")
        time.sleep(1.0)

def make_semantic_scholar_request_with_backoff(url, headers, params, max_retries=5):
    """
    Make a request to Semantic Scholar API with exponential backoff retry logic.
    """
    for attempt in range(max_retries):
        try:
            # Apply cross-process rate limiting
            wait_for_rate_limit()
            
            response = requests.get(url, headers=headers, params=params, timeout=30)
            
            if response.status_code == 200:
                return response
            elif response.status_code == 429:  # Rate limited
                # Exponential backoff with jitter
                base_delay = 2 ** attempt
                jitter = random.uniform(0, 1)
                delay = base_delay + jitter
                print(f"Rate limited (429). Retrying in {delay:.2f} seconds (attempt {attempt + 1}/{max_retries})")
                time.sleep(delay)
            elif response.status_code >= 500:  # Server error
                # Exponential backoff for server errors
                base_delay = 2 ** attempt
                jitter = random.uniform(0, 1)
                delay = base_delay + jitter
                print(f"Server error ({response.status_code}). Retrying in {delay:.2f} seconds (attempt {attempt + 1}/{max_retries})")
                time.sleep(delay)
            else:
                # For other errors, raise immediately
                response.raise_for_status()
                
        except requests.exceptions.RequestException as e:
            if attempt == max_retries - 1:
                raise e
            
            # Exponential backoff for network errors
            base_delay = 2 ** attempt
            jitter = random.uniform(0, 1)
            delay = base_delay + jitter
            print(f"Network error: {e}. Retrying in {delay:.2f} seconds (attempt {attempt + 1}/{max_retries})")
            time.sleep(delay)
    
    # If we get here, all retries failed
    raise Exception(f"Failed to make request after {max_retries} attempts")

def search_engine(query, engine="ddg", directory=None, population_dir=None):
    if engine == "ddg":
        return ddg_search(query)
    elif engine == "serper":
        return enhanced_serper_search(query)
    elif engine == "semantic_scholar":
        return semantic_scholar_search(query)
    elif engine == "rag_master":
        return rag_master_parameters(query)
    else:
        raise ValueError("Unsupported search engine. Use 'ddg', 'serper', 'semantic_scholar', 'rag', or 'rag_master'.")

def ddg_search(query):
    results = DDGS().text(query, max_results=5)
    urls = []
    for result in results:
        url = result['href']
        urls.append(url)

    docs = asyncio.run(get_page(urls))
    content = []
    citations = []
    for doc, url in zip(docs, urls):
        page_text = "\n".join(doc)
        content.append(page_text)
        citations.append(url)
    return content, citations

def search_for_papers(query, result_limit=20) -> Union[None, List[Dict]]:
    """Search with more papers initially to allow for filtering"""
    if not query:
        return None
    
    try:
        rsp = make_semantic_scholar_request_with_backoff(
            "https://api.semanticscholar.org/graph/v1/paper/search",
            headers={"X-API-KEY": os.environ.get('S2_API_KEY')},
            params={
                "query": query,
                "limit": result_limit,
                "fieldsOfStudy": ["Biology", "Mathematics", "Environmental Science"],
                "fields": "title,abstract,venue,year,citationCount",
            }
        )
        
        print(f"Response Status Code: {rsp.status_code}")
        print(f"Enhanced query: {query}")
        
        results = rsp.json()
        total = results["total"]
        print(f"Total results before filtering: {total}")
        
        if not total:
            return None

        papers = results["data"]
        return papers
        
    except Exception as e:
        print(f"Error searching for papers: {e}")
        return None

def semantic_scholar_search(query):
    papers = search_for_papers(query, result_limit=20)

    if not papers:
        print('No papers found.')
        return [], []

    citations = []
    content = []
        
    # Take up to 10 relevant papers
    for paper in papers:
        if paper.get('paperId') and paper.get('abstract'):
            citations.append(f"https://www.semanticscholar.org/paper/{paper['paperId']}")
            content.append(paper['abstract'])

    return content, citations

async def fetch(session, url):
    async with session.get(url) as response:
        return await response.text(encoding='utf-8', errors='ignore')

async def get_page(urls):
    async with aiohttp.ClientSession(headers={"User-Agent": "MyAppUserAgent"}) as session:
        tasks = [fetch(session, url) for url in urls]
        html_pages = await asyncio.gather(*tasks)

    docs_transformed = []
    for html in html_pages:
        try:
            soup = BeautifulSoup(html, 'html.parser')
            for tag in soup.find_all('a'):
                tag.decompose()
            p_tags = soup.find_all('p')
            docs_transformed.append([tag.get_text() for tag in p_tags])
        except:
            print("Can't access that page.")

    return docs_transformed

def truncate(text):
    words = text.split()
    truncated = " ".join(words[:400])
    return truncated
