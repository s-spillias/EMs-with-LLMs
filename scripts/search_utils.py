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

from scripts.rag_utils import rag_setup, rag_query, rag_master_parameters
from scripts.serper_search import enhanced_serper_search

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
    
    rsp = requests.get(
        "https://api.semanticscholar.org/graph/v1/paper/search",
        headers={"X-API-KEY": os.environ.get('S2_API_KEY')},
        params={
            "query": query,
            "limit": result_limit,
            "fieldsOfStudy": ["Biology", "Mathematics", "Environmental Science"],
            "fields": "title,abstract,venue,year,citationCount",
        },
    )
    print(f"Response Status Code: {rsp.status_code}")
    print(f"Enhanced query: {query}")
    rsp.raise_for_status()
    results = rsp.json()
    total = results["total"]
    print(f"Total results before filtering: {total}")
    time.sleep(0.5)  # Rate limiting
    if not total:
        return None

    papers = results["data"]
    return papers

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
