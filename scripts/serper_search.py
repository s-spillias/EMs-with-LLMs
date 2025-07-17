from anthropic import Anthropic
from dotenv import load_dotenv
import os
import requests
import json

load_dotenv()

def search_serper(query):
    """
    Perform a search using the Serper API.
    
    Args:
        query (str): The search query
        
    Returns:
        dict: The search results from Serper
    """
    headers = {
        'X-API-KEY': os.environ.get('SERPER_API_KEY'),
        'Content-Type': 'application/json'
    }
    
    payload = {
        'q': query,
        'gl': 'us',  # Geolocation
        'hl': 'en'   # Language
    }
    
    response = requests.post(
        'https://google.serper.dev/search',
        headers=headers,
        json=payload
    )
    
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: Serper API returned status code {response.status_code}")
        return None

def process_search_results(results):
    """
    Process and format the search results.
    
    Args:
        results (dict): The raw search results from Serper
        
    Returns:
        str: Formatted search results
    """
    if not results or 'organic' not in results:
        return "No results found."
    
    formatted_results = []
    for item in results['organic'][:3]:  # Take top 3 results
        title = item.get('title', '')
        snippet = item.get('snippet', '')
        link = item.get('link', '')
        formatted_results.append(f"Title: {title}\nSnippet: {snippet}\nSource: {link}\n")
    
    return "\n".join(formatted_results)

def enhanced_serper_search(query_raw, citation_required=True):
    """
    Perform an enhanced search using Google Serper API with Anthropic.
    
    Args:
        query_raw (str): The base search query
        citation_required (bool): Whether to explicitly request citations in the query
        
    Returns:
        tuple: (list of content strings, list of citation strings)
    """
    # Initialize Anthropic client
    client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    
    # Perform the search
    search_results = search_serper(query_raw)
    if not search_results:
        return [], []
    
    # Format search results
    formatted_results = process_search_results(search_results)
    
    # Create prompt for Claude
    system_prompt = """You are a research assistant helping to analyze search results and provide accurate information with citations. 
    Based on the search results provided, synthesize a clear and concise answer to the query. 
    Include specific citations to sources when possible."""
    
    user_prompt = f"""Query: {query_raw}

Search Results:
{formatted_results}

Please provide a comprehensive answer based on these search results, including relevant citations."""
    
    try:
        # Get Claude's analysis
        response = client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=1000,
            temperature=0,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}]
        )
        
        result = response.content[0].text
        
        # Extract citations from the search results
        citations = [item.get('link', '') for item in search_results.get('organic', [])[:3]]
        
        return [result], citations
            
    except Exception as e:
        print(f"Error occurred during analysis: {e}")
        return [], []

if __name__ == "__main__":
    # Example usage
    query = "maximum sustainable Crown-of-thorns starfish density per square meter"
    content, citations = enhanced_serper_search(query)
    
    print("\nSearch Results:")
    print("-" * 50)
    for text, citation in zip(content, citations):
        print(f"Content: {text}")
        print(f"Sources:")
        for cite in citations:
            print(f"- {cite}")
        print("-" * 50)
