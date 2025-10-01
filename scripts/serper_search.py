from dotenv import load_dotenv
import os
import json
import sys
from scripts.ask_AI import ask_ai

# Add path for importing from scripts
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.rag_utils import get_model_choices_from_metadata

load_dotenv()

def search_serper(query, population_dir=None):
    """
    Perform a search using ask_ai instead of Serper API.
    
    Args:
        query (str): The search query
        
    Returns:
        dict: The search results formatted to match Serper API response
    """
    # Construct a search prompt for the AI
    search_prompt = f"""
    Please perform a web search for: "{query}"
    
    Return the results in JSON format that matches the Serper API response structure:
    {{
        "organic": [
            {{
                "title": "Result title",
                "link": "https://example.com/result-url",
                "snippet": "Brief description of the result"
            }},
            ...more results...
        ]
    }}
    
    Provide at least 3 relevant results with accurate titles, links, and snippets.
    """
    

    rag_choice, _ = get_model_choices_from_metadata(population_dir)
    if rag_choice:
        model = rag_choice

    
    # Use ask_ai instead of Serper API
    ai_response = ask_ai(search_prompt, model=model)
    
    # Try to parse the response as JSON
    try:
        # Extract JSON if it's embedded in markdown or explanatory text
        if "```json" in ai_response:
            json_str = ai_response.split("```json")[1].split("```")[0].strip()
            results = json.loads(json_str)
        elif "```" in ai_response:
            json_str = ai_response.split("```")[1].split("```")[0].strip()
            results = json.loads(json_str)
        else:
            # Try to parse the whole response as JSON
            results = json.loads(ai_response)
            
        # Ensure the response has the expected structure
        if "organic" not in results:
            results = {"organic": results.get("results", [])}
            
        return results
    except json.JSONDecodeError:
        # If JSON parsing fails, create a structured response manually
        print(f"Failed to parse AI response as JSON. Creating fallback structure.")
        fallback_results = {
            "organic": [
                {
                    "title": f"Search results for: {query}",
                    "link": "https://example.com/search",
                    "snippet": ai_response[:200] + "..."
                }
            ]
        }
        return fallback_results
    except Exception as e:
        print(f"Error using ask_ai for search: {e}")
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

def enhanced_serper_search(query_raw, citation_required=True, population_dir=None):
    """
    Perform an enhanced search using ask_ai instead of Serper API.
    
    Args:
        query_raw (str): The base search query
        citation_required (bool): Whether to explicitly request citations in the query
        population_dir (str, optional): Path to population directory for model selection
        
    Returns:
        tuple: (list of content strings, list of citation strings)
    """
    # Perform the search
    search_results = search_serper(query_raw, population_dir)
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
        # Get model from population metadata if available
        model = "claude-3-haiku-20240307"  # Default fallback
        if population_dir:
            try:
                rag_choice, _ = get_model_choices_from_metadata(population_dir)
                if rag_choice:
                    model = rag_choice
            except Exception as e:
                print(f"Warning: Could not get model from metadata: {e}. Using default.")
        
        # Use ask_ai instead of direct Anthropic client
        analysis_prompt = f"{system_prompt}\n\n{user_prompt}"
        result = ask_ai(analysis_prompt, model=model)
        
        # Extract citations from the search results
        citations = [item.get('link', '') for item in search_results.get('organic', [])[:3]]
        
        return [result], citations
            
    except Exception as e:
        print(f"Error occurred during analysis: {e}")
        return [], []

if __name__ == "__main__":
    # Example usage
    import sys
    
    # Get population directory from command line if provided
    population_dir = None
    if len(sys.argv) > 1:
        population_dir = sys.argv[1]
    
    query = "maximum sustainable Crown-of-thorns starfish density per square meter"
    content, citations = enhanced_serper_search(query, population_dir=population_dir)
    
    print("\nSearch Results:")
    print("-" * 50)
    for text, citation in zip(content, citations):
        print(f"Content: {text}")
        print(f"Sources:")
        for cite in citations:
            print(f"- {cite}")
        print("-" * 50)
