#!/usr/bin/env python3
"""
LLM Client Library with Usage Tracking

A unified interface for multiple Large Language Model APIs with comprehensive
error handling, retry logic, and usage tracking using actual model names.
"""

import json
import logging
import os
import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Union
from threading import Lock



import requests
from dotenv import load_dotenv

# Optional imports with graceful fallback
try:
    import boto3
except ImportError:
    print("no boto3")
    
try:
    import anthropic
except ImportError:
    Anthropic = None
try:
    from groq import Groq
except ImportError:
    Groq = None

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    import google.generativeai as genai
except ImportError:
    genai = None


# Configuration and Data Classes
@dataclass
class UsageInfo:
    """Data class for tracking token usage and costs"""
    timestamp: str
    model: str
    input_tokens: int
    output_tokens: int
    input_cost: float
    output_cost: float
    total_cost: float
    prompt_preview: str
    response_preview: str


@dataclass
class ModelConfig:
    """Configuration for each model"""
    provider: str
    model_name: str
    max_tokens_limit: int
    input_cost_per_million: float
    output_cost_per_million: float
    temperature_default: float = 0.1


class Config:
    """Application configuration"""
    USAGE_FILE = "llm_usage.json"
    MAX_PREVIEW_LENGTH = 100
    DEFAULT_MAX_TOKENS = 10000
    OLLAMA_DEFAULT_MAX_TOKENS = 16384
    OLLAMA_API_BASE = "http://localhost:11434"
    
    # Model configurations using actual model names
    MODELS = {
        # Anthropic Models
        # Claude 4 Models
        'claude-opus-4-20250514': ModelConfig(
            provider='anthropic',
            model_name='claude-opus-4-20250514',
            max_tokens_limit=8192,  # 200K context window
            input_cost_per_million=15.00,
            output_cost_per_million=75.00
        ),
        'claude-sonnet-4-20250514': ModelConfig(
            provider='anthropic',
            model_name='claude-sonnet-4-20250514',
            max_tokens_limit=8192,  # 200K context window
            input_cost_per_million=3.00,
            output_cost_per_million=15.00
        ),
        # Claude 3.7 Models
        'claude-3-7-sonnet-20250219': ModelConfig(
            provider='anthropic',
            model_name='claude-3-7-sonnet-20250219',
            max_tokens_limit=8192,  # 200K context window
            input_cost_per_million=3.00,
            output_cost_per_million=15.00
        ),
        # Claude 3.5 Models
        'claude-3-5-sonnet-20241022': ModelConfig(
            provider='anthropic',
            model_name='claude-3-5-sonnet-20241022',
            max_tokens_limit=8192,  # 200K context window
            input_cost_per_million=3.00,
            output_cost_per_million=15.00
        ),
        'claude-3-5-haiku-20241022': ModelConfig(
            provider='anthropic',
            model_name='claude-3-5-haiku-20241022',
            max_tokens_limit=8192,  # 200K context window
            input_cost_per_million=1.00,
            output_cost_per_million=5.00
        ),
        # Claude 3 Models
        'claude-3-opus-20240229': ModelConfig(
            provider='anthropic',
            model_name='claude-3-opus-20240229',
            max_tokens_limit=8192,  # 200K context window
            input_cost_per_million=15.00,
            output_cost_per_million=75.00
        ),
        
        # AWS Bedrock Claude Models
        'anthropic.claude-3-5-sonnet-20240620-v1:0': ModelConfig(
            provider='aws_bedrock',
            model_name='anthropic.claude-3-5-sonnet-20240620-v1:0',
            max_tokens_limit=200000,
            input_cost_per_million=3.00,
            output_cost_per_million=15.00
        ),
        'anthropic.claude-3-haiku-20240307-v1:0': ModelConfig(
            provider='aws_bedrock',
            model_name='anthropic.claude-3-haiku-20240307-v1:0',
            max_tokens_limit=200000,
            input_cost_per_million=0.25,
            output_cost_per_million=1.25
        ),
        
        # Google Models
        # Gemini 2.5 Models
        'gemini-2.5-pro': ModelConfig(
            provider='google',
            model_name='gemini-2.5-pro',
            max_tokens_limit=2000000,
            input_cost_per_million=1.25,  # $1.25 for prompts <= 200k tokens
            output_cost_per_million=10.00  # $10.00 for prompts <= 200k tokens
        ),
        'gemini-2.5-flash': ModelConfig(
            provider='google',
            model_name='gemini-2.5-flash',
            max_tokens_limit=1000000,
            input_cost_per_million=0.30,  # text/image/video
            output_cost_per_million=2.50
        ),
        'gemini-2.5-flash-lite': ModelConfig(
            provider='google',
            model_name='gemini-2.5-flash-lite',
            max_tokens_limit=1000000,
            input_cost_per_million=0.10,  # text/image/video
            output_cost_per_million=0.40
        ),
        'gemini-2.5-flash-native-audio': ModelConfig(
            provider='google',
            model_name='gemini-2.5-flash-native-audio',
            max_tokens_limit=1000000,
            input_cost_per_million=0.50,  # text input
            output_cost_per_million=2.00  # text output
        ),
        'gemini-2.5-pro-preview-tts': ModelConfig(
            provider='google',
            model_name='gemini-2.5-pro-preview-tts',
            max_tokens_limit=1000000,
            input_cost_per_million=1.00,  # text input
            output_cost_per_million=20.00  # audio output
        ),
        'gemini-2.5-flash-preview-tts': ModelConfig(
            provider='google',
            model_name='gemini-2.5-flash-preview-tts',
            max_tokens_limit=1000000,
            input_cost_per_million=0.50,  # text input
            output_cost_per_million=10.00  # audio output
        ),
        # Gemini 2.0 Models
        'gemini-2.0-flash': ModelConfig(
            provider='google',
            model_name='gemini-2.0-flash',
            max_tokens_limit=1000000,
            input_cost_per_million=0.10,  # text/image/video
            output_cost_per_million=0.40
        ),
        'gemini-2.0-flash-lite': ModelConfig(
            provider='google',
            model_name='gemini-2.0-flash-lite',
            max_tokens_limit=1000000,
            input_cost_per_million=0.075,
            output_cost_per_million=0.30
        ),
        # Gemini 1.5 Models (keeping for backward compatibility)
        'gemini-1.5-pro': ModelConfig(
            provider='google',
            model_name='gemini-1.5-pro',
            max_tokens_limit=2000000,
            input_cost_per_million=3.50,
            output_cost_per_million=10.50
        ),
        'gemini-1.5-flash': ModelConfig(
            provider='google',
            model_name='gemini-1.5-flash',
            max_tokens_limit=1000000,
            input_cost_per_million=0.15,
            output_cost_per_million=0.60
        ),
        
        # Groq Models
        'llama-3.3-70b-versatile': ModelConfig(
            provider='groq',
            model_name='llama-3.3-70b-versatile',
            max_tokens_limit=32768,
            input_cost_per_million=0.59,
            output_cost_per_million=0.79
        ),
        'llama-3.1-8b-instant': ModelConfig(
            provider='groq',
            model_name='llama-3.1-8b-instant',
            max_tokens_limit=131072,
            input_cost_per_million=0.05,
            output_cost_per_million=0.08
        ),
        'mixtral-8x7b-32768': ModelConfig(
            provider='groq',
            model_name='mixtral-8x7b-32768',
            max_tokens_limit=32768,
            input_cost_per_million=0.24,
            output_cost_per_million=0.24
        ),
        'gemma2-9b-it': ModelConfig(
            provider='groq',
            model_name='gemma2-9b-it',
            max_tokens_limit=8192,
            input_cost_per_million=0.20,
            output_cost_per_million=0.20
        ),
        
        # OpenAI Models
        'o1-mini': ModelConfig(
            provider='openai',
            model_name='o1-mini',
            max_tokens_limit=65536,
            input_cost_per_million=3.00,
            output_cost_per_million=12.00
        ),
        'gpt-4o': ModelConfig(
            provider='openai',
            model_name='gpt-4o',
            max_tokens_limit=128000,
            input_cost_per_million=2.50,
            output_cost_per_million=10.00
        ),
        'gpt-4o-mini': ModelConfig(
            provider='openai',
            model_name='gpt-4o-mini',
            max_tokens_limit=128000,
            input_cost_per_million=0.15,
            output_cost_per_million=0.60
        ),
        'gpt-5-mini': ModelConfig(
            provider='openai',
            model_name='gpt-5-mini',
            max_tokens_limit=128000,
            input_cost_per_million=0.25,
            output_cost_per_million=2.0
        ),
        'gpt-4.1-mini': ModelConfig(
            provider='openai',
            model_name='gpt-4.1-mini',
            max_tokens_limit=128000,
            input_cost_per_million=0.40,
            output_cost_per_million=1.60
        ),
        # === OpenRouter models ===
        # GPT-5 (OpenAI via OpenRouter)
        'openai/gpt-5': ModelConfig(
            provider='openrouter',
            model_name='openai/gpt-5',
            max_tokens_limit=128000,        # 128K max output; 400K context
            input_cost_per_million=1.25,    # $1.25/M input
            output_cost_per_million=10.00   # $10.00/M output
        ),
        'openai/gpt-5-mini': ModelConfig(
            provider='openrouter',
            model_name='openai/gpt-5-mini',
            max_tokens_limit=128000,
            input_cost_per_million=0.25,
            output_cost_per_million=2.00
        ),

        # Claude Sonnet 4.5 (Anthropic via OpenRouter)
        'anthropic/claude-sonnet-4.5': ModelConfig(
            provider='openrouter',
            model_name='anthropic/claude-sonnet-4.5',
            max_tokens_limit=65536,         # ~64K max output; up to 1M context
            # NOTE: OpenRouter shows tiered pricing by provider for very long contexts (>200K).
            # We use the common "starting at" rate here for consistent cost tracking.
            input_cost_per_million=3.00,    # Starting at $3/M input
            output_cost_per_million=15.00   # Starting at $15/M output
        ),

        # Gemini 2.5 Pro (Google via OpenRouter)
        'google/gemini-2.5-pro': ModelConfig(
            provider='openrouter',
            model_name='google/gemini-2.5-pro',
            max_tokens_limit=65536,         # ~65.5K max output; ~1.05M context
            input_cost_per_million=1.25,    # Starting at $1.25/M input
            output_cost_per_million=10.00   # Starting at $10/M output
        ),
    }


# Utility Classes
class UsageTracker:
    """Thread-safe usage tracking with JSON persistence"""
    
    def __init__(self, file_path: str = Config.USAGE_FILE):
        self.file_path = Path(file_path)
        self._lock = Lock()
        self._ensure_file_exists()
    
    def _ensure_file_exists(self) -> None:
        """Create usage file if it doesn't exist"""
        if not self.file_path.exists():
            self._write_usage_data({"usage": [], "summary": self._calculate_summary([])})
    
    def _read_usage_data(self) -> Dict[str, Any]:
        """Read usage data from JSON file"""
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Handle both old format (list) and new format (dict with 'usage' key)
                if isinstance(data, list):
                    return {"usage": data}
                return data
        except (json.JSONDecodeError, FileNotFoundError) as e:
            logging.warning(f"Error reading usage file: {e}. Starting with empty data.")
            return {"usage": []}
    
    def _write_usage_data(self, data: Dict[str, Any]) -> None:
        """Write usage data to JSON file"""
        with open(self.file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def log_usage(self, usage_info: UsageInfo) -> None:
        """Thread-safely log usage information and update summary"""
        with self._lock:
            try:
                data = self._read_usage_data()
                
                # Add new usage record
                data["usage"].append(asdict(usage_info))
                
                # Update summary statistics
                stats = self._calculate_summary(data["usage"])
                data["summary"] = stats
                
                # Save updated data
                self._write_usage_data(data)
                
                logging.info(f"Usage logged for {usage_info.model}: "
                           f"${usage_info.total_cost:.6f}")
            except Exception as e:
                logging.error(f"Failed to log usage: {e}")
    
    def _calculate_summary(self, usage_data: list) -> Dict[str, Any]:
        """Calculate summary statistics from usage data"""
        if not usage_data:
            return {
                "total_cost": 0,
                "total_input_tokens": 0,
                "total_output_tokens": 0,
                "total_tokens": 0,
                "total_calls": 0
            }
        
        total_input_tokens = sum(entry.get("input_tokens", 0) for entry in usage_data)
        total_output_tokens = sum(entry.get("output_tokens", 0) for entry in usage_data)
        
        return {
            "total_cost": sum(entry.get("total_cost", 0) for entry in usage_data),
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "total_tokens": total_input_tokens + total_output_tokens,
            "total_calls": len(usage_data)
        }
    
    def get_total_usage(self) -> Dict[str, Any]:
        """Get aggregated usage statistics"""
        with self._lock:
            data = self._read_usage_data()
            usage_records = data.get("usage", [])
            
            if not usage_records:
                return {"total_cost": 0, "total_calls": 0, "models": {}}
            
            # Calculate basic summary stats
            summary = self._calculate_summary(usage_records)
            
            # Add model-specific stats
            models = {}
            for entry in usage_records:
                model = entry.get("model", "unknown")
                if model not in models:
                    models[model] = {
                        "calls": 0,
                        "total_cost": 0,
                        "input_tokens": 0,
                        "output_tokens": 0
                    }
                
                model_stats = models[model]
                model_stats["calls"] += 1
                model_stats["total_cost"] += entry.get("total_cost", 0)
                model_stats["input_tokens"] += entry.get("input_tokens", 0)
                model_stats["output_tokens"] += entry.get("output_tokens", 0)
            
            summary["models"] = models
            return summary


class RetryHandler:
    """Handles retry logic with exponential backoff"""
    
    RETRYABLE_ERRORS = [
        'rate limit', 'timeout', 'internal server', 'server error',
        'too many requests', '429', '500', '502', '503', '504',
        'connection', 'network', 'unavailable', 'capacity',
        'overloaded', 'throttle', 'exhausted', 'quota',
        'api error', 'service unavailable'
    ]
    
    RETRYABLE_EXCEPTIONS = (
        requests.exceptions.RequestException,
        ConnectionError,
        TimeoutError
    )
    
    @classmethod
    def is_retryable_error(cls, error: Exception) -> bool:
        """Determine if an error should trigger a retry"""
        error_str = str(error).lower()
        
        # Check for error keywords
        if any(indicator in error_str for indicator in cls.RETRYABLE_ERRORS):
            return True
        
        # Check for specific exception types
        if isinstance(error, cls.RETRYABLE_EXCEPTIONS):
            return True
        
        # Check for API-specific errors
        try:
            if isinstance(error, (anthropic.APIError, anthropic.APIConnectionError,
                                anthropic.InternalServerError, anthropic.RateLimitError)):
                return True
        except NameError:
            pass
        
        try:
            if genai and isinstance(error, genai.types.BlockedPromptException):
                return True
        except (NameError, AttributeError):
            pass
        
        return False
    
    @staticmethod
    def exponential_backoff(func, max_retries: int = 10, initial_delay: float = 1,
                          factor: float = 2, jitter: float = 0.1,
                          max_delay: float = 300):
        """Decorator implementing exponential backoff retry logic"""
        def wrapper(*args, **kwargs):
            retries = 0
            delay = initial_delay
            
            while True:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    retries += 1
                    if not RetryHandler.is_retryable_error(e) or retries >= max_retries:
                        logging.error(f"Error in {func.__name__} after {retries} retries: {e}")
                        raise e
                    
                    # Calculate delay with jitter
                    jitter_amount = random.uniform(-jitter * delay, jitter * delay)
                    sleep_time = min(delay + jitter_amount, max_delay)
                    
                    logging.warning(f"API error in {func.__name__}: {e}. "
                                  f"Retry {retries}/{max_retries} in {sleep_time:.2f}s...")
                    time.sleep(sleep_time)
                    delay = min(delay * factor, max_delay)
        
        return wrapper


# Base Classes
class BaseLLMClient(ABC):
    """Abstract base class for LLM clients"""
    
    def __init__(self, model_name: str):
        if model_name not in Config.MODELS:
            raise ValueError(f"Unsupported model: {model_name}")
        
        self.model_name = model_name
        self.config = Config.MODELS[model_name]
        self.usage_tracker = UsageTracker()
    
    @abstractmethod
    def _make_api_call(self, prompt: str, max_tokens: Optional[int]) -> Tuple[str, Dict[str, Any]]:
        """Make the actual API call. Must return (response, usage_info)"""
        pass
    
    def _create_preview(self, text: str) -> str:
        """Create a preview of text for logging"""
        if len(text) <= Config.MAX_PREVIEW_LENGTH:
            return text
        return text[:Config.MAX_PREVIEW_LENGTH] + "..."
    
    def _calculate_cost(self, input_tokens: int, output_tokens: int) -> Dict[str, float]:
        """Calculate costs based on token usage"""
        input_cost = (input_tokens / 1_000_000) * self.config.input_cost_per_million
        output_cost = (output_tokens / 1_000_000) * self.config.output_cost_per_million
        return {
            'input_cost': input_cost,
            'output_cost': output_cost,
            'total_cost': input_cost + output_cost
        }
    
    def _validate_max_tokens(self, max_tokens: Optional[int]) -> int:
        """Validate and normalize max_tokens parameter"""
        if max_tokens is None:
            return min(Config.DEFAULT_MAX_TOKENS, self.config.max_tokens_limit)
        return min(int(max_tokens), self.config.max_tokens_limit)
    
    @RetryHandler.exponential_backoff
    def generate(self, prompt: str, max_tokens: Optional[int] = None) -> str:
        """Generate response with retry logic and usage tracking"""
        max_tokens = self._validate_max_tokens(max_tokens)
        
        try:
            response, usage_data = self._make_api_call(prompt, max_tokens)
            
            # Create usage info for tracking
            usage_info = UsageInfo(
                timestamp=datetime.now().isoformat(),
                model=self.model_name,
                input_tokens=usage_data.get('input_tokens', 0),
                output_tokens=usage_data.get('output_tokens', 0),
                input_cost=usage_data.get('input_cost', 0),
                output_cost=usage_data.get('output_cost', 0),
                total_cost=usage_data.get('total_cost', 0),
                prompt_preview=self._create_preview(prompt),
                response_preview=self._create_preview(response)
            )
            
            # Log usage
            self.usage_tracker.log_usage(usage_info)
            
            return response
            
        except Exception as e:
            logging.error(f"Error in {self.__class__.__name__}.generate: {e}")
            raise


# Concrete Implementations
class AnthropicClient(BaseLLMClient):
    """Anthropic Claude API client"""
    
    def __init__(self, model_name: str):
        super().__init__(model_name)
        self.client = anthropic.Anthropic(
            api_key=os.environ.get("ANTHROPIC_API_KEY")
        )
    
    def _make_api_call(self, prompt: str, max_tokens: int) -> Tuple[str, Dict[str, Any]]:
        message = self.client.messages.create(
            model=self.config.model_name,
            max_tokens=max_tokens,
            temperature=self.config.temperature_default,
            messages=[{"role": "user", "content": prompt}]
        )
        
        response = message.content[0].text
        usage_data = {
            'input_tokens': message.usage.input_tokens,
            'output_tokens': message.usage.output_tokens,
        }
        usage_data.update(self._calculate_cost(
            usage_data['input_tokens'], 
            usage_data['output_tokens']
        ))
        
        return response, usage_data

class OpenRouterClient(BaseLLMClient):
    """OpenRouter API client (OpenAI-compatible Chat Completions)."""
    def __init__(self, model_name: str):
        super().__init__(model_name)
        self.api_base = os.environ.get("OPENROUTER_API_BASE", "https://openrouter.ai/api/v1")
        self.api_key = os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("Missing OPENROUTER_API_KEY environment variable.")

        # Optional headers to attribute usage/rankings on openrouter.ai
        self.http_referer = os.environ.get("OPENROUTER_HTTP_REFERER")  # e.g., https://your-site.example
        self.x_title = os.environ.get("OPENROUTER_X_TITLE")            # e.g., "Scott's LLM Tools"

    def _make_api_call(self, prompt: str, max_tokens: int) -> Tuple[str, Dict[str, Any]]:
        url = f"{self.api_base}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.http_referer:
            headers["HTTP-Referer"] = self.http_referer
        if self.x_title:
            headers["X-Title"] = self.x_title

        payload = {
            "model": self.config.model_name,  # e.g., "openai/gpt-5"
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": max_tokens,
            "temperature": self.config.temperature_default,
        }

        resp = requests.post(url, headers=headers, json=payload, timeout=600)
        resp.raise_for_status()
        data = resp.json()

        # OpenAI-compatible response shape
        response_text = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {}) or {}
        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)

        usage_data = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }
        usage_data.update(self._calculate_cost(input_tokens, output_tokens))
        return response_text, usage_data

class AWSBedrockClient(BaseLLMClient):
    """AWS Bedrock Claude client"""
    
    def __init__(self, model_name: str):
        super().__init__(model_name)
        # Uncomment and configure as needed
        # session = boto3.Session(profile_name='bedrockprofile')
        # self.client = session.client('bedrock-runtime', region_name='us-east-1')
        self.client = boto3.client('bedrock-runtime', region_name='us-east-1')
    
    def _make_api_call(self, prompt: str, max_tokens: int) -> Tuple[str, Dict[str, Any]]:
        body = json.dumps({
            "max_tokens": max_tokens,
            "temperature": 0,
            "messages": [{"role": "user", "content": f"\n\nHuman: {prompt}\n\nAssistant:"}],
            "anthropic_version": "bedrock-2023-05-31"
        })
        
        response = self.client.invoke_model(
            body=body,
            modelId=self.config.model_name,
            accept='application/json',
            contentType='application/json'
        )
        
        response_body = json.loads(response.get("body").read())
        text_response = response_body.get("content")[0]["text"]
        
        # AWS Bedrock doesn't provide token counts, so we estimate
        estimated_input_tokens = len(prompt.split()) * 1.3  # Rough estimate
        estimated_output_tokens = len(text_response.split()) * 1.3
        
        usage_data = {
            'input_tokens': int(estimated_input_tokens),
            'output_tokens': int(estimated_output_tokens),
        }
        usage_data.update(self._calculate_cost(
            usage_data['input_tokens'],
            usage_data['output_tokens']
        ))
        
        return text_response, usage_data


class GoogleClient(BaseLLMClient):
    """Google Gemini API client"""
    
    def __init__(self, model_name: str):
        if genai is None:
            raise ImportError("google-generativeai package is required for Google models")
        super().__init__(model_name)
        genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
        self.model = genai.GenerativeModel(
            self.config.model_name,
            generation_config=genai.types.GenerationConfig(temperature=0)
        )
    
    def _make_api_call(self, prompt: str, max_tokens: int) -> Tuple[str, Dict[str, Any]]:
        response = self.model.generate_content(prompt)
        
        # Gemini doesn't provide detailed token usage, estimate
        estimated_input_tokens = len(prompt.split()) * 1.3
        estimated_output_tokens = len(response.text.split()) * 1.3
        
        usage_data = {
            'input_tokens': int(estimated_input_tokens),
            'output_tokens': int(estimated_output_tokens),
        }
        usage_data.update(self._calculate_cost(
            usage_data['input_tokens'],
            usage_data['output_tokens']
        ))
        
        return response.text, usage_data


class GroqClient(BaseLLMClient):
    """Groq API client for various models"""
    
    def __init__(self, model_name: str):
        if Groq is None:
            raise ImportError("groq package is required for Groq models")
        super().__init__(model_name)
        self.client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    
    def _make_api_call(self, prompt: str, max_tokens: int) -> Tuple[str, Dict[str, Any]]:
        completion = self.client.chat.completions.create(
            model=self.config.model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=0
        )
        
        response = completion.choices[0].message.content
        
        # Groq provides token usage
        usage = completion.usage
        usage_data = {
            'input_tokens': usage.prompt_tokens,
            'output_tokens': usage.completion_tokens,
        }
        usage_data.update(self._calculate_cost(
            usage_data['input_tokens'],
            usage_data['output_tokens']
        ))
        
        return response, usage_data


class OpenAIClient(BaseLLMClient):
    """OpenAI API client"""
    
    def __init__(self, model_name: str):
        if OpenAI is None:
            raise ImportError("openai package is required for OpenAI models")
        super().__init__(model_name)
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    
    def _make_api_call(self, prompt: str, max_tokens: int) -> Tuple[str, Dict[str, Any]]:
        # Handle o1 models differently (no system messages, different parameter)
        if 'o1' in self.config.model_name:
            completion = self.client.chat.completions.create(
                model=self.config.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=max_tokens
            )
        else:
            completion = self.client.chat.completions.create(
                model=self.config.model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=self.config.temperature_default
            )
        
        response = completion.choices[0].message.content
        usage = completion.usage
        
        usage_data = {
            'input_tokens': usage.prompt_tokens,
            'output_tokens': usage.completion_tokens,
        }
        usage_data.update(self._calculate_cost(
            usage_data['input_tokens'],
            usage_data['output_tokens']
        ))
        
        return response, usage_data


class OllamaClient(BaseLLMClient):
    """Ollama API client for local LLM inference"""
    
    def __init__(self, model_name: str):
        super().__init__(model_name)
        # Extract the actual model name from the prefixed name (ollama_XXX or ollama:XXX)
        if self.model_name.startswith('ollama_'):
            self.actual_model_name = self.model_name.replace('ollama_', '', 1)
        elif self.model_name.startswith('ollama:'):
            self.actual_model_name = self.model_name.replace('ollama:', '', 1)
        else:
            self.actual_model_name = self.model_name
        self.api_base = Config.OLLAMA_API_BASE
    
    def _make_api_call(self, prompt: str, max_tokens: int) -> Tuple[str, Dict[str, Any]]:
        """Make API call to Ollama"""
        payload = {
            "model": self.actual_model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.config.temperature_default,
                "num_predict": max_tokens
            }
        }
        
        try:
            response = requests.post(f"{self.api_base}/api/generate", json=payload)
            response.raise_for_status()
            result = response.json()
            
            response_text = result.get("response", "")
            
            # Ollama doesn't provide token counts, so we estimate
            estimated_input_tokens = len(prompt.split()) * 1.3  # Rough estimate
            estimated_output_tokens = len(response_text.split()) * 1.3
            
            usage_data = {
                'input_tokens': int(estimated_input_tokens),
                'output_tokens': int(estimated_output_tokens),
            }
            usage_data.update(self._calculate_cost(
                usage_data['input_tokens'],
                usage_data['output_tokens']
            ))
            
            return response_text, usage_data
        except Exception as e:
            logging.error(f"Ollama API error: {e}")
            raise


# Main Interface
class LLMClientFactory:
    """Factory for creating LLM clients based on provider"""
    
    _provider_clients = {
        'anthropic': AnthropicClient,
        'aws_bedrock': AWSBedrockClient,
        'google': GoogleClient,
        'groq': GroqClient,
        'openai': OpenAIClient,
        'ollama': OllamaClient,
        'openrouter': OpenRouterClient
    }
    
    @classmethod
    def get_client(cls, model_name: str) -> BaseLLMClient:
        """Get a client instance for the specified model"""
        # Handle Ollama models before checking if they exist in Config.MODELS
        if (model_name.startswith('ollama_') or model_name.startswith('ollama:')) and model_name not in Config.MODELS:
            # Standardize the model name format for internal use
            standardized_name = model_name
            if model_name.startswith('ollama:'):
                # Convert ollama: prefix to ollama_ for consistency
                standardized_name = 'ollama_' + model_name[7:]
                # Also register with the original name to handle direct lookups
                Config.MODELS[model_name] = ModelConfig(
                    provider='ollama',
                    model_name=standardized_name,
                    max_tokens_limit=Config.OLLAMA_DEFAULT_MAX_TOKENS,
                    input_cost_per_million=0.0,  # Free/local
                    output_cost_per_million=0.0,  # Free/local
                    temperature_default=0.1
                )
            
            Config.MODELS[standardized_name] = ModelConfig(
                provider='ollama',
                model_name=standardized_name,
                max_tokens_limit=Config.OLLAMA_DEFAULT_MAX_TOKENS,
                input_cost_per_million=0.0,  # Free/local
                output_cost_per_million=0.0,  # Free/local
                temperature_default=0.1
            )
            
            # If we converted the name, use the standardized version
            if model_name.startswith('ollama:'):
                model_name = standardized_name
        # Handle OpenRouter models before checking Config.MODELS
        # Accept both "openrouter:<MODEL_ID>" and "openrouter:<MODEL_ID>"
        if (model_name.startswith('openrouter_') or model_name.startswith('openrouter:')) and model_name not in Config.MODELS:
            standardized_name = model_name
            if model_name.startswith('openrouter:'):
                # Convert "openrouter:" prefix to "openrouter_" for consistency
                standardized_name = 'openrouter:' + model_name[11:]

            # Actual OpenRouter model id after the prefix (e.g., "openai/gpt-5")
            actual_model_id = standardized_name.replace('openrouter:', '', 1)

            # Known models with accurate pricing & output limits
            known_openrouter = {
                # Sources: OpenRouter model pages & announcements
                # GPT-5: 400K context, ~$1.25/M in, $10/M out; ~128K max output
                'openai/gpt-5': {
                    'max_tokens_limit': 128000,
                    'input_cost_per_million': 1.25,
                    'output_cost_per_million': 10.00,
                },
                # Claude Sonnet 4.5: up to 1M context, starting at $3/$15; ~64K max output
                'anthropic/claude-sonnet-4.5': {
                    'max_tokens_limit': 65536,
                    'input_cost_per_million': 3.00,
                    'output_cost_per_million': 15.00,
                },
                # Gemini 2.5 Pro: ~1.05M context, starting at $1.25/$10; ~65.5K max output
                'google/gemini-2.5-pro': {
                    'max_tokens_limit': 65536,
                    'input_cost_per_million': 1.25,
                    'output_cost_per_million': 10.00,
                },
            }

            cfg = known_openrouter.get(actual_model_id)

            if cfg:
                # Accurate known config
                Config.MODELS[standardized_name] = ModelConfig(
                    provider='openrouter',
                    model_name=actual_model_id,
                    max_tokens_limit=cfg['max_tokens_limit'],
                    input_cost_per_million=cfg['input_cost_per_million'],
                    output_cost_per_million=cfg['output_cost_per_million'],
                    temperature_default=0.1,
                )
                # Also register the original name to handle direct lookups
                Config.MODELS[model_name] = Config.MODELS[standardized_name]
            else:
                # Generic fallback for any OpenRouter model id
                Config.MODELS[standardized_name] = ModelConfig(
                    provider='openrouter',
                    model_name=actual_model_id,
                    max_tokens_limit=Config.DEFAULT_MAX_TOKENS,
                    input_cost_per_million=0.0,   # Unknown price
                    output_cost_per_million=0.0,  # Unknown price
                    temperature_default=0.1,
                )
                Config.MODELS[model_name] = Config.MODELS[standardized_name]

            # If we converted the name, use the standardized version
            if model_name.startswith('openrouter:'):
                model_name = standardized_name

        if model_name not in Config.MODELS:
            available_models = list(Config.MODELS.keys())
            raise ValueError(f"Unknown model: {model_name}. Available models: {available_models}")
        
        model_config = Config.MODELS[model_name]
        provider = model_config.provider
        
        if provider not in cls._provider_clients:
            raise ValueError(f"Unknown provider: {provider}")
        
        try:
            return cls._provider_clients[provider](model_name)
        except Exception as e:
            logging.error(f"Failed to initialize {model_name} client: {e}")
            raise


def ask_ai(prompt: str, model_name: str, max_tokens: Optional[int] = None) -> str:
    """
    Main function to interact with AI models using actual model names
    
    Args:
        prompt: The input prompt
        model_name: Exact model name (e.g., 'claude-3-5-sonnet-20241022')
                   For Ollama models, use either 'ollama_' prefix or 'ollama:' prefix
                   Examples: 'ollama_qwen3:30b-a3b' or 'ollama:hf.co/unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF:UD-Q4_K_XL'
        max_tokens: Maximum tokens to generate
        
    Returns:
        AI response as string
        
    Raises:
        ValueError: If model is not supported
        Exception: If API call fails after retries
    """
    print(f"Asking {model_name}")
    
    # Ollama model registration is now handled in LLMClientFactory.get_client
    client = LLMClientFactory.get_client(model_name)
    return client.generate(prompt, max_tokens)


def list_available_models() -> Dict[str, Dict[str, Any]]:
    """Get list of all available models with their details"""
    models = {}
    for model_name, config in Config.MODELS.items():
        models[model_name] = {
            'provider': config.provider,
            'max_tokens': config.max_tokens_limit,
            'input_cost_per_million': config.input_cost_per_million,
            'output_cost_per_million': config.output_cost_per_million
        }
    return models


def get_usage_stats() -> Dict[str, Any]:
    """Get aggregated usage statistics"""
    tracker = UsageTracker()
    return tracker.get_total_usage()


def main():
    """Main function for testing and CLI usage"""
    load_dotenv()
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Show available models
    print("Available models:")
    models = list_available_models()
    for model_name, details in models.items():
        print(f"  {model_name} ({details['provider']}) - "
              f"${details['input_cost_per_million']:.2f}/${details['output_cost_per_million']:.2f} per 1M tokens")
    
    # Test the system
    try:
        # Use actual model name
        response = ask_ai('Why is the sky blue?', 'openrouter:gpt-5')
        print(f"\nResponse: {response}")
        
        # Show usage stats
        stats = get_usage_stats()
        print(f"\nUsage Statistics: {json.dumps(stats, indent=2)}")
        
    except Exception as e:
        logging.error(f"Test failed: {e}")


if __name__ == "__main__":
    main()