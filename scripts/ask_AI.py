import boto3
import json
import os
import time
import random
import requests
import base64
from groq import Groq
from openai import AzureOpenAI, OpenAI, AsyncAzureOpenAI, AsyncOpenAI
from dotenv import load_dotenv
import google.generativeai as genai
import anthropic

load_dotenv()

# session = boto3.Session(profile_name='bedrockprofile')
# brt = boto3.client(service_name='bedrock-runtime')

class RateLimitException(Exception):
    pass

def exponential_backoff(func, max_retries=10, initial_delay=1, factor=2, jitter=0.1):
    def wrapper(*args, **kwargs):
        retries = 0
        delay = initial_delay
        while retries < max_retries:
            try:
                return func(*args, **kwargs)
            except RateLimitException as e:
                if retries == max_retries - 1:
                    raise e
                sleep_time = delay * (random.uniform(1 - jitter, 1 + jitter))
                print(f"Rate limit hit. Retrying in {sleep_time:.2f} seconds...")
                time.sleep(sleep_time)
                retries += 1
                delay *= factor
    return wrapper

@exponential_backoff
def ask_aws_claude(prompt, max_tokens=200000):
    load_dotenv()
    newline = "\n\n"
    body = json.dumps({
        "max_tokens": int(max_tokens) if max_tokens is not None else 200000,
        "messages": [{"role": "user",
                    "content": f"{newline}Human: {prompt}{newline}Assistant:"}],
        "anthropic_version": "bedrock-2023-05-31"
    })
 
    modelId = 'anthropic.claude-3-5-sonnet-20240620-v1:0'
    accept = 'application/json'
    contentType = 'application/json'
    
    response = brt.invoke_model(body=body, modelId=modelId, accept=accept, contentType=contentType)
    
    response_body = json.loads(response.get("body").read())
    out = response_body.get("content")[0]["text"]
    print(out)
    return out

@exponential_backoff
def ask_claude(prompt, max_tokens=8192):
    client = anthropic.Anthropic(
        api_key=os.environ.get("ANTHROPIC_API_KEY")
    )

    message = client.messages.create(
        model="claude-3-5-haiku-20241022",
        max_tokens=int(max_tokens) if max_tokens is not None else 8192,
        temperature=0,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    out = message.content[0].text
    print(out)
    return out

@exponential_backoff
def ask_gemini(prompt, max_tokens=None):
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
    model = genai.GenerativeModel("gemini-1.5-flash")
    
    try:
        response = model.generate_content(prompt)
    except genai.types.BlockedPromptException as e:
        if "exhausted" in str(e).lower() or "429" in str(e):
            print(e)
            raise RateLimitException("Gemini API rate limit exceeded")
        raise
    except Exception as e:
        if "429" in str(e) or "exhausted" in str(e).lower():
            print(e)
            raise RateLimitException("Gemini API rate limit exceeded")
        raise
    
    print(response.text)
    return response.text

@exponential_backoff
def ask_gemma2(prompt, max_tokens=200000):
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    model = 'gemma2-9b-it'
    
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=int(max_tokens) if max_tokens is not None else 200000
    )
    
    output = completion.choices[0].message.content
    print(output)
    return output

def ask_llama3(prompt, max_tokens=200000):
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    model = 'llama-3.2-90b-text-preview'
    
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=int(max_tokens) if max_tokens is not None else 200000
    )
    
    output = completion.choices[0].message.content
    print(output)
    return output

@exponential_backoff
def ask_mixtral(prompt, max_tokens=200000):
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    model = "mixtral-8x7b-32768"
    
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=int(max_tokens) if max_tokens is not None else 200000
    )
    
    output = completion.choices[0].message.content
    print(output)
    return output

def ask_ai(prompt, ai_model='claude', max_tokens=None):
    print(f"Asking {ai_model}")
    if ai_model == "claude" or ai_model == 'anthropic_haiku':
        return ask_claude(prompt, max_tokens)
    elif ai_model == "aws_claude":
        return ask_aws_claude(prompt, max_tokens)
    elif ai_model == "gemini":
        return ask_gemini(prompt, max_tokens)
    elif ai_model == "gemma2":
        return ask_gemma2(prompt, max_tokens)
    elif ai_model == "llama3":
        return ask_llama3(prompt, max_tokens)
    elif ai_model == "mixtral":
        return ask_mixtral(prompt, max_tokens)
    else:
        raise ValueError(f"Unknown AI model: {ai_model}")

