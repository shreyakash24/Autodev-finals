"""
LLM Configuration Utility
Supports both OpenAI and Groq API providers
"""

import os
from typing import Optional


def get_llm(temperature: float = 0.2, model: Optional[str] = None):
    """
    Get configured LLM instance based on environment variables.
    
    Args:
        temperature: Temperature for generation (0.0 to 1.0)
        model: Override model name (optional)
        
    Returns:
        LLM instance (ChatOpenAI or ChatGroq) or None if no API key available
        
    Environment Variables:
        LLM_PROVIDER: 'openai' or 'groq' (default: 'openai')
        OPENAI_API_KEY: Required if LLM_PROVIDER=openai
        GROQ_API_KEY: Required if LLM_PROVIDER=groq
        LLM_MODEL: Model name override (optional)
    """
    provider = os.getenv('LLM_PROVIDER', 'openai').lower()
    model_override = os.getenv('LLM_MODEL', model)
    
    if provider == 'groq':
        groq_api_key = os.getenv('GROQ_API_KEY')
        if not groq_api_key:
            return None
        
        try:
            
            from langchain_groq import ChatGroq
            
            # Default Groq model if not specified
            default_model = 'llama3-70b-8192'
            final_model = model_override or default_model
            print(f"Using Groq model: {final_model}")
            return ChatGroq(
                groq_api_key=groq_api_key,
                model_name=final_model,
                temperature=temperature
            )
        except ImportError:
            return None
        except Exception as e:
            print(f"Error initializing Groq LLM: {e}")
            return None
    
    else:  # Default to OpenAI
        openai_api_key = os.getenv('OPENAI_API_KEY')
        if not openai_api_key:
            return None
        
        try:
            from langchain_openai import ChatOpenAI
            
            # Default OpenAI model if not specified
            default_model = 'gpt-4'
            final_model = model_override or default_model
            
            return ChatOpenAI(
                model=final_model,
                temperature=temperature
            )
        except ImportError:
            return None
        except Exception as e:
            print(f"Error initializing OpenAI LLM: {e}")
            return None
            return None


def get_llm_info() -> dict:
    """
    Get information about current LLM configuration.
    
    Returns:
        Dictionary with provider, model, and availability info
    """
    provider = os.getenv('LLM_PROVIDER', 'openai').lower()
    model = os.getenv('LLM_MODEL')
    
    if provider == 'groq':
        api_key = os.getenv('GROQ_API_KEY')
        default_model = 'llama3-70b-8192'
        # Check if library is installed
        try:
            import langchain_groq
            library_installed = True
        except ImportError:
            library_installed = False
    else:
        api_key = os.getenv('OPENAI_API_KEY')
        default_model = 'gpt-4'
        # Check if library is installed
        try:
            import langchain_openai
            library_installed = True
        except ImportError:
            library_installed = False
    
    return {
        'provider': provider,
        'model': model or default_model,
        'available': bool(api_key) and library_installed,
        'api_key_set': bool(api_key),
        'library_installed': library_installed
    }
