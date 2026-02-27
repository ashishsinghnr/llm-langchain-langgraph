"""
Shared Configuration — LLM Client Setup
=========================================
Centralizes the NERD_COMPLETION proxy configuration for both OpenAI and
Google Gemini models.  Each sample file can import from here instead of
duplicating the setup code.

Usage:
    from config import get_openai_llm, get_google_llm, get_embeddings

    llm = get_openai_llm()               # GPT-5 via proxy
    llm = get_google_llm(temperature=0)   # Gemini via proxy
    emb = get_embeddings()                # text-embedding-3-small via proxy
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── Environment variables ─────────────────────────────────────────────
NERD_API_TOKEN = os.environ.get("NERD_COMPLETION_API_TOKEN")
NERD_BASE_URL = os.environ.get("NERD_COMPLETION_BASE_URL")


# ── OpenAI (via NERD_COMPLETION proxy) ────────────────────────────────

def get_openai_llm(model: str = "gpt-5", temperature: float = 0.7):
    """Return a ChatOpenAI instance configured for the NERD_COMPLETION proxy."""
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=model,
        temperature=temperature,
        api_key=NERD_API_TOKEN,
        base_url=NERD_BASE_URL,
    )


# ── Google Gemini (via NERD_COMPLETION proxy) ─────────────────────────

def get_google_llm(model: str = "gemini-2.5-flash", temperature: float = 0.7):
    """Return a ChatGoogleGenerativeAI instance configured for NERD_COMPLETION.

    The proxy requires api_version='v1' (the SDK defaults to v1beta which
    returns 404).  We create a genai.Client with the correct http_options
    and override llm.client after init (the pydantic validator in
    ChatGoogleGenerativeAI always overwrites the client field).
    """
    from google import genai
    from langchain_google_genai import ChatGoogleGenerativeAI

    google_client = genai.Client(
        api_key=NERD_API_TOKEN,
        http_options={
            "base_url": NERD_BASE_URL,
            "api_version": "v1",
            "headers": {
                "Authorization": NERD_API_TOKEN,
            },
        },
    )

    llm = ChatGoogleGenerativeAI(
        model=model,
        temperature=temperature,
        api_key=NERD_API_TOKEN,
    )
    llm.client = google_client  # Override — see docstring above
    return llm


# ── Embeddings (OpenAI via NERD_COMPLETION proxy) ─────────────────────

def get_embeddings(model: str = "text-embedding-3-small"):
    """Return an OpenAIEmbeddings instance configured for the NERD_COMPLETION proxy."""
    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings(
        model=model,
        api_key=NERD_API_TOKEN,
        base_url=NERD_BASE_URL,
    )
