"""
Sample 9G: Structured Output with Pydantic Models (Google Gemini)
==================================================================
Same as 09_structured_output.py but uses Google's Gemini model.

Forces the LLM to return typed, validated data instead of free text.
This is the most reliable way to get structured responses in production.

Key concepts:
  - Pydantic BaseModel as an output schema
  - llm.with_structured_output(Schema) to constrain the LLM
  - Nested models for complex data structures
  - include_raw=True for debugging and logging

Run with New Relic:
  NEW_RELIC_CONFIG_FILE=newrelic.ini newrelic-admin run-program python 09_structured_output_google.py
"""

import time
from typing import Literal
from pydantic import BaseModel, Field
from config import get_google_llm
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai.chat_models import ChatGoogleGenerativeAIError

import newrelic.agent

llm = get_google_llm(temperature=0)


# ===========================================================================
# Step 1: Define Pydantic models (these ARE the output schema)
# ===========================================================================

class MovieReview(BaseModel):
    """Structured representation of a movie review."""
    title: str = Field(description="The movie title")
    rating: int = Field(description="Rating from 1 to 10")
    sentiment: Literal["positive", "negative", "mixed"] = Field(
        description="Overall sentiment of the review"
    )
    summary: str = Field(description="One-sentence summary of the review")


class Person(BaseModel):
    """A team member with their role and skills."""
    name: str = Field(description="Full name")
    role: str = Field(description="Job title or role")
    skills: list[str] = Field(description="List of key skills")


class TeamRoster(BaseModel):
    """A team with its members."""
    team_name: str = Field(description="Name of the team")
    members: list[Person] = Field(description="List of team members")
    size: int = Field(description="Number of people on the team")


# ===========================================================================
# Step 2: Retry helper
# ===========================================================================

def invoke_with_retry(runnable, input_data, max_retries=3):
    """Invoke a LangChain runnable with retry on errors."""
    for attempt in range(max_retries):
        try:
            return runnable.invoke(input_data)
        except ChatGoogleGenerativeAIError as e:
            wait = 30 * (attempt + 1)
            print(f"  [Error: {e}]")
            print(f"  [Retrying in {wait}s — attempt {attempt + 1}/{max_retries}]")
            time.sleep(wait)
    print("  [Max retries reached, skipping this call]")
    return None


# ===========================================================================
# Step 3: Run examples
# ===========================================================================

app = newrelic.agent.register_application(timeout=30)

try:
    with newrelic.agent.BackgroundTask(app, name="structured-output-google", group="LangChain"):

        # -------------------------------------------------------------------
        # Example A: Basic extraction — parse a movie review
        # -------------------------------------------------------------------
        print("=" * 60)
        print("Example A: Extract structured data from a movie review (Gemini)")
        print("=" * 60)

        structured_llm = llm.with_structured_output(MovieReview)

        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are a movie review analyzer. Extract structured data "
             "from the user's review."),
            ("human", "{review}"),
        ])

        chain = prompt | structured_llm

        review_text = (
            "I just watched Inception and it completely blew my mind! "
            "The layered dream sequences were incredibly creative, and "
            "Leonardo DiCaprio delivered a powerful performance. The only "
            "downside was the pacing in the first 20 minutes felt a bit "
            "slow. Overall though, a masterpiece — I'd give it a 9 out of 10."
        )

        result = invoke_with_retry(chain, {"review": review_text})
        if result:
            print(f"  Type:      {type(result).__name__}")
            print(f"  Title:     {result.title}")
            print(f"  Rating:    {result.rating}/10")
            print(f"  Sentiment: {result.sentiment}")
            print(f"  Summary:   {result.summary}")

        # -------------------------------------------------------------------
        # Example B: Nested models — extract a team roster
        # -------------------------------------------------------------------
        print("\n" + "=" * 60)
        print("Example B: Nested Pydantic models — team extraction")
        print("=" * 60)

        structured_llm_team = llm.with_structured_output(TeamRoster)

        prompt_team = ChatPromptTemplate.from_messages([
            ("system",
             "You are an HR assistant. Extract the team information "
             "from the description into a structured roster."),
            ("human", "{description}"),
        ])

        chain_team = prompt_team | structured_llm_team

        team_text = (
            "The AI Platform team has three members. Sarah Chen is the "
            "tech lead — she's an expert in Python, distributed systems, "
            "and ML infrastructure. Marcus Johnson is a senior engineer "
            "who specializes in Kubernetes, Go, and CI/CD pipelines. "
            "Priya Patel is a data scientist skilled in PyTorch, "
            "statistics, and data visualization."
        )

        result = invoke_with_retry(chain_team, {"description": team_text})
        if result:
            print(f"  Team: {result.team_name} ({result.size} members)")
            for person in result.members:
                print(f"    - {person.name} ({person.role})")
                print(f"      Skills: {', '.join(person.skills)}")

        # -------------------------------------------------------------------
        # Example C: include_raw=True — see raw LLM output + parsed result
        # -------------------------------------------------------------------
        print("\n" + "=" * 60)
        print("Example C: include_raw=True — raw + parsed output")
        print("=" * 60)

        structured_llm_raw = llm.with_structured_output(
            MovieReview, include_raw=True
        )

        chain_raw = prompt | structured_llm_raw

        result = invoke_with_retry(chain_raw, {"review": review_text})
        if result:
            print(f"  Keys returned: {list(result.keys())}")
            print(f"  Parsing error: {result['parsing_error']}")
            print(f"  Parsed type:   {type(result['parsed']).__name__}")
            print(f"  Parsed title:  {result['parsed'].title}")
            raw_content = str(result['raw'].content)[:120]
            print(f"  Raw content:   {raw_content}...")

finally:
    newrelic.agent.shutdown_agent(timeout=10)
