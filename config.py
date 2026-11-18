"""
config.py - Model provider setup for the Patient Message Triage demo.

Supports two providers:
  • Ollama         (default, local model served by Ollama)
  • AWS Bedrock    (optional, set USE_BEDROCK=true plus standard AWS credentials)
"""

import os

from dotenv import load_dotenv

load_dotenv()


def get_model():
    """Return a Strands model instance based on environment configuration."""

    use_bedrock = os.getenv("USE_BEDROCK", "false").lower() == "true"

    if use_bedrock:
        from strands.models import BedrockModel

        model_id = os.getenv("BEDROCK_MODEL_ID")
        if not model_id:
            raise ValueError("Set BEDROCK_MODEL_ID when USE_BEDROCK=true.")
        region = os.getenv("AWS_REGION", "us-east-1")
        print(f"[config] Using AWS Bedrock — model: {model_id} | region: {region}")
        return BedrockModel(model_id=model_id, region_name=region)

    # Default: Ollama
    from strands.models.ollama import OllamaModel

    model_id = os.getenv("OLLAMA_MODEL", "llama3.2")
    host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    return OllamaModel(
        host=host,
        model_id=model_id,
        max_tokens=4096,
        temperature=0.1,
    )
