from typing import Mapping, Optional, Literal

EmbeddingProvider = Literal["google", "openai"]


def resolve_embedding_provider(env: Mapping[str, str]) -> EmbeddingProvider:
    """Determine which embedding provider to use based on env vars."""
    raw_value = env.get("EMBEDDING_PROVIDER", "").strip().lower()
    if raw_value == "openai":
        return "openai"
    return "google"


def resolve_embedding_model(env: Mapping[str, str]) -> str:
    """Resolve the embedding model name based on provider."""
    provider = resolve_embedding_provider(env)
    
    if provider == "openai":
        default_model = "text-embedding-3-small"
        raw_value = env.get("OPENAI_EMBEDDING_MODEL", "").strip()
        return raw_value if raw_value else default_model
    else:
        default_model = "models/gemini-embedding-001"
        raw_value = env.get("GOOGLE_EMBEDDING_MODEL", "").strip()
        if not raw_value:
            return default_model
        if raw_value.startswith("models/"):
            return raw_value
        return f"models/{raw_value}"


def resolve_embedding_dimensions(env: Mapping[str, str]) -> Optional[int]:
    """Resolve embedding dimensions based on provider."""
    provider = resolve_embedding_provider(env)
    
    if provider == "openai":
        raw_value = env.get("OPENAI_EMBEDDING_DIMENSIONS", "").strip()
    else:
        raw_value = env.get("GOOGLE_EMBEDDING_DIMENSIONS", "").strip()
    
    if not raw_value:
        return None
    try:
        value = int(raw_value)
    except ValueError:
        return None
    return value if value > 0 else None
