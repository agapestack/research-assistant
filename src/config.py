from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="RAG_")

    # Qdrant
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333

    # Embeddings
    embedding_model: str = "minilm"

    # LLM
    llm_model: str = "qwen3:14b"
    llm_temperature: float = 0.1

    # RAG
    retrieval_k: int = 5


settings = Settings()
