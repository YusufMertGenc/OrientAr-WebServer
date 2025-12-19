from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # LLM (PC'deki Qwen)
    llm_base_url: str
    llm_model: str

    # Embedding (PC'deki Ollama)
    embedding_base_url: str
    embedding_model: str

    # Chroma
    chroma_dir: str

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore"
    )


settings = Settings()
