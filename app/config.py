from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    llm_base_url: str
    llm_model: str
    embedding_model: str
    chroma_dir: str

    class Config:
        env_file = ".env"


settings = Settings()
