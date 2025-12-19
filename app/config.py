from pathlib import Path
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    llm_base_url: str
    llm_model: str
    embedding_base_url: str
    embedding_model: str
    chroma_dir: str

    class Config:
        env_file = ".env"

    @property
    def chroma_path(self) -> str:
        return str(Path(self.chroma_dir).resolve())

settings = Settings()
