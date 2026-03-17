from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    llm_base_url: str
    llm_model: str
    embedding_base_url: str
    embedding_model: str
    chroma_dir: str
    firebase_sa_b64: Optional[str] = None
    google_application_credentials: Optional[str] = None

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore"
    )

    @property
    def chroma_path(self) -> str:
        return str(Path(self.chroma_dir).resolve())

settings = Settings()