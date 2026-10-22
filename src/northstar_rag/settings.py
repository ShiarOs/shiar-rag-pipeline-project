"""Validated application settings sourced from environment variables."""

from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=None,
        case_sensitive=False,
        extra="ignore",
    )

    groq_api_key: str = Field(default="", repr=False)
    groq_model: str = "openai/gpt-oss-20b"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_revision: str = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
    top_k: int = Field(default=5, ge=1, le=50)
    candidate_k: int = Field(default=12, ge=1, le=200)
    min_score: float = Field(default=0.45, ge=0.0, le=1.0)
    rrf_k: int = Field(default=60, ge=1)
    max_chunk_words: int = Field(default=100, ge=10, le=500)
    overlap_words: int = Field(default=12, ge=0, le=100)
    data_dir: Path = PROJECT_ROOT / "data" / "knowledge_base"
    calibration_file: Path = PROJECT_ROOT / "data" / "evaluation" / "calibration.json"
    evaluation_file: Path = PROJECT_ROOT / "data" / "evaluation" / "questions.json"
    index_dir: Path = PROJECT_ROOT / "storage" / "hybrid-index"

    @field_validator("embedding_revision", mode="before")
    @classmethod
    def blank_revision_uses_pinned_default(cls, value: str | None) -> str:
        if value is None or not str(value).strip():
            return "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
        return str(value).strip()

    @field_validator("min_score")
    @classmethod
    def score_must_be_finite(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError("MIN_SCORE must be a finite value between 0 and 1")
        return value

    @field_validator("overlap_words")
    @classmethod
    def overlap_must_be_smaller_than_chunk(cls, value: int, info) -> int:
        max_words = info.data.get("max_chunk_words", 100)
        if value >= max_words:
            raise ValueError("OVERLAP_WORDS must be smaller than MAX_CHUNK_WORDS")
        return value


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
