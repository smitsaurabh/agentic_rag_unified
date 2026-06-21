"""
Central configuration — reads from environment / .env file.
All other modules import from here; nothing imports os.getenv directly.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from typing import Literal


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── LLM ──────────────────────────────────────────────────────────────────
    llm_provider: Literal["anthropic", "openai"] = "anthropic"

    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")
    anthropic_model: str = "claude-opus-4-8"

    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    openai_model: str = "gpt-4o"

    # ── MySQL ─────────────────────────────────────────────────────────────────
    mysql_host: str = "localhost"
    mysql_port: int = 3306
    mysql_user: str = "root"
    mysql_password: str = ""
    mysql_database: str = "financial_rag"

    @property
    def mysql_url(self) -> str:
        return (
            f"mysql+aiomysql://{self.mysql_user}:{self.mysql_password}"
            f"@{self.mysql_host}:{self.mysql_port}/{self.mysql_database}"
        )

    @property
    def mysql_sync_url(self) -> str:
        return (
            f"mysql+pymysql://{self.mysql_user}:{self.mysql_password}"
            f"@{self.mysql_host}:{self.mysql_port}/{self.mysql_database}"
        )

    # ── Web search ────────────────────────────────────────────────────────────
    tavily_api_key: str = Field(default="", alias="TAVILY_API_KEY")

    # ── Agent knobs ───────────────────────────────────────────────────────────
    max_sql_rows: int = 50
    web_search_results: int = 5
    synthesis_temperature: float = 0.2
    enable_streaming: bool = True

    # ── Bond RAG (PDF) ────────────────────────────────────────────────────────
    # These mirror BOND_RAG__* env vars so the main config can expose them.
    # The bond_rag package reads its own settings via its own Settings class;
    # these are here only so FastAPI / CLI can read them without importing bond_rag.
    bond_rag_db_dir: str = Field(default="db/bond_rag", alias="BOND_RAG__DB_DIR")
    bond_rag_data_dir: str = Field(default="data/pdfs", alias="BOND_RAG__DATA_DIR")
    bond_rag_top_k: int = Field(default=6, alias="BOND_RAG__RETRIEVER__TOP_K")


settings = Settings()
