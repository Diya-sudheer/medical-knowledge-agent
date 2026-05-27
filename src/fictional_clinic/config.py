from functools import lru_cache
import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel


ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data" / "clinic_docs"
WEB_DIR = Path(__file__).resolve().parent / "web"


class Settings(BaseModel):
    openai_api_key: str | None = None
    openai_model: str = "gpt-4.1-mini"
    openai_fine_tuned_model: str | None = None
    use_openai: bool = False
    enable_live_kg: bool = False

    @property
    def response_model(self) -> str:
        return self.openai_fine_tuned_model or self.openai_model


@lru_cache
def get_settings() -> Settings:
    load_dotenv()
    return Settings(
        openai_api_key=os.getenv("OPENAI_API_KEY") or None,
        openai_model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
        openai_fine_tuned_model=os.getenv("OPENAI_FINE_TUNED_MODEL") or None,
        use_openai=os.getenv("USE_OPENAI", "false").lower() in {"1", "true", "yes"},
        enable_live_kg=os.getenv("ENABLE_LIVE_KG", "false").lower() in {"1", "true", "yes"},
    )
