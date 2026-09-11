import os
from pathlib import Path
from dotenv import load_dotenv

# Search for .env.openai in the current working directory, workspace root, or module directory
def _load_environment() -> None:
    current_file = Path(__file__).resolve()
    potential_paths = [
        Path.cwd() / ".env.openai",
        current_file.parent.parent / ".env.openai",          # src/shoppingassistantservice/.env.openai
        current_file.parent.parent.parent.parent / ".env.openai", # repo root
        Path.cwd() / ".env",
        current_file.parent.parent / ".env",
        current_file.parent.parent.parent.parent / ".env",
    ]
    for env_path in potential_paths:
        if env_path.is_file():
            load_dotenv(dotenv_path=env_path, override=False)
            break
    else:
        # Fallback to standard load_dotenv search
        load_dotenv()

_load_environment()

class Settings:
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    OPENAI_EMBEDDING_MODEL: str = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

    # Chroma DB configurations
    CHROMA_HOST: str = os.getenv("CHROMA_HOST", "localhost")
    CHROMA_PORT: int = int(os.getenv("CHROMA_PORT", "8000"))
    CHROMA_COLLECTION: str = os.getenv("CHROMA_COLLECTION", "online_boutique_products")

    # Service configuration
    PORT: int = int(os.getenv("PORT", "8080"))
    HOST: str = os.getenv("HOST", "0.0.0.0")

    @property
    def is_openai_configured(self) -> bool:
        return bool(self.OPENAI_API_KEY and self.OPENAI_API_KEY != "your_openai_api_key_here")

settings = Settings()
