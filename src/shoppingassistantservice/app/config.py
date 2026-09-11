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

    # Pinecone Vector DB configurations
    PINECONE_API_KEY: str = os.getenv("PINECONE_API_KEY", "")
    PINECONE_INDEX_NAME: str = os.getenv("PINECONE_INDEX_NAME", "online-boutique-products")
    PINECONE_ENVIRONMENT: str = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")
    PINECONE_NAMESPACE: str = os.getenv("PINECONE_NAMESPACE", "products")
    PINECONE_DIMENSION: int = int(os.getenv("PINECONE_DIMENSION", "1536"))

    # LangChain & LangSmith Observability configurations
    LANGCHAIN_TRACING_V2: str = os.getenv("LANGCHAIN_TRACING_V2", "false")
    LANGCHAIN_ENDPOINT: str = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
    LANGCHAIN_API_KEY: str = os.getenv("LANGCHAIN_API_KEY", "")
    LANGCHAIN_PROJECT: str = os.getenv("LANGCHAIN_PROJECT", "online-boutique-shopping-assistant")

    # Service configuration
    PORT: int = int(os.getenv("PORT", "8080"))
    HOST: str = os.getenv("HOST", "0.0.0.0")

    @property
    def is_openai_configured(self) -> bool:
        return bool(self.OPENAI_API_KEY and self.OPENAI_API_KEY != "your_openai_api_key_here")

    @property
    def is_pinecone_configured(self) -> bool:
        return bool(self.PINECONE_API_KEY and self.PINECONE_API_KEY != "your_pinecone_api_key_here")

    @property
    def is_langsmith_configured(self) -> bool:
        return bool(
            self.LANGCHAIN_API_KEY
            and self.LANGCHAIN_API_KEY != "your_langchain_api_key_here"
            and self.LANGCHAIN_TRACING_V2.lower() in ("true", "1")
        )

settings = Settings()

def setup_langsmith_environment():
    if settings.is_langsmith_configured:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_ENDPOINT"] = settings.LANGCHAIN_ENDPOINT
        os.environ["LANGCHAIN_API_KEY"] = settings.LANGCHAIN_API_KEY
        os.environ["LANGCHAIN_PROJECT"] = settings.LANGCHAIN_PROJECT

setup_langsmith_environment()
