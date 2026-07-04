"""
Environment-specific configuration for NyayaQuest.

Uses pydantic-settings to load and validate all environment variables
from .env files with typed defaults and computed properties.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM
    groq_api_key: str = ""

    # Storage
    chroma_persist_dir: str = "./chroma_db_groq_legal"
    redis_url: str = "redis://localhost:6379"
    google_drive_artifact_id: str = ""

    # Security
    allowed_origins: str = ""
    admin_emails: str = ""
    admin_api_key: str = ""

    # Intent Router
    classification_threshold: float = 0.70

    # Observability
    sentry_dsn: str = ""
    environment: str = "development"
    git_commit: str = ""
    log_level: str = "INFO"

    # Firebase
    firebase_project_id: str = ""
    firebase_api_key: str = ""
    firebase_auth_domain: str = ""
    firebase_storage_bucket: str = ""
    firebase_messaging_sender_id: str = ""
    firebase_app_id: str = ""

    @property
    def allowed_origins_list(self) -> list:
        return [o.strip() for o in self.allowed_origins.split(",") if o.strip()]

    @property
    def admin_emails_set(self) -> set:
        return {e.strip() for e in self.admin_emails.split(",") if e.strip()}


settings = Settings()
