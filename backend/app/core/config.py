from pydantic_settings import (
    BaseSettings,
    SettingsConfigDict,
)


class Settings(BaseSettings):
    app_name: str = "MIRA Backend"
    debug: bool = True

    database_url: str = (
        "postgresql://postgres:postgres@localhost:5432/mira"
    )

    # ------------------------------------------------------------------
    # Security and JWT configuration
    # ------------------------------------------------------------------

    secret_key: str = (
        "mira-development-secret-key-change-in-production-min32chars"
    )

    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 60 * 12

    # ------------------------------------------------------------------
    # Seed administrator
    # ------------------------------------------------------------------

    seed_admin_email: str = "admin@mira.gov.in"
    seed_admin_password: str = "Admin@123"

    # ------------------------------------------------------------------
    # Milvus vector store
    # ------------------------------------------------------------------

    milvus_host: str = "localhost"
    milvus_port: str = "19530"
    milvus_collection: str = "material_embeddings"

    # Milvus is an additive candidate source. Matching must continue
    # using rule-based blocking when Milvus is unavailable.
    milvus_enabled: bool = True

    # Final MIRA vector-search budget.
    milvus_top_k: int = 100

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()