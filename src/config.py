from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "TRACEIQ API"
    environment: str = "development"
    log_level: str = "INFO"

    model_config = SettingsConfigDict(env_prefix="TRACEIQ_")


settings = Settings()
