import os
from typing import Optional
from pydantic import BaseModel, Field, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict

from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential

# DOTENV = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
# print(f"DOTENV file path: {DOTENV}")

class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    All settings are validated through Pydantic's type system.
    """
    # project_endpoint: str = Field(..., env="PROJECT_ENDPOINT")
    project_endpoint: str = Field(..., env="PROJECT_ENDPOINT")
    model_deployment_name: str = Field(..., env="MODEL_DEPLOYMENT_NAME")
    agent_name: str = Field(..., env="AGENT_NAME")
    azure_openai_endpoint: str = Field(..., env="AZURE_OPENAI_ENDPOINT")
    azure_openai_api_key: str = Field(..., env="AZURE_OPENAI_API_KEY")
    azure_openai_api_version: str = Field(..., env="AZURE_OPENAI_API_VERSION")

    # bing_connection_name: str = Field(..., env="BING_CONNECTION_NAME")
    # project_api_version: str = Field(..., env="PROJECT_API_VERSION")
    # azure_search_connection_name: str = Field(..., env="AZURE_SEARCH_CONNECTION_NAME")
    # azure_search_index_name: str = Field(..., env="AZURE_SEARCH_INDEX_NAME")

    # enable_telemetry: bool = Field(True, env="ENABLE_TELEMETRY")
    # azure_monitor_connection_string: Optional[str] = Field(None, env="AZURE_MONITOR_CONNECTION_STRING")
 
    class Config:
        """Pydantic model configuration"""
        # env_file = f"{DOTENV}"
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False # variable names are lowcase in the Settings

class AuthHelper:
    @classmethod
    def load_settings(cls):
        try:
            return Settings()
        except ValidationError as e:
            missing = [err['loc'][0] for err in e.errors()]
            raise EnvironmentError(f"Missing required env keys: {missing}")

    @staticmethod
    def test_credential():
        """
        Test Azure authentication and return a credential object.
        Falls back to InteractiveBrowserCredential if DefaultAzureCredential fails.
        """
        credential = None
        try:
            credential = DefaultAzureCredential()
            # credential.get_token("https://management.azure.com/.default")
            # print("DefaultAzureCredential authentication OK")
        except Exception:
            credential = InteractiveBrowserCredential()
            # print("Falling back to InteractiveBrowserCredential")
        return credential
    
# import pydantic 
# print(f"Pydantic version: {pydantic.VERSION}")