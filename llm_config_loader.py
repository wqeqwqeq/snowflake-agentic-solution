import json
import os
from typing import Dict, Any
from dotenv import load_dotenv
from openai import AzureOpenAI

# Load environment variables from .env file
load_dotenv()

class LLMConfigLoader:
    """
    Utility class to load LLM configuration and create Azure OpenAI clients.
    """
    
    def __init__(self, config_file: str = "llm_config.json"):
        """
        Initialize the config loader.
        
        Args:
            config_file (str): Path to the configuration JSON file
        """
        self.config_file = config_file
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """
        Load configuration from JSON file.
        
        Returns:
            Dict[str, Any]: Configuration dictionary
        """
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file {self.config_file} not found")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in configuration file: {e}")
    
    def create_azure_openai_client(self, agent_type: str) -> AzureOpenAI:
        """
        Create and return an Azure OpenAI client using environment variables and agent-specific config.
        
        Args:
            agent_type (str): Either 'text_to_sql_agent' or 'sql_execution_agent'
            
        Returns:
            AzureOpenAI: Configured Azure OpenAI client
        """
        agent_config = self.config.get(agent_type, {})
        azure_config = agent_config.get("azure_openai", {})
        
        return AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=azure_config.get("api_version", "2024-02-15-preview"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
        )
    
    def get_model_name(self, agent_type: str) -> str:
        """
        Get the model name from configuration for a specific agent.
        
        Args:
            agent_type (str): Either 'text_to_sql_agent' or 'sql_execution_agent'
            
        Returns:
            str: Model name
        """
        agent_config = self.config.get(agent_type, {})
        azure_config = agent_config.get("azure_openai", {})
        return azure_config.get("model", "gpt-4o")
    
    def get_temperature(self, agent_type: str) -> float:
        """
        Get the temperature setting from configuration for a specific agent.
        
        Args:
            agent_type (str): Either 'text_to_sql_agent' or 'sql_execution_agent'
            
        Returns:
            float: Temperature value
        """
        agent_config = self.config.get(agent_type, {})
        azure_config = agent_config.get("azure_openai", {})
        return azure_config.get("temperature", 0)
    
    def get_text_to_sql_config(self) -> Dict[str, Any]:
        """
        Get configuration for TextToSQLAgent.
        
        Returns:
            Dict[str, Any]: TextToSQLAgent configuration
        """
        return self.config.get("text_to_sql_agent", {})
    
    def get_sql_execution_config(self) -> Dict[str, Any]:
        """
        Get configuration for SQLExecutionAgent.
        
        Returns:
            Dict[str, Any]: SQLExecutionAgent configuration
        """
        return self.config.get("sql_execution_agent", {})
    
    def get_system_prompt(self, agent_type: str) -> str:
        """
        Get system prompt for a specific agent type.
        
        Args:
            agent_type (str): Either 'text_to_sql_agent' or 'sql_execution_agent'
            
        Returns:
            str: System prompt
        """
        agent_config = self.config.get(agent_type, {})
        return agent_config.get("system_prompt", "")
    
    def get_tools(self, agent_type: str) -> list:
        """
        Get tools configuration for a specific agent type.
        
        Args:
            agent_type (str): Either 'text_to_sql_agent' or 'sql_execution_agent'
            
        Returns:
            list: Tools configuration
        """
        agent_config = self.config.get(agent_type, {})
        return agent_config.get("tools", []) 