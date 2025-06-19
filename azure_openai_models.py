import json
import os
from typing import List, Dict, Any
from dotenv import load_dotenv
from openai import AzureOpenAI
from snowflake_conn import snowflake_conn
from llm_config_loader import LLMConfigLoader
from llm_tools import LLMToolRegistry

# Load environment variables from .env file
load_dotenv()

class TextToSQLAgent:
    """
    Model 1: Converts natural language to SQL using Azure OpenAI with function calling.
    Has access to get_table_metadata tool.
    """
    
    def __init__(self, conn, config_loader: LLMConfigLoader = None):
        if config_loader is None:
            config_loader = LLMConfigLoader()
        
        self.config_loader = config_loader
        self.agent_type = "text_to_sql_agent"
        self.client = config_loader.create_azure_openai_client(self.agent_type)
        self.conn = conn
        self.tool_registry = LLMToolRegistry(conn)
        
    def get_table_metadata(self, table_name: str = "H1B_clean") -> str:
        """
        Tool function to get metadata of a Snowflake table.
        
        Args:
            table_name (str): Name of the table to get metadata for
            
        Returns:
            str: Formatted table schema information
        """
        return self.tool_registry.get_table_metadata_tool(table_name)
    
    def generate_sql(self, user_question: str) -> str:
        """
        Convert natural language question to SQL using Azure OpenAI with function calling.
        
        Args:
            user_question (str): Natural language question from user
            
        Returns:
            str: Generated SQL query
        """
        # Get tools and system message from config
        tools = self.config_loader.get_tools("text_to_sql_agent")
        system_message = self.config_loader.get_system_prompt("text_to_sql_agent")

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": f"Generate SQL for this question: {user_question}"}
        ]
        
        # First call to get table metadata
        response = self.client.chat.completions.create(
            model=self.config_loader.get_model_name(self.agent_type),
            messages=messages,
            tools=tools,
            tool_choice="auto",
            temperature=self.config_loader.get_temperature(self.agent_type)
        )
        
        # Handle tool calls
        while response.choices[0].message.tool_calls:
            messages.append(response.choices[0].message)
            
            for tool_call in response.choices[0].message.tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)
                
                if function_name == "get_table_metadata":
                    table_name = function_args.get("table_name", "H1B_clean")
                    function_result = self.get_table_metadata(table_name)
                    
                    messages.append({
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": function_name,
                        "content": function_result
                    })
            
            # Continue the conversation
            response = self.client.chat.completions.create(
                model=self.config_loader.get_model_name(self.agent_type),
                messages=messages,
                tools=tools,
                tool_choice="auto",
                temperature=self.config_loader.get_temperature(self.agent_type)
            )
        
        return response.choices[0].message.content.strip()


class SQLExecutionAgent:
    """
    Model 2: Executes SQL queries and provides natural language responses.
    Has access to execute_sql tool.
    """
    
    def __init__(self, conn, config_loader: LLMConfigLoader = None):
        if config_loader is None:
            config_loader = LLMConfigLoader()
        
        self.config_loader = config_loader
        self.agent_type = "sql_execution_agent"
        self.client = config_loader.create_azure_openai_client(self.agent_type)
        self.conn = conn
        self.tool_registry = LLMToolRegistry(conn)
    
    def execute_sql(self, sql_query: str) -> str:
        """
        Tool function to execute SQL query and fetch results.
        
        Args:
            sql_query (str): SQL query to execute
            
        Returns:
            str: Formatted query results
        """
        return self.tool_registry.execute_sql_tool(sql_query)
    
    def generate_response(self, user_question: str, sql_query: str) -> str:
        """
        Execute SQL query and generate natural language response.
        
        Args:
            user_question (str): Original user question
            sql_query (str): SQL query to execute
            
        Returns:
            str: Natural language response
        """
        # Get tools and system message from config
        tools = self.config_loader.get_tools("sql_execution_agent")
        system_message = self.config_loader.get_system_prompt("sql_execution_agent")

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": f"""
User's Question: {user_question}
SQL Query to Execute: {sql_query}

Please execute this SQL query and provide a natural language answer to the user's question.
"""}
        ]
        
        # Execute the SQL and get response
        response = self.client.chat.completions.create(
            model=self.config_loader.get_model_name(self.agent_type),
            messages=messages,
            tools=tools,
            tool_choice="auto",
            temperature=self.config_loader.get_temperature(self.agent_type)
        )
        
        # Handle tool calls
        while response.choices[0].message.tool_calls:
            messages.append(response.choices[0].message)
            
            for tool_call in response.choices[0].message.tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)
                
                if function_name == "execute_sql":
                    sql_query = function_args["sql_query"]
                    function_result = self.execute_sql(sql_query)
                    
                    messages.append({
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": function_name,
                        "content": function_result
                    })
            
            # Continue the conversation
            response = self.client.chat.completions.create(
                model=self.config_loader.get_model_name(self.agent_type),
                messages=messages,
                tools=tools,
                tool_choice="auto",
                temperature=self.config_loader.get_temperature(self.agent_type)
            )
        
        return response.choices[0].message.content


class TextToSQLPipeline:
    """
    Complete pipeline that combines both models for end-to-end text-to-SQL functionality.
    """
    
    def __init__(self, conn, config_loader: LLMConfigLoader = None):
        if config_loader is None:
            config_loader = LLMConfigLoader()
        
        self.sql_generator = TextToSQLAgent(conn, config_loader)
        self.sql_executor = SQLExecutionAgent(conn, config_loader)
    
    def process_question(self, user_question: str) -> Dict[str, Any]:
        """
        Process a natural language question through the complete pipeline.
        
        Args:
            user_question (str): Natural language question from user
            
        Returns:
            Dict[str, Any]: Results containing SQL query, execution results, and natural language response
        """
        print(f"Processing question: {user_question}")
        print("-" * 60)
        
        # Step 1: Generate SQL using Model 1
        print("Step 1: Generating SQL query...")
        sql_query = self.sql_generator.generate_sql(user_question)
        print(f"Generated SQL: {sql_query}")
        print()
        
        # Step 2: Execute SQL and generate response using Model 2
        print("Step 2: Executing SQL and generating response...")
        natural_response = self.sql_executor.generate_response(user_question, sql_query)
        print(f"Natural Language Response: {natural_response}")
        print()
        
        return {
            "question": user_question,
            "sql_query": sql_query,
            "natural_response": natural_response
        }


def main():
    """
    Main function to demonstrate the two-model text-to-SQL system.
    """
    print("Initializing Snowflake connection...")
    conn = snowflake_conn()
    
    print("Initializing Text-to-SQL Pipeline with Azure OpenAI...")
    pipeline = TextToSQLPipeline(conn)
    
    # Test questions
    test_questions = [
        "What's the average salary for carmax engineer in 2024?",
        "How many H1B applications were filed for software engineer positions?",
        "Show me the top 5 companies by number of H1B applications"
    ]
    
    try:
        for question in test_questions:
            result = pipeline.process_question(question)
            print("=" * 80)
            print()
    finally:
        # Close the connection when done
        print("Closing Snowflake connection...")
        conn.close()


if __name__ == "__main__":
    main() 