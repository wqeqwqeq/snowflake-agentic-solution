#!/usr/bin/env python3
"""
Example usage of the two Azure OpenAI models for text-to-SQL conversion.

This file demonstrates how to use:
1. TextToSQLAgent (Model 1) - Converts natural language to SQL with get_table_metadata tool
2. SQLExecutionAgent (Model 2) - Executes SQL and provides natural language responses
3. TextToSQLPipeline - Complete end-to-end pipeline
"""

from azure_openai_models import TextToSQLAgent, SQLExecutionAgent, TextToSQLPipeline
from snowflake_conn import snowflake_conn

def demo_model_1(conn):
    """Demonstrate Model 1: Natural Language to SQL conversion."""
    print("=" * 60)
    print("DEMO: Model 1 - Natural Language to SQL Conversion")
    print("=" * 60)
    
    # Initialize the text-to-SQL agent
    sql_agent = TextToSQLAgent(conn)
    
    # Test questions
    questions = [
        "What's the average salary for carmax engineer in 2024?",
        "How many H1B applications were filed for software engineer positions?",
        "Show me the top 5 companies by number of H1B applications"
    ]
    
    for question in questions:
        print(f"\nQuestion: {question}")
        print("-" * 40)
        
        # Generate SQL using Model 1
        sql_query = sql_agent.generate_sql(question)
        print(f"Generated SQL: {sql_query}")
        print()

def demo_model_2(conn):
    """Demonstrate Model 2: SQL Execution and Natural Language Response."""
    print("=" * 60)
    print("DEMO: Model 2 - SQL Execution and Natural Language Response")
    print("=" * 60)
    
    # Initialize the SQL execution agent
    execution_agent = SQLExecutionAgent(conn)
    
    # Example: manually provide a SQL query and original question
    user_question = "What's the average salary for carmax engineer in 2024?"
    sql_query = """
    SELECT AVG(prevailing_wage)
    FROM parsed.combined.h1b_clean
    WHERE employer_name LIKE '%CARMAX%'
    AND job_title LIKE '%ENGINEER%'
    AND EXTRACT(year FROM start_date) = 2024
    """
    
    print(f"Original Question: {user_question}")
    print(f"SQL Query: {sql_query}")
    print("-" * 40)
    
    # Generate natural language response using Model 2
    response = execution_agent.generate_response(user_question, sql_query)
    print(f"Natural Language Response: {response}")
    print()

def demo_complete_pipeline(conn):
    """Demonstrate the complete pipeline using both models."""
    print("=" * 60)
    print("DEMO: Complete Pipeline (Model 1 + Model 2)")
    print("=" * 60)
    
    # Initialize the complete pipeline
    pipeline = TextToSQLPipeline(conn)
    
    # Test questions
    questions = [
        "What's the average salary for carmax engineer in 2024?",
        "How many H1B applications were approved last year?",
        "Show me the top 3 companies with highest average salaries"
    ]
    
    for question in questions:
        print(f"\n{'='*80}")
        result = pipeline.process_question(question)
        print(f"Result Summary:")
        print(f"- Question: {result['question']}")
        print(f"- Generated SQL: {result['sql_query']}")
        print(f"- Final Answer: {result['natural_response']}")

def demo_individual_tools(conn):
    """Demonstrate individual tools separately."""
    print("=" * 60)
    print("DEMO: Individual Tools")
    print("=" * 60)
    
    # Test Model 1's get_table_metadata tool
    sql_agent = TextToSQLAgent(conn)
    print("Model 1 - get_table_metadata tool:")
    metadata = sql_agent.get_table_metadata("H1B_clean")
    print(metadata)
    print()
    
    # Test Model 2's execute_sql tool
    execution_agent = SQLExecutionAgent(conn)
    print("Model 2 - execute_sql tool:")
    test_query = "SELECT COUNT(*) FROM parsed.combined.h1b_clean LIMIT 1"
    result = execution_agent.execute_sql(test_query)
    print(f"Query: {test_query}")
    print(f"Result: {result}")

def main():
    """Main function to run all demonstrations."""
    print("Azure OpenAI Text-to-SQL Models Demonstration")
    print("=" * 60)
    
    # Create a single Snowflake connection to be shared across all demos
    print("Initializing Snowflake connection...")
    conn = snowflake_conn()
    
    try:
        # Demo individual models
        demo_model_1(conn)
        demo_model_2(conn)
        
        # Demo complete pipeline
        demo_complete_pipeline(conn)
        
        # Demo individual tools
        demo_individual_tools(conn)
        
    except Exception as e:
        print(f"Error during demonstration: {str(e)}")
        print("Make sure you have:")
        print("1. Set up your .env file with AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT")
        print("2. Configured your Snowflake connection in snowflake_conn.py")
        print("3. Installed required dependencies: openai, python-dotenv, snowflake-connector-python")
    finally:
        # Always close the connection when done
        print("Closing Snowflake connection...")
        conn.close()

if __name__ == "__main__":
    main() 