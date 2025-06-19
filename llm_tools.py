"""
LLM Tool Functions for Text-to-SQL System

This module contains standalone tool functions that can be used by different agents
for database operations and metadata retrieval.
"""

def get_table_metadata(conn, table_name: str = "H1B_clean") -> str:
    """
    Tool function to get metadata of a Snowflake table.
    
    Args:
        conn: Database connection object
        table_name (str): Name of the table to get metadata for
        
    Returns:
        str: Formatted table schema information
    """
    try:
        conn.execute('USE DATABASE parsed')
        conn.execute(f"""
            SELECT column_name, data_type, is_nullable, column_default
            FROM parsed.information_schema.columns 
            WHERE table_name = '{table_name.upper()}'
            ORDER BY ordinal_position
        """)
        schema_data = conn.fetch()
        
        if schema_data:
            schema_str = f"Table: parsed.combined.{table_name.lower()}\nColumns:\n"
            for row in schema_data:
                nullable = "NULL" if row[2] == "YES" else "NOT NULL"
                default = f", DEFAULT: {row[3]}" if row[3] else ""
                schema_str += f"- {row[0]} ({row[1]}, {nullable}{default})\n"
            return schema_str
        else:
            return f"No schema information found for table {table_name}"
    except Exception as e:
        return f"Error retrieving schema: {str(e)}"


def execute_sql(conn, sql_query: str) -> str:
    """
    Tool function to execute SQL query and fetch results.
    
    Args:
        conn: Database connection object
        sql_query (str): SQL query to execute
        
    Returns:
        str: Formatted query results
    """
    try:
        conn.execute(sql_query)
        results = conn.fetch()
        
        if results:
            # Format results as a readable string
            if len(results) == 1 and len(results[0]) == 1:
                # Single value result
                return str(results[0][0])
            else:
                # Multiple rows/columns
                formatted_results = []
                for i, row in enumerate(results):
                    if i < 10:  # Limit to first 10 rows for readability
                        formatted_results.append(str(row))
                    else:
                        formatted_results.append(f"... and {len(results) - 10} more rows")
                        break
                return "\n".join(formatted_results)
        else:
            return "No results returned from the query"
            
    except Exception as e:
        return f"Error executing SQL: {str(e)}"


class LLMToolRegistry:
    """
    Registry class to manage and provide access to LLM tool functions.
    This class acts as a bridge between the agent classes and the tool functions.
    """
    
    def __init__(self, conn):
        """
        Initialize the tool registry with a database connection.
        
        Args:
            conn: Database connection object
        """
        self.conn = conn
    
    def get_table_metadata_tool(self, table_name: str = "H1B_clean") -> str:
        """
        Wrapper for get_table_metadata tool function.
        
        Args:
            table_name (str): Name of the table to get metadata for
            
        Returns:
            str: Formatted table schema information
        """
        return get_table_metadata(self.conn, table_name)
    
    def execute_sql_tool(self, sql_query: str) -> str:
        """
        Wrapper for execute_sql tool function.
        
        Args:
            sql_query (str): SQL query to execute
            
        Returns:
            str: Formatted query results
        """
        return execute_sql(self.conn, sql_query) 