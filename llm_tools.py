import pandas as pd 

def get_table_metadata(conn, db_schema_tbl: str) -> str:
    """
    Tool function to get metadata of a Snowflake table.
    
    Args:
        conn: Database connection object
        db_schema_tbl (str): Full table reference in format database.schema.table
        
    Returns:
        str: Formatted table schema information
    """
    try:
        # Parse database.schema.table format
        parts = db_schema_tbl.split('.')
        if len(parts) != 3:
            return f"Invalid table reference format. Expected database.schema.table, got: {db_schema_tbl}"
        
        database, schema, table = parts
        
        # Use the specified database
        conn.execute(f'USE DATABASE {database}')
        conn.execute(f"""
            SELECT column_name, data_type, is_nullable, column_default
            FROM {database}.information_schema.columns 
            WHERE table_schema = '{schema.upper()}' 
            AND table_name = '{table.upper()}'
            ORDER BY ordinal_position
        """)
        schema_data = conn.fetch()
        
        if schema_data:
            schema_str = f"Table: {database}.{schema}.{table}\nColumns:\n"
            for row in schema_data:
                nullable = "NULL" if row[2] == "YES" else "NOT NULL"
                default = f", DEFAULT: {row[3]}" if row[3] else ""
                schema_str += f"- {row[0]} ({row[1]}, {nullable}{default})\n"
            return schema_str
        else:
            return f"No schema information found for table {db_schema_tbl}"
    except Exception as e:
        return f"Error retrieving schema: {str(e)}"


def execute_sql(conn, sql_query: str, nrows: int = 10) -> tuple:
    """
    Tool function to execute SQL query and fetch results.
    
    Args:
        conn: Database connection object
        sql_query (str): SQL query to execute
        nrows (int): Number of rows to fetch (default: 10)
        
    Returns:
        tuple: (pandas.DataFrame, str) - DataFrame and string representation of results
    """
    try:
        # Use the query_to_pandas method from the connection object
        df = conn.query_to_pandas(sql_query, nrows=nrows)
        
        if not df.empty:
            # Convert DataFrame to string representation
            df_string = df.to_string(index=False)
            return df, df_string
        else:
            # Return empty DataFrame and message
            return df, "No results returned from the query"
            
    except Exception as e:
        # Return error as single-row DataFrame
        error_df = pd.DataFrame({"Error": [str(e)]})
        error_string = f"Error executing SQL: {str(e)}"
        return error_df, error_string


def read_table_descriptions(file_path: str) -> str:
    """
    Tool function to read table descriptions from a markdown file.
    
    Args:
        file_path (str): Path to the table description file
        
    Returns:
        str: Content of the table description file
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        if content.strip():
            return content
        else:
            return "Table description file is empty"
            
    except FileNotFoundError:
        return f"Table description file not found: {file_path}"
    except Exception as e:
        return f"Error reading table description file: {str(e)}" 