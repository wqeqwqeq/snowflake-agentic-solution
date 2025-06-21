from typing import Optional, Any, List, Tuple, Dict
from snowflake.connector import SnowflakeConnection
from snowflake.connector.cursor import SnowflakeCursor
from dotenv import load_dotenv
import os
import pandas as pd
load_dotenv()

class snowflake_conn(SnowflakeConnection):
    """
    A Snowflake database connection wrapper class that extends SnowflakeConnection.
    
    This class provides a simplified interface for connecting to and interacting with
    a Snowflake database. It automatically configures the connection with predefined
    credentials and provides convenient methods for executing queries and fetching results.
    
    Attributes:
        cur (SnowflakeCursor): The database cursor for executing queries.
        
    Connection Details:
        - User: STANLEYSNOWFLAKE
        - Account: HTQEADI-ZNA89116
        - Database: parsed
        - Schema: combined
        
    Example:
        >>> conn = snowflake_conn()
        >>> conn.execute("SELECT * FROM my_table LIMIT 10")
        >>> conn.fetch()
        >>> conn.close()
    """
    
    def __init__(self, **kwargs: Any) -> None:
        """
        Initialize the Snowflake connection with predefined configuration.
        
        Creates a connection to Snowflake using hardcoded credentials and database settings.
        Additional keyword arguments can be passed to override default connection parameters.
        
        Args:
            **kwargs: Additional keyword arguments to pass to the parent SnowflakeConnection.
                     These will override any default connection parameters.
                     
        Raises:
            snowflake.connector.errors.Error: If connection to Snowflake fails.
            
        Note:
            The connection uses hardcoded credentials. In production, consider using
            environment variables or secure credential management.
        """
        super().__init__(
            user="STANLEYSNOWFLAKE",
            account="HTQEADI-ZNA89116",
            password=os.getenv("SNOWFLAKE_PASSWORD"),
            database="parsed",
            schema="combined",
            **kwargs
        )

        self.cur: SnowflakeCursor = self.cursor()

    def execute(self, query: str) -> None:
        """
        Execute a SQL query on the Snowflake database.
        
        This method executes the provided SQL query and prints the query to stdout
        for debugging and logging purposes. The results remain in the cursor and
        can be retrieved using the fetch() method.
        
        Args:
            query (str): The SQL query string to execute. Can be any valid SQL statement
                        including SELECT, INSERT, UPDATE, DELETE, CREATE, etc.
                        
        Raises:
            snowflake.connector.errors.ProgrammingError: If the SQL query is invalid.
            snowflake.connector.errors.DatabaseError: If there's a database-related error.
            
        Example:
            >>> conn.execute("SELECT COUNT(*) FROM users WHERE active = true")
            >>> conn.execute("CREATE TABLE temp_table AS SELECT * FROM source_table")
        """
        self.cur.execute(query)

    def fetch(self) -> None:
        """
        Fetch and print all results from the last executed query.
        
        This method retrieves all rows from the result set of the most recently
        executed query and prints them to stdout. The results are fetched as a
        list of tuples, where each tuple represents a row.
        
        Note:
            This method prints results directly and doesn't return them. For programmatic
            access to results, consider using self.cur.fetchall() directly.
            
        Raises:
            snowflake.connector.errors.Error: If there's an error fetching results.
            
        Example:
            >>> conn.execute("SELECT name, age FROM users LIMIT 5")
            >>> conn.fetch()  # Prints: [('John', 25), ('Jane', 30), ...]
        """
        results = self.cur.fetchall()
        return results
    
    def query_to_pandas(self, query: str, nrows: int = 10) -> pd.DataFrame:
        """
        Execute a SQL query and return the results as a pandas DataFrame.
        
        This method executes the provided SQL query and returns the results as a pandas DataFrame.
        Args:
            query (str): The SQL query string to execute.
            nrows (int): The number of rows to fetch from the result set.
        Returns:
            pd.DataFrame: A pandas DataFrame containing the query results.
        """
        self.execute(query)
        result = self.cur.fetchmany(nrows) 
        result2 = [list(i) for i in result]
        cols = [i[0] for i in self.cur.description]
        return pd.DataFrame(result2, columns=cols)

    def close(self) -> None:
        """
        Close the Snowflake database connection.
        
        This method properly closes the database connection and releases any
        associated resources. It's important to call this method when finished
        with the database operations to prevent connection leaks.
        
        Note:
            After calling this method, the connection object should not be used
            for further database operations.
            
        Example:
            >>> conn = snowflake_conn()
            >>> # ... perform database operations ...
            >>> conn.close()  # Always close when done
        """
        super().close()

    def get_metadata(self, db_schema_tbl: str) -> List[Dict]:
        """
        Get comprehensive metadata for a given table including column info and distinct values.
        
        This method analyzes a table's structure and data distribution by:
        1. Getting column names and data types from information schema
        2. For text columns, calculating distinct counts
        3. For low-cardinality columns (distinct < 10), retrieving actual distinct values
        
        Args:
            db_schema_tbl (str): Full table reference in format database.schema.table
            
        Returns:
            List[Dict]: List of dictionaries containing metadata for each column:
                - table_name: Full table name
                - column_name: Column name
                - data_type: Column data type
                - distinct_count: Number of distinct values (for text columns)
                - distinct_values: List of distinct values (if distinct_count < 10)
                
        Example:
            >>> conn.get_table_metadata("mydb.myschema.mytable")
            [{'table_name': 'MYDB.MYSCHEMA.MYTABLE', 'column_name': 'STATUS', 
              'data_type': 'TEXT', 'distinct_count': 2, 'distinct_values': ['yes', 'no']}]
        """
        # Step 1: Split database.schema.table format
        parts = db_schema_tbl.split('.')
        if len(parts) != 3:
            raise ValueError(f"Invalid table reference format. Expected database.schema.table, got: {db_schema_tbl}")
        
        database, schema, table = parts
        full_table_name = f"{database.upper()}.{schema.upper()}.{table.upper()}"
        
        # Step 2: Get column names and data types
        schema_query = f"""
        SELECT column_name, data_type,comment
        FROM {database.upper()}.information_schema.columns 
        WHERE table_schema = '{schema.upper()}' 
        AND table_name = '{table.upper()}'
        ORDER BY ordinal_position
        """
        
        self.execute(schema_query)
        columns_info = self.fetch()
        
        if not columns_info:
            return []
        
        # Create base metadata structure
        metadata = []
        text_columns = []
        
        for col_name, data_type, comment in columns_info:
            col_metadata = {
                'table_name': full_table_name.lower(),
                'column_name': col_name,
                'data_type': data_type,
                'comment': comment if comment is not None else "",
                'usage':""
            }
            metadata.append(col_metadata)
            
            # Track text columns for distinct value analysis
            if 'TEXT' in data_type.upper() or 'VARCHAR' in data_type.upper() or 'CHAR' in data_type.upper():
                text_columns.append(col_name)
        
        # Step 3: Get distinct counts for text columns
        if text_columns:
            distinct_select_parts = [f"COUNT(DISTINCT {col}) as cnt_distinct_{col}" for col in text_columns]
            distinct_query = f"""
            SELECT {', '.join(distinct_select_parts)}
            FROM {full_table_name}
            """
            
            self.execute(distinct_query)
            distinct_counts = self.fetch()[0]  # Single row result
            
            # Map distinct counts back to metadata
            text_col_index = 0
            for i, col_metadata in enumerate(metadata):
                if col_metadata['column_name'] in text_columns:
                    distinct_count = distinct_counts[text_col_index]
                    col_metadata['distinct_count'] = distinct_count
                    text_col_index += 1
        
        # Step 4: Get actual distinct values for low-cardinality columns
        for col_metadata in metadata:
            if 'distinct_count' in col_metadata and col_metadata['distinct_count'] < 10:
                col_name = col_metadata['column_name']
                distinct_values_query = f"""
                SELECT DISTINCT {col_name}
                FROM {full_table_name}
                WHERE {col_name} IS NOT NULL
                ORDER BY {col_name}
                """
                
                self.execute(distinct_values_query)
                distinct_values_result = self.fetch()
                distinct_values = [row[0] for row in distinct_values_result]
                col_metadata['distinct_values'] = distinct_values
        
        return metadata




