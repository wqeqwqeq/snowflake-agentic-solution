# %%
import json
import openai
from typing import List, Dict, Any, Optional
from snowflake_conn import snowflake_conn
import os
from dotenv import load_dotenv
import numpy as np
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SearchField,
    SearchFieldDataType,
    SimpleField,
    SearchableField,
    ComplexField,
    VectorSearch,
    HnswAlgorithmConfiguration,
    VectorSearchProfile,
    SemanticConfiguration,
    SemanticSearch,
    SemanticPrioritizedFields,
    SemanticField
)
from azure.core.credentials import AzureKeyCredential

load_dotenv()

# Set up OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

def write_schema_in_json(db_schema_tbl: str, output_file: str = "schema_metadata.json") -> None:
    """
    Get metadata from Snowflake table and write it to a JSON file.
    
    This function connects to Snowflake, retrieves table metadata using the get_metadata method,
    and saves the resulting list of dictionaries as a JSON file.
    
    Args:
        db_schema_tbl (str): Full table reference in format database.schema.table
        output_file (str): Path to the output JSON file. Defaults to "schema_metadata.json"
        
    Example:
        >>> write_schema_in_json("mydb.myschema.mytable", "my_schema.json")
    """
    # Create Snowflake connection
    conn = snowflake_conn()
    
    try:
        # Get metadata using the get_metadata method
        metadata = conn.get_metadata(db_schema_tbl)
        
        # Write metadata to JSON file
        with open(output_file, 'w', encoding='utf-8') as file:
            json.dump(metadata, file, indent=2, ensure_ascii=False)
        
        print(f"Schema metadata successfully written to {output_file}")
        print(f"Total columns processed: {len(metadata)}")
        
    except Exception as e:
        print(f"Error writing schema to JSON: {str(e)}")
        raise
    finally:
        # Always close the connection
        conn.close()

def read_json_schema(json_file: str = "schema_metadata.json") -> List[Dict]:
    """
    Read schema metadata from a JSON file and return as list of dictionaries.
    
    Args:
        json_file (str): Path to the JSON file containing schema metadata.
                        Defaults to "schema_metadata.json"
        
    Returns:
        List[Dict]: List of dictionaries containing table schema metadata
        
    Raises:
        FileNotFoundError: If the JSON file doesn't exist
        json.JSONDecodeError: If there's an error parsing the JSON file
        
    Example:
        >>> schema_data = read_json_schema("my_schema.json")
        >>> print(f"Loaded {len(schema_data)} columns")
    """
    try:
        with open(json_file, 'r', encoding='utf-8') as file:
            metadata = json.load(file)
        
        if not isinstance(metadata, list):
            raise ValueError("JSON file should contain a list of dictionaries")
        
        print(f"Successfully loaded schema metadata from {json_file}")
        print(f"Total columns loaded: {len(metadata)}")
        
        return metadata
        
    except FileNotFoundError:
        print(f"Error: JSON file '{json_file}' not found")
        raise
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON file: {str(e)}")
        raise
    except Exception as e:
        print(f"Unexpected error reading JSON file: {str(e)}")
        raise

def embed_schema_comments(schema_metadata: List[Dict], model: str = "text-embedding-3-small") -> List[Dict]:
    """
    Embed the comment field of schema metadata using OpenAI's embedding model.
    
    This function takes a list of schema metadata dictionaries and adds embedding vectors
    for the comment field of each column using OpenAI's text embedding API.
    
    Args:
        schema_metadata (List[Dict]): List of dictionaries containing schema metadata
        model (str): OpenAI embedding model to use. Defaults to "text-embedding-3-small"
        
    Returns:
        List[Dict]: Updated schema metadata with 'comment_embedding' field added to each dictionary
        
    Raises:
        openai.OpenAIError: If there's an error with the OpenAI API
        
            Example:
        >>> schema_data = read_json_schema("my_schema.json")
        >>> embedded_schema = embed_schema_comments(schema_data)
        >>> print(f"Generated embeddings for {len(embedded_schema)} columns")
    """
    if not openai.api_key:
        raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY environment variable.")
    
    embedded_metadata = []
    successful_embeddings = 0
    
    for column_data in schema_metadata:
        # Create a copy of the original data
        embedded_column = column_data.copy()
        
        # Get the comment text
        comment_text = column_data.get('comment', '')
        
        # Skip empty comments
        if not comment_text or comment_text.strip() == '':
            embedded_column['comment_embedding'] = None
            embedded_column['embedding_status'] = 'skipped_empty_comment'
        else:
            try:
                # Create embedding using OpenAI API
                response = openai.embeddings.create(
                    model=model,
                    input=comment_text
                )
                
                # Extract the embedding vector
                embedding_vector = response.data[0].embedding
                
                # Add embedding to the metadata
                embedded_column['comment_embedding'] = embedding_vector

                successful_embeddings += 1
                
            except Exception as e:
                print(f"Error embedding comment for column '{column_data.get('column_name', 'unknown')}': {str(e)}")
                embedded_column['comment_embedding'] = None
        
        embedded_metadata.append(embedded_column)
    
    print(f"Embedding complete: {successful_embeddings}/{len(schema_metadata)} comments successfully embedded")
    
    return embedded_metadata

def save_embedded_schema(embedded_metadata: List[Dict], output_file: str = "embedded_schema.json") -> None:
    """
    Save embedded schema metadata to a JSON file.
    
    Args:
        embedded_metadata (List[Dict]): Schema metadata with embeddings
        output_file (str): Output JSON file path
    """
    try:
        with open(output_file, 'w', encoding='utf-8') as file:
            json.dump(embedded_metadata, file, indent=2, ensure_ascii=False)
        
        print(f"Embedded schema metadata saved to {output_file}")
        
    except Exception as e:
        print(f"Error saving embedded schema: {str(e)}")
        raise



def prepare_search_index_data(json_file: str = "h1b_meta_embed.json") -> List[Dict[str, Any]]:
    """
    Read the JSON schema file and prepare data for Azure AI Search indexing.
    
    This function reads the schema metadata and converts each column entry into
    a search document with appropriate fields for Azure AI Search.
    
    Args:
        json_file (str): Path to the JSON file containing schema metadata.
                        Defaults to "h1b_meta_embed.json"
        
    Returns:
        List[Dict[str, Any]]: List of documents ready for Azure AI Search indexing
        
    Example:
        >>> search_docs = prepare_search_index_data("h1b_meta_embed.json")
        >>> print(f"Prepared {len(search_docs)} documents for indexing")
    """
    # Read the schema metadata
    schema_metadata = read_json_schema(json_file)
    
    search_documents = []
    
    for idx, column_data in enumerate(schema_metadata):
        # Create a unique document ID
        doc_id = f"{column_data.get('table_name', 'unknown').replace('.', '_')}_{column_data.get('column_name', 'unknown')}_{idx}"
        
        # Prepare search document
        search_doc = {
            "id": doc_id,
            "table_name": column_data.get('table_name', ''),
            "column_name": column_data.get('column_name', ''),
            "comment": column_data.get('comment', ''),
            "data_type": column_data.get('data_type', ''),
            "distinct_count": column_data.get('distinct_count', 0) if column_data.get('distinct_count') is not None else 0,
            "distinct_values_text": ', '.join(str(v) for v in column_data.get('distinct_values', [])) if column_data.get('distinct_values') else '',
        }
        
        # Add embedding vector if available (for vector search)
        if column_data.get('comment_embedding'):
            search_doc["comment_embedding"] = column_data['comment_embedding']
        
        search_documents.append(search_doc)
    
    print(f"Prepared {len(search_documents)} documents for Azure AI Search indexing")
    return search_documents

def create_azure_search_index(index_name: str = "column-metadata-index") -> SearchIndex:
    """
    Create the Azure AI Search index definition for schema metadata with enhanced vector and semantic search.
    
    Args:
        index_name (str): Name of the search index. Defaults to "column-metadata-index"
        
    Returns:
        SearchIndex: The search index definition with vector and semantic search capabilities
    """
    fields = [
        SimpleField(
            name="id", 
            type=SearchFieldDataType.String, 
            key=True,
            sortable=True,
            filterable=True,
            facetable=True
        ),
        SearchableField(name="table_name", type=SearchFieldDataType.String, analyzer_name="standard.lucene"),
        SearchableField(name="column_name", type=SearchFieldDataType.String, analyzer_name="standard.lucene"),
        SearchableField(name="comment", type=SearchFieldDataType.String, analyzer_name="standard.lucene"),
        SearchableField(name="distinct_values_text", type=SearchFieldDataType.String, analyzer_name="standard.lucene"),
        SimpleField(name="data_type", type=SearchFieldDataType.String, filterable=True, facetable=True),
        SimpleField(name="distinct_count", type=SearchFieldDataType.Int32, filterable=True, facetable=True, sortable=True),
        SearchField(
            name="comment_embedding", 
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=1536,  # text-embedding-3-small dimensions
            vector_search_profile_name="schemaHnswProfile"
        )
    ]
    
    # Configure the vector search with HNSW algorithm for better performance
    vector_search = VectorSearch(
        algorithms=[
            HnswAlgorithmConfiguration(
                name="schemaHnsw"
            )
        ],
        profiles=[
            VectorSearchProfile(
                name="schemaHnswProfile",
                algorithm_configuration_name="schemaHnsw",
            )
        ]
    )
    
    # Configure semantic search for natural language queries
    semantic_config = SemanticConfiguration(
        name="schema-semantic-config",
        prioritized_fields=SemanticPrioritizedFields(
            title_field=SemanticField(field_name="column_name"),
            content_fields=[
                SemanticField(field_name="comment"),
                SemanticField(field_name="table_name"),
                SemanticField(field_name="distinct_values_text")
            ]
        )
    )
    
    # Create the semantic search settings
    semantic_search = SemanticSearch(configurations=[semantic_config])
    
    # Create the search index with vector and semantic search capabilities
    index = SearchIndex(
        name=index_name, 
        fields=fields,
        vector_search=vector_search, 
        semantic_search=semantic_search
    )
    return index

def upload_to_azure_search(
    search_documents: List[Dict[str, Any]], 
    index_name: str = "column-metadata-index",
    search_endpoint: str = None,
    search_key: str = None
) -> bool:
    """
    Upload documents to Azure AI Search.
    
    This function creates the Azure AI Search client, creates or updates the search index,
    and uploads the provided documents.
    
    Args:
        search_documents (List[Dict[str, Any]]): List of documents to upload
        index_name (str): Name of the search index. Defaults to "column-metadata-index"
        search_endpoint (str): Azure Search service endpoint. If None, reads from AZURE_SEARCH_ENDPOINT env var
        search_key (str): Azure Search admin key. If None, reads from AZURE_SEARCH_KEY env var
        
    Returns:
        bool: True if upload successful, False otherwise
        
    Raises:
        ValueError: If Azure Search credentials are not provided
        
    Example:
        >>> docs = prepare_search_index_data("h1b_meta_embed.json")
        >>> success = upload_to_azure_search(docs, "my-schema-index")
        >>> print(f"Upload successful: {success}")
    """
    # Get Azure Search credentials
    endpoint = search_endpoint or os.getenv("AZURE_SEARCH_ENDPOINT")
    key = search_key or os.getenv("AZURE_SEARCH_KEY")
    
    if not endpoint or not key:
        raise ValueError(
            "Azure Search credentials not found. Please provide search_endpoint and search_key "
            "or set AZURE_SEARCH_ENDPOINT and AZURE_SEARCH_KEY environment variables."
        )
    
    try:
        # Create credentials
        credential = AzureKeyCredential(key)
        
        # Create index client
        index_client = SearchIndexClient(endpoint=endpoint, credential=credential)
        
        print(f"Creating or updating search index: {index_name}")
        
        # Create the search index
        index = create_azure_search_index(index_name)
        
        # Create or update the index
        result = index_client.create_or_update_index(index)
        print(f"Index '{result.name}' created/updated successfully")
        
        # Create search client for uploading documents
        search_client = SearchClient(endpoint=endpoint, index_name=index_name, credential=credential)
        
        print(f"Uploading {len(search_documents)} documents to Azure AI Search...")
        
        # Upload documents in batches
        batch_size = 100  # Azure Search recommends batches of 100 or fewer
        for i in range(0, len(search_documents), batch_size):
            batch = search_documents[i:i + batch_size]
            
            # Filter out documents with embedding vectors that are None
            valid_batch = []
            for doc in batch:
                if 'comment_embedding' in doc and doc['comment_embedding'] is None:
                    # Remove the embedding field if it's None
                    doc_copy = doc.copy()
                    del doc_copy['comment_embedding']
                    valid_batch.append(doc_copy)
                else:
                    valid_batch.append(doc)
            
            # Upload batch
            result = search_client.upload_documents(documents=valid_batch)
            
            successful_uploads = sum(1 for r in result if r.succeeded)
            failed_uploads = len(valid_batch) - successful_uploads
            
            print(f"Batch {i//batch_size + 1}: {successful_uploads} successful, {failed_uploads} failed")
            
            if failed_uploads > 0:
                for r in result:
                    if not r.succeeded:
                        print(f"  Failed to upload document {r.key}: {r.error_message}")
        
        print(f"Document upload completed for index: {index_name}")
        return True
        
    except Exception as e:
        print(f"Error uploading to Azure Search: {str(e)}")
        return False

def full_indexing_pipeline(
    json_file: str = "h1b_meta_embed.json",
    index_name: str = "column-metadata-index",
    search_endpoint: str = None,
    search_key: str = None
) -> bool:
    """
    Complete pipeline to prepare data and upload to Azure AI Search.
    
    This function combines data preparation and upload in a single operation.
    
    Args:
        json_file (str): Path to the JSON schema file
        index_name (str): Name of the Azure Search index
        search_endpoint (str): Azure Search endpoint (optional, uses env var if None)
        search_key (str): Azure Search key (optional, uses env var if None)
        
    Returns:
        bool: True if the entire pipeline completed successfully
        
    Example:
        >>> success = full_indexing_pipeline("h1b_meta_embed.json", "my-index")
        >>> print(f"Indexing pipeline completed: {success}")
    """
    try:
        print("Starting Azure AI Search indexing pipeline...")
        
        # Step 1: Prepare search documents
        print("Step 1: Preparing search documents...")
        search_docs = prepare_search_index_data(json_file)
        
        # Step 2: Upload to Azure Search
        print("Step 2: Uploading to Azure AI Search...")
        success = upload_to_azure_search(
            search_documents=search_docs,
            index_name=index_name,
            search_endpoint=search_endpoint,
            search_key=search_key
        )
        
        if success:
            print(f"✅ Pipeline completed successfully! Index '{index_name}' is ready for search.")
        else:
            print("❌ Pipeline failed during upload.")
            
        return success
        
    except Exception as e:
        print(f"❌ Pipeline failed with error: {str(e)}")
        return False

# Example usage and testing function
def main():
    """
    Example usage of the schema embedding functions.
    """
    try:
        # Example table - replace with your actual table reference
        table_reference = "parsed.combined.your_table_name"
        
        print("Step 1: Writing schema to JSON...")
        write_schema_in_json(table_reference, "example_schema.json")
        
        print("\nStep 2: Reading schema from JSON...")
        schema_data = read_json_schema("example_schema.json")
        
        print("\nStep 3: Embedding comments...")
        embedded_schema = embed_schema_comments(schema_data)
        
        print("\nStep 4: Saving embedded schema...")
        save_embedded_schema(embedded_schema, "example_embedded_schema.json")

            
    except Exception as e:
        print(f"Error in main execution: {str(e)}")


# %%
