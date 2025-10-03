#!/usr/bin/env python3
"""
Script to delete specific records from Pinecone vector database
Usage: python3 delete_pinecone_record.py --face-id FACE_ID
"""

import os
import argparse
import logging
from pinecone import Pinecone

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def delete_pinecone_record(face_id: str, api_key: str):
    """Delete a specific record from Pinecone by face ID"""
    try:
        pc = Pinecone(api_key=api_key)
        index_name = "face-recognition-index"
        
        if index_name not in pc.list_indexes():
            logger.error(f"Pinecone index '{index_name}' does not exist!")
            return False

        index = pc.Index(index_name)
        
        # Check if the record exists
        try:
            # Query to see if the record exists
            query_result = index.query(
                vector=[0.0] * 512,  # Dummy vector for query
                filter={"face_id": face_id},
                top_k=1,
                include_metadata=True
            )
            
            if not query_result.matches:
                logger.warning(f"Record with face_id '{face_id}' not found in Pinecone")
                return False
                
            logger.info(f"Found record: {query_result.matches[0].metadata}")
            
        except Exception as e:
            logger.info(f"Could not query record (might not exist): {e}")
        
        # Delete the record
        index.delete(ids=[face_id])
        logger.info(f"✅ Successfully deleted record with face_id: {face_id}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error deleting record: {e}")
        return False

def list_all_records(api_key: str):
    """List all records in Pinecone for reference"""
    try:
        pc = Pinecone(api_key=api_key)
        index_name = "face-recognition-index"
        
        if index_name not in pc.list_indexes():
            logger.error(f"Pinecone index '{index_name}' does not exist!")
            return

        index = pc.Index(index_name)
        
        # Get index stats
        stats = index.describe_index_stats()
        logger.info(f"📊 Total vectors in index: {stats.total_vector_count}")
        
        # Query all records (this is a workaround since Pinecone doesn't have a direct "list all" method)
        # We'll use a dummy vector to get some records
        try:
            query_result = index.query(
                vector=[0.0] * 512,  # Dummy vector
                top_k=100,  # Get up to 100 records
                include_metadata=True
            )
            
            if query_result.matches:
                logger.info("📋 Current records in Pinecone:")
                for match in query_result.matches:
                    metadata = match.metadata
                    logger.info(f"  - ID: {match.id}")
                    logger.info(f"    Name: {metadata.get('name', 'Unknown')}")
                    logger.info(f"    Face ID: {metadata.get('face_id', 'Unknown')}")
                    logger.info(f"    Person Type: {metadata.get('person_type', 'Unknown')}")
                    logger.info(f"    Image Count: {metadata.get('image_count', 1)}")
                    logger.info("    ---")
            else:
                logger.info("📋 No records found in Pinecone")
                
        except Exception as e:
            logger.info(f"Could not query records: {e}")
            
    except Exception as e:
        logger.error(f"❌ Error listing records: {e}")

def main():
    parser = argparse.ArgumentParser(description='Delete records from Pinecone vector database')
    parser.add_argument('--face-id', help='Face ID to delete')
    parser.add_argument('--list', action='store_true', help='List all records')
    parser.add_argument('--api-key', help='Pinecone API key (or set PINECONE_API_KEY env var)')
    
    args = parser.parse_args()
    
    # Get API key
    api_key = args.api_key or os.getenv('PINECONE_API_KEY')
    if not api_key:
        logger.error("❌ Pinecone API key not provided! Set PINECONE_API_KEY environment variable or use --api-key")
        return
    
    if args.list:
        logger.info("🔍 Listing all records in Pinecone...")
        list_all_records(api_key)
    elif args.face_id:
        logger.info(f"🗑️ Deleting record with face_id: {args.face_id}")
        success = delete_pinecone_record(args.face_id, api_key)
        if success:
            logger.info("✅ Deletion completed successfully!")
        else:
            logger.error("❌ Deletion failed!")
    else:
        logger.error("❌ Please provide --face-id to delete or --list to list records")
        parser.print_help()

if __name__ == "__main__":
    main()
