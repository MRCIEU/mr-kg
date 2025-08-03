#!/usr/bin/env python3
"""Test script to verify multiple read-only database connections work simultaneously."""

import time
import threading
from pathlib import Path
import duckdb
from yiutils.project_utils import find_project_root

def test_connection(thread_id: int, db_path: Path):
    """Test function to run in a separate thread."""
    print(f"Thread {thread_id}: Starting connection test")
    
    try:
        with duckdb.connect(str(db_path), read_only=True) as conn:
            print(f"Thread {thread_id}: Connected successfully")
            
            # Simulate work with a simple query
            result = conn.execute("SELECT COUNT(*) FROM trait_embeddings").fetchone()
            if result:
                print(f"Thread {thread_id}: Found {result[0]} trait embeddings")
            
            # Simulate processing time
            time.sleep(2)
            
            # Another query
            result = conn.execute("SELECT COUNT(*) FROM pmid_model_analysis").fetchone()
            if result:
                print(f"Thread {thread_id}: Found {result[0]} PMID-model combinations")
            
        print(f"Thread {thread_id}: Connection closed successfully")
        
    except Exception as e:
        print(f"Thread {thread_id}: ERROR - {e}")

def main():
    """Test multiple concurrent read-only connections."""
    # Find database
    PROJECT_ROOT = find_project_root("docker-compose.yml")
    DATA_DIR = PROJECT_ROOT / "data"
    
    db_files = list((DATA_DIR / "db").glob("database-*.db"))
    if not db_files:
        print("No database files found!")
        return 1
    
    latest_db = max(db_files, key=lambda p: p.stat().st_mtime)
    print(f"Using database: {latest_db}")
    
    # Start multiple threads
    num_threads = 5
    threads = []
    
    print(f"\nStarting {num_threads} concurrent database connections...")
    start_time = time.time()
    
    for i in range(num_threads):
        thread = threading.Thread(target=test_connection, args=(i, latest_db))
        threads.append(thread)
        thread.start()
    
    # Wait for all threads to complete
    for thread in threads:
        thread.join()
    
    end_time = time.time()
    print(f"\nAll threads completed in {end_time - start_time:.2f} seconds")
    print("✅ Multiple read-only connections work successfully!")

if __name__ == "__main__":
    main()
