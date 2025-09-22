from src.db.qdrant_db import QdrantChunksDB

if __name__ == "__main__":
    qdrant_db = QdrantChunksDB(url="http://localhost:1237")

    # Step 1: List collections
    # First you need to list the collections in order to know which collection to back up
    print("Step 1: List collections")
    list_collection = qdrant_db.list_collections()
    print(f"Collections: {list_collection}")

    # Step 2: List snapshots for each collection
    # Then you can list the snapshots (if exist) for each collection
    if list_collection['status'] == 'success':
        for collection in list_collection['collections']:
            snapshots = qdrant_db.list_snapshots(collection)
            print(f"Snapshots for {collection}: {snapshots}")
    else:
        print("Failed to list collections.")

    # Step 3: Create a snapshot for each collection
    # Create backup for each collection
    if list_collection['status'] == 'success':
        for collection in list_collection['collections']:
            backup = qdrant_db.backup_collection(collection, "backups")
            print(f"Backup result for {collection}: {backup}")
    else:
        print("Failed to list collections.")

    # Step 4: Restore from the backup file
    # You can restore from a specific backup file by providing the path to the backup file
    # You can get the path in folder "backups"
    qdrant_db.restore_collection(backup_path= "./backups/your_backup_collection.tar", new_collection_name= "restore_collection_name")