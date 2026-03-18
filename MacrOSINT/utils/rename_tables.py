"""
HDF5 Table Renaming Utility

Script to rename tables in the HDF5 stores, specifically to fix naming inconsistencies
like "soybean" -> "soybeans".
"""

from MacrOSINT.data.data_tables import ESRTableClient, FASTable, NASSTable
import os


def rename_soybean_tables():
    """Rename soybean tables to soybeans in all relevant stores."""
    
    print("🔧 HDF5 Table Renaming Utility")
    print("=" * 50)
    
    # Initialize different table clients to access their stores
    clients = {
        "ESR": ESRTableClient(),
        "FAS": FASTable(),
        "NASS": NASSTable(
            source_desc="NASS", 
            sector_desc="CROPS", 
            group_desc="FIELD CROPS", 
            commodity_desc="SOYBEANS",
            prefix="soybeans"
        )
    }
    
    results = {}
    
    for client_name, client in clients.items():
        print(f"\n📊 Processing {client_name} tables...")
        print(f"Store location: {client.table_db}")
        
        if not os.path.exists(client.table_db):
            print(f"⚠️ Store file doesn't exist: {client.table_db}")
            continue
        
        try:
            # First, list all tables to see what exists
            print(f"\n📋 Current tables in {client_name} store:")
            all_tables = client.list_all_tables()
            
            # Check for soybean tables
            soybean_tables = [t for t in all_tables if 'soybean' in t.lower() and 'soybeans' not in t.lower()]
            
            if not soybean_tables:
                print(f"✓ No 'soybean' tables found in {client_name} store")
                continue
            
            print(f"\n🔍 Found {len(soybean_tables)} tables with 'soybean' pattern:")
            for table in soybean_tables:
                print(f"  {table}")
            
            # Perform dry run first
            print(f"\n🧪 Dry run - showing planned renames for {client_name}:")
            dry_run_results = client.rename_table_keys('soybean', 'soybeans', dry_run=True)
            
            if dry_run_results:
                # Ask for confirmation
                print(f"\n❓ Proceed with renaming {len(dry_run_results)} tables in {client_name}? (y/N): ")
                
                # For automated execution, you can set this to True
                # or modify to read from command line arguments
                proceed = True  # Change to False for interactive mode
                
                if proceed:
                    print(f"🚀 Executing renames for {client_name}...")
                    actual_results = client.rename_table_keys('soybean', 'soybeans', dry_run=False)
                    results[client_name] = actual_results
                    print(f"✅ Completed {client_name} renames")
                else:
                    print(f"⏭️ Skipped {client_name} renames")
            
        except Exception as e:
            print(f"❌ Error processing {client_name}: {e}")
            results[client_name] = f"Error: {e}"
    
    # Summary
    print(f"\n📈 SUMMARY")
    print("=" * 50)
    for client_name, result in results.items():
        if isinstance(result, dict):
            print(f"{client_name}: ✅ Renamed {len(result)} tables")
        else:
            print(f"{client_name}: ❌ {result}")
    
    return results


def verify_renames():
    """Verify that the renames were successful."""
    print("\n🔍 Verifying renames...")
    
    client = ESRTableClient()
    try:
        all_tables = client.list_all_tables()
        soybean_tables = [t for t in all_tables if 'soybean' in t.lower()]
        soybeans_tables = [t for t in all_tables if 'soybeans' in t.lower()]
        
        print(f"\nRemaining 'soybean' tables: {len(soybean_tables)}")
        for table in soybean_tables:
            print(f"  {table}")
            
        print(f"\nFound 'soybeans' tables: {len(soybeans_tables)}")
        for table in soybeans_tables:
            print(f"  {table}")
            
        if not soybean_tables and soybeans_tables:
            print("✅ Rename verification successful!")
        else:
            print("⚠️ Some renames may not have completed successfully")
            
    except Exception as e:
        print(f"❌ Error during verification: {e}")


if __name__ == "__main__":
    # Execute the rename operation
    results = rename_soybean_tables()
    
    # Verify the results
    verify_renames()
    
    print("\n🎉 Table renaming process completed!")