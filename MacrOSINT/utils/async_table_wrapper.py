"""
Async Table Client Wrapper

This module provides async wrappers for the existing TableClient, EIATable, NASSTable, 
and ESRTableClient classes to leverage the AsyncFileManager for improved performance.
"""

import asyncio
from typing import Dict, List, Optional, Any, Callable
import pandas as pd

from MacrOSINT.utils.async_file_manager import AsyncFileManager
from MacrOSINT.data.data_tables import TableClient, EIATable, NASSTable, ESRTableClient


class AsyncTableMixin:
    """Mixin to add async file operations to existing table clients"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._async_manager: Optional[AsyncFileManager] = None
        self._auto_shutdown_manager = True
    
    async def _get_async_manager(self) -> AsyncFileManager:
        """Get or create the async file manager"""
        if self._async_manager is None:
            self._async_manager = AsyncFileManager(
                max_concurrent_operations=5,
                max_workers=4,
                enable_logging=True
            )
        return self._async_manager
    
    async def _shutdown_async_manager(self):
        """Shutdown the async manager if auto-shutdown is enabled"""
        if self._async_manager and self._auto_shutdown_manager:
            await self._async_manager.shutdown()
            self._async_manager = None
    
    def set_async_manager(self, manager: AsyncFileManager, auto_shutdown: bool = False):
        """Set external async manager (prevents auto-shutdown)"""
        self._async_manager = manager
        self._auto_shutdown_manager = auto_shutdown
    
    async def async_update_table_data(self, 
                                    key: str, 
                                    data: pd.DataFrame, 
                                    use_prefix: bool = True, 
                                    metadata: Optional[Dict] = None,
                                    priority: int = 0,
                                    callback: Optional[Callable] = None) -> str:
        """
        Async version of update_table_data method
        
        Returns:
            operation_id for tracking the operation
        """
        manager = await self._get_async_manager()
        
        # Construct the full key
        if use_prefix and hasattr(self, 'prefix') and self.prefix:
            full_key = f"{self.prefix}/{key}"
        else:
            full_key = key
        
        # Queue the write operation
        operation_id = await manager.write_hdf(
            file_path=self.table_db,
            key=full_key,
            data=data,
            metadata=metadata,
            priority=priority,
            callback=callback
        )
        
        return operation_id
    
    async def async_batch_update_table_data(self, 
                                          updates: List[Dict[str, Any]], 
                                          execute_immediately: bool = True) -> Dict[str, Any]:
        """
        Batch update multiple table entries asynchronously
        
        Args:
            updates: List of dictionaries with keys: 'key', 'data', 'use_prefix', 'metadata', 'priority'
            execute_immediately: Whether to execute all operations immediately
        
        Returns:
            Dictionary with operation results
        """
        manager = await self._get_async_manager()
        
        # Prepare operations for batch execution
        operations = []
        for update in updates:
            key = update['key']
            data = update['data']
            use_prefix = update.get('use_prefix', True)
            metadata = update.get('metadata')
            priority = update.get('priority', 0)
            
            # Construct full key
            if use_prefix and hasattr(self, 'prefix') and self.prefix:
                full_key = f"{self.prefix}/{key}"
            else:
                full_key = key
            
            operations.append({
                'file_path': self.table_db,
                'key': full_key,
                'data': data,
                'metadata': metadata,
                'priority': priority
            })
        
        return await manager.batch_write_hdf(operations, execute_immediately)
    
    async def async_get_key(self, key: str, use_prefix: bool = True, 
                           callback: Optional[Callable] = None) -> str:
        """
        Async version of get_key method
        
        Returns:
            operation_id for tracking the operation
        """
        manager = await self._get_async_manager()
        
        # Construct the full key
        if use_prefix and hasattr(self, 'prefix') and self.prefix:
            full_key = f"{self.prefix}/{key}"
        else:
            full_key = key
        
        # Queue the read operation
        operation_id = await manager.read_hdf(
            file_path=self.table_db,
            key=full_key,
            callback=callback
        )
        
        return operation_id
    
    async def wait_for_operations(self, timeout: Optional[float] = None) -> bool:
        """Wait for all async operations to complete"""
        if self._async_manager:
            return await self._async_manager.wait_for_completion(timeout)
        return True
    
    def get_async_progress(self) -> Optional[Dict[str, Any]]:
        """Get progress information for async operations"""
        if self._async_manager:
            return self._async_manager.get_progress()
        return None


class AsyncTableClient(AsyncTableMixin, TableClient):
    """Async-enabled TableClient"""
    pass


class AsyncEIATable(AsyncTableMixin, EIATable):
    """Async-enabled EIATable with specialized energy data methods"""
    
    async def async_update_petroleum_stocks(self, 
                                          start: str = "2001-01", 
                                          end: str = "2025-08", 
                                          max_concurrent: int = 3,
                                          progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Async version of update_petroleum_stocks with improved concurrency control
        """
        # Use existing async implementation but with our unified manager
        manager = await self._get_async_manager()
        
        # Set up progress tracking callback
        def track_progress(result, operation):
            if progress_callback:
                progress = manager.get_progress()
                progress_callback(progress, operation)
        
        # Call the existing async method but ensure we use our manager
        original_manager = self._async_manager
        try:
            # Temporarily override the manager settings
            return await self.update_petroleum_stocks_async(
                start=start, 
                end=end, 
                max_concurrent=max_concurrent
            )
        finally:
            self._async_manager = original_manager
    
    async def async_bulk_update_natural_gas(self, 
                                          start: Optional[str] = None, 
                                          end: Optional[str] = None, 
                                          max_concurrent: int = 3,
                                          progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Async bulk update for natural gas data using unified file manager
        """
        manager = await self._get_async_manager()
        
        # Prepare operations for all natural gas updates
        operations = []
        update_methods = [
            ('storage', self.update_natural_gas_storage),
            ('production', self.update_natural_gas_production),
            ('trade', self.update_natural_gas_trade),
            ('consumption', self.update_consumption_by_state)
        ]
        
        # Queue all operations
        operation_ids = []
        for operation_name, method in update_methods:
            # Call method to get data but don't save yet
            try:
                # This is a simplified approach - you may need to adapt based on actual method signatures
                data = method(start, end) if start and end else method()
                if isinstance(data, pd.DataFrame) and not data.empty:
                    op_id = await self.async_update_table_data(
                        key=f"NG/{operation_name}",
                        data=data,
                        priority=1,
                        callback=progress_callback
                    )
                    operation_ids.append(op_id)
            except Exception as e:
                print(f"Error queuing {operation_name}: {e}")
        
        # Execute all operations
        return await manager.execute_queued_operations(max_concurrent)


class AsyncNASSTable(AsyncTableMixin, NASSTable):
    """Async-enabled NASSTable with USDA data methods"""
    
    async def async_api_update_multi_year(self, 
                                        short_desc: str,
                                        start_year: int, 
                                        end_year: int,
                                        max_concurrent: int = 3,
                                        progress_callback: Optional[Callable] = None,
                                        **kwargs) -> Dict[str, Any]:
        """
        Enhanced async multi-year update using unified file manager
        """
        manager = await self._get_async_manager()
        
        # Prepare data for batch processing
        years = list(range(start_year, end_year + 1))
        operations = []
        
        async def fetch_year_data(year: int) -> Optional[pd.DataFrame]:
            """Fetch data for a single year"""
            try:
                # Call the API for this year
                year_kwargs = {**kwargs, 'year': year}
                result = self.api_update(short_desc, **year_kwargs)
                return result
            except Exception as e:
                print(f"Error fetching data for year {year}: {e}")
                return None
        
        # Fetch data for all years concurrently
        year_data_tasks = [fetch_year_data(year) for year in years]
        year_results = await asyncio.gather(*year_data_tasks, return_exceptions=True)
        
        # Queue write operations for successful results
        successful_operations = 0
        for year, result in zip(years, year_results):
            if isinstance(result, pd.DataFrame) and not result.empty:
                key = f"{self.commodity}/{short_desc.replace(' ', '_').lower()}/{year}"
                op_id = await self.async_update_table_data(
                    key=key,
                    data=result,
                    priority=1,
                    callback=progress_callback
                )
                successful_operations += 1
        
        # Execute all write operations
        write_results = await manager.execute_queued_operations(max_concurrent)
        
        return {
            **write_results,
            'years_requested': len(years),
            'years_successful': successful_operations,
            'success_rate': successful_operations / len(years) if years else 0
        }


class AsyncESRTableClient(AsyncTableMixin, ESRTableClient):
    """Async-enabled ESRTableClient with export sales reporting methods"""
    
    async def async_esr_multi_year_update(self, 
                                        commodity: str,
                                        start_year: int, 
                                        end_year: int,
                                        max_concurrent: int = 3,
                                        progress_callback: Optional[Callable] = None,
                                        **kwargs) -> Dict[str, Any]:
        """
        Enhanced async ESR multi-year update using unified file manager
        """
        manager = await self._get_async_manager()
        
        years = list(range(start_year, end_year + 1))
        operations = []
        
        async def fetch_esr_year_data(year: int) -> Optional[pd.DataFrame]:
            """Fetch ESR data for a single year"""
            try:
                # Use existing ESR staging API
                if hasattr(self, 'staging_client'):
                    data = await self.staging_client.get_esr_data_async(
                        commodity=commodity,
                        year=year,
                        **kwargs
                    )
                    return data
                else:
                    print(f"No staging client available for year {year}")
                    return None
            except Exception as e:
                print(f"Error fetching ESR data for year {year}: {e}")
                return None
        
        # Fetch data for all years concurrently
        year_data_tasks = [fetch_esr_year_data(year) for year in years]
        year_results = await asyncio.gather(*year_data_tasks, return_exceptions=True)
        
        # Queue write operations for successful results
        successful_operations = 0

        for year, result in zip(years, year_results):
            if isinstance(result, pd.DataFrame) and not result.empty:
                key = f"ESR/{commodity}/{year}"
                op_id = await self.async_update_table_data(
                    key=key,
                    data=result,
                    metadata={'commodity': commodity, 'year': year, 'updated': pd.Timestamp.now()},
                    priority=1,
                    callback=progress_callback
                )
                successful_operations += 1
        
        # Execute all write operations
        write_results = await manager.execute_queued_operations(max_concurrent)
        
        return {
            **write_results,
            'commodity': commodity,
            'years_requested': len(years),
            'years_successful': successful_operations,
            'success_rate': successful_operations / len(years) if years else 0
        }


def async_table_factory(table_type: str, *args, **kwargs) -> AsyncTableMixin:
    """
    Factory function to create async-enabled table clients
    
    Args:
        table_type: Type of table ('base', 'eia', 'nass', 'esr')
        *args, **kwargs: Arguments for the specific table client
    
    Returns:
        Async-enabled table client
    """
    if table_type.lower() == 'base':
        return AsyncTableClient(*args, **kwargs)
    elif table_type.lower() == 'eia':
        return AsyncEIATable(*args, **kwargs)
    elif table_type.lower() == 'nass':
        return AsyncNASSTable(*args, **kwargs)
    elif table_type.lower() == 'esr':
        return AsyncESRTableClient(*args, **kwargs)
    else:
        raise ValueError(f"Unknown table type: {table_type}")


# Context manager for unified async operations across multiple clients
class AsyncTableSession:
    """
    Context manager for managing multiple async table clients with shared file manager
    """
    
    def __init__(self, max_concurrent_operations: int = 10, max_workers: int = 4):
        self.max_concurrent_operations = max_concurrent_operations
        self.max_workers = max_workers
        self.manager: Optional[AsyncFileManager] = None
        self.clients: List[AsyncTableMixin] = []
    
    async def __aenter__(self):
        self.manager = AsyncFileManager(
            max_concurrent_operations=self.max_concurrent_operations,
            max_workers=self.max_workers,
            enable_logging=True
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.manager:
            await self.manager.shutdown()
    
    def create_client(self, table_type: str, *args, **kwargs) -> AsyncTableMixin:
        """Create a new async table client that shares the session's file manager"""
        client = async_table_factory(table_type, *args, **kwargs)
        client.set_async_manager(self.manager, auto_shutdown=False)
        self.clients.append(client)
        return client
    
    async def execute_all_operations(self, max_concurrent: Optional[int] = None) -> Dict[str, Any]:
        """Execute all queued operations across all clients"""
        if self.manager:
            return await self.manager.execute_queued_operations(max_concurrent)
        return {"completed": 0, "failed": 0, "results": []}
    
    def get_session_progress(self) -> Optional[Dict[str, Any]]:
        """Get progress information for the entire session"""
        if self.manager:
            return self.manager.get_progress()
        return None
    
    async def wait_for_all_operations(self, timeout: Optional[float] = None) -> bool:
        """Wait for all operations across all clients to complete"""
        if self.manager:
            return await self.manager.wait_for_completion(timeout)
        return True


# Example usage functions
async def example_usage():
    """Example of how to use the async table clients"""
    
    print("Example 1: Single client with async operations")
    
    # Create an async EIA client
    async_eia = AsyncEIATable('PET', rename_key_cols=True)
    
    # Queue multiple operations
    operations = [
        {'key': 'supply/crude_stocks', 'data': pd.DataFrame({'test': [1, 2, 3]})},
        {'key': 'supply/refined_stocks', 'data': pd.DataFrame({'test': [4, 5, 6]})},
        {'key': 'production/refined_products', 'data': pd.DataFrame({'test': [7, 8, 9]})}
    ]
    
    result = await async_eia.async_batch_update_table_data(operations)
    print(f"Batch update result: {result['completed']} completed, {result['failed']} failed")
    
    # Wait for completion and cleanup
    await async_eia._shutdown_async_manager()
    
    print("\nExample 2: Multi-client session")
    
    # Use session context manager for multiple clients
    async with AsyncTableSession(max_concurrent_operations=8) as session:
        # Create multiple clients sharing the same file manager
        eia_client = session.create_client('eia', 'NG')
        nass_client = session.create_client('nass', 'CORN')
        esr_client = session.create_client('esr')
        
        # Queue operations on different clients
        await eia_client.async_update_table_data(
            'consumption/by_state', 
            pd.DataFrame({'state': ['TX', 'CA'], 'consumption': [100, 200]})
        )
        
        await nass_client.async_update_table_data(
            'acres_planted/2024', 
            pd.DataFrame({'state': ['IA', 'IL'], 'acres': [1000, 2000]})
        )
        
        await esr_client.async_update_table_data(
            'cattle/exports/2024', 
            pd.DataFrame({'country': ['Japan', 'Korea'], 'exports': [50, 75]})
        )
        
        # Execute all operations across all clients
        final_result = await session.execute_all_operations()
        print(f"Session result: {final_result['completed']} completed, {final_result['failed']} failed")
        
        # Get session progress
        progress = session.get_session_progress()
        if progress:
            print(f"Session progress: {progress['progress_percentage']:.1f}% complete")


if __name__ == "__main__":
    asyncio.run(example_usage())