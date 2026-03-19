"""
Async Natural Gas Table Client

Flexible, granular approach with:
- Individual method updates for different endpoints
- Bulk category operations when appropriate
- Custom update task creation
- Organized by data types but allows endpoint-specific methods

Design allows mixing different API endpoints within categories.
"""

import asyncio
from typing import Dict, List, Optional, Any, Union, Callable
import pandas as pd
from datetime import datetime
import inspect

from MacrOSINT.utils.async_table_wrapper import AsyncTableMixin
from MacrOSINT.data.data_tables import EIATable


class AsyncNGTable(AsyncTableMixin, EIATable):
    """
    Flexible Async Natural Gas Table Client
    
    Features:
    - Granular method updates for different API endpoints
    - Bulk operations when appropriate
    - Custom update task creation
    - Flexible data organization
    """
    
    def __init__(self, rename_key_cols=True):
        super().__init__("NG", rename_key_cols)
        
        # Define available update methods with their metadata
        self.update_methods = {
            # Storage methods (single endpoint typically)
            'underground_storage': {
                'key': 'storage/underground',
                'helper_method': 'get_storage_data',
                'category': 'storage',
                'description': 'Underground natural gas storage data',
                'endpoint_type': 'single'
            },
            
            # Consumption methods (multiple endpoints)
            'state_consumption': {
                'key': 'consumption/by_state',
                'helper_method': 'consumption_breakdown',
                'category': 'consumption',
                'description': 'State-level consumption data',
                'endpoint_type': 'single'
            },
            'regional_consumption': {
                'key': 'consumption/by_region',
                'helper_method': 'get_regional_consumption',
                'category': 'consumption', 
                'description': 'Regional consumption aggregated data',
                'endpoint_type': 'single'
            },
            'state_consumption_pct': {
                'key': 'consumption/state_pct',
                'helper_method': 'state_consumption_as_pct_async',
                'category': 'consumption',
                'description': 'State consumption as percentage of total (async)',
                'endpoint_type': 'single'
            },
            'residential_consumption': {
                'key': 'consumption/residential',
                'helper_method': 'get_residential_consumption',
                'category': 'consumption',
                'description': 'Residential sector consumption',
                'endpoint_type': 'single'
            },
            'commercial_consumption': {
                'key': 'consumption/commercial',
                'helper_method': 'get_commercial_consumption',
                'category': 'consumption',
                'description': 'Commercial sector consumption',
                'endpoint_type': 'single'
            },
            'industrial_consumption': {
                'key': 'consumption/industrial',
                'helper_method': 'get_industrial_consumption',
                'category': 'consumption',
                'description': 'Industrial sector consumption',
                'endpoint_type': 'single'
            },
            
            # Production methods
            'dry_gas_production': {
                'key': 'production/dry_gas',
                'helper_method': 'get_production_data',
                'category': 'production',
                'description': 'Dry natural gas production',
                'endpoint_type': 'single'
            },

            'state_production': {
                'key': 'production/by_state',
                'helper_method': 'get_state_production_detailed_async',
                'category': 'production',
                'description': 'State-level production including offshore and federal areas',
                'endpoint_type': 'single'
            },
            
            # Trade methods
            'imports': {
                'key': 'movement/imports',
                'helper_method': 'get_imports_data',
                'category': 'trade',
                'description': 'Natural gas imports',
                'endpoint_type': 'single'
            },
            'exports': {
                'key': 'movement/exports',
                'helper_method': 'get_exports_data',
                'category': 'trade',
                'description': 'Natural gas exports',
                'endpoint_type': 'single'
            },
            
            # Price methods (if available)
            'henry_hub_prices': {
                'key': 'prices/henry_hub',
                'helper_method': 'get_henry_hub_prices',
                'category': 'prices',
                'description': 'Henry Hub spot prices',
                'endpoint_type': 'single'
            }
        }
        
        # Group methods by category for bulk operations
        self.categories = {}
        for method_name, method_info in self.update_methods.items():
            category = method_info['category']
            if category not in self.categories:
                self.categories[category] = []
            self.categories[category].append(method_name)
    
    async def update_method_async(self,
                                 method_name: str,
                                 start: Optional[str] = None,
                                 end: Optional[str] = None,
                                 priority: int = 1,
                                 progress_callback: Optional[Callable] = None,
                                 **kwargs) -> Dict[str, Any]:
        """
        Update a specific data method asynchronously
        
        Args:
            method_name: Name of the method to update (e.g., 'state_consumption', 'underground_storage')
            start: Start date in YYYY-MM format (default: 2001-01)
            end: End date in YYYY-MM format (default: current year)
            priority: Priority for the operation (higher = more important)
            progress_callback: Optional callback for progress tracking
            **kwargs: Additional arguments to pass to the helper method
        
        Returns:
            Dictionary with update results
        """
        if method_name not in self.update_methods:
            available = list(self.update_methods.keys())
            raise ValueError(f"Unknown method: {method_name}. Available: {available}")
        
        method_info = self.update_methods[method_name]
        
        # Set default date range
        if not start:
            start = "2001-01"
        if not end:
            end = datetime.now().strftime("%Y-%m")
        
        print(f"Updating NG {method_name} data ({start} to {end})...")
        
        manager = await self._get_async_manager()
        helper = self.clients["NG"]  # NGHelper from your existing code
        
        try:
            # Fetch data using the specified helper method
            helper_method_name = method_info['helper_method']
            
            if not hasattr(helper, helper_method_name):
                raise AttributeError(f"Helper method {helper_method_name} not available")
            
            helper_method = getattr(helper, helper_method_name)
            
            # Call the helper method with appropriate arguments
            method_signature = inspect.signature(helper_method)
            method_params = {}
            
            # Add start/end if method accepts them
            if 'start' in method_signature.parameters:
                method_params['start'] = start
            if 'end' in method_signature.parameters:
                method_params['end'] = end
            
            # Add any additional kwargs that the method accepts
            for key, value in kwargs.items():
                if key in method_signature.parameters:
                    method_params[key] = value
            
            # Fetch the data - check if method is async
            if asyncio.iscoroutinefunction(helper_method):
                data = await helper_method(**method_params)
            else:
                data = helper_method(**method_params)
            
            if data is None or data.empty:
                print(f"No data returned for {method_name}")
                return {'completed': 0, 'failed': 0, 'method': method_name, 'message': 'No data returned'}
            
            # Queue write operation
            full_key = f"NG/{method_info['key']}"
            await manager.write_hdf(
                self.table_db,
                full_key,
                data,
                metadata={
                    'method': method_name,
                    'category': method_info['category'],
                    'updated': datetime.now(),
                    'date_range': f"{start} to {end}"
                },
                priority=priority,
                callback=progress_callback
            )
            
            # Execute the write operation
            result = await manager.execute_queued_operations()
            
            print(f"NG {method_name} update completed: {result['completed']} successful, {result['failed']} failed")
            
            return {
                **result,
                'method': method_name,
                'category': method_info['category'],
                'date_range': f"{start} to {end}",
                'records_updated': len(data) if data is not None else 0
            }
            
        except Exception as e:
            print(f"Error updating NG {method_name} data: {e}")
            return {
                'completed': 0,
                'failed': 1,
                'method': method_name,
                'error': str(e)
            }
    async def update_methods_async(self,
                                  methods: List[str],
                                  start: Optional[str] = None,
                                  end: Optional[str] = None,
                                  max_concurrent: int = 3,
                                  progress_callback: Optional[Callable] = None,
                                  **kwargs) -> Dict[str, Any]:
        """
        Update multiple specific methods asynchronously
        
        Args:
            methods: List of method names to update
            start: Start date in YYYY-MM format (default: 2001-01)
            end: End date in YYYY-MM format (default: current year)
            max_concurrent: Maximum concurrent operations
            progress_callback: Optional callback for progress tracking
            **kwargs: Additional arguments to pass to helper methods
        
        Returns:
            Dictionary with results for each method
        """
        print(f"Starting bulk update for NG methods: {methods}")
        
        # Create tasks for each method
        tasks = []
        for method_name in methods:
            if method_name in self.update_methods:
                task = self.update_method_async(
                    method_name=method_name,
                    start=start,
                    end=end,
                    progress_callback=progress_callback,
                    **kwargs
                )
                tasks.append((method_name, task))
            else:
                print(f"Warning: Unknown method {method_name}, skipping")
        
        # Execute all method updates concurrently
        results = {}
        completed_tasks = await asyncio.gather(*[task for _, task in tasks], return_exceptions=True)
        
        for (method_name, _), result in zip(tasks, completed_tasks):
            if isinstance(result, Exception):
                results[method_name] = {'error': str(result), 'completed': 0, 'failed': 1}
            else:
                results[method_name] = result
        
        # Summary
        total_completed = sum(r.get('completed', 0) for r in results.values())
        total_failed = sum(r.get('failed', 0) for r in results.values())
        
        print(f"Bulk NG method update completed: {total_completed} total successful, {total_failed} total failed")
        
        return {
            'summary': {
                'total_completed': total_completed,
                'total_failed': total_failed,
                'methods_processed': len(methods)
            },
            'method_results': results
        }
    
    async def update_category_async(self,
                                  category: str,
                                  start: Optional[str] = None,
                                  end: Optional[str] = None,
                                  max_concurrent: int = 3,
                                  progress_callback: Optional[Callable] = None,
                                  **kwargs) -> Dict[str, Any]:
        """
        Update all methods in a category
        
        Args:
            category: Category name ('storage', 'consumption', 'production', 'trade', 'prices')
            start: Start date in YYYY-MM format
            end: End date in YYYY-MM format
            max_concurrent: Maximum concurrent operations
            progress_callback: Optional callback for progress tracking
            **kwargs: Additional arguments to pass to helper methods
        
        Returns:
            Dictionary with update results
        """
        if category not in self.categories:
            available = list(self.categories.keys())
            raise ValueError(f"Unknown category: {category}. Available: {available}")
        
        methods_in_category = self.categories[category]
        print(f"Updating NG {category} category: {len(methods_in_category)} methods")
        
        return await self.update_methods_async(
            methods=methods_in_category,
            start=start,
            end=end,
            max_concurrent=max_concurrent,
            progress_callback=progress_callback,
            **kwargs
        )
    
    async def create_custom_update_task(self,
                                      task_name: str,
                                      key: str,
                                      helper_function: Union[str, Callable],
                                      start: Optional[str] = None,
                                      end: Optional[str] = None,
                                      priority: int = 1,
                                      progress_callback: Optional[Callable] = None,
                                      **helper_kwargs) -> Dict[str, Any]:
        """
        Create and execute a custom update task with specified helper function
        
        Args:
            task_name: Name for this custom task
            key: Storage key (e.g., 'custom/special_data')
            helper_function: Helper function name (str) or callable
            start: Start date
            end: End date
            priority: Priority for the operation
            progress_callback: Optional progress callback
            **helper_kwargs: Arguments to pass to the helper function
        
        Returns:
            Dictionary with update results
        """
        print(f"Executing custom NG update task: {task_name}")
        
        # Set default date range
        if not start:
            start = "2001-01"
        if not end:
            end = datetime.now().strftime("%Y-%m")
        
        manager = await self._get_async_manager()
        helper = self.clients["NG"]
        
        try:
            # Get the helper function
            if isinstance(helper_function, str):
                if not hasattr(helper, helper_function):
                    raise AttributeError(f"Helper function {helper_function} not available")
                func = getattr(helper, helper_function)
            elif callable(helper_function):
                func = helper_function
            else:
                raise ValueError("helper_function must be string (method name) or callable")
            
            # Prepare function arguments
            func_signature = inspect.signature(func)
            func_params = {}
            
            # Add start/end if function accepts them
            if 'start' in func_signature.parameters:
                func_params['start'] = start
            if 'end' in func_signature.parameters:
                func_params['end'] = end
            
            # Add custom kwargs
            for arg_key, arg_value in helper_kwargs.items():
                if arg_key in func_signature.parameters:
                    func_params[arg_key] = arg_value
            
            # Execute the helper function - check if it's async
            if asyncio.iscoroutinefunction(func):
                data = await func(**func_params)
            else:
                data = func(**func_params)
            
            if data is None or data.empty:
                print(f"No data returned for custom task {task_name}")
                return {'completed': 0, 'failed': 0, 'task': task_name, 'message': 'No data returned'}
            
            # Queue write operation
            full_key = f"NG/{key}"
            await manager.write_hdf(
                self.table_db,
                full_key,
                data,
                metadata={
                    'task_name': task_name,
                    'custom_task': True,
                    'updated': datetime.now(),
                    'date_range': f"{start} to {end}",
                    'helper_function': str(helper_function)
                },
                priority=priority,
                callback=progress_callback
            )
            
            # Execute the write operation
            result = await manager.execute_queued_operations()
            
            print(f"Custom NG task {task_name} completed: {result['completed']} successful, {result['failed']} failed")
            
            return {
                **result,
                'task_name': task_name,
                'key': key,
                'date_range': f"{start} to {end}",
                'records_updated': len(data) if data is not None else 0
            }
            
        except Exception as e:
            print(f"Error executing custom NG task {task_name}: {e}")
            return {
                'completed': 0,
                'failed': 1,
                'task_name': task_name,
                'error': str(e)
            }
    
    def list_methods(self) -> Dict[str, str]:
        """List all available update methods and their descriptions"""
        return {name: info['description'] for name, info in self.update_methods.items()}
    
    def get_methods_by_category(self, category: str) -> List[str]:
        """Get all method names for a specific category"""
        if category not in self.categories:
            raise ValueError(f"Unknown category: {category}. Available: {list(self.categories.keys())}")
        return self.categories[category]
    
    def get_method_info(self, method_name: str) -> Dict[str, Any]:
        """Get detailed information about a specific method"""
        if method_name not in self.update_methods:
            raise ValueError(f"Unknown method: {method_name}. Available: {list(self.update_methods.keys())}")
        return self.update_methods[method_name].copy()
    
    
    async def update_all_categories_async(self, 
                                        start: Optional[str] = None, 
                                        end: Optional[str] = None,
                                        categories: Optional[List[str]] = None,
                                        max_concurrent: int = 5,
                                        **kwargs) -> Dict[str, Any]:
        """
        Update all NG categories concurrently
        
        Args:
            start: Start date (default: 2001-01)
            end: End date (default: current)
            categories: List of categories to update (default: all)
            max_concurrent: Maximum concurrent operations
        
        Returns:
            Dictionary with results for all categories
        """
        if categories is None:
            categories = list(self.categories.keys())
        
        print(f"Starting bulk NG update for categories: {categories}")
        
        # Create tasks for each category
        tasks = []
        for category in categories:
            if category in self.categories:
                task = self.update_category_async(
                    category=category,
                    start=start,
                    end=end,
                    max_concurrent=max_concurrent
                )
                tasks.append((category, task))
        
        # Execute all category updates concurrently
        results = {}
        completed_tasks = await asyncio.gather(*[task for _, task in tasks], return_exceptions=True)
        
        for (category, _), result in zip(tasks, completed_tasks):
            if isinstance(result, Exception):
                results[category] = {'error': str(result), 'completed': 0, 'failed': 1}
            else:
                results[category] = result
        
        # Summary
        total_completed = sum(r.get('completed', 0) for r in results.values())
        total_failed = sum(r.get('failed', 0) for r in results.values())
        
        print(f"Bulk NG update completed: {total_completed} total successful, {total_failed} total failed")
        
        return {
            'summary': {
                'total_completed': total_completed,
                'total_failed': total_failed,
                'categories_processed': len(categories)
            },
            'category_results': results
        }
    
    def get_category_keys(self, category: str) -> List[str]:
        """Get all keys for a specific category"""
        if category not in self.categories:
            raise ValueError(f"Unknown category: {category}")
        return self.categories[category]['keys']
    
    def list_categories(self) -> Dict[str, str]:
        """List all available categories and their descriptions"""
        return {cat: info['description'] for cat, info in self.categories.items()}
    
    async def get_category_data_async(self, 
                                    category: str, 
                                    keys: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
        """
        Retrieve data for a category asynchronously
        
        Args:
            category: Category name
            keys: Specific keys to retrieve (default: all keys in category)
        
        Returns:
            Dictionary of {key: DataFrame}
        """
        if category not in self.categories:
            raise ValueError(f"Unknown category: {category}")
        
        if keys is None:
            keys = self.get_category_keys(category)
        
        manager = await self._get_async_manager()
        
        # Queue read operations
        read_operations = []
        for key in keys:
            full_key = f"NG/{key}"
            try:
                op_id = await manager.read_hdf(self.table_db, full_key)
                read_operations.append((key, op_id))
            except Exception as e:
                print(f"Error queuing read for {key}: {e}")
        
        # Execute reads
        result = await manager.execute_queued_operations()
        
        # Compile results
        data_dict = {}
        for key, op_id in read_operations:
            # Note: In a full implementation, you'd need to map op_id to results
            # This is a simplified version
            try:
                data = pd.read_hdf(self.table_db, f"NG/{key}")
                data_dict[key] = data
            except Exception as e:
                print(f"Error reading {key}: {e}")
        
        return data_dict


# Convenience functions for common operations
async def update_ng_storage(start: str = None, end: str = None, max_concurrent: int = 3) -> Dict[str, Any]:
    """Quick function to update just NG storage data"""
    ng_client = AsyncNGTable()
    try:
        return await ng_client.update_category_async('storage', start, end, max_concurrent)
    finally:
        await ng_client._shutdown_async_manager()


async def update_ng_consumption(start: str = None, end: str = None, max_concurrent: int = 3) -> Dict[str, Any]:
    """Quick function to update just NG consumption data"""
    ng_client = AsyncNGTable()
    try:
        return await ng_client.update_category_async('consumption', start, end, max_concurrent)
    finally:
        await ng_client._shutdown_async_manager()


async def bulk_update_ng_data(categories: List[str] = None, start: str = None, end: str = None) -> Dict[str, Any]:
    """Quick function to update multiple NG categories"""
    ng_client = AsyncNGTable()
    try:
        return await ng_client.update_all_categories_async(start, end, categories)
    finally:
        await ng_client._shutdown_async_manager()


# Example usage and testing
async def example_usage():
    """Example of how to use the new AsyncNGTable"""
    print("AsyncNGTable Example Usage")
    print("=" * 40)
    
    # Create client
    ng_client = AsyncNGTable()
    
    try:
        # List available categories
        categories = ng_client.list_categories()
        print("Available categories:")
        for cat, desc in categories.items():
            print(f"  {cat}: {desc}")
        
        print("\n1. Updating storage data...")
        storage_result = await ng_client.update_category_async('storage', start="2023-01", end="2024-08")
        print(f"Storage update: {storage_result['completed']} completed, {storage_result['failed']} failed")
        
        print("\n2. Updating consumption data...")
        consumption_result = await ng_client.update_category_async('consumption', start="2023-01", end="2024-08")
        print(f"Consumption update: {consumption_result['completed']} completed, {consumption_result['failed']} failed")
        
        print("\n3. Bulk update all categories...")
        bulk_result = await ng_client.update_all_categories_async(
            start="2024-01", 
            end="2024-08",
            categories=['storage', 'consumption', 'production']
        )
        print(f"Bulk update summary: {bulk_result['summary']}")
        
    finally:
        await ng_client._shutdown_async_manager()


class AsyncPetrolTable(AsyncTableMixin, EIATable):
    """
    Flexible Async Petroleum Table Client

    Features:
    - Granular method updates for different API endpoints
    - Bulk operations when appropriate
    - Custom update task creation
    - Flexible data organization
    """

    def __init__(self, rename_key_cols=True):
        super().__init__("PET", rename_key_cols)

        # Define available update methods with their metadata
        self.update_methods = {
            # Stocks methods (multiple endpoints)
            'refinery_crude_stocks': {
                'key': 'stocks/refinery_crude',
                'helper_method': 'get_refinery_crude_stocks_async',
                'category': 'stocks',
                'description': 'Refinery crude oil stocks',
                'endpoint_type': 'single'
            },
            'product_stocks': {
                'key': 'stocks/by_padd',
                'helper_method': 'get_product_stocks_async',
                'category': 'stocks',
                'description': 'Refined petroleum product stocks',
                'endpoint_type': 'single'
            },
            'tank_farm_stocks': {
                'key': 'stocks/tank_farm',
                'helper_method': 'get_tank_farm_stocks_async',
                'category': 'stocks',
                'description': 'Tank farm crude oil stocks',
                'endpoint_type': 'single'
            },

            # Production methods
            'crude_production': {
                'key': 'production/crude',
                'helper_method': 'get_production_by_area',
                'category': 'production',
                'description': 'Crude oil production by area',
                'endpoint_type': 'single'
            },
            'refined_production': {
                'key': 'production/refined',
                'helper_method': 'get_refined_products_production',
                'category': 'production',
                'description': 'Refined products production',
                'endpoint_type': 'single'
            },
            'refinery_utilization': {
                'key': 'production/utilization',
                'helper_method': 'get_refinery_utilization',
                'category': 'production',
                'description': 'Refinery utilization rates',
                'endpoint_type': 'single'
            },
            'refinery_consumption': {
                'key': 'production/refinery_consumption',
                'helper_method': 'get_refinery_consumption',
                'category': 'production',
                'description': 'Refinery crude oil consumption',
                'endpoint_type': 'single'
            },

            # Trade methods (multiple endpoints)
            'weekly_imports': {
                'key': 'movements/imports/weekly',
                'helper_method': 'get_imports_async',
                'category': 'trade',
                'description': 'Petroleum imports',
                'endpoint_type': 'single'
            },
            'imports_by_source': {
                'key': 'movements/imports',
                'helper_method': 'get_multiple_padd_imports_async',
                'category': 'trade',
                'description': 'Petroleum imports by source country and dest padd',
                'endpoint_type': 'single'
            },
            'exports':{
                "key":"movements/exports",
                "helper_method":"get_exports_async",
                "category":"trade",
                "description":"Refined product and Crude exports",
                "endpoint_type":'single'
            },
            'crude_movements': {
                'key': 'movements/crude_movements',
                'helper_method': 'get_crude_movements_async',
                'category': 'trade',
                'description': 'Inter-PADD crude oil movements',
                'endpoint_type': 'single'
            },
            'refined_product_movements': {
                'key': 'movements/refined_movements',
                'helper_method': 'get_refined_product_movements_async',
                'category': 'trade',
                'description': 'Net receipts for refined product movements between PADDs',
                'endpoint_type': 'single'
            },
            'inter-padd_crude_movements': {
                'key': 'movements/crude/inter_padd',
                'helper_method': 'get_crude_movements_async',
                'category': 'trade',
                'description': 'Total receipts for crude oil from producers and refineries between PADDs',
                'endpoint_type': 'single'
            },
            # Consumption methods
            'product_supplied': {
                'key': 'consumption/product_supplied',
                'helper_method': 'get_product_supplied_async',
                'category': 'consumption',
                'description': 'Refined products supplied (consumption proxy)',
                'endpoint_type': 'single'
            },
            'weekly_consumption': {
                'key': 'consumption/weekly',
                'helper_method': 'us_weekly_consumption_breakdown',
                'category': 'consumption',
                'description': 'US weekly consumption breakdown',
                'endpoint_type': 'single'
            }
        }

        # Group methods by category for bulk operations
        self.categories = {}
        for method_name, method_info in self.update_methods.items():
            category = method_info['category']
            if category not in self.categories:
                self.categories[category] = []
            self.categories[category].append(method_name)

    async def update_method_async(self,
                                 method_name: str,
                                 start: Optional[str] = None,
                                 end: Optional[str] = None,
                                 priority: int = 1,
                                 progress_callback: Optional[Callable] = None,
                                 **kwargs) -> Dict[str, Any]:
        """
        Update a specific data method asynchronously

        Args:
            method_name: Name of the method to update (e.g., 'refinery_crude_stocks', 'imports')
            start: Start date in YYYY-MM format (default: 2001-01)
            end: End date in YYYY-MM format (default: current year)
            priority: Priority for the operation (higher = more important)
            progress_callback: Optional callback for progress tracking
            **kwargs: Additional arguments to pass to the helper method

        Returns:
            Dictionary with update results
        """
        if method_name not in self.update_methods:
            available = list(self.update_methods.keys())
            raise ValueError(f"Unknown method: {method_name}. Available: {available}")

        method_info = self.update_methods[method_name]

        # Set default date range
        if not start:
            start = "2001-01"
        if not end:
            end = datetime.now().strftime("%Y-%m")

        print(f"Updating PET {method_name} data ({start} to {end})...")

        manager = await self._get_async_manager()
        helper = self.clients["PET"]  # PetroleumHelper from your existing code

        try:
            # Fetch data using the specified helper method
            helper_method_name = method_info['helper_method']

            if not hasattr(helper, helper_method_name):
                raise AttributeError(f"Helper method {helper_method_name} not available")

            helper_method = getattr(helper, helper_method_name)

            # Call the helper method with appropriate arguments
            method_signature = inspect.signature(helper_method)
            method_params = {}

            # Add start/end if method accepts them
            if 'start' in method_signature.parameters:
                method_params['start'] = start
            if 'end' in method_signature.parameters:
                method_params['end'] = end

            # Add any additional kwargs that the method accepts
            for key, value in kwargs.items():
                if key in method_signature.parameters:
                    method_params[key] = value

            # Fetch the data - check if method is async
            if asyncio.iscoroutinefunction(helper_method):
                data = await helper_method(**method_params)
            else:
                data = helper_method(**method_params)

            if data is None or data.empty:
                print(f"No data returned for {method_name}")
                return {'completed': 0, 'failed': 0, 'method': method_name, 'message': 'No data returned'}

            # Queue write operation
            full_key = f"PET/{method_info['key']}"
            await manager.write_hdf(
                self.table_db,
                full_key,
                data,
                metadata={
                    'method': method_name,
                    'category': method_info['category'],
                    'updated': datetime.now(),
                    'date_range': f"{start} to {end}"
                },
                priority=priority,
                callback=progress_callback
            )

            # Execute the write operation
            result = await manager.execute_queued_operations()

            print(f"PET {method_name} update completed: {result['completed']} successful, {result['failed']} failed")

            return {
                **result,
                'method': method_name,
                'category': method_info['category'],
                'date_range': f"{start} to {end}",
                'records_updated': len(data) if data is not None else 0
            }

        except Exception as e:
            print(f"Error updating PET {method_name} data: {e}")
            return {
                'completed': 0,
                'failed': 1,
                'method': method_name,
                'error': str(e)
            }

    async def update_methods_async(self,
                                  methods: List[str],
                                  start: Optional[str] = None,
                                  end: Optional[str] = None,
                                  max_concurrent: int = 3,
                                  progress_callback: Optional[Callable] = None,
                                  **kwargs) -> Dict[str, Any]:
        """
        Update multiple specific methods asynchronously

        Args:
            methods: List of method names to update
            start: Start date in YYYY-MM format (default: 2001-01)
            end: End date in YYYY-MM format (default: current year)
            max_concurrent: Maximum concurrent operations
            progress_callback: Optional callback for progress tracking
            **kwargs: Additional arguments to pass to helper methods

        Returns:
            Dictionary with results for each method
        """
        print(f"Starting bulk update for PET methods: {methods}")

        # Create tasks for each method
        tasks = []
        for method_name in methods:
            if method_name in self.update_methods:
                task = self.update_method_async(
                    method_name=method_name,
                    start=start,
                    end=end,
                    progress_callback=progress_callback,
                    **kwargs
                )
                tasks.append((method_name, task))
            else:
                print(f"Warning: Unknown method {method_name}, skipping")

        # Execute all method updates concurrently
        results = {}
        completed_tasks = await asyncio.gather(*[task for _, task in tasks], return_exceptions=True)

        for (method_name, _), result in zip(tasks, completed_tasks):
            if isinstance(result, Exception):
                results[method_name] = {'error': str(result), 'completed': 0, 'failed': 1}
            else:
                results[method_name] = result

        # Summary
        total_completed = sum(r.get('completed', 0) for r in results.values())
        total_failed = sum(r.get('failed', 0) for r in results.values())

        print(f"Bulk PET method update completed: {total_completed} total successful, {total_failed} total failed")

        return {
            'summary': {
                'total_completed': total_completed,
                'total_failed': total_failed,
                'methods_processed': len(methods)
            },
            'method_results': results
        }

    async def update_category_async(self,
                                  category: str,
                                  start: Optional[str] = None,
                                  end: Optional[str] = None,
                                  max_concurrent: int = 3,
                                  progress_callback: Optional[Callable] = None,
                                  **kwargs) -> Dict[str, Any]:
        """
        Update all methods in a category

        Args:
            category: Category name ('stocks', 'production', 'trade', 'consumption')
            start: Start date in YYYY-MM format
            end: End date in YYYY-MM format
            max_concurrent: Maximum concurrent operations
            progress_callback: Optional callback for progress tracking
            **kwargs: Additional arguments to pass to helper methods

        Returns:
            Dictionary with update results
        """
        if category not in self.categories:
            available = list(self.categories.keys())
            raise ValueError(f"Unknown category: {category}. Available: {available}")

        methods_in_category = self.categories[category]
        print(f"Updating PET {category} category: {len(methods_in_category)} methods")

        return await self.update_methods_async(
            methods=methods_in_category,
            start=start,
            end=end,
            max_concurrent=max_concurrent,
            progress_callback=progress_callback,
            **kwargs
        )

    async def create_custom_update_task(self,
                                      task_name: str,
                                      key: str,
                                      helper_function: Union[str, Callable],
                                      start: Optional[str] = None,
                                      end: Optional[str] = None,
                                      priority: int = 1,
                                      progress_callback: Optional[Callable] = None,
                                      **helper_kwargs) -> Dict[str, Any]:
        """
        Create and execute a custom update task with specified helper function

        Args:
            task_name: Name for this custom task
            key: Storage key (e.g., 'custom/special_data')
            helper_function: Helper function name (str) or callable
            start: Start date
            end: End date
            priority: Priority for the operation
            progress_callback: Optional progress callback
            **helper_kwargs: Arguments to pass to the helper function

        Returns:
            Dictionary with update results
        """
        print(f"Executing custom PET update task: {task_name}")

        # Set default date range
        if not start:
            start = "2001-01"
        if not end:
            end = datetime.now().strftime("%Y-%m")

        manager = await self._get_async_manager()
        helper = self.clients["PET"]

        try:
            # Get the helper function
            if isinstance(helper_function, str):
                if not hasattr(helper, helper_function):
                    raise AttributeError(f"Helper function {helper_function} not available")
                func = getattr(helper, helper_function)
            elif callable(helper_function):
                func = helper_function
            else:
                raise ValueError("helper_function must be string (method name) or callable")

            # Prepare function arguments
            func_signature = inspect.signature(func)
            func_params = {}

            # Add start/end if function accepts them
            if 'start' in func_signature.parameters:
                func_params['start'] = start
            if 'end' in func_signature.parameters:
                func_params['end'] = end

            # Add custom kwargs
            for arg_key, arg_value in helper_kwargs.items():
                if arg_key in func_signature.parameters:
                    func_params[arg_key] = arg_value

            # Execute the helper function - check if it's async
            if asyncio.iscoroutinefunction(func):
                data = await func(**func_params)
            else:
                data = func(**func_params)

            if data is None or data.empty:
                print(f"No data returned for custom task {task_name}")
                return {'completed': 0, 'failed': 0, 'task': task_name, 'message': 'No data returned'}

            # Queue write operation
            full_key = f"PET/{key}"
            await manager.write_hdf(
                self.table_db,
                full_key,
                data,
                metadata={
                    'task_name': task_name,
                    'custom_task': True,
                    'updated': datetime.now(),
                    'date_range': f"{start} to {end}",
                    'helper_function': str(helper_function)
                },
                priority=priority,
                callback=progress_callback
            )

            # Execute the write operation
            result = await manager.execute_queued_operations()

            print(f"Custom PET task {task_name} completed: {result['completed']} successful, {result['failed']} failed")

            return {
                **result,
                'task_name': task_name,
                'key': key,
                'date_range': f"{start} to {end}",
                'records_updated': len(data) if data is not None else 0
            }

        except Exception as e:
            print(f"Error executing custom PET task {task_name}: {e}")
            return {
                'completed': 0,
                'failed': 1,
                'task_name': task_name,
                'error': str(e)
            }

    def list_methods(self) -> Dict[str, str]:
        """List all available update methods and their descriptions"""
        return {name: info['description'] for name, info in self.update_methods.items()}

    def get_methods_by_category(self, category: str) -> List[str]:
        """Get all method names for a specific category"""
        if category not in self.categories:
            raise ValueError(f"Unknown category: {category}. Available: {list(self.categories.keys())}")
        return self.categories[category]

    def get_method_info(self, method_name: str) -> Dict[str, Any]:
        """Get detailed information about a specific method"""
        if method_name not in self.update_methods:
            raise ValueError(f"Unknown method: {method_name}. Available: {list(self.update_methods.keys())}")
        return self.update_methods[method_name].copy()

    async def update_all_categories_async(self,
                                        start: Optional[str] = None,
                                        end: Optional[str] = None,
                                        categories: Optional[List[str]] = None,
                                        max_concurrent: int = 5,
                                        **kwargs) -> Dict[str, Any]:
        """
        Update all PET categories concurrently

        Args:
            start: Start date (default: 2001-01)
            end: End date (default: current)
            categories: List of categories to update (default: all)
            max_concurrent: Maximum concurrent operations
            **kwargs: Additional arguments to pass to helper methods

        Returns:
            Dictionary with results for all categories
        """
        if categories is None:
            categories = list(self.categories.keys())

        print(f"Starting bulk PET update for categories: {categories}")

        # Create tasks for each category
        tasks = []
        for category in categories:
            if category in self.categories:
                task = self.update_category_async(
                    category=category,
                    start=start,
                    end=end,
                    max_concurrent=max_concurrent,
                    **kwargs
                )
                tasks.append((category, task))

        # Execute all category updates concurrently
        results = {}
        completed_tasks = await asyncio.gather(*[task for _, task in tasks], return_exceptions=True)

        for (category, _), result in zip(tasks, completed_tasks):
            if isinstance(result, Exception):
                results[category] = {'error': str(result), 'completed': 0, 'failed': 1}
            else:
                results[category] = result

        # Summary
        total_completed = sum(r.get('summary', {}).get('total_completed', r.get('completed', 0)) for r in results.values())
        total_failed = sum(r.get('summary', {}).get('total_failed', r.get('failed', 0)) for r in results.values())

        print(f"Bulk PET update completed: {total_completed} total successful, {total_failed} total failed")

        return {
            'summary': {
                'total_completed': total_completed,
                'total_failed': total_failed,
                'categories_processed': len(categories)
            },
            'category_results': results
        }
