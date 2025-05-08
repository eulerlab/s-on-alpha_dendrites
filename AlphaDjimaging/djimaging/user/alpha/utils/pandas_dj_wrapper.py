"""
Module to use Pandas DataFrames as if they were DataJoint tables.
Does only support a small subset of functions for plotting data.
Does not support any populate-related functions.

Code written by claude.ai
"""

import os
import re
from typing import Union, List, Dict, Any, Tuple

import numpy as np
import pandas as pd


class DataJointTable:
    def __init__(self, df: pd.DataFrame, primary_keys: Union[List[str], None] = None, table_name: str = None):
        """
        Initialize a DataJoint-like table from a pandas DataFrame.

        Args:
            df: The pandas DataFrame containing the table data
            primary_keys: List of column names that serve as primary keys.
                         If None, will try to infer from DataFrame index.
            table_name: Optional name for the table
        """
        self.df = df.copy()
        self.table_name = table_name

        # Infer primary keys if not provided
        if primary_keys is None:
            if isinstance(df.index, pd.MultiIndex):
                self.primary_keys = list(df.index.names)
                # Reset index to make primary keys regular columns for easier operations
                self.df = self.df.reset_index()
            else:
                # Use index as primary key if it has a name
                if df.index.name is not None:
                    self.primary_keys = [df.index.name]
                    self.df = self.df.reset_index()
                else:
                    raise ValueError("Primary keys must be provided if DataFrame doesn't have a named index")
        else:
            self.primary_keys = primary_keys

        # Validate that all primary keys exist in the DataFrame
        missing_keys = [key for key in self.primary_keys if key not in self.df.columns]
        if missing_keys:
            raise ValueError(f"Primary keys {missing_keys} not found in DataFrame columns")

    def __and__(self, other: Union['DataJointTable', Dict[str, Any], str, List, Tuple],
                allow_missing=True, ignore_missing=True) -> 'DataJointTable':
        """
        Implement restriction operation (&) similar to DataJoint.

        Args:
            other: Another DataJointTable, a dictionary of {key: value} pairs,
                  a string condition (e.g., "experimenter='jane'"),
                  a list of any of these types (which will be OR'd together),
                  or a tuple whose values correspond to the first n primary keys
            allow_missing: Whether to allow missing keys without raising errors
            ignore_missing: Whether to silently ignore dictionary keys that don't exist in the table

        Returns:
            A new DataJointTable with the restrictions applied
        """
        # Handle tuple operand as a restriction on primary keys
        if isinstance(other, tuple):
            if not self.primary_keys:
                raise ValueError("Cannot restrict by tuple when table has no primary keys")

            # Create a dictionary mapping primary keys to tuple values
            if len(other) > len(self.primary_keys):
                raise ValueError(
                    f"Tuple contains {len(other)} values but table only has {len(self.primary_keys)} primary keys")

            # Map the tuple values to the first n primary keys
            restriction_dict = {
                self.primary_keys[i]: other[i]
                for i in range(len(other))
            }

            # Apply the restriction using the dictionary
            return self & restriction_dict

        # Handle list operand as an OR condition
        # Modify the list handling section to handle unhashable types
        elif isinstance(other, list):
            if not other:  # Empty list
                # Return empty table (no matches)
                return DataJointTable(self.df.head(0), self.primary_keys, self.table_name)

            # Apply the first restriction
            result = self & other[0]

            # OR with the remaining restrictions
            for item in other[1:]:
                item_result = self & item
                # Combine the results but only consider primary keys for duplicates
                # This avoids hashing problems with array columns
                combined_df = pd.concat([result.df, item_result.df])
                if self.primary_keys:
                    # Drop duplicates only by primary key columns which should be hashable
                    result_df = combined_df.drop_duplicates(subset=self.primary_keys)
                else:
                    # If no primary keys, just keep all rows (could have duplicates)
                    result_df = combined_df
                result = DataJointTable(result_df, self.primary_keys, self.table_name)

            return result

        elif isinstance(other, str):
            # Parse the string restriction
            return self._restrict_by_string(other)
        elif isinstance(other, dict):
            # Restrict by key-value pairs
            query_conditions = []
            query_params = {}

            # Filter out keys that don't exist in the table if ignore_missing is True
            filtered_items = {}
            for key, value in other.items():
                if key in self.df.columns:
                    filtered_items[key] = value
                elif not allow_missing and not ignore_missing:
                    raise ValueError(f"Restriction key '{key}' not found in table columns")
                # Keys not in columns are silently ignored if ignore_missing is True

            for key, value in filtered_items.items():
                if isinstance(value, (list, tuple, set)):
                    # Handle list/tuple/set of values (IN clause)
                    if len(value) == 0:
                        # Empty list means no matches
                        return DataJointTable(self.df.head(0), self.primary_keys, self.table_name)
                    param_name = f"value_{len(query_params)}"
                    query_params[param_name] = list(value)
                    query_conditions.append(f"{key} in @{param_name}")
                else:
                    # Simple equality condition
                    param_name = f"value_{len(query_params)}"
                    query_params[param_name] = value
                    query_conditions.append(f"{key} == @{param_name}")

            if query_conditions:
                query = " & ".join(query_conditions)
                result_df = self.df.query(query, local_dict=query_params)
            else:
                result_df = self.df.copy()

            return DataJointTable(result_df, self.primary_keys, self.table_name)

        elif isinstance(other, DataJointTable):
            # Restrict by another table (join on common primary keys)
            common_keys = set(self.primary_keys) & set(other.primary_keys)
            if not common_keys:
                raise ValueError("No common primary keys found for restriction between tables")

            # Get unique combinations of the common keys from the other table
            restriction_df = other.df[list(common_keys)].drop_duplicates()

            # Perform a merge (equivalent to SQL INNER JOIN)
            result_df = pd.merge(self.df, restriction_df, on=list(common_keys), how='inner')

            return DataJointTable(result_df, self.primary_keys, self.table_name)
        else:
            raise TypeError(f"Unsupported operand type for &: {type(other)}")

    def __mul__(self, other: 'DataJointTable') -> 'DataJointTable':
        """
        Implement join operation (*) similar to DataJoint.
        Performs a natural join between tables based on matching attributes.

        Args:
            other: Another DataJointTable to join with

        Returns:
            A new DataJointTable representing the join result
        """
        if not isinstance(other, DataJointTable):
            raise TypeError(f"Unsupported operand type for *: {type(other)}")

        # Find common column names (matching attributes)
        common_cols = set(self.df.columns) & set(other.df.columns)
        if not common_cols:
            raise ValueError("No matching attributes found for natural join between tables")

        # Perform natural join (SQL INNER JOIN on matching columns)
        result_df = pd.merge(self.df, other.df, on=list(common_cols), how='inner')

        # Determine the primary keys for the joined table
        # In DataJoint, the primary key of a join is the union of the primary keys
        joined_primary_keys = list(set(self.primary_keys) | set(other.primary_keys))

        # For the table name, combine the names if available
        if self.table_name and other.table_name:
            joined_table_name = f"{self.table_name} * {other.table_name}"
        else:
            joined_table_name = None

        return DataJointTable(result_df, joined_primary_keys, joined_table_name)

    def _restrict_by_string(self, condition: str) -> 'DataJointTable':
        """
        Implement string-based restriction similar to DataJoint.

        Args:
            condition: String condition (e.g., "experimenter='jane'")

        Returns:
            A new DataJointTable with the restrictions applied
        """
        # Convert DataJoint-style string restriction to pandas query syntax

        # Handle equality conditions (e.g., "field='value'")
        equality_pattern = r"([a-zA-Z0-9_]+)\s*=\s*'([^']*)'"
        condition = re.sub(equality_pattern, r"\1 == '\2'", condition)

        # Handle equality with double quotes (e.g., 'field="value"')
        equality_pattern_dquote = r'([a-zA-Z0-9_]+)\s*=\s*"([^"]*)"'
        condition = re.sub(equality_pattern_dquote, r'\1 == "\2"', condition)

        # Handle numeric equality (e.g., "field=123")
        numeric_pattern = r"([a-zA-Z0-9_]+)\s*=\s*([0-9.]+)"
        condition = re.sub(numeric_pattern, r"\1 == \2", condition)

        # Handle IN clause (e.g., "field in ('a', 'b', 'c')")
        # No direct conversion needed as pandas query supports this syntax

        # Handle NOT clause (e.g., "not field='value'")
        condition = re.sub(r"NOT\s+", "not ", condition, flags=re.IGNORECASE)

        # Verify that the referenced columns exist
        column_pattern = r"([a-zA-Z0-9_]+)\s*[=<>!]"
        for match in re.finditer(column_pattern, condition):
            column = match.group(1)
            if column not in self.df.columns:
                raise ValueError(f"Column '{column}' in string restriction not found in table")

        try:
            result_df = self.df.query(condition)
            return DataJointTable(result_df, self.primary_keys, self.table_name)
        except Exception as e:
            raise ValueError(f"Failed to parse string restriction: {condition}. Error: {str(e)}")

    def __len__(self) -> int:
        """Return the number of rows in the table."""
        return len(self.df)

    def fetch(self, *args, as_dict=False, format=None, order_by=None) -> Union[
        List[np.ndarray], List[Dict[str, Any]], pd.DataFrame, Tuple[np.ndarray, ...], np.ndarray]:
        """
        Fetch data from the table, similar to DataJoint's fetch method.

        Args:
            *args: Column names to fetch. If empty, fetch all columns.
                   Special value 'KEY' can be included to fetch primary key values.
            as_dict: If True, return a list of dictionaries (one per row).
                     Otherwise, return a list of numpy arrays (one per column).
            format: Optional format specifier:
                   - 'frame': Return a DataFrame with primary keys as multi-index
                   - None: Return based on as_dict parameter
            order_by: Column name(s) to order the results by. Can be:
                    - A single column name string
                    - A list of column names
                    - A tuple of (column_name, ascending) pairs
                    - A string of comma-separated column names (prefixed with - for descending)

        Returns:
            - Results in the exact order specified in args
            - If no arguments: List of dictionaries or tuple of arrays for all columns
            - If a single column (not 'KEY'): A numpy array of values
            - If multiple columns: A tuple of results in the same order as the args
            - If format='frame': A pandas DataFrame
        """
        import numpy as np

        # Format parameter overrides as_dict
        if format == 'frame':
            return self._fetch_as_frame(*args, order_by=order_by)

        # Create a copy of the DataFrame for sorting and selection
        working_df = self.df.copy()

        # Apply ordering if specified
        if order_by is not None:
            working_df = self._apply_ordering(working_df, order_by)

        # Process each argument in order
        results = []

        # If no arguments were provided, fetch all columns
        if not args:
            if as_dict:
                return [dict(row) for _, row in working_df.iterrows()]
            else:
                # For no args, return a tuple of arrays for all columns
                columns = list(working_df.columns)
                arrays = [working_df[col].to_numpy() for col in columns]
                return tuple(arrays) if len(arrays) > 1 else arrays[0]

        # Process each argument in the exact order provided
        for arg in args:
            if arg == 'KEY':
                # Handle KEY - return a list of primary key dictionaries
                key_result = [
                    {pk: row[pk] for pk in self.primary_keys}
                    for _, row in working_df.iterrows()
                ]
                results.append(key_result)
            elif arg in working_df.columns:
                # Handle regular column
                if as_dict:
                    col_result = [{arg: val} for val in working_df[arg].tolist()]
                else:
                    col_result = working_df[arg].to_numpy()
                results.append(col_result)
            else:
                raise ValueError(f"Column '{arg}' not found in table")

        # If only one argument was provided, return it directly without wrapping in a tuple
        if len(args) == 1:
            return results[0]

        # Otherwise return a tuple of results in the exact order requested
        return tuple(results)

    def _fetch_as_frame(self, *args, order_by=None) -> pd.DataFrame:
        """
        Fetch data and return as a DataFrame with primary keys as multi-index.

        Args:
            *args: Column names to fetch. If empty, fetch all columns except primary keys.
            order_by: Column name(s) to order the results by.

        Returns:
            DataFrame with primary keys as multi-index
        """
        # Create a copy of the DataFrame for sorting and selection
        working_df = self.df.copy()

        # Apply ordering if specified
        if order_by is not None:
            working_df = self._apply_ordering(working_df, order_by)

        # Determine which columns to include
        if not args:
            # Include all non-primary key columns if none specified
            columns_to_fetch = [col for col in working_df.columns if col not in self.primary_keys]
        else:
            # Handle KEY special argument
            if 'KEY' in args:
                # If KEY is included, it's already handled by including primary keys as index
                args = [arg for arg in args if arg != 'KEY']

            # Check if all requested columns exist
            missing_cols = [col for col in args if col not in working_df.columns]
            if missing_cols:
                raise ValueError(f"Columns {missing_cols} not found in table")

            # Remove primary keys from the args (they'll be part of the index)
            columns_to_fetch = [col for col in args if col not in self.primary_keys]

        # If no columns left after removing primary keys, just return primary keys as DataFrame
        if not columns_to_fetch:
            result = working_df[self.primary_keys].copy()
            result = result.set_index(self.primary_keys)
            return result

        # Create a copy of the DataFrame with the needed columns
        cols_to_include = self.primary_keys + columns_to_fetch
        result = working_df[cols_to_include].copy()

        # Set primary keys as index
        if len(self.primary_keys) == 1:
            # Single-level index
            result = result.set_index(self.primary_keys[0])
        else:
            # Multi-level index
            result = result.set_index(self.primary_keys)

        return result

    def _apply_ordering(self, df, order_by) -> pd.DataFrame:
        """
        Apply ordering to a DataFrame based on the order_by parameter.

        Args:
            df: DataFrame to sort
            order_by: Sorting specification, which can be:
                     - A single column name string
                     - A list of column names (all ascending)
                     - A tuple of (column_name, ascending) pairs
                     - A string of comma-separated column names (prefixed with - for descending)

        Returns:
            Sorted DataFrame
        """
        # Handle different formats of order_by
        if isinstance(order_by, str):
            if ',' in order_by:
                # Parse a comma-separated string like "column1,-column2"
                sort_specs = []
                for col_spec in order_by.split(','):
                    col_spec = col_spec.strip()
                    if col_spec.startswith('-'):
                        col_name = col_spec[1:]
                        ascending = False
                    else:
                        col_name = col_spec
                        ascending = True

                    if col_name not in df.columns:
                        raise ValueError(f"Column '{col_name}' in order_by not found in table")
                    sort_specs.append((col_name, ascending))

                # Sort by multiple columns with specified directions
                return df.sort_values(
                    by=[spec[0] for spec in sort_specs],
                    ascending=[spec[1] for spec in sort_specs]
                )
            else:
                # Simple single column
                if order_by not in df.columns:
                    raise ValueError(f"Column '{order_by}' in order_by not found in table")
                return df.sort_values(by=order_by)
        elif isinstance(order_by, (list, tuple)):
            if all(isinstance(item, str) for item in order_by):
                # List of column names (all ascending)
                missing_cols = [col for col in order_by if col not in df.columns]
                if missing_cols:
                    raise ValueError(f"Columns {missing_cols} in order_by not found in table")
                return df.sort_values(by=list(order_by))
            elif all(isinstance(item, tuple) and len(item) == 2 for item in order_by):
                # List of (column, ascending) tuples
                cols = [item[0] for item in order_by]
                ascending = [item[1] for item in order_by]

                missing_cols = [col for col in cols if col not in df.columns]
                if missing_cols:
                    raise ValueError(f"Columns {missing_cols} in order_by not found in table")

                return df.sort_values(by=cols, ascending=ascending)
            else:
                raise ValueError(
                    "Invalid format for order_by. Must be a list of column names or a list of (column, ascending) tuples.")
        else:
            raise ValueError("Invalid type for order_by. Must be a string, list, or tuple.")

    def fetch1(self, *args) -> Union[Dict[str, Any], Any, Tuple]:
        """
        Fetch a single row from the table, similar to DataJoint's fetch1.
        Raises an error if the table contains more than one row.

        Args:
            *args: Column names to fetch. If empty, fetch all columns.
                   If a single column name is provided (not KEY), returns the value directly.
                   Special value 'KEY' can be included to fetch primary key values.

        Returns:
            If a single non-KEY column is requested, returns the value directly.
            If 'KEY' is included among regular columns, returns a tuple where the last element
            is a dictionary containing primary key values.
            If only 'KEY' is specified, returns a dictionary with primary key values.
            If multiple columns (but not KEY) are specified, returns a tuple of values.
            If no arguments are provided, returns a dictionary with all values.
        """
        if len(self.df) != 1:
            raise ValueError(f"Expected exactly one row, but found {len(self.df)}")

        # Check if KEY is among the arguments
        has_key = 'KEY' in args
        regular_columns = [arg for arg in args if arg != 'KEY' and arg in self.df.columns]

        # Validate all columns exist
        invalid_columns = [arg for arg in args if arg != 'KEY' and arg not in self.df.columns]
        if invalid_columns:
            raise ValueError(f"Columns {invalid_columns} not found in table")

        # If only KEY was specified
        if not regular_columns and has_key:
            # Return a dictionary containing only primary key values
            return {pk: self.df.iloc[0][pk] for pk in self.primary_keys}

        # If no arguments were provided
        if not args:
            # Return a dictionary with all values
            return {col: self.df.iloc[0][col] for col in self.df.columns}

        # Handle regular columns
        if len(regular_columns) == 1 and not has_key:
            # Return the value directly if only one non-KEY column
            return self.df.iloc[0][regular_columns[0]]
        else:
            # Return a tuple of values for regular columns
            regular_result = tuple(self.df.iloc[0][col] for col in regular_columns)

        # If KEY wasn't requested, just return the regular columns
        if not has_key:
            return regular_result

        # If KEY was requested, also return primary keys
        key_result = {pk: self.df.iloc[0][pk] for pk in self.primary_keys}

        # Return a tuple of (regular_result, key_result)
        return (*regular_result, key_result) if len(regular_columns) > 0 else key_result

    def proj(self, *args, **kwargs) -> 'DataJointTable':
        """
        Project (select) columns from the table, similar to DataJoint's proj.

        Args:
            *args: Column names to include in the projection.
            **kwargs: New column names as {new_name: old_name} or {new_name: 'expr'}

        Returns:
            A new DataJointTable with the projected columns
        """
        # Start with the primary keys
        columns_to_include = list(self.primary_keys)

        # Add any additional columns from args
        for col in args:
            if col not in self.df.columns:
                raise ValueError(f"Column '{col}' not found in table")
            if col not in columns_to_include:
                columns_to_include.append(col)

        result_df = self.df[columns_to_include].copy()

        # Apply any column renames or expressions from kwargs
        for new_name, old_name_or_expr in kwargs.items():
            if isinstance(old_name_or_expr, str) and old_name_or_expr in self.df.columns:
                # Simple renaming of existing column
                result_df[new_name] = self.df[old_name_or_expr]
            else:
                # This is a simplified approach - real SQL expressions aren't supported
                # For more complex expressions, you would need a SQL parser
                raise ValueError("Complex expressions in proj() are not yet supported")

        # Determine primary keys for the projected table (keep the original primary keys)
        projected_primary_keys = [pk for pk in self.primary_keys if pk in result_df.columns]

        return DataJointTable(result_df, projected_primary_keys, self.table_name)

    def head(self, n=5) -> 'DataJointTable':
        """Get the first n rows of the table."""
        return DataJointTable(self.df.head(n), self.primary_keys, self.table_name)

    def tail(self, n=5) -> 'DataJointTable':
        """Get the last n rows of the table."""
        return DataJointTable(self.df.tail(n), self.primary_keys, self.table_name)

    def __repr__(self) -> str:
        """String representation of the table."""
        table_info = f"DataJointTable"
        if self.table_name:
            table_info += f" '{self.table_name}'"
        table_info += f" with {len(self.df)} rows, primary keys: {self.primary_keys}"
        return table_info + "\n" + str(self.df.head())


def load_dj_table(filepath: str, primary_keys: Union[List[str], None] = None) -> DataJointTable:
    """
    Load a DataJoint table from an HDF5 file.

    Args:
        filepath: Path to the HDF5 file
        primary_keys: Primary key column names. If None, will try to infer from the DataFrame

    Returns:
        A DataJointTable object
    """
    df = pd.read_hdf(filepath)
    table_name = os.path.splitext(os.path.basename(filepath))[0]
    return DataJointTable(df, primary_keys, table_name)


def load_dj_database(database_export_root: str, table_map: Dict[str, Union[List[str], None]] = None) -> Dict[
    str, DataJointTable]:
    """
    Load multiple DataJoint tables from a directory.

    Args:
        database_export_root: Directory containing HDF5 files
        table_map: Dictionary mapping filenames (without .h5) to primary key lists.
                  If a table is not in the map, will try to infer primary keys.

    Returns:
        Dictionary of DataJointTable objects keyed by table name
    """
    if table_map is None:
        table_map = {}

    result = {}
    for filename in os.listdir(database_export_root):
        if filename.endswith('.h5'):
            table_name = os.path.splitext(filename)[0]
            filepath = os.path.join(database_export_root, filename)
            primary_keys = table_map.get(table_name, None)
            result[table_name] = load_dj_table(filepath, primary_keys)

    return result
