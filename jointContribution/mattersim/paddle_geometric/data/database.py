import io
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cached_property
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import paddle
from paddle import Tensor
from tqdm import tqdm
import pickle  # Used for serializing and deserializing complex objects


@dataclass
class TensorInfo:
    """Describes the type information of a tensor, including data type, size,
    whether it's an index or an edge index."""
    dtype: paddle.dtype
    size: Tuple[int, ...] = (-1,)
    is_index: bool = False
    is_edge_index: bool = False

    def __post_init__(self) -> None:
        # A tensor cannot be both an index and an edge index simultaneously
        if self.is_index and self.is_edge_index:
            raise ValueError("Tensor cannot be both 'Index' and 'EdgeIndex' at the same time.")
        if self.is_index:
            self.size = (-1,)  # Dynamic size for index tensors
        if self.is_edge_index:
            self.size = (2, -1)  # Edge indices are two-dimensional


def maybe_cast_to_tensor_info(value: Any) -> Union[Any, TensorInfo]:
    """Converts input to TensorInfo if it meets the criteria."""
    if not isinstance(value, dict):
        return value
    if len(value) < 1 or len(value) > 3:
        return value
    if 'dtype' not in value:
        return value
    valid_keys = {'dtype', 'size', 'is_index', 'is_edge_index'}
    if len(set(value.keys()) | valid_keys) != len(valid_keys):
        return value
    return TensorInfo(**value)


Schema = Union[Any, Dict[str, Any], Tuple[Any], List[Any]]


class Database(ABC):
    """Abstract base class for a database that supports inserting and retrieving data.

    A database acts as an index-based key-value store for tensors and other custom data.
    """
    def __init__(self, schema: Schema = object) -> None:
        schema_dict = self._to_dict(schema)
        self.schema: Dict[Union[str, int], Any] = schema_dict

    @abstractmethod
    def insert(self, index: int, data: Any) -> None:
        """Insert data at a specified index."""
        raise NotImplementedError

    def multi_insert(self, indices: Union[Sequence[int], slice], data_list: Sequence[Any]) -> None:
        """Insert multiple data entries at specified indices."""
        if isinstance(indices, slice):
            indices = self.slice_to_range(indices)
        for index, data in zip(indices, data_list):
            self.insert(index, data)

    @abstractmethod
    def get(self, index: int) -> Any:
        """Retrieve data from a specified index."""
        raise NotImplementedError

    def multi_get(self, indices: Union[Sequence[int], slice]) -> List[Any]:
        """Retrieve data from multiple indices."""
        if isinstance(indices, slice):
            indices = self.slice_to_range(indices)
        return [self.get(index) for index in indices]

    @staticmethod
    def _to_dict(value: Any) -> Dict[Union[str, int], Any]:
        """Convert the input value to a dictionary."""
        if isinstance(value, dict):
            return value
        if isinstance(value, (tuple, list)):
            return {i: v for i, v in enumerate(value)}
        return {0: value}

    def slice_to_range(self, indices: slice) -> range:
        """Convert a slice object into a range object."""
        start = indices.start or 0
        stop = indices.stop or len(self)
        step = indices.step or 1
        return range(start, stop, step)

    def __len__(self) -> int:
        """Return the number of entries in the database."""
        raise NotImplementedError

    def __getitem__(self, key: Union[int, Sequence[int], slice]) -> Union[Any, List[Any]]:
        """Retrieve data using index or slice."""
        if isinstance(key, int):
            return self.get(key)
        return self.multi_get(key)

    def __setitem__(self, key: Union[int, Sequence[int], slice], value: Union[Any, Sequence[Any]]) -> None:
        """Insert data using index or slice."""
        if isinstance(key, int):
            self.insert(key, value)
        else:
            self.multi_insert(key, value)

    def __repr__(self) -> str:
        try:
            return f"{self.__class__.__name__}({len(self)})"
        except NotImplementedError:
            return f"{self.__class__.__name__}()"


class SQLiteDatabase(Database):
    """An SQLite-based key-value database implementation.

    Uses SQLite to store tensors and other data types.
    """
    def __init__(self, path: str, name: str, schema: Schema = object) -> None:
        super().__init__(schema)
        import sqlite3
        self.path = path
        self.name = name
        self._connection: Optional[sqlite3.Connection] = None
        self._cursor: Optional[sqlite3.Cursor] = None
        self.connect()

        # Create table if it does not exist
        schema_str = ", ".join(
            f"{key} BLOB NOT NULL" for key in self.schema.keys()
        )
        query = f"CREATE TABLE IF NOT EXISTS {self.name} (id INTEGER PRIMARY KEY, {schema_str})"
        self.cursor.execute(query)

    def connect(self) -> None:
        """Connect to the SQLite database."""
        import sqlite3
        self._connection = sqlite3.connect(self.path)
        self._cursor = self._connection.cursor()

    def close(self) -> None:
        """Close the database connection."""
        if self._connection is not None:
            self._connection.commit()
            self._connection.close()
            self._connection = None
            self._cursor = None

    @property
    def connection(self) -> Any:
        """Return the database connection object."""
        if self._connection is None:
            raise RuntimeError("No open database connection")
        return self._connection

    @property
    def cursor(self) -> Any:
        """Return the database cursor object."""
        if self._cursor is None:
            raise RuntimeError("No open database connection")
        return self._cursor

    def insert(self, index: int, data: Any) -> None:
        """Insert a single data entry."""
        query = f"INSERT INTO {self.name} (id, {', '.join(self.schema.keys())}) VALUES (?, {', '.join(['?'] * len(self.schema))})"
        self.cursor.execute(query, (index, *self._serialize(data)))
        self.connection.commit()

    def get(self, index: int) -> Any:
        """Retrieve a single data entry."""
        query = f"SELECT {', '.join(self.schema.keys())} FROM {self.name} WHERE id = ?"
        self.cursor.execute(query, (index,))
        row = self.cursor.fetchone()
        if row is None:
            raise KeyError(f"Index {index} not found in database")
        return self._deserialize(row)

    def __len__(self) -> int:
        """Get the total number of entries in the database."""
        query = f"SELECT COUNT(*) FROM {self.name}"
        self.cursor.execute(query)
        return self.cursor.fetchone()[0]

    def _serialize(self, data: Any) -> List[bytes]:
        """Serialize data into a byte stream."""
        return [pickle.dumps(data.get(key)) for key in self.schema.keys()]

    def _deserialize(self, row: Tuple[bytes]) -> Dict[str, Any]:
        """Deserialize a byte stream into original data."""
        return {key: pickle.loads(value) for key, value in zip(self.schema.keys(), row)}


class RocksDatabase(Database):
    """A RocksDB-based key-value database implementation.

    Uses RocksDB to store tensors and other data types.
    """
    def __init__(self, path: str, schema: Schema = object) -> None:
        super().__init__(schema)
        import rocksdict

        self.path = path
        self._db: Optional[rocksdict.Rdict] = None
        self.connect()

    def connect(self) -> None:
        """Connect to the RocksDB database."""
        import rocksdict
        self._db = rocksdict.Rdict(self.path, options=rocksdict.Options(raw_mode=True))

    def close(self) -> None:
        """Close the database connection."""
        if self._db is not None:
            self._db.close()
            self._db = None

    @property
    def db(self) -> Any:
        """Return the database object."""
        if self._db is None:
            raise RuntimeError("No open database connection")
        return self._db

    @staticmethod
    def to_key(index: int) -> bytes:
        """Convert an integer index to bytes."""
        return index.to_bytes(8, byteorder="big", signed=True)

    def insert(self, index: int, data: Any) -> None:
        """Insert a single data entry."""
        self.db[self.to_key(index)] = self._serialize(data)

    def get(self, index: int) -> Any:
        """Retrieve a single data entry."""
        return self._deserialize(self.db[self.to_key(index)])

    def _serialize(self, data: Any) -> bytes:
        """Serialize data into a byte stream."""
        buffer = io.BytesIO()
        pickle.dump(data, buffer)
        return buffer.getvalue()

    def _deserialize(self, row: bytes) -> Any:
        """Deserialize a byte stream into original data."""
        return pickle.loads(row)
