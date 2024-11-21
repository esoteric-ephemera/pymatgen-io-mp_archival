"""Archival class definitions independent of data type."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

import h5py
import numpy as np
from numcodecs import Blosc
import zarr

if TYPE_CHECKING:
    from typing import Any


class ArchivalFormat(Enum):
    HDF5 = "h5"
    ZARR = "zarr"


@dataclass
class Archiver:
    """Mixin class to define base archival methods"""

    parsed_objects: dict[str, Any]

    metadata: dict[str, Any] | None = None
    format: ArchivalFormat | str = ArchivalFormat.HDF5
    compression: dict | None = None
    float_dtype: np.dtype = np.float32

    def __post_init__(self) -> None:
        """Ensure that attributes have correct type."""

        if isinstance(self.format, str):
            self.format = ArchivalFormat(self.format)

        if self.compression is None:
            if self.format == ArchivalFormat.HDF5:
                self.compression = {
                    "compression": 9,
                }
            elif self.format == ArchivalFormat.ZARR:
                self.compression = {
                    "compressor": Blosc(clevel=9),
                }

    def __getattr__(self, name: Any) -> Any:
        """Allow accessing parsed objects with dot notation."""
        if name.upper() in self.parsed_objects:
            return self.parsed_objects[name.upper()]
        raise AttributeError(name)

    def to_group(self, group: h5py.Group | zarr.Group, group_key: str = "group") -> None:
        """Append data to an existing HDF5-like file group."""
        raise NotImplementedError

    def to_archive(self, file_name: str = "archive") -> None:
        """Create a new archive for this class of data."""

        if len(file_split := file_name.split(".")) > 1:
            file_name = ".".join(file_split[:-1])
        file_name += f".{self.format.value}"  # type: ignore[union-attr]

        if self.format == ArchivalFormat.HDF5:
            with h5py.File(file_name, "w") as hf5:
                self.to_group(hf5)
        elif self.format == ArchivalFormat.ZARR:
            with zarr.open(file_name, "w") as zg:
                self.to_group(zg)
