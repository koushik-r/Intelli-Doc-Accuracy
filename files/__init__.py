"""File (blob) storage package — Azure Blob Storage for document bytes."""

from .blob import BlobStore, get_blob_store

__all__ = ["BlobStore", "get_blob_store"]
