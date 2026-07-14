"""Azure Blob Storage helper for document bytes.

Uploaded documents live in a single container; Mongo stores only the reference
(container + blob name + url). The frontend previews a document by streaming it
back through the API, which pulls the bytes from here.

Settings:
  AZURE_STORAGE_CONNECTION_STRING — storage account connection string
  AZURE_STORAGE_CONTAINER         — container name (default: idp-documents)
"""

from __future__ import annotations

import os
import uuid
from functools import lru_cache

from azure.storage.blob import BlobServiceClient, ContentSettings

DEFAULT_CONTAINER = "idp-documents"


class BlobStore:
    def __init__(self, connection_string: str, container: str):
        self._service = BlobServiceClient.from_connection_string(connection_string)
        self._container = container
        self._ensure_container()

    def _ensure_container(self) -> None:
        try:
            self._service.create_container(self._container)
        except Exception:
            # Already exists (or race) — safe to ignore.
            pass

    @property
    def container(self) -> str:
        return self._container

    def upload(self, data: bytes, filename: str, content_type: str) -> tuple[str, str]:
        """Store bytes under a unique blob name. Returns (blob_name, blob_url)."""
        safe = filename.replace("/", "_").replace("\\", "_")
        blob_name = f"{uuid.uuid4().hex}/{safe}"
        client = self._service.get_blob_client(self._container, blob_name)
        client.upload_blob(
            data,
            overwrite=True,
            content_settings=ContentSettings(content_type=content_type),
        )
        return blob_name, client.url

    def download(self, blob_name: str) -> bytes:
        client = self._service.get_blob_client(self._container, blob_name)
        return client.download_blob().readall()

    def delete(self, blob_name: str) -> None:
        client = self._service.get_blob_client(self._container, blob_name)
        try:
            client.delete_blob()
        except Exception:
            pass


@lru_cache(maxsize=1)
def get_blob_store() -> BlobStore:
    conn = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    if not conn:
        raise RuntimeError(
            "AZURE_STORAGE_CONNECTION_STRING is not set. Configure it in "
            "local.settings.json / app settings."
        )
    container = os.getenv("AZURE_STORAGE_CONTAINER", DEFAULT_CONTAINER)
    return BlobStore(conn, container)
