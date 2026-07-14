"""Datasource package — MongoDB connection, collections, and index setup."""

from .datasource import Collections, collections, ensure_indexes, get_db

__all__ = ["Collections", "collections", "ensure_indexes", "get_db"]
