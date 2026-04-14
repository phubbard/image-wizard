"""Lazy-loaded ML models.

Each model module exposes a singleton that loads weights on first call.
Models are kept alive for the process lifetime to avoid re-loading.
"""
