"""
Vercel serverless entry point for Data Drift Detection API.
Exports the FastAPI app instance for Vercel Python runtime.
"""

from api.app import app

__all__ = ["app"]

