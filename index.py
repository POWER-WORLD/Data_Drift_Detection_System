"""
Root entry point for Vercel Serverless Function runtime.
Exports the FastAPI app instance from api.app.
"""

from api.app import app

__all__ = ["app"]

