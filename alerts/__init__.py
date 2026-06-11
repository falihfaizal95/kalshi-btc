"""alerts package — market scanning, edge detection, and auto-trading."""
from .engine import scan_markets, compute_edge

__all__ = ["scan_markets", "compute_edge"]
