"""Minimal import utilities for graceful degradation."""

def safe_import(module_name, fallback=None):
    """Safely import a module with fallback."""
    try:
        return __import__(module_name)
    except ImportError:
        return fallback

# Optional dependency imports
numpy = safe_import('numpy')
psutil = safe_import('psutil')
torch = safe_import('torch')
transformers = safe_import('transformers')

# Simple fallback functions
def get_cpu_percent():
    """Get CPU percentage with fallback."""
    if psutil:
        return psutil.cpu_percent()
    return 50.0  # Fallback value

def get_memory_percent():
    """Get memory percentage with fallback."""
    if psutil:
        return psutil.virtual_memory().percent
    return 60.0  # Fallback value