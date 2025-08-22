"""Scaling modules for BCI-GPT."""

try:
    from .auto_scaler import AutoScaler, ResourceManager, LoadBalancer, global_auto_scaler, global_load_balancer
    __all__ = ["AutoScaler", "ResourceManager", "LoadBalancer", "global_auto_scaler", "global_load_balancer"]
except ImportError:
    __all__ = []
