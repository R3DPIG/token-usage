"""token-usage: a plugin-based CLI for tracking LLM token usage and cost.

Stable public surface (everything else is reachable via its module):

* :mod:`token_usage.models`    -- immutable data structures.
* :mod:`token_usage.pricing`   -- pricing catalog and cost math.
* :mod:`token_usage.storage`   -- append-oriented persistence.
* :mod:`token_usage.analytics` -- aggregation and derived statistics.
* :mod:`token_usage.registry`  -- the plugin contract and registry.
* :mod:`token_usage.cli`       -- the command-line entry point.
"""

from token_usage.models import UsageRecord, ModelPricing, Budget
from token_usage.pricing import PricingCatalog, default_catalog
from token_usage.storage import JsonStore
from token_usage.registry import Plugin, PluginRegistry, registry

__all__ = [
    "UsageRecord", "ModelPricing", "Budget", "PricingCatalog", "default_catalog",
    "JsonStore", "Plugin", "PluginRegistry", "registry",
]

__version__ = "0.1.0"
