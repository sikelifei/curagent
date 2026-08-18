"""Environment plugins, tool registration, and dataset prompt adapters."""

from .base import AgentEnvironment, EnvironmentDependencyError
from ..tools import Capabilities, Capability, CapabilityCollection
from .registry import available_environments, create_environment, register_environment
from .runner import EnvironmentRunResult, run_environment, run_registered_environment
from .trace_analysis import aggregate_trace_metrics, analyze_environment_trace

# Import built-in plugins for registration. External plugins can call
# register_environment without modifying this package.
from . import browsecomp_plus as browsecomp_plus
from . import deepdive as deepdive
from . import oolong as oolong
from . import oolong_synth as oolong_synth
from . import textcraft_synth as textcraft_synth
from . import webshop as webshop

__all__ = [
    "AgentEnvironment",
    "Capability",
    "Capabilities",
    "CapabilityCollection",
    "EnvironmentDependencyError",
    "EnvironmentRunResult",
    "available_environments",
    "aggregate_trace_metrics",
    "analyze_environment_trace",
    "create_environment",
    "register_environment",
    "run_environment",
    "run_registered_environment",
    "browsecomp_plus",
    "deepdive",
    "oolong",
    "oolong_synth",
    "textcraft_synth",
    "webshop",
]
