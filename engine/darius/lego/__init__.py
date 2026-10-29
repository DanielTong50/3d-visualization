"""LEGO-first assembly task compiler and public instruction discovery."""

from .catalog import get_assembly, library_index
from .compiler import compile_assembly
from .schema import CompiledPlan, LegoAssembly

__all__ = ["CompiledPlan", "LegoAssembly", "compile_assembly", "get_assembly", "library_index"]
