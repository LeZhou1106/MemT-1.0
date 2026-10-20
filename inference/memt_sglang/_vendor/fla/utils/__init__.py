"""Exports used by the copied inference closure only."""
from ._compat import autotune_cache_kwargs, find_spec_cached
from ._decorators import input_guard, tensor_cache
from ._device import (IS_AMD, IS_GATHER_SUPPORTED, IS_TF32_SUPPORTED,
    IS_NVIDIA_BLACKWELL, check_shared_mem, autocast_custom_fwd)
