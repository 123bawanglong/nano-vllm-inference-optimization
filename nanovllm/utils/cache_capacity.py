def choose_cache_capacity(available: int, requested: int) -> int:
    """Honor an explicit experiment capacity, retaining upstream auto mode."""
    if available <= 0:
        raise RuntimeError("No KV cache blocks fit the memory budget")
    if requested > available:
        raise RuntimeError(f"Requested {requested} KV blocks exceeds available {available}")
    return requested if requested > 0 else available
