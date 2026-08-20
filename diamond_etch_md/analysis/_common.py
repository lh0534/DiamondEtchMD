"""Shared helpers used by multiple analysis modules."""


def _phase_of_impact(impact: int, spec, ml: int) -> int:
    """Return 0-based phase index for a given impact number in a cycling simulation."""
    total_cycle = sum(p.fluence_ml for p in spec.phases) * ml
    pos = (impact - 1) % total_cycle
    cum = 0
    for pi, p in enumerate(spec.phases):
        cum += p.fluence_ml * ml
        if pos < cum:
            return pi
    return len(spec.phases) - 1
