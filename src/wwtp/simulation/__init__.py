"""Simulation sub-package.

Exposes a :func:`get_backend` factory so callers never need to import
concrete backends directly.  This keeps the ``try/except ImportError``
for PeePyPoo in one place.

Usage
-----
    from wwtp.simulation import get_backend

    backend = get_backend()         # auto-detects PeePyPoo; falls back to synthetic
    backend = get_backend("synthetic")   # force synthetic
    backend = get_backend("peepypoo")    # force PeePyPoo (raises if not installed)

    df = backend.run(duration_h=12, q_inf_base=3200)
"""

from __future__ import annotations

from wwtp.simulation.base import SimulationBackend
from wwtp.logging_cfg import get_logger

logger = get_logger(__name__)


def get_backend(name: str = "auto") -> SimulationBackend:
    """Return a :class:`SimulationBackend` instance.

    Args:
        name: One of ``"auto"`` (default), ``"synthetic"``, or ``"peepypoo"``.
            - ``"auto"`` tries PeePyPoo first, falls back to synthetic on
              :class:`ImportError`.
            - ``"peepypoo"`` raises :class:`ImportError` if Julia is absent.
            - ``"synthetic"`` always returns the scipy ODE backend.

    Returns:
        A concrete :class:`SimulationBackend` instance ready to call
        :meth:`~SimulationBackend.run`.

    Raises:
        ImportError: When *name* is ``"peepypoo"`` and the package is absent.
        ValueError: When *name* is not a recognised option.
    """
    from wwtp.simulation.synthetic import SyntheticSimulation

    if name == "synthetic":
        return SyntheticSimulation()

    if name == "peepypoo":
        from wwtp.simulation.peepypoo import PeePyPooSimulation  # raises ImportError if absent

        return PeePyPooSimulation()

    if name == "auto":
        try:
            from wwtp.simulation.peepypoo import PeePyPooSimulation

            backend = PeePyPooSimulation()
            logger.info("Simulation backend: PeePyPoo (Julia/ASM1)")
            return backend
        except ImportError:
            logger.info("PeePyPoo not available — using synthetic scipy ODE backend")
            return SyntheticSimulation()

    raise ValueError(
        f"Unknown simulation backend {name!r}. Expected 'auto', 'synthetic', or 'peepypoo'."
    )


__all__ = ["SimulationBackend", "get_backend"]
