"""JAX-resident moola vector adaptors for use with the L2 Riesz inner product.

This module is **import-guarded**: importing :mod:`moola` without JAX installed
will not raise because this file is imported lazily (only when JAX is present).

Classes
-------
JaxPrimalVector
    Primal-space vector backed by a ``jnp.ndarray``.
JaxDualVector
    Dual-space vector backed by a ``jnp.ndarray``.
JaxAllAtOnceVector
    Block-diagonal vector over multiple control spaces.
JaxFunctional
    Wraps callable ``J`` and gradient ``grad_J`` for use with moola optimisers.
"""
from __future__ import annotations

from math import sqrt
from typing import Callable

import jax.numpy as jnp

from moola.linalg.vector import Vector
from moola.problem.functional import Functional
from tfm_cardiax.riesz import JaxRieszMap


class JaxPrimalVector(Vector):
    """Primal vector backed by a jnp.ndarray with L2-Riesz inner product.

    Parameters
    ----------
    data : jnp.ndarray
        Flat coefficient array.
    riesz_map : JaxRieszMap
        Assembled mass-matrix Riesz map for the underlying control space.
    """

    def __init__(self, data: jnp.ndarray, riesz_map: JaxRieszMap):
        self._data = jnp.asarray(data, dtype=jnp.float64)
        self._riesz = riesz_map

    # ------------------------------------------------------------------
    # Primal / dual conversion
    # ------------------------------------------------------------------

    def primal(self) -> "JaxPrimalVector":
        """Return self (already the primal representation)."""
        return self

    def dual(self) -> "JaxDualVector":
        """Return the dual embedding M x."""
        return JaxDualVector(self._riesz.apply(self._data), self._riesz)

    # ------------------------------------------------------------------
    # Inner product and norm
    # ------------------------------------------------------------------

    def inner(self, other: "JaxPrimalVector") -> float:
        """Compute the Riesz inner product <self, other>_M = self^T M other."""
        return float(jnp.vdot(self._data, self._riesz.apply(other._data)))

    def norm(self) -> float:
        """Compute the Riesz norm sqrt(<self, self>_M)."""
        return sqrt(self.inner(self))

    primal_norm = norm

    # ------------------------------------------------------------------
    # Vector arithmetic (required by moola algorithms)
    # ------------------------------------------------------------------

    def axpy(self, a, x: "JaxPrimalVector") -> "JaxPrimalVector":
        """In-place self += a * x."""
        self._data = self._data + a * x._data
        return self

    def scale(self, a) -> "JaxPrimalVector":
        """In-place self *= a."""
        self._data = a * self._data
        return self

    def zero(self) -> "JaxPrimalVector":
        """Zero out the vector in-place."""
        self._data = jnp.zeros_like(self._data)
        return self

    def copy(self) -> "JaxPrimalVector":
        """Return a deep copy."""
        return JaxPrimalVector(self._data, self._riesz)

    def assign(self, other: "JaxPrimalVector") -> "JaxPrimalVector":
        """Copy values from *other* into self."""
        self._data = other._data
        return self

    # ------------------------------------------------------------------
    # Array access (required by moola algorithms)
    # ------------------------------------------------------------------

    def array(self) -> jnp.ndarray:
        """Return the underlying jnp array."""
        return self._data

    def set(self, arr, local: bool = True) -> None:
        """Set the vector to *arr*."""
        self._data = jnp.asarray(arr, dtype=jnp.float64)

    # ------------------------------------------------------------------
    # Size (required by moola.linalg.Vector ABC)
    # ------------------------------------------------------------------

    def local_size(self) -> int:
        return int(self._data.size)

    def size(self) -> int:
        return int(self._data.size)

    def has_petsc_support(self) -> bool:
        return False

    # ------------------------------------------------------------------
    # Operator overloads (default impls in Vector use axpy/scale/copy)
    # ------------------------------------------------------------------

    def __getitem__(self, index):
        return float(self._data[index])

    def __setitem__(self, index, value):
        # JAX arrays are immutable; build a new one via index_update
        import numpy as np
        arr = np.asarray(self._data)
        arr[index] = value
        self._data = jnp.asarray(arr, dtype=jnp.float64)


class JaxDualVector(Vector):
    """Dual vector backed by a jnp.ndarray, paired with a JaxRieszMap.

    Parameters
    ----------
    data : jnp.ndarray
        Coefficient array in the dual space.
    riesz_map : JaxRieszMap
        Assembled mass-matrix Riesz map for the underlying control space.
    """

    def __init__(self, data: jnp.ndarray, riesz_map: JaxRieszMap):
        self._data = jnp.asarray(data, dtype=jnp.float64)
        self._riesz = riesz_map

    # ------------------------------------------------------------------
    # Primal / dual conversion
    # ------------------------------------------------------------------

    def primal(self) -> JaxPrimalVector:
        """Return M^{-1} self as a JaxPrimalVector."""
        return JaxPrimalVector(self._riesz.solve(self._data), self._riesz)

    def dual(self) -> "JaxDualVector":
        """Return self (already the dual representation)."""
        return self

    # ------------------------------------------------------------------
    # Duality pairing
    # ------------------------------------------------------------------

    def apply(self, primal: JaxPrimalVector) -> float:
        """Compute the natural duality pairing <self, primal>."""
        return float(jnp.vdot(self._data, primal._data))

    # ------------------------------------------------------------------
    # Inner product and norm
    # ------------------------------------------------------------------

    def inner(self, other: "JaxDualVector") -> float:
        """Compute the dual inner product <self, other>_{M^{-1}} = self^T M^{-1} other."""
        return float(jnp.vdot(self._data, self._riesz.solve(other._data)))

    def norm(self) -> float:
        """Compute the dual norm sqrt(<self, self>_{M^{-1}})."""
        return sqrt(self.inner(self))

    def primal_norm(self) -> float:
        """Compute the norm of the corresponding primal vector."""
        return self.primal().norm()

    # ------------------------------------------------------------------
    # Vector arithmetic
    # ------------------------------------------------------------------

    def axpy(self, a, x: "JaxDualVector") -> "JaxDualVector":
        """In-place self += a * x."""
        self._data = self._data + a * x._data
        return self

    def scale(self, a) -> "JaxDualVector":
        """In-place self *= a."""
        self._data = a * self._data
        return self

    def zero(self) -> "JaxDualVector":
        """Zero out the vector in-place."""
        self._data = jnp.zeros_like(self._data)
        return self

    def copy(self) -> "JaxDualVector":
        """Return a deep copy."""
        return JaxDualVector(self._data, self._riesz)

    def assign(self, other: "JaxDualVector") -> "JaxDualVector":
        """Copy values from *other* into self."""
        self._data = other._data
        return self

    # ------------------------------------------------------------------
    # Operator overloads (needed for BFGS two-loop recursion)
    # ------------------------------------------------------------------

    def __add__(self, other: "JaxDualVector") -> "JaxDualVector":
        return JaxDualVector(self._data + other._data, self._riesz)

    def __sub__(self, other: "JaxDualVector") -> "JaxDualVector":
        return JaxDualVector(self._data - other._data, self._riesz)

    def __mul__(self, a) -> "JaxDualVector":
        return JaxDualVector(a * self._data, self._riesz)

    __rmul__ = __mul__

    def __neg__(self) -> "JaxDualVector":
        return JaxDualVector(-self._data, self._riesz)

    # ------------------------------------------------------------------
    # Array access
    # ------------------------------------------------------------------

    def array(self) -> jnp.ndarray:
        """Return the underlying jnp array."""
        return self._data

    def set(self, arr, local: bool = True) -> None:
        self._data = jnp.asarray(arr, dtype=jnp.float64)

    def local_size(self) -> int:
        return int(self._data.size)

    def size(self) -> int:
        return int(self._data.size)

    def has_petsc_support(self) -> bool:
        return False

    def __getitem__(self, index):
        return float(self._data[index])

    def __setitem__(self, index, value):
        import numpy as np
        arr = np.asarray(self._data)
        arr[index] = value
        self._data = jnp.asarray(arr, dtype=jnp.float64)


class JaxAllAtOnceVector(Vector):
    """Block-diagonal vector over multiple named control spaces.

    Each block carries its own :class:`JaxRieszMap`; cross-block inner
    products are zero (block-diagonal Riesz map).

    Parameters
    ----------
    blocks : dict[str, JaxPrimalVector | JaxDualVector]
        Ordered mapping from control name to the corresponding block vector.
    """

    def __init__(self, blocks: dict):
        self._blocks = dict(blocks)

    # ------------------------------------------------------------------
    # Primal / dual conversion
    # ------------------------------------------------------------------

    def primal(self) -> "JaxAllAtOnceVector":
        """Return primal representation of every block."""
        return JaxAllAtOnceVector({k: v.primal() for k, v in self._blocks.items()})

    def dual(self) -> "JaxAllAtOnceVector":
        """Return dual representation of every block."""
        return JaxAllAtOnceVector({k: v.dual() for k, v in self._blocks.items()})

    # ------------------------------------------------------------------
    # Inner product
    # ------------------------------------------------------------------

    def inner(self, other: "JaxAllAtOnceVector") -> float:
        """Sum of block-wise inner products."""
        return sum(self._blocks[k].inner(other._blocks[k]) for k in self._blocks)

    def norm(self) -> float:
        return sqrt(self.inner(self))

    def primal_norm(self) -> float:
        return self.primal().norm()

    # ------------------------------------------------------------------
    # Arithmetic
    # ------------------------------------------------------------------

    def axpy(self, a, x: "JaxAllAtOnceVector") -> "JaxAllAtOnceVector":
        for k in self._blocks:
            self._blocks[k].axpy(a, x._blocks[k])
        return self

    def scale(self, a) -> "JaxAllAtOnceVector":
        for k in self._blocks:
            self._blocks[k].scale(a)
        return self

    def zero(self) -> "JaxAllAtOnceVector":
        for v in self._blocks.values():
            v.zero()
        return self

    def copy(self) -> "JaxAllAtOnceVector":
        return JaxAllAtOnceVector({k: v.copy() for k, v in self._blocks.items()})

    def assign(self, other: "JaxAllAtOnceVector") -> "JaxAllAtOnceVector":
        for k in self._blocks:
            self._blocks[k].assign(other._blocks[k])
        return self

    # ------------------------------------------------------------------
    # Operator overloads
    # ------------------------------------------------------------------

    def __add__(self, other: "JaxAllAtOnceVector") -> "JaxAllAtOnceVector":
        return JaxAllAtOnceVector({k: self._blocks[k] + other._blocks[k]
                                   for k in self._blocks})

    def __sub__(self, other: "JaxAllAtOnceVector") -> "JaxAllAtOnceVector":
        return JaxAllAtOnceVector({k: self._blocks[k] - other._blocks[k]
                                   for k in self._blocks})

    def __mul__(self, a) -> "JaxAllAtOnceVector":
        return JaxAllAtOnceVector({k: self._blocks[k] * a for k in self._blocks})

    __rmul__ = __mul__

    def __neg__(self) -> "JaxAllAtOnceVector":
        return JaxAllAtOnceVector({k: -self._blocks[k] for k in self._blocks})

    # ------------------------------------------------------------------
    # Size / array
    # ------------------------------------------------------------------

    def local_size(self) -> int:
        return sum(v.local_size() for v in self._blocks.values())

    def size(self) -> int:
        return self.local_size()

    def has_petsc_support(self) -> bool:
        return False

    def array(self):
        """Return a concatenated numpy array of all blocks (for diagnostics)."""
        import numpy as np
        return jnp.concatenate([jnp.ravel(v.array()) for v in self._blocks.values()])

    def set(self, arr, local: bool = True) -> None:
        raise NotImplementedError("Use block-wise set for JaxAllAtOnceVector.")

    def __getitem__(self, key):
        return self._blocks[key]

    def __setitem__(self, key, value):
        self._blocks[key] = value


class JaxFunctional(Functional):
    """Wraps a JAX-callable objective and gradient for use with moola optimisers.

    The instance doubles as a *problem* object (exposes ``.obj = self``) so it
    can be passed directly to :class:`moola.algorithms.bfgs.BFGS` without
    wrapping in ``moola.Problem``.

    Parameters
    ----------
    assemble_fn : callable
        ``assemble_fn(x: jnp.ndarray) -> float`` — evaluates the objective.
    grad_fn : callable
        ``grad_fn(x: jnp.ndarray) -> jnp.ndarray`` — returns the gradient
        (a dual-space array, i.e. the derivative in the *dual* representation).
    riesz_map : JaxRieszMap
        Riesz map for the single control space.  For multi-block problems,
        use :class:`JaxAllAtOnceVector` and wire separately.
    """

    def __init__(
        self,
        assemble_fn: Callable,
        grad_fn: Callable,
        riesz_map: JaxRieszMap,
    ):
        self._asm = assemble_fn
        self._grd = grad_fn
        self._riesz = riesz_map
        # Allow passing self directly to BFGS as the "problem" argument.
        self.obj = self

    def __call__(self, x: JaxPrimalVector) -> float:
        """Evaluate the objective at *x*."""
        return float(self._asm(x.array()))

    def derivative(self, x: JaxPrimalVector) -> JaxDualVector:
        """Return the derivative (gradient in the dual space) at *x*."""
        return JaxDualVector(self._grd(x.array()), self._riesz)