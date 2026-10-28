"""The l-infinity step bound, scalar and per-DoF.

A per-DoF bound whose entries are all equal is the scalar bound, iterate for
iterate; and a per-DoF bound holds each coordinate's realized step within its
own entry, which is the point of having one -- a control made of blocks with
different units cannot share one number.
"""
import numpy as np
import pytest

from moola import BFGS, NumpyPrimalVector, Problem

from .test_rosenbrock import MyFunctional


def _iterates(bound, maxiter=25):
    xs = []
    solver = BFGS(Problem(MyFunctional()), NumpyPrimalVector((-3, -4)),
                  options={"jtol": None, "gtol": None, "display": 0,
                           "maxiter": maxiter, "max_step_linf": bound},
                  hooks={"after_iteration": lambda x: xs.append(np.array(x.data, float))})
    solver.solve()
    return np.array(xs)


def test_equal_per_dof_bounds_are_the_scalar_bound():
    a = _iterates(0.5)
    b = _iterates(np.array([0.5, 0.5]))
    assert a.shape == b.shape
    assert np.array_equal(a, b)


@pytest.mark.parametrize("bound", [np.array([0.05, np.inf]), np.array([np.inf, 0.2])])
def test_each_dof_stays_within_its_own_bound(bound):
    xs = _iterates(bound)
    steps = np.abs(np.diff(np.vstack([[-3.0, -4.0], xs]), axis=0))
    assert np.all(steps <= bound[None, :] * (1 + 1e-12))
    # the bound binds: some step is AT the bound for the bounded coordinate
    k = int(np.argmin(bound))
    assert np.isclose(steps[:, k].max(), bound[k], rtol=1e-6)
