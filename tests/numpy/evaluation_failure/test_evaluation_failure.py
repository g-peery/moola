"""StrongWolfeLineSearch with trial steps that raise EvaluationFailure (s227).

phi(a) = (a - 2)^2 along the line, minimizer a = 2. A failure region a > cut
stands in for a PDE solve that does not converge. The search must never return
a failed step, must keep a sufficient-decrease step it already has when it was
extrapolating past it, must backtrack when the first trial fails, and must
raise EvaluationFailure when nothing below the failures evaluates.
"""
import logging

import pytest

from moola import EvaluationFailure
from moola.linesearch import StrongWolfeLineSearch

FTOL = 1e-3


def _line(cut, fail_from_zero=False):
    calls = []

    def phi_dphi(a):
        calls.append(a)
        if fail_from_zero or a > cut:
            raise EvaluationFailure(f"no solve at {a}")
        return (a - 2.0) ** 2, 2.0 * (a - 2.0)

    return phi_dphi, calls


def _search(cut, start, fail_from_zero=False, gtol=0.9, **kw):
    phi_dphi, calls = _line(cut, fail_from_zero)
    ls = StrongWolfeLineSearch(ftol=FTOL, gtol=gtol, xtol=0.1, start_stp=start, stpmax=10.0, **kw)
    stp = ls.search(lambda a: phi_dphi(a)[0], phi_dphi, (4.0, -4.0))
    return stp, calls, ls


def _sufficient(a):
    return (a - 2.0) ** 2 <= 4.0 + FTOL * a * (-4.0)


def test_no_failure_is_unchanged():
    stp, calls, ls = _search(cut=100.0, start=1.0)
    assert ls.last_failures == 0
    assert stp == 1.0 and calls == [1.0]        # strong Wolfe holds at 1 already


def test_extrapolation_failure_keeps_the_good_step_and_searches_the_gap():
    # gtol 0.1: step 1 decreases f but is still too steep for strong Wolfe, so
    # dcsrch extrapolates (to the exact minimizer 2), which fails (cut 1.5). The
    # search keeps step 1, restarts with stpmax 1 + (2 - 1)/2 = 1.5, and ends at
    # that cap: the largest step known to evaluate, never below step 1.
    stp, calls, ls = _search(cut=1.5, start=1.0, gtol=0.1)
    assert calls[:2] == [1.0, 2.0]
    assert ls.last_failures == 1
    assert stp == 1.5 and _sufficient(stp)
    assert calls.count(1.0) == 1                # the restart re-used step 1 from the cache


def test_first_trial_failure_backtracks():
    stp, calls, ls = _search(cut=0.5, start=1.0)
    assert calls[0] == 1.0 and ls.last_failures >= 1
    assert 0 < stp <= 0.5 and _sufficient(stp)


def test_every_trial_failing_raises():
    with pytest.raises(EvaluationFailure, match="no step"):
        _search(cut=0.0, start=1.0, fail_from_zero=True)


def test_failures_are_logged(caplog):
    with caplog.at_level(logging.WARNING, logger="moola"):
        _search(cut=0.5, start=1.0)
    text = caplog.text
    assert "evaluation failed at step 1" in text
    assert "restarting with stpmax" in text
    assert "accepted step" in text and "failed evaluation(s)" in text
