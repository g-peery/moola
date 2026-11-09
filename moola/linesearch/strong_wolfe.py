import math

from .line_search import LineSearch, EvaluationFailure
from .dcsrch import dcsrch
from numpy import zeros
from logging import error
import logging

class StrongWolfeLineSearch(LineSearch):
    def __init__(self, ftol=1e-4, gtol=0.9, xtol=1e-1, start_stp=1.0, stpmin = None, stpmax="automatic", verify=False, ignore_warnings=True,
                 max_failures=6, failure_gap_fraction=0.5):
        '''
        This class implements a line search algorithm whose steps
        satisfy the strong Wolfe conditions (i.e. they satisfies a
        sufficient decrease condition and a curvature condition).

        The algorithm is designed to find a step 'stp' that satisfies
        the sufficient decrease condition

               f(stp) <= f(0) + ftol*stp*f'(0),

        and the curvature condition

               abs(f'(stp)) <= gtol*abs(f'(0)).

        If ftol is less than gtol and if, for example, the function
        is bounded below, then there is always a step which satisfies
        both conditions.

        If no step can be found that satisfies both conditions, then
        the algorithm stops with a warning. In this case stp only
        satisfies the sufficient decrease condition.

        The function arguments are:

           ftol           | a nonnegative tolerance for the sufficient decrease condition.
           gtol           | a nonnegative tolerance for the curvature condition.
           xtol           | a nonnegative relative tolerance for an acceptable step.
           start_stp      | a guess for an initial step size.
           stpmin         | a nonnegative lower bound for the step.
           stpmax         | a nonnegative upper bound for the step.
           verify         | if True, the step is compared to the Fortran implementation
                          | (note: this assumes that the Python module dcsrch_fortran is compiled
                          |  and in the PYTHONPATH)
          ignore_warnings | Continue if the line search ends with a warnings (e.g. stp = stpmax).
                          | Default: True
          max_failures    | how many trial steps may raise EvaluationFailure in one search
                          | before it gives up (see below). Default: 6
     failure_gap_fraction | where the next trial goes in the gap between the largest
                          | sufficient-decrease step and the smallest failed step. Default: 0.5

        Failed evaluations. A trial step whose evaluation raises EvaluationFailure
        (e.g. a PDE solve that did not converge) is not a function value, so it is
        never handed to dcsrch. The search records a_fail, the smallest failed step,
        and restarts dcsrch from step 0 with stpmax below a_fail:
          - if an evaluated step a_ok < a_fail satisfies sufficient decrease (e.g. the
            search was extrapolating past it), stpmax = a_ok + failure_gap_fraction *
            (a_fail - a_ok): the search keeps a_ok and looks in the gap above it;
          - otherwise stpmax is the largest evaluated step below a_fail (dcsrch then
            interpolates down from it), or a_fail / 2 if there is none.
        Evaluated steps are cached, so a restart re-uses them without new
        evaluations. Ending at stpmax with sufficient decrease is the usual
        "stp = stpmax" warning: the largest step known to evaluate. The search gives
        up after max_failures failures or when a_fail - a_ok <= xtol * a_fail: it
        returns a_ok if there is one, and otherwise raises EvaluationFailure. Every
        returned step was evaluated without failure. The number of failures of the
        last search is kept in self.last_failures.

        References:
         Mor'e, J.J., and Thuente, D.J., 1992, Line search algorithms with guaranteed
              sufficient decrease: Preprint MCS-P330-1092, Argonne National Laboratory.
         Averick, B.M., and Mor'e, J.J., 1993, FORTRAN subroutines dcstep and dcsrch
              from MINPACK-2, 1993, Argonne National Laboratory and University of Minnesota.
        '''

        self.ftol            = ftol
        self.gtol            = gtol
        self.xtol            = xtol
        self.start_stp       = start_stp
        self.stpmin          = stpmin
        self.stpmax          = stpmax
        self.verify          = verify
        self.ignore_warnings = ignore_warnings
        self.max_failures    = max_failures
        self.failure_gap_fraction = failure_gap_fraction
        self.last_failures   = 0
        self.logger = logging.getLogger("moola")

    def search(self, phi, phi_dphi, phi_dphi0):
        ''' Performs the line search on the function phi.

            phi must be a function [0, oo] -> R.
            phi_dphi must evaluate phi and its derivative, and
            must be a function [0, oo] -> (R, R).

            The return value is a step that satisfies the strong Wolfe condition.
        '''

        f0, g0 = phi_dphi0

        # Compute an estimate for the maximum step size
        if not self.stpmin:
            self.stpmin = 0.
        if self.stpmax == "automatic":
            stpmax = max(4*min(self.start_stp, 1.0), 0.1*f0/(-g0*self.ftol))
            self.logger.info(
                f"Automatic maximum step size enabled. Along the line:"
                f" -gradient = {-g0}\tmax step size = {stpmax}"
            )
        else:
            stpmax = self.stpmax

        # Steps evaluated without failure in this search, and the step on trial.
        seen = {}
        trial = [None]

        def evaluate(stp):
            if stp not in seen:
                trial[0] = stp
                seen[stp] = phi_dphi(stp)
            return seen[stp]

        self.last_failures = 0
        a_fail = math.inf
        cap = stpmax
        while True:
            try:
                stp = self._search_below(phi, phi_dphi, f0, g0, min(self.start_stp, cap), cap, evaluate)
                if self.last_failures:
                    self.logger.warning(
                        f"Line search: accepted step {stp:.6g} after {self.last_failures} failed "
                        f"evaluation(s); smallest failed step {a_fail:.6g}"
                    )
                return stp
            except EvaluationFailure as exc:
                a_fail = min(a_fail, trial[0])
                self.last_failures += 1
                below = [a for a in seen if 0 < a < a_fail]
                ok = [a for a in below if seen[a][0] <= f0 + self.ftol*a*g0]
                a_ok = max(ok, default=0.0)
                self.logger.warning(
                    f"Line search: evaluation failed at step {a_fail:.6g} ({exc}); "
                    f"failure {self.last_failures}/{self.max_failures}, "
                    f"best sufficient-decrease step {a_ok:.6g}"
                )
                if self.last_failures >= self.max_failures or (a_fail - a_ok) <= self.xtol*a_fail:
                    if a_ok > 0:
                        self.logger.warning(
                            f"Line search: giving up below the failed step; returning step {a_ok:.6g}"
                        )
                        return a_ok
                    raise EvaluationFailure(
                        f"line search found no step that evaluates and decreases the objective "
                        f"({self.last_failures} failed, smallest failed step {a_fail:.6g})"
                    ) from exc
                if a_ok > 0:
                    cap = a_ok + self.failure_gap_fraction*(a_fail - a_ok)
                    how = f"searching the gap above sufficient-decrease step {a_ok:.6g}"
                else:
                    cap = max(below, default=0.5*a_fail)
                    how = ("backtracking from evaluated step" if below else "backtracking: halving")
                self.logger.warning(f"Line search: restarting with stpmax {cap:.6g} ({how})")

    def _search_below(self, phi, phi_dphi, f, g, stp, stpmax, evaluate):
        """One dcsrch run from step 0 with the given stpmax; trial steps go
        through ``evaluate`` (cached, and may raise EvaluationFailure)."""
        # Set up the variables for dscrch
        isave = zeros(3)
        dsave = zeros(14)
        task = "START"

        while True:
            stp, task, isave, dsave = self.__csrch__(f, g, stp, task, isave, dsave, stpmax)

            if task in ("START", "FG"):
                f, g = evaluate(stp)
            else:
                break

        if "Error" in task:
            raise RuntimeError(task)
        elif "Warning" in task:
            if not self.ignore_warnings:
                raise Warning(task)
            else:
                self.logger.info(
                    "Warning in line search: %s." % task.replace("Warning: ","")
                )
                return stp
        else:
            assert task=="Convergence" or ("Warning" in task and self.ignore_warnings)

            if self.verify:
                # Recompute the step with the Fortran implementation and compare
                stp_fort = self.search_fortran(phi, phi_dphi, stpmax)
                if stp_fort is not None and stp_fort != stp:
                    raise RuntimeError("The line search verification failed!")

            return stp

    def search_fortran(self, phi, phi_dphi, stpmax):
        ''' Performs the line search on the function phi using the Fortran implementation
            of the line search algorithm.

            phi must be a function [0, oo] -> R.
            phi_dphi must evaluate phi and its derivative, and
            must be a function [0, oo] -> (R, R).

            The return value is a step that satisfies the strong Wolfe condition.
        '''

        try:
            import pyswolfe
            dphi = lambda x: phi_dphi(x)[1]
            ls_fort = pyswolfe.StrongWolfeLineSearch(phi(0), dphi(0), 1.0, phi, dphi, gtol=self.gtol,
                                                     xtol=self.xtol, ftol=self.ftol, stp=self.start_stp,
                                                     stpmin=self.stpmin, stpmax=stpmax)
            ls_fort.search()
            return ls_fort.stp

        except ImportError:
            error("The line search could not be verified. Did you compile the pyswolfe Fortran module?")

    def __csrch__(self, f, g, stp, task, isave, dsave, stpmax):
        self.logger.debug(
            "Line search subroutine inputs: "
            f"stp={stp:.6f}\tf={f:.6f}\tg={g:.6f}"
            f"\tftol={self.ftol:.6f}\tgtol={self.gtol:.6f}\txtol={self.xtol:.6f}"
            f"\ttask={task}\tstpmin={self.stpmin:.6f}\tstpmax={stpmax:.6f}"
        )
        stp, task, isave, dsave = dcsrch(stp, f, g, self.ftol, self.gtol, self.xtol, task, self.stpmin, stpmax, isave, dsave)
        self.logger.debug(
            f"Line search subroutine outputs: stp={stp:.6f}\ttask={task}"
        )
        return stp, task, isave, dsave

