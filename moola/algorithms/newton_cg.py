from .optimisation_algorithm import *
from .bfgs import LimitedMemoryInverseHessian, LinearOperator
from numpy import sqrt
from ..linesearch import EvaluationFailure


def dual_to_primal(x):
    return x.primal()

class NewtonCG(OptimisationAlgorithm):
    '''
    An inexact newton method.
    '''
    __name__ = 'NewtonCG'
    def __init__(self, problem, initial_point = None, precond=LinearOperator(dual_to_primal), options = {}, hooks={}):
        '''
        Initialises the Hybrid CG method. The valid options are:
         * options: A dictionary containing additional options for the steepest descent algorithm. Valid options are:
            - tol: Not supported yet - must be None.
            - maxiter: Maximum number of iterations before the algorithm terminates. Default: 200.
            - disp: dis/enable outputs to screen during the optimisation. Default: True
            - gtol: Gradient norm stopping tolerance: ||grad j|| < gtol.
            - line_search: defines the line search algorithm to use. Default: strong_wolfe
            - line_search_options: additional options for the line search algorithm. The specific options read the help
              for the line search algorithm.
            - an optional callback method which is called after every optimisation iteration.
            - ncg_forcing: the forcing term eta_k (CG stops when ||r||_B <= eta_k ||g_k||_B).
              "quadratic" (default, historical): eta_k = min(ncg_reltol, ||g_k||_B), absolute in
              ||g_k||. "superlinear": eta_k = min(ncg_reltol, sqrt(||g_k||_B / ncg_gref)), the
              relative Eisenstat-Walker choice of hIPPYlib (Villa et al. 2021, Alg. 1), which does
              not depend on the scale of the gradient.
            - ncg_gref: the reference gradient norm of the superlinear forcing term. None (default)
              uses the norm at the initial point.
         * hooks: A dictionariy containing user-defined "hook" functions that are called at certain events during the optimisation.
            - after_iteration: Is called after each each iteration.
          '''

        # Set the default options values
        self.hooks = hooks
        self.problem = problem
        self.set_options(options)
        self.linesearch = get_line_search_method(self.options['line_search'], self.options['line_search_options'])
        self.data = {'control'   : initial_point,
                     'iteration' : 0,
                     'precond'   : precond }

    def __str__(self):
        s = "Newton CG method.\n"
        s += "-"*30 + "\n"
        s += "Line search:\t\t %s\n" % self.options['line_search']
        s += "Maximum iterations:\t %i\n" % self.options['maxiter']
        return s

    # set default parameters

    @classmethod
    def default_options(cls):
        # this is defined as a function to prevent defaults from being changed at runtime.
        default = OptimisationAlgorithm.default_options()
        default.update(
            # generic parameters:
            {"jtol"                   : None,
             "gtol"                   : 1e-4,
             "maxiter"                :  200,
             "display"                :    2,
             "line_search"            : "strong_wolfe",
             "line_search_options"    : {"start_stp": 1},
             "callback"               : None,
             "record"                 : ("grad_norm", "objective"),

             # method specific parameters:
             "ncg_reltol"             :  .5,
             "ncg_maxiter"            : 200,
             "ncg_hesstol"            : "default",
             "ncg_forcing"            : "quadratic",
             "ncg_gref"               : None,
             })
        return default

    def solve(self):
        '''
            Arguments:
             * problem: The optimisation problem.

            Return value:
              * solution: The solution to the optimisation problem
         '''
        self.display( self.__str__(), 1)

        objective = self.problem.obj
        options = self.options


        B = self.data['precond']
        x = self.data['control']
        i = self.data['iteration']

        # compute initial objective and gradient
        J = objective(x)
        dJ = objective.derivative(x)

        r = dJ.copy()  # initial residual ( with dk = 0)
        r.scale(-1.)

        self.update({'objective' : J,
                     'grad_norm' : r.primal_norm()})
        self.record_progress()

        gref = options['ncg_gref']
        if gref is None:
            gref = self.data['grad_norm']
        gref = float(gref)
        if not gref > 0:
            raise ValueError(f"ncg_gref must be positive, got {gref}")

        if options['ncg_hesstol'] == "default":
            import numpy
            eps = numpy.finfo(numpy.float64).eps
            ncg_hesstol = eps*numpy.sqrt(len(x))
        elif options['ncg_hesstol'] == "adaptive":
            import numpy
            eps = numpy.finfo(numpy.float64).eps
            ncg_hesstol = eps*r.apply(B*r)
        else:
            ncg_hesstol = options['ncg_hesstol']

        # Start the optimisation loop
        while self.check_convergence() == 0:
            self.display(self.iter_status, 2)
            # p = current CG search direction
            p = Br = (B * r) # mapping residual to primal space
            # Steepest-descent direction -B g, kept apart from p and Br, which CG
            # overwrites (p is scaled in place; Br is rebound to B r_cg).
            sd = Br.copy()
            # d = Newton search direction
            d = p.copy().zero()
            rBr = r.apply(Br)
            H = objective.hessian(x)


            # CG iterations. Stop when r^T B r <= eta_k^2 * g^T B g (the residual in the
            # preconditioner's norm, which is the gradient norm moola reports).
            if options['ncg_forcing'] == "quadratic":
                eta = min(options['ncg_reltol'], sqrt(rBr))
            elif options['ncg_forcing'] == "superlinear":
                eta = min(options['ncg_reltol'], sqrt(sqrt(rBr) / gref))
            else:
                raise ValueError(f"Unknown ncg_forcing {options['ncg_forcing']!r}; "
                                 "use 'quadratic' or 'superlinear'")
            cg_tol = eta**2 * rBr
            cg_iter  = 0
            cg_break = 0
            low_curvature_vals = None # Sentinel
            while cg_iter < options['ncg_maxiter'] and rBr >= cg_tol:
                self.display(f"rBr = {rBr}\ttolerance = {cg_tol}", 3)

                self.display("Forming product Hp", 3)
                Hp  = H(p)
                self.display("Computing curvature <Hp, p> by duality pairing", 3)
                pHp = Hp.apply(p)

                self.display('cg_iter = {}\tcurve = {}\thesstol = {}'.format(cg_iter, pHp, ncg_hesstol), 3)
                if pHp < 0:
                    if cg_iter == 0:
                        self.display(
                            "Curvature negative on first iteration of CG. "
                            "Falling back to steppest descent.",
                            3
                        )

                        # Fall back to steepest descent
                        d = sd

                    # otherwise use the last computed pk
                    self.display(
                        "Curvature negative. "
                        f"Stopping CG at iteration {cg_iter}",
                        3
                    )
                    break

                if 0 <= pHp < ncg_hesstol:
                    # Curvature along p is too small to trust a CG step (alpha = rBr / pHp
                    # would be huge). End CG and take a step along the direction already in
                    # hand -- the CG iterate d, or steepest descent if CG has not moved yet --
                    # by a line search here, whose result replaces the outer one.
                    self.display("Curvature positive but within ncg_hesstol; stopping CG "
                                 f"at iteration {cg_iter}.", 3)
                    if cg_iter == 0:
                        d = sd
                    try:
                        low_curvature_vals = self.do_linesearch(objective, x, d, prev=(J, dJ))
                        self.display("Low-curvature line search found a step.", 3)
                    except (RuntimeError, Warning, EvaluationFailure) as exc:
                        if cg_iter == 0:
                            raise
                        # The CG iterate may not be a good direction; steepest descent
                        # always is, so try it once before giving up.
                        self.display(f"Low-curvature line search along the CG iterate failed "
                                     f"({exc}); retrying along steepest descent.", 3)
                        d = sd
                        low_curvature_vals = self.do_linesearch(objective, x, d, prev=(J, dJ))
                    break

                # Standard CG iterations
                alpha = rBr / pHp

                # Update Newton search direction
                d.axpy(alpha, p)            # update cg iterate

                # Implicit update to residual
                r.axpy(-alpha, Hp)          # update residual

                Br = B*r
                t = r.apply(Br)
                rBr, beta = t, t / rBr,

                p.scale(beta)
                p.axpy(1., Br)

                cg_iter +=1


            # do a line search and update
            self.display("CG completed.", 3)
            if low_curvature_vals is None:
                self.display("Now performing line search.", 3)
                x, a = self.do_linesearch(objective, x, d, prev=(J, dJ))
            else:
                self.display("Reusing previous line search", 3)
                x, a = low_curvature_vals

            J, oldJ = objective(x), J

            # evaluate gradient at the new point
            dJ = objective.derivative(x)

            r = dJ.copy() 
            r.scale(-1)

            i += 1

            if options['callback'] is not None:
                options['callback'](J, r)

            # store current iteration variables
            self.update({'iteration' : i,
                         'control'   : x,
                         'grad_norm' : r.primal_norm(),
                         'delta_J'   : oldJ-J,
                         'objective' : J,
                         'cg_iter'   : cg_iter,
                         'eta'       : eta,
                         'lbfgs'     : B })
            self.record_progress()

            if "after_iteration" in self.hooks:
                self.hooks["after_iteration"](x)

        self.display(self.convergence_status, 1)
        self.display(self.iter_status, 1)
        return self.data
