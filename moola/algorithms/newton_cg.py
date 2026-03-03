from .optimisation_algorithm import *
from .bfgs import LimitedMemoryInverseHessian, LinearOperator
from numpy import sqrt


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
            # d = Newton search direction
            d = p.copy().zero()
            rBr = r.apply(Br)
            H = objective.hessian(x)


            # CG iterations
            cg_tol =  min(options['ncg_reltol']**2, rBr)*rBr
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
                        d = Br

                    # otherwise use the last computed pk
                    self.display(
                        "Curvature negative. "
                        f"Stopping CG at iteration {cg_iter}",
                        3
                    )
                    break

                if 0 <= pHp < ncg_hesstol:
                    self.display("Curvature positive but within ncg_hesstol.", 3)

                    if cg_iter == 0:
                        self.display("Low curvature occurred on 1st CG iteration, so falling back to line search.", 3)
                        # Fall back to steepest descent
                        d = Br
                    
                    # try to use what we have
                    try:
                        low_curvature_vals = self.do_linesearch(objective, x, d, prev=(J, dJ)) #TODO: fix this hack

                        self.display("Linesearch failed to find a better point. Stopping CG.", 3)

                        break
                    except:
                        pass

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
                         'lbfgs'     : B })
            self.record_progress()

            if "after_iteration" in self.hooks:
                self.hooks["after_iteration"](x)

        self.display(self.convergence_status, 1)
        self.display(self.iter_status, 1)
        return self.data
