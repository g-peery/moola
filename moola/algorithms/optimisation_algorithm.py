from moola.linesearch import FixedLineSearch
from moola.linesearch import ArmijoLineSearch
from moola.linesearch import StrongWolfeLineSearch
from moola.linesearch import HagerZhangLineSearch
from moola.problem import Solution
import logging

class OptimisationAlgorithm(object):
    '''
    An abstract implementation of an optimisation algorithm.
    All implementations of optimisation algorithms should use this is a base class.
    '''
    def __init__(self, tol, options={}, **args):
        ''' Initialises the optmization algorithm. '''
        raise NotImplementedError('OptimisationAlgorithm.__init__ is not implemented')

    def __str__(self):
        ''' Prints out a description of the algorithm settings. '''
        raise NotImplementedError('OptimisationAlgorithm.__str__ is not implemented')


    @classmethod
    def default_options(cls):
        # this should set any parameter that all algorithms needs to know.
        default = {'display': 10,
                   'maxiter': 100,
                   # Number of CONSECUTIVE iterations that must fail the
                   # functional-decrease test (jtol / rjtol) before the run is
                   # declared converged. 1 reproduces the historical behaviour
                   # exactly: because delta_J = oldJ - J is signed, a single
                   # iteration that fails to decrease J -- one unlucky line
                   # search -- ends the run there and then. Raising this lets a
                   # run ride out a stalled step and distinguishes a genuine
                   # plateau from a transient, at the cost of that many extra
                   # iterations before stopping.
                   'delta_J_patience': 1,}
        return default

    def set_options(self, user_options):
        self.logger = logging.getLogger("moola")
        # Update options with provided dictionary.
        if not isinstance(user_options, dict):
            raise TypeError("Options have to be set with a dictionary object.")
        if hasattr(self,  'options'):
            options = self.options
        else:
            options = self.default_options()
        for key, val in user_options.items():
            if key not in options:
                raise KeyError("'{}' not a valid setting for {}".format(key, self.__class__.__name__))
            # TODO: check also that the provided value is admissible.
            options[key] = val
        self.options = options
        return options

    def display(self, text, level):
        '''
        Function for providing information to the user.
        Only prints text if level is not smaller than the treshold setting.

        Tentative level structuring:
          0: no output (user may have provided a callback function to store or print data)
          1: start / end of optimization loop
          2: general information for each optimization loop, e.g. iterations number, gradient norm etc
          3: specific information for each optimization loop, e.g. inner iteration loop information
           :
           :
         10: debug
        '''
        if level <= self.options['display']:
            self.logger.info(text)

    def do_linesearch(self, obj, m, s, prev=None):
        ''' Performs a linesearch on obj starting from m in direction s. '''
        self.logger.debug("Beginning line-search.")

        m_new = m.copy()

        def update_m_new(alpha):
            if update_m_new.alpha_new != alpha:
                m_new.assign(m)
                m_new.axpy(alpha, s)
                update_m_new.alpha_new = alpha
        update_m_new.alpha_new = 0

        # Define the real-valued functions in the s-direction
        def phi(alpha):
            update_m_new(alpha)

            return obj(m_new)

        def phi_dphi(alpha):
            update_m_new(alpha)

            p = obj(m_new)
            djs = obj.derivative(m_new).apply(s)
            return p, djs

        #get values at current point
        if prev is None:
            phi_dphi0 = phi_dphi(0)
        else:
            p = prev[0]
            djs = prev[1].apply(s)
            phi_dphi0 = (p, djs)

        # Perform the line search
        alpha = self.linesearch.search(phi, phi_dphi, phi_dphi0)

        self.logger.debug(f"Line search complete. Stepping size (alpha): {alpha:.3f}")
        update_m_new(alpha)
        return m_new, float(alpha)

    def check_convergence(self):
        '''
        Unified convergence test for the implemented optimization algorithms.
        '''
        data = self.data
        options = self.options
        status = 0
        if data['iteration'] >= self.options['maxiter']:
            status = -1
        if 'gtol' in options and options['gtol'] != None and 'grad_norm' in data:
            if data['grad_norm'] < options['gtol']:
                status = 1
        if 'rgtol' in options and options['rgtol'] != None and 'grad_norm' in data and 'initial_grad_norm' in data:
            if data['grad_norm'] < options['rgtol'] * data['initial_grad_norm']:
                status = 1
        j_stalled = False
        if 'jtol' in options and options['jtol'] != None and 'delta_J' in data:
            if data['delta_J'] < options['jtol']:
                j_stalled = True
        if 'rjtol' in options and options['rjtol'] != None and 'delta_J' in data and 'initial_J' in data:
            if data['delta_J'] < options['rjtol'] * data['initial_J']:
                j_stalled = True
        # Require delta_J_patience CONSECUTIVE stalled iterations. The counter
        # resets on any iteration that makes the required progress, so this
        # tests for a plateau rather than for a single bad step.
        if 'delta_J' in data:
            patience = options.get('delta_J_patience', 1) or 1
            if j_stalled:
                data['delta_J_strikes'] = data.get('delta_J_strikes', 0) + 1
                if data['delta_J_strikes'] >= patience:
                    status = 2
            else:
                data['delta_J_strikes'] = 0
        # TODO: implement more tests
        data['status'] = status
        return status

    @property
    def iter_status(self):
        keys = ['iteration', 'objective', 'grad_norm', 'delta_J', 'delta_x']
        return '\t'.join(['{} = {}:'.format(k, self.data[k]) for k in keys if k in self.data])

    @property
    def convergence_status(self):
        reasons = {-1: '\nMaximum number of iterations reached.\n',
                    2: '\nTolerance reached: delta_j < jtol.\n',
                    1: '\nTolerance reached: grad_norm < gtol.\n',
                   -4: 'Linesearch failed.',
                   -5: 'Algorithm breakdown.'}
        return reasons[self.data['status']]


    def solve(self):
        ''' Solves the optimisation problem.
            Arguments:
             * problem: The optimisation problem.

            Return value:
              * solution: The solution to the optimisation problem
        '''
        raise NotImplementedError('OptimisationAlgorithm.solve is not implemented')

    def update(self, d=None, **args):
        if not hasattr(self, 'data'):
            self.data = {}
        if d is not None and hasattr(d, 'keys'):
            self.data.update(d)
        else:
            self.data.update(**args)
    def record_progress(self, **args):
        if args == {}:
            args = self.options['record']
        if not hasattr(self, 'history'):
            # then initialize it
            self.history = {arg: [] for arg in args}
        for arg in args:
            self.history[arg].append(self.data[arg])



def get_line_search_method(line_search, line_search_options):
    ''' Takes a name of a line search method and returns its Python implementation as a function. '''
    if line_search == "strong_wolfe":
        ls = StrongWolfeLineSearch(**line_search_options)
    elif line_search == "approximate_wolfe":
        ls = HagerZhangLineSearch(**line_search_options)
    elif line_search == "armijo":
        ls = ArmijoLineSearch(**line_search_options)
    elif line_search == "fixed":
        ls = FixedLineSearch(**line_search_options)
    else:
        raise ValueError("Unknown line search specified. Valid values are 'armijo', 'strong_wolfe', 'approximate_wolfe' and 'fixed'.")

    return ls


