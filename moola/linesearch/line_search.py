class EvaluationFailure(Exception):
    """Raised by an objective or gradient evaluation that has no trustworthy
    value at the requested point -- e.g. a PDE solve inside the functional did
    not converge. The strong Wolfe line search catches it at a trial step and
    searches below that step instead (see StrongWolfeLineSearch); raised
    anywhere else it propagates to the caller of the optimiser."""


class LineSearch:
    ''' A base class for a line search implementation. '''

    def search(self, phi, phi_dphi):
        ''' This method performs the line search. 

            The two passed-in arguments are functions:
               phi:  [0, oo] -> R
               phi_dphi: [0, oo] -> (R, R)
            where phi_dphi evaluates both phi and its derivative.

            For multidimensional optimisation problems min_m f(m), dim(m) >= 1, phi is typically defined as:
            phi(a) = f(m_ + a*s),
            where m_ and s are the current parameter point and search direction, respectively and a \in R.

            The return value must be the computed step size.
        '''
        raise NotImplementedError("Must be overloaded.")

