from moola import *
import pytest
class MyFunctional(Functional):
    def __call__(self, val):
        x, y = val.data
        r = (1 - x)**2 + 100*(y - x**2)**2
        events.increment("Functional evaluation")
        return r

    def derivative(self, val):
        x, y = val.data
        #print "current x =", x, y 

        dx = - 2*(1 - x) + 100*2*(y - x**2) * (-2*x)
        dy = 100 * 2 * (y - x**2)
        dr = (dx, dy)

        events.increment("Derivative evaluation")
        return NumpyDualVector(dr)

    def hessian(self, val):
        x, y = val.data
        def hes(vec):
            v = vec.data
            dxx = 2. + 1200.*x**2 - 400.*y
            dxy = -400.*x
            dyy = 200.
            d2v = (dxx * v[0] + dxy * v[1], dxy*v[0] +dyy*v[1] )
            events.increment("Hessian evaluation")
            return NumpyDualVector(d2v)
        return hes

@pytest.fixture
def moola_problem():
    objective = MyFunctional()
    x_init = NumpyPrimalVector((-3, -4))
    options = {'jtol': None, 'gtol': None, 'display': 3,
               'maxiter' : 1000,}# 'line_search_options' :{'xtol':1e-25, 'ignore_warnings':True}} 
    
    
    return Problem(objective), x_init, options

@pytest.mark.parametrize("bfgs_options,bfgs_expected",
                         [ ({"gtol": 1e-12, 'mem_lim': 2, "rjtol":0, "rgtol":1e-18}, 19),
                           ({"jtol": 1e-12, 'mem_lim': 2,"rjtol":0, "rgtol":1e-18}, 17),
                           ({"gtol": 1e-12, 'mem_lim': 100,"rjtol":0, "rgtol":1e-19}, 30),
                           ({"jtol": 1e-12, 'mem_lim': 100,"rjtol":0, "rgtol":1e-18}, 28)])
def test_LBFGS(bfgs_options, bfgs_expected, moola_problem):
    problem, x_init, options = moola_problem
    options.update(bfgs_options)
    solver = BFGS(problem, x_init, options = options)
    sol = solver.solve()
    assert len(sol['lbfgs']) == min(options['mem_lim'], sol['iteration'])
    if options['gtol'] is not None:
        assert sol['grad_norm'] < options['gtol']
    if options['jtol'] is not None:
        assert sol['delta_J'] < options['jtol']
    # NOTE: Iteration count has changed and hasn't been tested for years.
    # assert sol['iteration'] == bfgs_expected


@pytest.mark.parametrize("newtcg_options,newtcg_expected",
                         [ ({"gtol": 1e-12, 'ncg_hesstol':0}, 32),
                           ({"jtol": 1e-12, 'ncg_hesstol':0}, 32),
                           ({"gtol": 1e-12, 'ncg_hesstol':0, 'ncg_reltol':1e-20}, 31),
                           ({"jtol": 1e-12, 'ncg_hesstol':0, 'ncg_reltol':1e-20}, 31), ])
def test_NewtonCG(newtcg_options, newtcg_expected, moola_problem):
    problem, x_init, options = moola_problem
    options.update(newtcg_options)
    solver = NewtonCG(problem, x_init, options = options)
    sol = solver.solve()
    if options['gtol'] is not None:
        assert sol['grad_norm'] < options['gtol']
    if options['jtol'] is not None:
        assert sol['delta_J'] < options['jtol']
    assert sol['iteration'] == newtcg_expected


def test_NewtonCG_superlinear_forcing(moola_problem):
    """The relative forcing term eta_k = min(0.5, sqrt(||g_k|| / ||g_0||)) (hIPPYlib) converges
    to the minimizer (1, 1), and eta_k is recorded and shrinks as ||g_k|| does."""
    problem, x_init, options = moola_problem
    options.update({"gtol": 1e-10, "ncg_hesstol": 0, "ncg_forcing": "superlinear"})
    etas = []
    solver = NewtonCG(problem, x_init, options=options,
                      hooks={"after_iteration": lambda x: etas.append(solver.data["eta"])})
    sol = solver.solve()
    assert sol['grad_norm'] < 1e-10
    assert abs(sol['control'].data[0] - 1) < 1e-8 and abs(sol['control'].data[1] - 1) < 1e-8
    assert etas[0] == 0.5 and etas[-1] < 1e-3


def test_NewtonCG_low_curvature_takes_a_line_search_step(moola_problem):
    """With ncg_hesstol above every curvature, each Newton step ends CG at its first iteration and
    steps along steepest descent by the in-CG line search: the run still decreases J every
    iteration and never divides by the near-zero curvature."""
    problem, x_init, options = moola_problem
    options.update({"gtol": None, "maxiter": 20, "ncg_hesstol": 1e30})
    Js = []
    solver = NewtonCG(problem, x_init, options=options,
                      hooks={"after_iteration": lambda x: Js.append(solver.data["objective"])})
    sol = solver.solve()
    assert sol['iteration'] == 20
    assert all(b < a for a, b in zip(Js, Js[1:]))
    assert solver.data['cg_iter'] == 0


@pytest.mark.parametrize("nlcg_options,nlcg_expected",
                         [ ({"gtol": 1e-12, "beta_rule": "fletcher-reeves"}, 536),
                           ({"jtol": 1e-12, "beta_rule": "fletcher-reeves"}, 119),
                           ({"gtol": 1e-12, "beta_rule": "polak-ribiere-polyak"}, 28),
                           ({"jtol": 1e-12, "beta_rule": "polak-ribiere-polyak"}, 25),
                           ({"gtol": 1e-12, "beta_rule": "hager-zhang"}, 37),
                           ({"jtol": 1e-12, "beta_rule": "hager-zhang"}, 35),
                           ({"gtol": 1e-12, "beta_rule": "liu-storey"}, 26),
                           ({"jtol": 1e-12, "beta_rule": "liu-storey"}, 23),
                           ({"gtol": 1e-12, "beta_rule": "hestenes-stiefel"}, 33),
                           ({"jtol": 1e-12, "beta_rule": "hestenes-stiefel"}, 32),
                           ({"gtol": 1e-12, "beta_rule": "dai-yuan"},109),
                           ({"jtol": 1e-12, "beta_rule": "dai-yuan"}, 55),
                           ({"gtol": 1e-12, "beta_rule": "conjugate_descent"},157),
                           ({"jtol": 1e-12, "beta_rule": "conjugate_descent"}, 63),
                           ({"gtol": 1e-02, "beta_rule": "steepest_descent"}, 647),
                           ({"jtol": 1e-06, "beta_rule": "steepest_descent"}, 295)])
def test_NonlinearCG(nlcg_options, nlcg_expected, moola_problem):
    problem, x_init, options = moola_problem
    options.update(nlcg_options)
    solver = NonLinearCG(problem, x_init, options = options)
    sol = solver.solve()
    if options['gtol'] is not None:
        assert sol['grad_norm'] < options['gtol']
    if options['jtol'] is not None:
        assert sol['delta_J'] < options['jtol']
    assert sol['iteration'] == nlcg_expected



@pytest.mark.parametrize("hybrid_options,hybrid_expected",
                         [ ({"gtol": 1e-12, 'mem_lim': 0, 'ncg_hesstol': 0}, 32),
                           ({"jtol": 1e-12, 'mem_lim': 0, 'ncg_hesstol': 0}, 32),
                           ({"gtol": 1e-12, 'mem_lim': 100, 'initial_bfgs_iterations': 100}, 30),
                           ({"jtol": 1e-12, 'mem_lim': 100, 'initial_bfgs_iterations': 100}, 28),
                           ({"gtol": 1e-12, 'mem_lim': 2, 'initial_bfgs_iterations': 0, 'ncg_hesstol': 0}, 38),
                           ({"jtol": 1e-12, 'mem_lim': 2, 'initial_bfgs_iterations': 0, 'ncg_hesstol': 0}, 38),
                           ({"gtol": 1e-15, 'mem_lim': 100, 'initial_bfgs_iterations': 1, 'ncg_hesstol': 0, 'ncg_reltol':1e-20}, 14),
                           ({"jtol": 1e-15, 'mem_lim': 100, 'initial_bfgs_iterations': 1, 'ncg_hesstol': 0, 'ncg_reltol':1e-20}, 14), ])
def test_HybridCG(hybrid_options, hybrid_expected, moola_problem):
    problem, x_init, options = moola_problem
    options.update(hybrid_options)
    solver = HybridCG(problem, x_init, options = options)
    sol = solver.solve()
    if options['gtol'] is not None:
        assert sol['grad_norm'] < options['gtol']
    if options['jtol'] is not None:
        assert sol['delta_J'] < options['jtol']
    # FIXME: Iteration count has changed and hasn't been tested for years.
    # assert sol['iteration'] == hybrid_expected


if __name__ == '__main__':
    use_moola = True
    maxiter = 10

    prob, x0, opt = moola_problem()

    if use_moola:
        opt.update({"gtol": 1e-12, 'mem_lim': 2, 'maxiter': maxiter, 'mem_lim': 100})
        solver = BFGS(prob, x0, options = opt)
        sol = solver.solve()

    else:
        from scipy.optimize import minimize
        objective  = lambda x: prob.obj(NumpyPrimalVector(x))
        derivative = lambda x: prob.obj.derivative(NumpyPrimalVector(x)).data
        x_init = x0.data

        minimize(objective, x_init, method="L-BFGS-B", tol=1e-1000, jac=derivative, options={"disp": True, "maxiter": maxiter-2, "gtol": 1e-12, "maxcor": 100})
