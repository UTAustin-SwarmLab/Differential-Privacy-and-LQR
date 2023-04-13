from configparser import Interpolation
import cvxpy as cp
import numpy as np

# from src_class import src


def ACS(Psi, src_list, rho_total, show_text=False, eta = 1e-4):
    ### c: linear combination coefficient shape(n_src x T x 1)
    ### rho: utility shape(n_src)
    ### eta: convergenve_criteria

    ### initialize c and rho
    n_src = len(src_list)
    T = Psi.shape[0]
    c_init = np.ones((n_src, T, 1)) / n_src
    rho_init = np.ones((n_src)) / n_src * rho_total
    assert np.isclose( np.sum(rho_init), rho_total ), "Sum of initial rho is not rho_total!"
    assert np.allclose( np.sum(c_init, axis=0 ), np.ones((T,1))), "Sum of initial c is not 1!"

    c = c_init
    rho = rho_init

    ### initialize evaluation of regret
    regret_new = EvaluateExRegret(c, rho, Psi, src_list)
    regret_last = 0
    regret_unifrom = regret_new

    iteration = 0
    if show_text:
        print(iteration, ':', rho, c, regret_new)
    while np.abs(regret_new-regret_last) >= eta:
        ### new rho
        rho, _ = MinRegret_rho(c, Psi, rho_total, src_list, eta)
        assert np.isclose( np.sum(rho), rho_total ), "Sum of rho is not rho_total!"

        ### new c
        c, _ = MinRegret_c(rho, Psi, src_list, eta)
        assert np.allclose( np.sum(c, axis=0 ), np.ones((T,1))), "Sum of c is not 1!"
        
        ### evaluate
        regret_last = regret_new
        regret_new = EvaluateExRegret(c, rho, Psi, src_list)

        iteration += 1
        if show_text:       
            print(iteration, ':', rho, c, regret_new)
        
    return c, rho, regret_new, regret_unifrom


def MinRegret_rho(c, Psi, rho_total, src_list, eta):#  -> tuple[np.ndarray, np.float64]:
    n_src = len(src_list)
    T = Psi.shape[0]

    rho_cp = cp.Variable(n_src)
    constr = []

    obj = 0
    constr.append( rho_cp >= 0 )
    constr.append( cp.sum(rho_cp) == rho_total )
    for i_src, src in enumerate(src_list):
        power = 1 + 2 * cp.exp(- src.beta * (rho_cp[i_src] - src.gamma)) + cp.exp(- 2 * src.beta * (rho_cp[i_src] - src.gamma))
        sigma_square = 2 * ( src.deltaS / src.alpha )**2 * power
        Psi_rho = sigma_square * ( Psi * np.eye(T) )
        obj += c[i_src].T @ Psi_rho @ c[i_src]

    prob = cp.Problem(cp.Minimize(obj), constr)
    prob.solve()# solver=cp.MOSEK, verbose=False)

    return rho_cp.value, prob.value

def MinRegret_c(rho, Psi, src_list, eta): #-> tuple[np.ndarray, np.float64]:
    n_src = len(src_list)
    T = Psi.shape[0]

    obj = 0
    constr = []

    c_cp = []
    sum_c = 0
    for i_src, src in enumerate(src_list):
        c_cpi = cp.Variable((T, 1))
        c_cp.append(c_cpi)
        constr.append( c_cp[-1] >= 0 )
        sum_c += c_cp[-1]
    
    constr.append( sum_c == np.ones((T, 1)) )

    for i_src, src in enumerate(src_list):
        power = - src.beta * (rho[i_src] - src.gamma)
        sigma_square = 2 * ( src.deltaS * (1 + np.exp(power)) / src.alpha )**2
        Psi_rho = Psi * ( src.covar + sigma_square * np.eye(T) )
        obj += cp.atoms.quad_form(c_cp[i_src], Psi_rho)

    prob = cp.Problem(cp.Minimize(obj), constr)
    prob.solve()# solver=cp.MOSEK, verbose=False)

    c_value = np.zeros((n_src, T, 1))
    for i_src in range(n_src):
        c_value[i_src] = c_cp[i_src].value

    return c_value, prob.value

def EvaluateExRegret(c, rho, Psi, src_list) -> np.float64:
    regret = 0
    for i_src, src in enumerate(src_list):
        sigma_square = 2 * (src.deltaS * (1 + np.exp(-src.beta * (rho[i_src] - src.gamma)))/src.alpha )**2
        regret += c[i_src].T @ ( Psi * (src.covar + sigma_square*np.eye(Psi.shape[0]) ) ) @ c[i_src]

    assert regret >= 0, "Regret is negative, why??"

    return regret.item(0)
