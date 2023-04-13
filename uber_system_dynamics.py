import numpy as np
import yaml
import pandas as pd

yaml_file = open("parameters.yml")
parsed_yaml_file = yaml.load(yaml_file, Loader=yaml.FullLoader)["uber_para"]
T = parsed_yaml_file['T']
n = parsed_yaml_file['n']
m = parsed_yaml_file['m']
p = parsed_yaml_file['p']
# sample_n = parsed_yaml_file['sample_n']

def CtrlTask(s, random_seed=326, x0=np.array([1]*4).reshape(4, 1), T=T):
    #This function returns the co-design matrix, Phi.

    #T: total time steps
    #n: x-dim
    #m: u-dim
    #p: s-dim

    np.random.seed(random_seed)

    A = np.random.rand(n,n) + 0.1
    B = np.random.rand(n,m) + 0.1
    C = np.random.rand(n,p) + 0.1

    M_list = []
    N_list = []

    for i in range(T):
        if i == 0:
            M = np.concatenate((B, np.zeros((n, m*(T-1)))), axis=1)
            N = np.concatenate((C, np.zeros((n, p*(T-1)))), axis=1)
            M_list.append(M)
            N_list.append(N)
            continue

        _ = np.dot(A, M_list[-1][:, :m]).reshape((n, m))
        __ = np.dot(A, N_list[-1][:, :p]).reshape((n, p))

        M = np.concatenate((_, M_list[-1][:, :m*(T-1)]), axis=1)
        N = np.concatenate((__, N_list[-1][:, :p*(T-1)]), axis=1)
        M_list.append(M)
        N_list.append(N)

    ### Q R must be PSD
    Q = np.random.rand(n,n) + 0.1
    Q = Q.T @ Q
    R = np.random.rand(m,m) + 0.1
    R = R.T @ R

    s = s.T

    # K = np.zeros((m*T, m*T)) + R
    K = np.kron(np.eye(T), R)
    k = np.zeros((m*T,1))
    L = np.zeros((m*T, p*T))

    for t in range(T):
        # K
        _ = np.dot(M_list[t].T, Q)
        _ = np.dot(_, M_list[t])
        K += _

        #k
        __ = np.dot(M_list[t].T, Q)
        Nts = np.dot(N_list[t], s)
        A_pow_t1_x0 = np.dot(np.linalg.matrix_power(A, t+1), x0)
        __ = np.dot(__, (A_pow_t1_x0 + Nts) )
        k += __

        #L
        ___ = np.dot(M_list[t].T, Q)
        ___ = np.dot(___, N_list[t])
        L += ___

    u_opt = -1 * np.dot(np.linalg.inv(K), k)

    Phi = np.dot(L.T, np.linalg.inv(K))
    Phi = np.dot(Phi, L)

    # xt
    x_list = [np.array(x0)]

    for t in range(0, T-1):
        xt = np.dot(np.linalg.matrix_power(A, t+1), x0)
        xt += np.dot(M_list[t], u_opt)
        xt += np.dot(N_list[t], s)
        x_list.append(xt)

    cost = u_opt.T @ K @ u_opt + 2 * k.T @ u_opt
    for t in range(0, T):
        const = np.dot(np.linalg.matrix_power(A, t+1), x0) + N_list[t] @ s
        cost += const.T @ Q @ const

    return Phi, float(cost)


def CtrlTaskIndentity(s, x0=np.array([1]*4).reshape(4, 1), T=T):
    #This function returns the co-design matrix, Phi.

    #T: total time steps
    #n: x-dim
    #m: u-dim
    #p: s-dim

    # s = np.zeros((p*T,1))
    s = s.T

    A = np.eye(n, dtype=np.float64)
    B = - np.eye(m, dtype=np.float64)
    C = np.eye(p, dtype=np.float64)

    M_list = []
    N_list = []

    for i in range(T):
        if i == 0:
            M = np.concatenate((B, np.zeros((n, m*(T-1)))), axis=1)
            N = np.concatenate((C, np.zeros((n, p*(T-1)))), axis=1)
            M_list.append(M)
            N_list.append(N)

        _ = np.dot(A, M_list[-1][:, :m]).reshape((n, m))
        __ = np.dot(A, N_list[-1][:, :p]).reshape((n, p))

        M = np.concatenate((_, M_list[-1][:, :m*(T-1)]), axis=1)
        N = np.concatenate((__, N_list[-1][:, :p*(T-1)]), axis=1)
        M_list.append(M)
        N_list.append(N)

    ### Q R must be PSD
    Q = np.eye(m, dtype=np.float64)
    R = np.eye(m, dtype=np.float64)
    # Q = np.random.rand(n,n) + 0.1
    # Q = Q.T @ Q
    # R = np.random.rand(m,m) + 0.1
    # R = R.T @ R

    # print(Q, R)


    K = np.kron(np.eye(T,dtype=int), R)
    k = np.zeros((m*T, 1)) # 20x1
    L = np.zeros((m*T, p*T))

    for t in range(T):
        # K
        _ = np.dot(M_list[t].T, Q)
        _ = np.dot(_, M_list[t])
        K += _

        #k
        __ = np.dot(M_list[t].T, Q) # 20x4
        Nts = np.dot(N_list[t], s) # 4x1
        A_pow_t1_x0 = np.dot(np.linalg.matrix_power(A, t+1), x0) # 4x4
        __ = np.dot(__, (A_pow_t1_x0 + Nts) ) 
        k += __

        #L
        ___ = np.dot(M_list[t].T, Q)
        ___ = np.dot(___, N_list[t])
        L += ___

    u_opt = -1 * np.dot(np.linalg.inv(K), k)

    Phi = np.dot(L.T, np.linalg.inv(K))
    Phi = np.dot(Phi, L)
    
    # xt
    x_list = [np.array(x0)]
    # x_list_ = [x0]

    for t in range(0, T):
        xtplus1 = np.dot(np.linalg.matrix_power(A, t+1), x0)
        xtplus1 += np.dot(M_list[t], u_opt)
        xtplus1 += np.dot(N_list[t], s)
        x_list.append(xtplus1)

        # x = np.dot(A, x_list_[-1]) + np.dot(B, u_opt[t]) + np.dot(C, s[t])
        # x_list_.append(x)

    cost = u_opt.T @ K @ u_opt + 2 * k.T @ u_opt
    for t in range(0, T):
        const = np.dot(np.linalg.matrix_power(A, t+1), x0) + N_list[t] @ s
        cost += const.T @ Q @ const

    return Phi, float(cost)

if __name__ == '__main__':
    # Phi = CtrlTask()
    Phi, cost = CtrlTaskIndentity(s=np.zeros((1,p*T)))
    print(Phi.shape, cost)
