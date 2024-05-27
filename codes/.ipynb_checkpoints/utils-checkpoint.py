import matplotlib
import numpy as np
import time
from numpy.linalg import norm, cond
from numba import njit, objmode
from sklearn import linear_model
from sklearn.datasets import load_svmlight_file
from scipy.io import savemat, loadmat
import matplotlib.pyplot as plt

# have more readable plots with increased fontsize:
fontsize = 25
# pl.rcParams['axes.linewidth'] = 0.1
plt.rcParams.update({'axes.labelsize': fontsize,
              'font.size': fontsize,
              'font.weight': 'bold',
              'axes.linewidth': 2.,
              'legend.fontsize': fontsize,
              'xtick.labelsize': fontsize - 8,
              'ytick.labelsize': fontsize - 8,
              'agg.path.chunksize':200
                    })

@njit
def least_square_loss(A, b, x):
    """Value of Least square objective at x."""
    return .5 * np.mean((A @ x - b) ** 2)

def create_mat(dim_row, dim_col, cond_numb, mu=1, mu2=1, mean=0, std_dev=1, control=True):
    # M = 2*np.random.randn(dim_row, dim_col) + 10
    M = std_dev * np.random.randn(dim_row, dim_col) + mean
    if control:
        U,b,V=np.linalg.svd(M, full_matrices=False)
        # s = b[0] # b is ordered in the decreasing order
        # linear streching of b
        if cond_numb < 1:
            raise ValueError("The condition number should be at least one (>= 1)")
        elif not mu:
            # diag = cond_numb * (s-b) / (s-b[-1])
            s = b[0]
            # diag = np.sqrt(dim_row)*(cond_numb - (cond_numb-1) * (s-b) / (s-b[-2])) 
            diag = mu2*(cond_numb - (cond_numb-1) * (s-b) / (s-b[-2]))
            # diag = diag / cond_numb
            diag[-1] = 0.
        else:
            s = b[0] # b is ordered in the decreasing order
            diag = mu*cond_numb - mu*(cond_numb-1) * (s-b) / (s-b[-1])
            # diag = diag / cond_numb
        # print(s, b[-1], diag, (s-b) / (s-b[-1]), mu*(cond_numb-1) * (s-b) / (s-b[-1]))
        # print(diag)
        return U @ np.diag(diag) @ V, diag[-2]
    else:
        return M#, np.linalg.norm(M, ord=-2)

def make_data_reg(dim_row, dim_col, cond_numb_A=1, noise_scale=0., mu=1., mu2=1, mean=0., std_dev=1., control=True):
    # np.random.seed(0) # The same number will appear every time
    A, _ = create_mat(dim_row, dim_col, cond_numb_A, mu=mu, mu2=mu2, mean=mean, std_dev=std_dev, control=control)
    x_star = np.random.randn(dim_col)
    b = A @ x_star + (noise_scale * np.random.randn(dim_row))
    return A, b, x_star, _

def make_data_class(dim_row, dim_col, cond_numb_A=1, prob_noise=0, mu=1, control=False):
    A, _ = create_mat(dim_row, dim_col, cond_numb_A, control=control)
    x_star = 5*np.random.randn(dim_col)
    b = np.sign(np.sign(A @ x_star) + .5) # We make sign(0) = 1 instead of the build-in 0.
    # b[b<0] = 0
    if not np.ceil(prob_noise):
        return A, b, x_star
    else:
        prob_list = np.random.choice([0,1], dim_row, p=[1-prob_noise, prob_noise])
        for i in prob_list:
            if i:
                b[i] = - b[i]
        return A, b, x_star
    
@njit
def prox_logit(x, gamma):
    """
    This code was taken from http://proximity-operator.net
    """
    limit = 5e2
    size_x = x.size
    w = np.zeros(size_x)
    ex = np.exp(x)
    z = gamma*ex
    
    # INITIALIZATION
    approx = gamma * (1 - np.exp(gamma-x))
    w[z>1] = approx[z>1]
    
    # RUN
    max_iter = 20;
    test_end = np.zeros(size_x)
    precision = 1e-8
    epsilon = 1e-20
    
    for _ in range(max_iter):
        e = np.exp(w)
        y = w*e + ex*w - z
        v = e*(1 + w) + ex
        u = e*(2 + w)
        w_new = w -  y/( v -  y * u/(2*v) )
        test = (np.abs(w_new-w)/(epsilon + np.abs(w))  < precision)
        tmp = (test_end == 0)
        test_end[np.logical_and(test, tmp)] = 1
        idx_update = np.logical_and(np.logical_not(test), tmp)
        w[idx_update] = w_new[idx_update] #the rest stays constant !
        if(np.sum(test_end) == size_x): # stop !
            break
    p = x - w
    
    # ASYMPTOTIC DVP
    test = (x>limit)
    p[test] = x[test] - approx[test]
    
    return p

@njit
def prox_log_reg(x, b_i, a_i, step_size):
    if b_i == 0:
        return x
    else:
        u = -b_i * a_i
        ux = u @ x
        uu = u @ u
        v = np.array([ux])
        # We have to compute the prox of a function composed with a linear one.
        tmp = prox_logit(v, uu*step_size)
        return x + ((tmp[0] - ux)/uu) * u

@njit
def compute_prox(a_i, b_i, step_size, v):
    numerator = b_i - a_i @ v
    denominator = step_size * (a_i @ a_i) + 1
    frac = numerator / denominator
    tmp = step_size * a_i * frac
    # return v + tmp
    tmp += v
    return tmp

@njit
def compute_prox_matrix(A, b, step_size, v):
    first_term = (A.T @ A) + ((1/step_size)*np.diag(np.ones(A.shape[1])))
    inv_first_term = np.linalg.inv(first_term)
    second_term = (A.T @ b) +((1/step_size)*v)
    return inv_first_term @ second_term

@njit
def logit_log1pexp(x, scalar=False):
    """
    Log(1 + exp(-x)) = logsumexp(0,-x), avoid overflow
    and more accurate than the naive computation.
    https://fa.bianp.net/blog/2019/evaluate_logistic/
    """
    if scalar is True:
        if x < -33.3:
            return -x
        elif x <= -18:
            return np.exp(x) - x
        elif x <= 37:
            return np.log1p(np.exp(-x))
        else:
            return np.exp(-x)
    else:
        out = np.zeros_like(x)
        idx0 = x < -33
        out[idx0] = -x[idx0]
        idx1 = (x >= -33) & (x < -18)
        out[idx1] =  np.exp(x[idx1]) - x[idx1]
        idx2 = (x >= -18) & (x < 37)
        out[idx2] = np.log1p(np.exp(-x[idx2]))
        idx3 = x >= 37
        out[idx3] = np.exp(-x[idx3])

        return out
@njit
def logistic_loss(A, b, x):
    z = A @ x
    return np.mean(logit_log1pexp(b*z))

@njit
def grad_logistic_func(a,x,b):
    v = -b * a @ x
    num = -b * np.exp(v)
    dem = 1. + np.exp(v)
    return (num/dem) * a

@njit
def full_grad_logistic_func(A,x,b):
    return -b * A.T @ (1. / (1. + np.exp(b * (A @ x)))) / A.shape[0]

@njit
def stoch_prox_logistic(A, b, x_0, max_iter, L, exp=.55, c=1., rnd_seed=None):
    n_features = A.shape[1]
    n_samples = A.shape[0]
    new_x = x_0.copy() #np.zeros(n_features)
    fx_tmp = np.zeros(max_iter+1)
    fx_tmp[0] = logistic_loss(A, b, x_0)
    sum_step = 0
    # erg_it = 0
    erg_x = x_0.copy()
    
    if not (rnd_seed == None):
        np.random.seed(rnd_seed)
        
    w_t = 1
    weighted_x = new_x.copy()
        
    for t in range(max_iter):
        # step_size = min((1. / ((t+1)**exp)) * c, 1. / (4.*L))
        step_size = 1. / ((t+1)**exp) * c
        # if t % 5000 == 0:
        #     print(step_size, 1. / (4.*L))
        # step_size = 1. / (4.*L)
        row_i = np.random.randint(n_samples)
        a_i = A[row_i]
        b_i = b[row_i]
        # We compute the weighted iterates according to 
        # http://proceedings.mlr.press/v134/sebbouh21a/sebbouh21a.pdf
        sum_step += step_size
        omega = step_size / sum_step
        erg_x = omega * new_x + (1 - omega)*erg_x
        
#         if t == 0:
#             step_prev = 0
#         else:
#             step_prev = min((1. / (t**exp)) * c, 1. / (4.*L)) # 1. / (t**exp) 
#         w_t = step_size*w_t / (step_prev + step_size*w_t)
#         weighted_x = w_t*new_x + (1 - w_t)*weighted_x
        
#         print(weighted_x, erg_x)
        
        fx_tmp[t+1] = logistic_loss(A, b, erg_x)
        # erg_it 
        new_x = prox_log_reg(new_x, b_i, a_i, step_size)
        # fx_tmp[t] = logistic_loss(A, b, new_x)
    return fx_tmp

@njit
def svrp_logistic(A, b, step_size, x_0, s, inner_iter=1, rnd_seed=None, last_it=False):
    n_features = A.shape[1]
    n_samples = A.shape[0]
    new_x = x_0.copy()
    prev_x = new_x.copy()

    fx_tmp_s = np.zeros(s+1)
    fx_tmp_s[0] = logistic_loss(A, b, x_0)
    fx_tmp = np.zeros(s*inner_iter+1)
    fx_tmp[0] = logistic_loss(A, b, x_0)

    # erg_x = np.zeros(n_features)

    k = 0
    if not (rnd_seed == None):
        np.random.seed(rnd_seed)
    for t in range(s):
        full_grad_previous = full_grad_logistic_func(A, prev_x, b)
        for i in range(inner_iter):
            # erg_x = erg_x + new_x
            row_i = np.random.randint(n_samples)
            a_i = A[row_i]
            b_i = b[row_i]
            stoch_grad_previous = grad_logistic_func(a_i, prev_x, b_i)
            v = (new_x + step_size*stoch_grad_previous
                 - step_size*full_grad_previous)
            new_x = prox_log_reg(v, b_i, a_i, step_size)
            fx_tmp[k+1] = logistic_loss(A, b, new_x)
            k += 1

        if not last_it:
            s_t = np.random.choice(inner_iter)
            ind = k - inner_iter + s_t
            fx_tmp_s[t+1] = fx_tmp[ind]
        else:
            fx_tmp_s[t+1] = fx_tmp[k-1]
        prev_x = new_x.copy()
    return fx_tmp_s, fx_tmp

@njit
def svrp_logistic_acc(A, b, step_size, x_0, max_iter, inner_iter=1, accuracy=5e-2
                 ,f_star=0., rnd_seed=None, max_it=2.5e7):
    n_features = A.shape[1]
    n_samples = A.shape[0]
    new_x = x_0.copy()
    prev_x = new_x.copy()

    # fx_tmp_ = np.zeros(max_iter)
    # fx_tmp = np.zeros(max_iter*inner_iter)
    fx_tmp = 34e10

    # erg_x = np.zeros(n_features)

    k = 0
    # t = 0
    
    n_increase = 0
    max_n_increase = 100
    
    if not (rnd_seed == None):
        np.random.seed(rnd_seed)
        
    # for t in range(max_iter):
    while (fx_tmp - f_star > accuracy): #and (k <= 1e7):
        # if t == -1:
        #     k = 0
        #     t = 0
        full_grad_previous = full_grad_logistic_func(A, prev_x, b)
        # for i in range(inner_iter):
        for i in range(inner_iter): 
            # erg_x = erg_x + new_x
            prev_f = fx_tmp
            row_i = np.random.randint(n_samples)
            a_i = A[row_i]
            b_i = b[row_i]
            stoch_grad_previous = grad_logistic_func(a_i, prev_x, b_i)
            v = (new_x + step_size*stoch_grad_previous
                 - step_size*full_grad_previous)
            new_x = prox_log_reg(v, b_i, a_i, step_size)
            fx_tmp = logistic_loss(A, b, new_x)
            if fx_tmp - f_star <= accuracy:
                return k + 1
            # If we have max_n_increase consecutive increases we stop
            if prev_f <= fx_tmp:
                n_increase += 1
            else:
                n_increase = 0
            if n_increase == max_n_increase or fx_tmp == np.inf or fx_tmp == np.nan:
            # if fx_tmp == np.inf or fx_tmp == np.nan:
                return max_it
                # return -1 #int(1.5*max_iter*inner_iter)
            k += 1
            if k > max_it:
                return max_it
            # if k % 1e6 == 0:
            #     with objmode():
            #         start = time.perf_counter()
            #         # start = time.time()
            #         print(f"iterations: {k}, time per {1e6}: {time.perf_counter() - start}")
        #     if s.any():
        #         if i == s[t]:
        #             tmp = new_x.copy()
        # t += 1
        # if s.any():
        #     new_x = tmp.copy()
        prev_x = new_x.copy()
    # if fx_tmp[-1] > fx_tmp[0]:
    #     return int(1.5*max_iter*inner_iter)
    return k #max_iter*inner_iter

@njit
def sapa_logistic(A, b, step_size, x_0, max_iter, rnd_seed=None):
    n_features = A.shape[1]
    n_samples = A.shape[0]
    new_x = x_0.copy()
    grad_tab = np.zeros((n_samples, n_features))

    fx_tmp = np.zeros(max_iter+1)
    fx_tmp[0] = logistic_loss(A, b, x_0)
    sum_x = np.zeros(x_0.shape[0])

    for i in range(n_samples):
        a_i = A[i]
        b_i = b[i]
        grad_tab[i] = grad_logistic_func(a_i, new_x, b_i)
    
    full_grad = np.sum(grad_tab, axis=0) / n_samples
    if not (rnd_seed==None):
        np.random.seed(rnd_seed)
    for t in range(max_iter):
        row_i = np.random.randint(n_samples)
        a_i = A[row_i]
        b_i = b[row_i]
        # print(a_i.shape, new_x.shape)
        stoch_grad = grad_logistic_func(a_i, new_x, b_i)
        # grad_phi = grad_tab[row_i].copy()
        # grad_tab[row_i] = stoch_grad.copy()
        # g_k =  grad_phi - full_grad
        # v = (new_x + step_size*g_k)
        # new_x = prox_log_reg(v, b_i, a_i, step_size)
        # fx_tmp[t+1] = logistic_loss(A, b, sum_x / (t+1))
        # sum_x += new_x
        # # fx_tmp[t] = logistic_loss(A, b, new_x)
        # full_grad = full_grad + (stoch_grad - grad_phi) / n_samples
        ########################################################################
        g_k =  grad_tab[row_i] - full_grad
        v = (new_x + step_size*g_k)
        new_x = prox_log_reg(v, b_i, a_i, step_size)
        fx_tmp[t+1] = logistic_loss(A, b, sum_x / (t+1))
        sum_x += new_x
        # fx_tmp[t] = logistic_loss(A, b, new_x)
        full_grad = full_grad + (stoch_grad - grad_tab[row_i]) / n_samples
        grad_tab[row_i] = stoch_grad

    return fx_tmp 

@njit
def sapa_logistic_acc(A, b, step_size, x_0, max_iter, accuracy=5e-2, f_star=0., rnd_seed=None, max_it=2.5e7):
    n_features = A.shape[1]
    n_samples = A.shape[0]
    new_x = x_0.copy()
    grad_tab = np.zeros((n_samples, n_features))

    # fx_tmp = np.zeros(max_iter)
    fx_tmp = 34e10

    for i in range(n_samples):
        a_i = A[i]
        b_i = b[i]
        grad_tab[i] = grad_logistic_func(a_i, new_x, b_i)
    
    full_grad = np.sum(grad_tab, axis=0) / n_samples
    
    t = 0
    # k = 0
    
    n_increase = 0
    max_n_increase = 100
    
    if not (rnd_seed == None):
        np.random.seed(rnd_seed)    
    
    # for t in range(max_iter):
    while (fx_tmp - f_star > accuracy): # and (t <= 1e7):
        # if k == -1:
        #     t = 0
        #     k = 0
        prev_f = fx_tmp
        row_i = np.random.randint(n_samples)
        a_i = A[row_i]
        b_i = b[row_i]
        stoch_grad = grad_logistic_func(a_i, new_x, b_i)
        grad_phi = grad_tab[row_i].copy()
        grad_tab[row_i] = stoch_grad.copy()
        g_k =  grad_phi - full_grad
        v = (new_x + step_size*g_k)
        new_x = prox_log_reg(v, b_i, a_i, step_size)
        # fx_tmp[t] = least_square_loss(A, b, new_x)
        fx_tmp = logistic_loss(A, b, new_x)
        # If we have max_n_increase consecutive increases we stop
        if fx_tmp - f_star <= accuracy:
            return t + 1
        if prev_f <= fx_tmp:
            n_increase += 1
        else:
            n_increase = 0
        if n_increase == max_n_increase or fx_tmp == np.inf or fx_tmp == np.nan:
        # if fx_tmp == np.inf or fx_tmp == np.nan:
            return max_it
                # return -1 #int(1.5*max_iter)
        t += 1
        if t > max_it:
            return t
            # return t
        # if t % 1e6 == 0:
        #     with objmode():
        #         start = time.perf_counter()
        #         # start = time.time()
        #         print(f"iterations: {k}, time per {1e6}: {time.perf_counter() - start}")
        full_grad = full_grad + (stoch_grad - grad_phi) / n_samples
    # if fx_tmp[-1] > fx_tmp[0]:
    #     return int(1.5*max_iter)
    return t #max_iter

@njit
def saga_logistic_acc(A, b, step_size, x_0, max_iter, accuracy=5e-2, f_star=0., rnd_seed=None, max_it=2.5e7):
    n_features = A.shape[1]
    n_samples = A.shape[0]
    new_x = x_0.copy()
    grad_tab = np.zeros((n_samples, n_features))

    # fx_tmp = np.zeros(max_iter)
    fx_tmp = 34e10

    for i in range(n_samples):
        a_i = A[i]
        b_i = b[i]
        grad_tab[i] = grad_logistic_func(a_i, new_x, b_i)

    full_grad = np.sum(grad_tab, axis=0) / n_samples
    full_grad_1 = full_grad

    t = 0
    # k = - 1
    
    n_increase = 0
    max_n_increase = 100
    if not (rnd_seed == None):
        np.random.seed(rnd_seed)
    
    # for t in range(max_iter):
    while (fx_tmp - f_star > accuracy): # and (t <= 1e7):
        # if k == -1:
        #     t = 0
        #     k = 0
        prev_f = fx_tmp
        row_i = np.random.randint(n_samples)
        a_i = A[row_i]
        b_i = b[row_i]
        stoch_grad = grad_logistic_func(a_i, new_x, b_i)
        # full_grad = np.sum(grad_tab, axis=0) / n_samples
        grad_phi = grad_tab[row_i].copy()
        grad_tab[row_i] = stoch_grad.copy()
        g_k = full_grad - grad_phi
        v = stoch_grad + g_k
        new_x -= step_size * v
        # fx_tmp[t] = least_square_loss(A, b, new_x)
        fx_tmp = logistic_loss(A, b, new_x)
        if fx_tmp - f_star <= accuracy:
            return t + 1
        # If we have max_n_increase consecutive increases we stop
        if prev_f <= fx_tmp:
            n_increase += 1
        else:
            n_increase = 0
        # if t != 0:
        #     if fx_tmp[t] >= fx_tmp[t-1]:
        #         n_increase += 1
        #     else:
        #         n_increase = 0
        if n_increase == max_n_increase or fx_tmp == np.inf or fx_tmp == np.nan:
        # if fx_tmp == np.inf or fx_tmp == np.nan:
            return max_it
            # return -1 # int(1.5*max_iter)
        # if least_square_loss(A, b, new_x) - f_star <= accuracy:
        #     return t + 1
        t += 1
        if t > max_it:
            return t
            # return t
        full_grad = full_grad + (stoch_grad - grad_phi) / n_samples
    # if fx_tmp[-1] > fx_tmp[0]:
    #     return int(1.5*max_iter)
    return t #max_iter

@njit
# def svrg_ols(A, b, step_size, x_0, max_iter, inner_iter=1, s=np.zeros(2)):
def svrg_logistic(A, b, step_size, x_0, max_iter, inner_iter=1, rnd_seed=None, last_it=False):
    n_features = A.shape[1]
    n_samples = A.shape[0]
    new_x = x_0.copy()
    prev_x = new_x.copy()
    fx_tmp = np.zeros(max_iter*inner_iter+1)
    fx_tmp[0] = least_square_loss(A, b, x_0)
    fx_tmp_s = np.zeros(max_iter+1)
    fx_tmp_s[0] = least_square_loss(A, b, x_0)

    k = 0
    if not (rnd_seed == None):
        np.random.seed(rnd_seed)
    for t in range(max_iter):
        full_grad_previous = full_grad_logistic_func(A, prev_x, b)
        for i in range(inner_iter):
            row_i = np.random.randint(n_samples)
            a_i = A[row_i]
            b_i = b[row_i]
            stoch_grad_previous = grad_logistic_func(a_i, prev_x, b_i)
            stoch_grad_actual = grad_logistic_func(a_i, new_x, b_i)
            v = stoch_grad_actual - stoch_grad_previous + full_grad_previous
            new_x -= step_size * v
            fx_tmp[k+1] = logistic_loss(A, b, new_x)
            k += 1
        #     if s.any():
        #         if i == s[t]:
        #             tmp = new_x.copy()
        # if s.any():
        #     new_x = tmp.copy()
        if not last_it:
            s_t = np.random.choice(inner_iter)
            ind = k - inner_iter + s_t
            fx_tmp_s[t+1] = fx_tmp[ind]
        else:
            fx_tmp_s[t+1] = fx_tmp[k-1]
        # fx_tmp_s[t] = least_square_loss(A, b, new_x)
        prev_x = new_x.copy()
    return fx_tmp_s, fx_tmp

@njit
def svrg_logistic_acc(A, b, step_size, x_0, max_iter, inner_iter=1, accuracy=5e-2
                 , f_star=0., rnd_seed=None, max_it=2.5e7):
    n_features = A.shape[1]
    n_samples = A.shape[0]
    new_x = x_0.copy()
    prev_x = new_x.copy()
    # fx_tmp = np.zeros(max_iter*inner_iter)
    fx_tmp = 34e10

    k = 0
    
    n_increase = 0
    max_n_increase = 100

    if not (rnd_seed == None):
        np.random.seed(rnd_seed)
    
    # t = -1
    # for t in range(max_iter):
    while (fx_tmp - f_star > accuracy): # and (k <= 1e7):
        full_grad_previous = full_grad_logistic_func(A, prev_x, b)
        # if t == -1:
        #     k = 0
        #     t = 0
        for i in range(inner_iter):
            prev_f = fx_tmp
            row_i = np.random.randint(n_samples)
            a_i = A[row_i]
            b_i = b[row_i]
            stoch_grad_previous = grad_logistic_func(a_i, prev_x, b_i)
            stoch_grad_actual = grad_logistic_func(a_i, new_x, b_i)
            v = stoch_grad_actual - stoch_grad_previous + full_grad_previous
            new_x -= step_size * v
            # fx_tmp[k] = least_square_loss(A, b, new_x)
            fx_tmp = logistic_loss(A, b, new_x)
            if fx_tmp - f_star <= accuracy:
                return k + 1
            # If we have max_n_increase consecutive increases we stop
            if prev_f <= fx_tmp:
                n_increase += 1
            else:
                n_increase = 0
            # if k != 0:
            #     if fx_tmp[k] >= fx_tmp[k-1]:
            #         n_increase += 1
            #     else:
            #         n_increase = 0
            # if n_increase == max_n_increase or fx_tmp == np.inf or fx_tmp == np.nan:
            if fx_tmp == np.inf or fx_tmp == np.nan:
                return max_it
                # return -1 # int(1.5*max_iter*inner_iter)
            k += 1
            if k > max_it:
                return max_it
                # return k
            # if s.any():
            #     if i == s[t]:
            #         tmp = new_x.copy()
            # t += 1
        # if s.any():
        #     new_x = tmp.copy()
        prev_x = new_x.copy()
    # if fx_tmp[-1] > fx_tmp[0]:
    #     return int(1.5*max_iter*inner_iter)
    return k #max_iter*inner_iter

@njit
def grad_desc_logistic(A, b, L, max_iter):
    n_features = A.shape[1]
    n_samples = A.shape[0]
    new_x = np.zeros(n_features)
    step_size = 1. / L
    
    for t in range(max_iter):
        grad = full_grad_logistic_func(A, new_x, b)
        new_x -= step_size*grad
    return new_x
    
@njit
def stoch_prox_least_square(A, b, x_0, max_iter, exp=.55, c=1., rnd_seed=None):
    n_features = A.shape[1]
    n_samples = A.shape[0]
    new_x = x_0.copy()
    erg_x = x_0.copy()
    sum_step = 0
    
    fx_tmp = np.zeros(max_iter+1)
    fx_tmp[0] = least_square_loss(A, b, x_0)

    if not (rnd_seed == None):
        np.random.seed(rnd_seed)

    for t in range(max_iter):
        step_size = (1. / (t+1)**exp) * c
        row_i = np.random.randint(n_samples)
        a_i = A[row_i]
        b_i = b[row_i]
        # We compute the weighted iterates according to 
        # http://proceedings.mlr.press/v134/sebbouh21a/sebbouh21a.pdf
        # sum_step += step_size
        # omega = step_size / sum_step
        # erg_x = omega * new_x + (1 - omega)*erg_x
        # fx_tmp[t] = least_square_loss(A, b, erg_x)
        
        new_x = compute_prox(a_i, b_i, step_size, new_x)
        fx_tmp[t+1] = least_square_loss(A, b, new_x)
        
        # all_x[t] = new_x.copy()
        #####################################################################
    return fx_tmp#, all_x

@njit
def stoch_grad_least_square(A, b, x_0, max_iter, exp=.55, c=1.):
    n_features = A.shape[1]
    n_samples = A.shape[0]
    new_x = x_0.copy()
    weighted_x = new_x.copy()    
    fx_tmp = np.zeros(max_iter)
    # all_x = np.zeros((max_iter,n_features))
    
    w_t = 1
    
    full_grad_previous = A.T @ (A @ new_x - b) / n_samples

    for t in range(max_iter):
        # We compute stepsize #####################
        # step_size = min(1. / (t+1)**c, 1. / 6.*L)
        step_size = (1. / (t+1)**exp) * c
        # print(step_size, (1. / (t+1)**exp))
        ##################################################
        row_i = np.random.randint(n_samples)
        a_i = A[row_i]
        b_i = b[row_i]
        stoch_grad = (a_i @ new_x - b_i) * a_i
        if t >= 0:
            new_x -= step_size * stoch_grad
        else:
            new_x -= step_size*full_grad_previous
        # We compute the weighted iterates according to 
        # http://proceedings.mlr.press/v134/sebbouh21a/sebbouh21a.pdf
        if t == 0:
            step_prev = 0
        else:
            step_prev = 1. / (t+1)**c #min(1. / t, 1. / 4. * L)
        w_t = step_size*w_t / (step_prev + step_size*w_t)
        weighted_x = w_t*new_x + (1 - w_t)*weighted_x
        #######################################################################
        fx_tmp[t] = least_square_loss(A, b, weighted_x)
        # all_x[t] = new_x.copy()
    return fx_tmp #, all_x

@njit
# def svrp_ols(A, b, step_size, x_0, max_iter, inner_iter=1, s=np.zeros(2)):
def svrp_ols(A, b, step_size, x_0, max_iter, inner_iter=1, rnd_seed=None, last_it=False):
    n_features = A.shape[1]
    n_samples = A.shape[0]
    new_x = x_0.copy()
    prev_x = new_x.copy()

    fx_tmp_s = np.zeros(max_iter+1)
    fx_tmp_s[0] = least_square_loss(A, b, x_0)
    fx_tmp = np.zeros(max_iter*inner_iter+1)
    fx_tmp[0] = least_square_loss(A, b, x_0)

    # erg_x = np.zeros(n_features)

    k = 0
    if not (rnd_seed == None):
        np.random.seed(rnd_seed)
    for t in range(max_iter):
        full_grad_previous = (A.T @ (A @ prev_x - b)) / n_samples
        for i in range(inner_iter):
            # erg_x = erg_x + new_x
            row_i = np.random.randint(n_samples)
            a_i = A[row_i]
            b_i = b[row_i]
            stoch_grad_previous = ((a_i @ prev_x) - b_i) * a_i
            v = (new_x + step_size*stoch_grad_previous
                 - step_size*full_grad_previous)
            new_x = compute_prox(a_i, b_i, step_size, v)
            fx_tmp[k+1] = least_square_loss(A, b, new_x)
            k += 1
#             if s.any():
#                 if i == s[t]:
#                     tmp = new_x.copy()

#         if s.any():
#             new_x = tmp.copy()
        if not last_it:
            s_t = np.random.choice(inner_iter)
            ind = k - inner_iter + s_t
            fx_tmp_s[t+1] = fx_tmp[ind]
        else:
            fx_tmp_s[t+1] = fx_tmp[k-1]
        # fx_tmp_s[t] = least_square_loss(A, b, new_x)
        prev_x = new_x.copy()
    return fx_tmp_s, fx_tmp 

@njit
def svrp_ols_acc(A, b, step_size, x_0, max_iter, inner_iter=1, accuracy=5e-2
                 ,f_star=0., rnd_seed=None, max_it=int(4e4)):
    n_features = A.shape[1]
    n_samples = A.shape[0]
    new_x = x_0.copy()
    prev_x = new_x.copy()

    # fx_tmp_ = np.zeros(max_iter)
    # fx_tmp = np.zeros(max_iter*inner_iter)
    fx_tmp = 34e10

    # erg_x = np.zeros(n_features)

    k = 0
    # t = 0
    
    n_increase = 0
    max_n_increase = 100
    
    if not (rnd_seed == None):
        np.random.seed(rnd_seed)
        
    # for t in range(max_iter):
    while (fx_tmp - f_star > accuracy): #and (k <= 1e7):
        # if t == -1:
        #     k = 0
        #     t = 0
        full_grad_previous = (A.T @ (A @ prev_x - b)) / n_samples
        # for i in range(inner_iter):
        for i in range(inner_iter): 
            # erg_x = erg_x + new_x
            prev_f = fx_tmp
            row_i = np.random.randint(n_samples)
            a_i = A[row_i]
            b_i = b[row_i]
            stoch_grad_previous = ((a_i @ prev_x) - b_i) * a_i
            v = (new_x + step_size*stoch_grad_previous
                 - step_size*full_grad_previous)
            new_x = compute_prox(a_i, b_i, step_size, v)
            # fx_tmp[k] = least_square_loss(A, b, new_x)
            fx_tmp = least_square_loss(A, b, new_x)
            if fx_tmp - f_star <= accuracy:
                return k + 1
            # If we have max_n_increase consecutive increases we stop
            if prev_f <= fx_tmp:
                n_increase += 1
            else:
                n_increase = 0
            # if n_increase == max_n_increase or fx_tmp == np.inf or fx_tmp == np.nan:
            if fx_tmp == np.inf or fx_tmp == np.nan:
            # if fx_tmp == np.inf or fx_tmp == np.nan:
                return max_it
                # return -1 #int(1.5*max_iter*inner_iter)
            k += 1
            if k > max_it:
                return max_it
            # if k % 1e6 == 0:
            #     with objmode():
            #         start = time.perf_counter()
            #         # start = time.time()
            #         print(f"iterations: {k}, time per {1e6}: {time.perf_counter() - start}")
        #     if s.any():
        #         if i == s[t]:
        #             tmp = new_x.copy()
        # t += 1
        # if s.any():
        #     new_x = tmp.copy()
        prev_x = new_x.copy()
    # if fx_tmp[-1] > fx_tmp[0]:
    #     return int(1.5*max_iter*inner_iter)
    return k #max_iter*inner_iter

@njit
def sapa_ols(A, b, step_size, x_0, max_iter, rnd_seed=None):
    n_features = A.shape[1]
    n_samples = A.shape[0]
    new_x = x_0.copy()
    grad_tab = np.zeros((n_samples, n_features))

    # all_x = np.zeros((max_iter, n_features))
    fx_tmp = np.zeros(max_iter+1)
    fx_tmp[0] = least_square_loss(A, b, x_0)

    for i in range(n_samples):
        a_i = A[i]
        b_i = b[i]
        grad_tab[i] = ((a_i @ new_x) - b_i) * a_i
    
    full_grad = np.sum(grad_tab, axis=0) / n_samples
    if not (rnd_seed==None):
        np.random.seed(rnd_seed)
    for t in range(max_iter):
        row_i = np.random.randint(n_samples)
        a_i = A[row_i]
        b_i = b[row_i]
        stoch_grad = ((a_i @ new_x) - b_i) * a_i
        # grad_phi = grad_tab[row_i].copy()
        # grad_tab[row_i] = stoch_grad.copy()
        # g_k =  grad_phi - full_grad
        # v = (new_x + step_size*g_k)
        # new_x = compute_prox(a_i, b_i, step_size, v)
        # fx_tmp[t] = least_square_loss(A, b, new_x)
        # # all_x[t] = new_x.copy()
        # full_grad = full_grad + (stoch_grad - grad_phi) / n_samples
        #################################################################################
        # grad_phi = grad_tab[row_i].copy()
        g_k =  grad_tab[row_i] - full_grad
        v = (new_x + step_size*g_k)
        new_x = compute_prox(a_i, b_i, step_size, v)
        fx_tmp[t+1] = least_square_loss(A, b, new_x)
        # all_x[t] = new_x.copy()
        full_grad = full_grad + ((stoch_grad - grad_tab[row_i]) / n_samples)
        grad_tab[row_i] = stoch_grad

    return fx_tmp #, all_x

@njit
def sapa_ols_acc(A, b, step_size, x_0, max_iter, accuracy=5e-2, f_star=0., rnd_seed=None):
    n_features = A.shape[1]
    n_samples = A.shape[0]
    new_x = x_0.copy()
    grad_tab = np.zeros((n_samples, n_features))

    # fx_tmp = np.zeros(max_iter)
    fx_tmp = 34e10

    for i in range(n_samples):
        a_i = A[i]
        b_i = b[i]
        grad_tab[i] = ((a_i @ new_x) - b_i) * a_i
    
    full_grad = np.sum(grad_tab, axis=0) / n_samples
    
    t = 0
    # k = 0
    
    n_increase = 0
    max_n_increase = 100
    
    if not (rnd_seed == None):
        np.random.seed(rnd_seed)    
    
    # for t in range(max_iter):
    while (fx_tmp - f_star > accuracy): # and (t <= 1e7):
        # if k == -1:
        #     t = 0
        #     k = 0
        prev_f = fx_tmp
        row_i = np.random.randint(n_samples)
        a_i = A[row_i]
        b_i = b[row_i]
        stoch_grad = ((a_i @ new_x) - b_i) * a_i
        grad_phi = grad_tab[row_i].copy()
        grad_tab[row_i] = stoch_grad.copy()
        g_k =  grad_phi - full_grad
        v = (new_x + step_size*g_k)
        new_x = compute_prox(a_i, b_i, step_size, v)
        # fx_tmp[t] = least_square_loss(A, b, new_x)
        fx_tmp = least_square_loss(A, b, new_x)
        # If we have max_n_increase consecutive increases we stop
        if fx_tmp - f_star <= accuracy:
            return t + 1
        if prev_f <= fx_tmp:
            n_increase += 1
        else:
            n_increase = 0
        if n_increase == max_n_increase or fx_tmp == np.inf or fx_tmp == np.nan:
        # if fx_tmp == np.inf or fx_tmp == np.nan:
            return 40000
                # return -1 #int(1.5*max_iter)
        t += 1
        if t > 40000:
            return 40000
            # return t
        # if t % 1e6 == 0:
        #     with objmode():
        #         start = time.perf_counter()
        #         # start = time.time()
        #         print(f"iterations: {k}, time per {1e6}: {time.perf_counter() - start}")
        full_grad = full_grad + (stoch_grad - grad_phi) / n_samples
    # if fx_tmp[-1] > fx_tmp[0]:
    #     return int(1.5*max_iter)
    return t #max_iter




@njit
# def svrg_ols(A, b, step_size, x_0, max_iter, inner_iter=1, s=np.zeros(2)):
def svrg_ols(A, b, step_size, x_0, max_iter, inner_iter=1, rnd_seed=None, last_it=False):
    n_features = A.shape[1]
    n_samples = A.shape[0]
    new_x = x_0.copy()
    prev_x = new_x.copy()
    fx_tmp = np.zeros(max_iter*inner_iter+1)
    fx_tmp[0] = least_square_loss(A, b, x_0)
    fx_tmp_s = np.zeros(max_iter+1)
    fx_tmp_s[0] = least_square_loss(A, b, x_0)

    k = 0
    if not (rnd_seed == None):
        np.random.seed(rnd_seed)
    for t in range(max_iter):
        full_grad_previous = (A.T @ (A @ prev_x - b)) / n_samples
        for i in range(inner_iter):
            row_i = np.random.randint(n_samples)
            a_i = A[row_i]
            b_i = b[row_i]
            stoch_grad_previous = ((a_i @ prev_x) - b_i) * a_i
            stoch_grad_actual = ((a_i @ new_x) - b_i) * a_i
            v = stoch_grad_actual - stoch_grad_previous + full_grad_previous
            new_x -= step_size * v
            fx_tmp[k+1] = least_square_loss(A, b, new_x)
            k += 1
        #     if s.any():
        #         if i == s[t]:
        #             tmp = new_x.copy()
        # if s.any():
        #     new_x = tmp.copy()
        if not last_it:
            s_t = np.random.choice(inner_iter)
            ind = k - inner_iter + s_t
            fx_tmp_s[t+1] = fx_tmp[ind]
        else:
            fx_tmp_s[t+1] = fx_tmp[k-1]
        # fx_tmp_s[t] = least_square_loss(A, b, new_x)
        prev_x = new_x.copy()
    return fx_tmp_s, fx_tmp

@njit
def svrg_ols_acc(A, b, step_size, x_0, max_iter, inner_iter=1, accuracy=5e-2
                 , f_star=0., rnd_seed=None, max_it=int(4e4)):
    n_features = A.shape[1]
    n_samples = A.shape[0]
    new_x = x_0.copy()
    prev_x = new_x.copy()
    # fx_tmp = np.zeros(max_iter*inner_iter)
    fx_tmp = 34e10

    k = 0
    
    n_increase = 0
    max_n_increase = 100

    if not (rnd_seed == None):
        np.random.seed(rnd_seed)
    
    # t = -1
    # for t in range(max_iter):
    while (fx_tmp - f_star > accuracy): # and (k <= 1e7):
        full_grad_previous = (A.T @ (A @ prev_x - b)) / n_samples
        # if t == -1:
        #     k = 0
        #     t = 0
        for i in range(inner_iter):
            prev_f = fx_tmp
            row_i = np.random.randint(n_samples)
            a_i = A[row_i]
            b_i = b[row_i]
            stoch_grad_previous = ((a_i @ prev_x) - b_i) * a_i
            stoch_grad_actual = ((a_i @ new_x) - b_i) * a_i
            v = stoch_grad_actual - stoch_grad_previous + full_grad_previous
            new_x -= step_size * v
            # fx_tmp[k] = least_square_loss(A, b, new_x)
            fx_tmp = least_square_loss(A, b, new_x)
            if fx_tmp - f_star <= accuracy:
                return k + 1
            # If we have max_n_increase consecutive increases we stop
            if prev_f <= fx_tmp:
                n_increase += 1
            else:
                n_increase = 0
            # if k != 0:
            #     if fx_tmp[k] >= fx_tmp[k-1]:
            #         n_increase += 1
            #     else:
            #         n_increase = 0
            # if n_increase == max_n_increase or fx_tmp == np.inf or fx_tmp == np.nan:
            if fx_tmp == np.inf or fx_tmp == np.nan:
                return max_it
                # return -1 # int(1.5*max_iter*inner_iter)
            k += 1
            if k > max_it:
                return max_it
                # return k
            # if s.any():
            #     if i == s[t]:
            #         tmp = new_x.copy()
            # t += 1
        # if s.any():
        #     new_x = tmp.copy()
        prev_x = new_x.copy()
    # if fx_tmp[-1] > fx_tmp[0]:
    #     return int(1.5*max_iter*inner_iter)
    return k #max_iter*inner_iter

@njit
def saga_ols(A, b, step_size, x_0, max_iter, inner_iter=1, rnd_seed=None):
    n_features = A.shape[1]
    n_samples = A.shape[0]
    new_x = x_0.copy()
    grad_tab = np.zeros((n_samples, n_features))

    fx_tmp = np.zeros(max_iter+1)
    fx_tmp[0] = least_square_loss(A, b, x_0)

    for i in range(n_samples):
        a_i = A[i]
        b_i = b[i]
        grad_tab[i] = ((a_i @ new_x) - b_i) * a_i

    full_grad = np.sum(grad_tab, axis=0) / n_samples
    # full_grad_1 = full_grad
    if not (rnd_seed==None):
        np.random.seed(rnd_seed)
    for t in range(max_iter):
        row_i = np.random.randint(n_samples)
        a_i = A[row_i]
        b_i = b[row_i]
        stoch_grad = ((a_i @ new_x) - b_i) * a_i
        # full_grad = np.sum(grad_tab, axis=0) / n_samples
        # grad_phi = grad_tab[row_i].copy()
        # grad_tab[row_i] = stoch_grad.copy()
        # g_k = full_grad - grad_phi
        # v = stoch_grad + g_k
        # new_x -= step_size * v
        # fx_tmp[t] = least_square_loss(A, b, new_x)
        # full_grad = full_grad + (stoch_grad - grad_phi) / n_samples
        ###########################################################################grad_phi = grad_tab[row_i].copy()
        g_k = full_grad - grad_tab[row_i]
        v = stoch_grad + g_k
        new_x -= step_size * v
        fx_tmp[t+1] = least_square_loss(A, b, new_x)
        full_grad = full_grad + ((stoch_grad - grad_tab[row_i]) / n_samples)
        grad_tab[row_i] = stoch_grad
 
    return fx_tmp

@njit
def saga_ols_acc(A, b, step_size, x_0, max_iter, accuracy=5e-2, f_star=0., rnd_seed=None):
    n_features = A.shape[1]
    n_samples = A.shape[0]
    new_x = x_0.copy()
    grad_tab = np.zeros((n_samples, n_features))

    # fx_tmp = np.zeros(max_iter)
    fx_tmp = 34e10

    for i in range(n_samples):
        a_i = A[i]
        b_i = b[i]
        grad_tab[i] = ((a_i @ new_x) - b_i) * a_i

    full_grad = np.sum(grad_tab, axis=0) / n_samples
    full_grad_1 = full_grad

    t = 0
    # k = - 1
    
    n_increase = 0
    max_n_increase = 100
    if not (rnd_seed == None):
        np.random.seed(rnd_seed)
    
    # for t in range(max_iter):
    while (fx_tmp - f_star > accuracy): # and (t <= 1e7):
        # if k == -1:
        #     t = 0
        #     k = 0
        prev_f = fx_tmp
        row_i = np.random.randint(n_samples)
        a_i = A[row_i]
        b_i = b[row_i]
        stoch_grad = ((a_i @ new_x) - b_i) * a_i
        # full_grad = np.sum(grad_tab, axis=0) / n_samples
        grad_phi = grad_tab[row_i].copy()
        grad_tab[row_i] = stoch_grad.copy()
        g_k = full_grad - grad_phi
        v = stoch_grad + g_k
        new_x -= step_size * v
        # fx_tmp[t] = least_square_loss(A, b, new_x)
        fx_tmp = least_square_loss(A, b, new_x)
        if fx_tmp - f_star <= accuracy:
            return t + 1
        # If we have max_n_increase consecutive increases we stop
        if prev_f <= fx_tmp:
            n_increase += 1
        else:
            n_increase = 0
        # if t != 0:
        #     if fx_tmp[t] >= fx_tmp[t-1]:
        #         n_increase += 1
        #     else:
        #         n_increase = 0
        if n_increase == max_n_increase or fx_tmp == np.inf or fx_tmp == np.nan:
        # if fx_tmp == np.inf or fx_tmp == np.nan:
            return 40000
            # return -1 # int(1.5*max_iter)
        # if least_square_loss(A, b, new_x) - f_star <= accuracy:
        #     return t + 1
        t += 1
        if t > 40000:
            return 40000
            # return t
        full_grad = full_grad + (stoch_grad - grad_phi) / n_samples
    # if fx_tmp[-1] > fx_tmp[0]:
    #     return int(1.5*max_iter)
    return t #max_iter

@njit
def loopless_svrg_ols(A, b, step_size, x_0, n_iter, rndm_list, inner_iter=1):
    n_features = A.shape[1]
    n_samples = A.shape[0]
    # new_x = np.zeros(n_features)
    new_x = x_0.copy()
    fx_tmp = np.zeros(n_iter)
    # all_x = np.zeros((max_iter, n_features))
    full_grad_previous = A.T @ (A @ new_x - b)/n_samples
    for t in range(n_iter):
        row_i = np.random.randint(n_samples)
        a_i = A[row_i]
        b_i = b[row_i]
        if t == 0:
            stoch_grad_previous = ((a_i @ new_x )  - b_i)*a_i
        stoch_grad_actual = ((a_i @ new_x) - b_i) * a_i
        v = stoch_grad_actual - stoch_grad_previous + full_grad_previous
        if rndm_list[t]:
            full_grad_previous = A.T @ (A @ new_x - b)/n_samples
            stoch_grad_previous = ((a_i @ new_x )  - b_i)*a_i
        new_x -= step_size * v
        fx_tmp[t] = least_square_loss(A, b, new_x)
        # all_x[t] = new_x.copy()

    return fx_tmp#, all_x

@njit
def loopless_svrp_ols(A, b, step_size, x_0, n_iter, rndm_list, inner_iter=1):
    n_features = A.shape[1]
    n_samples = A.shape[0]
    new_x = x_0.copy()
    fx_tmp = np.zeros(n_iter)
    # all_x = np.zeros((max_iter, n_features))
    full_grad_previous = A.T @ (A @ new_x - b)/n_samples
    for t in range(n_iter):
        row_i = np.random.randint(n_samples)
        a_i = A[row_i]
        b_i = b[row_i]
        if t == 0:
            stoch_grad_previous = ((a_i @ new_x )  - b_i)*a_i
        v = new_x + step_size*stoch_grad_previous - step_size*full_grad_previous
        if rndm_list[t]:
            full_grad_previous = A.T @ (A @ new_x - b)/n_samples
            stoch_grad_previous = ((a_i @ new_x )  - b_i)*a_i
        new_x = compute_prox(a_i, b_i, step_size, v)
        fx_tmp[t] = least_square_loss(A, b, new_x)
        # all_x[t] = new_x.copy()
    return fx_tmp#, all_x


@njit
def grad_desc(A, b, L, max_iter):
    n_features = A.shape[1]
    n_samples = A.shape[0]
    new_x = np.zeros(n_features)
    step_size = 1. / L
    
    for t in range(max_iter):
        grad = A.T @ (A @ new_x - b) / n_samples
        new_x -= step_size*grad

    return new_x

def plot_acc(dico, mu, sigma, cond_n, n_steps, stepsizes, accuracy, excepted=[], save=False, step_rule="", root="", svrp=True, n=1, d=1, plot=False, log=False):
    root = root
    p = [d,n]
    if not svrp:
        mu_str = str(mu)[:5]
        # title = f"$cond(A^T A)$ = {int(cond_n**2)}, $\mu$ = {mu_str}"
        title_l = "$cond(A^T A)$= {}".format(100)
        title_r = "$d$, $n$= {}".format(p)
    else:
        mu_str = str(mu)[:5]
        # title = f"$\kappa$ = {str(cond_n)[:6]}, $\mu$ = {mu_str}"
        title_l = "$cond(A^T A)$= {}".format(100)
        title_r = "$d$, $n$= {}".format(p)

    # axis_x = np.arange(n_steps)
    dict_data = {}
    
    steps = np.zeros(n_steps)
    t = 0
    for k, v in dico.items():
        steps[t] = k
        t +=1
    steps.sort()
    stepsizes = list(steps)
    
    max_step = 0
    # print(dico, stepsizes)
    for index, k in enumerate(stepsizes):
        for key, value in dico[k].items():
            try:
                # if key in excepted or not value[1]:
                if key in excepted or not value[0]:
                    continue
            except ValueError:
                pass
            plt.figure(key[:2], constrained_layout=True, figsize=(12, 8))

            # val = value

            # if value[-1] > value[0] or (value == np.inf).any() or value[-1] > accuracy:
            #     num_steps = value.shape[0]# + 5000
            # else:
            #     # num_steps = val[val > accuracy].shape[0]
            #     # try:
            #     num_steps = np.argmax(value < accuracy)
            #     # tft = np.where(value < accuracy)[0][0]
            #     # if num_steps == 0:
            #     #     print(key, value < accuracy, np.argmax(value[value < accuracy]), tft)
            #     #     ppap
            #     # # except ValueError:
            #     # #     num_steps = value.shape[0]
            #     # # print("I am",np.argmax(val[val < accuracy]), num_steps)
            num_steps = value[0]
            max_nsteps = value[1]
            min_nsteps = value[2]
            if value[0] > max_step:
                max_step = value[0]
            try:
                dict_data[key]["nsteps"][index] = num_steps
                dict_data[key]["max_nsteps"][index] = max_nsteps
                dict_data[key]["min_nsteps"][index] = min_nsteps
            except KeyError:
                tmp = np.zeros(n_steps)
                tmp_max = np.zeros(n_steps)
                tmp_min = np.zeros(n_steps)
                tmp[0] = num_steps
                tmp_max[0] = max_nsteps
                tmp_min[0] = min_nsteps
                dict_data[key] = {"nsteps": tmp, "max_nsteps": tmp_max, "min_nsteps": tmp_min}
                # dict_data[key] = tmp
            except IndexError:
                print(index, dict_data[key], key)
            # if value[0] > max_step:
            #     max_step = value[0]
            # try:
            #     dict_data[key][index] = num_steps
            # except KeyError:
            #     tmp = np.zeros(n_steps)
            #     tmp[0] = num_steps 
            #     dict_data[key] = tmp
            # except IndexError:
            #     print(index, dict_data[key], key


    # not_any = []
    # for k, v in dict_data.items():
    #     if not v[0].any():
    #         not_any.append(k)
    # for k in not_any:
    #     dict_data.pop(k, None)
    
    # max_step_log10 = 10^(-(int(np.log10(max_step)) + 1))
    markers = [ '*', 'o', 'v', '1', 's', '+']
    id_mrkr = 0
    # print(dict_data)
    for key, value in dict_data.items():
        # if key in excepted or not value[1]:
        #     continue
        ###### replace values equal to -1 by something greater that max steps #######
        # value["nsteps"][value["nsteps"] == -1] = 40000 # max_step * 10
        # value["max_nsteps"][value["steps"] == -1] = 40000 # max_step * 10
        # value["min_nsteps"][value["steps"] == -1] = 40000 # max_step * 10
        # value["nsteps"][value["nsteps"] == -1] =
        marker = markers[id_mrkr]
        id_mrkr += 1

        abs_x = steps #list(range(n_steps))
        # abs_x = stepsizes #list(range(n_steps))
        plt.figure(key[:2])
        # plt.plot(abs_x, value, marker=marker,label=key, linewidth=3)
        # plt.semilogy(abs_x, value, marker=marker,label=key, linewidth=3)
#         plt.loglog(abs_x, value, marker=marker,label=key, linewidth=3)

#         plt.ylabel(f"#iter to accuracy = {accuracy}", fontweight='bold')
#         plt.xlabel("Stepsizes", fontweight='bold')
#         # plt.xlabel("Coefficients", fontweight='bold')
#         # plt.xticks(np.ar§ange(1, 1+n_steps, 2)) #(stepsizes)
#         plt.title(title, fontweight='bold')
#         plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.09),
#               fancybox=True, shadow=True, ncol=3)

        # plt.title('$\kappa$= {}'.format(100), fontsize='x-large', loc = "left",weight='bold')
        # plt.title('$d$, $n$= {}'.format(p), fontsize='x-large', loc = "right",weight='bold')
        #plt.title('f model: L, d= {}'.format(LU, d))
        plt.title(title_l, fontsize='x-large', loc = "left",weight='bold')
        plt.title(title_r, fontsize='x-large', loc = "right",weight='bold')
        plt.plot(abs_x,  value["nsteps"], marker=marker, lw=5, label=key)
        plt.fill_between(abs_x,  value["max_nsteps"],  value["min_nsteps"], alpha=.3)
        plt.grid(axis='y', color = 'grey', linestyle = '-', linewidth = .3)
        # plt.plot(list(stp.T), CountSVRG, '-o',lw=5, label="SVRG")
        plt.xlabel("Stepsizes",fontsize=36,weight='bold')
        plt.ylabel(f"#Iter. to accuracy$={accuracy}$",fontsize=32,weight='bold')
        if log:
            plt.yscale("log")
            #ax1.set_xscale("log")
            #ax1.set_yscale("log")
        plt.legend()
#         value[value == -1] = 40000 # max_step * 10
#         marker = markers[id_mrkr]
#         id_mrkr += 1

#         abs_x = steps #list(range(n_steps))
#         # abs_x = stepsizes #list(range(n_steps))
#         plt.figure(key[:2])
#         # plt.plot(abs_x, value, marker=marker,label=key, linewidth=3)
#         # plt.semilogy(abs_x, value, marker=marker,label=key, linewidth=3)
# #         plt.loglog(abs_x, value, marker=marker,label=key, linewidth=3)

# #         plt.ylabel(f"#iter to accuracy = {accuracy}", fontweight='bold')
# #         plt.xlabel("Stepsizes", fontweight='bold')
# #         # plt.xlabel("Coefficients", fontweight='bold')
# #         # plt.xticks(np.ar§ange(1, 1+n_steps, 2)) #(stepsizes)
# #         plt.title(title, fontweight='bold')
# #         plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.09),
# #               fancybox=True, shadow=True, ncol=3)

#         # plt.title('$\kappa$= {}'.format(100), fontsize='x-large', loc = "left",weight='bold')
#         # plt.title('$d$, $n$= {}'.format(p), fontsize='x-large', loc = "right",weight='bold')
#         #plt.title('f model: L, d= {}'.format(LU, d))
#         plt.title(title_l, fontsize='x-large', loc = "left",weight='bold')
#         plt.title(title_r, fontsize='x-large', loc = "right",weight='bold')
#         plt.plot(abs_x,  value, marker=marker, lw=5, label=key)
#         plt.grid(axis='y', color = 'grey', linestyle = '-', linewidth = .3)
#         # plt.plot(list(stp.T), CountSVRG, '-o',lw=5, label="SVRG")
#         plt.xlabel("Stepsizes",fontsize=36,weight='bold')
#         plt.ylabel(f"#Iter. to accuracy$={accuracy}$",fontsize=32,weight='bold')
#         #ax1.set_xscale("log")
#         #ax1.set_yscale("log")
#         plt.legend()

        if save:
            plt.savefig(f"{root}{key[:2]}_{step_rule}", format="png",
                            bbox_inches='tight', dpi=200)
    if plot:
        plt.show(block=False)
    plt.close('all')



def plot(dico, mu, sigma, cond_n, max_iter, norm_x, max_bound=0, excepted=[], normalized=True, save=False, step_rule="", root="", ols=True, n=1, d=1, plot=False):
    root = root
    x_true = 0.
    p = [d, n]
    # String for title
    if not ols:
        # title = f"$cond(A^T A)$ = {int(cond_n**2)}"
        # title = f"$n$ = {n}"
        title_l = "$cond(A^T A)$= {}".format(100)
        title_r = "$d$, $n$= {}".format(p)
    else:
        mu_str = str(mu)[:5]
        # title = f"$cond(A^T A)$ = {int(cond_n**2)}, $\mu$ = {mu_str}"
        # title = f"$n$ = {n}, $\mu$ = {mu_str}"
        title_l = "$cond(A^T A)$= {}".format(100)
        title_r = "$d$, $n$= {}".format(p)

    # Checking for algo that should be excluded and the same time creating corresponding figures.
    # for key, value in dico.items():
    #     try:
    #         if key in excepted or not value[1]:
    #             continue
    #     except ValueError:
    #         pass
    #     if (key[:2] == "SP"):
    #         # plot with the function values of stochastic prox
    #         plt.figure("Stoch_f", constrained_layout=True, figsize = [10, 9])
    #         # plt.figure("Stoch_f", constrained_layout=True, figsize = [6.4, 5.5], linewidth=3)
    #         plt.title(title)
    #     elif (key[:2] == "SG"):
    #         pass
    #     else:
    #         plt.figure(key[:2], constrained_layout=True, figsize = [10, 9]) #[6.4, 4.8])
    #         plt.title(title)
    plt.figure("all_fig", constrained_layout=True, figsize = [12, 8])
    for key, value in dico.items():
        try:
            if key in excepted or not value:
                continue
        except ValueError:
            pass

        if max_bound and max_bound < value.shape[1]:
            length = max_bound
        else:
            length = value.shape[1]

        if normalized and (max_iter+1 != length):
            normalize_f = max_iter / length
            abs_x_f = normalize_f * np.arange(length)
            abs_x_f[-1] = max_iter 
        else:
            # normalizer_f = 1
            abs_x_f = np.arange(length) 

        # abs_x_f = normalize_f * np.arange(length)
        # print(key, abs_x_f[:length], abs_x_f)
        
        value[value > 1e+30] = 1e+30

#         if (key[:2] == "SP") or (key[:2] == "SG"):
#             plt.figure("Stoch_f")
#             plt.semilogy(abs_x_f[:length], value[:length], '-', label=key, linewidth=3)
#             continue
#         else:
#             plt.figure("Stoch_f")
#             plt.semilogy(abs_x_f[:length], value[:length], '-',label=key, linewidth=3)#, color='sky blue')
#             plt.figure(key[:2])

#         plt.semilogy(abs_x_f[:length], value[:length], '-',label=key, linewidth=3)#, color='sky blue')
        
#         plt.ylabel(f"{string}")
#         plt.xlabel("Normalized iterations", fontweight='bold')
#         plt.title(title, fontweight="bold")
#         plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.11),
#               fancybox=True, shadow=True, ncol=3)

#         if save:
#             plt.savefig(f"{root}{key[:2]}_{step_rule}", format="png",
#                         bbox_inches='tight', dpi=200)
        
        # plt.semilogy(abs_x_f[:length], value[:length], '-', label=key, linewidth=3)
        
        mean = np.mean(value, axis=0)
        # # print(mean.shape)
        # std = np.std(fx, axis=0)
        val_max = np.max(value, axis=0)
        val_min = np.min(value, axis=0)
        # plt.plot(abs_x_f, mean, label=key, linewidth=3
        #     , fillstyle='none')
        # plt.fill_between(abs_x_f, mean + std, mean - std, alpha=.3)
        plt.plot(abs_x_f[:length], mean[:length], '-', label=key, lw=3)
        plt.fill_between(abs_x_f[:length], val_max[:length], val_min[:length], alpha=.3)
        
        # plt.semilogy(abs_x_f[:length], value[:length], '-', label=key, lw=5)
                
    if ols:
        # string = "$F(\\mathbf{x}^k) - F^*$"
        # plt.ylabel(f"{string}")
        # plt.xlabel("Normalized iterations", fontweight='bold')
        # plt.title(title, fontweight="bold")
        # plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.11),
        #       fancybox=True, shadow=True, ncol=3) 

        string = "$F(\\mathbf{\\tilde{x}}^s) - F^*$"
        plt.yscale("log")
        plt.title(title_l, fontsize='x-large', loc = "left",weight='bold')
        plt.title(title_r, fontsize='x-large', loc = "right",weight='bold')
        # plt.plot(abs_x,  value, marker=marker, lw=5, label=key)
        # plt.plot(list(stp.T), CountSVRG, '-o',lw=5, label="SVRG")
        plt.xlabel("Normalized iterations",fontsize=36,weight='bold')
        plt.ylabel(f"{string}",fontsize=32,weight='bold')
        #ax1.set_xscale("log")
        #ax1.set_yscale("log")
        # plt.grid(axis = 'y')
        plt.legend()

        # plt.figure("Stoch_f")
        # plt.ylabel(f"{string}")
        # plt.xlabel("Normalized iterations", fontweight='bold')
        # plt.title(title, fontweight="bold")
        # plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.11),
        #       fancybox=True, shadow=True, ncol=3)
        if save:
            plt.savefig(f"{root}ols_{step_rule}", format="png",
                                bbox_inches='tight', dpi=150)

        # plt.show(block=False)
        plt.close('all')
    else:
        string = "$F(\\mathbf{\\tilde{x}}^s) - F^*$"
        plt.yscale("log")
        plt.title(title_l, fontsize='x-large', loc = "left",weight='bold')
        plt.title(title_r, fontsize='x-large', loc = "right",weight='bold')
        # plt.plot(abs_x,  value, marker=marker, lw=5, label=key)
        # plt.plot(list(stp.T), CountSVRG, '-o',lw=5, label="SVRG")
        plt.xlabel("Normalized iterations",fontsize=36,weight='bold')
        plt.ylabel(f"{string}",fontsize=32,weight='bold')
        #ax1.set_xscale("log")
        #ax1.set_yscale("log")
        # plt.grid(axis = 'y')
        plt.legend()
        
        # string = "$F(\\mathbf{\\hat{x}}^k) - F^*$"
        # plt.ylabel(f"{string}")
        # plt.xlabel("Normalized iterations", fontweight='bold')
        # plt.title(title, fontweight="bold")
        # plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.11),
        #       fancybox=True, shadow=True, ncol=3)   
        if save:
            plt.savefig(f"{root}logistic_{step_rule}", format="png",
                                bbox_inches='tight', dpi=150)

        if plot:
            plt.show(block=False)
        plt.close('all')