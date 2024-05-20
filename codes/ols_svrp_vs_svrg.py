import sys
import numpy as np
from numpy.linalg import norm
# from utils import run_all_for_iter_to_accuracy_plot
from utils import plot_acc, make_data_reg, sapa_ols_acc, saga_ols_acc, svrg_ols_acc, svrp_ols_acc, grad_desc, least_square_loss

def run_all_for_iter_to_accuracy_plot(n, d, mu, mu2, cond_n, s, x_0, coef_step, n_steps,
                                      mean=0.,std_dev=1., inner_iters=1, numb_exp=1, 
                                      save=False, acc=1e-2,
                                      root="", last_it=True, max_it=40000):
    
    sigma = mu
    np.random.seed(6)
    dict_data = {}

    for exp in range(numb_exp):

        A, b, x_true, _ = make_data_reg(n, d, cond_numb_A=cond_n, noise_scale=0., 
                                        mu=np.sqrt(mu), mu2=np.sqrt(mu2), mean=mean, std_dev=std_dev)
        
        if not mu:
            mu_ = (_**2) #/ n
        else:
            mu_ = mu
    
        L = np.max((A ** 2).sum(axis=1))
        
        L_grad = norm(A,ord=2)**2 / n
        x_opt = grad_desc(A, b, L_grad, 100000)
        f_star = least_square_loss(A, b, x_opt)
    
        m_fix = int((8*(2*L-mu_)) / mu_) + 1
        delta = mu_**2 * m_fix**2 - 8 * (2*L - mu_)*mu_*m_fix
        dem = 4*(2*L - mu_)*mu_*m_fix
        step = .5 * ( ((mu_*m_fix + np.sqrt(delta)) / dem) 
                  - ((mu_*m_fix + np.sqrt(delta)) / dem) )
        inner_iter = m_fix
    
        total_elt_op = s * (n + inner_iter + 1) - n # + (inner_iters + 1)
        total_svrg = s # + 1
        total_sppa = s * (n + inner_iter + 1)
    
        if not exp:
            # bounds_dict = {"500": (0.01, 1.5), "1000": (0.05, 1.1), "1500": (0.05, 0.85), "2000": (0.05, 0.7),\
            #                "3000": (0.05, 0.76), "4000": (0.05, 0.7)}
            # low_bound, upp_bound = bounds_dict[str(d)]
            # stepsizes = (upp_bound + low_bound) - np.geomspace(upp_bound,low_bound,n_steps)
            ##################################
            step = 1./ (4*L)
            tmp = np.linspace(0,30,n_steps)
            tmp = tmp[tmp!=0]
            upper_bound = 1./ (2*(2*L-mu_))
            stepsizes = tmp * step
            stepsizes = stepsizes[stepsizes!=upper_bound]
        
        for index, step in enumerate(stepsizes):

            tmp_svrg = svrg_ols_acc(A, b, step, x_0, s, f_star = f_star, 
                                  inner_iter=inner_iters, 
                                  accuracy=acc, rnd_seed=3, max_it=max_it)

            tmp_svrp = svrp_ols_acc(A, b, step, x_0, s, inner_iter=inner_iters, f_star = f_star
                                  , accuracy=acc, rnd_seed=3, max_it=max_it)
            try:
                # print(f"I am doing svrg now for step = {step}")
                dict_data[step]["SVRP"][0] += tmp_svrp / numb_exp
                dict_data[step]["SVRP"][1] = max(dict_data[step]["SVRP"][1], tmp_svrp)
                dict_data[step]["SVRP"][2] = min(dict_data[step]["SVRP"][2], tmp_svrp)

                # print(f"I am doing svrp now for step = {step}")
                dict_data[step]["SVRG"][0] += tmp_svrg / numb_exp
                dict_data[step]["SVRG"][1] = max(dict_data[step]["SVRG"][1], tmp_svrg)
                dict_data[step]["SVRG"][2] = min(dict_data[step]["SVRG"][2], tmp_svrg)
            except:
                dict_data[step] = {"SVRP": [tmp_svrp  / numb_exp, tmp_svrp, tmp_svrp], "SVRG": [tmp_svrg / numb_exp, tmp_svrg, tmp_svrg]}                    

    
    accuracy = acc
    except_ = []
    # print("Hi Cheik")
    # plot_acc(dict_data, mu_, sigma, cond_n, n_steps-1, coeffs, accuracy, excepted=except_,
    plot_acc(dict_data, mu_, sigma, L / mu_, len(stepsizes), stepsizes, accuracy, excepted=except_,
step_rule=f"ols_svrpVSsvrg_acc_{acc}_mu_{mu}_cond_{int(cond_n)}_n_{n}_blocks_{d}_in_{inner_iters}.png", root="../plots/svrpVSsvrg/",
             save=save, svrp=True, n=n, d=d)

    return dict_data

if __name__ == "__main__":
    # cond_ns = [1., 5., 20.]
    # cond_ns = [1., 2., 5.]
    cond_ns = [20.]
    # mus = [0., 0.6, 1]
    mus = [0.]
    # n, m = 1000, 500
    # d = 50
    c = .55
    max_iter = 40
    inner_iters=1000
    # mu = 0.
    coef_step = 6.
    
    n_exp = 10
    
    accuracy = 1e-2

    cond_n = 10.
    mu = 0.
    mu2 = 0.1
    d = 500
    
    n_steps = 51

    n = 2000
    for d in [1000, 1500, 2000, 3000]:
    # for d in [500]:
        x_0 = np.zeros(d)
        dict_steps = run_all_for_iter_to_accuracy_plot(n, d, mu, mu2, cond_n, max_iter, x_0, 
                                                       coef_step, n_steps, mean=0., 
                                                       std_dev=1., save=True, acc=accuracy,
                                                       inner_iters=inner_iters, 
                                                       numb_exp=n_exp, numb_pb=n_pb,
                                                       last_it=True, max_it=60000)