import sys
import numpy as np
from numpy.linalg import norm, cond
# from utils import run_all_for_iter_to_accuracy_plot
from utils import logistic_loss, grad_desc_logistic, plot_acc, make_data_class, sapa_logistic_acc, saga_logistic_acc, svrg_logistic_acc, svrp_logistic_acc

def run_all_for_iter_to_accuracy_plot(n, d, mu, cond_n, s, x_0, coef_step, n_steps,
                                      mean=0.,std_dev=1., inner_iters=0, numb_exp=1, 
                                      save=False, acc=5e-2,
                                      root="", last_it=True, plot=False):
    
    sigma = mu
    mu_ = mu
    np.random.seed(42)
    dict_data = {}

    for exp in range(numb_exp):

        A, b, x_true = make_data_class(n, d,cond_numb_A=cond_n, prob_noise=0., mu=mu, control=True)

         # https://arxiv.org/pdf/1810.08727.pdf
        L = .25 * np.max((A**2).sum(axis=1))

        L_grad = .25 * (norm(A,ord=2)**2) / n
        
        # f_star_ = logistic_loss(A, b, grad_desc_logistic(A, b, L_grad, 10000000))
        f_star = logistic_loss(A, b, grad_desc_logistic(A, b, L_grad, 100000))
        # f_star = 0

        #         step_svrp = 1. / (coeffs[0]*L)
        #         m = 1. /(mu_*step_svrp*(1 - 2*step_svrp*(2*L-mu_)))
        #         inner_iters = int(m) + 1

        #         step_sapa = 1. / (coeffs[1]*L)

        step_svrp = 1. / (5*L)
        # m = 2*n

        step_sapa = 1. / (5*L)

        m = 2*n
                
#         step_sapa = 1. / (5*L)
 
        total_elt_op = s * (n + m + 1) - n
        total_svrg = s 
        total_sppa = s * (n + m + 1)
    
        if not exp:
            # bounds_dict = {"500": (0.01, 4.5), "1000": (0.05, 2.7), "1500": (0.05, 2.5), "2000": (0.05, 1.4)}
            # bounds_dict = {"500": (0.01, 5.5), "1000": (0.05, 3.2), "1500": (0.05, 3), "2000": (0.05, 2)}
            bounds_dict = {"500": (0.01, 6.5), "1000": (0.05, 4.2), "1500": (0.05, 4), "2000": (0.05, 3), "3000": (0.05, 3), "4000": (0.05, 3)}
            low_bound, upp_bound = bounds_dict[str(d)]
            stepsizes = (upp_bound + low_bound) - np.geomspace(upp_bound,low_bound,n_steps)
            ##################################
            # step = 1./ (4*L)
            # tmp = np.linspace(0,160,n_steps)
            # tmp = tmp[tmp!=0]
            # # upper_bound = 1./ (2*(2*L-mu_))
            # upper_bound = 1./ (2*(2*L))
            # stepsizes = tmp * step
            # stepsizes = stepsizes[stepsizes!=upper_bound]
        for index, step in enumerate(stepsizes):
 
            tmp_svrg = svrg_logistic_acc(A, b, step, x_0, total_svrg, f_star = f_star,
                                  inner_iter=inner_iters, 
                                  accuracy=acc, rnd_seed=3)

            tmp_svrp = svrp_logistic_acc(A, b, step, x_0, total_svrg, inner_iter=inner_iters, f_star = f_star
                                  , accuracy=acc, rnd_seed=3)
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

    except_ = []
    # print("Hi Cheik")
    # plot_acc(dict_data, mu_, sigma, cond_n, n_steps-1, coeffs, accuracy, excepted=except_,
    plot_acc(dict_data, mu_, sigma, L / mu_, len(stepsizes), stepsizes, accuracy, excepted=except_,
step_rule=f"logistic_svrpVSsvrg_acc_{acc}_mu_{mu}_cond_{int(cond_n)}_n_{n}_blocks_{d}_in_{inner_iters}.png", root="../plots/svrpVSsvrg/",
             save=save, svrp=True, n=n, d=d, plot=plot, log=True)

    return dict_data

if __name__ == "__main__":
    # cond_ns = [1., 5., 20.]
    # cond_ns = [1., 2., 5.]
    # cond_ns = [20.]
    # mus = [0., 0.6, 1]
    mus = [0.]
    # n, m = 1000, 500
    # d = 500
    c = .55
    max_iter = 40
    inner_iters=1000
    # mu = 0.
    coef_step = 6.
    
    
    n_exp = 10
    
    accuracy = 1e-2

    cond_n = 10.
    mu = 0.
    n_steps = 51

    n = 2000
    print(n)
    for d in [500, 1000, 1500, 2000]:
    # for n in [1000]:
        # np.random.seed(42)
        # n_steps = 101
        x_0 = np.zeros(d)
        dict_steps = run_all_for_iter_to_accuracy_plot(n, d, mu, cond_n, max_iter, x_0, 
                                                       coef_step, n_steps, mean=0., 
                                                       std_dev=1., save=True, acc=accuracy,
                                                       inner_iters=inner_iters, 
                                                       numb_exp=n_exp,
                                                       last_it=True)