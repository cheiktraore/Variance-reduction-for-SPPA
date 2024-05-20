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
            step = 1./ (4*L)
            tmp = np.linspace(0,40,n_steps)
            tmp = tmp[tmp!=0]
            stepsizes = tmp * step

        for index, step in enumerate(stepsizes):
 
            tmp_sapa = sapa_logistic_acc(A, b, step, x_0, total_elt_op, f_star = f_star,
                                     accuracy=acc, rnd_seed=3)
            tmp_saga = saga_logistic_acc(A, b, step, x_0, total_elt_op, f_star = f_star,
                                     accuracy=acc, rnd_seed=3)
            try:
                # print(f"I am doing sapa now for step = {step}")
                dict_data[step]["SAPA"][0] += tmp_sapa / numb_exp
                dict_data[step]["SAPA"][1] = max(dict_data[step]["SAPA"][1], tmp_sapa)
                dict_data[step]["SAPA"][2] = min(dict_data[step]["SAPA"][2], tmp_sapa)

                # print(f"I am doing saga now for step = {step}")
                dict_data[step]["SAGA"][0] += tmp_saga / numb_exp
                dict_data[step]["SAGA"][1] = max(dict_data[step]["SAGA"][1], tmp_saga)
                dict_data[step]["SAGA"][2] = min(dict_data[step]["SAGA"][2], tmp_saga)
            except:
                dict_data[step] = {"SAPA": [tmp_sapa / numb_exp, tmp_sapa, tmp_sapa], "SAGA": [tmp_saga / numb_exp, tmp_saga, tmp_saga]}

    except_ = []
    # print("Hi Cheik")
    # plot_acc(dict_data, mu_, sigma, cond_n, n_steps-1, coeffs, accuracy, excepted=except_,
    plot_acc(dict_data, mu_, sigma, cond_n, len(stepsizes), stepsizes, accuracy, excepted=except_,
step_rule=f"logistic_sapaVSsaga_acc_{acc}_mu_{mu}_cond_{int(cond_n)}_n_{n}_blocks_{d}.png", root="./plots/sapaVSsaga/",
             save=save, svrp=False, n=n, d=d, plot=plot,log=True)

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

    d = 500
    x_0 = np.zeros(d)
    for n in [1000, 5000, 10000]:
    # for n in [5000, 10000]:
        # np.random.seed(42)
        # n_steps = 101
        dict_steps = run_all_for_iter_to_accuracy_plot(n, d, mu, cond_n, max_iter, x_0, 
                                                       coef_step, n_steps, mean=0., 
                                                       std_dev=1., save=True, acc=accuracy,
                                                       inner_iters=1, 
                                                       numb_exp=n_exp,
                                                       last_it=True)