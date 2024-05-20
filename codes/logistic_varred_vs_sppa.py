import numpy as np
from utils import plot, make_data_class, norm, grad_desc_logistic, logistic_loss, sapa_logistic, stoch_prox_logistic, svrp_logistic

if __name__ == "__main__":
    cond_n = 10.

    d = 500
    exp = .55
    s = 50
    mu = 0. #np.sqrt(n*0.)
    x_0 = np.zeros(d)
    c = 1.
    n_exp = 10
    
    mean = 0.
    std_dev = 1.
    
    save = True
    
    for n in [1000, 5000, 10000]:
        np.random.seed(42)

        m = 2*n
                
 
        total_elt_op = s * (n + m + 1) - n
        total_svrg = s 
        total_sppa = s * (n + m + 1)
        
        dict_data = {}

        if n_exp >= 1:

            f_star = np.zeros((n_exp,1))
            fx_2 = np.zeros((n_exp, total_sppa+1))

            fx_8 = np.zeros((n_exp, total_elt_op+1))

            # fx_3 = np.zeros(max_iter * inner_iters)
            fx_3 = np.zeros((n_exp, s+1))


            fx_2_ = np.zeros(total_sppa+1)

            fx_8_ = np.zeros(total_elt_op+1)

            fx_3_ = np.zeros(s+1)

            for i in range(n_exp):
                A, b, x_true = make_data_class(n, d,cond_numb_A=cond_n, prob_noise=0., mu=mu, control=True)

                 # https://arxiv.org/pdf/1810.08727.pdf
                L = .25 * np.max((A**2).sum(axis=1))

                L_grad = .25 * (norm(A,ord=2)**2) / n
                
                f_star_ = logistic_loss(A, b, grad_desc_logistic(A, b, L_grad, 100000))
 
                step_svrp = 1. / (5*L)
                # m = 2*n

                step_sapa = 1. / (5*L)
                
                print(f"cond {cond_n}, sigma {mu}, experience {i+1}")
                
                fx_8[i] = sapa_logistic(A, b, step_sapa, x_0, total_elt_op, rnd_seed=3)

                # s = np.random.choice(np.arange(inner_iters), max_iter)
                fx_3[i] =svrp_logistic(A, b, step_sapa, x_0, s, inner_iter=m, rnd_seed=3, last_it=False)[0]

                fx_2[i] = stoch_prox_logistic(A, b, x_0, total_sppa, L, c=c, exp=exp, rnd_seed=3)
                
                f_star[i] = min((np.min(fx_8[i]), np.min(fx_3[i]),
                                 np.min(fx_2[i]), f_star_))
                
        else:
            print("Wrong number of experiments. It should be greater or equal to 1")
            
        f_star = min((np.min(fx_8), np.min(fx_3),
                         np.min(fx_2), f_star_))
        dict_data  = {
                            "SAPA": (fx_8 - f_star),
                            "SPPA": (fx_2 - f_star),
                            "SVRP": (fx_3 - f_star),
                            }

        except_ = []
        plot(dict_data, mu, mu, cond_n, s, total_elt_op, max_bound=0, excepted=except_, 
             normalized=True, ols=False,
             step_rule=f"mu_{mu}_cond_{int(cond_n)}_n_func_{n}.png",
             root="images/timing/test_2/", save=save, n=n, d=d)