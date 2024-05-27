import numpy as np
from utils import plot, make_data_reg, norm, grad_desc, least_square_loss, sapa_ols, stoch_prox_least_square, svrp_ols

if __name__ == "__main__":
    cond_n = 10.
    d = 500
    exp = .55
    s = 50
    mu = 0.
    mu2 = 1
    # cond_n = 15.
    x_0 = np.zeros(d)
    c = 1.
    
    mean = 0.
    std_dev = 1.
    
    n_exp = 10
    
    save = True    
    
    sigma = mu


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

            fx_3 = np.zeros((n_exp, s+1))


            fx_2_ = np.zeros(total_sppa+1)

            fx_8_ = np.zeros(total_elt_op+1)

            fx_3_ = np.zeros(s+1)

            for i in range(n_exp):
                A, b, x_true, _ = make_data_reg(n, d, cond_numb_A=cond_n, noise_scale=0.2, 
                                                mu=np.sqrt(mu), mu2=np.sqrt(mu2),
                                                mean=mean, std_dev=std_dev)

                if not mu:
                    mu_ = mu2 #(_**2) #/ n
                else:
                    mu_ = mu

                L_grad = norm(A,ord=2)**2 / n
                x_opt = grad_desc(A, b, L_grad, 100000)
                f_star_ = least_square_loss(A, b, x_opt)

                L = np.max((A ** 2).sum(axis=1))

                step_svrp = 1. / (5*L)

                step_sapa = 1. / (5*L)
                
                print(f"cond {cond_n}, sigma {sigma}, experience {i+1}")
                
                fx_8[i] = sapa_ols(A, b, step_sapa, x_0, total_elt_op, rnd_seed=3)

                fx_3[i] = svrp_ols(A, b, step_svrp, x_0, total_svrg, inner_iter=m, rnd_seed=3)[0]

                fx_2[i] = stoch_prox_least_square(A, b, x_0, total_sppa, c=c, exp=exp, rnd_seed=3)
                f_star[i] = min((np.min(fx_8[i]), np.min(fx_3[i]),
                                 np.min(fx_2[i]), f_star_))

        else:
            print("Wrong number of experiments. It should be greater or equal to 1")
            
        
        dict_data  = {      "SAPA": (fx_8 - f_star),
                            "SPPA": (fx_2 - f_star),
                            "SVRP": (fx_3 - f_star),
                            }

        except_ = []
        plot(dict_data, mu_, sigma, cond_n, s, total_elt_op, max_bound=0, excepted=except_, 
             normalized=True, ols=True,
             step_rule=f"mu_{mu}_cond_{int(cond_n)}_n_func_{n}_n_exp_{n_exp}_review.png",
             root="../plots/varredVSsppa/", save=save, n=n, d=d)