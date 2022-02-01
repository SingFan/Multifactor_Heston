# -*- coding: utf-8 -*-

# py_HestonModel2 from teramonagi

import numpy as np
import matplotlib.pyplot as plt
import two_factor_vol
from scipy.optimize import fmin, rosen

#sample market data
def sample_data():
    x = [x.split() for x in open('marketdata_hk.txt')]
    header = x[0]
    market_datas = []
    for market_data in x[1:]:
        market_datas.append([float(z) for z in market_data]) # [map(lambda z:float(z), market_data)]
    return (header, market_datas)


#parameter calibration(kappa, theta, sigma, rho, v0)
def calibrate(init_val, market_datas):
    def error(pars, market_datas):
        a_1, a_2, b_1, b_2, sigma_1, sigma_2, rho_1, rho_2, v1, v2 = pars
        print( a_1, a_2, b_1, b_2, sigma_1, sigma_2, rho_1, rho_2, v1, v2)
        result = 0.0
        for market_data in market_datas:
            s0, k, market_price, r, T = market_data 
            #print s0, k, market_price, r, T
            model_price = two_factor_vol.call_price(a_1, a_2, b_1, b_2, sigma_1, sigma_2, rho_1, rho_2, v1, v2 , r ,T , s0 , k)
            result += ((model_price - market_price)/market_price)**2
        
        if min(sigma_1, sigma_2) <= 0:
            result += 1e6
        if min(a_1, a_2) <= 0:
            result += 1e6
        if min(b_1, b_2) <= 0:
            result += 1e6
        if min(v1, v2) <= 0:
            result += 1e6
        # if max(sigma_1**2 - 2*a_1, sigma_2**2 - 2*a_2) >= 0:
        #     result += 1e6
        print( result)   
        return result/len(market_datas)*100
    
    opt = fmin(error, init_val, args = (market_datas,), maxiter = 400)
    return opt
    
if __name__ == '__main__':
    #load market data
    header, market_datas = sample_data()
    #Initialize {a_1, a_2, b_1, b_2, sigma_1, sigma_2, rho_1, rho_2, v1, v2}
    init_val = [0.6, 0.6, 0.1, 0.02, 0.6, 0.6, 0, 0, 0.5, 0.5]
    #calibration of parameters
    theta_1, theta_2, k_1, k_2, sigma_1, sigma_2, rho_1, rho_2, v1, v2 = calibrate(init_val, market_datas)
    #
    market_prices = np.array([])
    model_prices  = np.array([])
    K = np.array([])
    for market_data in market_datas:
        s0, k, market_price, r, T = market_data 
        model_prices  = np.append(model_prices, two_factor_vol.call_price(theta_1, theta_2, k_1, k_2, sigma_1, sigma_2, rho_1, rho_2, v1, v2 , r ,T , s0 , k))
        market_prices = np.append(market_prices, market_price)
        K = np.append(K,k)
    #plot result
    plt.figure()
    plt.plot(K, market_prices, 'g*', K, model_prices, 'b*')
    plt.xlabel('Strike (K)')
    plt.ylabel('Price')
    plt.show()

    plt.title('Calibration for call option with multifactor Heston model')
    plt.savefig('Calibration for call option with multifactor Heston model.png')










