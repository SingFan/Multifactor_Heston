# -*- coding: utf-8 -*-
from scipy import *
from scipy.integrate import quad
import numpy as np
#public

# call_price = s0 * p1 - K * exp(-r * T) * p2
# p1 = __p1, where __p1 = 1/2 + 1/pi * integrand_1
# p2 = __p2, where __p2 = 1/2 + 1/pi * integrand_2

# =============================================================================
# 
# =============================================================================

def call_price(a_1, a_2, b_1, b_2, sigma_1, sigma_2, rho_1, rho_2, v1, v2 ,r ,T ,s0 ,K): # {parameters, vol_factors, option spec.}
    p1 = p(a_1, a_2, b_1, b_2, sigma_1, sigma_2, rho_1, rho_2, v1, v2 ,r ,T ,s0 ,K, 1)
    p2 = p(a_1, a_2, b_1, b_2, sigma_1, sigma_2, rho_1, rho_2, v1, v2 ,r ,T ,s0 ,K, 2)
    return (s0 * p1 - K * np.exp(-r * T) * p2)

#private
def p(a_1, a_2, b_1, b_2, sigma_1, sigma_2, rho_1, rho_2, v1, v2 ,r ,T ,s0 , K, status):
    
    if status == 1:
        integrand = lambda phi: (np.exp(1j * phi * log(s0/K)) * f(phi+1, a_1, a_2, b_1, b_2, sigma_1, sigma_2, rho_1, rho_2, v1, v2, r, T, s0, status) / (1j * phi * s0 * np.exp(r*T))).real  
    else:
        integrand = lambda phi: (np.exp(1j * phi * log(s0/K)) * f(phi, a_1, a_2, b_1, b_2, sigma_1, sigma_2, rho_1, rho_2, v1, v2, r, T, s0, status) / (1j * phi)).real  
    
    return (0.5 + (1 / pi) * quad(integrand, 0, 100)[0])

def f(x, a_1, a_2, b_1, b_2, sigma_1, sigma_2, rho_1, rho_2, v1, v2, r, T, s0, status):# { phi, parameters, vol_factors, option spec., status}
    
    # if status == 1:
    #     u_status = 1
    #     b_1 = b_1 - rho_1 * sigma_1
    #     b_2 = b_2 - rho_2 * sigma_2
    # else:
    #     u_status = -1
    #     b_1 = b_1
    #     b_2 = b_2

    # d_1 = np.sqrt((rho_1*sigma_1*x*1j - b_1)**2 + sigma_1**2*(u_status*x*1j + x**2))
    # d_2 = np.sqrt((rho_2*sigma_2*x*1j - b_2)**2 + sigma_2**2*(u_status*x*1j + x**2))
    
    # g_1 = (b_1 - rho_1*sigma_1*x*1j + d_1)/(b_1 - rho_1*sigma_1*x*1j - d_1)
    # g_2 = (b_2 - rho_2*sigma_2*x*1j + d_2)/(b_2 - rho_2*sigma_2*x*1j - d_2)
    
    # B_1 = (b_1-rho_1*sigma_1*x*1j+d_1)/sigma_1**2*((1-np.exp(d_1*T))/(1-g_1*np.exp(d_1*T)))
    # B_2 = (b_2-rho_2*sigma_2*x*1j+d_2)/sigma_2**2*((1-np.exp(d_2*T))/(1-g_2*np.exp(d_2*T)))
    
    # A = r*x*1j*T + \
    #     a_1/(sigma_1**2)*((b_1 - rho_1*sigma_1*x*1j + d_1)*T - 2*np.log((1-g_1*np.exp(d_1*T))/(1-g_1))) + \
    #     a_2/(sigma_2**2)*((b_2 - rho_2*sigma_2*x*1j + d_2)*T - 2*np.log((1-g_2*np.exp(d_2*T))/(1-g_2)))
    
    d_1 = np.sqrt((rho_1*sigma_1*x*1j - b_1)**2 + sigma_1**2*(x*1j + x**2))
    d_2 = np.sqrt((rho_2*sigma_2*x*1j - b_2)**2 + sigma_2**2*(x*1j + x**2))
    
    g_1 = (b_1 - rho_1*sigma_1*x*1j + d_1)/(b_1 - rho_1*sigma_1*x*1j - d_1)
    g_2 = (b_2 - rho_2*sigma_2*x*1j + d_2)/(b_2 - rho_2*sigma_2*x*1j - d_2)
    
    B_1 = (b_1-rho_1*sigma_1*x*1j+d_1)/sigma_1**2*((1-np.exp(d_1*T))/(1-g_1*np.exp(d_1*T)))
    B_2 = (b_2-rho_2*sigma_2*x*1j+d_2)/sigma_2**2*((1-np.exp(d_2*T))/(1-g_2*np.exp(d_2*T)))
    
    A = r*x*1j*T + \
        a_1/(sigma_1**2)*((b_1 - rho_1*sigma_1*x*1j + d_1)*T - 2*np.log((1-g_1*np.exp(d_1*T))/(1-g_1))) + \
        a_2/(sigma_2**2)*((b_2 - rho_2*sigma_2*x*1j + d_2)*T - 2*np.log((1-g_2*np.exp(d_2*T))/(1-g_2)))
    return np.exp(A+B_1*v1+B_2*v2)

    
