# -*- coding: utf-8 -*-
from scipy import *
from scipy.integrate import quad
import numpy as np


def call_price(theta_1, theta_2, k_1, k_2, sigma_1, sigma_2, rho_1, rho_2, v1, v2 ,r ,T ,s0 ,K): # {parameters, vol_factors, option spec.}
    integrand = lambda phi: (np.exp(- 1j * phi * np.log(K))/ (1j * phi) *
                            (f(phi-1j, theta_1, theta_2, k_1, k_2, sigma_1, sigma_2, rho_1, rho_2, v1, v2, r, T, s0) + \
                             - K* f(phi, theta_1, theta_2, k_1, k_2, sigma_1, sigma_2, rho_1, rho_2, v1, v2, r, T, s0))).real
    integral = quad(integrand, 0, 100)[0]
    
    return 0.5*(s0 - K*np.exp(-r*T)) + (np.exp(-r*T)/np.pi)*integral

def f(x, theta_1, theta_2, k_1, k_2, sigma_1, sigma_2, rho_1, rho_2, v1, v2, r, T, s0):# { phi, parameters, vol_factors, option spec., status}
        
    d_1 = np.sqrt((k_1 - rho_1*sigma_1*x*1j)**2 + sigma_1**2*(x*1j + x**2))
    d_2 = np.sqrt((k_2 - rho_2*sigma_2*x*1j)**2 + sigma_2**2*(x*1j + x**2))
    
    # g_1 = (k_1 - rho_1*sigma_1*x*1j + d_1)/(k_1 - rho_1*sigma_1*x*1j - d_1)
    # g_2 = (k_2 - rho_2*sigma_2*x*1j + d_2)/(k_2 - rho_2*sigma_2*x*1j - d_2)
    
    # B_1 = (k_1-rho_1*sigma_1*x*1j-d_1)/sigma_1**2*((1-np.exp(d_1*T))/(1-g_1*np.exp(d_1*T)))
    # B_2 = (k_2-rho_2*sigma_2*x*1j-d_2)/sigma_2**2*((1-np.exp(d_2*T))/(1-g_2*np.exp(d_2*T)))
    
    # A = r*x*1j*T + \
    #     theta_1*_1/(sigma_1**2)*((k_1 - rho_1*sigma_1*x*1j + d_1)*T - 2*np.log((1-g_1*np.exp(d_1*T))/(1-g_1))) + \
    #     theta_2*_2/(sigma_2**2)*((k_2 - rho_2*sigma_2*x*1j + d_2)*T - 2*np.log((1-g_2*np.exp(d_2*T))/(1-g_2)))
    
    c_1 = (k_1 - rho_1*sigma_1*x*1j - d_1)/(k_1 - rho_1*sigma_1*x*1j + d_1)
    c_2 = (k_2 - rho_2*sigma_2*x*1j - d_2)/(k_2 - rho_2*sigma_2*x*1j + d_2)
    
    B_1 = (k_1-rho_1*sigma_1*x*1j-d_1)/sigma_1**2*((1-np.exp(-d_1*T))/(1-c_1*np.exp(-d_1*T)))
    B_2 = (k_2-rho_2*sigma_2*x*1j-d_2)/sigma_2**2*((1-np.exp(-d_2*T))/(1-c_2*np.exp(-d_2*T)))
    
    A = r*x*1j*T + \
    theta_1*k_1/(sigma_1**2)*((k_1 - rho_1*sigma_1*x*1j - d_1)*T - 2*np.log((1-c_1*np.exp(-d_1*T))/(1-c_1))) + \
    theta_2*k_2/(sigma_2**2)*((k_2 - rho_2*sigma_2*x*1j - d_2)*T - 2*np.log((1-c_2*np.exp(-d_2*T))/(1-c_2)))
    
    return np.exp(A+ 1j*x*np.log(s0)+B_1*v1+B_2*v2)
    
