'''
Code developed by Tomás Vélez Acosta
'''

from math import pi
import numpy as np
import math as mt
import pdb
import time
from matplotlib import pyplot as plt

def ftx(input):
    fts = np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(input)))
    return fts


def iftx(input):
    fts = np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(input)))
    return fts


def SAASM(field, z, wavelength, pixel_pitch_in ,pixel_pitch_out):
    '''
    # Function to diffract a complex field using the angular spectrum approach with a Semi-Analytical spherical wavefront.
    # For further reference review: 

    
    # Inputs:
    # field - complex field
    # z - propagation distance
    # wavelength - wavelength
    # dx/dy - sampling pitches
    '''
    k_wl = 2 * pi / wavelength
    

    M, N = field.shape
    x = np.arange(0, N, 1)  # array x
    fx = np.fft.fftshift(np.fft.fftfreq(N,pixel_pitch_in[0]))
    fx_out = np.fft.fftshift(np.fft.fftfreq(N,pixel_pitch_out[0]))

    y = np.arange(0, M, 1)  # array y
    fy = np.fft.fftshift(np.fft.fftfreq(M,pixel_pitch_in[1]))
    fy_out = np.fft.fftshift(np.fft.fftfreq(M,pixel_pitch_out[1]))

    X, Y = np.meshgrid(x - (N / 2), y - (M / 2), indexing='xy')
    FX, FY = np.meshgrid(fx, fy, indexing='xy')
    FX_out, FY_out = np.meshgrid(fx_out, fy_out, indexing='xy')
    
    MR = (X**2 + Y**2)
    MF = np.sqrt(FX**2 + FY**2)
    MF_out = np.sqrt(FX_out**2 + FY_out**2)
    
    
    kmax = np.amax(MF)
    h = np.ones_like(MR)


    bX = -kmax * FX / (2*d*z)
    bY = -kmax * FY / (2*d*z)
    Mbeta = np.sqrt(np.power(bX,2)+np.power(bY,2))
    
    kz = (k_wl/kmax)
    c = 2/3 * kz + 2/3 * np.sqrt(kz**2 - 0.5)- 1/3 * np.sqrt(kz**2 -1)
    d = 1/3 * np.sqrt(kz**2 - 1) - kz
    alpha = np.exp(-1j* c * kmax * z/(2j * d * z)) * np.exp((1j * kmax * (MR))/(4*d*z))


    kernel = np.exp(1j * kmax * np.power(Mbeta,2)/(4*d*z))
    



    E_out = iftx(ftx(ftx(np.divide(field/alpha)) * kernel) * np.exp(1j * z * h))
    
	
    return E_out
































