# -*- coding: utf-8 -*-
"""
Created on Wed May 17 08:35:42 2023

@author: Tomas
"""

import numpy as np
from numpy import asarray
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray
from math import pi
from matplotlib import pyplot as plt
from PIL import Image
import time
def RS1_Free(Field_Input,z,wavelength,pixel_pitch_in,pixel_pitch_out,Output_shape):
    '''
    Function to cumpute the Raleygh Sommerfeld 1 diffraction integral wothout approximations or the use of FFT,
    but allowing to change the output sampling (pixel pitch and shape).
    ### Inputs: 
    * field - complex field to be diffracted
    * z - propagation distance
    * wavelength - wavelength of the light used
    * pixel_pitch_in - Sampling pitches of the input field as a (2,) list
    * pixel_pitch_out - Sampling pitches of the output field as a (2,) list
    * Output_shape - Shape of the output field as an tuple of integers
    '''


    dx = pixel_pitch_in[0] #Input Pixel Size X
    dy = pixel_pitch_in[1] #Input Pixel Size Y
    ds = dx*dy # Area differential for the integral
    dx_out = pixel_pitch_out[0] #Output Pixel Size X
    dy_out = pixel_pitch_out[1] #Output Pixel Size Y
    
    M,N = np.shape(Field_Input)
    (M2,N2) = Output_shape
    k = (2*np.pi)/wavelength # Wave number of the ilumination source
    

    U0 = np.zeros(Output_shape,dtype='complex_')
    U1 = Field_Input  #This will be the hologram plane 


    x_inp_lim = dx*int(N/2)
    y_inp_lim = dy*int(M/2)

    x_cord = np.linspace(-x_inp_lim , x_inp_lim , num = N)
    y_cord = np.linspace(-y_inp_lim , y_inp_lim , num = M)

    [X_inp,Y_inp] = np.meshgrid(x_cord,y_cord,indexing='xy')


    x_out_lim = dx_out*int(N2/2)
    y_out_lim = dy_out*int(M2/2)

    x_cord_out = np.linspace(-x_out_lim , x_out_lim , num = N2)
    y_cord_out = np.linspace(-y_out_lim , y_out_lim , num = M2)

    # The first pair of loops ranges over the points in the output plane in order to determine r01
    for x_sample in range(OutputShape[0]):
        x_fis_out = x_cord_out[x_sample]
        for y_sample in range(OutputShape[1]):
            y_fis_out = y_cord_out[y_sample]
            mr01 = np.sqrt(np.power(x_fis_out-X_inp,2)+np.power(y_fis_out-Y_inp,2)+(z)**2)
            Obliquity = (z)/ mr01
            kernel = np.exp(1j * k * mr01)/mr01
            dif = (1j*k)+(1/mr01)
            U0[y_sample,x_sample] = np.sum(U1 * dif * kernel * Obliquity * ds)
    U0 = -U0/(2*np.pi)
    Viewing_window = [-x_out_lim,x_out_lim,-y_out_lim,y_out_lim]
    return U0,Viewing_window
    



mod = SourceModule('''
__global__ void multiplicacion(float *idata_remap_real, float *idata_remap_imag, float *matriz_holo_real, float *matriz_holo_imag, float *d_temp13x, int width, int height)
{
	
	//Descriptores de cada hilo
	int col = blockIdx.x*blockDim.x + threadIdx.x;
	int fila = blockIdx.y*blockDim.y + threadIdx.y;

	d_temp13x[fila*width + col] = ((idata_remap_real[fila*width + col])*(matriz_holo_real[fila*width + col])) -
		((idata_remap_imag[fila*width + col])*(matriz_holo_imag[fila*width + col]));

	matriz_holo_imag[fila*width + col] = ((idata_remap_imag[fila*width + col])*(matriz_holo_real[fila*width + col])) +
		((idata_remap_real[fila*width + col])*(matriz_holo_imag[fila*width + col]));

	matriz_holo_real[fila*width + col] = d_temp13x[fila*width + col];

}''')

multiplicacion = mod.get_function("multiplicacion")
print(multiplicacion)
# Light source
wavelength = 4.338e-07 # Wavelength of the illumination Source
k = 2*pi/wavelength # Wave number of the ilumination source

#Simulation Control variables
L = 100e-3
output_z = 15e-3   # Z Component of the observation screen coordinates in m
prop_z = 100e-6
# Magn = np.abs(L/output_z) # Magnification of the LM
Magn = 1


signal_size = 128
OutputShape = 128
# dx = dy = 3.3e-6 #Pixel Size.
dx = dy = 1.69e-3 / signal_size
dx_out = dy_out =  dx/Magn # MULTIPLY BY THE MAGNIFICATION
# dx = dy = dx_out/Magn # COMMENT FOR RECONSTRUCTION

M = N = signal_size # Control of the size of the matrices


U_img = gpuarray.empty((N, M), np.float32) # Matrix to save the output intensity
ref_img = gpuarray.empty((N, M), np.float32)
var2_real = gpuarray.empty((N, M), np.float32)
var2_img = gpuarray.empty((N, M), np.float32)



