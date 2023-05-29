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
    
def amplitude (inp, log):
    '''
    # Function to calcule the amplitude representation of a given complex field
    # Inputs:
    # inp - The input complex field
    # log - boolean variable to determine if a log representation is applied    
    '''
    out = np.abs(inp)
    out = out / np.amax(out)
    if log == True:
        out = 20 * np.log(out)
    return out

def intensity (inp, log):
    out1 = amplitude(inp,False)
    out = np.power(out1,2)
    if log == True:
        out = 20 * np.log(out)
    return out

def ploty2(Input,Output):
    fig,axs = plt.subplots(1,2)

    axs[0].imshow(intensity(Input,'False'), cmap='gray')
    axs[0].set_title('Input Intensity')
    
    axs[1].imshow(intensity(Output,'False'), cmap='gray')
    axs[1].set_title('Output Intensity')

    plt.subplots_adjust(wspace=0.2)
    plt.subplots_adjust(hspace=0.4)
    plt.show()

mod = SourceModule('''
                   
#include <math.h>
#include <cuComplex.h>
#include <complex.h>

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

}

__global__ void matrix(cuComplex *U_field, cuComplex *U_temp,float *z,float *k,float *x_out, float *y_out,float *X_inp,float *Y_inp, float *mr01, float *Obliquity, cuComplex *kernel,cuComplex *dif ,int width, int height)
{
	//Descriptores de cada hilo
	int col = blockIdx.x*blockDim.x + threadIdx.x;
	int fila = blockIdx.y*blockDim.y + threadIdx.y;
    int i2= ((fila*N)+col);
    
    mr01[i2] = sqrt(pow(x_fis_out - X_inp[i2], 2) + pow(y_fis_out - Y_inp[i2], 2) + pow(z, 2));
    Obliquity[i2] = z / mr01[];
    kernel[i2] = cuCdivf(cuCexpf(cuCmulf(make_cuComplex(0, k), make_cuComplex(mr01[i2], 0))), make_cuComplex(mr01[i2], 0))
    dif[i2] = make_cuComplex(0, k);
    dif[i2] = cuCaddf(dif[i2], cuCdivf(make_cuComplex(1, 0), make_cuComplex(mr01[i2], 0)));
    U_temp[i2] = cuCmulf(Obliquity[i2],cuCmulf(kernel[i2],cuCmulf(U_field[i2],dif[i2])));
}

__global__ void sum2DArray(cuComplex* U_temp, int rows, int columns, cuComplex* result) {
    __shared__ cuComplex sharedSum[BLOCK_SIZE];

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int localIdx = threadIdx.x;

    // Perform reduction within each block
    cuComplex sum = make_cuComplex(0, 0);
    for (int i = tid; i < rows * columns; i += blockDim.x * gridDim.x) {
        sum = cuCaddf(sum, U_temp[i]);
    }
    sharedSum[localIdx] = sum;

    // Synchronize within the block
    __syncthreads();

    // Perform reduction between threads in each block
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (localIdx < stride) {
            sharedSum[localIdx] = cuCaddf(sharedSum[localIdx], sharedSum[localIdx + stride]);
        }
        __syncthreads();
    }

    // Store the block sum in global memory
    if (localIdx == 0) {
        result[blockIdx.x] = sharedSum[0];
    }
}

__global__ void out_calc(cuComplex *U_out, cuComplex *result, int width, int height, int row, int column)
{
    U_out[row * width + column] = result[0]
}




''')

multiplicacion = mod.get_function("multiplicacion")
sum2DArray = mod.get_function("sum2DArray")
matrix = mod.get_function("matrix")
out_calc = mod.get_function("out_calc")

# Light source
wavelength = 4.338e-07 # Wavelength of the illumination Source
k = 2*pi/wavelength # Wave number of the ilumination source
k = k.astype(np.float32)

#Simulation Control variables
L = 100e-3
z = 100e-6
z = z.astype(np.float32)
# Magn = np.abs(L/output_z) # Magnification of the LM
Magn = 1




signal_size = 128
Outputsize = 128
OutputShape = (Outputsize,Outputsize)
# dx = dy = 3.3e-6 #Pixel Size.
dx = dy = 1.69e-3 / signal_size
ds = dx*dy # Area differential for the integral
dx_out = dy_out =  dx/Magn # MULTIPLY BY THE MAGNIFICATION
# dx = dy = dx_out/Magn # COMMENT FOR RECONSTRUCTION


im = Image.open(r"D:\OneDrive - Universidad EAFIT\Semestre IX\Advanced Project 2\USAF-1951.svg.png").convert('L')
im = im.resize((signal_size,signal_size))
im = np.asarray(im)/255
im = im.astype(np.complex64)

(M,N) = np.shape(im) # Control of the size of the matrices

# GPU Parameters
block_dim = (16, 16, 1)

grid_dim = (N // (block_dim[0]), M // (block_dim[1]), 1)

U_field = gpuarray.to_gpu(im) # Matrix to save the output field
U_out = gpuarray.empty((OutputShape, OutputShape), np.complex64)
U_temp = gpuarray.empty((N, M), np.complex64)
mr01 = gpuarray.empty((N, M), np.float32)
Obliquity = gpuarray.empty((N, M), np.float32)
kernel = gpuarray.empty((N, M), np.complex64)
dif = gpuarray.empty((N, M), np.complex64)






M,N = np.shape(U_field)
(M2,N2) = OutputShape
k = (2*np.pi)/wavelength # Wave number of the ilumination source


U0 = np.zeros(OutputShape,dtype='complex_')
U1 = im  #This will be the hologram plane 


x_inp_lim = dx*int(N/2)
y_inp_lim = dy*int(M/2)

x_cord = np.linspace(-x_inp_lim , x_inp_lim , num = N)
y_cord = np.linspace(-y_inp_lim , y_inp_lim , num = M)

[X_inp,Y_inp] = np.meshgrid(x_cord,y_cord,indexing='xy')
X_gpu = gpuarray.to_gpu(X_inp)
Y_gpu = gpuarray.to_gpu(Y_inp)

x_out_lim = dx_out*int(N2/2)
y_out_lim = dy_out*int(M2/2)

x_cord_out = np.linspace(-x_out_lim , x_out_lim , num = N2)
y_cord_out = np.linspace(-y_out_lim , y_out_lim , num = M2)

# The first pair of loops ranges over the points in the output plane in order to determine r01
for x_sample in range(OutputShape[0]):
    x_fis_out = x_cord_out[x_sample]
    for y_sample in range(OutputShape[1]):
        y_fis_out = y_cord_out[y_sample]
        result = np.complex64(0)
        x_gpu = x_fis_out.astype(np.float32)
        y_gpu = y_fis_out.astype(np.float32)
        matrix(U_field,U_temp,z,k,x_gpu,y_gpu,X_gpu,Y_gpu,mr01,Obliquity, kernel, dif,N, M,block=block_dim, grid=grid_dim)
        sum2DArray(U_temp,M,N,result,block=block_dim, grid=grid_dim)
        out_calc(U_out,result, N, M,np.int32(y_sample),np.int32(x_sample),block=block_dim, grid=grid_dim)

U0 = U_out.get()
U0 = -U0/(2*np.pi)

figure = ploty2(U1,U0)





