'''
Code developed by Tomás Vélez Acosta
'''

from math import pi
import numpy as np
import time
from PIL import Image
from matplotlib import pyplot as plt

def ftx(input):
    fts = np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(input)))
    return fts

def phase(inp):
    out = np.angle(inp)+np.pi
    return(out)

def iftx(input):
    fts = np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(input)))
    return fts

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

def rect2D(size, width_x, width_y, center=None):
    '''
    Make a 2D rect function.
    Size is the length of the signal
    width is width of the rect function
    '''
    if center is None:
        x0 = y0 = size // 2
    else:
        x0 = center[0]
        y0 = center[1]

    data = np.zeros((size,size))
    data[int(x0-width_x/2):int(x0+width_x/2), int(y0-width_y/2):int(y0+width_y/2)] = 1

    return data

def circ2D(size, pradius, center=None):
    '''
    Makes a 2D circ function.
    Size is the length of the signal
    pradius is the pradius of the circ function
    '''
    if center is None:
        x0 = y0 = size // 2
    else:
        x0 = center[0]
        y0 = center[1]

    data = np.zeros((size,size),dtype='complex_')
    for j in range (size):
        for i in range (size):
            if np.power(j-x0, 2) + np.power(i-y0, 2) < np.power(pradius, 2):
                data[i,j] = 1
    return data

def imageShow (inp, title):
    '''
    # Function to display an image
    # Inputs:
    # inp - The input complex field
    # title - The title of the displayed image        
    '''
    plt.imshow(inp, cmap='gray'), plt.title(title)  # image in gray scale
    plt.show()  # show image

    return

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
    if z<0:
        field = np.flip(field)

    M, N = field.shape
    x = np.arange(0, N, 1)  # array x
    fx = np.fft.fftshift(np.fft.fftfreq(N,pixel_pitch_in[0]))

    y = np.arange(0, M, 1)  # array y
    fy = np.fft.fftshift(np.fft.fftfreq(M,pixel_pitch_in[1]))


    X_in, Y_in = np.meshgrid((x - (N / 2))*pixel_pitch_in[0], (y - (M / 2))*pixel_pitch_in[1], indexing='xy')
    X_out,Y_out = np.meshgrid((x - (N / 2))*pixel_pitch_out[0], (y - (M / 2))*pixel_pitch_out[1], indexing='xy')
    FX, FY = np.meshgrid(fx, fy, indexing='xy')
    KX = FX * 2 * pi
    KY = FY * 2 * pi


    
    MR_in = (X_in**2 + Y_in**2)
    MK = np.sqrt(KX**2 + KY**2)
    
    
    
    kmax = np.amax(MK)
    k_interm = (k_wl/kmax)
    c = (2/3 * k_interm) + 2/3 * np.sqrt(k_interm**2 - 0.5)- 1/3 * np.sqrt(k_interm**2 -1)
    d = np.sqrt(k_interm**2 - 1) - k_interm

    bX = -kmax * X_out / (2*d*z)
    bY = -kmax * Y_out / (2*d*z)
    Mbeta = np.sqrt(np.power(bX,2)+np.power(bY,2))
    
    taylor_no_sup = (c*kmax + d *(MK**2)/kmax )
    
    



    spherical_ideal = np.sqrt(k_wl**2 - MK**2)
    


    h = spherical_ideal - taylor_no_sup

    
    alpha = np.exp(1j* c * kmax * z)*kmax/(2j * d * z) * np.exp((1j * kmax * MR_in)/(4*d*z))

    kernel = np.exp(-1j * d * z * np.power(Mbeta,2)/(kmax))
    
    


    E_out = iftx(ftx(ftx(np.divide(field,alpha)) * kernel) * np.exp(1j * z * h))
    

	
    return E_out

def plotea(U1,U0,limits_in,limits_out):
    fig,axs = plt.subplots(1, 2)
    # gs = fig.add_gridspec(1,3, hspace=0, wspace=0)
    # axs = gs.subplots(sharex=False, sharey=True)
    axs[0].imshow(intensity(U1,'False'), cmap='gray')
    axs[0].set_title('Input')
    # axs[2].imshow(amplitude(U0,'False'), cmap='gray',extent=limits_out)
    # axs[2].set_title('Amplitude Pattern \n Screen-Aperture distance = '+str(output_z)+' m \n Aperture radius = ' +str(radius*1000) + ' mm '+'(Coordinates in [m])')
    axs[1].imshow(intensity(U0,False), cmap='gray',extent=limits_out)
    axs[1].set_title('SAASM \n Screen-Aperture distance = '+str(output_z)+' m \n Aperture radius = ' +str(radius*1000) + ' mm '+'(Coordinates in [m])')
    plt.subplots_adjust(wspace=0.171)
    plt.show()

    return fig



signal_size = 1024 # Size of visualization
Magn = 1e0
dx = dy = 3.3e-6 #Pixel Size
dx_out = dy_out = dx/Magn
M = N = signal_size # Control of the size of the matrices

x_center = signal_size/2# Optical...
y_center = signal_size/2# ...axis of the system
radius = 5e-5 # Radius of the aperture in meters
Pradius = int(radius/dx) #Radius of the aperture in pixels

x_inp_lim = dx*int(N/2)
y_inp_lim = dy*int(M/2)


# Light source+
wavelength = 4.52e-7 # Wavelength of the illumination Source
k = 2*pi/wavelength # Wave number of the ilumination source



output_z = 2.5e-2   # Z Component of the observation screen coordinates
# output_z = -2.5e-2   # Z Component of the observation screen coordinates
Input_Z = 0 # Z Component of the aperture coordinates


# Aperture defines the geometry of the apperture, for circular apperture use circ2D, for rectangular apperture use rect2D

 


# im = Image.open(r"D:\OneDrive - Universidad EAFIT\Semestre VII\Advanced Project I\Holograms\0106\USINTFINraw.png").convert('L')
# im = circ2D(signal_size,Pradius,center=None)
im = Image.open(r"D:\OneDrive - Universidad EAFIT\Semestre IX\Advanced Project 2\USAFFULL.jpg").convert('L')
# im = Image.open(r"D:\OneDrive - Universidad EAFIT\Semestre IX\Advanced Project 2\USAF-1951.svg.png").convert('L')

im = im.resize((signal_size,signal_size))
im = np.asarray(im)/255

U1 = im.copy()


U0_temp = SAASM(U1, output_z, wavelength, [dx,dy],[dx_out,dy_out])
U0 = SAASM(U0_temp, -output_z, wavelength, [dx_out,dy_out],[dx,dy])
VW_in = [-x_inp_lim,x_inp_lim,-y_inp_lim,y_inp_lim]

plotea(U0_temp,U0,VW_in,VW_in)

























