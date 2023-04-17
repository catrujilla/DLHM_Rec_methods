'''
Code developed by Tomás Vélez Acosta
'''

from math import pi
import numpy as np
import time
from PIL import Image
from matplotlib import pyplot as plt
from scipy.interpolate import RegularGridInterpolator

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
    out1 = amplitude(inp,log)
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

def resample(array,grad,pix_pit,Gamma=None):
    shape = np.shape(array)
    M = int(shape[0]*(1+grad*pix_pit[0]/np.pi))
    N = int(shape[1]*(1+grad*pix_pit[1]/np.pi))
    fts = np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(array)))
    fts = np.pad(fts,((M,M),(N,N)),'constant',constant_values=0)
    out = np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(fts)))
    shape_out = np.shape(out)
    if type(Gamma) != 'NoneType':
        shape_Gamma = np.shape(Gamma)
        dM = int((shape_out[0]-shape_Gamma[0])/2)
        dN = int((shape_out[1]-shape_Gamma[1])/2)
        Gamma = np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(Gamma)))
        if dM%2 == 1:
            Gamma = np.pad(Gamma,((dM,dM),(dN,dN)),'constant',constant_values=0)
        else:
            Gamma = np.pad(Gamma,((dM,),(dN,)),'constant',constant_values=0)
        Gamma = np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(Gamma)))
    return out,Gamma

def SAASM(field, z, wavelength, pixel_pitch_in,pixel_pitch_out):
    '''
    # Function to diffract a complex field using the angular spectrum approach with a Semi-Analytical spherical wavefront.
    # For further reference review: 

    
    # Inputs:
    # field - complex field
    # z - propagation distance
    # wavelength - wavelength
    # dx/dy - sampling pitches
    '''


    # Starting cooridnates computation
    k_wl = 2 * pi / wavelength
    M, N = field.shape
    #Linear Coordinates
    x = np.arange(0, N, 1)  # array x
    fx = np.fft.fftshift(np.fft.fftfreq(N,pixel_pitch_in[0]))
    y = np.arange(0, M, 1)  # array y
    fy = np.fft.fftshift(np.fft.fftfreq(M,pixel_pitch_in[1]))
    #Grids
    X_in, Y_in = np.meshgrid((x - (N / 2))*pixel_pitch_in[0], (y - (M / 2))*pixel_pitch_in[1], indexing='xy')
    FX, FY = np.meshgrid(fx, fy, indexing='xy')
    KX = FX * 2 * pi
    KY = FY * 2 * pi
    MR_in = (X_in**2 + Y_in**2)
    MK = np.sqrt(KX**2 + KY**2)
    kmax = np.amax(MK)

    # Fitting parameters for the parabolic fase
    k_interm = (k_wl/kmax)
    c = (2/3 * k_interm) + 2/3 * np.sqrt(k_interm**2 - 0.5)- 1/3 * np.sqrt(k_interm**2 -1)
    d = np.sqrt(k_interm**2 - 1) - k_interm
    pp0 = pixel_pitch_in[0]
    


    #Calculating the beta coordinates as output for the first fourier transform
    X_out, Y_out = np.meshgrid((x - (N / 2))*pixel_pitch_out[0], (y- (M / 2))*pixel_pitch_out[1], indexing='xy')
    bX = -kmax * X_out / (2*d*z)
    bY = -kmax * Y_out / (2*d*z)
    Mbeta = np.sqrt(np.power(bX,2)+np.power(bY,2))
    

    ''' IN THIS STEP THE FIRST FOURIER TRANSFORM OF THE FIELD IS CALCULATED DOING A RESAMPLING USING THE
    FAST FOURIER TRASNSFORM AND A PADDING. THIS TRANSFORM HAS AS OUTPUT COORDINATE THE SCALED COORDINATE
    BETA, THAT IS NOT RELEVANT FOR THIS STEP BUT THE NEXT ONE'''
    # Initial interpolation for j=1
    max_grad_alpha = -kmax/(2*d*z) * np.amax(MR_in)
    pp1 = np.pi * pp0 /(pp0*max_grad_alpha+2*np.pi)
    alpha = np.exp(1j* c * kmax * z)*kmax/(2j * d * z) * np.exp((1j * kmax * MR_in)/(4*d*z))

    #Interpolation of the input field Scipy
    xin = (x - (N / 2))*pp0
    yin = (y - (M / 2))*pp0
    N2 = int(N*(2+max_grad_alpha*pp0/np.pi))
    M2 = int(M*(2+max_grad_alpha*pp0/np.pi))
    x1 = np.arange(0, N2-1, 1)
    y1 = np.arange(0, M2-1, 1)
    

    X1,Y1 = np.meshgrid((x1 - (N2 / 2))*pp1, (y1 - (M2 / 2))*pp1,indexing='ij')
    inter = RegularGridInterpolator((xin,yin),field,bounds_error=False, fill_value=None)
    E_interpolated = inter((X1,Y1))
    FE = np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(E_interpolated)))
    MR1 = (X1**2 + Y1**2)
    # alpha = np.exp(1j* c * kmax * z)*kmax/(2j * d * z) * np.exp((1j * kmax * MR1)/(4*d*z))

    # Interpolation using FFT
    E_interpolated,alpha = resample(field,max_grad_alpha,[pp0,pp0],Gamma=alpha)
    E_interpolated = E_interpolated - np.amin(E_interpolated)
    E_interpolated = E_interpolated/np.amax(E_interpolated)


    # Computation of the j=1 step
    FE1 = np.fft.ifftshift(np.fft.fft2(np.fft.fftshift((np.divide(E_interpolated,alpha)))))
    #Slicing of the input field in the inner region where the field is valid
    half_size1 = [int(np.shape(FE1)[0]/2),int(np.shape(FE1)[1]/2)]
    FE1 = FE1[half_size1[0]-int(M/2):half_size1[0]+int(M/2),half_size1[1]-int(N/2):half_size1[1]+int(N/2)]
    
    

    '''IN THIS STEP THE SECOND FOURIER TRANSFORM IS CALCULATED. HERE THE COORDINATES BETA ARE RELEVANT
    SINCE THE ELEMENT-WISE PRODUCT OF THE FE1 WITH THE PROPAGATION KERNEL REQUIRES THE KERNEL'S 
    ARGUMENT TO BE THE MAGNITUDE OF BETA INSTEAD OF THE MAGNITUD OF RHO'''
    #Padding variables for j=2
    max_grad_kernel = np.amax(Mbeta)
    pp2 = np.pi * pp1 /(pp1*max_grad_kernel+2*np.pi)
    pad2 = [int(np.shape(FE1)[0]*(1+max_grad_kernel*pp1/np.pi)/2),int(np.shape(FE1)[1]*(1+max_grad_kernel*pp1/np.pi)/2)]
    E2 = np.pad(FE1,((pad2[0],pad2[0]),(pad2[1],pad2[1])),'constant',constant_values=np.mean(FE1))
    # E2 = FE1
    


    # Calculation of the oversampled kernel
    M2,N2 = np.shape(E2)
    x2 = np.arange(0,N2,1)
    y2 = np.arange(0,M2,1)
    # If required, check the pixel size
    X_out, Y_out = np.meshgrid((x2 - (N2 / 2))*pixel_pitch_out[0], (y2- (M2 / 2))*pixel_pitch_out[1], indexing='xy')
    bX = -kmax * X_out / (2*d*z)
    bY = -kmax * Y_out / (2*d*z)
    Mbeta = np.sqrt(np.power(bX,2)+np.power(bY,2))
    kernel = np.exp(-1j * d * z * np.power(Mbeta,2)/(kmax))



    FE2 = ftx(E2*kernel)
    # half_size2 = [int(np.shape(FE2)[0]/2),int(np.shape(FE2)[1]/2)]
    # FE2 = FE2[half_size2[0]-int(M2/2):half_size2[0]+int(M/2),half_size2[1]-int(N2/2):half_size2[1]+int(N/2)]

    

    '''IN THIS STEP THE THIRD FOURIER TRANSFORM IS CALCULATED. HERE THE SUPERIOR ORDER TERMS (H) ARE CALCULATED
    TO FIND NUMERICALLY THE MAXIMUM GRADIENT OF ITS ARGUMENT, THEN, A PADDING OF FE2 IS DONE AND FINALLY H
    IS RESAMPLED IN TERMS OF FE2'''
    # Calculation of the superior order phases
    M3,N3 = np.shape(FE2)
    fx_out = np.fft.fftshift(np.fft.fftfreq(N3,pixel_pitch_out[0]))
    fy_out = np.fft.fftshift(np.fft.fftfreq(M3,pixel_pitch_out[1]))
    FX_out, FY_out = np.meshgrid(fx_out, fy_out, indexing='xy')
    KX_out = FX_out * 2 * pi
    KY_out = FY_out * 2 * pi
    MK_out = np.sqrt(KX_out**2 + KY_out**2)
    taylor_no_sup = (c*kmax + d *(MK_out**2)/kmax)
    spherical_ideal = np.sqrt(k_wl**2 - MK_out**2)
    h = spherical_ideal - taylor_no_sup
    grad_h = np.sqrt(np.gradient(h)[0]**2 + np.gradient(h)[1]**2)
    
    
    # Trial
    E_npad = iftx(FE2 * np.exp(1j * z * h))
    half_size3 = [int(np.shape(E_npad)[0]/2),int(np.shape(E_npad)[1]/2)]
    E_npad = E_npad[half_size3[0]-int(M/2):half_size3[0]+int(M/2),half_size3[1]-int(N/2):half_size3[1]+int(N/2)]
    
    # Padding variables for j=3
    max_grad_h = np.amax(grad_h)
    pad3 = [int(np.shape(FE2)[0]*(1+max_grad_h*pp2/np.pi)/2),int(np.shape(FE2)[1]*(1+max_grad_h*pp2/np.pi)/2)]
    E3 = np.pad(FE2,((pad3[0],pad3[0]),(pad3[1],pad3[1])),'constant',constant_values=0)

    #Calculation of the new h
    M3,N3 = np.shape(E3)
    fx_out = np.fft.fftshift(np.fft.fftfreq(N3,pp2))
    fy_out = np.fft.fftshift(np.fft.fftfreq(M3,pp2))
    FX_out, FY_out = np.meshgrid(fx_out, fy_out, indexing='xy')
    KX_out = FX_out * 2 * pi
    KY_out = FY_out * 2 * pi
    MK_out = np.sqrt(KX_out**2 + KY_out**2)
    taylor_no_sup = (c*kmax + d *(MK_out**2)/kmax)
    spherical_ideal = np.sqrt(k_wl**2 - MK_out**2)
    h = spherical_ideal - taylor_no_sup
    
    E_out = iftx(E3 * np.exp(1j * z * h))
    half_size3 = [int(np.shape(E3)[0]/2),int(np.shape(E3)[1]/2)]
    E_out = E_out[half_size3[0]-int(M/2):half_size3[0]+int(M/2),half_size3[1]-int(N/2):half_size3[1]+int(N/2)]
    
    
    return E_npad

def plotea(U1,U0):
    fig,axs = plt.subplots(1, 2)
    # gs = fig.add_gridspec(1,3, hspace=0, wspace=0)
    # axs = gs.subplots(sharex=False, sharey=True)
    axs[0].imshow(intensity(U1,'False'), cmap='gray')
    axs[0].set_title('Input')
    # axs[2].imshow(amplitude(U0,'False'), cmap='gray',extent=limits_out)
    # axs[2].set_title('Amplitude Pattern \n Screen-Aperture distance = '+str(output_z)+' m \n Aperture radius = ' +str(radius*1000) + ' mm '+'(Coordinates in [m])')
    axs[1].imshow(intensity(U0,False), cmap='gray')
    axs[1].set_title('SAASM \n Screen-Aperture distance = '+str(output_z)+' m \n Aperture radius = ' +str(radius*1000) + ' mm '+'(Coordinates in [m])')
    plt.subplots_adjust(wspace=0.171)
    plt.show()

    return fig




signal_size = 512 # Size of visualization
Magn = 1e0
dx = dy = 3.3e-6 #Pixel Size
dx_out = dy_out = dx*Magn
M = N = signal_size # Control of the size of the matrices

x_center = signal_size/2# Optical...
y_center = signal_size/2# ...axis of the system
radius = 5e-5 # Radius of the aperture in meters
Pradius = int(radius/dx) #Radius of the aperture in pixels

x_inp_lim = dx*int(N/2)
y_inp_lim = dy*int(M/2)


# Light source+
wavelength = 6.32e-7 # Wavelength of the illumination Source
k = 2*pi/wavelength # Wave number of the ilumination source


output_z = 8e-3   # Z Component of the observation screen coordinates
# output_z = 2.5e-2   # Z Component of the observation screen coordinates
Input_Z = 0 # Z Component of the aperture coordinates


# Aperture defines the geometry of the apperture, for circular apperture use circ2D, for rectangular apperture use rect2D

 


# im = Image.open(r"D:\OneDrive - Universidad EAFIT\Semestre VII\Advanced Project I\Holograms\0106\USINTFINraw.png").convert('L')
# im = circ2D(signal_size,Pradius,center=None)
im = Image.open(r"D:\OneDrive - Universidad EAFIT\Semestre IX\Advanced Project 2\USAFFULL.jpg").convert('L')
# im = Image.open(r"D:\OneDrive - Universidad EAFIT\Semestre IX\Advanced Project 2\USAF-1951.svg.png").convert('L')

im = im.resize((signal_size,signal_size))
im = np.asarray(im)/255
# im = np.pad(im,1024)
U1 = im.copy()


U0_temp_pure = SAASM(U1, output_z, wavelength, [dx,dy],[dx_out,dy_out])
U0_temp = intensity(U0_temp_pure,False)
U0_temp = Image.fromarray(U0_temp)
U0_temp = np.asarray(U0_temp.resize((signal_size,signal_size)))
U0 = SAASM(U0_temp, -output_z, wavelength, [dx_out,dy_out],[dx,dy])
VW_in = [-x_inp_lim,x_inp_lim,-y_inp_lim,y_inp_lim]

plotea(U0,U0_temp_pure)

























