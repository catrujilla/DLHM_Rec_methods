'''
Code developed by Tomás Vélez Acosta
'''

from math import pi
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import cv2 as cv
import time

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

def imageShow (inp, title,limits = None):
    '''
    # Function to display an image
    # Inputs:
    # inp - The input complex field
    # title - The title of the displayed image        
    '''
    M,N = np.shape(inp)
    if all (limits == None):
        limits = [0,N,0,M]
    plt.imshow(inp, cmap='gray',extent=limits), plt.title(title)  # image in gray scale
    plt.show()  # show image

    return

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

def phase(inp):
    out = np.angle(inp)+np.pi
    return(out)

def plotea(Aperture,U0,limits_in,limits_out):
    fig,axs = plt.subplots(1, 2)
    # gs = fig.add_gridspec(1,3, hspace=0, wspace=0)
    # axs = gs.subplots(sharex=False, sharey=True)
    axs[0].imshow(intensity(Aperture,'False'), cmap='gray',extent=limits_in)
    axs[0].set_title('Angular Spectrum')
    # axs[2].imshow(amplitude(U0,'False'), cmap='gray',extent=limits_out)
    # axs[2].set_title('Amplitude Pattern \n Screen-Aperture distance = '+str(output_z)+' m \n Aperture radius = ' +str(radius*1000) + ' mm '+'(Coordinates in [m])')
    axs[1].imshow(intensity(U0,'False'), cmap='gray',extent=limits_out)
    axs[1].set_title('RS1 \n Screen-Aperture distance = '+str(output_z)+' m \n Aperture radius = ' +str(radius*1000) + ' mm '+'(Coordinates in [m])')
    plt.subplots_adjust(wspace=0.171)
    plt.show()

    return fig

def RS1(Field_Input,z,wavelength,pixel_pitch_in,pixel_pitch_out):
    '''
    # SOME COMMENTS ARE MISSING HERE

    
    # Inputs:
    # field - complex field
    # z - propagation distance
    # wavelength - wavelength
    # dx/dy - sampling pitches
    '''


    dx = pixel_pitch_in[0] #Input Pixel Size X
    dy = pixel_pitch_in[1] #Input Pixel Size Y
    ds = dx*dy
    dx_out = pixel_pitch_out[0] #Output Pixel Size X
    dy_out = pixel_pitch_out[1] #Output Pixel Size Y
    
    M,N = np.shape(Field_Input)
    k = (2*pi)/wavelength # Wave number of the ilumination source
    
    output_z = z

    U0 = np.zeros((M,N),dtype='complex_')
    U1 = Field_Input  #This will be the hologram plane 


    x_inp_lim = dx*int(N/2)
    y_inp_lim = dy*int(M/2)

    x_cord = np.linspace(-x_inp_lim , x_inp_lim , num = N)
    y_cord = np.linspace(-y_inp_lim , y_inp_lim , num = M)

    [X_inp,Y_inp] = np.meshgrid(x_cord,y_cord)


    x_out_lim = dx_out*int(N/2)
    y_out_lim = dy_out*int(M/2)

    x_cord_out = np.linspace(-x_out_lim , x_out_lim , num = N)
    y_cord_out = np.linspace(-y_out_lim , y_out_lim , num = M)

    
    # The first pair of loops ranges over the points in the viewing screen in order to determine r01
    for x_sample in range(N):
        x_fis_out = x_cord_out[x_sample]
        for y_sample in range(M):
            y_fis_out = y_cord_out[y_sample]
            mr01 = np.sqrt(np.power(X_inp-x_fis_out,2)+np.power(Y_inp-y_fis_out,2)+(output_z)**2)
            Obliquity = (output_z)/ mr01
            kernel = np.exp(1j * k * mr01)/mr01
            dif = 1j*k+(1/mr01)
            U0[y_sample,x_sample] = np.sum(U1 * dif * kernel * Obliquity * ds)
    # U0 = U0 /(1j*wavelength)
    # U0 = U0 / (-2*pi)
    Viewing_window = [-x_out_lim,x_out_lim,-y_out_lim,y_out_lim]
    return U0,Viewing_window

def RS1_size_variable(Field_Input,z,wavelength,pixel_pitch_in,Output_shape):
    '''
    # SOME COMMENTS ARE MISSING HERE

    
    # Inputs:
    # field - complex field
    # z - propagation distance
    # wavelength - wavelength
    # dx/dy - sampling pitches
    '''


    dx = pixel_pitch_in[0] #Input Pixel Size X
    dy = pixel_pitch_in[1] #Input Pixel Size Y
    
    
    M,N = np.shape(Field_Input)
    k = 2*pi/wavelength # Wave number of the ilumination source
    
    output_z = z
    Input_Z = 0 # Z Component of the aperture coordinates

    U0 = np.zeros(Output_shape,dtype='complex_')
    U1 = Field_Input  #This will be the hologram plane 
    x_inp_lim = dx*int(N/2)
    y_inp_lim = dy*int(M/2)

    x_cord = np.linspace(-x_inp_lim , x_inp_lim , num = N)
    y_cord = np.linspace(-y_inp_lim , y_inp_lim , num = M)

    [X_inp,Y_inp] = np.meshgrid(x_cord,y_cord)
    


    x_out_lim = dx*int(Output_shape[0]/2)
    y_out_lim = dy*int(Output_shape[1]/2)

    x_cord_out = np.linspace(-x_out_lim , x_out_lim , num = Output_shape[0])
    y_cord_out = np.linspace(-y_out_lim , y_out_lim , num = Output_shape[1])

    [X_out,Y_out] = np.meshgrid(x_cord_out,y_cord_out)
    
    # The first pair of loops ranges over the points in the viewing screen in order to determine r01
    for x_sample in range(Output_shape[0]):
        x_fis_out = X_out[1,x_sample]
        for y_sample in range(Output_shape[1]):
            y_fis_out = Y_out[y_sample,1]

            U1with_phase = U1.copy()
            mr01 = np.sqrt(np.power(X_inp-x_fis_out,2)+np.power(Y_inp-y_fis_out,2)+(Input_Z-output_z)**2)
            Obliquity = (Input_Z-output_z) / mr01
            kernel = np.exp(1j * k * mr01)/mr01
            U0[y_sample,x_sample] = np.sum(U1with_phase * kernel * Obliquity * dx * dy)
    U0 = U0 /(1j*wavelength)
    Viewing_window = [-x_out_lim,x_out_lim,-y_out_lim,y_out_lim]
    return U0,Viewing_window

def angularSpectrum(field, z, wavelength, dx, dy):
    '''
    # Function to diffract a complex field using the angular spectrum approach
    # Inputs:
    # field - complex field
    # z - propagation distance
    # wavelength - wavelength
    # dx/dy - sampling pitches
    '''
    M, N = np.shape(field)
    x = np.arange(0, N, 1)  # array x
    y = np.arange(0, M, 1)  # array y
    X, Y = np.meshgrid(x - (N / 2), y - (M / 2), indexing='xy')

    dfx = 1 / (dx * M)
    dfy = 1 / (dy * N)
    
    field_spec = np.fft.fftshift(field)
    field_spec = np.fft.fft2(field_spec)
    field_spec = np.fft.fftshift(field_spec)
        
    phase = np.exp2(1j * z * pi * np.sqrt(np.power(1/wavelength, 2) - (np.power(X * dfx, 2) + np.power(Y * dfy, 2))))
	
    tmp = field_spec*phase
    
    out = np.fft.ifftshift(tmp)
    out = np.fft.ifft2(out)
    out = np.fft.ifftshift(out)
	
    return out

# Light source
wavelength = 4.05e-07 # Wavelength of the illumination Source
k = 2*pi/wavelength # Wave number of the ilumination source

#Simulation Control variables
L = 5e-3
output_z = 2e-3   # Z Component of the observation screen coordinates in m
signal_size = 256  # Size of visualization 
Magn = np.abs(L/output_z)
dx = dy = 3.3e-6 #Pixel Size
dx_out = dy_out =  dx/Magn
M = N = signal_size # Control of the size of the matrices
zcrit = np.sqrt(4*dx**2 - wavelength**2)*(N*dx + N*dx_out)/wavelength

print('La distancia crítica de propagación es: ', zcrit, 'm')

x_center = signal_size/2# Optical...
y_center = signal_size/2# ...axis of the system
radius = 2e-5 # Radius of the aperture in meters
Pradius = int(radius/dx) #Radius of the aperture in pixels

x_inp_lim = dx*int(N/2)
y_inp_lim = dy*int(M/2)










# Aperture defines the geometry of the apperture, for circular apperture use circ2D, for rectangular apperture use rect2D

Aperture = circ2D(signal_size,Pradius,center=None) 
# Aperture = rect2D(signal_size,10,10,center=None)



im = Image.open(r"D:\OneDrive - Universidad EAFIT\Semestre IX\Advanced Project 2\ep.png").convert('L')
# im = Image.open(r"D:\OneDrive - Universidad EAFIT\Semestre IX\Advanced Project 2\USAFFULL.jpg").convert('L')
# im = Image.open(r"D:\OneDrive - Universidad EAFIT\Semestre IX\Advanced Project 2\USAF-1951.svg.png").convert('L')
# im = im.resize((signal_size,signal_size))
im = np.asarray(im)/255





x_cord = np.linspace(-x_inp_lim , x_inp_lim , num = N)
y_cord = np.linspace(-y_inp_lim , y_inp_lim , num = M)
[X_inp,Y_inp] = np.meshgrid(x_cord,y_cord)
Rinp = np.sqrt(np.power(X_inp,2)+np.power(Y_inp,2) + (L-output_z)**2)

ill = np.ones_like(im,dtype='complex') * np.exp(1j*k*Rinp)/Rinp


# U1 = Aperture.copy()
U1 = im.copy() *ill


AS = angularSpectrum(Aperture,output_z,wavelength,dx,dy)
start = time.time()
U0,VW = RS1(im,output_z,wavelength,[dx,dy],[dx,dy])
end = time.time()
delta = end-start
print('El tiempo de ejecucion es: ',delta)
# U0,VW = RS1_size_variable(Aperture,output_z,wavelength,[dx,dy],[dx_out,dy_out],[2*signal_size,2*signal_size])

VW_in = [-x_inp_lim,x_inp_lim,-y_inp_lim,y_inp_lim]
figure = plotea(U1,U0,VW_in,VW)



Aperture,output_z,wavelength,[dx,dy],[dx_out,dy_out]


