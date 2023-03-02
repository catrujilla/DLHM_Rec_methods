'''
Code developed by Tomás Vélez Acosta
'''

from math import pi
import numpy as np
import math as mt
import pdb
import time
from matplotlib import pyplot as plt


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

def imageShow (inp, title,limits):
    '''
    # Function to display an image
    # Inputs:
    # inp - The input complex field
    # title - The title of the displayed image        
    '''
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

def RS1(Field_Input,z,wavelength,pixel_pitch_in,pixel_pitch_out,Output_shape):
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
    dx_out = pixel_pitch_out[0] #Output Pixel Size X
    dy_out = pixel_pitch_out[1] #Output Pixel Size Y
    
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
    


    x_out_lim = dx_out*int(Output_shape[0]/2)
    y_out_lim = dy_out*int(Output_shape[1]/2)

    x_cord_out = np.linspace(-x_out_lim , x_out_lim , num = Output_shape[0])
    y_cord_out = np.linspace(-y_out_lim , y_out_lim , num = Output_shape[1])

    [X_out,Y_out] = np.meshgrid(x_cord_out,y_cord_out)
    
    # The first pair of loops ranges over the points in the viewing screen in order to determine r01
    for x_sample in range(Output_shape[0]):
        x_fis_out = X_out[1,x_sample]
        for y_sample in range(Output_shape[1]):
            y_fis_out = Y_out[y_sample,1]
            # pdb.set_trace()
            # print(U1)
            
            U1with_phase = U1.copy()
            mr01 = np.sqrt(np.power(X_inp-x_fis_out,2)+np.power(Y_inp-y_fis_out,2)+(Input_Z-output_z)**2)
            Obliquity = (Input_Z-output_z) / mr01
            kernel = np.exp(1j * k * mr01)/mr01
            U0[y_sample,x_sample] = np.sum(U1with_phase * kernel * Obliquity * dx * dy)
            # The second pair of loops ranges over the points in the apperture to determine r21
    U0 = U0 /(1j*wavelength)
    Viewing_window = [-x_out_lim,x_out_lim,-y_out_lim,y_out_lim]
    return U0,Viewing_window

def plotea(Aperture,U0,limits_in,limits_out):
    fig, axs = plt.subplots(2, 2)




    axs[0,0].imshow(amplitude(Aperture,'False'), cmap='gray',extent=limits_in)
    axs[0,1].imshow(amplitude(U0,'False'), cmap='gray',extent=limits_out)
    axs[1,0].imshow(intensity(U0,'False'), cmap='gray',extent=limits_out)
    fig.show()

    return fig


#Simulation Control variables

signal_size_in = 128 # Size of visualization
signal_size_out = 128 # Size of visualization
dx = dy = 0.04/signal_size_in #Pixel Size
dx_out = dy_out = dx
M = N = signal_size_in # Control of the size of the matrices

x_center = signal_size_in/2# Optical...
y_center = signal_size_in/2# ...axis of the system
radius = 0.002 # Radius of the aperture in meters
Pradius = int(radius/dx) #Radius of the aperture in pixels

x_inp_lim = dx*int(N/2)
y_inp_lim = dy*int(M/2)


# Light source
wavelength = 0.6328/(1000000) # Wavelength of the illumination Source
k = 2*pi/wavelength # Wave number of the ilumination source




z = -1 # Z Component of the Source's coordinates 
SourceZ = np.array([0,0,z]) # Coordinates of the source
output_z = 10   # Z Component of the observation screen coordinates
Input_Z = 0 # Z Component of the aperture coordinates


# Aperture defines the geometry of the apperture, for circular apperture use circ2D, for rectangular apperture use rect2D

Aperture = circ2D(signal_size_in,Pradius,center=None) 
# Aperture = rect2D(signal_size,10,10,center=None)


U1 = Aperture.copy()

U0,VW = RS1(U1,output_z,wavelength,[dx,dy],[dx_out,dy_out],[signal_size_out,signal_size_out])


# Finally, the code plots the amplitude of the diffraction pattern

# fig, axs = plt.subplots(2, 2)



# plotea(Aperture,U0)
imageShow(amplitude(Aperture,'False'),'Aperture (Coordinates in [m])',[-x_inp_lim,x_inp_lim,-y_inp_lim,y_inp_lim])
imageShow(intensity(U0,False),('Diffraction Pattern \n Screen-Aperture distance = '+str(output_z)+' m \n Aperture radius = ' +str(radius*1000) + ' mm '+'(Coordinates in [m])' ),VW)
imageShow(amplitude(U0,False),('Amplitude Diffraction Pattern \n Screen-Aperture distance = '+str(output_z)+' m \n Aperture radius = ' +str(radius*1000) + ' mm '+'(Coordinates in [m])' ),VW)

