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

#Simulation Control variables

signal_size = 32 # Size of visualization
dx = dy = 0.04/signal_size #Pixel Size
dx_out = dy_out = 0.04/signal_size
M = N = signal_size # Control of the size of the matrices

x_center = signal_size/2# Optical...
y_center = signal_size/2# ...axis of the system
radius = 0.01 # Radius of the aperture in meters
Pradius = int(radius/dx) #Radius of the aperture in pixels


# Light source
wavelength = 0.6328/(1000000) # Wavelength of the illumination Source
k = 2*pi/wavelength # Wave number of the ilumination source




z = -1 # Z Component of the Source's coordinates 
SourceZ = np.array([0,0,z]) # Coordinates of the source
output_z = 0.01 # Z Component of the observation screen coordinates
Input_Z = 0 # Z Component of the aperture coordinates


# Aperture defines the geometry of the apperture, for circular apperture use circ2D, for rectangular apperture use rect2D

Aperture = circ2D(signal_size,Pradius,center=None) 
# Aperture = rect2D(signal_size,10,10,center=None)


U0 = np.zeros((signal_size,signal_size),dtype='complex_') #This will be the viewing screen, the output plane

U1 = Aperture.copy()  #This will be the hologram plane 
x_inp_lim = dx*int(N/2)
y_inp_lim = dy*int(M/2)

x_cord = np.linspace(-x_inp_lim , x_inp_lim , num = N)
y_cord = np.linspace(-y_inp_lim , y_inp_lim , num = M)

[X_inp,Y_inp] = np.meshgrid(x_cord,y_cord)
Z_inp = Input_Z*np.ones_like(X_inp)


x_out_lim = dx_out*int(N/2)
y_out_lim = dy_out*int(M/2)

x_cord_out = np.linspace(-dx_out*int(N/2) , dx_out*int(N/2) , num = N)
y_cord_out = np.linspace(-dy_out*int(M/2) , dy_out*int(M/2) , num = M)

[X_out,Y_out] = np.meshgrid(x_cord_out,y_cord_out)
Z_out = output_z*np.ones_like(X_out)


r01_X = X_inp-np.transpose(X_out)
r01_Y = Y_inp-np.transpose(Y_out)
r01_Z = Z_inp-np.transpose(Z_out)

# imageShow(r01_X,'X component r01')
# imageShow(r01_Y,'Y component r01')
# imageShow(r01_Z,'Z component r01')

start = time.time()
# The first pair of loops ranges over the points in the viewing screen in order to determine r01
for x_sample in range(N):
    for y_sample in range(M):

        # pdb.set_trace()
        # print(U1)

        U1with_phase = U1.copy()
        # The second pair of loops ranges over the points in the apperture to determine r21
        for x_holo in range(N):
            for y_holo in range(M):                


                r01 = [r01_X[x_sample,x_holo],r01_Y[y_sample,y_holo],Input_Z-output_z]
                mr01 = np.sqrt(r01[0]**2 + r01[1]**2 + r01[2]**2)
                
                '''By knowing the geometrical parameters of the system, the following two lines compute the weight of
                each point source in the input plane to form the output.
                '''
                # pdb.set_trace()
                U1with_phase[y_holo,x_holo] = U1with_phase[y_holo,x_holo] * (r01[2]/mr01)
                U1with_phase[y_holo,x_holo] = U1with_phase[y_holo,x_holo] * np.exp(mr01*1j*k)/mr01
                U0[y_sample,x_sample] = U0[y_sample,x_sample] + (dx*dy) * U1with_phase[y_holo,x_holo]



U0 = U0 /(1j*wavelength)
# Finally, the code plots the amplitude of the diffraction pattern
end = time.time()

delta = end-start
print('El tiempo de ejecución es ',delta)
# ampU0 = amplitude(U0,False)
imageShow(amplitude(Aperture,'False'),'Aperture (Coordinates in [m])',[-x_inp_lim,x_inp_lim,-y_inp_lim,y_inp_lim])
imageShow(intensity(U0,False),('Diffraction Pattern \n Screen-Aperture distance = '+str(output_z)+' m \n Aperture radius = ' +str(radius*1000) + ' mm '+'(Coordinates in [m])' ),[-x_out_lim,x_out_lim,-y_out_lim,y_out_lim])


