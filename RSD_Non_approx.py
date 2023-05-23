'''
Code developed by Tomás Vélez Acosta
'''

from math import pi
import numpy as np
from matplotlib import pyplot as plt
import plotly.express as px
from plotly.subplots import make_subplots
from PIL import Image
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

def ploty(Input,Output,VWI,VWO):

    fig,axs = plt.subplots(2, 2)

    axs[0,0].imshow(intensity(Input,'False'), cmap='gray',extent=VWI)
    axs[0,0].set_title('Input Intensity')
    

    axs[0,1].imshow(phase(Input), cmap='gray',extent=VWI)
    axs[0,1].set_title('Input Phase')

    axs[1,0].imshow(intensity(Output,'False'), cmap='gray',extent=VWO)
    axs[1,0].set_title('Output Intensity')


    axs[1,1].imshow(phase(Output), cmap='gray',extent=VWO)
    axs[1,1].set_title('Output Phase')

    plt.subplots_adjust(wspace=0.2)
    plt.subplots_adjust(hspace=0.4)
    plt.show()

    return fig

def ishow(Field):
    fig,axs = plt.subplots(1, 2)
    # gs = fig.add_gridspec(1,3, hspace=0, wspace=0)
    # axs = gs.subplots(sharex=False, sharey=True)
    axs[0].imshow(intensity(Field,'False'), cmap='gray')
    axs[0].set_title('Intensity')
    # axs[2].imshow(amplitude(U0,'False'), cmap='gray',extent=limits_out)
    # axs[2].set_title('Amplitude Pattern \n Screen-Aperture distance = '+str(output_z)+' m \n Aperture radius = ' +str(radius*1000) + ' mm '+'(Coordinates in [m])')
    axs[1].imshow(phase(Field), cmap='gray')
    axs[1].set_title('Phase')
    plt.subplots_adjust(wspace=0.171)
    plt.show()

    return fig

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
    ds = dx*dy
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
            # start = time.time()
            y_fis_out = y_cord_out[y_sample]
            mr01 = np.sqrt(np.power(x_fis_out-X_inp,2)+np.power(y_fis_out-Y_inp,2)+(z)**2)
            Obliquity = (z)/ mr01
            kernel = np.exp(1j * k * mr01)/mr01
            dif = (1j*k)+(1/mr01)
            U0[y_sample,x_sample] = np.sum(U1 * dif * kernel * Obliquity * ds)
            # stop = time.time()
            # print('Tiempo de ejecución: ', 1000*(stop-start))
    U0 = -U0/(2*np.pi)
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

def ploty2(Input,Output):
    fig,axs = plt.subplots(1,2)

    axs[0].imshow(intensity(Input,'False'), cmap='gray')
    axs[0].set_title('Input Intensity')
    
    axs[1].imshow(intensity(Output,'False'), cmap='gray')
    axs[1].set_title('Output Intensity')

    plt.subplots_adjust(wspace=0.2)
    plt.subplots_adjust(hspace=0.4)
    plt.show()

def ploty3(Input,Output):
    fig,axs = plt.subplots(1,2)

    axs[0].imshow(phase(Input), cmap='gray')
    axs[0].set_title('Input phase')
    
    axs[1].imshow(phase(Output), cmap='gray')
    axs[1].set_title('Output phase')

    plt.subplots_adjust(wspace=0.2)
    plt.subplots_adjust(hspace=0.4)
    plt.show()

# Light source
wavelength = 6.33e-07 # Wavelength of the illumination Source
k = 2*pi/wavelength # Wave number of the ilumination source
prop_z = 3.5e-3
Magn = 1
dx = dy = 3.3e-6
signal_size = 128
OutputShape = 314



#Simulation Control variables
L = 100e-3
output_z = 15e-3   # Z Component of the observation screen coordinates in m





# dx = dy = 3.3e-6 #Pixel Size.

dx_out = dy_out =  Magn*(signal_size*dx)/OutputShape # MULTIPLY BY THE MAGNIFICATION
# dx_out = dy_out = dx*Magn # COMMENT FOR RECONSTRUCTION

M = N = signal_size # Control of the size of the matrices
zcrit = np.sqrt(4*dx**2 - wavelength**2)*(N*dx + N*dx_out)/(2*wavelength)


# Aperture defines the geometry of the apperture, for circular apperture use circ2D, for rectangular apperture use rect2D
radius = 7*dx # Radius of the aperture in meters
Pradius = int(radius/dx) #Radius of the aperture in pixels
Aperture = circ2D(signal_size,Pradius,center=None) 
# Aperture = rect2D(signal_size,10,10,center=None)



x_inp_lim = dx*int(N/2)
y_inp_lim = dy*int(M/2)


#---------------------------Choosing the image to diffract----------------------------
im = Aperture.copy()
# im = Image.open(r"USAF_EXP.png").convert('L')
# im = Image.open(r"D:\OneDrive - Universidad EAFIT\Semestre IX\Advanced Project 2\epiteliales_L=5_z=2.png").convert('L')
# im = Image.open(r"D:\OneDrive - Universidad EAFIT\Semestre IX\Advanced Project 2\ep.png").convert('L')
# im = Image.open(r"D:\OneDrive - Universidad EAFIT\Semestre IX\Advanced Project 2\USAFFULL.jpg").convert('L')
# im = Image.open(r"D:\OneDrive - Universidad EAFIT\Semestre IX\Advanced Project 2\USAF-1951.svg.png").convert('L')
# im = im.resize((signal_size,signal_size))
# im = np.asarray(im)/255
# im = 1-im # This line inverts 1 to 0 and visceversa to have illumination in the back


#--------------------------- Illumination calculation -----------------------------
M,N = np.shape(im)
x_cord = np.linspace(-x_inp_lim , x_inp_lim , num = N)
y_cord = np.linspace(-y_inp_lim , y_inp_lim , num = M)
[X_inp,Y_inp] = np.meshgrid(x_cord,y_cord)
Rinp = np.sqrt(np.power(X_inp,2)+np.power(Y_inp,2) + (100e-2)**2)
ill = np.exp(1j*im)
# ill = np.ones_like(im,dtype='complex') * np.exp(1j*k*Rinp)/Rinp
# ill = np.ones_like(im,dtype='complex')



U1 = im.copy() * ill






OutputShape = (OutputShape,OutputShape)

start = time.time()
U0,VW = RS1_Free(U1,prop_z,wavelength,[dx,dy],[dx_out,dy_out],OutputShape)
# U0,VW = RS1_Free(U1,-30e-3,4.31e-7,[40e-3,40e-3],[20e-3,20e-3],OutputShape)
end = time.time()
delta = end-start

print('El tiempo de ejecucion es: ',delta)

VW_in = [-x_inp_lim,x_inp_lim,-y_inp_lim,y_inp_lim]
# figure = ploty(U1,U0,VW_in,VW)




figure = ploty3(U1,U0)

# Reconstruction,VW_sample = RS1(U0,output_z-L,wavelength,[dx_out,dy_out],[dx,dy])

# figure2 = ploty(U0,Reconstruction,VW,VW_sample)
