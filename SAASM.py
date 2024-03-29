'''
Code developed by Tomás Vélez Acosta
'''

from math import pi
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from scipy.interpolate import RegularGridInterpolator

def phase(inp):
    out = np.angle(inp)+np.pi
    return(out)

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

def plotea(U1,U0):
    fig,axs = plt.subplots(1, 2)
    # gs = fig.add_gridspec(1,3, hspace=0, wspace=0)
    # axs = gs.subplots(sharex=False, sharey=True)
    axs[0].imshow(intensity(U1,False), cmap='gray')
    axs[0].set_title('Input')
    # axs[2].imshow(amplitude(U0,'False'), cmap='gray',extent=limits_out)
    # axs[2].set_title('Amplitude Pattern \n Screen-Aperture distance = '+str(output_z)+' m \n Aperture radius = ' +str(radius*1000) + ' mm '+'(Coordinates in [m])')
    axs[1].imshow(intensity(U0,False), cmap='gray')
    axs[1].set_title('SAASM Output \n Input-Output distance = '+str(output_z)+' m')
    plt.subplots_adjust(wspace=0.171)
    plt.show()

    return fig

def ishow(Field,title='title'):
    fig,axs = plt.subplots(1, 2)
    # gs = fig.add_gridspec(1,3, hspace=0, wspace=0)
    # axs = gs.subplots(sharex=False, sharey=True)
    axs[0].imshow(intensity(Field,'False'), cmap='gray')
    axs[0].set_title('Intensity')
    # axs[2].imshow(amplitude(U0,'False'), cmap='gray',extent=limits_out)
    # axs[2].set_title('Amplitude Pattern \n Screen-Aperture distance = '+str(output_z)+' m \n Aperture radius = ' +str(radius*1000) + ' mm '+'(Coordinates in [m])')
    axs[1].imshow(phase(Field), cmap='gray')
    axs[1].set_title('Phase')
    if title =='title':
        plt.subplots_adjust(wspace=0.171)
        plt.show(block=False)
        return fig
    fig.suptitle(title)
    plt.subplots_adjust(wspace=0.171)
    mng = plt.get_current_fig_manager()
    mng.full_screen_toggle()
    plt.show(block=False)

    return fig

def CONV_SAASM(field, z, wavelength, pixel_pitch_in,pixel_pitch_out):
    '''
    Function to diffract a complex field using the angular spectrum approach with a Semi-Analytical spherical wavefront.
    This operator only works for convergent fields, for divergent fields see DIV_SAASM
    For further reference review: https://opg.optica.org/josaa/abstract.cfm?uri=josaa-31-3-591 and https://doi.org/10.1117/12.2642760

    
    ### Inputs:
    * field - complex field to be diffracted
    * z - propagation distance
    * wavelength - wavelength of the light used
    * pixel_pitch_in - Sampling pitches of the input field as a (2,) list
    * pixel_pitch_out - Sampling pitches of the output field as a (2,) list
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
    alpha = (np.exp(-1j* c * kmax * z)*kmax/(2j * d * z)) * np.exp((1j * kmax * MR_in)/(4*d*z))

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
    MR1 = (X1**2 + Y1**2)
    alpha = np.exp(-1j* c * kmax * z)*kmax/(2j * d * z) * np.exp((1j * kmax * MR1)/(4*d*z))
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
    # E2 = np.copy(FE1)

    # Calculation of the oversampled kernel
    M0,N0 = np.shape(E2)
    x2 = np.arange(0,N0,1)
    y2 = np.arange(0,M0,1)
    # If required, check the pixel size
    X_out, Y_out = np.meshgrid((x2 - (N0 / 2))*pixel_pitch_out[0], (y2- (M0 / 2))*pixel_pitch_out[1], indexing='xy')
    bX = -kmax * X_out / (2*d*z)    
    bY = -kmax * Y_out / (2*d*z)
    Mbeta = np.sqrt(np.power(bX,2)+np.power(bY,2))
    kernel = np.exp(-1j * d * z * np.power(Mbeta,2)/(kmax))


    # Computation of the j=2 step
    FE2 = np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(E2*kernel)))
    half_size2 = [int(np.shape(FE2)[0]/2),int(np.shape(FE2)[1]/2)]
    FE2 = FE2[half_size2[0]-int(M/2):half_size2[0]+int(M/2),half_size2[1]-int(N/2):half_size2[1]+int(N/2)]
    # FE2 = FE2[230:2140,230:2140]

    

    '''IN THIS STEP THE THIRD FOURIER TRANSFORM IS CALCULATED. HERE THE SUPERIOR ORDER TERMS (H) ARE CALCULATED
    TO FIND NUMERICALLY THE MAXIMUM GRADIENT OF ITS ARGUMENT, THEN, A PADDING OF FE2 IS DONE AND FINALLY H
    IS RESAMPLED IN TERMS OF FE2'''
    # Calculation of the superior order phases
    Mfin,Nfin = np.shape(FE2)
    fx_out = np.fft.fftshift(np.fft.fftfreq(Nfin,pixel_pitch_out[0]))
    fy_out = np.fft.fftshift(np.fft.fftfreq(Mfin,pixel_pitch_out[1]))
    FX_out, FY_out = np.meshgrid(fx_out, fy_out, indexing='xy')
    KX_out = FX_out * 2 * pi
    KY_out = FY_out * 2 * pi
    MK_out = np.sqrt(KX_out**2 + KY_out**2)
    taylor_no_sup = (c*kmax + d *(MK_out**2)/kmax)
    spherical_ideal = np.sqrt(k_wl**2 - MK_out**2)
    h = spherical_ideal - taylor_no_sup
    grad_h = np.sqrt(np.gradient(h)[0]**2 + np.gradient(h)[1]**2)
    
    
    
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
    MK_out = np.where(MK_out>k_wl,k_wl,MK_out) # FIXING THE OVERLOADED FREQUENCIES
    spherical_ideal = np.sqrt(k_wl**2 - MK_out**2)
    h = spherical_ideal - taylor_no_sup

    # Computation of the j=3 step
    E_out = np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(E3 * np.exp(1j * z * h))))
    half_size3 = [int(np.shape(E3)[0]/2),int(np.shape(E3)[1]/2)]
    E_out = E_out[half_size3[0]-int(Mfin/2):half_size3[0]+int(Mfin/2),half_size3[1]-int(Nfin/2):half_size3[1]+int(Nfin/2)]
    # E_out = E_out[half_size3[0]-int(5017/2):half_size3[0]+int(5017/2),half_size3[1]-int(5017/2):half_size3[1]+int(5017/2)]
    
    print(pp2)
    return E_out

def CONV_SAASM_V2(field, z, wavelength, pixel_pitch_in,pixel_pitch_out):
    '''
    Function to diffract a complex field using the angular spectrum approach with a Semi-Analytical spherical wavefront.
    This operator only works for convergent fields, for divergent fields see DIV_SAASM
    For further reference review: https://opg.optica.org/josaa/abstract.cfm?uri=josaa-31-3-591 and https://doi.org/10.1117/12.2642760

    
    ### Inputs:
    * field - complex field to be diffracted
    * z - propagation distance
    * wavelength - wavelength of the light used
    * pixel_pitch_in - Sampling pitches of the input field as a (2,) list
    * pixel_pitch_out - Sampling pitches of the output field as a (2,) list
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
    alpha = (np.exp(-1j* c * kmax * z)*kmax/(2j * d * z)) * np.exp((1j * kmax * MR_in)/(4*d*z))

    #Interpolation of the input field Scipy
    xin = (x - (N / 2))*pp0
    yin = (y - (M / 2))*pp0
    N2 = int(N*(2+max_grad_alpha*pp0/np.pi))
    M2 = int(M*(2+max_grad_alpha*pp0/np.pi))

    pp1 = M*pixel_pitch_in[0]/M2
    x1 = np.arange(0, N2-1, 1)
    y1 = np.arange(0, M2-1, 1)
    

    X1,Y1 = np.meshgrid((x1 - (N2 / 2))*pp1, (y1 - (M2 / 2))*pp1,indexing='ij')
    inter = RegularGridInterpolator((xin,yin),field,bounds_error=False, fill_value=None)
    E_interpolated = inter((X1,Y1))
    MR1 = (X1**2 + Y1**2)
    alpha = np.exp(-1j* c * kmax * z)*kmax/(2j * d * z) * np.exp((1j * kmax * MR1)/(4*d*z))
    E_interpolated = E_interpolated - np.amin(E_interpolated)
    E_interpolated = E_interpolated/np.amax(E_interpolated)
    EM1 = np.divide(E_interpolated,alpha)

    #Padding variables for j=2
    max_grad_kernel = np.amax(Mbeta)
    
    
    # Computation of the j=1 step
    FE1 = np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(EM1)))
    #Slicing of the input field in the inner region where the field is valid
    # half_size1 = [int(np.shape(FE1)[0]/2),int(np.shape(FE1)[1]/2)]
    # FE1 = FE1[half_size1[0]-int(M/2):half_size1[0]+int(M/2),half_size1[1]-int(N/2):half_size1[1]+int(N/2)]

    

    '''IN THIS STEP THE SECOND FOURIER TRANSFORM IS CALCULATED. HERE THE COORDINATES BETA ARE RELEVANT
    SINCE THE ELEMENT-WISE PRODUCT OF THE FE1 WITH THE PROPAGATION KERNEL REQUIRES THE KERNEL'S 
    ARGUMENT TO BE THE MAGNITUDE OF BETA INSTEAD OF THE MAGNITUD OF RHO'''
    # Calculation of the oversampled kernel
    M0,N0 = np.shape(FE1)
    x2 = np.arange(0,N0,1)
    y2 = np.arange(0,M0,1)
    # If required, check the pixel size
    X_out, Y_out = np.meshgrid((x2 - (N0 / 2))*pixel_pitch_out[0], (y2- (M0 / 2))*pixel_pitch_out[1], indexing='xy')
    Mrho = np.sqrt(np.power(X_out,2)+np.power(Y_out,2))
    bX = -kmax * X_out / (2*d*z)    
    bY = -kmax * Y_out / (2*d*z)
    print('C = ',str(-kmax/(2*d*z)))
    Mbeta = np.sqrt(np.power(bX,2)+np.power(bY,2))
    kernel = np.exp(-1j * d * z * np.power(Mbeta,2)/(kmax))
    # kernel = np.exp(-1j * kmax * np.power(Mrho,2)/(4 * d * z))
    EM2 = FE1*kernel

    # Computation of the j=2 step
    FE2 = np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(EM2)))
    half_size2 = [int(np.shape(FE2)[0]/2),int(np.shape(FE2)[1]/2)]
    # FE2 = FE2[half_size2[0]-int(M/2):half_size2[0]+int(M/2),half_size2[1]-int(N/2):half_size2[1]+int(N/2)]
    

    '''IN THIS STEP THE THIRD FOURIER TRANSFORM IS CALCULATED. HERE THE SUPERIOR ORDER TERMS (H) ARE CALCULATED
    TO FIND NUMERICALLY THE MAXIMUM GRADIENT OF ITS ARGUMENT, THEN, A PADDING OF FE2 IS DONE AND FINALLY H
    IS RESAMPLED IN TERMS OF FE2'''
    # Calculation of the superior order phases
    Mfin,Nfin = np.shape(FE2)
    fx_out = np.fft.fftshift(np.fft.fftfreq(Nfin,pixel_pitch_out[0]))
    fy_out = np.fft.fftshift(np.fft.fftfreq(Mfin,pixel_pitch_out[1]))
    FX_out, FY_out = np.meshgrid(fx_out, fy_out, indexing='xy')
    KX_out = FX_out * 2 * pi
    KY_out = FY_out * 2 * pi
    MK_out = np.sqrt(KX_out**2 + KY_out**2)
    taylor_no_sup = (c*kmax + d *(MK_out**2)/kmax)
    spherical_ideal = np.sqrt(k_wl**2 - MK_out**2)
    h = spherical_ideal - taylor_no_sup
    
    # Computation of the j=3 step
    E_out = np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(FE2 * np.exp(1j * z * h))))
    half_size3 = [int(np.shape(E_out)[0]/2),int(np.shape(E_out)[1]/2)]
    # E_out = E_out[half_size3[0]-int(M/2):half_size3[0]+int(M/2),half_size3[1]-int(N/2):half_size3[1]+int(N/2)]
    # E_out = E_out[half_size3[0]-int(5017/2):half_size3[0]+int(5017/2),half_size3[1]-int(5017/2):half_size3[1]+int(5017/2)]
    print('Output pixel pitch: ',pp1* 10**6,'um')
    return E_out

def CONV_SAASM_V3(field, z, wavelength, pixel_pitch_in,pixel_pitch_out):
    '''
    Function to diffract a complex field using the angular spectrum approach with a Semi-Analytical spherical wavefront.
    This operator only works for convergent fields, for divergent fields see DIV_SAASM
    For further reference review: https://opg.optica.org/josaa/abstract.cfm?uri=josaa-31-3-591 and https://doi.org/10.1117/12.2642760

    
    ### Inputs:
    * field - complex field to be diffracted
    * z - propagation distance
    * wavelength - wavelength of the light used
    * pixel_pitch_in - Sampling pitches of the input field as a (2,) list
    * pixel_pitch_out - Sampling pitches of the output field as a (2,) list
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
    kmax = np.abs(np.amax(MK))
  

    ''' IN THIS STEP THE FIRST FOURIER TRANSFORM OF THE FIELD IS CALCULATED DOING A RESAMPLING USING THE
    FAST FOURIER TRASNSFORM AND A PADDING. THIS TRANSFORM HAS AS OUTPUT COORDINATE THE SCALED COORDINATE
    BETA, THAT IS NOT RELEVANT FOR THIS STEP BUT THE NEXT ONE'''
    # Fitting parameters for the parabolic fase
    k_interm = (k_wl/kmax)
    c = (2/3 * k_interm) + 2/3 * np.sqrt(k_interm**2 - 0.5)- 1/3 * np.sqrt(k_interm**2 -1)
    d = np.sqrt(k_interm**2 - 1) - k_interm
    pp0 = pixel_pitch_in[0]

    # Initial interpolation for j=1
    max_grad_alpha = -kmax/(2*d*z) * np.amax(MR_in)
    alpha = (np.exp(-1j* c * kmax * z)*kmax/(2j * d * z)) * np.exp((1j * kmax * MR_in)/(4*d*z))

    #Interpolation of the input field Scipy
    
    N2 = int(N*(2+max_grad_alpha*pp0/np.pi))
    M2 = int(M*(2+max_grad_alpha*pp0/np.pi))
    # M2 = 4*M
    # N2 = 4*N

    pp1 = M*pixel_pitch_in[0]/M2
    x1 = np.arange(0, N2-1, 1)
    y1 = np.arange(0, M2-1, 1)
    X1,Y1 = np.meshgrid((x1 - (N2 / 2))*pp1, (y1 - (M2 / 2))*pp1,indexing='ij')
    fx1 = np.fft.fftshift(np.fft.fftfreq(N2,pp1))
    fy1 = np.fft.fftshift(np.fft.fftfreq(M2,pp1))
    FX1, FY1 = np.meshgrid(fx1, fy1, indexing='xy')
    # THIS LINEs ARE FOR TRIALS ONLY
    X1 = Y_in
    Y1 = X_in
    FX1 = FX
    FY1 = FY
    #_______________________________
    KX1 = FX1 * 2 * pi
    KY1 = FY1 * 2 * pi
    MK1 = np.sqrt(KX1**2 + KY1**2)
    kmax = np.abs(np.amax(MK1))

    
    xin = (x - (N / 2))*pp0
    yin = (y - (M / 2))*pp0
    inter = RegularGridInterpolator((xin,yin),field,bounds_error=False, fill_value=None)
    E_interpolated = inter((X1,Y1))

    MR1 = (X1**2 + Y1**2)
    k_interm = (k_wl/kmax)
    c = (2/3 * k_interm) + 2/3 * np.sqrt(k_interm**2 - 0.5)- 1/3 * np.sqrt(k_interm**2 -1)
    d = np.sqrt(k_interm**2 - 1) - k_interm

    alpha = np.exp(-1j* c * kmax * z)*kmax/(2j * d * z) * np.exp((1j * kmax * MR1)/(4*d*z))
    E_interpolated = E_interpolated - np.amin(E_interpolated)
    E_interpolated = E_interpolated/np.amax(E_interpolated)
    EM1 = np.divide(E_interpolated,alpha)

    #Padding variables for j=2
    # max_grad_kernel = np.amax(Mbeta)
    
    # Computation of the j=1 step
    FE1 = np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(EM1)))

    

    '''IN THIS STEP THE SECOND FOURIER TRANSFORM IS CALCULATED. HERE THE COORDINATES BETA ARE RELEVANT
    SINCE THE ELEMENT-WISE PRODUCT OF THE FE1 WITH THE PROPAGATION KERNEL REQUIRES THE KERNEL'S 
    ARGUMENT TO BE THE MAGNITUDE OF BETA INSTEAD OF THE MAGNITUD OF RHO'''
    # Calculation of the oversampled kernel
    M0,N0 = np.shape(FE1)
    x2 = np.arange(0,N0,1)
    y2 = np.arange(0,M0,1)
    # If required, check the pixel size
    # X_out, Y_out = np.meshgrid((x2 - (N0 / 2))*pp1, (y2- (M0 / 2))*pp1, indexing='xy')#<----------------------erase this
    X_out, Y_out = np.meshgrid((x2 - (N0 / 2))*pixel_pitch_out[0], (y2- (M0 / 2))*pixel_pitch_out[1], indexing='xy')
    Mrho = np.sqrt(np.power(X_out,2)+np.power(Y_out,2))
    bX = -kmax * X_out / (2*d*z)    
    bY = -kmax * Y_out / (2*d*z)
    Mbeta = np.sqrt(np.power(bX,2)+np.power(bY,2))
    kernel = np.exp(-1j * d * z * np.power(Mbeta,2)/(kmax))
    # kernel = np.exp(-1j * kmax * np.power(Mrho,2)/(4 * d * z))
    EM2 = FE1*kernel

    # Computation of the j=2 step
    FE2 = np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(EM2)))
    # half_size2 = [int(np.shape(FE2)[0]/2),int(np.shape(FE2)[1]/2)]
    # FE2 = FE2[half_size2[0]-int(M/2):half_size2[0]+int(M/2),half_size2[1]-int(N/2):half_size2[1]+int(N/2)]
    

    '''IN THIS STEP THE THIRD FOURIER TRANSFORM IS CALCULATED. HERE THE SUPERIOR ORDER TERMS (H) ARE CALCULATED
    TO FIND NUMERICALLY THE MAXIMUM GRADIENT OF ITS ARGUMENT, THEN, A PADDING OF FE2 IS DONE AND FINALLY H
    IS RESAMPLED IN TERMS OF FE2'''
    # Calculation of the superior order phases
    Mfin,Nfin = np.shape(FE2)
    #----------------ERASE THIS-----------------
    # fx_out = np.fft.fftshift(np.fft.fftfreq(Nfin,pp1))
    # fy_out = np.fft.fftshift(np.fft.fftfreq(Mfin,pp1))
    #-------------------------------------------
    
    fx_out = np.fft.fftshift(np.fft.fftfreq(Nfin,pixel_pitch_out[0]))
    fy_out = np.fft.fftshift(np.fft.fftfreq(Mfin,pixel_pitch_out[1]))
    FX_out, FY_out = np.meshgrid(fx_out, fy_out, indexing='xy')
    KX_out = FX_out * 2 * pi
    KY_out = FY_out * 2 * pi
    MK_out = np.sqrt(KX_out**2 + KY_out**2)
    taylor_no_sup = (c*kmax + d *(MK_out**2)/kmax)
    etay = np.exp(1j*z*taylor_no_sup)
    spherical_ideal = np.sqrt(k_wl**2 - MK_out**2)
    esph = np.exp(1j*z*spherical_ideal)
    h = spherical_ideal - taylor_no_sup
    
    # Computation of the j=3 step
    phase_h = np.exp(1j * z * h)
    EM3 = FE2 * phase_h
    E_out = np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(EM3)))
    # half_size3 = [int(np.shape(E_out)[0]/2),int(np.shape(E_out)[1]/2)]
    # E_out = E_out[half_size3[0]-int(M/2):half_size3[0]+int(M/2),half_size3[1]-int(N/2):half_size3[1]+int(N/2)]
    # E_out = E_out[half_size3[0]-int(5017/2):half_size3[0]+int(5017/2),half_size3[1]-int(5017/2):half_size3[1]+int(5017/2)]
    print('Output pixel pitch: ',pixel_pitch_out[0]* 10**6,'um')
    return E_out




L = 65e-3
z_micro = 5e-3
output_z = 800e-3
signal_size = 1024 # Size of visualization
# Magn = L/z_micro
Magn = 1
#-------------Pixel Size Reconstruction-------------
dx = (20.48e-3)/(1024)
# dx = 5.32e-6 
dy = dx
dx_out = dx/Magn
dy_out = dy/Magn
print('Input pixel pitch: ',dx* 10**6,'um')
# print('Output pixel pitch: ',dx_out* 10**6,'um')
#-------------Pixel Size Simulation-------------
# dx_out = 5.86e-6 
# dy_out = dx_out
# dx = dx_out/Magn
# dy = dy_out/Magn

M = N = signal_size # Control of the size of the matrices


radius = 1e-5 # Radius of the aperture in meters
Pradius = int(radius/dx) #Radius of the aperture in pixels

x_inp_lim = dx*int(N/2)
y_inp_lim = dy*int(M/2)


# Light source
wavelength = 532e-9 # Wavelength of the illumination Source
k = 2*pi/wavelength # Wave number of the ilumination source


# output_z = L-z_micro   # Z Component of the observation screen coordinates


Input_Z = 0 # Z Component of the aperture coordinates


# Aperture defines the geometry of the apperture, for circular apperture use circ2D, for rectangular apperture use rect2D

# amporg = Image.open(r"D:\OneDrive - Universidad EAFIT\Semestre X\rec_Am_holo_org.png").convert('L')
# Amp_org = np.asarray(amporg)/255
# mask = Image.open(r"D:\OneDrive - Universidad EAFIT\Semestre X\mask.png").convert('L')
# mask = np.asarray(mask)/255
# im = 0.1*mask + 0.9*Amp_org

# im = Image.open(r"D:\OneDrive - Universidad EAFIT\Semestre IX\Advanced Project 2\Final Presentation\USF_B.bmp").convert('L')
# im = np.asarray(im)/255
# ref = Image.open(r"D:\OneDrive - Universidad EAFIT\Semestre IX\Advanced Project 2\Final Presentation\ref_b.bmp").convert('L')
# ref = np.asarray(ref)/255
# im = (ref-im)[454:454+512,625:625+512]

# im = Image.open(r"USAF_EXP.png").convert('L')
# im = Image.open(r"D:\OneDrive - Universidad EAFIT\Semestre X\Holo_USAFFULL-1_633_1000_5_3_3_rec__1100.jpg").convert('L')
# im = Image.open(r"D:\OneDrive - Universidad EAFIT\Semestre X\Holo_USAFFULL-1_633_1000_5_3_3_rec__1100.bmp").convert('L')
im = Image.open(r"F:\OneDrive - Universidad EAFIT\Semestre X\TDG\Images\HIGH_NA.jpg").convert('L')
# im = Image.open(r"D:\OneDrive - Universidad EAFIT\Semestre IX\Advanced Project 2\Final Presentation\contrast2.bmp").convert('L')
# im = Image.open(r"D:\OneDrive - Universidad EAFIT\Semestre IX\Advanced Project 2\Final Presentation\arago.bmp").convert('L')
# im = Image.open(r"D:\OneDrive - Universidad EAFIT\Semestre IX\Advanced Project 2\Final Presentation\Holofase.bmp").convert('L')
# im = circ2D(signal_size,Pradius,center=None)
# im = Image.open(r"D:\OneDrive - Universidad EAFIT\Semestre IX\Advanced Project 2\USAFFULL.jpg").convert('L')
# im = Image.open(r"D:\OneDrive - Universidad EAFIT\Semestre IX\Advanced Project 2\USAF-1951.svg.png").convert('L')

# im = im.resize((signal_size,signal_size))
im = np.asarray(im)/255
# im = 1 - im

M,N = np.shape(im)
x_cord = np.linspace(-x_inp_lim , x_inp_lim , num = N)
y_cord = np.linspace(-y_inp_lim , y_inp_lim , num = M)
[X_inp,Y_inp] = np.meshgrid(x_cord,y_cord)
Rinp = np.sqrt(np.power(X_inp,2)+np.power(Y_inp,2) + (5e-3)**2)
# ill = np.exp(1j*im)
# ill = np.ones_like(im,dtype='co   lex') * np.exp(1j*k*Rinp)/Rinp
ill = np.ones_like(im,dtype='complex')

U1 = im.copy()*ill

# U0_temp_pure = CONV_SAASM(U1, output_z, wavelength, [dx,dy],[dx_out,dy_out])
output_zs = np.linspace(0.5e-3,2e-3,10)
# for output_z in output_zs:
#     U0_temp_pure = CONV_SAASM_V2(U1, output_z, wavelength, [dx,dy],[dx_out,dy_out])
#     # U0_temp_pure = CONV_SAASM(U1, output_z, wavelength, [dx,dy],[dx_out,dy_out])
#     U0_temp_pure = U0_temp_pure[710:1333,710:1333] # ERASE FOR GENERAL RESULTS
#     title = 'Z = '+str(output_z*1e3)+' mm'
#     ishow(U0_temp_pure,title=title)


U0_temp_pure = CONV_SAASM_V3(U1, output_z, wavelength, [dx,dy],[dx_out,dy_out])
# U0_temp = intensity(U0_temp_pure,False)
# U0_temp = Image.fromarray(U0_temp)
# U0_temp = np.asarray(U0_temp.resize((signal_size,signal_size)))
# U0 = CONV_SAASM(U0_temp, -output_z, wavelength, [dx_out,dy_out],[dx,dy])
VW_in = [-x_inp_lim,x_inp_lim,-y_inp_lim,y_inp_lim]


# U0_temp_pure = U0_temp_pure[710:1333,710:1333] # ERASE FOR GENERAL RESULTS
plotea(U1,U0_temp_pure)
inte = intensity(U0_temp_pure,False)*255
inte = inte.astype(np.uint8)
output = Image.fromarray(inte)
output = output.convert("L")
# output.save(r"D:\OneDrive - Universidad EAFIT\Semestre X\SAASM_experimental.bmp")























