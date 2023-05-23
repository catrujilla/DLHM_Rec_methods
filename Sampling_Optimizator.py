import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import NonlinearConstraint 
from scipy import optimize
from timeit import default_timer as timer
import plotly.graph_objects as go


def fmincon(minfunc, Constraints, Initial_values): 
    '''
    Variables:
    minfunc: function to minimize
    lb and ub: float variables indicating low and upper bounderies of the search range
    Cy: float variable indicating the initial phase factor curvature (seed)
    '''

    start = timer() #Start to count time
    # nlc = NonlinearConstraint(fun = minfunc, lb = lb, ub = ub)
    out = optimize.minimize(fun = minfunc, x0 = Initial_values, method='SLSQP', constraints=Constraints) #The 'SLSQP' is equivalent to the default 'interior-point' algorithm that MATLAB's 'fmincon' function uses.
    if out.success:
        print("Optimization successful!")
    else:
        print("Optimization failed.")
    Cy_opt = out.x
    print("Processing time FMC:", timer()-start) #Time for FMC execution
    return Cy_opt


z = 3.5e-3
wvl = 6.33e-7
in_size = [128,128]
Magn = 0.5
ppitch = 3.3e-6
coef = 2 * ppitch**2 * (Magn+Magn**2)/wvl


comp2 = lambda x: x[0]*in_size[0]*x[1]*in_size[1]

nyquistx = lambda x: x[0]
lbx = (coef * in_size[0]**2 /np.sqrt(z**2 + (1+Magn)**2 * ppitch**2 * in_size[0]**2))
x_constraints = NonlinearConstraint(nyquistx, lb = lbx, ub = np.inf)

nyquisty = lambda x: x[1]
lby = (coef * in_size[1]**2 /np.sqrt(z**2 + (1+Magn)**2 * ppitch**2 * in_size[1]**2)) 
y_constraints = NonlinearConstraint(nyquisty, lb = lby, ub = np.inf)
cns = [x_constraints,y_constraints]

Cy = [lbx,lby]
sol = fmincon(comp2,cns,Cy)
print(sol)
x=np.linspace(in_size[0]/2,2*sol[0],500)
y=np.linspace(in_size[1]/2,2*sol[1],500)
[X,Y]=np.meshgrid(x,y,indexing='xy')
Z = X*Y*in_size[0]*in_size[1]
restrx = go.Scatter3d(x=x,y=lbx*np.ones_like(x),z=np.zeros_like(x),marker={'size':3,'color':'#1ec7ab'})
restry = go.Scatter3d(x=lby*np.ones_like(y),y=y,z=np.zeros_like(y),marker={'size':3,'color':'#1ec7ab'})
surf = go.Surface(z=Z,x=X,y=Y,opacity=0.7)
scat = go.Scatter3d(x=[sol[0]],y=[sol[1]],z=[comp2(sol)])
fig = go.Figure(data=[surf,scat,restrx,restry])
fig.show()

