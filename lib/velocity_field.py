import numpy as np
from matplotlib import pyplot as plt
from astropy.io import fits 

class VelField2D(object):
    
    def __init__(self, VelCurve1D):
        self.__velcurve1D = VelCurve1D
        
    def model(self, x, y, parameters_dict):
        
        #Taking necessary parameters from parameter dictonary with 
        PA_rad = parameters_dict["PA"]/180.0*np.pi 
        inclination = parameters_dict["inclination"]/180.0*np.pi
        
        # Applying projection regarding inclination and projection angle 
        sini = np.sin(inclination)
        cosi = np.cos(inclination)
        cos2i = 1.0-sini**2
        x_rot = x*np.cos(PA_rad)-y*np.sin(PA_rad)
        y_rot = x*np.sin(PA_rad)+y*np.cos(PA_rad)
        r = np.sqrt(x_rot**2+(y_rot**2)/cos2i)
        
        # Mapping 1D velocity curve onto the 
        velocity = self.__velcurve1D(r, parameters_dict)*(x_rot/r)*np.sin(inclination)
        select = r==0
        velocity[select] = 0.0 
        return velocity


class VelCurve1D(object):
    def __init__(self, function):
        self.__function1D = function
        
    def model(self, r, parameter_dict):
        velocity = self.__function1D(r, parameter_dict)
        return velocity
    

def ArcTan1D(r, parameters_dict):
    v_out = (2.0 / np.pi) * parameters_dict['v_asympt'] * np.arctan(r / parameters_dict['r_turnover'])
    return v_out


if __name__ == "__main__":
    r = np.arange(-10,10,0.1)
    parameters = {"v_asympt":200, "r_turnover":0.2, "PA": 10, "inclination": 89.0}
    vel_2D = VelField2D(ArcTan1D)
    
    (x,y) = np.indices([100,100])
    x_cent = 50.1
    y_cent = 50.1
    pix_scale = 0.1 
    vel_field = vel_2D.model((x-x_cent)*pix_scale,(y-y_cent)*pix_scale, parameters)
    hdu = fits.PrimaryHDU(vel_field)
    hdu.writeto('test_vel.fits',overwrite=True)
