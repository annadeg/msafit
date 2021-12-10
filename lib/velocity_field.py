import numpy as np
from matplotlib import pyplot as plt
from astropy.io import fits 

class VelField2D(object):
    
    def __init__(self, VelCurve1D):
        self.__velcurve1D = VelCurve1D
        
    def model(self, x, y, parameters_dict):
        
        #Taking necessary parameters from parameter dictonary with 
        PA_rad = (parameters_dict["PA"]+90.0)/180.0*np.pi 
        inclination = parameters_dict["inclination"]/180.0*np.pi
        x_cent = parameters_dict["cent_x_vel"]
        y_cent = parameters_dict["cent_y_vel"]
        # Applying projection regarding inclination and projection angle 
        sini = np.sin(inclination)
        cosi = np.cos(inclination)
        cos2i = 1.0-sini**2
        x_rot = (x-x_cent)*np.cos(PA_rad)-(y-y_cent)*np.sin(PA_rad)
        y_rot = (x-x_cent)*np.sin(PA_rad)+(y-y_cent)*np.cos(PA_rad)
        
        # Mapping 1D velocity curve onto the grid
        if parameters_dict["inclination"] == 90:
            velocity = self.__velcurve1D(x_rot, parameters_dict)
            pscale = np.diff(y)[0][0]
            select = np.abs(y_rot) > (pscale/2)
            velocity[select] = 0.

        else:
            r = np.sqrt(x_rot**2+(y_rot**2/cos2i))        
            velocity = self.__velcurve1D(r, parameters_dict)*(x_rot/r)*sini
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
    r = np.arange(-10,10,0.05)
    parameters = {"v_asympt":200, "r_turnover":0.2, "PA": 20.0, "inclination": 80.0, 'cent_x_vel':0.0, 'cent_y_vel':0.0,}
    vel_2D = VelField2D(ArcTan1D)
    
    
    size_x = 3.0
    size_y = 3.0
    sampling = 0.1
    xs= np.linspace(size_x/-2.0,size_x/2.0,int(size_x/sampling))
    ys= np.linspace(size_y/-2.0,size_y/2.0,int(size_y/sampling))
    xs,ys = np.meshgrid(xs, ys)
    
    #(x,y) = np.indices([30,30])
    #x_cent = 15.1
    #y_cent = 15.1
    #pix_scale = 0.01
    #print((y-y_cent)*pix_scale,ys*pix_scale)
    vel_field = vel_2D.model(xs,ys, parameters)
    hdu = fits.PrimaryHDU(vel_field)
    hdu.writeto('test_vel.fits',overwrite=True)
