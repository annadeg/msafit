import numpy as np
from astropy.modeling import models
from astropy.io import fits


class LightDistribution2D(object):
    
    def __init__(self, LightFunction2D):
        self.__function2D = LightFunction2D

    def model(self, x, y, parameter_dict, out_surfacebrightness=True):
        light_distribution2D = self.__function2D(x, y, parameter_dict)
        
        total_flux = parameter_dict['total_flux']
        light_distribution2D = light_distribution2D/np.sum(light_distribution2D)*total_flux
        
        if out_surfacebrightness:
            pixel_area = np.fabs(x[1,0]-x[0,0])*np.fabs(y[0,1]-y[0,0])
            light_distribution2D = light_distribution2D/pixel_area
        
        return light_distribution2D
    
    
def Sersic2D(x, y, parameter_dict):
    model = models.Sersic2D(r_eff=parameter_dict['r_eff_light'], n=parameter_dict['n_index'], x_0=parameter_dict['cent_x_light'],y_0=parameter_dict['cent_y_light'], ellip=1-parameter_dict['axisratio_light'], theta=(parameter_dict['PA_light']-90)/180.0*np.pi)
    light_distribution2D = model(x, y)
    return light_distribution2D
        
        
if __name__ == "__main__":
    parameter_dict = {'r_eff_light':0.2, 'n_index':1.0, 'cent_x_light':0.0, 'cent_y_light':0.0, 'axisratio_light':np.cos(40.0/180.0*np.pi), 'PA_light':0.0, 'total_flux':10.0}
    
    size_x = 3.0
    size_y = 3.0
    sampling = 0.02
    xs= np.linspace(size_x/-2.0,size_x/2.0,int(size_x/sampling))
    ys= np.linspace(size_y/-2.0,size_y/2.0,int(size_y/sampling))
    ys,xs = np.meshgrid(ys, xs)

    Light2D = LightDistribution2D(Sersic2D)
    image = Light2D.model(xs, ys, parameter_dict)
    
    hdu = fits.PrimaryHDU(image)
    hdu.writeto('test_light.fits', overwrite=True)






        
