import numpy as np
from velocity_field import ArcTan1D, ConstantVdisp, VelField2D
from light_profile import Sersic2D, LightDistribution2D
from astropy.io import fits

fact = np.sqrt(2.*np.pi)

def Gaussian1D(x, flux, cent, width):
    if len(flux.shape)==2:
        y=flux[np.newaxis,:, :]*np.exp(-0.5*((x[:,np.newaxis, np. newaxis]-cent[np.newaxis,:, :])/abs(width[np.newaxis,:, :]))**2)/(fact*abs(width[np.newaxis,:, :]))
    else:
        y=flux*np.exp(-0.5*((x-cent)/abs(width))**2)/(fact*abs(width))
    return y


class Cube(object):
    def __init__(self, x_grid, y_grid, wave_grid):
        self.__x = x_grid
        self.__y = y_grid
        self.__wave = wave_grid
        self.__dim = [wave_grid.shape[0], x_grid.shape[0],x_grid.shape[1],]
        self.__data = None
        
        
    def computeLine(self, modelLight2D, modelVel2D, parameters_dict, rest_wave, redshift):
        
        if (len(modelLight2D) == len(modelVel2D)) and (len(modelLight2D) == len(parameters_dict)):
            cube_out = np.zeros(self.__dim)
            for i in range(len(modelLight2D)):
                velocity_map, dispersion_map = modelVel2D[i].model(self.__x, self.__y, parameters_dict[i])
                light_map = modelLight2D[i].model(self.__x, self.__y, parameters_dict[i])
                
                wave_cent = ((velocity_map/300000.0)+1)*rest_wave*(redshift+1)
                wave_width = (dispersion_map/300000.0)*rest_wave 
                cube_out += Gaussian1D(self.__wave, light_map, wave_cent, wave_width) 
            self.__data = cube_out
            
    def writeFitsData(self, fileout):
        hdu = fits.PrimaryHDU(self.__data)
        hdu.header['CRVAL3'] = self.__wave[0]
        hdu.header['CDELT3'] = self.__wave[1]-self.__wave[0]
        hdu.header['CRPIX3'] = 1
        hdu.writeto(fileout, overwrite=True)
        
if __name__ == "__main__":
    parameters = {'r_eff_light':0.2, 'n_index':1.0, 'cent_x_light':0.0, 'cent_y_light':0.0, 'axisratio_light':np.cos(50.0/180.0*np.pi), 'PA_light':20.0, 'total_flux':10.0, "v_asympt":200, "r_turnover":0.2, "PA": 20.0, "inclination": 50.0, 'cent_x_vel':0.0, 'cent_y_vel':0.0,'sigma_0':50.}
    
    size_x = 3.0
    size_y = 3.0
    sampling = 0.05
    xs= np.linspace(size_x/-2.0,size_x/2.0,int(size_x/sampling))
    ys= np.linspace(size_y/-2.0,size_y/2.0,int(size_y/sampling))
    xs,ys = np.meshgrid(xs, ys)
    wave = np.arange(2.6,2.65,0.0001)
    
    vel2D = VelField2D(ArcTan1D,ConstantVdisp)
    Light2D = LightDistribution2D(Sersic2D)
    Cube3D = Cube(xs, ys, wave)
    Cube3D.computeLine([Light2D], [vel2D], [parameters], 0.65648, 3.0)
    Cube3D.writeFitsData("test_cube.fits")
    
    
