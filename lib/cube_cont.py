import numpy as np
from astropy.io import fits
import scipy.ndimage
import tqdm
import warnings

import ppxf.ppxf
import ppxf.ppxf_util

import sedpy

from .velocity_field import ArcTan1D, ConstantVdisp, VelField2D
from .light_profile import Sersic2D, LightDistribution2D
from .cube import Cube


fact = np.sqrt(2.*np.pi)

class CubeCont(Cube):

    def computeSpectrum(
        self, modelLight2D, modelVel2D, parameters_dict, input_wave, input_spectrum,
        redshift, h3=0, h4=0, velscale=None, resample_factor=10):

        assert parameters_dict.get('total_flux', 1.)==1., f'`total_flux` must be 1 in `parameters_dict`'
        cube_out = np.zeros(self._dim)

        astro_filter = sedpy.observate.load_filters(
            (parameters_dict['tot_AB_filter'],))[0]
        norm_factor = astro_filter.ab_mag(input_wave, input_spectrum)
        norm_factor = 10**(0.4 * (norm_factor - 1.))
        input_spectrum = input_spectrum * norm_factor

        n_spec = 1 # = 2 for two-sided fits. Not relevant here.
        sigma_diff = 0 # Used for unconvolved templates.
        n_comp = 1 # Number of distinct kinematic components. Not relevant here.
        vsyst = 0 # TODO Change according to redshift

        # Obtain velscale.
        _, _, velscale = ppxf.ppxf_util.log_rebin(
            input_wave[[0, -1]], input_wave*0, velscale=velscale)
        velscale = np.round(velscale, decimals=10) # To avoid rounding issues.

        # Obtain log-binned spectrum.
        log_input_spectrum, log_wave, _ = ppxf.ppxf_util.log_rebin(
            input_wave[[0, -1]], input_spectrum, velscale=velscale)
        if len(log_wave)<len(input_wave):
            velscale -= 1.e-8 # Solve rounding error.
            log_input_spectrum, log_wave, _ = ppxf.ppxf_util.log_rebin(
                input_wave[[0, -1]], input_spectrum, velscale=velscale)

        # Need high-res spectrum to oversample and ensure accuracy of convolution with LOSVD
        log_input_spectrum_hires = np.copy(log_input_spectrum)
        log_input_spectrum_hires = scipy.ndimage.interpolation.zoom(
           log_input_spectrum_hires, resample_factor, order=3)

        # Hi-resolution output grid.
        log_wave_hires = scipy.ndimage.interpolation.zoom(
            log_wave, resample_factor, order=3)
        wave_hires = np.linspace(*np.exp(log_wave_hires[[0, -1]]), num=len(log_wave_hires))
        #wave_hires = np.linspace(
        #    input_wave[0], input_wave[-1], len(input_wave)*resample_factor)
        #_, log_wave_hires, _ = ppxf.ppxf_util.log_rebin(
        #    input_wave[[0, -1]], wave_hires, velscale=velscale/resample_factor)

        # Get FFT of spectrum.
        npad = 2**int(np.ceil(np.log2(len(log_input_spectrum_hires))))
        spectrum_rfft = np.fft.rfft(log_input_spectrum_hires, npad)

        light_map = modelLight2D.model(
            self._x, self._y, parameters_dict,
            out_surfacebrightness=False)

        velocity_map, dispersion_map = modelVel2D.model(
            self._x, self._y, parameters_dict)

        losvd_parameters = np.array([
            velocity_map, dispersion_map]) / velscale

        if h3!=0:
            h3_map = np.full_like(velocity_map, h3)
            losvd_parameters = np.vstack([
                losvd_parameters, h3_map[None, :, :]])

        if h4!=0:
            if h3==0: # Always need an h3 map if h4 is used. Use 0 if necessary.
                h3_map = np.full_like(velocity_map, 0.)
                losvd_parameters = np.vstack([
                    losvd_parameters, h3_map[None, :, :]])

            h4_map = np.full_like(velocity_map, h4)
            losvd_parameters = np.vstack([
                losvd_parameters, h4_map[None, :, :]])

        n_moments = (len(losvd_parameters),)

        # Nested loop over all spaxels; heavy lifting here.
        for j in (pbarj:=tqdm.tqdm(range(len(light_map)), colour='blue')):
            pbarj.set_description('Cube     ')
            for k in (pbark:=tqdm.tqdm(range(len(light_map[j])), colour='green', leave=False)):
                pbark.set_description(f'Slice {j: 3d}')

                losvd_rfft = ppxf.ppxf.losvd_rfft(
                    losvd_parameters[:, j, k], n_spec, n_moments,
                    len(spectrum_rfft), n_comp, vsyst, resample_factor,
                    sigma_diff)
                
                comp_index, nspec_index = 0, 0
                spec_out = np.fft.irfft(
                    spectrum_rfft * losvd_rfft[:, comp_index, nspec_index],
                    npad)

                # Remove 0 padding.
                spec_out = spec_out[:len(input_wave)*resample_factor]

                # Rebin to linear, then downsample to original grid.
                spec_out = np.interp(
                    wave_hires, np.exp(log_wave_hires), spec_out)
                spec_out = ppxf.ppxf.rebin(
                    spec_out, resample_factor)
                """
                # This is a viable alternative to the two-steps process above.
                spec_out = np.interp(
                    input_wave, np.exp(log_wave_hires), spec_out)
                """
                
                cube_out[:, j, k] = spec_out * light_map[j, k]


        temp_flux    = np.nansum(cube_out, axis=(1, 2))
        temp_mag     = astro_filter.ab_mag(input_wave, temp_flux)
        rescale_flux = 10**(0.4*(temp_mag - parameters_dict['tot_AB_mag']))
                  
        self.data = cube_out * rescale_flux # cgs

            
            
if __name__ == "__main__":

    parameters = { 
        'r_eff_light':0.2, 'n_index':1.0, 'cent_x_light':0.0, 'cent_y_light':0.0,
        'axisratio_light':np.cos(50.0/180.0*np.pi), 'PA_light':20.0,
        'total_flux':10.0,
        "v_asympt":200, "r_turnover":0.2, "PA": 20.0, "inclination": 50.0,
        'cent_x_vel':0.0, 'cent_y_vel':0.0,'sigma_0':50.}
    h3=-.1 
    h4=-0.3
    
    size_x = 3.0
    size_y = 3.0
    sampling = 0.05
    xs = np.linspace(size_x/-2.0, size_x/2.0, int(size_x/sampling))
    ys = np.linspace(size_y/-2.0, size_y/2.0, int(size_y/sampling))
    xs, ys = np.meshgrid(xs, ys)
    wave = np.arange(2.6,2.65,0.0001)
    
    vel2D = VelField2D(ArcTan1D,ConstantVdisp)
    Light2D = LightDistribution2D(Sersic2D)

    # Create the input spectrum.
    wave = np.arange(2.6, 2.65, 0.0001)

    spec0 = 1/np.sqrt(2*np.pi)/0.0005*np.exp(-0.5*((wave-2.63)/0.0005)**2)
    spec1 = 0.5/np.sqrt(2*np.pi)/0.0005*np.exp(-0.5*((wave-2.605)/0.0005)**2)
    spec = spec0+spec1

    Cube3D = Cube(xs, ys, wave)
    Cube3D.computeSpectrum(
        [Light2D], [vel2D], [parameters], wave, spec, 0, h3=h3, h4=h4)
    Cube3D.writeFitsData("test_cont_cube.fits")
    Cube3D.writeIPSObject("test_cube.fits")
