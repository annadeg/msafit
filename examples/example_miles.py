# coding: utf-8
import glob
import os
import warnings

import numpy as np

import astropy.constants, astropy.units, astropy.wcs
from  astropy.io import fits

import sedpy

from ..lib import velocity_field
from ..lib import light_profile
from ..lib import cube_cont

import ppxf
import ppxf.ppxf_util
import ppxf.miles_util


def main():

    # size_x, size_y = 3.0, 3.0
    size_x, size_y = 0.5, 0.5 # Smaller cube for faster computing.
    sampling = 0.05
    xs = np.linspace(size_x/-2.0, size_x/2.0, int(size_x/sampling))
    ys = np.linspace(size_y/-2.0, size_y/2.0, int(size_y/sampling))
    xs,ys = np.meshgrid(xs, ys)

    vel2D = velocity_field.VelField2D(
        velocity_field.ArcTan1D, velocity_field.ConstantVdisp)
    Light2D = light_profile.LightDistribution2D(
        light_profile.Sersic2D)
    parameters = {
        'r_eff_light':0.2, 'n_index':1.0, 'cent_x_light':0.0, 'cent_y_light':0.0,
        'axisratio_light':np.cos(50.0/180.0*np.pi), 'PA_light':20.0,
        'total_flux':1, # Not relevant for cube.
        "v_asympt":200, "r_turnover":0.2, "PA": 20.0, "inclination": 50.0,
        'cent_x_vel':0.0, 'cent_y_vel':0.0,'sigma_0':50.,
        'tot_AB_mag': 27., 'tot_AB_filter': 'sdss_r0'}


    # Load library of SSP spectra.
    miles_path = os.path.dirname(os.path.realpath(ppxf.miles_util.__file__))
    filenames = sorted(glob.glob(os.path.join(miles_path, 'miles_models/*Z*T*fits')))
    ages, mets = [np.zeros(len(filenames)) for _ in 'ab']
    for i,f in enumerate(filenames):
        ages[i], mets[i] = ppxf.miles_util.age_metal(f)

    age, met = 2.5, 0.
    index = np.argmin((age-ages)**2 + (met-mets**2))
    with fits.open(filenames[index]) as hdu:
        spec_wcs = astropy.wcs.WCS(hdu[0].header)
        wave = spec_wcs.all_pix2world(np.arange(spec_wcs._naxis[0]), 0)[0]
        #select_range = (wave>=3600.) & (wave<=7500.)
        select_range = (wave>=-np.inf) & (wave<=np.inf)
        spec = hdu[0].data * astropy.constants.L_sun / astropy.units.AA
        dist = 10*astropy.units.pc
        spec /= (4.*np.pi*dist**2)
        spec = spec.to('erg/(s AA cm2)')
        spec = spec.value # in erg/(s AA cm2)
        spec, wave = spec[select_range], wave[select_range]

    Cube3D = cube_cont.CubeCont(xs, ys, wave)
    Cube3D.computeSpectrum(
        Light2D, vel2D, parameters, wave, spec, 0, h3=-.1, h4=-0.1,
        resample_factor=10)
    Cube3D.writeIPSObject('test_MILES.fits')
    Cube3D.writeFitsData('test_MILES.fits')

    # Check output.
    data = fits.open('test_MILES.fits')
    cube_wcs = astropy.wcs.WCS(data[0].header)
    wave = cube_wcs.all_pix2world(0, 0, np.arange(cube_wcs._naxis[2]), 0)[2]
    flux = np.sum(data[0].data, axis=(1,2))
    astro_filter = sedpy.observate.load_filters(('sdss_r0',))[0]

    assert np.isclose(astro_filter.ab_mag(wave, flux), parameters['tot_AB_mag']), 'Mismatch in magnitude'


    # Check sol.
    spec = data[0].data[:, 4, 4]
    _, _, velscale = ppxf.ppxf_util.log_rebin(wave[[0, -1]], wave)
    velscale = np.round(velscale, decimals=7)
    galaxy_spec, log_wave, _ = ppxf.ppxf_util.log_rebin(
        wave[[0, -1]], spec, velscale=velscale)
    if len(log_wave)<len(wave):
        print('Beekj')
        velscale -= 1.e-8 # Solve rounding error.
        galaxy_spec, log_wave, _ = ppxf.ppxf_util.log_rebin(
            wave[[0, -1]], spec, velscale=velscale)

    FWHM_gal = 2.51 * 1.01 # This is the MILES resolution, we used MILES for our "galaxy".
    templ = ppxf.miles_util.miles(
        os.path.join(miles_path, 'miles_models/*Z*T*fits'), velscale, FWHM_gal)
    templates = templ.templates
    templates = templates.reshape(len(templates), -1)
    c = 299792.458   # km/s
    dv = c*(templ.ln_lam_temp[0]-log_wave[0])
    noise = galaxy_spec / 50.
    #galaxy_spec = np.random.normal(galaxy_spec, noise)
    pp = ppxf.ppxf.ppxf(
        templates, galaxy_spec, noise*0+1, velscale, [0, 200], moments=4,
        lam=np.exp(log_wave), vsyst=dv)
    pp.plot()
    pp.wave = wave
    pp.spec = spec

    warnings.warn(
        'Note: h3/h4 different from input. This goes away if we avoid rebinning to linear scale (e.g. change cube_cont to write log-binned spectrum, then replace galaxy_spec with spec in ppxf.\n'
        'The order of down-sampling and linear binning does not change the results',
        UserWarning)

    return pp


if __name__=="__main__":
    main()
