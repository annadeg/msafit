# coding: utf-8
import glob
import numpy as np

import scipy.interpolate
import astropy.constants, astropy.units, astropy.wcs
from  astropy.io import fits

import ppxf
import ppxf.ppxf_util
import ppxf.miles_util

import sedpy

from ..lib import velocity_field
from ..lib import light_profile
from ..lib import cube_cont

try:
    import fsps_mist_c3k_util
except ModuleNotFoundError as e:
    print('It seems you are not Francesco, so you don\'t have this module')
    raise e

def main():
    size_x, size_y = 3.0, 3.0
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

    filenames = sorted(glob.glob('fsps_with____nebcont/*fits'))
    ages, mets = [np.zeros(len(filenames)) for _ in 'ab']
    for i,f in enumerate(filenames):
        # Now this is really clever, and equally bad. `age_metal` should be
        # a `classmethod`, not a bound method. But meh, for the sake of not
        # breaking backwards compatibility men will go to unfathomable lows.
        ages[i], mets[i] = fsps_mist_c3k_util.fsps_mist_c3k.age_metal(
            None, f)

    age, met = 2.5, 0.
    index = np.argmin((age-ages)**2 + (met-mets**2))
    with fits.open(filenames[index]) as hdu:
        spec_wcs = astropy.wcs.WCS(hdu[0].header)
        wave = spec_wcs.all_pix2world(np.arange(spec_wcs._naxis[0]), 0)[0]
        spec = hdu[0].data * astropy.constants.L_sun / astropy.units.AA
        dist = 10*astropy.units.pc
        spec /= (4.*np.pi*dist**2)
        spec = spec.to('erg/(s AA cm2)')
        spec = spec.value # in erg/(s AA cm2)

    Cube3D = cube_cont.CubeCont(xs, ys, wave)
    Cube3D.computeSpectrum(
        Light2D, vel2D, parameters, wave, spec, 0, h3=-.1, h4=-0.1)
    Cube3D.writeIPSObject('test_FSPS.fits')
    Cube3D.writeFitsData('test_FSPS.fits')

    # Check output.
    data = fits.open('test_FSPS.fits')
    cube_wcs = astropy.wcs.WCS(data[0].header)
    wave = cube_wcs.all_pix2world(0, 0, np.arange(cube_wcs._naxis[2]), 0)[2]
    flux = np.sum(data[0].data, axis=(1,2))
    astro_filter = sedpy.observate.load_filters(('sdss_r0',))[0]

    assert np.isclose(astro_filter.ab_mag(wave, flux), parameters['tot_AB_mag']), 'Mismatch in magnitude'

    # Check sol.
    spec = data[0].data[:, 29, 29]
    _, _, velscale = ppxf.ppxf_util.log_rebin(wave[[0, -1]], wave)
    velscale = np.round(velscale, decimals=7)
    galaxy_spec, log_wave, _ = ppxf.ppxf_util.log_rebin(
        wave[[0, -1]], spec, velscale=velscale)
    if len(log_wave)<len(wave):
        velscale -= 1.e-8 # Address rounding error.
        galaxy_spec, log_wave, _ = ppxf.ppxf_util.log_rebin(
            wave[[0, -1]], spec, velscale=velscale)

    w,f = np.loadtxt('fsps_with____nebcont/LSF-Config_fsps').T
    fwhm_interp = scipy.interpolate.interp1d(w, f*1.01, bounds_error=True)
    templ = fsps_mist_c3k_util.fsps_mist_c3k(
        'fsps_with____nebcont/*fits', velscale[0],
        fwhm_interp, 'fsps_with____nebcont/LSF-Config_fsps')
    templates = templ.templates
    templates = templates.reshape(len(templates), -1)
    pp = ppxf.ppxf.ppxf(
        templates, galaxy_spec, galaxy_spec*0+1, velscale, [0, 200], moments=4,
        lam=np.exp(log_wave))
    pp.plot()
    pp.wave = wave
    pp.spec = spec

    return pp


if __name__=="__main__":
    main()
