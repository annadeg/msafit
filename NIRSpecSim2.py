from astropy.io import fits as pyfits
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt
from astropy import constants as const
import numpy
from skimage.feature import register_translation
from scipy import ndimage
from scipy import interpolate

class NIRSpec_PCE:
    def __init__(self,file_pce):
        hdu = pyfits.open(file_pce)
        self.lib = hdu[1].data
        self.wave = self.lib.field('WAVE')
        hdu.close()
    
    def define_PCE(self,fwa,gwa):
        self.pce = self.lib.field('%s_%s'%(fwa,gwa))
    
    def get_PCE(self,wave=None):
        if wave is None:
            return self.wave, self.pce
        else:
            interp = interp1d(self.wave, self.pce, kind='cubic')
            return interp(wave)
        
    def get_counts(self,wave,flux):
        (wave,pce) = self.get_PCE
        electron = flux*pce

class NIRSpec_TRACE_LIB:
    def __init__(self,file_trace):
        self.hdu = pyfits.open(file_trace,memmap=True)
        self.wave = self.hdu[5].data.field('WAVE')

    def getTrace(self,QD,i,j):
        (trace_FPA_x,trace_FPA_y) = self.hdu[QD].data[:,:,j-1,i-1].T
        trace_x = trace_FPA_x
        trace_y = trace_FPA_y
        return NIRSpec_TRACE(wave=self.wave,trace_x=trace_x,trace_y=trace_y)

        
class NIRSpec_TRACE:
    def __init__(self,wave=None,trace_x=None,trace_y=None):
        self.wave = wave
        self.trace_x = trace_x
        self.trace_y = trace_y
        
    def interpolateTrace(self,wave_new):
        x_int = interp1d(self.wave,self.trace_x,bounds_error=False,fill_value=0)
        y_int = interp1d(self.wave,self.trace_y,bounds_error=False,fill_value=0)
        return NIRSpec_TRACE(wave_new, x_int(wave_new), y_int(wave_new))
        

class NIRSpec_PSF:
    def __init__(self,psf,oversample):
        self.psf = psf
        self.dim = psf.shape
        self.oversample = oversample
        
    def rebin(self,binning=100):
        psf_shape = self.psf.reshape((self.dim[0],int(self.dim[1]//binning),int(binning)))
        psf_out = numpy.sum(numpy.sum(psf_shape,-1).reshape(int(self.dim[0]//binning),int(binning),int(self.dim[1]//binning)),1)
        return NIRSpec_PSF(psf_out,oversample=self.oversample/binning)
    
    def shift_PSF(self,shift_x,shift_y):
        psf_out = ndimage.interpolation.shift(self.psf,(shift_y,shift_x),mode='nearest')
        return NIRSpec_PSF(psf_out,oversample=self.oversample)

class NIRSpec_PSFcube:
    def __init__(self,psfs,wave,oversample):
        self.psf_cube = psfs
        self.wave = wave
        self.oversample = oversample
    
    def getPSFwave(self,wave):
        wave_samp=self.wave[1]-self.wave[0]
        if wave not in self.wave:
            select = numpy.fabs(self.wave-wave)<=wave_samp
            wave_cut = self.wave[select]
            if len(wave_cut)>1:
                psf1 = self.psf_cube[select,:,:][0,:,:]
                psf2 = self.psf_cube[select,:,:][1,:,:]
                dist = (wave-wave_cut[0])/wave_samp
                psf_out = (psf2-psf1)*dist+psf1
            else:
                psf_out = self.psf_cube[select,:,:][0,:,:]
        else:
            select = self.wave==wave
            psf_out = self.psf_cube[select,:,:][0,:,:]
        return NIRSpec_PSF(psf_out,oversample=self.oversample)
    
    def getPSFindex(i):
        return NIRSpec_PSF(self._psf_cube[i,:,:],oversample=self.oversample)
        
    
class NIRSpec_PSFlib:
    def __init__(self,lib_name):
        self.hdu_lib = pyfits.open(lib_name,memmap=True)
        self.wave = self.hdu_lib['WAVELENGTH'].data
        self.xgrid = self.hdu_lib['XGRID'].data
        self.ygrid = self.hdu_lib['YGRID'].data
        self.irange = self.hdu_lib['IRANGE'].data
        self.jrange = self.hdu_lib['JRANGE'].data
        self.oversample = int(self.hdu_lib[0].header['OVERSAMP'])
        self.dim_x = self.hdu_lib[1].header['NAXIS1']
        self.dim_y = self.hdu_lib[1].header['NAXIS2']
        
    def get_PSF_shutter(self,i,j,x,y):
        if i in self.irange and j in self.jrange and x in self.xgrid and y in self.ygrid:
                select_x = self.xgrid==x
                select_y = self.ygrid==y
                select_i = self.irange==i
                select_j = self.jrange==j
                psfs = NIRSpec_PSFcube(self.hdu_lib[1].data[select_i,select_j,:,select_x,select_y,:,:][0,:,:,:],self.wave,oversample = self.oversample)
        elif i in self.irange and j in self.jrange:
            select_i = self.irange==i
            select_j = self.jrange==j
            arg_x=numpy.argsort((self.xgrid-x)**2)
            arg_y=numpy.argsort((self.ygrid-y)**2)
            psfs_out = numpy.zeros((len(self.wave),self.dim_y,self.dim_x),dtype=numpy.float32)
            for w in range(len(self.wave)):
                (shift_x, error, diffphase) = register_translation(self.hdu_lib[1].data[select_i,select_j,w,arg_x[0],arg_y[0],:,:][0,:,:], self.hdu_lib[1].data[select_i,select_j,w,arg_x[1],arg_y[0],:,:][0,:,:], 100)
                (shift_y, error, diffphase) = register_translation(self.hdu_lib[1].data[select_i,select_j,w,arg_x[0],arg_y[0],:,:][0,:,:], self.hdu_lib[1].data[select_i,select_j,w,arg_x[0],arg_y[1],:,:][0,:,:], 100)
                diff_x=x-self.xgrid[arg_x[:2]]
                diff_y=y-self.ygrid[arg_y[:2]]
                shift_x = numpy.fabs(shift_x[1])
                shift_y = numpy.fabs(shift_y[0])
                psf1=ndimage.interpolation.shift(self.hdu_lib[1].data[select_i,select_j,w,arg_x[0],arg_y[0],:,:][0,:,:],(shift_y*(diff_y[0]/numpy.sum(numpy.fabs(diff_y))),shift_x*(diff_x[0]/numpy.sum(numpy.fabs(diff_x)))),mode='nearest')
                psf2=ndimage.interpolation.shift(self.hdu_lib[1].data[select_i,select_j,w,arg_x[1],arg_y[0],:,:][0,:,:],(shift_y*(diff_y[0]/numpy.sum(numpy.fabs(diff_y))),shift_x*(diff_x[1]/numpy.sum(numpy.fabs(diff_x)))),mode='nearest')
                psf3=ndimage.interpolation.shift(self.hdu_lib[1].data[select_i,select_j,w,arg_x[0],arg_y[1],:,:][0,:,:],(shift_y*(diff_y[1]/numpy.sum(numpy.fabs(diff_y))),shift_x*(diff_x[0]/numpy.sum(numpy.fabs(diff_x)))),mode='nearest')
                psf4=ndimage.interpolation.shift(self.hdu_lib[1].data[select_i,select_j,w,arg_x[1],arg_y[1],:,:][0,:,:],(shift_y*(diff_y[1]/numpy.sum(numpy.fabs(diff_y))),shift_x*(diff_x[1]/numpy.sum(numpy.fabs(diff_x)))),mode='nearest')
                weight1 = 1.0/numpy.sqrt((diff_y[0]**2+diff_x[0]**2))
                weight2 = 1.0/numpy.sqrt((diff_y[0]**2+diff_x[1]**2))
                weight3 = 1.0/numpy.sqrt((diff_y[1]**2+diff_x[0]**2))
                weight4 = 1.0/numpy.sqrt((diff_y[1]**2+diff_x[1]**2))
                psfs_out[w,:,:] = (weight1*psf1+weight2*psf2+weight3*psf3+weight4*psf4)/(weight1+weight2+weight3+weight4)
            psfs = NIRSpec_PSFcube(psfs_out,self.wave,oversample=self.oversample)
        else:
            arg_i=numpy.argsort((self.irange-i)**2)
            arg_j=numpy.argsort((self.jrange-j)**2)
            diff_i=(i-self.irange[arg_i[:2]])/365.0
            diff_j=(j-self.jrange[arg_j[:2]])/171.0
            
            weight1 = 1.0/numpy.sqrt((diff_j[0]**2+diff_i[0]**2))
            weight2 = 1.0/numpy.sqrt((diff_j[0]**2+diff_i[1]**2))
            weight3 = 1.0/numpy.sqrt((diff_j[1]**2+diff_i[0]**2))
            weight4 = 1.0/numpy.sqrt((diff_j[1]**2+diff_i[1]**2))
            psfs_shutter = (weight1*self.hdu_lib[1].data[arg_i[0],arg_j[0],:,:,:,:,:]+weight2*self.hdu_lib[1].data[arg_i[1],arg_j[0],:,:,:,:,:]+weight3*self.hdu_lib[1].data[arg_i[0],arg_j[1],:,:,:,:,:]+weight4*self.hdu_lib[1].data[arg_i[1],arg_j[1],:,:,:,:,:])/(weight1+weight2+weight3+weight4)
            
            if x in self.xgrid and y in self.ygrid:
                psfs_out = psfs_shutter[:,select_x,select_y,:,:][0,:,:,:]
            else:
                arg_x=numpy.argsort((self.xgrid-x)**2)
                arg_y=numpy.argsort((self.ygrid-y)**2)
                psfs_out = numpy.zeros((len(self.wave),self.dim_y,self.dim_x),dtype=numpy.float32)
                for w in range(len(self.wave)):
                    (shift_x, error, diffphase) = register_translation(psfs_shutter[w,arg_x[0],arg_y[0],:,:], psfs_shutter[w,arg_x[1],arg_y[0],:,:], 100)
                    (shift_y, error, diffphase) = register_translation(psfs_shutter[w,arg_x[0],arg_y[0],:,:], psfs_shutter[w,arg_x[0],arg_y[1],:,:], 100)
                    diff_x=x-self.xgrid[arg_x[:2]]
                    diff_y=y-self.ygrid[arg_y[:2]]
                    shift_x = numpy.fabs(shift_x[1])
                    shift_y = numpy.fabs(shift_y[0])
                    psf1=ndimage.interpolation.shift(psfs_shutter[w,arg_x[0],arg_y[0],:,:],(shift_y*(diff_y[0]/numpy.sum(numpy.fabs(diff_y))),shift_x*(diff_x[0]/numpy.sum(numpy.fabs(diff_x)))),mode='nearest')
                    psf2=ndimage.interpolation.shift(psfs_shutter[w,arg_x[1],arg_y[0],:,:],(shift_y*(diff_y[0]/numpy.sum(numpy.fabs(diff_y))),shift_x*(diff_x[1]/numpy.sum(numpy.fabs(diff_x)))),mode='nearest')
                    psf3=ndimage.interpolation.shift(psfs_shutter[w,arg_x[0],arg_y[1],:,:],(shift_y*(diff_y[1]/numpy.sum(numpy.fabs(diff_y))),shift_x*(diff_x[0]/numpy.sum(numpy.fabs(diff_x)))),mode='nearest')
                    psf4=ndimage.interpolation.shift(psfs_shutter[w,arg_x[1],arg_y[1],:,:],(shift_y*(diff_y[1]/numpy.sum(numpy.fabs(diff_y))),shift_x*(diff_x[1]/numpy.sum(numpy.fabs(diff_x)))),mode='nearest')
                    weight1 = 1.0/numpy.sqrt((diff_y[0]**2+diff_x[0]**2))
                    weight2 = 1.0/numpy.sqrt((diff_y[0]**2+diff_x[1]**2))
                    weight3 = 1.0/numpy.sqrt((diff_y[1]**2+diff_x[0]**2))
                    weight4 = 1.0/numpy.sqrt((diff_y[1]**2+diff_x[1]**2))
                    psfs_out[w,:,:] = (weight1*psf1+weight2*psf2+weight3*psf3+weight4*psf4)/(weight1+weight2+weight3+weight4)
            psfs = NIRSpec_PSFcube(psfs_out,self.wave,oversample=self.oversample)
        return psfs

class NIRSpec_PSFlib2:
    def __init__(self,lib_name):
        self.hdu_lib = pyfits.open(lib_name,memmap=True)
        self.wave = self.hdu_lib['WAVELENGTH'].data
        self.xgrid = self.hdu_lib['XGRID'].data
        self.ygrid = self.hdu_lib['YGRID'].data
        self.oversample = int(self.hdu_lib[0].header['OVERSAMP'])
        self.dim_x = self.hdu_lib[1].header['NAXIS1']
        self.dim_y = self.hdu_lib[1].header['NAXIS2']

    def get_PSF_shutter(self,x,y):
        if x in self.xgrid and y in self.ygrid:
                select_x = self.xgrid==x
                select_y = self.ygrid==y
                psfs = NIRSpec_PSFcube(self.hdu_lib[1].data[:,select_x,select_y,:,:][0,:,:,:],self.wave,oversample = self.oversample)
        else:
            arg_x=numpy.argsort((self.xgrid-x)**2)
            arg_y=numpy.argsort((self.ygrid-y)**2)
            psfs_out = numpy.zeros((len(self.wave),self.dim_y,self.dim_x),dtype=numpy.float32)
            for w in range(len(self.wave)):
                print(self.hdu_lib[1].data[w,arg_x[0],arg_y[0],:,:].shape)
                (shift_x, error, diffphase) = register_translation(self.hdu_lib[1].data[w,arg_x[0],arg_y[0],:,:], self.hdu_lib[1].data[w,arg_x[1],arg_y[0],:,:], 100)
                (shift_y, error, diffphase) = register_translation(self.hdu_lib[1].data[w,arg_x[0],arg_y[0],:,:], self.hdu_lib[1].data[w,arg_x[0],arg_y[1],:,:], 100)
                diff_x=x-self.xgrid[arg_x[:2]]
                diff_y=y-self.ygrid[arg_y[:2]]
                shift_x = numpy.fabs(shift_x[1])
                shift_y = numpy.fabs(shift_y[0])
                psf1=ndimage.interpolation.shift(self.hdu_lib[1].data[w,arg_x[0],arg_y[0],:,:],(shift_y*(diff_y[0]/numpy.sum(numpy.fabs(diff_y))),shift_x*(diff_x[0]/numpy.sum(numpy.fabs(diff_x)))),mode='nearest')
                psf2=ndimage.interpolation.shift(self.hdu_lib[1].data[w,arg_x[1],arg_y[0],:,:],(shift_y*(diff_y[0]/numpy.sum(numpy.fabs(diff_y))),shift_x*(diff_x[1]/numpy.sum(numpy.fabs(diff_x)))),mode='nearest')
                psf3=ndimage.interpolation.shift(self.hdu_lib[1].data[w,arg_x[0],arg_y[1],:,:],(shift_y*(diff_y[1]/numpy.sum(numpy.fabs(diff_y))),shift_x*(diff_x[0]/numpy.sum(numpy.fabs(diff_x)))),mode='nearest')
                psf4=ndimage.interpolation.shift(self.hdu_lib[1].data[w,arg_x[1],arg_y[1],:,:],(shift_y*(diff_y[1]/numpy.sum(numpy.fabs(diff_y))),shift_x*(diff_x[1]/numpy.sum(numpy.fabs(diff_x)))),mode='nearest')
                weight1 = 1.0/numpy.sqrt((diff_y[0]**2+diff_x[0]**2))
                weight2 = 1.0/numpy.sqrt((diff_y[0]**2+diff_x[1]**2))
                weight3 = 1.0/numpy.sqrt((diff_y[1]**2+diff_x[0]**2))
                weight4 = 1.0/numpy.sqrt((diff_y[1]**2+diff_x[1]**2))
                psfs_out[w,:,:] = (weight1*psf1+weight2*psf2+weight3*psf3+weight4*psf4)/(weight1+weight2+weight3+weight4)
            psfs = NIRSpec_PSFcube(psfs_out,self.wave,oversample=self.oversample)
        return psfs
    
class Spectrum:
    def __init__(self,wave,flux):
        self.wave = wave # in units of m
        self.flux = flux # in units of erg/s/cm2/pixel
        
    def convert_to_photon(self,collecting_area=25.4):
        photon_energy = const.h.value*1e7*const.c.value/(self.wave)
        photon_flux_rate = self.flux*(100.0)**2*collecting_area/photon_energy
        self.photon_rate = photon_flux_rate
        
    
    def compute_ElectronRate(self,PCE_model):
        self.pce = PCE_model.get_PCE(self.wave/1e-6)
        self.electron_rate = self.photon_rate*self.pce

class Detector:
    def __init__(self):
        self.det_plane491=numpy.zeros((2048,2048))
        self.det_plane492=numpy.zeros((2048,2048))
        self.x491=numpy.array([-0.03817084,-0.03817084,-0.00132484,-0.00132484])
        self.y491=numpy.array([-0.018423,0.018423,0.018423,-0.018423])
        self.x492=numpy.array([0.03817084,0.03817139,0.00132539,0.00132484])
        self.y492=numpy.array([0.01842235,-0.01842365,-0.0184242,0.0184218])
        self.x_cor_491 = numpy.linspace(numpy.min(self.x491),numpy.max(self.x491),2048)
        self.y_cor_491 = numpy.linspace(numpy.min(self.y491),numpy.max(self.y491),2048)
        self.x_cor_492 = numpy.linspace(numpy.min(self.x492),numpy.max(self.x492),2048)
        self.y_cor_492 = numpy.linspace(numpy.min(self.y492),numpy.max(self.y492),2048)
    
    def add_psf_grid(self,PSF,x_cor,y_cor,factor,oversampling=20):
        dim_out = (PSF.dim[0]//oversampling,PSF.dim[1]//oversampling)
        x_cor_ref = x_cor-(dim_out[1]-1)/2.0*18e-6
        y_cor_ref = y_cor-(dim_out[0]-1)/2.0*18e-6
        print (x_cor_ref,self.x_cor_491[-1],self.x_cor_492[0])
        if x_cor_ref>self.x_cor_492[0]:
            idx_x = numpy.argmin(numpy.fabs(x_cor_ref-self.x_cor_492))
            shift_x = (x_cor_ref-self.x_cor_492[idx_x])/18e-6*oversampling
            idx_y = numpy.argmin(numpy.fabs(y_cor_ref-self.y_cor_492))
            shift_y = (y_cor_ref-self.y_cor_492[idx_y])/18e-6*oversampling
            shifted_PSF = PSF.shift_PSF(shift_x,shift_y)
            binned_psf = shifted_PSF.rebin(binning=oversampling)
            if idx_x+dim_out[1]<2048 and idx_y+dim_out[0]<2048:
                self.det_plane492[idx_y:idx_y+dim_out[0],idx_x:idx_x+dim_out[1]] += binned_psf.psf*factor
        elif x_cor_ref<self.x_cor_491[-1]:
            idx_x = numpy.argmin(numpy.fabs(x_cor_ref-self.x_cor_491))
            shift_x = (x_cor_ref-self.x_cor_491[idx_x])/18e-6*oversampling
            idx_y = numpy.argmin(numpy.fabs(y_cor_ref-self.y_cor_491))
            shift_y = (y_cor_ref-self.y_cor_491[idx_y])/18e-6*oversampling
            print(shift_x,shift_y)
            shifted_PSF = PSF.shift_PSF(shift_x,shift_y)
            binned_psf = shifted_PSF.rebin(binning=oversampling)
            if idx_x+dim_out[1]<2048 and idx_y+dim_out[0]<2048:
                self.det_plane491[idx_y:idx_y+dim_out[0],idx_x:idx_x+dim_out[1]] += binned_psf.psf*factor
        #print numpy.sum(binned_psf)
    
    def addSpectrum(self,spec,PSFcube,x_cor,y_cor,oversampling=20):
        for i in range(len(spec.wave)):
            if spec.electron_rate[i]>0:
               print(spec.electron_rate[i])
               PSF = PSFcube.getPSFwave(spec.wave[i])
               print(spec.wave[i],numpy.sum(PSF.psf))
               self.add_psf_grid(PSF,x_cor[i],y_cor[i],spec.electron_rate[i],oversampling)
               
               
    def saveFITS(self,file_out):
        hdulist=[]
        hdulist.append(pyfits.PrimaryHDU())
        hdulist.append(pyfits.ImageHDU(self.det_plane491,name='DET491'))
        hdulist.append(pyfits.ImageHDU(numpy.rot90(self.det_plane492,2),name='DET492'))
        hdu = pyfits.HDUList(hdulist)
        hdu.writeto(file_out,overwrite=True)
        
        
if __name__=="__main__":
    ##load PSF library
    PSF_lib = NIRSpec_PSFlib2('library_data/G235H_psf_lib.fits')
    ## get the PSF offset in arcsec within shutter
    PSF_source = PSF_lib.get_PSF_shutter(0.034,0.339)
    ## load the library of spectral traces across the detector
    TRACE_LIB = NIRSpec_TRACE_LIB('library_data/MSA_traces_G235H.fits')
    ## get trace for the center of a given shutter
    MSA_TRACE = TRACE_LIB.getTrace(1,197,116)
    print(MSA_TRACE.wave,MSA_TRACE.trace_x)
    ## load the PCE curves for NIRSpec (was not clear if it works)
    PCE_model = NIRSpec_PCE('library_data/NIRSpec_PCE_curves.fits')
    ## Load a specific setup for the PCE curve
    PCE_model.define_PCE('F170LP','G235H')


    #hdu=pyfits.open('/home/husemann/Projects/NIRSpecGTO/NIPS_training/DataPackageTraining/scenes/WIDE_PS-CONT_MOS/JAGUAR_matching_SED.fits')
    #SED = hdu[2].data
    #select = SED<0.0
    #SED[select]=0.0
    #JAGUAR_info = hdu[1].data
    #ID_JAGUAR = JAGUAR_info['ID']
    #z_spec = JAGUAR_info['z']
    #scale_spec = JAGUAR_info['flux_scale']
    #select = ID_JAGUAR == 39629
    #SED_wave = hdu[3].data
    #hdu.close()

    ### define wavelength sampling in microns
    sampling = 0.0003
    spec_wave = numpy.arange(2.4,2.6,sampling)
    #wavelength=SED_wave*1e-10*(1+z_spec[select])
    #print(SED[select,:],z_spec[select],scale_spec[select])
    #spectrum=SED[select ,:][0,:]/(1+z_spec[select])*scale_spec[select]
    #spec_intp = interpolate.interp1d(wavelength*1e6,spectrum)
    #spec_flux = spec_intp(spec_wave)*sampling*1e4
    trace_spec = MSA_TRACE.interpolateTrace(spec_wave*1e-6)
    
    spec_sim = Spectrum(spec_wave*1e-6,100.0)
    spec_sim.convert_to_photon()
    spec_sim.compute_ElectronRate(PCE_model)

    NIRSpec_det = Detector()
    NIRSpec_det.addSpectrum(spec_sim,PSF_source,trace_spec.trace_x,trace_spec.trace_y,oversampling=PSF_source.oversample)
    NIRSpec_det.saveFITS('test_det4.fits')

    ##(wave,PCE) = PCE_model.get_PCE(2.51)
    #plt.plot(spec_sim.wave,spec_sim.electron_rate)
    #plt.show()

    
    ##
    #print(MSA_TRACE.wave,trace_FPA_x,trace_FPA_y)
    #plt.plot(trace_FPA_x,trace_FPA_y)
    #plt.show()
        

    
