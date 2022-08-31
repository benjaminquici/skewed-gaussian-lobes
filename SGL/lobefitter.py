
# standard imports
import numpy as np
import math
import matplotlib
from matplotlib import rc
import matplotlib.pyplot as plt
import matplotlib.path as mpltPath
from astropy.io import fits
from astropy import wcs
from matplotlib.widgets import Slider, Button, TextBox
from astropy.visualization import PercentileInterval
# import Interact

# scipy imports
from scipy.stats import norm, skewnorm
from scipy.optimize import least_squares

# astropy imports
from astropy.visualization import simple_norm

# import cosmology
from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=67.74, Om0=0.3089) # consistent with RAiSEHD

# warnings
import warnings
warnings.filterwarnings('ignore')

# Need to place this inside the Class(es)
rc('font', **{'family':'serif', 'serif':['Times'], 'weight':'medium'})
rc('text', usetex=True)
matplotlib.rcParams['axes.linewidth'] = 0.5

import pandas as pd
import gc


centermarker = {"marker": "o", "linestyle": "None", "ec":"cyan", "fc":"cyan", "s":2}
centermarker_ = {"marker": "o", "linestyle": "None", "ec":"cyan", "fc":"None", "s":50}
hotspotmarker = {"marker": "o", "linestyle": "None", "ec":"green", "fc":"cyan", "s":2}
hotspotmarker_ = {"marker": "o", "linestyle": "None", "ec":"green", "fc":"None", "s":50}

excludemarker = {"marker": "x", "linestyle": "None", "color": "red", "linewidth": 0.75}
excludemarker_ = {"marker": "None", "linestyle": "-", "color": "red", "linewidth": 0.75}
fillmarker = {"marker": "x", "linestyle": "None", "color": "blue", "linewidth": 0.75}
fillmarker_ = {"marker": "None", "linestyle": "-", "color": "blue", "linewidth": 0.75}
lobemarker_0 = {"marker": "x", "linestyle": "None", "color": "green", "linewidth": 0.75}
lobemarker_1 = {"marker": "None", "linestyle": "-", "color": "green", "linewidth": 0.75}


class Coords:
    """
    Simple template to store i and j locations
    """
    def __init__(self,i,j):
        self.i = i
        self.j = j



class SelectCoords:
    """
    Simple template to append i and j locations to list.
    """
    def __init__(self):
        self.i = []
        self.j = []



class Interact:
    """
    Class for interactively extracting pixel location from matplotlib axis
    """
    def __init__(self, ax=None):
        if ax is None:
            self.ax = plt.gca()
        else:
            self.ax = ax
        self.cid = ax.figure.canvas.mpl_connect('key_press_event', self)
        self.center = SelectCoords()
        self.hotspot = SelectCoords()
        self.lobe = SelectCoords()
        self.exclude = SelectCoords()
        self.fill = SelectCoords()
    
    def __call__(self, event):
        if event.inaxes != self.ax.axes: return
        if event.key == 'c':
            self.center.j.append(int(event.xdata))
            self.center.i.append(int(event.ydata))
        if event.key == 'h':
            self.hotspot.j.append(int(event.xdata))
            self.hotspot.i.append(int(event.ydata))
        if event.key == 'e':
            self.exclude.j.append(int(event.xdata))
            self.exclude.i.append(int(event.ydata))
        if event.key == 'f':
            self.fill.j.append(int(event.xdata))
            self.fill.i.append(int(event.ydata))
        if event.key == 'b':
            self.lobe.j.append(int(event.xdata))
            self.lobe.i.append(int(event.ydata))

        # This draws points onto the canvas      
        self.ax.scatter(self.center.j, self.center.i, **centermarker)
        self.ax.scatter(self.center.j, self.center.i, **centermarker_)
        self.ax.scatter(self.hotspot.j, self.hotspot.i, **hotspotmarker)
        self.ax.scatter(self.hotspot.j, self.hotspot.i, **hotspotmarker_)
        
        if len(self.lobe.j)==1:
            line, = self.ax.plot(self.lobe.j, self.lobe.i, **lobemarker_0)
        else:
            line, = self.ax.plot(self.lobe.j, self.lobe.i, **lobemarker_1)
        
        if len(self.exclude.j)==1:
            line, = self.ax.plot(self.exclude.j, self.exclude.i, **excludemarker)
        else:
            line, = self.ax.plot(self.exclude.j, self.exclude.i, **excludemarker_)
        
        if len(self.fill.j)==1:
            line, = self.ax.plot(self.fill.j, self.fill.i, **fillmarker)
        else:
            line, = self.ax.plot(self.fill.j, self.fill.i, **fillmarker_)

        # This makes it plot without needing to change focus back to the terminal
        self.ax.figure.canvas.draw()



def showimdata(imdata):
    fig = plt.figure()
    ax = fig.add_axes([0.05,0.05,0.9,0.9])
    ax.imshow(imdata)
    plt.show()



def peakpixel(imdata):
    """
    Find the peak pixel and return its value and position
    """

    value = np.nanmax(imdata)
    i,j = np.where(imdata==value)
    position = Coords(i[0], j[0])

    return value, position


def importdf(filename, log=False, normalise=True):
    """
    Extract image data, pixel scales and wcs projection from input fits image
    """
    
    dataframe = pd.read_csv(filename, index_col=0)
    x = (dataframe.index).astype(np.float_)
    imdata = dataframe.to_numpy()

    imdata = imdata.T
    
    cdelt = [np.abs(x[1]-x[0])*(1/3600.), np.abs(x[1]-x[0])*(1/3600.)]

    mask = np.ones_like(imdata)
    mask[np.where(imdata<=1e-4)]=0
    index = mask > 0
    
    if log == True:
        # imdata = imdata**(1./factor)
        imdata = np.log10(imdata)
        if normalise is True:
            imdatarange = np.abs(np.nanmax(imdata[index])-np.nanmin(imdata[index]))
            imdata = np.interp(imdata, (np.nanmin(imdata[index]), np.nanmax(imdata[index])), (0, imdatarange))

    else:
        imdata = np.sqrt(imdata)

    
    gc.collect()
    # showimdata(imdata)

    return imdata, mask, cdelt



def importfits(fitsimage, log=False, normalise=True):
    """
    Extract image data, pixel scales and wcs projection from input fits image
    """
    with fits.open(fitsimage) as hdu:
    # hdu = fits.open(fitsimage)
    
        imdata = np.squeeze(hdu[0].data)

        mask = np.ones_like(imdata)
        mask[np.where(imdata<=1e-5)]=0
        index = mask > 0

        # stretch = AsinhStretch()
        # showimdata(stretch(imdata))
        
        if log == True:
            imdata = np.log10(imdata)
            if normalise is True:
                imdatarange = np.abs(np.nanmax(imdata[index])-np.nanmin(imdata[index]))
                imdata = np.interp(imdata, (np.nanmin(imdata[index]), np.nanmax(imdata[index])), (0, imdatarange))
            # imdata[np.where(np.isnan(imdata))]=-99
        
        imheader = hdu[0].header
        
        cdelt1, cdelt2 = np.abs(imheader['CDELT1']), np.abs(imheader['CDELT2'])
        cdelt = [cdelt1, cdelt2]
        
        naxis1, naxis2 = imheader['NAXIS1'], imheader['NAXIS2']
        naxis = [naxis2, naxis1]
        
        w=wcs.WCS(imheader, naxis=2)

        imdata = np.sqrt(imdata)

        hdu.close()
    
    return imdata, mask, cdelt, naxis, w



"""
Functions for estimating and fitting properties of observed radio lobes
"""


def SkewGauss2D(params, image_data, beam_params, image_x, image_y, image_mask=None):
    """
    Skewed 2D Gaussian model 
    """

    # if no mask supplied, fit all image pixels
    if image_mask is None:
        image_mask = np.ones_like(image_x)

    # export gaussian parameters
    amplitude, rotation, mu_x, mu_y, sigma_pc1, sigma_pc2, skew_pc1 = params
    
    # transform to principle component axes at each pixel (rotation centred on centre of ellipse)
    image_pc1 = np.cos(rotation)*(image_x - mu_x)*beam_params[0] - np.sin(rotation)*(image_y - mu_y)*beam_params[1]
    image_pc2 = np.sin(rotation)*(image_x - mu_x)*beam_params[0] + np.cos(rotation)*(image_y - mu_y)*beam_params[1]
        
    # calculate residual for each pixel in the lobe section of the image
    z_pred = np.zeros_like(image_data)
    index = image_mask > 0
    # z_pred[index] = 2*np.pi*amplitude*skewnorm.pdf(image_pc1[index]/(1e-7*sigma_pc1), a=skew_pc1)*norm.pdf(image_pc2[index]/(1e-7*sigma_pc2))
    z_pred[index] = 2*np.pi*amplitude*skewnorm.pdf(image_pc1[index]/(sigma_pc1), a=skew_pc1)*norm.pdf(image_pc2[index]/(sigma_pc2))

    return z_pred


def SkewGauss2D_residuals(params, image_data, beam_params, image_x, image_y, image_mask):
    """
    Calculate the residual between the lobe and a 2D Gaussian model
    """

    # calculate residual for each pixel in the lobe section of the image
    z_pred = SkewGauss2D(params, image_data, beam_params, image_x, image_y, image_mask=image_mask)

    # return the weighted residual vector
    return (image_data - z_pred)[image_mask > 0].flatten()
    
    # sigx, sigy = params[4], params[5]
    # if sigx < sigy:
    #     return 1e+2
    # else:

    #     # calculate residual for each pixel in the lobe section of the image
    #     z_pred = SkewGauss2D(params, image_data, beam_params, image_x, image_y, image_mask=image_mask)
    
    #     # return the weighted residual vector
    #     return (image_data - z_pred)[image_mask > 0].flatten()


def OptimizeSkewGauss2D(image_data, beam_params, center, hotspot, stretch, image_mask, padding=1, w=None):
    """
    Fit a skewed 2D Gaussian to observed radio lobes
    """

    # apply stretch
    if stretch == 'linear':
        pass
    if stretch == 'log':
        image_data = np.log10(image_data)
        # print(image_data[np.where(image_mask>0)])
    if stretch == 'sqrt':
        image_data = np.sqrt(image_data)

    # work out size of image data in each direction
    image_height, image_width = np.shape(image_data)

    # find angle of jet axis from 'core'
    print('Hotspot coords: x = {}, y = {}'.format(hotspot.j, hotspot.i))
    print('Core coords: x = {}, y = {}'.format(center.j, center.i))
    print('x displacement = {}'.format((center.j - hotspot.j)*beam_params[0]))
    print('y displacement = {}'.format((hotspot.i - center.i)*beam_params[1]))
    # rotation = np.tan(((-hotspot.i + center.i)*beam_params[1])/((hotspot.j - center.j)*beam_params[0]))
    rotation = math.atan(((hotspot.i - center.i)*beam_params[1])/((center.j - hotspot.j)*beam_params[0]))

    # create mask to flag pixels on other side of 'core' to 'hotspot'
    image_x, image_y = np.meshgrid(range(0, len(image_data[0, :])), range(0, len(image_data[:, 0])))

    # define values for parameter initial guesses and boundaries
    if stretch == 'log':
        amp, amp_min, amp_max = np.nanmax(image_mask*image_data), -20, 2
    else:
        amp, amp_min, amp_max = np.nanmax(image_mask*image_data), 0, np.inf
    rot,   rot_min,   rot_max = rotation, rotation-np.pi/18., rotation+np.pi/18. #rotation - np.pi/18, rotation + np.pi/18
    # print(rot*180/np.pi)ss
    mux,   mux_min,   mux_max   = hotspot.j, 0, 2*image_width
    muy,   muy_min,   muy_max   = hotspot.i, 0, 2*image_height
    sigx,  sigx_min,  sigx_max  = np.sqrt(beam_params[0]**2 + beam_params[1]**2), 0, np.inf
    sigy,  sigy_min,  sigy_max  = np.sqrt(beam_params[0]**2 + beam_params[1]**2), 0, np.inf
    skewx, skewx_min, skewx_max = -10, -1000, 0

    # define initial guess for Gaussian fit parameters
    x0 = [amp, rot, mux, muy, sigx, sigy, skewx]

    # define parameter boundaries
    bounds = [[amp_min, rot_min, mux_min, muy_min, sigx_min, sigy_min, skewx_min], [amp_max, rot_max, mux_max, muy_max, sigx_max, sigy_max, skewx_max]]

    # parse argument list
    args = (image_data, beam_params, image_x, image_y, image_mask)

    # do the least squares fitting
    result = least_squares(SkewGauss2D_residuals, args=args, x0=x0, bounds=bounds, method='dogbox', verbose=1)
    result = result.x
    ampfit, rotfit, muxfit, muyfit, sig1fit, sig2fit, skewfit = result
    for i in range(len(x0)):
        bmin, bmax = bounds[0][i], bounds[1][i]
        print('Bounds: {:.6g} < {:.6g} < {:.6g}. Fitted: {:.6g}'.format(bmin, x0[i], bmax, result[i]))
        if i == 1:
            print(result[i]*180/np.pi)

    # generate large meshgrid onto which to draw predicted image
    padding = int(padding)
    image_x, image_y = np.meshgrid(range(0, int(padding*len(image_data[0, :]))), range(0, int(padding*len(image_data[:, 0]))))
    image_x = image_x-int(0.5*(padding-1)*image_width)
    image_y = image_y-int(0.5*(padding-1)*image_height)

    # draw image on new grid
    image_pred = SkewGauss2D(result, image_data, beam_params, image_x, image_y, image_mask=None)

    if stretch == 'linear':
        pass
    if stretch == 'log':
        image_pred = 10**image_pred
    if stretch == 'sqrt':
        image_pred = image_pred**2

    # fig0 = plt.figure()
    # ax0 = fig0.add_axes([0.05,0.05,0.9,0.9], projection=w)
    # ax0.set_title('Image pred')
    # ax0.imshow(image_pred)
    # plt.show()

    return image_pred

    # get the location of the peak pixel in predicted image
    # hotspot_value, hotspot_loc = peakpixel(image_pred**2)

    # # get 1,2 and 3sigma values of peak
    # levels = [hotspot_value*np.exp(-4.5), hotspot_value*np.exp(-2), hotspot_value*np.exp(-0.5)]

    # twosig = hotspot_value*np.exp(-2)

    # fig1 = plt.figure()
    # ax = fig1.add_axes([0.05,0.05,0.9,0.9], projection=w)
    # ax.imshow(image_data**2)
    # ax.contour((image_pred**2)*image_mask, levels=levels, colors='red')
    # ax.scatter(hotspot_loc.j, hotspot_loc.i, s=10, marker='.', color='blue')
    # ax.scatter(center.j, center.i, marker='.',s=10,color='blue')
    # interact = Interact(ax)
    # plt.show()

    # # work out the angular diameter distance
    # # d_A = cosmo.angular_diameter_distance(z=0.2133)
    # d_A = cosmo.kpc_proper_per_arcmin(0.2133)

    # # export relevant coords
    # edge = Coords(interact.hotspot.i[0], interact.hotspot.j[0])
    # backflow = Coords(interact.hotspot.i[1], interact.hotspot.j[1])
    # width = Coords(interact.hotspot.i[2], interact.hotspot.j[2])
    # width1 = Coords(interact.hotspot.i[3], interact.hotspot.j[3])

    # # calculcate physical size
    # size_pix = np.sqrt(((center.i-edge.i)**2) + ((center.j-edge.j)**2))
    # print('Jet length: {} kpc'.format(size_pix*beam_params[0]*60*d_A.value))

    # # calculcate axis ratio
    # width_pix = np.sqrt(((hotspot_loc.i-width.i)**2) + ((hotspot_loc.j-width.j)**2))
    # print('Axis ratio: {}'.format(size_pix/width_pix))

    # # calculate extent 
    # extent_pix = np.sqrt(((edge.i-backflow.i)**2) + ((edge.j-backflow.j)**2))
    # print('Extent: {}'.format(extent_pix/size_pix))

    # # calculate width 
    # width_tot_pix = np.sqrt(((width.i-width1.i)**2) + ((width.j-width1.j)**2))
    # print('Width tot: {}'.format(width_tot_pix*beam_params[0]*d_A.value*1e+3))

    # # frac peak len
    # hotspot_pix = np.sqrt(((center.i-hotspot_loc.i)**2) + ((center.j-hotspot_loc.j)**2))
    # print('Frac peak len: {}'.format(hotspot_pix/size_pix))

    # print(int(center.i), int(center.j))
    # centerpix = (image_pred**2)[int(center.i):int(center.i)+1,int(center.j):int(center.j)+1]
    # print(centerpix/hotspot_value)

    # # reproject the coordinates of the RG center
    # center.i = center.i * padding
    # center.j = center.j * padding

    # return result, 10**image_pred, center


"""
Functions for estimating and fitting properties of simulated (RAiSE) lobes.
"""

def SkewGauss2DFixed(params, image_data, beam_params, image_x, image_y, mu_y, angle=0, image_mask=None):
    """
    Skewed 2D Gaussian model with a fixed y-axis coordinate
    """

    # if no mask supplied, fit all image pixels
    if image_mask is None:
        image_mask = np.ones_like(image_x)

    # export gaussian parameters
    amplitude, mu_x, sigma_pc1, sigma_pc2, skew_pc1 = params

    # transform to principle component axes at each pixel (rotation centred on centre of ellipse)
    image_pc1 = np.cos(angle)*(image_x - mu_x)*beam_params[0] - np.sin(angle)*(image_y - mu_y)*beam_params[1]
    image_pc2 = np.sin(angle)*(image_x - mu_x)*beam_params[0] + np.cos(angle)*(image_y - mu_y)*beam_params[1]

    # evaluate gaussian at coordinate location
    z_pred = np.zeros_like(image_x.astype(float))
    index = image_mask > 0
    z_pred[index] = 2*np.pi*amplitude*skewnorm.pdf(image_pc1[index]/(sigma_pc1), a=skew_pc1)*norm.pdf(image_pc2[index]/(sigma_pc2))
    
    return z_pred



def Gauss2DFixed(params, image_data, beam_params, image_x, image_y, mu_x, mu_y, angle=0, image_mask=None):
    """
    Skewed 2D Gaussian model with a fixed y-axis coordinate
    """

    # if no mask supplied, fit all image pixels
    if image_mask is None:
        image_mask = np.ones_like(image_x)

    # export gaussian parameters
    amplitude, sigma_pc1, sigma_pc2 = params

    # transform to principle component axes at each pixel (rotation centred on centre of ellipse)
    image_pc1 = np.cos(angle)*(image_x - mu_x)*beam_params[0] - np.sin(angle)*(image_y - mu_y)*beam_params[1]
    image_pc2 = np.sin(angle)*(image_x - mu_x)*beam_params[0] + np.cos(angle)*(image_y - mu_y)*beam_params[1]

    # evaluate gaussian at coordinate location
    z_pred = np.zeros_like(image_x.astype(float))
    index = image_mask > 0
    z_pred[index] = 2*np.pi*amplitude*norm.pdf(image_pc1[index]/(1e-7*sigma_pc1))*norm.pdf(image_pc2[index]/(1e-7*sigma_pc2))
    
    return z_pred



def SkewGauss2DFixed_residuals(params, image_data, beam_params, image_x, image_y, mu_y, image_mask, subtraction='linea'):
    """
    Calculate the residual between the lobe and a 2D Gaussian model
    """

    # evaluate 2D Gaussian model for input set of parameters
    z_pred = SkewGauss2DFixed(params, image_data, beam_params, image_x, image_y, mu_y, image_mask=image_mask)

    # perform subtraction in appropriate parameter space
    if subtraction == 'linear':
        # note, this assumes the original data is logged. #TODO generalize 
        z_pred = (10**z_pred)**0.5
        image_data = (10**image_data)**0.5

    return (image_data - z_pred)[image_mask > 0].flatten()



def Gauss2DFixed_residuals(params, image_data, beam_params, image_x, image_y, mu_x, mu_y, image_mask, subtraction='linar'):
    """
    Calculate the residual between the lobe and a 2D Gaussian model
    """

    # evaluate 2D Gaussian model for input set of parameters
    z_pred = Gauss2DFixed(params, image_data, beam_params, image_x, image_y, mu_x, mu_y, image_mask=image_mask)

    # perform subtraction in appropriate parameter space
    if subtraction == 'linear':
        # note, this assumes the original data is logged. #TODO generalize 
        z_pred = (10**z_pred)**factor
        image_data = (10**image_data)**factor

    return (image_data - z_pred)[image_mask > 0].flatten()



def Gauss2DFixedRegression(image_data, beam_params, image_mask, padding=1.1):
    """
    Optimise fit for a 2D Gaussian with a fixed center (center of image)
    """

    if image_mask is None:
        image_mask = np.ones_like(image_data)

    # work out size of image data in each direction
    image_height, image_width = np.shape(image_data)

    # instantiate arrays
    image_x, image_y = np.meshgrid(range(0, len(image_data[0, :])), range(0, len(image_data[:, 0])))

    # determine hotspot position
    hotspot_val, hotspot_pos = peakpixel(image_data*image_mask)

    # determine radio galaxy center
    center = Coords(0.5*image_height, 0.5*image_width)

    # set up initial guesses and boundaries for parameters
    amp, amp_min, amp_max = np.nanmax(image_mask*image_data), 0, np.inf
    sigx,  sigx_min,  sigx_max  = np.sqrt(beam_params[0]**2 + beam_params[1]**2)*1e7, 0, np.inf
    sigy,  sigy_min,  sigy_max  = np.sqrt(beam_params[0]**2 + beam_params[1]**2)*1e7, 0, np.inf

    # define initial guess for Gaussian fit parameters
    x0 = [amp, sigx, sigy]

    # define parameter boundaries
    bounds = [[amp_min, sigx_min, sigy_min], [amp_max, sigx_max, sigy_max]]

    # parse argument list
    args = (image_data, beam_params, image_x, image_y, center.j, center.i, image_mask)

    # do the least squares fitting
    result = least_squares(Gauss2DFixed_residuals, args=args, x0=x0, bounds=bounds)
    result = result.x
    ampfit, sig1fit, sig2fit = result

    # generate large meshgrid onto which to draw predicted image
    padding = int(padding)
    image_x, image_y = np.meshgrid(range(0, int(padding*len(image_data[0, :]))), range(0, int(padding*len(image_data[:, 0]))))
    image_x = image_x-int(0.5*(padding-1)*image_width)
    image_y = image_y-int(0.5*(padding-1)*image_height)

    # image data template
    template = np.zeros_like(image_x.astype(float))
    template[int(0.5*(padding-1)*image_height):(int(0.5*(padding+1)*image_height)), int(0.5*(padding-1)*image_width):(int(0.5*(padding+1)*image_width))] = image_data

    # draw image on new grid
    image_pred = Gauss2DFixed(result, image_data, beam_params, image_x, image_y, center.j, center.i, image_mask=None)

    # reproject the coordinates of the RG center
    center.i = center.i * padding
    center.j = center.j * padding
    gc.collect()

    return result, image_pred**2, center, template



def SkewGauss2DFixedRegression(image_data, beam_params, image_mask, padding=1):
    """
    Fit a skewed 2D Gaussian to simulated (RAiSE) radio lobes
    """
    beam_params = [1, 1]
    # work out size of image data in each direction
    image_height, image_width = np.shape(image_data)

    # instantiate arrays
    image_x, image_y = np.meshgrid(range(0, len(image_data[0, :])), range(0, len(image_data[:, 0])))

    # determine radio galaxy center
    center = Coords(0.5*image_height, 0.5*image_width)

    # mask pixels to left of center
    image_mask[np.where(image_x<center.j)]=0

    # determine hotspot position
    hotspot_val, hotspot_pos = peakpixel(image_data*image_mask)

    # set up initial guesses and boundaries for parameters
    amp, amp_min, amp_max = np.nanmax(image_mask*image_data), 0, np.inf
    mux,   mux_min,   mux_max   = 0.98*image_width, 0.9*image_width, 2*image_width
    muy,   muy_min,   muy_max   = center.i, 0, 2*image_height
    sigx,  sigx_min,  sigx_max  = 1*np.sqrt(beam_params[0]**2 + beam_params[1]**2), 0, image_width
    sigy,  sigy_min,  sigy_max  = 1*np.sqrt(beam_params[0]**2 + beam_params[1]**2), 0, image_height
    skewx, skewx_min, skewx_max = -0, -np.inf, 0

    # define initial guess for Gaussian fit parameters
    x0 = [amp, mux, sigx, sigy, skewx]

    # define parameter boundaries
    bounds = [[amp_min, mux_min, sigx_min, sigy_min, skewx_min], [amp_max, mux_max, sigx_max, sigy_max, skewx_max]]

    # parse argument list
    args = (image_data, beam_params, image_x, image_y, center.i, image_mask)

    # do the least squares fitting
    result = least_squares(SkewGauss2DFixed_residuals, args=args, x0=x0, bounds=bounds)
    result = result.x
    ampfit, muxfit, sig1fit, sig2fit, skewfit = result

    # generate large meshgrid onto which to draw predicted image
    # padding = int(padding)
    image_x, image_y = np.meshgrid(range(0, padding+int(len(image_data[0, :]))), range(0, padding+int(len(image_data[:, 0]))))
    image_x = image_x - 0.5*padding
    image_y = image_y - 0.5*padding

    # image data template
    template = np.zeros_like(image_x.astype(float))
    template[int(padding*0.5):int(padding*0.5)+len(image_data[:, 0]), int(padding*0.5):int(padding*0.5)+len(image_data[0, :])] = image_data

    # draw image on new grid
    image_pred = SkewGauss2DFixed(result, image_data, beam_params, image_x, image_y, center.i, image_mask=None)

    # reproject the coordinates of the RG center
    center.i = center.i + padding*0.5
    center.j = center.j + padding*0.5

    gc.collect()
    return result, image_pred**2, center, template



def MeasureLobeShape(image_pred, image_data, center, cdelt, z, sigma=2, name=None, dispname=None):
    """
    Analyze 2D Gaussian to measure the physical size, axis ratio, and extent.
    """

    # sigma = 1
    # work out the angular diameter distance
    # d_A = cosmo.angular_diameter_distance(z=z)
    d_A = cosmo.kpc_proper_per_arcmin(z)
    d_A = d_A.value*60

    # work out the sigma level
    factor = np.exp(-0.5*(sigma**2))

    # find peak in predicted image
    hotspot_val, hotspot_pos = peakpixel(image_pred)

    # work out the 1sigma, 2sigma and 3sigma values of the peak
    sigmaval = hotspot_val*factor

    # select jet-axis row, and everything to the West of the center
    jetax = image_pred[int(center.i):int(center.i)+1, int(center.j):]

    # measure the width of the lobe brighter than sigma level pixel units)
    lobewidth = len(jetax[np.where(jetax>=sigmaval)])

    # measure distance from center to hotspot (pixel units)
    jetax_ch_pix = int(np.abs(int(hotspot_pos.j) - int(center.j)))

    # select the pixel between the hotspot and the lobe edge
    jetax_he = image_pred[int(center.i):int(center.i)+1, int(hotspot_pos.j):]

    # measure the hotspot to lobe-edge distance (pixel units)
    jetax_he_pix = len(jetax_he[np.where(jetax_he>=sigmaval)])

    # measure the length of the jet (pixel units)
    jetlen_pix = jetax_ch_pix + jetax_he_pix

    if center.j > hotspot_pos.j:
        jetlen_pix = lobewidth

    # get position halfway along the lobe
    halfway_jet_pix = 0.5*jetlen_pix

    # convert jet length to kpc units
    jetlen_kpc = jetlen_pix*cdelt[0]*d_A

    # select column containing peak pixel, and everything below the jet axis
    peakcolumn = image_pred[int(center.i):, int(hotspot_pos.j):int(hotspot_pos.j)+1]

    # measure the North-South distance from the jet axis to the lobe edge
    lobeheight_pix = len(peakcolumn[np.where(peakcolumn>=sigmaval)])

    # measure axis ratio
    axisratio = jetlen_pix/lobeheight_pix

    # measure extent
    extent = lobewidth/jetlen_pix

    # measure extent sigma at the core
    center_val = float(image_pred[int(center.i):int(center.i+1), int(center.j):int(center.j+1)])
    factor = center_val/hotspot_val
    sigma = np.sqrt(-2*np.log(factor))
    print(sigma)
    # factor = np.exp(-0.5*(sigma**2))
    # sigmaval_ = hotspot_val*factor

    if name is not None:
        r = len(image_pred[0, :]) / len(image_pred[:, 0])
        
        fig = plt.figure(figsize=(7*r, 14))
        ax0 = fig.add_axes([0.05, 0.025, 0.9, 0.45])
        ax1 = fig.add_axes([0.05, 0.5, 0.9, 0.45])
        ax1.imshow(image_data**2, cmap='inferno', origin='lower')
        ax1.contour(image_pred, levels=[sigmaval], colors='white', linewidths=0.5)
        ax1.contour(image_pred, levels=[center_val], colors='white', linewidths=0.5)
        norm = simple_norm(image_pred, percent=99.9)
        
        ax0.imshow(image_pred, cmap='inferno', origin='lower', norm=norm)
        ax0.contour(image_pred, levels=[sigmaval], colors='white', linewidths=0.5)
        ax0.contour(image_pred, levels=[center_val], colors='white', linewidths=0.5)
        
        ax0.scatter(center.j, center.i, s=300, marker='o', ec='magenta', fc='none', lw=2)
        ax0.scatter(center.j, center.i, s=4, marker='o', ec='magenta', lw=2)

        ax0.scatter(center.j+jetlen_pix, center.i, s=200, marker='x', ec='cyan')
        ax0.scatter(hotspot_pos.j, center.i+lobeheight_pix, s=200, marker='x', ec='cyan')

        ax0.scatter(hotspot_pos.j, hotspot_pos.i, s=200, marker='s', ec='blue')
        ax0.scatter(center.j+halfway_jet_pix, center.i, s=200, marker='s', ec='green', fc='green')
        if dispname is not None:
            ax1.text(0.05*len(image_pred[:, 0]), 0.9*len(image_pred[:, 0]), dispname, color='white')
        
        plt.savefig('{}.pdf'.format(name), dpi=100)
        # Clear the current axes.
        plt.cla() 
        # Clear the current figure.
        plt.clf() 
        # Closes all the figure windows.
        plt.close('all')   
        plt.close(fig)
        gc.collect()

    return(jetlen_kpc*2, axisratio, extent)



def RAiSELobeType(imdata):
    peakpix, peakpixvloc = peakpixel(imdata)
    if peakpixvloc.j/len(imdata[0,:]) < 0.49 or peakpixvloc.j/len(imdata[0,:]) > 0.51:
    # if peakpixvloc.j/len(imdata[0,:]) == 0.5:
        fittype = 'skew'
    else:
        fittype = 'standard'
    # print(fittype)
    return fittype

def FitRAiSELobe(filename, z, sigma, name, dispname=None):

    imdata, mask, cdelt = importdf(filename)

    if np.all(imdata == 0):
        size, axisratio, extent = np.inf, np.inf, np.inf
        message = 'No emission present in image. Returning infs for all lobe properties.'
    else:
        result, image_pred, center, image_data = SkewGauss2DFixedRegression(imdata, cdelt, image_mask=mask)
        size, axisratio, extent = MeasureLobeShape(image_pred, image_data, center, cdelt, z, sigma=sigma, name=name, dispname=dispname)
        message = 'Fitting of RAiSE lobes concluded.'
    
    gc.collect()
    return size, axisratio, extent, np.abs(skewfit)

def create_index_array(hdu):
    # Initialise an array of co-ordinate pairs for the full array
    imdata = np.squeeze(hdu[0].data)
    indexes = np.empty(((imdata.shape[0])*(imdata.shape[1]),2),dtype=int)
    idx = np.array([ (j,0) for j in range(imdata.shape[1])])
    j=imdata.shape[1]
    for i in range(imdata.shape[0]):
        idx[:,1]=i
        indexes[i*j:(i+1)*j] = idx
    return indexes


def polyplot(fitsimage):
    """
    Interactively adjust a hexagonal grid onto the radio image.
    """

    # extract hdu properties
    hdu = fits.open(fitsimage)
    imdata = np.sqrt(np.squeeze(hdu[0].data))
    header = hdu[0].header
    w = wcs.WCS(header, naxis=2)

    # create figure
    fig = plt.figure(figsize=(10.5,7))
    ax = fig.add_axes([0.22, 0.05, 0.9, 0.9])#, projection=w)
    ticklabelcolor='black'
    display_minor_ticks=True
    minor_tick_frequency=6
    hide_labels=False, 
    maj_tick_length=2.2
    maj_tick_width=0.75
    min_tick_length=1.4
    xlabel='R. A. (J2000)'
    xlabelfontsize=8
    xfontweight='bold'
    xformatter='hh:mm:ss'
    xticklabelfontsize=8
    xminpad=1
    ylabel='Decl. (J2000)'
    ylabelfontsize=8
    yfontweight='bold'
    yformatter='dd:mm:ss'
    yticklabelfontsize=8
    yminpad=-1

    # ax.tick_params(axis='both', which='major', direction='in', length=maj_tick_length, width=maj_tick_width)
    # ax.tick_params(which='minor', length=min_tick_length)
    # # format Right Ascension axis
    # lon = ax.coords['ra']
    # lon.set_minor_frequency(minor_tick_frequency)
    # lon.set_axislabel(xlabel, minpad=xminpad, fontsize=xlabelfontsize, fontweight=xfontweight)
    # lon.set_major_formatter(xformatter)
    # lon.set_ticklabel(size=xticklabelfontsize, color=ticklabelcolor)
    # lon.display_minor_ticks(display_minor_ticks)
    
    # # format Declination axis
    # lat = ax.coords['dec']
    # lat.set_minor_frequency(minor_tick_frequency)
    # lat.set_axislabel(ylabel, minpad=yminpad, fontsize=ylabelfontsize, fontweight=yfontweight)
    # lat.set_major_formatter(yformatter)
    # lat.set_ticklabel(size=yticklabelfontsize, color=ticklabelcolor)
    # lat.display_minor_ticks(display_minor_ticks)

    im = ax.imshow(imdata, cmap='inferno')

    # setup normalization
    pct = 99.5
    interval = PercentileInterval(pct)
    vmin, vmax = interval.get_limits(imdata)

    # setup interactive sliders
    axmin = fig.add_axes([0.08, 0.9, 0.15, 0.02])
    axmax  = fig.add_axes([0.08, 0.8, 0.15, 0.02])
    axtheta  = fig.add_axes([0.08, 0.25, 0.15, 0.02])
    
    svmin = Slider(axmin, "vmin \n (Jy/beam)", 0, vmax, valinit=vmin)
    svmax = Slider(axmax, "vmax \n (Jy/beam)", 0, vmax, valinit=vmax)
    svtheta = Slider(axtheta, "theta \n (deg)", 0, 60, valinit=0)

    # send interactive commands to redraw plot
    def update(val=None):
        ax.clear()
        ax.imshow(imdata, cmap='inferno',vmin=svmin.val, vmax=svmax.val)
        fig.canvas.draw_idle()
    
    # update values
    svmin.on_changed(update)
    svmax.on_changed(update)
    svtheta.on_changed(update)

    # export the polypick-selected sky coords
    interact = Interact(ax)
    plt.show()

    center = Coords(interact.center.i[0], interact.center.j[0])
    hotspots.i, hotspots.j = interact.hotspot.i, interact.hotspot.j
    hotspot = Coords(hotspots.i[0], hotspots.j[0])

    path = mpltPath.Path(list(zip(lobeedge.i,lobeedge.j)))
    indexes = create_index_array(hdu)
    inside = path.contains_points(indexes)
    imdata_new = np.zeros_like(imdata)
    for ix in indexes[np.where(inside)]:
        if np.isnan(imdata[ix[0],ix[1]]) or imdata[ix[0],ix[1]]<0:
            imdata_new[ix[0],ix[1]] = 0
        else:
            imdata_new[ix[0],ix[1]] = imdata[ix[0],ix[1]]

    lobeedge = SelectCoords()
    lobeedge.i, lobeedge.j = interact.lobe.i, interact.lobe.j

    return center, hotspot, lobeedge

def format(ax, ticklabelcolor='black', display_minor_ticks=True, minor_tick_frequency=6, hide_labels=False, 
        maj_tick_length=2.2, maj_tick_width=0.75, min_tick_length=1.4, \
        xlabel='R. A. (J2000)', xlabelfontsize=8, xfontweight='bold', xformatter='hh:mm:ss', xticklabelfontsize=8, xminpad=1, \
        ylabel='Decl. (J2000)', ylabelfontsize=8, yfontweight='bold', yformatter='dd:mm:ss', yticklabelfontsize=8, yminpad=-1):

        # format major and minor ticks
        ax.tick_params(axis='both', which='major', direction='in', length=maj_tick_length, width=maj_tick_width)
        ax.tick_params(which='minor', length=min_tick_length)

        # # format Right Ascension axis
        # lon = ax.coords['ra']
        # lon.set_minor_frequency(minor_tick_frequency)
        # lon.set_axislabel(xlabel, minpad=xminpad, fontsize=xlabelfontsize, fontweight=xfontweight)
        # lon.set_major_formatter(xformatter)
        # lon.set_ticklabel(size=xticklabelfontsize, color=ticklabelcolor)
        # lon.display_minor_ticks(display_minor_ticks)
        
        # # format Declination axis
        # lat = ax.coords['dec']
        # lat.set_minor_frequency(minor_tick_frequency)
        # lat.set_axislabel(ylabel, minpad=yminpad, fontsize=ylabelfontsize, fontweight=yfontweight)
        # lat.set_major_formatter(yformatter)
        # lat.set_ticklabel(size=yticklabelfontsize, color=ticklabelcolor)
        # lat.display_minor_ticks(display_minor_ticks)
        return ax
        
def host_from_image(hostimage, radioimage):
    hdu_host = fits.open(hostimage)
    image_host = np.squeeze(hdu_host[0].data)
    w_host = wcs.WCS(hdu_host[0].header, naxis=2)

    hdu_radio = fits.open(radioimage)
    image_radio = np.squeeze(hdu_radio[0].data)
    w_radio = wcs.WCS(hdu_radio[0].header, naxis=2)

    min_,max_ = np.abs(np.nanmin(image_radio)), np.nanmax(image_radio)
    levels=np.geomspace(min_, max_, 7)
    # create figure
    fig = plt.figure(figsize=(7,7))
    ax = fig.add_axes([0.1, 0.1, 0.85, 0.85])#, projection=w_host)
    ax.set_title("Select SMBH centre using 'c'")
    norm = simple_norm(image_host, percent=99.5)
    ax.imshow(image_host, cmap='gray_r', norm=norm)
    ax.contour(image_radio, colors='black', linewidths=0.5, levels=levels)
    format(ax)
    interact = Interact(ax)
    plt.show()

    center = Coords(None, None)
    if len(interact.center.i):
        ra, dec = w_host.wcs_pix2world(interact.center.i[0], interact.center.j[0],0)
        xpix, ypix = w_radio.wcs_world2pix(ra, dec, 0)
        center = Coords(xpix, ypix)
    return center

def select_radio_source(radioimage, center):
    hdu = fits.open(radioimage)
    imdata = np.squeeze(hdu[0].data)
    height, width = np.shape(imdata)
    header = hdu[0].header
    # cdelt = 3600*np.abs(header['CDELT1'])
    cdelt = 1
    w = wcs.WCS(header, naxis=2)
    # create figure
    fig = plt.figure(figsize=(7,7*(height/width)))
    ax = fig.add_axes([0.075, 0.1, 0.85, 0.85])#, projection=w)
    ax.set_title("Select: hotspots using 'h'; radio centre using 'c'.")
    ax.axis('off')
    norm = simple_norm(imdata, percent=95, stretch='log')
    im = ax.imshow(imdata, cmap='inferno', norm=norm)
    if center.i is not None:
        y,x = center.i, center.j
        ax.scatter(x,y, marker='o', s=2, color='cyan')
        ax.scatter(x,y, marker='o', s=50, ec='cyan', fc='None', color='cyan')

    # setup normalization
    pct = 100
    interval = PercentileInterval(pct)
    vmin, vmax = interval.get_limits(imdata)

    # setup interactive sliders
    axmin = fig.add_axes([0.15, 0.05, 0.15, 0.02])
    axmax  = fig.add_axes([0.6, 0.05, 0.15, 0.02])
    # axtheta  = fig.add_axes([0.08, 0.25, 0.15, 0.02])
    
    svmin = Slider(axmin, "vmin \n (Jy/beam)", 0, vmax, valinit=vmin)
    svmax = Slider(axmax, "vmax \n (Jy/beam)", 0, 5*vmax, valinit=vmax)
    # svtheta = Slider(axtheta, "theta \n (deg)", 0, 60, valinit=0)

    # send interactive commands to redraw plot
    def update(val=None):
        ax.clear()
        if center.i is not None:
            y,x = center.i, center.j
            ax.scatter(x,y, marker='o', s=2, color='cyan')
            ax.scatter(x,y, marker='o', s=50, ec='cyan', fc='None', color='cyan')
        ax.imshow(imdata, cmap='inferno',vmin=svmin.val, vmax=svmax.val, norm=norm)
        fig.canvas.draw_idle()
    
    # update values
    svmin.on_changed(update)
    svmax.on_changed(update)
    # svtheta.on_changed(update)

    # export the polypick-selected sky coords
    interact = Interact(ax)
    plt.show()

    if center.i is None:
        if len(interact.center.i):
            center = Coords(interact.center.i[0], interact.center.j[0])
        else:
            raise(Exception('Coordindates of SMBH not specified from radio image.'))

    if len(interact.hotspot.i):
        hotspots = SelectCoords()
        hotspots.i, hotspots.j = interact.hotspot.i, interact.hotspot.j
    else:
        raise(Exception('Coordindates of hotspot not specified from radio image.'))

    return imdata, center, hotspots, w, [cdelt,cdelt]

def select_lobe(radioimage, hotspot, sigma=None):
    hdu = fits.open(radioimage)
    imdata = np.squeeze(hdu[0].data)
    height, width = np.shape(imdata)
    header = hdu[0].header
    w = wcs.WCS(header, naxis=2)
    # create figure
    fig = plt.figure(figsize=(1.5*7,7*(height/width)))
    ax = fig.add_axes([0.075-0.15, 0.05, 0.9, 0.9])#, projection=w)
    ax.set_title("Trace around lobe boundary using `b'.")
    ax.axis('off')
    # stretch='linear'
    norm = simple_norm(imdata, percent=95, stretch='linear')
    im = ax.imshow(imdata, cmap='inferno', norm=norm)
    ax.scatter(hotspot.j, hotspot.i, marker='o', s=50, color='green')

    # setup normalizationlogbu
    pct = 100
    interval = PercentileInterval(pct)
    vmin, vmax = interval.get_limits(imdata)

    # setup interactive sliders
    # axmin = fig.add_axes([0.15, 0.05, 0.1, 0.02])
    # axmax  = fig.add_axes([0.45, 0.05, 0.1, 0.02])
    axperc = fig.add_axes([0.7, 0.05, 0.02, 0.85])
    axcont = fig.add_axes([0.76, 0.05, 0.02, 0.85])
    axlin  = fig.add_axes([0.82, 0.85, 0.15, 0.05])
    axlog  = fig.add_axes([0.82, 0.75, 0.15, 0.05])
    axsqrt = fig.add_axes([0.82, 0.65, 0.15, 0.05])
    
    # svmin = Slider(axmin, "vmin", 0, vmax, valinit=vmin)
    # svmax = Slider(axmax, "vmax", 0, vmax, valinit=vmax)
    svperc = Slider(axperc, "percent", 0, 100, valinit=95, orientation="vertical")
    if sigma is None:
        svcont = Slider(axcont, "contour\n(Jy/beam)", np.log10(np.abs(vmin)), np.log10(vmax), valinit=np.log10(vmax), orientation="vertical", valfmt="$10^{%0.3f}$")
    elif sigma is not None:
        svcont = Slider(axcont, "contour\n(S/N)", -5, 20, valinit=3, orientation="vertical")

    logbutton = Button(axlog, 'log stretch', hovercolor='0.975')
    linbutton = Button(axlin, 'linear stretch', hovercolor='0.975')
    sqrtbutton = Button(axsqrt, 'sqrt stretch', hovercolor='0.975')

    # send interactive commands to redraw plot
    def contour(val=None):
        try:
            stretch
        except NameError:
            ax.clear()
            ax.set_title("Trace around lobe boundary using `b'.")
            ax.scatter(hotspot.j, hotspot.i, marker='o', s=50, color='green')
            norm = simple_norm(imdata, percent=svperc.val, stretch='linear')
            ax.imshow(imdata, cmap='inferno',norm=norm)
            if sigma is not None:
                cont = ax.contour(imdata, colors='cyan', linewidths=0.5, levels=[svcont.val*sigma])
            else:
                cont = ax.contour(imdata, colors='cyan', linewidths=0.5, levels=[10**svcont.val])
            fig.canvas.draw_idle()
        else:
            ax.clear()
            ax.set_title("Trace around lobe boundary using `b'.")
            ax.scatter(hotspot.j, hotspot.i, marker='o', s=50, color='green')
            norm = simple_norm(imdata, percent=svperc.val, stretch=stretch)
            ax.imshow(imdata, cmap='inferno',norm=norm)
            if sigma is not None:
                cont = ax.contour(imdata, colors='cyan', linewidths=0.5, levels=[svcont.val*sigma])
            else:
                cont = ax.contour(imdata, colors='cyan', linewidths=0.5, levels=[10**svcont.val])
            fig.canvas.draw_idle()

    def percent(val=None):
        try:
            stretch
        except NameError:
            ax.clear()
            ax.set_title("Trace around lobe boundary using `b'.")
            ax.scatter(hotspot.j, hotspot.i, marker='o', s=50, color='green')
            norm = simple_norm(imdata, percent=svperc.val, stretch='linear')
            ax.imshow(imdata, cmap='inferno', norm=norm)
            if sigma is not None:
                cont = ax.contour(imdata, colors='cyan', linewidths=0.5, levels=[svcont.val*sigma])
            else:
                cont = ax.contour(imdata, colors='cyan', linewidths=0.5, levels=[10**svcont.val])
            fig.canvas.draw_idle()
        else:
            ax.clear()
            ax.set_title("Trace around lobe boundary using `b'.")
            ax.scatter(hotspot.j, hotspot.i, marker='o', s=50, color='green')
            norm = simple_norm(imdata, percent=svperc.val, stretch=stretch)
            ax.imshow(imdata, cmap='inferno', norm=norm)
            if sigma is not None:
                cont = ax.contour(imdata, colors='cyan', linewidths=0.5, levels=[svcont.val*sigma])
            else:
                cont = ax.contour(imdata, colors='cyan', linewidths=0.5, levels=[10**svcont.val])
            fig.canvas.draw_idle()

    def lin_stretch(val=None):
        ax.clear()
        ax.set_title("Trace around lobe boundary using `b'.")
        global stretch
        ax.scatter(hotspot.j, hotspot.i, marker='o', s=50, color='green')
        stretch = 'linear'
        norm = simple_norm(imdata, percent=svperc.val, stretch='linear')
        ax.imshow(imdata, cmap='inferno', norm=norm)
        if sigma is not None:
            cont = ax.contour(imdata, colors='cyan', linewidths=0.5, levels=[svcont.val*sigma])
        else:
            cont = ax.contour(imdata, colors='cyan', linewidths=0.5, levels=[10**svcont.val])
        fig.canvas.draw_idle()

    def log_stretch(val=None):
        ax.clear()
        ax.set_title("Trace around lobe boundary using `b'.")
        global stretch
        ax.scatter(hotspot.j, hotspot.i, marker='o', s=50, color='green')
        stretch = 'log'
        norm = simple_norm(imdata, percent=svperc.val, stretch='log')
        ax.imshow(imdata, cmap='inferno', norm=norm)
        if sigma is not None:
            cont = ax.contour(imdata, colors='cyan', linewidths=0.5, levels=[svcont.val*sigma])
        else:
            cont = ax.contour(imdata, colors='cyan', linewidths=0.5, levels=[10**svcont.val])
        fig.canvas.draw_idle()

    def sqrt_stretch(val=None):
        ax.clear()
        ax.set_title("Trace around lobe boundary using `b'.")
        global stretch
        ax.scatter(hotspot.j, hotspot.i, marker='o', s=50, color='green')
        stretch = 'sqrt'
        norm = simple_norm(imdata, percent=svperc.val, stretch='sqrt')
        ax.imshow(imdata, cmap='inferno', norm=norm)
        if sigma is not None:
            cont = ax.contour(imdata, colors='cyan', linewidths=0.5, levels=[svcont.val*sigma])
        else:
            cont = ax.contour(imdata, colors='cyan', linewidths=0.5, levels=[10**svcont.val])
        fig.canvas.draw_idle()

    # def localrms(text):
    #     ax.clear()
    #     ax.set_title("Trace around lobe boundary using `b'.")
    #     global sigma
    #     sigma = 1e-3*eval(text)
    #     norm = simple_norm(imdata, percent=svperc.val, stretch='linear')
    #     ax.imshow(imdata, cmap='inferno', norm=norm)
    #     cont = ax.contour(imdata, colors='cyan', linewidths=0.5, levels=[svcont.val*sigma])
    #     fig.canvas.draw_idle()

    # axbox = fig.add_axes([0.82, 0.55, 0.15, 0.05])
    # text_box = TextBox(axbox, 'Local RMS\n(mJy/beam)', initial=1000)
    # text_box.on_submit(localrms)
    
    # update values
    svperc.on_changed(percent)
    svcont.on_changed(contour)
    logbutton.on_clicked(log_stretch)
    linbutton.on_clicked(lin_stretch)
    sqrtbutton.on_clicked(sqrt_stretch)

    # export the polypick-selected sky coords
    interact = Interact(ax)
    plt.show()

    if len(interact.lobe.i):
        indexes = create_index_array(hdu)
        
        lobeedge = SelectCoords()
        lobeedge.i, lobeedge.j = interact.lobe.i, interact.lobe.j

        path = mpltPath.Path(list(zip(lobeedge.i,lobeedge.j)))
        inside = path.contains_points(indexes)
        
        imdata_new = np.zeros_like(imdata)
        image_mask = np.zeros_like(imdata_new)
        for ix in indexes[np.where(inside)]:
            if np.isnan(imdata[ix[0],ix[1]]) or imdata[ix[0],ix[1]]<=0:
                imdata_new[ix[0],ix[1]] = 0
            else:
                imdata_new[ix[0],ix[1]] = imdata[ix[0],ix[1]]
                image_mask[ix[0],ix[1]] = 1
        
        if len(interact.exclude.i) and len(interact.fill.i):
            fill = SelectCoords()
            fill.i, fill.j = interact.fill.i, interact.fill.j
            exclude = SelectCoords()
            exclude.i, exclude.j = interact.exclude.i, interact.exclude.j
            path = mpltPath.Path(list(zip(fill.i,fill.j)))
            inside = path.contains_points(indexes)
            means = []
            for ix in indexes[np.where(inside)]:
                means.append(np.mean(imdata[ix[0],ix[1]]))
            means = np.mean(means)

            path = mpltPath.Path(list(zip(exclude.i,exclude.j)))
            inside = path.contains_points(indexes)
            for ix in indexes[np.where(inside)]:
                imdata_new[ix[0],ix[1]] = np.nan
                image_mask[ix[0],ix[1]] = 0
        
        # fig = plt.figure(figsize=(7,7))
        # ax = fig.add_axes([0.05, 0.05, 0.9, 0.9])#, projection=w)
        # im = ax.imshow(image_mask, cmap='inferno', norm=norm, origin='lower')
        # plt.show()
        
        return imdata_new, image_mask
    else:
        raise(Exception('Boundary around lobe not drawn.'))


def FitObservedLobes(radioimage, hostimage):
    # file containing NAME, core_RA/host_ra, hotspot_ra/dec
    # interactively draw position angle: use either as fixed value, allow free, or constrain within boundary
    # save output images into file

    center = Coords(None, None)
    if hostimage is not None:
        center = host_from_image(hostimage, radioimage)

    imdata, center, hotspots, w, cdelt =  select_radio_source(radioimage, center)

    image_preds = []
    for hotspotPointer in range(0, len(hotspots.i)):
        hotspot = Coords(hotspots.i[hotspotPointer], hotspots.j[hotspotPointer])
        imdata_, image_mask = select_lobe(radioimage, hotspot, sigma=None)
        image_pred = OptimizeSkewGauss2D(imdata_, cdelt, center=center, hotspot=hotspot, image_mask=image_mask, stretch='sqrt', w=w)
        image_preds.append(image_pred)

    fig = plt.figure(figsize=(7,7))
    ax = fig.add_axes([0.1, 0.1, 0.85, 0.85])#, projection=w)
    norm = simple_norm(imdata, percent=99.5, stretch='sqrt')
    ax.imshow(imdata, cmap='inferno', norm=norm)
    ax.scatter(center.j, center.i, **centermarker)
    ax.scatter(center.j, center.i, **centermarker_)
    format(ax)
    for hotspotPointer in range(0, len(hotspots.i)):
        #get the location of the peak pixel in predicted image
        hotspot_value, hotspot_loc = peakpixel(image_preds[hotspotPointer])

        # get 1,2 and 3sigma values of peak
        levels = [hotspot_value*np.exp(-2)]
        ax.contour(image_preds[hotspotPointer], levels=levels, colors='red', linewidths=1)
    plt.show()