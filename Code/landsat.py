# -*- coding: utf-8 -*-
"""
Created on Monday August 14, 2023 - 14:19:51
(Last Updated: 10/29/2024)

@author: Kelly Wang


Landsat 8 Bands 4,5,10 Processing

"""
import pandas as pd
import geopandas as gpd
import numpy as np

import altair as alt
import hvplot.xarray
import hvplot.pandas
import holoviews as hv
from matplotlib import pyplot as plt
from matplotlib import colors as color
import seaborn as sns

import rasterio as rio
from rasterio.mask import mask
from rasterstats import zonal_stats

import xarray as xr
import rioxarray as rxr

from pysal.lib import weights
from pysal.model import spreg
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn import linear_model as lm

import esda


def masked_tif(rio_file, boundary, folder):
    """
    Create a masked image.

    Parameters
    ----------
    rio_file : TIF file
        Opened TIF file using Rasterio.
    boundary : GeoPandas Dataframe
        Geography of the boundaries to mask the image to.
    folder : str
        Directory where final processed image is stored.

    Returns
    -------
    None.

    """
    masked, masked_transform = rio.mask.mask(rio_file, boundary.geometry,
                                             crop=True, all_touched=True)
    folder_name = folder + ".tif"
    masked_rio = rio.open(folder_name, "w", driver="GTiff", 
                          height=masked.shape[1], width=masked.shape[2],
                          count=1, dtype="float64", crs = 32618, 
                          transform=masked_transform) 
    masked_rio.write_band(1, masked[0])
    masked_rio.close()
    
    
def ndvi_tif(red_file, nir_file, folder):
    """
    Calculate NDVI using Landsat 8's bands 4 and 5 from Collection 2 Level 1 
        images.

    Parameters
    ----------
    red_file : str
        Directory where the red (R) image is stored.
        Band 4 in Landsat 8.
    nir_file : str
        Directory where the near infrared (NIR) image is stored.
        Band 5 in Landsat 8..
    folder : str
        Directory where the final processed image is stored.

    Returns
    -------
    None.

    """
    red = rio.open(red_file)
    nir = rio.open(nir_file)
    
    band4 = red.read(1).astype("float64")
    band5 = nir.read(1).astype("float64")
    np.seterr(divide="ignore", invalid="ignore")
    ndvi = np.where((band5 + band4)==0.0, 0, (band5 - band4) / (band5 + band4))
    
    folder_name = folder + ".tif"
    ndvi_rio = rio.open(folder_name, "w", driver="GTiff", height=ndvi.shape[0], 
                        width=ndvi.shape[1], count=1, dtype="float64", 
                        crs = 32618, transform = red.transform) 
    ndvi_rio.write(ndvi,1)
    ndvi_rio.close() 
    
    
def temp_tif(temp_file, ndvi_file, folder, band = 10, celsius=False, 
             boundary=pd.DataFrame()):
    """
    Calculate land surface temperature using Landsat 8's band 10 from
        Collection 2 Level 1 images.
    
    temp = BT/1 + w * (BT /p) * ln(e)
    
    BT = At Satellite temperature (brightness)
    w = wavelength of emitted radiance (μm)
    
    where p = h * c / s (1.439e-2 mK)
    
    h = Planck's constant (Js)
    s = Boltzmann constant (J/K)
    c = velocity of light (m/s)

    Parameters
    ----------
    temp_file : str
        Directory where the thermal infrared image (TIR) is stored.
        Bands 10 and 11 in Landsat 8.
    ndvi_file : str
        Directory where the NDVI image is stored.
    folder : str
        Directory where final processed image is stored.
    band : int, optional
        The input TIR's band number that will be used to calculate LST. 
        The default is 10 (but can be changed to 11 in Landsat 8).
    celsius : bool, optional
        Decides what unit to calculate LST. 
        The default is False (Fahrenheit).
    boundary : GeoPandas Dataframe, optional
        Geography of the boundaries where temperature will be calculated. 
        The default is pd.DataFrame().

    Returns
    -------
    None.

    """
    temp = rxr.open_rasterio(temp_file)
    ndvi = rxr.open_rasterio(ndvi_file)
    
    if not boundary.empty:
        xmin, ymin, xmax, ymax = boundary.total_bounds
        temp = temp.sel(x=slice(xmin, xmax), y=slice(ymax, ymin)) 
        ndvi = ndvi.sel(x=slice(xmin, xmax), y=slice(ymax, ymin)) 
        # NOTE: y-coord system is in descending order!
        
        temp = temp.compute()
        ndvi = ndvi.compute()

    
    toa_rad = 3.342e-4 * temp + 0.1 #- 0.29
    
    k_1 = 774.8853 
    k_2 = 1321.0789 
    
    toa_bright = k_2 / np.log((k_1 / toa_rad) + 1) # Kelvin
    # toa_bright_c = toa_bright_k - 273.15 # Celsius
    
    veg_prop = ((ndvi - ndvi.min()) / (ndvi.max() - ndvi.min())) ** 2
    
    emissivity = 0.004 * veg_prop + 0.986
    
    h = 6.62607e-34
    s = 1.38065e-23
    c = 2.9979e8

    rho = h * c / s
    
    wavelength = 10.895e-6
    if band == 11:
        wavelength = 12.005e-6

    lst = toa_bright / (1 + ((wavelength * toa_bright / rho) * np.log(emissivity)))
    
    if celsius:
        lst = lst - 273.15
    else:
        lst = (9*(lst - 273.15) / 5) + 32
    
    '''
    # Alternative code
    temp_rio = rio.open(folder_name, "w", driver="GTiff", height=lst.shape[0], 
                        width=lst.shape[1], count=1, dtype="float64", 
                        crs = 32618, transform = temp.transform) 
    temp_rio.write(lst,1)
    temp_rio.close() 
    '''
    folder_name = folder + ".tif"
    lst.rio.to_raster(folder_name)
    


def stats(groups, raster_file, rio_file = None, crs = 32618,
          stats=["median", "mean", "std", "min", "max"]):
    """
    Calculating the statistics of the image values based on the given
        geographical boundaries.

    Parameters
    ----------
    groups : GeoPandas Dataframe
        Geography of the regional groups that will be used partition the
            calculations of each statistic.
    raster_file : str
        DESCRIPTION.
    rio_file : TIF file, optional
        Can instead input an opened TIF file using Rasterio. 
        The default is None.
    crs : int, optional
        DESCRIPTION. The default is 32618.
    stats : list, optional
        A list of statistics that will be calculated. 
        The default is ["median", "mean", "std", "min", "max"].

    Returns
    -------
    geostats : GeoPandas Dataframe
        Dataframe with the given statistics on the input regions.

    """
    if rio_file == None:
        results = zonal_stats(groups, raster_file, stats=stats)
    else:
        results = zonal_stats(groups, raster_file, affine=rio_file.transform,
                              stats=stats)
    
    geostats = groups # gpd.GeoDataFrame.from_features(results)
    
    for s in stats:
        geostats[s] = [data[s] for data in results]
    
    return geostats
    

def geo_plot(rio_file, boundary, title, color, x_size=10, y_size=10, 
             borderwidth=2.5):
    """
    Graphing geographical images using Matplotlib.

    Parameters
    ----------
    rio_file : TIF file
        Opened TIF file using Rasterio.
    boundary : GeoPandas Dataframe
        Geography of the boundaries to graph the image.
    title : str
        Title of the plot.
    color : str
        Matplotlib color scheme.
    x_size : float, optional
        Length of the x-axis of the plot. The default is 10.
    y_size : float, optional
        Length of the y-axis of the plot. The default is 10.
    borderwidth : float, optional
        Line weight of the border outline. The default is 2.5.

    Returns
    -------
    fig : plt.figure()
        Final output plot's figure.
    ax : plt.axis() / plt.axes()
        Final output plot's axis/axes.

    """
    image, image_transform = mask(dataset=rio_file, shapes=boundary.geometry,
        crop=True,  # Remove pixels not within boundary
        all_touched=True,  # Get all pixels that touch the boudnary
        filled=False,  # Do not fill cropped pixels with a default value
    )
    
    
    fig, ax = plt.subplots(figsize=(x_size, y_size))

    # Boundary
    landsat_extent = [rio_file.bounds.left, rio_file.bounds.right, 
                      rio_file.bounds.bottom, rio_file.bounds.top]
    
    img = ax.imshow(image[0], extent=landsat_extent, cmap = color) 
    
    # Format and plot city limits
    boundary.plot(ax = ax, edgecolor="dimgray", facecolor="none", 
                  linewidth=borderwidth)
    plt.colorbar(img)
    
    ax.set_axis_off()
    ax.set_title(title, fontsize=15) # Use \n to create new line
    
    return fig, ax



# --- Past Code ---

def landsat_processing(data, bands, bounds, crs = 32618):
    """
    Process landsat images into a manipulable form.
    
    Parameters
    ----------
    data : TYPE
        DESCRIPTION.
    bands : TYPE
        DESCRIPTION.
    bounds : TYPE
        DESCRIPTION.
    crs : int, optional
        DESCRIPTION. The default is 32618.

    Returns
    -------
    landsat : xarray
        Concatenated layers of Landsat bands into for an area's boundary.

    """
    
    image = []
    
    for i in range(len(data)):
        layer = rxr.open_rasterio(data[i])
        layer = layer.assign_coords(band=[bands[i]])
        image.append(layer)
        
    landsat = xr.concat(image, "band", compat="identical")
    
    city_limits = bounds.to_crs(crs)
    xmin, ymin, xmax, ymax = city_limits.total_bounds
    
    # Slice our xarray data
    landsat = landsat.sel(x=slice(xmin, xmax), y=slice(ymax, ymin))
    landsat = landsat.compute()

    return landsat


def veg_index(landsat):
    """
    Calculate Normalized Difference Vegetation Index (ranges from -1 to 1).

    Parameters
    ----------
    landsat : TYPE
        DESCRIPTION.

    Returns
    -------
    ndvi : TYPE
        DESCRIPTION.

    """
    red = landsat.sel(band=4)
    nir = landsat.sel(band=5)

    # Calculate the NDVI
    ndvi = (nir - red) / (nir + red)
    ndvi = ndvi.where(ndvi < np.inf)
    
    return ndvi
    
    
def land_surface_temp(landsat, ndvi, celsius = True):
    """
    Calculate land surface temperature of Landsat 8 using band 10 
        Level 1 collection
    
    temp = BT/1 + w * (BT /p) * ln(e)
    
    BT = At Satellite temperature (brightness)
    w = wavelength of emitted radiance (μm)
    
    where p = h * c / s (1.439e-2 mK)
    
    h = Planck's constant (Js)
    s = Boltzmann constant (J/K)
    c = velocity of light (m/s)

    Parameters
    ----------
    landsat : TYPE
        DESCRIPTION.
    ndvi : TYPE
        DESCRIPTION.
    celsius : TYPE, optional
        DESCRIPTION. The default is True.

    Returns
    -------
    lst : TYPE
        DESCRIPTION.

    """
    
    toa = 3.342e-4 * landsat.sel(band=10) + 0.1 #- 0.29
    
    k_1 = 774.8853 # 1321.08
    k_2 = 1321.0789 # 777.89
    
    bt = k_2 / np.log((k_1 / toa) + 1)
    
    '''
    if celsius:
        bt = (k_2 / np.log((k_1 / toa) + 1)) - 273.15 # Celsius
    else:
        bt = (9*((k_2 / np.log((k_1 / toa) + 1)) - 273.15)/5) + 32
    '''
    veg_prop = ((ndvi - ndvi.min()) / (ndvi.max() - ndvi.min())) ** 2
    
    
    emissivity = 0.004 * veg_prop + 0.986
    
    h = 6.626e-34
    s = 1.38e-23
    c = 2.998e8

    p = h * c / s
    
    wavelength = 10.895e-6 #0.0010895

    lst = bt / (1 + ((wavelength * bt / p) * np.log(emissivity))) - 273.15
    
    if not celsius:
        lst = (9 * lst /5) + 32    

    return lst
