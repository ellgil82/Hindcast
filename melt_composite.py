# Define where the script is running
host = 'jasmin'

# Import packages
import iris
import iris.plot as iplt
import sys
import numpy as np
import pandas as pd
import os
import fnmatch
import scipy
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import matplotlib.dates as mdates
from matplotlib import rcParams
from matplotlib.lines import Line2D
if host == 'jasmin':
    sys.path.append('/gws/nopw/j04/bas_climate/users/ellgil82/scripts/Tools/')
elif host == 'bsl':
    sys.path.append('/users/ellgil82/scripts/Tools/')

from tools import compose_date, compose_time, find_gridbox
from find_gridbox import find_gridbox
from rotate_data import rotate_data
from divg_temp_colourmap import shiftedColorMap
import time
from sklearn.metrics import mean_squared_error
import datetime
import metpy
import metpy.calc


## Load data
def load_vars(year):
    # Set up filepath
    if host == 'jasmin':
        filepath = '/gws/nopw/j04/bas_climate/users/ellgil82/hindcast/output/alloutput/'
        ancil_path = '/gws/nopw/j04/bas_climate/users/ellgil82/hindcast/output/'
        lsm_name = 'land_binary_mask'
    elif host == 'bsl':
        filepath = '/data/mac/ellgil82/hindcast/output/'
        ancil_path = filepath
        lsm_name = 'LAND MASK (No halo) (LAND=TRUE)'
    try:
        melt_flux = iris.load_cube(filepath + year + '_land_snow_melt_flux.nc', 'Snow melt heating flux')  # W m-2
        melt_amnt = iris.load_cube(filepath + year + '_land_snow_melt_amnt.nc', 'Snowmelt')  # kg m-2
        melt_rate = iris.load_cube(filepath+year+'_Ts.nc', 'surface_temperature')
        #melt_rate = iris.load_cube(filepath + year + '_land_snow_melt_rate.nc', 'Rate of snow melt on land')  # kg m-2 s-1
        orog = iris.load_cube(ancil_path + 'orog.nc', 'surface_altitude')
        orog = orog[0, 0, :, :]
        LSM = iris.load_cube(ancil_path + 'new_mask.nc', lsm_name)
        LSM = LSM[0, 0, :, :]
    except iris.exceptions.ConstraintMismatchError:
        print('Files not found')
    var_list = [melt_rate, melt_amnt, melt_flux]
    for i in var_list:
        real_lon, real_lat = rotate_data(i, 2, 3)
    vars_yr = {'melt_flux': melt_flux[:,0,:,:], 'melt_rate': melt_rate[:,0,:,:], 'melt_amnt': melt_amnt[:,0,:,:],
               'orog': orog, 'lsm': LSM,'lon': real_lon, 'lat': real_lat, 'year': year}
    return vars_yr


## Set up plotting options
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Segoe UI', 'Helvetica', 'Liberation sans', 'Tahoma', 'DejaVu Sans',
                               'Verdana']


year_list = ['1998', '2000', '2001', '2002', '2003', '2004', '2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017']

def composite_plot(year_list):
    fig, ax = plt.subplots(7, 3, figsize=(8, 18))
    CbAx = fig.add_axes([0.25, 0.1, 0.5, 0.02])
    ax = ax.flatten()
    for axs in ax:
        axs.axis('off')
    total_melt = np.zeros((220,220))
    plot = 0
    for year in year_list:
        vars_yr = load_vars(year)
        c = ax[plot].pcolormesh(np.ma.masked_where(vars_yr['orog'].data > 100, a = np.cumsum(vars_yr['melt_amnt'].data, axis = 0)[-1]), vmin = 0, vmax = 4)
        ax[plot].contour(vars_yr['lsm'].data, colors = '#222222', lw = 2)
        ax[plot].contour(vars_yr['orog'].data, colors = '#222222', levels = [100])
        total_melt = total_melt + (np.cumsum(vars_yr['melt_amnt'].data, axis = 0)[-1])
        ax[plot].text(0.4, 1.1, s = year_list[plot], fontsize = 24,  color='dimgrey', transform = ax[plot].transAxes)
        plot = plot+1
    mean_melt_composite = np.ma.masked_where(vars_yr['orog'].data > 100, a =total_melt/len(year_list))
    ax[-1].contour(vars_yr['lsm'].data, colors='#222222', lw=2)
    ax[-1].pcolormesh(mean_melt_composite)
    ax[-1].contour(vars_yr['orog'].data, colors = '#222222', levels=[100])
    ax[-1].text(0., 1.1, s = 'Composite', fontsize = 24,  color='dimgrey', transform = ax[-1].transAxes)
    cb = plt.colorbar(c, orientation = 'horizontal', cax = CbAx, ticks = [0,2,4])
    cb.solids.set_edgecolor("face")
    cb.outline.set_edgecolor('dimgrey')
    cb.ax.tick_params(which='both', axis='both', labelsize=24, labelcolor='dimgrey', pad=10, size=0, tick1On=False, tick2On=False)
    cb.outline.set_linewidth(2)
    cb.ax.xaxis.set_ticks_position('bottom')
    #cb.ax.set_xticks([0,4,8])
    cb.set_label('Annual snow melt amount (kg m-$^{2}$)', fontsize = 24,  color='dimgrey', labelpad = 30)
    plt.subplots_adjust(left = 0.05, right = 0.95, top = 0.95, bottom = 0.15, hspace = 0.3, wspace = 0.05)
    if host == 'bsl':
        plt.savefig('/users/ellgil82/figures/Hindcast/SMB/melt_all_years.png', transparent = True)
        plt.savefig('/users/ellgil82/figures/Hindcast/SMB/melt_all_years.eps', transparent = True)
    elif host == 'jasmin':
        plt.savefig('/gws/nopw/j04/bas_climate/users/ellgil82/hindcast/figures/melt_all_years.png', transparent = True)
        plt.savefig('/gws/nopw/j04/bas_climate/users/ellgil82/hindcast/figures/melt_all_years.eps', transparent = True)
    plt.show()


composite_plot(year_list)

def total_melt(year_list):
    total_melt = np.zeros((220, 220))
    for year in year_list:
        vars_yr = load_vars(year)
        total_melt = total_melt + (np.cumsum(vars_yr['melt_amnt'].data, axis=0)[-1])
    total_melt_masked = np.ma.masked_where(vars_yr['orog'].data > 100, total_melt, copy = True)
    totm = total_melt_masked.sum()
    return totm, total_melt_masked

totm, totm_masked = total_melt(year_list)