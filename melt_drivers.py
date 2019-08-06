
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
import glob
from scipy import stats

# Set up filepath
if host == 'jasmin':
    filepath = '/gws/nopw/j04/bas_climate/users/ellgil82/hindcast/output/alloutput/'
    lsm_name = 'land_binary_mask'
elif host == 'bsl':
    filepath = '/data/mac/ellgil82/hindcast/output/'
    lsm_name = 'LAND MASK (No halo) (LAND=TRUE)'

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

surf_2012 = load_vars('2012')

year_list = ['1998', '2000', '2001', '2002', '2003', '2004', '2005', '2006', '2007', '2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017']


lon_index14, lat_index14 = find_gridbox(-67.01, -61.03, surf_2012['lat'], surf_2012['lon'])
lon_index15, lat_index15 = find_gridbox(-67.34, -62.09, surf_2012['lat'], surf_2012['lon'])
lon_index17, lat_index17 = find_gridbox(-65.93, -61.85, surf_2012['lat'], surf_2012['lon'])
lon_index18, lat_index18 = find_gridbox(-66.48272, -63.37105, surf_2012['lat'], surf_2012['lon'])

lat_dict = {'AWS14': lat_index14,
            'AWS15': lat_index15,
            'AWS17': lat_index17,
            'AWS18': lat_index18}

lon_dict = {'AWS14': lon_index14,
            'AWS15': lon_index15,
            'AWS17': lon_index17,
            'AWS18': lon_index18}

station_dict = {'AWS14_SEB_2009-2017_norp.csv': 'AWS14',
             'AWS15_hourly_2009-2014.csv': 'AWS15',
              'AWS17_SEB_2011-2015_norp.csv': 'AWS17',
                'AWS18_SEB_2014-2017_norp.csv': 'AWS18'}


## Diagnose correlation between shortwave flux and melting across model domain
def cloud_melt(station, year_list):
    cloud_melt = pd.DataFrame(index = ['r', 'r_squared', 'p', 'std_err'])
    for year in year_list:
        SW = iris.load_cube(filepath + year + '_surface_SW_down.nc', 'surface_downwelling_shortwave_flux_in_air')
        melt = iris.load_cube(filepath + year + '_land_snow_melt_amnt.nc', 'Snowmelt')
        SW = SW[:,0,:,:]
        melt = melt[:,0,:,:]
        if station == 'ice_shelf':
            orog = iris.load_cube(filepath + 'orog.nc', 'surface_altitude')
            orog = orog[0, 0, :, :]
            LSM = iris.load_cube(filepath + 'new_mask.nc', lsm_name)
            lsm = LSM[0, 0, :, :]
            # Make ice shelf mask
            Larsen_mask = np.zeros((220, 220))
            lsm_subset = lsm.data[:150, 90:160]
            Larsen_mask[:150, 90:160] = lsm_subset
            Larsen_mask[orog.data > 100] = 0
            Larsen_mask = np.logical_not(Larsen_mask)
            melt_masked = np.ma.masked_array(melt.data, mask=np.broadcast_to(Larsen_mask, melt.shape)).mean(axis=(1, 2))
            SW_masked = np.ma.masked_array(SW.data, mask=np.broadcast_to(Larsen_mask, SW.shape)).mean(axis=(1, 2))
        else:
            melt_masked = melt.data[:, lat_dict[station], lon_dict[station]]
            SW_masked =SW.data[:, lat_dict[station], lon_dict[station]]
        slope, intercept, r_value, p_value, std_err = stats.linregress(melt_masked, SW_masked)
        stats_yr = [r_value, r_value**2, p_value, std_err]
        cloud_melt[year] = pd.Series(stats_yr, index = ['r', 'r_squared', 'p', 'std_err'])
        cloud_melt.to_csv(filepath + 'cloud_v_melt_stats_model_' + station + '.csv')

#cloud_melt(station = 'AWS14', year_list= year_list)
#cloud_melt(station = 'AWS15', year_list = year_list)
#cloud_melt(station = 'AWS17', year_list= year_list)
#cloud_melt(station = 'AWS18', year_list = year_list)
#cloud_melt(station = 'ice_shelf', year_list= year_list)

for file in os.listdir(filepath):
    if fnmatch.fnmatch(file, 'Modelled_seasonal_foehn_frequency_%(station)s*.csv' % locals()):
        foehn_freq = pd.read_csv(str(file), na_values=-9999, header=0)
for year in year_list:
    print('Loading model data from ' + year)
    ANN = load_vars(year)
    DJF = load_vars('DJF_'+year)
    MAM = load_vars('MAM_'+year)
    JJA = load_vars('JJA_'+year)
    SON = load_vars('SON_'+year)
    total_melt_ANN = np.cumsum(ANN['melt_amnt'].data[:,lat_dict[station], lon_dict[station]], axis = 0)[-1]
    total_melt_DJF = np.cumsum(DJF['melt_amnt'].data[:, lat_dict[station], lon_dict[station]], axis=0)[-1]
    total_melt_MAM = np.cumsum(MAM['melt_amnt'].data[:, lat_dict[station], lon_dict[station]], axis=0)[-1]
    total_melt_JJA = np.cumsum(JJA['melt_amnt'].data[:, lat_dict[station], lon_dict[station]], axis=0)[-1]
    total_melt_SON = np.cumsum(SON['melt_amnt'].data[:, lat_dict[station], lon_dict[station]], axis=0)[-1]



stats_df = pd.DataFrame(index = ['r', 'r_squared', 'p', 'std_err'])
slope, intercept, r_value, p_value, std_err = stats.linregress(foehn_freq['ANN'],total_melt_ANN)
ANN_stats = [r_value, r_value**2, p, std_err]

slope, intercept, r_value, p_value, std_err = stats.linregress(foehn_freq['MAM'],total_melt_MAM)
MAM_stats = [r_value, r_value**2, p, std_err]

slope, intercept, r_value, p_value, std_err = stats.linregress(foehn_freq['DJF'],total_melt_DJF)
DJF_stats = [r_value, r_value**2, p, std_err]

slope, intercept, r_value, p_value, std_err = stats.linregress(foehn_freq['JJA'],total_melt_JJA)
JJA_stats = [r_value, r_value**2, p, std_err]

slope, intercept, r_value, p_value, std_err = stats.linregress(foehn_freq['SON'],total_melt_SON)
SON_stats = [r_value, r_value**2, p, std_err]