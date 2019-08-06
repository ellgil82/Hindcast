# Define where the script is running
host = 'bsl'

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

# Set up filepath
if host == 'jasmin':
    filepath = '/gws/nopw/j04/bas_climate/users/ellgil82/hindcast/output/test_run/'
elif host == 'bsl':
    filepath = '/data/mac/ellgil82/hindcast/output/'

## Load data
def load_vars():
    # Set up filepath
    if host == 'jasmin':
        filepath = '/gws/nopw/j04/bas_climate/users/ellgil82/hindcast/output/'
        ancil_path = '/gws/nopw/j04/bas_climate/users/ellgil82/hindcast/output/'
        lsm_name = 'land_binary_mask'
    elif host == 'bsl':
        filepath = '/data/mac/ellgil82/hindcast/output/'
        ancil_path = filepath
        lsm_name = 'LAND MASK (No halo) (LAND=TRUE)'
    try:
        os.chdir(filepath)
        Tair = iris.load_cube( '*_Tair_1p5m_daymn.nc', 'air_temperature')
        Ts = iris.load_cube( '*_Ts_daymn.nc', 'surface_temperature')
        melt_flux = iris.load_cube('*_land_snow_melt_flux_daymn.nc', 'Snow melt heating flux')
        melt_amnt = iris.load_cube('*_land_snow_melt_amnt_daymn.nc', 'Snow melt')
        change_me = [Ts, Tair]
        for i in change_me:
            i.convert_units('celsius')
            real_lon, real_lat = rotate_data(i, 2,3)
        orog = iris.load_cube(ancil_path + 'orog.nc', 'surface_altitude')
        orog = orog[0, 0, :, :]
        LSM = iris.load_cube(ancil_path + 'new_mask.nc', lsm_name)
        LSM = LSM[0, 0, :, :]
        print('You spin me right round baby, right round...')
        for i in [melt_flux, melt_rate, orog, LSM]:
            real_lon, real_lat = rotate_data(i, 1,2)
    except iris.exceptions.ConstraintMismatchError:
        print('Files not found')
    var_list = {'Ts': Ts, 'Tair': Tair, 'melt_flux': melt_flux, 'melt_amnt': melt_amnt, 'orog': orog, 'lsm': LSM, 'lat': real_lat, 'lon': real_lon}
    return var_list

all_yrs = load_vars()

# Create pandas dataframe with each file loaded in as a time series and then added as a column (would only work for specific locations/whole ice shelf mean)
ice_shelf = pd.DataFrame()
AWS14 = pd.DataFrame()
AWS15 = pd.DataFrame()
AWS17 = pd.DataFrame()
AWS18 = pd.DataFrame()
for i in ['Ts', 'Tair', 'melt_amnt', 'melt_flux']:
    ice_shelf_masked_srs = np.ma.masked_where(var_list['orog'].data >= 100. and var_list['lsm'].data == 0, var_list[i].data, copy = True)
    ice_shelf_srs = np.mean(ice_shelf_masked_srs, axis = (1,2))
    AWS14_srs = np.mean(i[:,lat14_index, lon14_index].data, axis = (1,2))
    AWS15_srs = np.mean(i[:,lat15_index, lon15_index].data, axis = (1,2))
    AWS17_srs = np.mean(i[:,lat17_index, lon17_index].data, axis = (1,2))
    AWS18_srs = np.mean(i[:,lat18_index, lon18_index].data, axis = (1,2))
    ice_shelf[i] = ice_shelf_srs
    AWS14[i] = AWS14_srs
    AWS15[i] = AWS15_srs
    AWS17[i] = AWS17_srs
    AWS18[i] = AWS18_srs





        MSLP = iris.load_cube( '*_MSLP_daymn.nc', 'air_pressure_at_sea_level')
        sfc_P = iris.load_cube('*_sfc_P_daymn.nc', 'surface_air_pressure')
        FF_10m = iris.load_cube( '*_FF_10m_daymn.nc', 'wind_speed')
        RH = iris.load_cube('*_RH_1p5m_daymn.nc', 'relative_humidity')
        u = iris.load_cube('*_u_10m_daymn.nc', 'x wind component (with respect to grid)')
        v = iris.load_cube('*_v_10m_daymn.nc', 'y wind component (with respect to grid)')
        LWnet = iris.load_cube('*_surface_LW_net_daymn.nc', 'surface_net_downward_longwave_flux')
        SWnet = iris.load_cube('*_surface_SW_net_daymn.nc','Net short wave radiation flux')
        LWdown = iris.load_cube('*_surface_LW_down_daymn.nc', 'IR down')
        SWdown = iris.load_cube('*_surface_SW_down_daymn.nc', 'surface_downwelling_shortwave_flux_in_air')
        HL = iris.load_cube('*_latent_heat_daymn.nc', 'Latent heat flux')
        HS = iris.load_cube('*_sensible_heat_daymn.nc', 'surface_upward_sensible_heat_flux')

        Tair.convert_units('celsius')
        Ts.convert_units('celsius')
        MSLP.convert_units('hPa')
        sfc_P.convert_units('hPa')
        FF_10m = FF_10m[:,:,1:,:]
        v = v[:,:,1:,:]
        var_list = [Tair, Ts, MSLP, sfc_P, FF_10m, RH, u, v, LWnet, SWnet, LWdown, SWdown, HL, HS]
        for i in var_list:
            real_lon, real_lat = rotate_data(i, 2, 3)
        WD = metpy.calc.wind_direction(u = u.data, v = v.data)
        WD = iris.cube.Cube(data = WD, standard_name='wind_from_direction')
        Etot = LWnet.data + SWnet.data - HL.data - HS.data
        for turb in [HS, HL]:
            turb.data = 0-turb.data
        Emelt_calc = iris.cube.Cube(Emelt_calc)
        Etot = iris.cube.Cube(Etot)
    vars_yr = {'Tair': Tair[:,0,:,:], 'Ts': Ts[:,0,:,:], 'MSLP': MSLP[:,0,:,:], 'sfc_P': sfc_P[:,0,:,:], 'FF_10m': FF_10m[:,0,:,:],
               'RH': RH[:,0,:,:], 'WD': WD[:,0,:,:], 'LWnet': LWnet[:,0,:,:], 'SWnet': SWnet[:,0,:,:], 'SWdown': SWdown[:,0,:,:],
               'LWdown': LWdown[:,0,:,:], 'HL': HL[:,0,:,:], 'HS': HS[:,0,:,:], 'Etot': Etot[:,0,:,:], 'Emelt': melt[:,0,:,:],
               'lon': real_lon, 'lat': real_lat, 'year': year, 'Emelt_calc': Emelt_calc[:,0,:,:]}
    return vars_yr


files = []
for file in os.listdir(os.getcwd()):
    if fnmatch.fnmatch(file, '????_Tair_1p5m.nc'):
        files.append(file)



Ts_files = []
Tair_files = []
FF_files = []
P_files = []
os.chdir(filepath)
for i in os.listdir(filepath):
	if fnmatch.fnmatch(i, '2016_Ts_daymn.nc*'):
		Ts_files.append(i)
	elif fnmatch.fnmatch(i, '*Tair_daymn.nc*'):
		Tair_files.append(i)
	elif fnmatch.fnmatch(i, '*MSLP_daymn.nc*'):
		P_files.append(i)
	elif fnmatch.fnmatch(i, '*FF_10m_daymn.nc*'):
		FF_files.append(i)

Ts_cubes = iris.load(Ts_files)
Tair_cubes = iris.load(Tair_files)
P_cubes = iris.load(P_files)
FF_cubes = iris.load(FF_files)
cube_list_list = [Ts_cubes, Tair_cubes, P_cubes, FF_cubes]
for cube_list in cube_list_list:
	for i in range(len(cube_list)):

		if len(cube_list[i].coord('time').points) == 367:
			print('huzzah')
			cube_list[i] = cube_list[i][:366,:,:,:]
	cube_list_list[k].concatenate_cube()




for i in range(len(Ts_cubes)):
	Ts_cubes[i] = Ts_cubes[i][:,0,:,:]

Ts = iris.load(files_to_load, 'surface_temperature')
Tair = iris.load_cube(files_to_load, 'air_temperature')
FF_10m = iris.load_cube(files_to_load, 'wind_speed')
MSLP = Tair = iris.load_cube(files_to_load, 'air_pressure_at_sea_level')