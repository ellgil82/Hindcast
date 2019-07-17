'''# Diagnose foehn from 3-hourly data

# How to diagnose foehn?
- options (Dan Bannister's thesis):
a) surface met-based, i.e. dT > 2 K AND increased wind speed, decreased RH and wind direction cross-AP for 6+ hours
b) isentrope-based, i.e if dZ isentrope > 1000 m for 6+ hours 
c) Froude number/h-hat based, i.e. if Fr > 0.9 for 6+ hours

Only calculate if wind direction within 30 deg region that makes it 'cross-peninsula'.

1. Select time periods when wind direction within range for 6+ hours in a row
2. At each timestep, determine whether conditions are met (for whichever method is being used) for 2+ timesteps in a row.


Foehn stats:
- number of events
- timing of events (so preserve times - use pandas)
- length of events
- strength of events

'''

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
def load_vars(year):
	# Load surface variables
	Tair = iris.load_cube( filepath + year+'_Tair_1p5m.nc', 'air_temperature')
	Ts = iris.load_cube( filepath + year+'_Ts.nc', 'surface_temperature')
	MSLP = iris.load_cube( filepath + year+'_MSLP.nc', 'air_pressure_at_sea_level')
	sfc_P = iris.load_cube(filepath  + year + '_sfc_P.nc', 'surface_air_pressure')
	FF_10m = iris.load_cube( filepath +year+'_FF_10m.nc', 'wind_speed')
	RH = iris.load_cube(filepath  + year + '_RH_1p5m.nc', 'relative_humidity')
	u = iris.load_cube(filepath  + year + '_u_10m.nc', 'x wind component (with respect to grid)')
	v = iris.load_cube(filepath  + year + '_v_10m.nc', 'y wind component (with respect to grid)')
	# Load profiles
	theta_prof = iris.load_cube(filepath + year + '_theta_full_profile.nc')
	theta_prof = theta_prof[:,:40,:,:]
	u_prof = iris.load_cube(filepath + year + '_u_wind_full_profile.nc')
	v_prof = iris.load_cube(filepath + year + '_v_wind_full_profile.nc')
	v_prof = v_prof[:,:, 1:,:]
	theta_pp = iris.load_cube(filepath + '20141013T0000Z_Peninsula_4km_hindcast_pe000.pp', 'air_potential_temperature')
	theta_pp = theta_pp[:,:40,:,:]
	try:
		LSM = iris.load_cube(filepath + 'new_mask.nc', 'LAND MASK (No halo) (LAND=TRUE)')
		orog = iris.load_cube(filepath + 'orog.nc', 'surface_altitude')
		orog = orog[0, 0, :, :]
		LSM = LSM[0, 0, :, :]
	except iris.exceptions.ConstraintMismatchError:
		print('Files not found')
	# Rotate data onto standard lat/lon grid
	for i in [theta_prof, theta_pp, u_prof, v_prof, orog]:
		real_lon, real_lat = rotate_data(i, np.ndim(i) - 2, np.ndim(i) - 1)
	# Convert model levels to altitude
	# Take orography data and use it to create hybrid height factory instance
	auxcoord = iris.coords.AuxCoord(orog.data, standard_name=str(orog.standard_name), long_name="orography",
									var_name="orog", units=orog.units)
	for x in [theta_prof, u_prof, v_prof]:
		x.add_aux_coord(auxcoord, (np.ndim(x) - 2, np.ndim(x) - 1))
		x.add_aux_coord(theta_pp.coord('sigma'), 1)
		x.add_aux_coord(theta_pp.coord('level_height'), 1)
		factory = iris.aux_factory.HybridHeightFactory(sigma=x.coord("sigma"), delta=x.coord("level_height"), orography=x.coord("surface_altitude"))
		x.add_aux_factory(factory)  # this should produce a 'derived coordinate', 'altitude' (test this with >>> print theta)
    #Tair.convert_units('celsius')
	#Ts.convert_units('celsius')
	#MSLP.convert_units('hPa')
	#sfc_P.convert_units('hPa')
	FF_10m = FF_10m[:,:,1:,:]
	v = v[:,:,1:,:]
	WD = metpy.calc.wind_direction(u = u.data, v = v.data)
	WD = iris.cube.Cube(data = WD, standard_name='wind_from_direction')
	surf_vars_yr = {'Tair': Tair[:,0,:,:], 'Ts': Ts[:,0,:,:], 'MSLP': MSLP[:,0,:,:], 'sfc_P': sfc_P[:,0,:,:], 'FF_10m': FF_10m[:,0,:,:],
               'RH': RH[:,0,:,:], 'WD': WD[:,0,:,:], 'lon': real_lon, 'lat': real_lat, 'year': year}
	prof_vars_yr = {'lon': real_lon, 'lat': real_lat, 'year': year, 'theta': theta_prof, 'u': u_prof, 'v': v_prof, 'altitude': theta_prof.coord('altitude'), 'orog': orog, 'lsm': LSM}
	return surf_vars_yr, prof_vars_yr

surf_2012, prof_2012 = load_vars('2012')


def Froude_number(u_wind):
	'''Calculate Froude number at a distance of one Rossby radius of deformation (150 km) from the Antarctic Peninsula
		mountain barrier.

		Inputs:

			- u_wind: derived from model or observations in m s-1. This can be output from a single model grid point at
			a single height, or averaged over a vertical range, as in Elvidge et al. (2016), who uses the range 200 m
			to 2000 m to be representative of flow impinging on the Peninsula.

		Outputs:

			- Fr: Froude number

			- h_hat: non-dimensional mountain height, h_hat = Nh/U

		'''
	mean_u = np.mean(u_wind)
	N = 0.01 # s-1 = Brunt-Vaisala frequency
	h = 2000 # m = height of AP mountains
	Fr = mean_u/(N*h)
	h_hat = (N*h)/mean_u
	return Fr, h_hat

def uv_to_ws(u_wind,v_wind):
	'''Calculate wind speed from u and v wind components

		Inputs:

			- u_wind: zonal component of wind (m s-1), where positive = westerly

			- v_wind: meridional component of wind (m s-1), where positive = southerly

		Outputs:

			- ws: wind speed (m s-1)

		'''
	ws = np.sqrt(np.square(u_wind) + np.square(v_wind))
	return ws

def uv_to_wd(u_wind,v_wind):
	'''Calculate wind direction from u and v wind components

		Inputs:

			- u_wind: zonal component of wind (m s-1), where positive = westerly

			- v_wind: meridional component of wind (m s-1), where positive = southerly

		Outputs:

			- wd: wind direction (in degrees), where North = 0

		'''
	wd = 270 - ((math.atan2(u_wind, v_wind)) * (180 / math.pi)) % 360
	return wd

# Determine whether undisturbed upstream conditions meet criteria for foehn
#if wd >= 240 and wd <=300 and ws >= 4.0:


## Inputs: 1) latitude at which you want the diagnosis (and longitude), i.e. coordinates of AWS, 2) dictionary of variables for year of interest
# Isentrope method

##AWS 14 coords
meas_lon = -61.50
meas_lat = -67.01




def diag_foehn_isentrope(meas_lat, meas_lon, prof_var):
	''' Diagnose whether foehn conditions are present in data using the isentrope method described by Bannister (2015) and King et al. (2017).

	Assumptions:

		- One Rossby wave of deformation from the mountain crest = ~150 km (longitude gridbox 42 in the domain used).

		- Assume representative mountain height of 2000 m.

	Criteria for foehn:

		- u at Z1 must exceed 2.0 m s-1

		- wind direction must be cross-peninsula

		- Difference between height Z1 and the height of the Z1 isentrope in the transect defined in lee of the barrier (Z2), i.e. Z3 = Z1-Z2, must exceed 1000 m over 6 hours.

	Inputs:

		- meas_lat, meas_lon: latitude and longitude of the location at which you would like to diagnose foehn, typically the location of an AWS.

		- prof_var: dictionary of profile variables for the year of interest, retrieved using the function load_vars.

	Returns:

	    - foehn_freq: a 1D Boolean time series showing whether foehn is occurring or not at the location requested.

	    '''
	# Find gridbox of latitudes and longitudes of AWS/location at which foehn is occurring
	lon_idx, lat_idx = find_gridbox(meas_lon, meas_lat, real_lon=prof_2012['lon'], real_lat=prof_2012['lat'])
	# Find model level closest to 2000 m
	Z1 = np.argmin((prof_var['altitude'][:, lat_idx, 42].points - 2000) ** 2)
	# Find representative u wind upstream of mountains by at least one Rossby wave of deformation and above the mountain crest
	u_Z1 = prof_var['u'][:,Z1, lat_idx, 42]
	v_Z1 = prof_var['v'][:,Z1, lat_idx, 42]
	# Calculate wind direction at this height
	WD = metpy.calc.wind_direction(u = u_Z1.data, v = v_Z1.data)
	# Calculate elevation of theta isentrope upstream
	isen = np.copy(prof_var['theta'][:, Z1, lat_idx, 42].data)
	# Define 40 km transect from peak of orography across ice shelf
	# At each latitude, find the location of the maximum height of orography
	max_alt = np.argmax(prof_var['orog'].data, axis=1)
	# Define a 40 km transect on the Eastern side, i.e. over Larsen, from the peak of orography at that latitude over which to measure Z3
	transect_lons = np.asarray((max_alt, max_alt + 10))
	theta_transect = np.copy(prof_var['theta'][:, :, lat_idx, transect_lons[0, lat_idx]:transect_lons[1, lat_idx]].data)
	foehn_freq = []
	Z2 = []
	for timestep in range(len(isen)):
		# Find timesteps where u >= 2.0 m s-1
		if u_Z1[timestep] > 2.0:
			# Find timesteps where wind direction is cross-peninsula
			if WD.magnitude[timestep] <= 300. and WD.magnitude[timestep] >= 240.:
				# Find the minimum height of the upstream isentrope theta_Z1 in the transect defined, Z2.
				ht = np.isclose( theta_transect[timestep,:,:], isen[timestep], atol=.25, rtol = 0.01)
				hts, lons = np.where(ht == True)
				min_ht = np.min(hts)
				Z2 = np.append(Z2, prof_var['altitude'].points[min_ht,0,0])
			else:
				Z2 = np.append(Z2, np.nan)
		else:
			Z2 = np.append(Z2, np.nan)
		# Find difference between Z1 and Z2
		Z3 = prof_var['altitude'].points[Z1, lat_idx, 42] - Z2
		# If Z3 > 1000 m for 6 hours or more (i.e. two timesteps for 3-hourly data)
		if Z3[timestep] >= 1000.:# + Z3[timestep+1] >= 2000.:
			foehn_freq = np.append(foehn_freq, 1.)
		else:
			foehn_freq = np.append(foehn_freq, 0.)
	return foehn_freq

foehn_freq_2012 = diag_foehn_isentrope(meas_lon = -61.50, meas_lat = -67.01, prof_var = prof_2012)

'''
between each 3-hourly timestep,
if:
1) mean flow is cross-peninsula and westerly (directional and speed conditions)
	2a) Fr > 0.9 (Dan Bannister) is above threshold / OR:
	2b) surface met changes sufficiently
foehn_stats.append(1)
else:
	foehn_stats.append(0)

	Should end up with a list of len-1 time series full of 1s and 0s (i.e. foehn event or not)

	# Calculate total frequency of foehn in time series
	# Calculate mean number of foehn in each year
	# Plot total number of foehn in each year (bar chart)
	# Calculate and plot Mean number of foehn in each season
	# Plot mean number of foehn per month, whole time series

	Correlations between:

	- foehn events + turbulent fluxes (seasonal/overall) - sim. 1st melt chapter
	- foehn events + melt rates (seasonal/overall) - sim. 1st melt chapter
	- foehn events and increased SWdown / decreased cloud cover (cloud clearance)

'''