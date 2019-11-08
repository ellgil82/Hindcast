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
elif host == 'bsl':
    filepath = '/data/mac/ellgil82/hindcast/output/'

## Load model data
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
	theta_pp = iris.load_cube(filepath + '*pe000.pp', 'air_potential_temperature')
	theta_pp = theta_pp[:,:40,:,:]
	if host == 'bsl':
		try:
			LSM = iris.load_cube(filepath + 'new_mask.nc', 'LAND MASK (No halo) (LAND=TRUE)')
			orog = iris.load_cube(filepath + 'orog.nc', 'surface_altitude')
			orog = orog[0, 0, :, :]
			LSM = LSM[0, 0, :, :]
		except iris.exceptions.ConstraintMismatchError:
			print('Files not found')
	elif host == 'jasmin':
		try:
			LSM = iris.load_cube(filepath + 'new_mask.nc', 'land_binary_mask')
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

# Load specific year of AWS observatioins
def load_AWS(station, year):
	## --------------------------------------------- SET UP VARIABLES ------------------------------------------------##
	## Load data from AWS 14 and AWS 15 for January 2011
    print('\nDayum grrrl, you got a sweet AWS...')
    if host == 'jasmin':
		os.chdir(filepath)
		for file in os.listdir(filepath):
			if fnmatch.fnmatch(file, '%(station)s*' % locals()):
				AWS_srs = pd.read_csv(str(file), na_values=-9999, header=0)
    elif host == 'bsl':
		os.chdir('/data/clivarm/wip/ellgil82/AWS/')
		for file in os.listdir('/data/clivarm/wip/ellgil82/AWS/'):
			if fnmatch.fnmatch(file, '%(station)s*' % locals()):
				AWS_srs = pd.read_csv(str(file), na_values = -9999, header = 0)
    # Calculate date, given list of years and day of year
    date_list = compose_date(AWS_srs['year'], days=AWS_srs['day'])
    AWS_srs['Date'] = date_list
    # Set date as index
    AWS_srs.index = AWS_srs['Date']
    # Calculate actual time from decimal DOY (seriously, what even IS that format?)
    try:
        AWS_srs['time'] = 24.*(AWS_srs['Time'] - AWS_srs['day'])
        time_list = []
        case = AWS_srs.loc[year + '-01-01':year + '-12-31']  # '2015-01-01':'2015-12-31'
        for i in case['time']:
            hrs = int(i)                 # will now be 1 (hour)
            mins = int((i-hrs)*60)       # will now be 4 minutes
            secs = int(0 - hrs*60*60 + mins*60) # will now be 30
            j = datetime.time(hour = hrs, minute=mins)
            time_list.append(j)
        case['Time'] = time_list
    except TypeError:
        print('Got time already m9')
        AWS_srs['Time'] = pd.to_datetime(AWS_srs['Time'], format='%H:%M:%S').dt.time
        case = AWS_srs.loc[year+'-01-01':year +'-12-31'] #'2015-01-01':'2015-12-31'
    print '\nconverting times...'
    # Convert times so that they can be plotted
    case['datetime'] = case.apply(lambda r : pd.datetime.combine(r['Date'],r['Time']),1)
    try:
        case['E'] = case['LWnet_corr'].values + case['SWnet_corr'].values + case['Hlat'].values + case['Hsen'].values - case['Gs'].values
    except:
        print('No full SEB \'ere pal...')
    case['WD'][case['WD'] < 0.] = np.nan
    case['FF_10m'][case['FF_10m'] < 0.] = np.nan
    case['WD'] = case['WD'].interpolate() # interpolate missing values
    case['FF_10m'] = case['FF_10m'].interpolate()
    if station == 'AWS14_SEB_2009-2017_norp':
        case = case.tail(1).append(case.iloc[:-1])
    # Calculate months
    months = [g for n, g in case.groupby(pd.TimeGrouper('M'))]
    DJF = pd.concat((months[11], months[0], months[1]), axis=0)
    MAM = pd.concat((months[2], months[3], months[4]), axis=0)
    JJA = pd.concat((months[5], months[6], months[7]), axis=0)
    SON = pd.concat((months[8], months[9], months[10]), axis=0)
    if host == 'jasmin':
        os.chdir('/gws/nopw/j04/bas_climate/users/ellgil82/hindcast/output/alloutput/')
    elif host == 'bsl':
        os.chdir('/data/mac/ellgil82/hindcast/output/')
    return case, DJF, MAM, JJA, SON

# Load entire AWS series
def load_all_AWS(station):
	## --------------------------------------------- SET UP VARIABLES ------------------------------------------------##
	## Load data from AWS 14 and AWS 15 for January 2011
    print('\nDayum grrrl, you got a sweet AWS...')
    if host == 'jasmin':
		os.chdir(filepath)
		for file in os.listdir(filepath):
			if fnmatch.fnmatch(file, '%(station)s*' % locals()):
				AWS_srs = pd.read_csv(str(file), na_values=-9999, header=0)
    elif host == 'bsl':
		os.chdir('/data/clivarm/wip/ellgil82/AWS/')
		for file in os.listdir('/data/clivarm/wip/ellgil82/AWS/'):
			if fnmatch.fnmatch(file, '%(station)s*' % locals()):
				AWS_srs = pd.read_csv(str(file), na_values = -9999, header = 0)
    # Calculate date, given list of years and day of year
    date_list = compose_date(AWS_srs['year'], days=AWS_srs['day'])
    AWS_srs['Date'] = date_list
    # Set date as index
    AWS_srs.index = AWS_srs['Date']
    # Calculate actual time from decimal DOY (seriously, what even IS that format?)
    try:
        AWS_srs['time'] = 24.*(AWS_srs['Time'] - AWS_srs['day'])
        time_list = []
        for i in AWS_srs['time']:
            hrs = int(i)                 # will now be 1 (hour)
            mins = int((i-hrs)*60)       # will now be 4 minutes
            secs = int(0 - hrs*60*60 + mins*60) # will now be 30
            j = datetime.time(hour = hrs, minute=mins)
            time_list.append(j)
        AWS_srs['Time'] = time_list
    except TypeError:
        print('Got time already m9')
        AWS_srs['Time'] = pd.to_datetime(AWS_srs['Time'], format='%H:%M:%S').dt.time
    print '\nconverting times...'
    # Convert times so that they can be plotted
    AWS_srs['datetime'] = AWS_srs.apply(lambda r : pd.datetime.combine(r['Date'],r['Time']),1)
    try:
        AWS_srs['E'] = AWS_srs['LWnet_corr'].values + AWS_srs['SWnet_corr'].values + AWS_srs['Hlat'].values + AWS_srs['Hsen'].values - AWS_srs['Gs'].values
    except:
        print('No full SEB \'ere pal...')
    AWS_srs['WD'][AWS_srs['WD'] < 0.] = np.nan
    AWS_srs['FF_10m'][AWS_srs['FF_10m'] < 0.] = np.nan
    AWS_srs['WD'] = AWS_srs['WD'].interpolate() # interpolate missing values
    AWS_srs['FF_10m'] = AWS_srs['FF_10m'].interpolate()
    AWS_srs['WD'][AWS_srs['WD'] == 0.] = np.nan
    if station == 'AWS14_SEB_2009-2017_norp':
        AWS_srs = AWS_srs.tail(1).append(AWS_srs.iloc[:-1])
    # Calculate months
    months = [g for n, g in AWS_srs.groupby(pd.TimeGrouper('M'))]
    DJF = pd.concat((months[11], months[0], months[1]), axis=0)
    MAM = pd.concat((months[2], months[3], months[4]), axis=0)
    JJA = pd.concat((months[5], months[6], months[7]), axis=0)
    SON = pd.concat((months[8], months[9], months[10]), axis=0)
    if host == 'jasmin':
        os.chdir('/gws/nopw/j04/bas_climate/users/ellgil82/hindcast/output/alloutput/')
    elif host == 'bsl':
        os.chdir('/data/mac/ellgil82/hindcast/output/')
    return AWS_srs, DJF, MAM, JJA, SON

# Calculate Froude number at one Rossby radius of deformation away from mountains
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
	N = 0.01 # s-1 = Brunt-Vaisala frequency
	h = 2000 # m = height of AP mountains
	Fr = u_wind/(N*h)
	h_hat = (N*h)/u_wind
	return Fr, h_hat

# Set up AWS-specific relative humidity thresholds for surface-based foehn detection method of Turton et al. (2018)
rh_thresh = {'AWS14': -15,
			 'AWS15': -15,
			 'AWS17': -15,
			 'AWS18': -17.5}

# Diagnose foehn occurrence in model time series using modified surface method of Turton et al. (2018)
def diag_foehn_surf(meas_lat, meas_lon, surf_var):
	''' Diagnose whether foehn conditions are present in data using the surface meteorology method described by Turton et al. (2018).

	Criteria for foehn:

		- surface temperature increase of >= 2 K for 6+ hours

		- wind direction must be cross-peninsula for 6+ hours

		- wind speed must increase for 6+ hours

		- relative humidity must decrease for 6+ hours

	Inputs:

		- meas_lat, meas_lon: latitude and longitude of the location at which you would like to diagnose foehn, typically the location of an AWS.

		- surf_var: dictionary of surface variables for the year of interest, retrieved using the function load_vars.

	Returns:

	    - foehn_freq: a 1D Boolean time series showing whether foehn is occurring or not at the location requested.

	    '''
	# Find gridbox of latitudes and longitudes of AWS/location at which foehn is occurring
	lon_idx, lat_idx = find_gridbox(meas_lat, meas_lon, real_lon=surf_var['lon'], real_lat=surf_var['lat'])
	model_df = pd.DataFrame()
	model_df['FF_10m'] = pd.Series(surf_var['FF_10m'][:, lat_idx, lon_idx].data)
	model_df['WD'] = pd.Series(surf_var['WD'][:, lat_idx, lon_idx].data)
	model_df['RH'] = pd.Series(surf_var['RH'][:, lat_idx, lon_idx].data)
	model_df['Tair'] = pd.Series(surf_var['Tair'][:, lat_idx, lon_idx].data)
	model_df['wind_difs'] = model_df['FF_10m'].diff(periods = 2)
	model_df['RH_difs'] = model_df['RH'].diff(periods=2)
	model_df['T_difs'] = model_df['Tair'].diff(periods=2)
	foehn_df = model_df.loc[((model_df.RH_difs <= rh_thresh[station_dict[station]]) & (model_df.T_difs > 0.)) | ((model_df.RH <= model_df.RH.quantile(q=0.1)) & (model_df.T_difs > 0.)) | ((model_df.RH <= model_df.RH.quantile(q=0.15)) & (model_df.T_difs > 3.))]
	foehn_freq = len(foehn_df)
	return foehn_freq, foehn_df

# Diagnose foehn occurrence in observed time series using modified surface method of Turton et al. (2018)
def diag_foehn_AWS(AWS_var):
	''' Diagnose whether foehn conditions are present in data using the surface meteorology method described by Bannister (2015) and XXX.

	Criteria for foehn:

		- surface temperature increase of >= 2 K for 6+ hours

		- wind direction must be cross-peninsula for 6+ hours

		- wind speed must increase for 6+ hours

		- relative humidity must decrease for 6+ hours

	Inputs:

		- meas_lat, meas_lon: latitude and longitude of the location at which you would like to diagnose foehn, typically the location of an AWS.

		- surf_var: dictionary of surface variables for the year of interest, retrieved using the function load_vars.

	Returns:

	    - foehn_freq: an integer indicating the number of foehn events occurring at the location requested during the period considered.

	    - foehn_df: a pandas dataframe containing information about the events highlighted.

	    '''
	AWS_var['wind_difs'] = AWS_var['FF_10m'].diff(periods=4)
	AWS_var['RH_difs'] = AWS_var['RH'].diff(periods=4)
	AWS_var['T_difs'] = AWS_var['Tair_2m'].diff(periods=4) #
	# Find timesteps where criteria are met
	foehn_df = AWS_var.loc[((AWS_var.RH_difs <= rh_thresh[station_dict[station]]) & (AWS_var.T_difs > 0.)) | ((AWS_var.RH <= AWS_var.RH.quantile(q=0.1)) & (AWS_var.T_difs > 0.)) | ((AWS_var.RH <= AWS_var.RH.quantile(q=0.15)) & (AWS_var.T_difs > 3.))]
	foehn_freq = len(foehn_df)
	return foehn_freq, foehn_df

# Run sensitivity test of thresholds for surface-based method, after Datta et al. (2019), Turton et al. (2018) and Wiesenekker et al. (2018)
def sens_test_AWS(station):
	AWS_var, D, M, J, S = load_all_AWS(station)
	AWS_var['wind_difs'] = AWS_var['FF_10m'].diff(periods=4)
	AWS_var['RH_difs'] = AWS_var['RH'].diff(periods=4)
	AWS_var['T_difs'] = AWS_var['Tair_2m'].diff(periods=4)  #
	# Find timesteps where criteria are met
	Turton_2018 = AWS_var.loc[((AWS_var.RH_difs <= rh_thresh[station_dict[station]]) & (AWS_var.T_difs > 0.)) | ((AWS_var.RH <= AWS_var.RH.quantile(q=0.1)) & (AWS_var.T_difs > 0.)) | ((AWS_var.RH <= AWS_var.RH.quantile(q=0.15)) & (AWS_var.T_difs > 3.))]
	Weisenneker_2018 = AWS_var.loc[((AWS_var.WD <= 360.) & (AWS_var.WD >= 225.) & (AWS_var.FF_10m >= 4.0))]
	Datta_2019 = AWS_var.loc[((AWS_var.WD <= 360.) & (AWS_var.WD >=220.) & (AWS_var.wind_difs >=3.5) & (AWS_var.RH_difs <= -5.) & (AWS_var.T_difs > 1.))]
	tsrs_len = np.float(len(AWS_var))
	return Turton_2018, Weisenneker_2018, Datta_2019, tsrs_len

# Diagnose foehn occurrence in time series using the Froude number method described in Bannister (2015)
def diag_foehn_Froude(meas_lat, meas_lon, prof_var):
	''' Diagnose whether foehn conditions are present in data using the Froude number method described by Bannister (2015).

	Assumptions:

		- One Rossby wave of deformation from the mountain crest = ~150 km (longitude gridbox 42 in the domain used).

		- Assume representative mountain height of 2000 m.

	Criteria for foehn:

		- u at Z1 must exceed 2.0 m s-1

		- wind direction must be cross-peninsula

		- Foehn wind detected if Froude number exceeds 0.9 for 6+ hours

	Inputs:

		- meas_lat, meas_lon: latitude and longitude of the location at which you would like to diagnose foehn, typically the location of an AWS.

		- prof_var: dictionary of profile variables for the year of interest, retrieved using the function load_vars.

	Returns:

	    - foehn_freq: a 1D Boolean time series showing whether foehn is occurring or not at the location requested.

	    '''
	# Find gridbox of latitudes and longitudes of AWS/location at which foehn is occurring
	lon_idx, lat_idx = find_gridbox(meas_lat, meas_lon,  real_lon=prof_var['lon'], real_lat=prof_var['lat'])
	# Find model level closest to 2000 m
	Z1 = np.argmin((prof_var['altitude'][:, lat_idx, 42].points - 2000) ** 2)
	# Find representative u wind upstream of mountains by at least one Rossby wave of deformation and above the mountain crest
	u_Z1 = np.mean(prof_var['u'][:,7:Z1, lat_idx, 42].data, axis =1) # take mean of flow between ~200 m - Z1 (levels 7-Z1) as per Elvidge et al. (2015)
	v_Z1 = np.mean(prof_var['v'][:,7:Z1, lat_idx, 42].data, axis = 1)
	# Calculate wind direction at this height
	WD = metpy.calc.wind_direction(u = u_Z1, v = v_Z1)
	# Calculate Froude number of upstream flow
	Fr, h_hat = Froude_number(u_Z1)
	foehn_freq = np.zeros(prof_var['u'][:,0,lat_idx, lon_idx].data.shape)
	for timestep in range(len(foehn_freq)-2):
		# If representative upstream wind direction is cross-peninsula
		for t in range(3):
			if WD.magnitude[timestep+t] <= 300. and WD.magnitude[timestep+t] >= 240.:
				# If Froude number > 0.9 for 6+ hours, diagnose foehn conditions
				if Fr[timestep] >= 0.9 and Fr[timestep+1] >= 0.9 and Fr[timestep+2] >= 0.9:
					foehn_freq[timestep] =  1.
			else:
				foehn_freq[timestep] =  0.
	return foehn_freq

# Diagnose foehn occurrence in time series using the isentrope method of Bannister (2015), King et al. (2017) and Turton et al. (2018)
def diag_foehn_isentrope(meas_lat, meas_lon, prof_var):
	''' Diagnose whether foehn conditions are present in data using the isentrope method described by Bannister (2015) and King et al. (2017).

	Assumptions:

		- One Rossby wave of deformation from the mountain crest = ~150 km (longitude gridbox 42 in the domain used).

		- Assume representative mountain height of 2000 m.

	Criteria for foehn:

		- u at Z1 must exceed 2.0 m s-1

		- wind direction must be cross-peninsula

		- Difference between height Z1 and the height of the Z1 isentrope in the transect defined in lee of the barrier (Z2), i.e. Z3 = Z1-Z2, must exceed 1000 m over 6+ hours.

	Inputs:

		- meas_lat, meas_lon: latitude and longitude of the location at which you would like to diagnose foehn, typically the location of an AWS.

		- prof_var: dictionary of profile variables for the year of interest, retrieved using the function load_vars.

	Returns:

	    - foehn_freq: a 1D Boolean time series showing whether foehn is occurring or not at the location requested.

	    '''
	# Find gridbox of latitudes and longitudes of AWS/location at which foehn is occurring
	lon_idx, lat_idx = find_gridbox(y = meas_lat, x = meas_lon, real_lat=prof_var['lat'],  real_lon=prof_var['lon'])
	# Find model level closest to 2000 m
	Z1 = np.argmin((prof_var['altitude'][:, lat_idx, 42].points - 2000) ** 2)
	# Find representative u wind upstream of mountains by at least one Rossby wave of deformation and above the mountain crest
	u_Z1 = np.mean(prof_var['u'][:, 7:Z1, lat_idx, 42].data, axis=1)
	# Calculate elevation of theta isentrope upstream
	isen = np.copy(prof_var['theta'][:, Z1, lat_idx, 42].data)
	# Define 40 km transect from peak of orography across ice shelf
	# At each latitude, find the location of the maximum height of orography
	max_alt = np.argmax(prof_var['orog'].data, axis = 1)
	# Define a 40 km transect on the Eastern side, i.e. over Larsen, from the peak of orography at that latitude over which to measure Z3
	transect_lons = np.asarray((max_alt, max_alt + 22))
	theta_transect = np.copy(prof_var['theta'][:, :, lat_idx, transect_lons[0, lat_idx]:transect_lons[1, lat_idx]].data)#:130
	foehn_freq = np.zeros(prof_var['u'][:,0,lat_idx, lon_idx].data.shape)
	Z2 = np.zeros(prof_var['u'][:,0,lat_idx, lon_idx].data.shape)
	Z3 = np.zeros(prof_var['u'][:,0,lat_idx, lon_idx].data.shape)
	for timestep in range(len(foehn_freq)-2):
		for t in range(3):
		# Find timesteps where u >= 2.0 m s-1
			if u_Z1[timestep+t] > 2.0:
				# Find the minimum height of the upstream isentrope theta_Z1 in the transect defined, Z2.
				try:
					hts, lons = np.where(theta_transect[timestep] == isen[timestep]) # try this method
					min_ht = np.min(hts)
					Z2[timestep] = prof_var['altitude'].points[min_ht,lat_idx,lon_idx]
				except ValueError:
					Z2[timestep] = np.nan
			else:
				Z2[timestep] = np.nan
		# Find difference between Z1 and Z2
		Z3[timestep] = prof_var['altitude'].points[Z1, lat_idx, 42] - Z2[timestep]
	model_df = pd.DataFrame()
	model_df['Z3'] = pd.Series(np.repeat(Z3,2))
	# If Z3 > 1000 m for 6 hours or more (two instantaneous timesteps for 6-hourly data = at least 6 hours) thresholds: FF = 1.0, T = 2.0, RH = -5
	foehn_df = model_df.loc[(model_df.Z3 >= 470.)]
	return foehn_freq, foehn_df

# Diagnose foehn occurrence in time series using combined isentrope method with surface stipulations, after Bannister (2015), King et al. (2017) and Turton et al. (2018)
def combo_foehn(meas_lat, meas_lon, prof_var, surf_var):
	''' Diagnose whether foehn conditions are present in data using the isentrope method described by Bannister (2015) and King et al. (2017).

	Assumptions:

		- One Rossby wave of deformation from the mountain crest = ~150 km (longitude gridbox 42 in the domain used).

		- Assume representative mountain height of 2000 m.

	Criteria for foehn:

		- u at Z1 must exceed 2.0 m s-1 (i.e. wind must have a cross-peninsula component).

		- Difference between height Z1 and the height of the Z1 isentrope in the transect defined in lee of the barrier (Z2), i.e. Z3 = Z1-Z2, must exceed 500 m over 6+ hours.

		- Surface warming (dT > 0.) must be observed.

		- Surface drying (dRH < 0.) must be observed.

	Inputs:

		- meas_lat, meas_lon: latitude and longitude of the location at which you would like to diagnose foehn, typically the location of an AWS.

		- prof_var: dictionary of profile variables for the year of interest, retrieved using the function load_vars.

	Returns:

	    - foehn_freq: an integer counting the number of timesteps at which foehn occurs at the location requested.

	    - foehn_df: a pandas DataFrame containing information about each timestep diagnosed as being foehn conditions.

	    '''
	# Find gridbox of latitudes and longitudes of AWS/location at which foehn is occurring
	lon_idx, lat_idx = find_gridbox(y = meas_lat, x = meas_lon, real_lat=prof_var['lat'],  real_lon=prof_var['lon'])
	# Find model level closest to 2000 m
	Z1 = np.argmin((prof_var['altitude'][:, lat_idx, 42].points - 2000) ** 2)
	# Find representative u wind upstream of mountains by at least one Rossby wave of deformation and above the mountain crest
	u_Z1 = np.mean(prof_var['u'][:, 7:Z1, lat_idx, 42].data, axis=1)
	# Calculate elevation of theta isentrope upstream
	isen = np.copy(prof_var['theta'][:, Z1, lat_idx, 42].data)
	# Define 40 km transect from peak of orography across ice shelf
	# At each latitude, find the location of the maximum height of orography
	max_alt = np.argmax(prof_var['orog'].data, axis = 1)
	# Define a 40 km transect on the Eastern side, i.e. over Larsen, from the peak of orography at that latitude over which to measure Z3
	transect_lons = np.asarray((max_alt, max_alt + 22))
	theta_transect = np.copy(prof_var['theta'][:, :, lat_idx, transect_lons[0, lat_idx]:transect_lons[1, lat_idx]].data)#:130
	foehn_freq = np.zeros(prof_var['u'][:,0,lat_idx, lon_idx].data.shape)
	Z2 = np.zeros(prof_var['u'][:,0,lat_idx, lon_idx].data.shape)
	Z3 = np.zeros(prof_var['u'][:,0,lat_idx, lon_idx].data.shape)
	for timestep in range(len(foehn_freq)-2):
		for t in range(3):
		# Find timesteps where u >= 2.0 m s-1
			if u_Z1[timestep+t] > 2.0:
				# Find the minimum height of the upstream isentrope theta_Z1 in the transect defined, Z2.
				try:
					hts, lons = np.where(theta_transect[timestep] == isen[timestep]) # try this method
					min_ht = np.min(hts)
					Z2[timestep] = prof_var['altitude'].points[min_ht,lat_idx,lon_idx]
				except ValueError:
					Z2[timestep] = np.nan
			else:
				Z2[timestep] = np.nan
		# Find difference between Z1 and Z2
		Z3[timestep] = prof_var['altitude'].points[Z1, lat_idx, 42] - Z2[timestep]
	model_df = pd.DataFrame()
	model_df['FF_10m'] = pd.Series(surf_var['FF_10m'][:, lat_idx, lon_idx].data)
	model_df['WD'] = pd.Series(surf_var['WD'][:, lat_idx, lon_idx].data)
	model_df['RH'] = pd.Series(surf_var['RH'][:, lat_idx, lon_idx].data)
	model_df['Tair'] = pd.Series(surf_var['Tair'][:, lat_idx, lon_idx].data)
	model_df['Z3'] = pd.Series(np.repeat(Z3,2)) # upsample to be compatible with 3-hourly data
	model_df['wind_difs'] = model_df['FF_10m'].diff(periods=2)
	model_df['RH_difs'] = model_df['RH'].diff(periods=2)
	model_df['T_difs'] = model_df['Tair'].diff(periods=2)
	# If Z3 > 1000 m for 6 hours or more (two instantaneous timesteps for 6-hourly data = at least 6 hours) thresholds: FF = 1.0, T = 2.0, RH = -5
	#foehn_df = model_df.loc[((model_df.RH_difs <= rh_thresh[station]) & (model_df.T_difs > 0.) & (model_df.Z3 >= 470.)) | ((model_df.RH <= model_df.RH.quantile(q=0.1)) & (model_df.T_difs > 0.) & (model_df.Z3 >= 470.)) | ((model_df.RH <= model_df.RH.quantile(q=0.15)) & (model_df.T_difs > 3.) & (model_df.Z3 >= 470.))]
	foehn_df = model_df.loc[((model_df.RH_difs < 0)  & (model_df.T_difs > 0.) & (model_df.Z3 >= 470.))]
	foehn_freq = len(foehn_df)
	return foehn_freq, foehn_df

# Run sensitivity test of each method to diagnose foehn occurrence in model data and create csv file with output
def sens_test_df():
	df = pd.DataFrame(index = ['Turton et al. (2018)', 'Weisenneker et al. (2018)', 'Datta et al. (2019)'])
	for station in station_dict.keys():
		try:
			Turton_2018, Weisenneker_2018, Datta_2019, tsrs_len = sens_test_AWS(station = station)
			print('Turton (2018) method: ' + str(len(Turton_2018)))
			print('\n\nWeisenekker (2018) method: ' + str(len(Weisenneker_2018)))
			print('\n\nDatta (2019) method: ' + str(len(Datta_2019)))
			df[station_dict[station]] = pd.Series([(len(Turton_2018)/tsrs_len), (len(Weisenneker_2018)/tsrs_len), (len(Datta_2019)/tsrs_len)], index = df.index)
		except:
			print('Sorry, no can do that year')
	try:
		print(df)
		df.to_csv(filepath + 'surface_method_sensitivity_test_all_stations.csv')
	except:
		print('AAAAAH')

# Create csv files of annual foehn frequency computed using each method
def foehn_freq_stats(station, yr_list):
	yr_stats = pd.DataFrame(index=['obs', 'surf', 'Froude', 'isen', 'combo'], columns=yr_list)
	for year in yr_list:
		surf_var, prof_var = load_vars(year=year)
		foehn_freq_isen, foehn_df = diag_foehn_isentrope(lon_dict[station_dict[station]],lat_dict[station_dict[station]], prof_var)
		foehn_freq_froude = diag_foehn_Froude(lon_dict[station_dict[station]],lat_dict[station_dict[station]], prof_var)
		foehn_freq_surf, surf_df = diag_foehn_surf(lon_dict[station_dict[station]],lat_dict[station_dict[station]], surf_var)
		combo_freq, combo_df = combo_foehn(lon_dict[station_dict[station]],lat_dict[station_dict[station]], prof_var, surf_var)
		try:
			AWS_ANN, AWS_DJF, AWS_MAM, AWS_JJA, AWS_SON = load_AWS(station, year)
			foehn_freq_AWS, foehn_df = diag_foehn_AWS(AWS_ANN)
			yr_stats[year] = pd.Series(data = [foehn_freq_AWS, foehn_freq_surf, np.count_nonzero(foehn_freq_froude), np.count_nonzero(foehn_freq_isen) + combo_freq], index = yr_stats.index)
		except:
			print('AWS data not available at ' + station_dict[station] + ' during ' + year)
			yr_stats[year] = pd.Series(data= [np.nan, foehn_freq_surf, np.count_nonzero(foehn_freq_froude), foehn_freq_isen + combo_freq], index = yr_stats.index)
		print(yr_stats[year])
	print(yr_stats)
	yr_stats.to_csv(filepath + 'Annual_foehn_frequency_modelled_'+station+'.csv')

# Create csv files of seasonal foehn statistics
def seas_foehn(year_list, station):
	seas_foehn_freq_mod = pd.DataFrame(index =['DJF', 'MAM', 'JJA', 'SON', 'ANN'], columns = year_list)
	seas_foehn_freq_obs = pd.DataFrame(index =['DJF', 'MAM', 'JJA', 'SON', 'ANN'], columns = year_list)
	seas_foehn_freq_bias = pd.DataFrame(index =['DJF', 'MAM', 'JJA', 'SON', 'ANN'], columns = year_list)
	for year in year_list:
		print('Loading model data from ' + year)
		ANN_surf, ANN_prof = load_vars(year)
		DJF_surf, DJF_prof = load_vars('DJF_'+year)
		MAM_surf, MAM_prof = load_vars('MAM_'+year)
		JJA_surf, JJA_prof = load_vars('JJA_'+year)
		SON_surf, SON_prof = load_vars('SON_'+year)
		# diagnose modelled number of foehn
		ANN_freq,df = combo_foehn(lon_dict[station_dict[station]],lat_dict[station_dict[station]], ANN_prof, ANN_surf)
		DJF_freq,df = combo_foehn(lon_dict[station_dict[station]],lat_dict[station_dict[station]], DJF_prof, DJF_surf)
		MAM_freq,df = combo_foehn(lon_dict[station_dict[station]],lat_dict[station_dict[station]], MAM_prof, MAM_surf)
		JJA_freq,df = combo_foehn(lon_dict[station_dict[station]],lat_dict[station_dict[station]], JJA_prof, JJA_surf)
		SON_freq,df = combo_foehn(lon_dict[station_dict[station]],lat_dict[station_dict[station]], SON_prof, SON_surf)
		model_stats = [DJF_freq, MAM_freq, JJA_freq, SON_freq, ANN_freq]
		seas_foehn_freq_mod[year] = pd.Series(model_stats, index=['DJF', 'MAM', 'JJA', 'SON', 'ANN'])
		# Import AWS data if available
		try:
			AWS_ANN, AWS_DJF, AWS_MAM, AWS_JJA, AWS_SON = load_AWS(station, year)
			success = 'yes'
		except:
			print('AWS data not available at ' + station_dict[station] + ' during ' + year)
			success = 'no'
			# Diagnose observed foehn frequency if available
		if success == 'yes':
			ANN_obs_freq,df = diag_foehn_AWS(AWS_ANN)
			DJF_obs_freq,df = diag_foehn_AWS(AWS_DJF)
			MAM_obs_freq,df = diag_foehn_AWS(AWS_MAM)
			JJA_obs_freq,df = diag_foehn_AWS(AWS_JJA)
			SON_obs_freq,df = diag_foehn_AWS(AWS_SON)
			# obs_stats.append([np.count_nonzero(DJF_obs_freq), np.count_nonzero(MAM_obs_freq), np.count_nonzero(JJA_obs_freq), np.count_nonzero(SON_obs_freq), np.count_nonzero(ANN_obs_freq)])
			obs = [DJF_obs_freq,MAM_obs_freq, JJA_obs_freq, SON_obs_freq, ANN_obs_freq]
			seas_foehn_freq_obs[year] = pd.Series(obs, index = ['DJF', 'MAM', 'JJA', 'SON', 'ANN'])
			biases = seas_foehn_freq_mod[year] - seas_foehn_freq_obs[year]
			seas_foehn_freq_bias[year] = pd.Series(biases)
		elif success == 'no':
			seas_foehn_freq_obs[year] = pd.Series([np.nan, np.nan, np.nan, np.nan, np.nan], index = ['DJF', 'MAM', 'JJA', 'SON', 'ANN'])
			seas_foehn_freq_bias[year] = pd.Series([np.nan, np.nan, np.nan, np.nan, np.nan], index=['DJF', 'MAM', 'JJA', 'SON', 'ANN'])
	#seas_foehn_freq_bias = pd.DataFrame(seas_bias, columns = ['DJF', 'MAM', 'JJA', 'SON', 'ANN'], index = year_list)
	#seas_foehn_freq_obs = pd.DataFrame(obs_stats, columns = ['DJF', 'MAM', 'JJA', 'SON', 'ANN'], index = year_list)
	seas_foehn_freq_obs.plot.bar()
	if host == 'bsl':
		plt.savefig('/users/ellgil82/figures/Hindcast/foehn/Observed_seasonal_foehn_frequency_' + station_dict[station] + '_' + year_list[0] + '_to_' + year_list[-1] + '.png')
	elif host == 'jasmin':
		plt.savefig(filepath + 'Observed_seasonal_foehn_frequency_' + station_dict[station] + '_' + year_list[0] + '_to_' + year_list[-1] + '.eps')
		plt.savefig(filepath + 'Observed_seasonal_foehn_frequency_' + station_dict[station] + '_' + year_list[0] + '_to_' + year_list[-1] + '.png')
	seas_foehn_freq_obs.to_csv(filepath + 'Observed_seasonal_foehn_frequency_' + station_dict[station] + '_' + year_list[0] + '_to_' + year_list[-1] + '.csv')
	seas_foehn_freq_bias.to_csv(filepath + 'Bias_seasonal_foehn_frequency_' + station_dict[station] + '_' + year_list[0] + '_to_' + year_list[-1] + '.csv')
	print(seas_foehn_freq_obs)
	print(seas_foehn_freq_bias)
	print(seas_foehn_freq_mod)
	seas_foehn_freq_mod.plot.bar()
	if host == 'bsl':
		plt.savefig('/users/ellgil82/figures/Hindcast/foehn/Seasonal_foehn_frequency_' + station_dict[station] + '_' + year_list[0] + '_to_' + year_list[-1] + '.png')
	elif host == 'jasmin':
		plt.savefig(filepath + 'Seasonal_foehn_frequency_' + station_dict[station] + '_' + year_list[0] + '_to_' + year_list[-1] + '.eps')
		plt.savefig(filepath + 'Seasonal_foehn_frequency_' + station_dict[station] + '_' + year_list[0] + '_to_' + year_list[-1] + '.png')
	seas_foehn_freq_mod.to_csv(filepath + 'Modelled_seasonal_foehn_frequency_' + station_dict[station] + '_' + year_list[0] + '_to_' + year_list[-1] + '.csv')
	plt.show()

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