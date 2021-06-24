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
    filepath = '/gws/nopw/j04/bas_climate/users/ellgil82/hindcast/output/alloutput/ceda_archive/'
elif host == 'bsl':
    filepath = '/data/mac/ellgil82/hindcast/output/'

## Load data
def load_vars(year):
	# Load surface variables
	Tair = iris.load_cube( filepath + year+'_air_temperature_1p5m.nc', 'air_temperature')
	Ts = iris.load_cube( filepath + year+'_surface_temperature.nc', 'surface_temperature')
	MSLP = iris.load_cube( filepath + year+'_air_pressure_at_mean_sea_level.nc', 'air_pressure_at_sea_level')
	#sfc_P = iris.load_cube(filepath  + year + '_air_pressure.nc', 'surface_air_pressure')
	FF_10m = iris.load_cube( filepath +year+'_wind_speed_10m.nc', 'wind_speed')
	RH = iris.load_cube(filepath  + year + '_relative_humidity_1p5m.nc', 'relative_humidity')
	u = iris.load_cube(filepath  + year + '_eastward_wind_10m.nc', 'x wind component (with respect to grid)')
	v = iris.load_cube(filepath  + year + '_northward_wind_10m.nc', 'y wind component (with respect to grid)')
	# Load profiles
	theta_prof = iris.load_cube(filepath + year + '_air_potential_temperature_model_levels.nc')
	theta_prof = theta_prof[:,:40,:,:]
	u_prof = iris.load_cube(filepath + year + '_eastward_wind_model_levels.nc')
	v_prof = iris.load_cube(filepath + year + '_northward_wind_model_levels.nc')
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

MAM_vars = load_vars(year = '1998-2017_MAM')

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

rh_thresh = {'AWS14': -15,
			 'AWS15': -15,
			 'AWS17': -15,
			 'AWS18': -17.5}

def diag_foehn_surf(meas_lat, meas_lon, surf_var, station):
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

def diag_foehn_AWS(AWS_var, station):
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

#sens_test_df()

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
			if u_Z1[timestep + t] > 2.0:
				# If Froude number > 0.9 for 6+ hours, diagnose foehn conditions
				if Fr[timestep + t] >= 0.9 :
					foehn_freq[timestep] =  1.
			else:
				foehn_freq[timestep] =  0.
	return foehn_freq

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
	foehn_freq = len(foehn_df)
	return foehn_freq, foehn_df

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

#foehn_freq, foehn_df = combo_foehn(lon_dict[station_dict[station]],lat_dict[station_dict[station]], prof_2012, surf_2012)#, station = 'AWS18')

def full_srs_foehn():
	# Load surface variables
	Tair = iris.load_cube(filepath + '1998-2017_Tair_1p5m.nc', 'air_temperature')
	# FF_10m = iris.load_cube(filepath + '1998-2017_FF_10m.nc', 'wind_speed')
	RH = iris.load_cube(filepath + '1998-2017_RH_1p5m.nc', 'relative_humidity')
	u_prof = iris.load_cube(filepath + '1998-2017_u_wind_full_profile.nc')
	FF_10m = FF_10m[:, 0, 1:, :]
	RH = RH[:, 0, :, :]
	dT = np.gradient(Tair.data, 2, axis=0)  # find gradient between timesteps with t=2 spacing (i.e. 6 hours)
	dRH = np.gradient(RH.data, 2, axis=0)
	u_Z1 = np.mean(u_prof[:, 27, 80:140, 4:42].data, axis=(1, 2))[
		   :29222] >= 2.  # take mean over area, not just at one point
	u_Z1 = np.repeat(u_Z1, 2)  # [2:-3]
	f = np.reshape(np.tile(u_Z1, (1, 1, 1)), ((u_Z1.shape[0]), 1, 1))
	u_Z1 = np.broadcast_to(f, (dT.shape))
	FI_noFF = dT - dRH
	FI_noFF[u_Z1 == 0] = np.nan
	foehn_cond_noFF_cube = iris.cube.Cube(data = FI_noFF)
	iris.save(foehn_cond_noFF, filepath + 'FI_noFF_calc_grad.nc')
	return foehn_cond, foehn_cond_noFF, dRH, dT, dFF, u_Z1

#foehn_cond, foehn_cond_noFF, dRH, dT, dFF, u_Z1 = full_srs_foehn()

surf_vars, prof_vars = load_vars('2012')

lon_index14, lat_index14, = find_gridbox(-67.01, -61.03, surf_vars['lat'], surf_vars['lon'])
lon_index15, lat_index15, = find_gridbox(-67.34, -62.09, surf_vars['lat'], surf_vars['lon'])
lon_index17, lat_index17, = find_gridbox(-65.93, -61.85, surf_vars['lat'], surf_vars['lon'])
lon_index18, lat_index18, = find_gridbox(-66.48272, -63.37105, surf_vars['lat'], surf_vars['lon'])

lat_dict = {'AWS14': -67.01,
            'AWS15': -67.34,
            'AWS17': -65.93,
            'AWS18': -66.48272}

lon_dict = {'AWS14': -61.03,
            'AWS15': -62.09,
            'AWS17': -61.85,
            'AWS18': -63.37105}

station_dict = {'AWS14_SEB_2009-2017_norp.csv': 'AWS14',
             'AWS15_hourly_2009-2014.csv': 'AWS15',
              'AWS17_SEB_2011-2015_norp.csv': 'AWS17',
                 'AWS18_SEB_2014-2017_norp.csv': 'AWS18'}

def foehn_freq_stats(station, yr_list):
	print('Running annual foehn stats at ' + station)
	yr_stats = pd.DataFrame(index=['Froude'], columns=yr_list)
	for year in yr_list:
		surf_var, prof_var = load_vars(year=year)
		#foehn_freq_isen, foehn_df = diag_foehn_isentrope(lon_dict[station_dict[station]],lat_dict[station_dict[station]], prof_var)
		foehn_freq_froude = diag_foehn_Froude(lon_dict[station_dict[station]],lat_dict[station_dict[station]], prof_var)
		#foehn_freq_surf, surf_df = diag_foehn_surf(lon_dict[station_dict[station]],lat_dict[station_dict[station]], surf_var, station)
		#combo_freq, combo_df = combo_foehn(lon_dict[station_dict[station]],lat_dict[station_dict[station]], prof_var, surf_var)
		#try:
		#	AWS_ANN, AWS_DJF, AWS_MAM, AWS_JJA, AWS_SON = load_AWS(station, year)
		#	foehn_freq_AWS, foehn_df = diag_foehn_AWS(AWS_ANN, station)
		#	yr_stats[year] = pd.Series(data = [foehn_freq_AWS, foehn_freq_surf, np.count_nonzero(foehn_freq_froude), foehn_freq_isen, combo_freq], index = yr_stats.index)
		#except:
		#	print('AWS data not available at ' + station_dict[station] + ' during ' + year)
		#	yr_stats[year] = pd.Series(data= [np.nan, foehn_freq_surf, np.count_nonzero(foehn_freq_froude), foehn_freq_isen,  combo_freq], index = yr_stats.index)
		yr_stats[year] = pd.Series([np.count_nonzero(foehn_freq_froude)], index = yr_stats.index)
		print(yr_stats[year])
	print(yr_stats)
	yr_stats.to_csv(filepath + 'Annual_foehn_frequency_modelled_FROUDE'+station+'.csv')

def seas_foehn(year_list, station):
	print('Running seasonal foehn stats at ' + station_dict[station])
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
			ANN_obs_freq,df = diag_foehn_AWS(AWS_ANN, station)
			DJF_obs_freq,df = diag_foehn_AWS(AWS_DJF, station)
			MAM_obs_freq,df = diag_foehn_AWS(AWS_MAM, station)
			JJA_obs_freq,df = diag_foehn_AWS(AWS_JJA, station)
			SON_obs_freq,df = diag_foehn_AWS(AWS_SON, station)
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
	try:
		seas_foehn_freq_obs.plot.bar()
	except:
		print('can\'t plot no bar chart')
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

#seas_foehn(year_list = ['1998', '1999', '2000','2001', '2002', '2003', '2004', '2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017'], station = 'AWS14_SEB_2009-2017_norp.csv')
#seas_foehn(year_list = ['1998', '1999', '2000','2001', '2002', '2003', '2004', '2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017'], station = 'AWS17_SEB_2011-2015_norp.csv')
#seas_foehn(year_list = ['1998', '1999', '2000','2001', '2002', '2003', '2004', '2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017'], station = 'AWS18_SEB_2014-2017_norp.csv')
#seas_foehn(year_list = ['1998', '1999', '2000','2001', '2002', '2003', '2004', '2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017'], station = 'AWS15_hourly_2009-2014.csv')

#for s in station_dict.keys():#'[ 'AWS15_hourly_2009-2014.csv','AWS17_SEB_2011-2015_norp.csv',  'AWS18_SEB_2014-2017_norp.csv']:
	#seas_foehn(station = s, year_list = ['1998', '1999', '2000','2001', '2002', '2003', '2004', '2005', '2006', '2007',  '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017'])
	#foehn_freq_stats(station = s, yr_list = ['1998', '1999', '2000','2001', '2002', '2003', '2004', '2005', '2006', '2007',  '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017'])

#FI, FI_noFF = full_srs_foehn()

def foehn_freq_bar(station, yr_list):
	foehn_stats = pd.DataFrame(index = ['observed', 'surface method', 'Froude method','isentrope method' ])
	#yr_list = ['1998', '1999', '2000', '2001', '2002', '2003', '2004', '2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017']
	for year in yr_list:
		surf_var, prof_var = load_vars(year=year)
		AWS_SEB, DJF, MAM, JJA, SON = load_AWS(station, year = year)
		foehn_freq_AWS = diag_foehn_AWS(AWS_var = AWS_SEB)
		foehn_freq_isen = diag_foehn_isentrope(meas_lon = lon_dict[station_dict[station]], meas_lat = lat_dict[station_dict[station]], prof_var = prof_var) # lats/lons of AWS 14
		foehn_freq_surf = diag_foehn_surf(meas_lon = lon_dict[station_dict[station]], meas_lat = lat_dict[station_dict[station]], surf_var = surf_var)
		foehn_freq_froude = diag_foehn_Froude(meas_lon = lon_dict[station_dict[station]], meas_lat = lat_dict[station_dict[station]], prof_var = prof_var)
		#foehn_freq_yr = diag_foehn_isentrope(meas_lon=, meas_lat=, prof_var=prof_var)  # lats/lons of AWS 18
		yr_stats = [foehn_freq_AWS, np.count_nonzero(foehn_freq_surf), np.count_nonzero(foehn_freq_froude), np.count_nonzero(foehn_freq_isen)]
		foehn_stats[year] = pd.Series(yr_stats, index = ['observed', 'surface method', 'Froude method','isentrope method' ])
	print(foehn_stats)
	foehn_stats.plot.bar()
	if host == 'bsl':
		plt.savefig('/users/ellgil82/figures/Hindcast/foehn/Annual_foehn_frequency_' + station_dict[station] + '_' + yr_list[0] + '_to_' + yr_list[-1] + '.png')
		plt.savefig('/users/ellgil82/figures/Hindcast/foehn/Annual_foehn_frequency_' + station_dict[station] + '_' + yr_list[0] + '_to_' + yr_list[-1] + '.eps')
	elif host == 'jasmin':
		plt.savefig('/gws/nopw/j04/bas_climate/users/ellgil82/hindcast/figures/Annual_foehn_frequency_' + station_dict[station] + '_' + yr_list[0] + '_to_' + yr_list[-1] + '.png')
		plt.savefig('/gws/nopw/j04/bas_climate/users/ellgil82/hindcast/figures/Annual_foehn_frequency_' + station_dict[station] + '_' + yr_list[0] + '_to_' + yr_list[-1] + '.eps')
	foehn_stats.to_csv(filepath + 'Modelled_foehn_frequency_various_methods_' + station_dict[station] + '_' + year_list[0] + '_to_' + year_list[-1] + '.csv')
	plt.show()

#foehn_freq_bar('AWS14_SEB_2009-2017_norp.csv', yr_list = ['1998', '1999', '2000','2001', '2002', '2003', '2004', '2005', '2006', '2007',  '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017'])
#foehn_freq_bar('AWS17_SEB_2011-2015_norp', yr_list = ['1998', '1999', '2000','2001', '2002', '2003', '2004', '2005', '2006', '2007',  '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017'])
#foehn_freq_bar('iWS18_SEB_hourly_until_nov17.txt', yr_list = ['1998', '1999', '2000','2001', '2002', '2003', '2004', '2005', '2006', '2007',  '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017'])
#foehn_freq_bar('AWS15_hourly_2009-2014.csv', yr_list = ['1998', '1999', '2000','2001', '2002', '2003', '2004', '2005', '2006', '2007',  '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017'])

surf_var, prof_var = load_vars('MetUM_v11p1_Antarctic_Peninsula_4km_19980101-20171231')

# trim for MAM 2016 only
for k in surf_var.keys():
	surf_var[k] = surf_var[k][53072:53807]
for k in prof_var.keys():
	prof_var[k] = prof_var[k][53072:53807]

def spatial_foehn(calc):
	if calc == 'yes':
		dT = np.diff(surf_var['Tair'].data, n=2, axis = 0) >= 0.
		dT3 = np.diff(surf_var['Tair'].data, n=2,axis = 0) >= 3.
		Z1 = np.argmin((prof_var['altitude'][:, 110, 42].points - 2000) ** 2)
		u_Z1 = prof_var['u'][:, Z1, 110, 42].data >= 2.
		u_Z1 = np.repeat(u_Z1[2:-3], 2)
		f = np.reshape(np.tile(u_Z1, (1,1,1)), ((u_Z1.shape[0]),1,1))
		u_Z1 = np.broadcast_to(f, (dT.shape))
		RH10 = surf_var['RH'][:-2].data <= np.quantile(surf_var['RH'][:-2].data, 0.1, axis = 0)
		RH15 = surf_var['RH'][:-2].data <= np.quantile(surf_var['RH'][:-2].data, 0.15, axis = 0)
		dRH = np.diff(surf_var['RH'].data, n=2, axis = 0) <= -15
		all_cond = (((RH10 == 1.) & (dT == 1.) & (u_Z1 == 1.))| ((RH15 == 1.) & (dT3 == 1.) & (u_Z1 == 1.)) | ((dRH == 1.)  & (dT == 1.) & (u_Z1 == 1.) )) # Criteria of Turton et al. (2018) plus wind component
		total_foehn = all_cond.sum(axis = 0)
		foehn_pct = np.ma.masked_where(condition = prof_var['lsm'].data == 0., a= (total_foehn/np.float(len(dT)))*100.)
	else:
		foehn_pct = iris.load_cube(filepath + 'foehn_pct.nc')
		try:
			LSM = iris.load_cube(filepath + 'new_mask.nc')
			orog = iris.load_cube(filepath + 'orog.nc')
			orog = orog[0, 0, :, :]
			lsm = LSM[0, 0, :, :]
			for i in [orog, lsm]:
				real_lon, real_lat = rotate_data(i, np.ndim(i) - 2, np.ndim(i) - 1)
		except iris.exceptions.ConstraintMismatchError:
			print('Files not found')
	# Plot
	fig, ax = plt.subplots(figsize=(8, 8))
	CbAx = fig.add_axes([0.25, 0.18, 0.5, 0.02])
	ax.axis('off')
	Larsen_mask = np.zeros((220,220))
	Larsen_mask[40:135, 90:155] = 1.
	c = ax.pcolormesh(np.ma.masked_where((Larsen_mask == 0.), foehn_pct.data), cmap = 'OrRd', vmin = 3, vmax = 12)
	#c = ax.pcolormesh(np.ma.masked_where((prof_var['orog'].data >= 100.), foehn_pct.data), cmap = 'OrRd', vmin = 3, vmax = 12)  #divide by 20 to get mean annual number of foehn/20.
	cb = plt.colorbar(c, cax = CbAx, orientation = 'horizontal', extend = 'both', ticks = [0,5,10, 15])#cb.solids.set_edgecolor("face")
	cb.outline.set_edgecolor('dimgrey')
	cb.ax.tick_params(which='both', axis='both', labelsize=24, labelcolor='dimgrey', pad=10, size=0, tick1On=False, tick2On=False)
	cb.outline.set_linewidth(2)
	cb.ax.xaxis.set_ticks_position('bottom')
	cb.set_label('Mean foehn occurrence (% of time)', fontsize = 24,  color='dimgrey', labelpad = 20)
	ax.contour(lsm.data, levels = [1], colors = '#222222')
	ax.contour(orog.data, levels = [50], colors = '#222222')
	plt.subplots_adjust( bottom = 0.25, top = 0.95)
	if host == 'bsl':
		plt.savefig('/users/ellgil82/figures/Hindcast/foehn/foehn_occurrence_spatial_composite.png', transparent=True)
		plt.savefig('/users/ellgil82/figures/Hindcast/foehn/foehn_occurrence_spatial_composite.eps', transparent=True)
	elif host == 'jasmin':
		plt.savefig('/gws/nopw/j04/bas_climate/users/ellgil82/hindcast/figures/foehn_occurrence_spatial_composite_surface_criteria.png', transparent=True)
		plt.savefig('/gws/nopw/j04/bas_climate/users/ellgil82/hindcast/figures/foehn_occurrence_spatial_composite_surface_criteria.eps', transparent=True)
	plt.show()
	if calc == 'yes':
		return total_foehn, foehn_pct, all_cond

#total_foehn, foehn, all_cond = spatial_foehn(calc = 'no')
total_foehn, foehn, all_cond = spatial_foehn(calc = 'yes')

fpct = iris.cube.Cube(data = foehn_pct, units = 'percent', long_name = 'foehn occurrence')
iris.save(fpct, filepath + 'MAM_2016_foehn_pct.nc')



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
	# Number of foehn per season - are relationships between melting and foehn occurrence/duration stronger in winter/summer?

	Correlations between:

	- foehn events + turbulent fluxes (seasonal/overall) - sim. 1st melt chapter
	- foehn events + melt rates (seasonal/overall) - sim. 1st melt chapter
	- foehn events and increased SWdown / decreased cloud cover (cloud clearance)


Figure:
	- Bar chart of "ice shelf" stations (AWS 14 and 15) vs. "inlet" stations (AWS 17 and 18) - annual foehn frequency with three methods (bar chart, each method a different colour, for each year)
	- Bar chart of seasonal/annual frequency (each year) for a) inlet stations and b) ice shelf stations using the three methods
	- Bar chart of whole 20 year period - seasonal breakdown plus inlet vs ice shelf (somehow)


'''
