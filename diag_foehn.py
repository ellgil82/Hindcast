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

surf_2012, prof_2012 = load_vars('2012')
surf_2016, prof_2016 = load_vars('2016')


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


AWS14_SEB, DJF_14, MAM_14, JJA_14, SON_14 = load_AWS('AWS14_SEB_2009-2017_norp.csv', '2016')
AWS15_SEB, DJF_15, MAM_15, JJA_15, SON_15 = load_AWS('AWS15_hourly_2009-2014.csv', '2012')
AWS17_SEB, DJF_17, MAM_17, JJA_17, SON_17 = load_AWS('AWS17_SEB_2011-2015_norp.csv', '2012')
AWS18_SEB, DJF_18, MAM_18, JJA_18, SON_18 = load_AWS('AWS18_SEB_2014-2017_norp.csv', '2016')

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

def diag_foehn_surf(meas_lat, meas_lon, surf_var):
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

	    - foehn_freq: a 1D Boolean time series showing whether foehn is occurring or not at the location requested.

	    '''
	# Find gridbox of latitudes and longitudes of AWS/location at which foehn is occurring
	lon_idx, lat_idx = find_gridbox(meas_lat, meas_lon, real_lon=surf_var['lon'], real_lat=surf_var['lat'])
	foehn_freq = np.zeros(surf_var['FF_10m'][:,lat_idx, lon_idx].data.shape)
	for timestep in range(len(foehn_freq)-2):
		# Find timesteps where wind direction is cross-peninsula
		for t in range(2):
			if surf_var['WD'][timestep + t, lat_idx, lon_idx].data <= 300. and surf_var['WD'][timestep + t, lat_idx, lon_idx].data >= 240.:
				#foehn_freq[timestep] = 1.
				# If wind speed increases by 2.0 m s-1 over 6 hours
				if (surf_var['FF_10m'][timestep+2, lat_idx, lon_idx].data - surf_var['FF_10m'][timestep, lat_idx, lon_idx].data) >= 2.0:
					#foehn_freq[timestep] = 1.
					# If relative humidity decreases over 6 hours
					if (surf_var['RH'][timestep+2, lat_idx, lon_idx].data - surf_var['RH'][timestep, lat_idx, lon_idx].data) <=-5.:
						#foehn_freq[timestep] = 1.
						# If temperature decreases by 2 K over 6 hours
						if (surf_var['Tair'][timestep+2, lat_idx, lon_idx] - surf_var['Tair'] [timestep, lat_idx, lon_idx]) >= 2.0:
							foehn_freq[timestep] = 1.
			else:
				foehn_freq[timestep] = 0.
	#plt.plot(foehn_freq)
	#plt.show()
	return foehn_freq

##AWS 18 coords
meas_lon = -63.37105
meas_lat = -66.48272

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

	    - foehn_freq: a 1D Boolean time series showing whether foehn is occurring or not at the location requested.

	    '''
	# Find gridbox of latitudes and longitudes of AWS/location at which foehn is occurring
	foehn_freq = np.zeros(AWS_var['FF_10m'].shape)
	for timestep in range(len(foehn_freq)-4):
		# Find timesteps where wind direction is cross-peninsula
		for t in range(5):
			if (AWS_var['WD'][timestep + t] <= 300. and AWS_var['WD'][timestep + t] >= 240.):
				# If wind speed increases by 2.0 m s-1 over 6 hours
				if AWS_var['FF_10m'][timestep+4] - AWS_var['FF_10m'][timestep] >= 2.0:
					# If relative humidity decreases over 6 hours
					if AWS_var['RH'][timestep+4] - AWS_var['RH'][timestep] <= -5.:
						# If temperature decreases by 2 K over 6 hours
						if AWS_var['Tair_2m'][timestep+4] - AWS_var['Tair_2m'] [timestep] >= 2.0:
							foehn_freq[timestep] = 1.
			else:
				foehn_freq[timestep] = 0.
	#plt.plot(foehn_freq)
	#plt.show()
	return foehn_freq

def diag_foehn_Froude(meas_lat, meas_lon, prof_var):
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
	lon_idx, lat_idx = find_gridbox(meas_lat, meas_lon,  real_lon=prof_var['lon'], real_lat=prof_var['lat'])
	# Find model level closest to 2000 m
	Z1 = np.argmin((prof_var['altitude'][:, lat_idx, 42].points - 2000) ** 2)
	# Find representative u wind upstream of mountains by at least one Rossby wave of deformation and above the mountain crest
	u_Z1 = np.mean(prof_var['u'][:,7:Z1, lat_idx, 42].data, axis =1)
	v_Z1 = np.mean(prof_var['v'][:,7:Z1, lat_idx, 42].data, axis = 1)
	# Calculate wind direction at this height
	WD = metpy.calc.wind_direction(u = u_Z1, v = v_Z1)
	# Calculate Froude number of upstream flow
	Fr, h_hat = Froude_number(u_Z1) # will this work on an array?
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
	lon_idx, lat_idx = find_gridbox(meas_lat, meas_lon, real_lon=prof_var['lon'], real_lat=prof_var['lat'])
	# Find model level closest to 2000 m
	Z1 = np.argmin((prof_var['altitude'][:, lat_idx, 42].points - 2000) ** 2)
	# Find representative u wind upstream of mountains by at least one Rossby wave of deformation and above the mountain crest
	u_Z1 = np.mean(prof_var['u'][:, 7:Z1, lat_idx, 42].data, axis=1)
	v_Z1 = np.mean(prof_var['v'][:, 7:Z1, lat_idx, 42].data, axis=1)
	# Calculate wind direction at this height
	WD = metpy.calc.wind_direction(u = u_Z1, v = v_Z1)
	# Calculate elevation of theta isentrope upstream
	isen = np.copy(prof_var['theta'][:, Z1, lat_idx, 42].data)
	# Define 40 km transect from peak of orography across ice shelf
	# At each latitude, find the location of the maximum height of orography
	max_alt = np.argmax(prof_var['orog'].data, axis = 1)
	# Define a 40 km transect on the Eastern side, i.e. over Larsen, from the peak of orography at that latitude over which to measure Z3
	transect_lons = np.asarray((max_alt, max_alt + 10))
	theta_transect = np.copy(prof_var['theta'][:, :, lat_idx, transect_lons[0, lat_idx]:transect_lons[1, lat_idx]].data)
	foehn_freq = np.zeros(prof_var['u'][:,0,lat_idx, lon_idx].data.shape)
	Z2 = np.zeros(prof_var['u'][:,0,lat_idx, lon_idx].data.shape)
	Z3 = np.zeros(prof_var['u'][:,0,lat_idx, lon_idx].data.shape)
	for timestep in range(len(foehn_freq)-2):
		for t in range(3):
		# Find timesteps where u >= 2.0 m s-1
			if u_Z1[timestep+t] > 2.0:
			# Find timesteps where wind direction is cross-peninsula
				if WD.magnitude[timestep+t] <= 300. and WD.magnitude[timestep+t] >= 240.:
					# Find the minimum height of the upstream isentrope theta_Z1 in the transect defined, Z2.
					try:
						hts, lons = np.where(theta_transect[timestep] == isen[timestep]) # try this method
							#np.isclose( theta_transect[timestep,:,:], isen[timestep], atol=0.00001, rtol = 0.00001)
						#hts, lons = np.where(ht == True)
						min_ht = np.min(hts)
						Z2[timestep] = prof_var['altitude'].points[min_ht,lat_idx,lon_idx]
					except ValueError:
						Z2[timestep] = np.nan
				else:
					Z2[timestep] = np.nan
			else:
				Z2[timestep] = np.nan
		# Find difference between Z1 and Z2
		Z3[timestep] = prof_var['altitude'].points[Z1, lat_idx, 42] - Z2[timestep]
		# If Z3 > 1000 m for 6 hours or more (two instantaneous timesteps for 6-hourly data = at least 6 hours)
		if Z3[timestep] >= 1000. and Z3[timestep-1] >= 1000. and Z3[timestep-2] >= 1000.:
			foehn_freq[timestep] = 1.
		else:
			foehn_freq[timestep] = 0.
	return foehn_freq

foehn_freq_AWS18 = diag_foehn_AWS(AWS_var = AWS18_SEB)
foehn_freq_isen_2016 = diag_foehn_isentrope(meas_lon = -63.37105, meas_lat = -66.48272, prof_var = prof_2016)
foehn_freq_surf_2016 = diag_foehn_surf(meas_lon = -63.37105, meas_lat = -66.48272, surf_var = surf_2016)
foehn_freq_froude_2016 = diag_foehn_Froude(meas_lon = -63.37105, meas_lat = -66.48272, prof_var = prof_2016)

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

def foehn_freq_stats(station, yr_list):
    foehn_stats = pd.DataFrame(index = ['DJF', 'MAM', 'JJA', 'SON', 'ANN'])
    for year in yr_list:
        surf_var, prof_var = load_vars(year=year)
        foehn_freq_isen = diag_foehn_isentrope(meas_lon=lon_dict[station_dict[station]], meas_lat=lat_dict[station_dict[station]], prof_var=prof_var)
        yr_stats = [np.count_nonzero(foehn_freq_AWS), np.count_nonzero(foehn_freq_surf), np.count_nonzero(foehn_freq_froude), np.count_nonzero(foehn_freq_isen) ]
        foehn_stats[year] = pd.Series(yr_stats)

#foehn_freq_stats(station = 'AWS14_SEB_2009-2017_norp.csv', yr_list = ['2016'])

def seas_foehn(year_list, station):
    seas_foehn_freq_mod = pd.DataFrame(index=['DJF', 'MAM', 'JJA', 'SON', 'ANN'])
    seas_foehn_freq_obs = pd.DataFrame(index=['DJF', 'MAM', 'JJA', 'SON', 'ANN'])
    seas_foehn_freq_bias = pd.DataFrame(index=['DJF', 'MAM', 'JJA', 'SON', 'ANN'])
    for year in year_list:
        print('Loading model data from ' + year)
        ANN_surf, ANN_prof = load_vars(year)
        DJF_surf, DJF_prof = load_vars('DJF_'+year)
        MAM_surf, MAM_prof = load_vars('MAM_'+year)
        JJA_surf, JJA_prof = load_vars('JJA_'+year)
        SON_surf, SON_prof = load_vars('SON_'+year)
        # Import AWS data if available
        try:
            AWS_ANN, AWS_DJF, AWS_MAM, AWS_JJA, AWS_SON = load_AWS(station, year)
        except:
            print('AWS data not available at ' + station_dict[station] + ' during ' + year)
		# diagnose modelled number of foehn
        ANN_freq = diag_foehn_isentrope(meas_lon=lon_dict[station_dict[station]],
                                        meas_lat=lat_dict[station_dict[station]], prof_var=ANN_prof)
        DJF_freq = diag_foehn_isentrope(meas_lon=lon_dict[station_dict[station]],
                                        meas_lat=lat_dict[station_dict[station]], prof_var=DJF_prof)
        MAM_freq = diag_foehn_isentrope(meas_lon=lon_dict[station_dict[station]],
                                        meas_lat=lat_dict[station_dict[station]], prof_var=MAM_prof)
        JJA_freq = diag_foehn_isentrope(meas_lon=lon_dict[station_dict[station]],
                                        meas_lat=lat_dict[station_dict[station]], prof_var=JJA_prof)
        SON_freq = diag_foehn_isentrope(meas_lon=lon_dict[station_dict[station]],
                                        meas_lat=lat_dict[station_dict[station]], prof_var=SON_prof)
        # Diagnose observed foehn frequency if available
        try:
            ANN_obs_freq = diag_foehn_AWS(ANN_freq)
            DJF_obs_freq = diag_foehn_AWS(DJF_freq)
            MAM_obs_freq = diag_foehn_AWS(MAM_freq)
            JJA_obs_freq = diag_foehn_AWS(JJA_freq)
            SON_obs_freq = diag_foehn_AWS(SON_freq)
        except:
            print('I already told you, AWS data are not available at ' + station_dict[station] + ' during ' + year)
		# add seasonal foehn count to
        model_stats = [np.count_nonzero(DJF_freq), np.count_nonzero(MAM_freq), np.count_nonzero(JJA_freq), np.count_nonzero(SON_freq), np.count_nonzero(ANN_freq)]
        try:
            obs_stats = [np.count_nonzero(DJF_obs_freq), np.count_nonzero(MAM_obs_freq), np.count_nonzero(JJA_obs_freq), np.count_nonzero(SON_obs_freq), np.count_nonzero(ANN_obs_freq)]
            seas_bias = model_stats-obs_stats
            seas_foehn_freq_obs[year] = pd.Series(obs_stats, index = ['DJF', 'MAM', 'JJA', 'SON', 'ANN'] )
            seas_foehn_freq_bias[year] = pd.Series(seas_bias, index = ['DJF', 'MAM', 'JJA', 'SON', 'ANN'] )
        except:
            print('OMG bro, do you even listen??? AWS data are not available at ' + station_dict[station] + ' during ' + year)
        seas_foehn_freq_mod[year] = pd.Series(model_stats, index = ['DJF', 'MAM', 'JJA', 'SON', 'ANN'] )
    print(seas_foehn_freq_mod)
    print(seas_foehn_freq_obs)
    print(seas_foehn_freq_bias)
    seas_foehn_freq_mod.plot.bar()
    seas_foehn_freq_obs.plot.bar()
    if host == 'bsl':
        plt.savefig('/users/ellgil82/figures/Hindcast/foehn/Seasonal_foehn_frequency_' + station_dict[station] + '_' + year_list[0] + '_to_' + year_list[-1] + '.png')
    elif host == 'jasmin':
		plt.savefig(filepath + 'Seasonal_foehn_frequency_' + station_dict[station] + '_' + year_list[0] + '_to_' + year_list[-1] + '.eps')
		plt.savefig(filepath + 'Seasonal_foehn_frequency_' + station_dict[station] + '_' + year_list[0] + '_to_' + year_list[-1] + '.png')
    seas_foehn_freq_mod.to_csv(filepath + 'Modelled_seasonal_foehn_frequency_' + station_dict[station] + '_' + year_list[0] + '_to_' + year_list[-1] + '.csv')
    seas_foehn_freq_obs.to_csv(filepath + 'Observed_seasonal_foehn_frequency_' + station_dict[station] + '_' + year_list[0] + '_to_' + year_list[-1] + '.csv')
    seas_foehn_freq_bias.to_csv(filepath + 'Bias_seasonal_foehn_frequency_' + station_dict[station] + '_' + year_list[0] + '_to_' + year_list[-1] + '.csv')
    plt.show()

seas_foehn(year_list = ['1998', '2000','2001', '2002', '2003', '2004', '2005', '2006', '2007',  '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017'], station = 'AWS14_SEB_2009-2017_norp.csv')
seas_foehn(year_list = ['1998', '2000','2001', '2002', '2003', '2004', '2005', '2006', '2007',   '2008','2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017'], station = 'AWS17_SEB_2011-2015_norp.csv')
seas_foehn(year_list = ['1998', '2000','2001', '2002', '2003', '2004', '2005', '2006', '2007',   '2008','2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017'], station = 'AWS18_SEB_2014-2017_norp.csv')
seas_foehn(year_list = ['1998', '2000','2001', '2002', '2003', '2004', '2005', '2006', '2007',   '2008','2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017'], station = 'AWS15_hourly_2009-2014.csv')

def foehn_freq_bar(station):
	yr_list = [ '2011', '2012', '2013', '2014', '2015', '2016']
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
		yr_stats = [np.count_nonzero(foehn_freq_AWS), np.count_nonzero(foehn_freq_surf), np.count_nonzero(foehn_freq_froude), np.count_nonzero(foehn_freq_isen)]
		foehn_stats[year] = pd.Series(yr_stats, index = ['observed', 'surface method', 'Froude method','isentrope method' ])
	print(foehn_stats)
	foehn_stats.plot.bar()
	plt.savefig('/users/ellgil82/figures/Hindcast/foehn/Annual_foehn_frequency_' + station_dict[station] + '_' + yr_list[0] + '_to_' + yr_list[-1] + '.png')
	plt.savefig('/users/ellgil82/figures/Hindcast/foehn/Annual_foehn_frequency_' + station_dict[station] + '_' + yr_list[0] + '_to_' + yr_list[-1] + '.eps')
	plt.show()

foehn_freq_bar('AWS14_SEB_2009-2017_norp')
foehn_freq_bar('AWS17_SEB_2011-2015_norp')
foehn_freq_bar('iWS18_SEB_hourly_until_nov17.txt')
foehn_freq_bar('AWS15_hourly_2009-2014.csv')





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
