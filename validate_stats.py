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
import cf_units

# Set up filepath
if host == 'jasmin':
    filepath = '/gws/nopw/j04/bas_climate/users/ellgil82/hindcast/output/alloutput/'
elif host == 'bsl':
    filepath = '/data/mac/ellgil82/hindcast/output/'

## Load data
def load_vars(year):
    Tair = iris.load_cube( filepath + year+'_Tair_1p5m_daymn.nc', 'air_temperature')
    Ts = iris.load_cube( filepath + year+'_Ts_daymn.nc', 'surface_temperature')
    MSLP = iris.load_cube( filepath + year+'_MSLP_daymn.nc', 'air_pressure_at_sea_level')
    sfc_P = iris.load_cube(filepath  + year + '_sfc_P_daymn.nc', 'surface_air_pressure')
    FF_10m = iris.load_cube( filepath +year+'_FF_10m_daymn.nc', 'wind_speed')
    RH = iris.load_cube(filepath  + year + '_RH_1p5m_daymn.nc', 'relative_humidity')
    u = iris.load_cube(filepath  + year + '_u_10m_daymn.nc', 'x wind component (with respect to grid)')
    v = iris.load_cube(filepath  + year + '_v_10m_daymn.nc', 'y wind component (with respect to grid)')
    LWnet = iris.load_cube( filepath +year+'_surface_LW_net_daymn.nc', 'surface_net_downward_longwave_flux')
    SWnet = iris.load_cube(filepath  + year + '_surface_SW_net_daymn.nc','Net short wave radiation flux')
    LWdown = iris.load_cube( filepath +year+'_surface_LW_down_daymn.nc', 'IR down')
    SWdown = iris.load_cube( filepath +year+'_surface_SW_down_daymn.nc', 'surface_downwelling_shortwave_flux_in_air')
    SWup = iris.load_cube( filepath +year+'_surface_SW_up_daymn.nc')
    LWup = iris.load_cube( filepath +year+'_surface_SW_up_daymn.nc')
    HL = iris.load_cube( filepath +year+'_latent_heat_daymn.nc', 'Latent heat flux')
    HS = iris.load_cube( filepath +year+'_sensible_heat_daymn.nc', 'surface_upward_sensible_heat_flux')
    melt = iris.load_cube( filepath +year+'_land_snow_melt_flux_daymn.nc', 'Snow melt heating flux')
    Tair.convert_units('celsius')
    Ts.convert_units('celsius')
    MSLP.convert_units('hPa')
    sfc_P.convert_units('hPa')
    FF_10m = FF_10m[:,:,1:,:]
    v = v[:,:,1:,:]
    var_list = [Tair, Ts, MSLP, sfc_P, FF_10m, RH, u, v, LWnet, SWnet, LWdown, SWdown, SWup, LWup, HL, HS, melt]
    for i in var_list:
        real_lon, real_lat = rotate_data(i, 2, 3)
        tcoord = i.coord('time')
        tcoord.units = cf_units.Unit(tcoord.units.origin, calendar = 'gregorian') # change to be compatible
        i.coord('time').convert_units('seconds since  1970-01-01 00:00:00')
    Time_srs = matplotlib.dates.num2date(matplotlib.dates.epoch2num(i.coord('time').points))
    WD = metpy.calc.wind_direction(u = u.data, v = v.data)
    WD = iris.cube.Cube(data = WD, standard_name='wind_from_direction')
    Etot = LWnet.data + SWnet.data - HL.data - HS.data
    Emelt_calc = np.copy(Etot)
    Emelt_calc[Ts.data < -0.025] = 0
    for turb in [HS, HL]:
        turb.data = 0-turb.data
    Emelt_calc = iris.cube.Cube(Emelt_calc)
    Etot = iris.cube.Cube(Etot)
    vars_yr = {'Tair': Tair[:,0,:,:], 'Ts': Ts[:,0,:,:], 'MSLP': MSLP[:,0,:,:], 'sfc_P': sfc_P[:,0,:,:], 'FF_10m': FF_10m[:,0,:,:],
               'RH': RH[:,0,:,:], 'WD': WD[:,0,:,:], 'LWnet': LWnet[:,0,:,:], 'SWnet': SWnet[:,0,:,:], 'SWdown': SWdown[:,0,:,:],'SWup': SWup[:,0,:,:],
               'LWdown': LWdown[:,0,:,:], 'HL': HL[:,0,:,:], 'HS': HS[:,0,:,:], 'Etot': Etot[:,0,:,:], 'Emelt': melt[:,0,:,:], 'LWup': LWup[:,0,:,:],
               'lon': real_lon, 'lat': real_lat, 'year': year, 'Emelt_calc': Emelt_calc[:,0,:,:], 'Time_srs': Time_srs}
    return vars_yr

full_srs = load_vars('1998-2017')

try:
    LSM = iris.load_cube(filepath+'new_mask.nc', 'land_binary_mask')
    orog = iris.load_cube(filepath+'orog.nc', 'surface_altitude')
    orog = orog[0,0,:,:]
    LSM = LSM[0,0,:,:]
except iris.exceptions.ConstraintMismatchError:
    print('Files not found')


def load_AWS(station):
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
    # Calculate months
    months = [g for n, g in AWS_srs.groupby(pd.TimeGrouper('M'))]
    DJF = pd.concat((months[11], months[0], months[1]), axis=0)
    MAM = pd.concat((months[2], months[3], months[4]), axis=0)
    JJA = pd.concat((months[5], months[6], months[7]), axis=0)
    SON = pd.concat((months[8], months[9], months[10]), axis=0)
    # Calculate daily means
    AWS_daymn = AWS_srs.resample('D', how = 'mean')
    if station == 'AWS14_SEB_2009-2017_norp.csv':
        AWS_daymn = AWS_daymn[21:-1]
    else:
        AWS_daymn = AWS_daymn[1:-1] # return only means made with complete days
    if host == 'jasmin':
        os.chdir('/gws/nopw/j04/bas_climate/users/ellgil82/hindcast/output/alloutput/')
    elif host == 'bsl':
        os.chdir('/data/mac/ellgil82/hindcast/output/')
    return AWS_srs, DJF, MAM, JJA, SON, AWS_daymn

AWS14_SEB, DJF_14, MAM_14, JJA_14, SON_14, AWS14_daymn = load_AWS('AWS14_SEB_2009-2017_norp.csv')
AWS15_SEB, DJF_15, MAM_15, JJA_15, SON_15, AWS15_daymn = load_AWS('AWS15_hourly_2009-2014.csv')
AWS17_SEB, DJF_17, MAM_17, JJA_17, SON_17, AWS17_daymn = load_AWS('AWS17_SEB_2011-2015_norp.csv')
AWS18_SEB, DJF_18, MAM_18, JJA_18, SON_18, AWS18_daymn = load_AWS('AWS18_SEB_2014-2017_norp.csv')

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

idx_dict = {'AWS14': (4039, 6944),
            'AWS15': (4039, 5912),
            'AWS17': (4793, 6225),
            'AWS18': (6173, 7256)}

def trim_model(AWS):
    trimmed = {}
    for i in ['Tair', 'Ts', 'MSLP', 'sfc_P', 'FF_10m', 'RH', 'WD', 'LWnet', 'SWnet', 'SWup', 'LWup', 'LWdown', 'SWdown', 'HL', 'HS', 'Emelt', 'Etot']:
        trimmed[i] = full_srs[i][idx_dict[AWS][0]:idx_dict[AWS][1], lon_dict[AWS]-1:lon_dict[AWS]+1, lat_dict[AWS]-1:lat_dict[AWS]+1]
    return trimmed

trimmed_14 = trim_model('AWS14')
trimmed_15 = trim_model('AWS15')
trimmed_17 = trim_model('AWS17')
trimmed_18 = trim_model('AWS18')

def calc_bias(AWS):
    A, D, M, J, S, AWS_var = load_AWS(AWS)
    model_var = trim_model(station_dict[AWS])
    surf_obs = [AWS_var['Tsobs'], AWS_var['Tair_2m'], AWS_var['RH'], AWS_var['FF_10m'], AWS_var['pres'], AWS_var['WD'], AWS_var['SWin_corr'], AWS_var['SWout'], AWS_var['SWnet_corr'], AWS_var['LWin'],
                AWS_var['LWout_corr'], AWS_var['LWnet_corr'], AWS_var['Hsen'], AWS_var['Hlat'], AWS_var['E'], AWS_var['melt_energy']]
    mod = [np.mean(model_var['Ts'].data, axis = (1,2)), np.mean(model_var['Tair'].data, axis = (1,2)),  np.mean(model_var['RH'].data, axis = (1,2)), np.mean(model_var['FF_10m'].data, axis = (1,2)),
           np.mean(model_var['sfc_P'].data, axis = (1,2)), np.mean(model_var['WD'].data, axis = (1,2)), np.mean(model_var['SWdown'].data, axis = (1,2)), np.mean(model_var['SWup'].data, axis = (1,2)),
           np.mean(model_var['SWnet'].data, axis = (1,2)), np.mean(model_var['LWdown'].data, axis = (1,2)), np.mean(model_var['LWup'].data, axis = (1,2)), np.mean(model_var['LWnet'].data, axis = (1,2)),
           np.mean(model_var['HS'].data, axis = (1,2)), np.mean(model_var['HL'].data, axis = (1,2)), np.mean(model_var['Etot'].data, axis = (1,2)),np.mean(model_var['Emelt'].data, axis = (1,2))]
    idx = ['Ts', 'Tair', 'RH', 'FF_10m', 'P', 'WD', 'SWdown', 'SWup', 'SWnet', 'LWdown', 'LWup', 'LWnet', 'HS', 'HL', 'E', 'melt']
    mean_obs = []
    mean_mod = []
    bias = []
    errors = []
    r2s = []
    rmses = []
    for i in np.arange(len(surf_obs)):
        b = mod[i] - surf_obs[i]
        errors.append(b) #find biases
        mean_obs.append(np.mean(surf_obs[i]))
        mean_mod.append(np.mean(mod[i]))
        bias.append(np.mean(b)) # find mean bias
        slope, intercept, r2, p, sterr = scipy.stats.linregress(surf_obs[i], mod[i])
        r2s.append(r2)
        mse = mean_squared_error(y_true = surf_obs[i], y_pred = mod[i])
        rmse = np.sqrt(mse)
        rmses.append(rmse)
    df = pd.DataFrame(index = idx)
    df['obs mean'] = pd.Series(mean_obs, index = idx)
    df['mod mean'] = pd.Series(mean_mod, index = idx)
    df['bias'] =pd.Series(bias, index=idx)
    df['rmse'] = pd.Series(rmses, index = idx)
    df['% RMSE'] = ( df['rmse']/df['obs mean'] ) * 100
    df['correl'] = pd.Series(r2s, index = idx)
    for i in range(len(mod)):
        slope, intercept, r2, p, sterr = scipy.stats.linregress(surf_obs[i], mod[i])
        print(idx[i])
        print('\nr2 = %s\n' % r2)
    print('RMSE/bias = \n\n\n')
    melt_nonzero = np.copy(surf_obs[-1])
    melt_nonzero[melt_nonzero == 0.] = np.nan
    AWS_nonz_mn = np.nanmean(melt_nonzero)
    mod_melt_nonzero = np.copy(mod[-2])
    mod_melt_nonzero[mod_melt_nonzero == 0.] = np.nan
    mod_nonz_mn = np.nanmean(mod_melt_nonzero)
    calc_mod_melt_nonzero = np.copy(mod[-1])
    calc_mod_melt_nonzero[calc_mod_melt_nonzero == 0.] = np.nan
    calc_mod_nonz_mn = np.nanmean(calc_mod_melt_nonzero)
    nonz_bias = np.nanmean(mod_melt_nonzero - melt_nonzero)
    print(' observed mean: \n%s\n' % AWS_nonz_mn)
    print(' model mean: \n%s\n' % mod_nonz_mn)
    print(' calculated model mean: \n%s\n' % calc_mod_nonz_mn)
    df.to_csv('/data/mac/ellgil82/hindcast/Total_stats_' + station_dict[station] + '_bias_RMSE.csv') # change to be station-specific
    print(df)

calc_bias('AWS14_SEB_2009-2017_norp.csv')
calc_bias('AWS17_SEB_2011-2015_norp.csv')
calc_bias('AWS18_SEB_2014-2017_norp.csv')




def remove_diurnal_cyc(input_var):
    if input_var.ndim >= 2:
        data = input_var.data
        nt, ny, nx = data.shape
        data = data.reshape(nt, ny*nx)
        tmax, ngrid = data.shape
        diur = np.zeros((8, ngrid))  # _number of hours = 3 hourly output
        for hr in np.arange(8):
            idx = np.arange(hr, tmax, 8)
            diur[hr, :] = data[idx].mean(axis=0)
        day_cyc = np.reshape(diur, (8, ny, nx))
        diur = np.tile(diur, [int(len(input_var.coord('t').points) / 8.), 1])  # 24 hours in 37 days
        data = (data - diur).reshape(nt, ny, nx)
    else:
        data = input_var.values
        diur = np.zeros((24))
        for hr in np.arange(24):
            idx = np.arange(hr, len(data), 24)
            diur[hr] = data[idx].mean(axis=0)
        day_cyc = np.reshape(diur, (24))
        diur = np.tile(diur, [int(len(data)/24.), 1])
    return data, day_cyc




from seasonal import fit_seasons, adjust_seasons
seasons, trend = fit_seasons(full_srs['Ts'].data[:,lon_dict['AWS14'], lat_dict['AWS14']])
adjusted = adjust_seasons(full_srs['Ts'][:,lon_dict['AWS14'], lat_dict['AWS14']].data, seasons=seasons)
seas = full_srs['Ts'][:,lon_dict['AWS14'], lat_dict['AWS14']].data - trend
residual = adjusted - trend