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
#from cftime import datetime
from datetime import datetime, time

# Set up filepath
if host == 'jasmin':
    filepath = '/gws/nopw/j04/bas_climate/users/ellgil82/hindcast/output/alloutput/'
elif host == 'bsl':
    filepath = '/data/mac/ellgil82/hindcast/output/'

## Load data
def load_vars(year, mn):
    if mn == 'yes':
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
        HL = iris.load_cube( filepath +year+'_latent_heat_daymn.nc', 'Latent heat flux')
        HS = iris.load_cube( filepath +year+'_sensible_heat_daymn.nc', 'surface_upward_sensible_heat_flux')
        #Etot = iris.load_cube(filepath + year + '_surface_E_tot_daymn.nc')
        melt = iris.load_cube( filepath +year+'_land_snow_melt_flux_daymn.nc', 'Snow melt heating flux')
    elif mn == 'no':
        Tair = iris.load_cube( filepath +year+'_Tair_1p5m.nc', 'air_temperature')
        Ts = iris.load_cube( filepath +year+'_Ts.nc', 'surface_temperature')
        MSLP = iris.load_cube( filepath +year+'_MSLP.nc', 'air_pressure_at_sea_level')
        sfc_P = iris.load_cube(filepath  + year + '_sfc_P.nc', 'surface_air_pressure')
        FF_10m = iris.load_cube( filepath +year+'_FF_10m.nc', 'wind_speed')
        RH = iris.load_cube(filepath  + year + '_RH_1p5m.nc', 'relative_humidity')
        u = iris.load_cube(filepath + year + '_u_10m.nc', 'x wind component (with respect to grid)')
        v = iris.load_cube(filepath  + year + '_v_10m.nc', 'y wind component (with respect to grid)')
        LWnet = iris.load_cube( filepath+year+'_surface_LW_net.nc', 'surface_net_downward_longwave_flux')
        SWnet = iris.load_cube(filepath  + year + '_surface_SW_net.nc','Net short wave radiation flux')
        LWdown = iris.load_cube( filepath +year+'_surface_LW_down.nc', 'IR down')
        SWdown = iris.load_cube( filepath +year+'_surface_SW_down.nc', 'surface_downwelling_shortwave_flux_in_air')
        HL = iris.load_cube( filepath +year+'_latent_heat.nc', 'Latent heat flux')
        HS = iris.load_cube( filepath +year+'_sensible_heat.nc', 'surface_upward_sensible_heat_flux')
        #Etot = iris.load_cube(filepath + year + '_surface_E_tot.nc')
        melt = iris.load_cube(filepath  + year + '_land_snow_melt_flux.nc', 'Snow melt heating flux')
    Tair.convert_units('celsius')
    Ts.convert_units('celsius')
    MSLP.convert_units('hPa')
    sfc_P.convert_units('hPa')
    FF_10m = FF_10m[:,:,1:,:]
    v = v[:,:,1:,:]
    # Flip direction of turbulent fluxes to match convention (positive towards surface)
    HS = iris.analysis.maths.multiply(HS, -1.)
    HL = iris.analysis.maths.multiply(HL, -1.)
    var_list = [Tair, Ts, MSLP, sfc_P, FF_10m, RH, u, v, melt, LWnet, SWnet, LWdown, SWdown,  HL, HS]
    for i in var_list:
        real_lon, real_lat = rotate_data(i, 2, 3)
    WD = metpy.calc.wind_direction(u = u.data, v = v.data)
    WD = iris.cube.Cube(data = WD, standard_name='wind_from_direction')
    #Etot = LWnet.data + SWnet.data + HL.data + HS.data
    #Etot = iris.cube.Cube(Etot)
    SWup = iris.cube.Cube(data = SWnet.data - SWdown.data)
    LWup = iris.cube.Cube(data=LWnet.data - LWdown.data)
    vars_yr = {'Tair': Tair[:,0,:,:], 'Ts': Ts[:,0,:,:], 'MSLP': MSLP[:,0,:,:], 'sfc_P': sfc_P[:,0,:,:], 'FF_10m': FF_10m[:,0,:,:],
               'RH': RH[:,0,:,:], 'WD': WD[:,0,:,:], 'u': u[:,0,:,:], 'v': v[:,0,:,:], 'LWnet': LWnet[:,0,:,:], 'SWnet': SWnet[:,0,:,:], 'SWdown': SWdown[:,0,:,:],
               'LWdown': LWdown[:,0,:,:], 'HL': HL[:,0,:,:], 'HS': HS[:,0,:,:], 'Emelt': melt[:,0,:,:], 'SWup': SWup[:,0,:,:], 'LWup': LWup[:,0,:,:],
               'lon': real_lon, 'lat': real_lat, 'year': year}
    return vars_yr

#full_srs = load_vars('2016', mn = 'no')

full_srs = load_vars('1998-2017', mn = 'yes')
full_srs['Etot'] = iris.cube.Cube(data = (full_srs['LWnet'].data + full_srs['SWnet'].data + full_srs['HL'].data + full_srs['HS'].data))

#mn_srs = load_vars('1998-2017', mn = 'yes')

try:
    LSM = iris.load_cube(filepath+'new_mask.nc')
    orog = iris.load_cube(filepath+'orog.nc')
    orog = orog[0,0,:,:]
    LSM = LSM[0,0,:,:]
except iris.exceptions.ConstraintMismatchError:
    print('Files not found')

def load_AWS(station, year):
    ## --------------------------------------------- SET UP VARIABLES ------------------------------------------------##
    ## Load data from AWS 14 and AWS 15 for January 2011
    print('\nDayum grrrl, you got a sweet AWS...')
    os.chdir(filepath)
    for file in os.listdir(filepath):
        if fnmatch.fnmatch(file, '%(station)s*' % locals()):
            AWS_srs = pd.read_csv(str(file), na_values = -9999, header = 0)
    # Calculate date, given list of years and day of year
    date_list = compose_date(AWS_srs['year'], days=AWS_srs['day'])
    AWS_srs['Date'] = date_list
    # Set date as index
    AWS_srs.index = AWS_srs['Date']
    # Calculate actual time from decimal DOY (seriously, what even IS that format?)
    AWS_srs['time'] = 24*(AWS_srs['Time'] - AWS_srs['day'])
    case = AWS_srs.loc[year+'-01-01':year +'-12-31'] #'2015-01-01':'2015-12-31'
    print '\nconverting times...'
    # Convert times so that they can be plotted
    time_list = []
    for i in case['time']:
        hrs = int(i)                 # will now be 1 (hour)
        mins = int((i-hrs)*60)       # will now be 4 minutes
        secs = int(0 - hrs*60*60 + mins*60) # will now be 30
        j = time(hour = hrs, minute=mins)
        time_list.append(j)
    case['Time'] = time_list
    case['datetime'] = case.apply(lambda r : pd.datetime.combine(r['Date'],r['Time']),1)
    case['E'] = case['LWnet_corr'].values + case['SWnet_corr'].values + case['Hlat'].values + case['Hsen'].values - case['Gs'].values
    case['WD'][case['WD'] < 0.] = np.nan
    case['FF_10m'][case['FF_10m'] < 0.] = np.nan
    case['WD'] = case['WD'].interpolate() # interpolate missing values
    case['FF_10m'] = case['FF_10m'].interpolate()
    daily_Tair = case.resample('D')['Tair_2m']
    case['Tair_min'] = daily_Tair.transform('min')
    case['Tair_max'] = daily_Tair.transform('max')
    u, v = metpy.calc.wind_components(case['FF_10m'], case['WD'])
    case['u'] = u
    case['v'] = v
    try:
        case['Tsobs'][case['Tsobs'] > -0.025] = 0.
        daily_Ts = case.resample('D')['Tsobs']
        case['Ts_min'] = daily_Ts.transform('min')
        case['Ts_max'] = daily_Ts.transform('max')
    except KeyError:
        print('Ts not available at AWS 15')
    if station == 'AWS18_SEB_2014-2017_norp.csv':
        AWS_srs = AWS_srs[::2]
    # Calculate months
    months = [g for n, g in case.groupby(pd.TimeGrouper('M'))]
    DJF = pd.concat((months[11], months[0], months[1]), axis=0)
    MAM = pd.concat((months[2], months[3], months[4]), axis=0)
    JJA = pd.concat((months[5], months[6], months[7]), axis=0)
    SON = pd.concat((months[8], months[9], months[10]), axis=0)
    return case, DJF, MAM, JJA, SON

#ANN_18, DJF, MAM, JJA, SON = load_AWS(station = 'AWS18_SEB_2014-2017_norp.csv', year = '2016')

def load_all_AWS(station, daily):
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
            j = time(hour = hrs, minute=mins)
            time_list.append(j)
        AWS_srs['Time'] = time_list
    except TypeError:
        print('Got time already m9')
        AWS_srs['Time'] = pd.to_datetime(AWS_srs['Time'], format='%H:%M:%S').dt.time
    print('\nconverting times...')
    # Convert times so that they can be plotted
    AWS_srs['datetime'] = AWS_srs.apply(lambda r : pd.datetime.combine(r['Date'],r['Time']),1)
    try:
        AWS_srs['E'] = AWS_srs['LWnet_corr'].values + AWS_srs['SWnet_corr'].values + AWS_srs['Hlat'].values + AWS_srs['Hsen'].values - AWS_srs['Gs'].values
    except:
        print('No full SEB \'ere pal...')
    AWS_srs['WD'][AWS_srs['WD'] < 0.] = np.nan
    AWS_srs['FF_10m'][AWS_srs['FF_10m'] < 0.] = np.nan
    #AWS_srs['WD'] = AWS_srs['WD'].interpolate() # interpolate missing values
    #AWS_srs['FF_10m'] = AWS_srs['FF_10m'].interpolate()
    AWS_srs['WD'][AWS_srs['WD'] == 0.] = np.nan
    u, v = metpy.calc.wind_components(AWS_srs['FF_10m'], AWS_srs['WD'])
    AWS_srs['u'] = u
    AWS_srs['v'] = v
    if station == 'AWS18_SEB_2014-2017_norp.csv':
        AWS_srs = AWS_srs[::2]
    if station == 'AWS15_hourly_2009-2014.csv':
        AWS_srs['P'][ AWS_srs['P'] < 800.] = np.nan
        for j in ['SWin', 'SWout', 'LWin', 'LWout']:
            AWS_srs[j][AWS_srs[j] < -999.] = np.nan
            #AWS_srs[j].interpolate(method = 'linear', limit_direction = 'both') # linearly interpolate missing values
        AWS_srs['SWnet'] = AWS_srs['SWin'] - AWS_srs['SWout']
        AWS_srs['LWnet'] = AWS_srs['LWin'] - AWS_srs['LWout']
    else:
        AWS_srs['Tsobs'][AWS_srs['Tsobs'] > -0.025] = 0.
        # Linearly interpolate missing values
    #AWS_srs.interpolate('linear', limit_direction = 'both')
        # Calculate daily means
    if daily == 'yes':
        AWS_srs = AWS_srs.groupby(AWS_srs.index).mean()
    if station == 'AWS14_SEB_2009-2017_norp.csv':
        AWS_srs = AWS_srs[1:]
    daily_Tair = AWS_srs.resample('D')['Tair_2m']
    AWS_srs['Tair_min'] = daily_Tair.transform('min')
    AWS_srs['Tair_max'] = daily_Tair.transform('max')
    try:
        daily_Ts = AWS_srs.resample('D')['Tsobs']
        AWS_srs['Ts_min'] = daily_Ts.transform('min')
        AWS_srs['Ts_max'] = daily_Ts.transform('max')
    except KeyError:
        print('Ts not available at AWS 15')
    ## Calculate months
    #months = [g for n, g in AWS_srs.groupby(pd.TimeGrouper('M'))]
    #DJF = pd.concat((months[11], months[0], months[1]))
    #MAM = pd.concat((months[2], months[3], months[4]))
    #JJA = pd.concat((months[5], months[6], months[7]))
    #SON = pd.concat((months[8], months[9], months[10]))
    if host == 'jasmin':
        os.chdir('/gws/nopw/j04/bas_climate/users/ellgil82/hindcast/output/alloutput/')
    elif host == 'bsl':
        os.chdir('/data/mac/ellgil82/hindcast/output/')
    return AWS_srs #DJF, MAM, JJA, SON

print('\nLoading in AWS data\n')

ANN_14 = load_all_AWS('AWS14_SEB_2009-2017_norp.csv', daily = 'yes')  # , DJF_14, MAM_14, JJA_14, SON_14
ANN_15 = load_all_AWS('AWS15_hourly_2009-2014.csv', daily   = 'yes')  # , DJF_15, MAM_15, JJA_15, SON_15
ANN_17 = load_all_AWS('AWS17_SEB_2011-2015_norp.csv', daily = 'yes')  # , DJF_17, MAM_17, JJA_17, SON_17
ANN_18 = load_all_AWS('AWS18_SEB_2014-2017_norp.csv', daily = 'yes')  # , DJF_18, MAM_18, JJA_18, SON_18

def make_model_timesrs(model_var, freq):
    try:
        Timesrs = model_var['Emelt'].coord('t').points
    except:
        Timesrs = model_var['Emelt'].coord('time').points
    Timesrs = Timesrs + 0.5 # account for offset from initialisation time (t-12 hr), then + 1 to map index day[0] onto day 1.
    df = pd.DataFrame()
    if model_var['year'] == '1998-2017':
        df['datetime'] = pd.date_range(datetime(1998,1,1,0,0,0),datetime(2017,12,31,23,59,59), freq = freq)
    else:
        yrs = np.repeat(np.int(model_var['year']), len(Timesrs))
        date_list = compose_date(yrs, days=Timesrs)
        time_list = [time(0,0,0), time(3,0,0), time(6,0,0), time(9,0,0), time(12,0,0), time(15,0,0), time(18,0,0), time(21,0,0)]
        time_list = np.tile(np.asarray(time_list), len(Timesrs)/8)
        df['times'] = pd.Series(time_list)
        df['dates'] = pd.Series(date_list)
        df['datetime'] = df.apply(lambda r : pd.datetime.combine(r['dates'],r['times']),1)
    model_var['Timesrs'] = df['datetime']

def trim_model(model_var, AWS_var, daily):
    #load model vars to match AWS
    if daily == 'yes':
        freq = 'D'
    else:
        freq = '3H'
    make_model_timesrs(model_var, freq)
    Timesrs = model_var['Timesrs']
    # find where times match
    if daily == 'yes':
        start = np.where(Timesrs == pd.to_datetime(AWS_var.index)[0])
        end = np.where(Timesrs == pd.to_datetime(AWS_var.index)[-1])
    else:
        start = np.where(Timesrs == pd.to_datetime(AWS_var['datetime'])[0])
        if start[0] > 0:
            start = start
        else:
            start = np.where(Timesrs == pd.to_datetime(AWS_var['datetime'])[1])
            if start[0] > 0:
                start = start
        end = np.where(Timesrs == pd.to_datetime(AWS_var['datetime'])[-1])
        if end[0] > 0:
            end = end
        else:
            end = np.where(Timesrs == pd.to_datetime(AWS_var['datetime'])[-2])
            if end[0] > 0:
                end = end
            else:
                end = np.where(Timesrs == pd.to_datetime(AWS_var['datetime'])[-3])
                if end[0] > 0:
                    end = end
                else:
                    end = np.where(Timesrs == pd.to_datetime(AWS_var['datetime'][-1]).replace(minute = AWS_var['datetime'][-1].minute +1))
                    if end[0] > 0:
                        end = end
                    else:
                        end = np.where(Timesrs == pd.to_datetime(AWS_var['datetime'][-2]).replace(minute=AWS_var['datetime'][-2].minute + 1))
                        if end[0] > 0:
                            end = end
    new_vars = {}
    new_vars['year'] = '1998-2017'
    for j in ['Tair','Ts', 'MSLP', 'sfc_P', 'FF_10m', 'u', 'v', 'RH', 'WD', 'LWnet', 'SWnet', 'SWdown', 'LWdown', 'LWup', 'SWup', 'HL', 'HS', 'Etot', 'Emelt', 'Timesrs']:
        new_vars[j] = model_var[j][start[0][0]:end[0][0]]
    new_vars['start'] = start[0][0]
    new_vars['end'] = end[0][0]
    return new_vars

srs_17_trimmed = trim_model(model_var = full_srs, AWS_var = ANN_17, daily = 'yes')
srs_18_trimmed = trim_model(model_var = full_srs, AWS_var = ANN_18, daily = 'yes')
srs_14_trimmed = trim_model(model_var = full_srs, AWS_var = ANN_14, daily = 'yes')
srs_15_trimmed = trim_model(model_var = full_srs, AWS_var = ANN_15, daily = 'yes')

lon_index14, lat_index14, = find_gridbox(-67.01, -61.03, full_srs['lat'], full_srs['lon'])
lon_index15, lat_index15, = find_gridbox(-67.34, -62.09, full_srs['lat'], full_srs['lon'])
lon_index17, lat_index17, = find_gridbox(-65.93, -61.85, full_srs['lat'], full_srs['lon'])
lon_index18, lat_index18, = find_gridbox(-66.48, -63.37, full_srs['lat'], full_srs['lon'])

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

print('\nTrimming model data to match observation period\n')

#srs_17_trimmed = trim_model(model_var = full_srs, AWS_var = ANN_17, daily = 'no')
#srs_18_trimmed = trim_model(model_var = full_srs, AWS_var = ANN_18, daily = 'no')
#srs_14_trimmed = trim_model(model_var = full_srs, AWS_var = ANN_14, daily = 'no')
#srs_15_trimmed = trim_model(model_var = full_srs, AWS_var = ANN_15, daily = 'no')

# Calculate daily mins and maxs for each AWS
def calc_minmax(station, trimmed_srs, daily):
    df = pd.DataFrame()
    df['mod_T'] = np.mean(trimmed_srs['Tair'].data[:,lat_dict[station_dict[station]] - 1:lat_dict[station_dict[station]] + 1, lon_dict[station_dict[station]] - 1:lon_dict[station_dict[station]] + 1], axis=(1, 2))
    Timesrs = pd.date_range(datetime(1998,1,1,0,0,0),datetime(2017,12,31,23,59,59), freq = '3H')
    #Timesrs = pd.date_range(datetime(2016,1,1,0,0,0),datetime(2017,1,1,0,0,0), freq = '3H')
    Timesrs = Timesrs[trimmed_srs['start']:trimmed_srs['end']]
    df.index = Timesrs
    daily_T = df.resample('D')['mod_T']
    Tmax = daily_T.transform('max')
    Tmin = daily_T.transform('min')
    trimmed_srs['Tmin'] = Tmin
    trimmed_srs['Tmax'] = Tmax
    return Tmax, Tmin

print('\nCalculating mins/maxes\n')

Tmax14, Tmin14 = calc_minmax('AWS14_SEB_2009-2017_norp.csv', srs_14_trimmed, daily = 'yes')
Tmax15, Tmin15 = calc_minmax('AWS15_hourly_2009-2014.csv', srs_15_trimmed, daily = 'yes')
Tmax17, Tmin17 = calc_minmax('AWS17_SEB_2011-2015_norp.csv', srs_17_trimmed, daily = 'yes')
Tmax18, Tmin18 = calc_minmax('AWS18_SEB_2014-2017_norp.csv', srs_18_trimmed, daily = 'yes')

mins = {'AWS14': Tmin14,
            'AWS15': Tmin15,
            'AWS17': Tmin17,
            'AWS18': Tmin18}

maxes = {'AWS14': Tmax14,
            'AWS15': Tmax15,
            'AWS17': Tmax17,
            'AWS18': Tmax18}#

def seas_mean(year_list, location):
    for each_year in year_list:
        seas_means = pd.DataFrame(index =['DJF', 'MAM', 'JJA', 'SON', 'ANN'])
        seas_vals = pd.DataFrame(index =['DJF', 'MAM', 'JJA', 'SON', 'ANN'])
        for each_var in file_dict.keys():
            ANN = iris.load_cube(filepath + each_year + file_dict[each_var], each_var)
            DJF = iris.load_cube(filepath + each_year + '/DJF_' + each_year +  file_dict[each_var], each_var)
            MAM = iris.load_cube(filepath + each_year + '/MAM_' + each_year + file_dict[each_var], each_var)
            JJA = iris.load_cube(filepath + each_year + '/JJA_' + each_year + file_dict[each_var], each_var)
            SON = iris.load_cube(filepath + each_year + '/SON_' + each_year +  file_dict[each_var], each_var)
            # Conversions
            if each_var == 'surface_air_pressure' or each_var == 'air_pressure_at_sea_level':
                for each_seas in [DJF, MAM, JJA, SON, ANN]:
                    each_seas.convert_units('hPa')
            elif each_var == 'surface_temperature' or each_var == 'air_temperature':
                for each_seas in [DJF, MAM, JJA, SON, ANN]:
                    each_seas.convert_units('celsius')
            # calculate mean at specified location
            ANN = ANN[:, 0, lat_dict[location], lon_dict[location]].data
            DJF = DJF[:, 0, lat_dict[location], lon_dict[location]].data
            MAM = MAM[:, 0, lat_dict[location], lon_dict[location]].data
            JJA = JJA[:, 0, lat_dict[location], lon_dict[location]].data
            SON = SON[:, 0, lat_dict[location], lon_dict[location]].data
            seas_vals[each_var] = [DJF, MAM, JJA, SON, ANN]
            seas_means[each_var] = [np.mean(DJF), np.mean(MAM), np.mean(JJA), np.mean(SON), np.mean(ANN) ]
        wd_djf = metpy.calc.wind_direction(u = seas_vals['x wind component (with respect to grid)']['DJF'], v = seas_vals['y wind component (with respect to grid)']['DJF'])
        wd_mam = metpy.calc.wind_direction(u = seas_vals['x wind component (with respect to grid)']['MAM'], v = seas_vals['y wind component (with respect to grid)']['MAM'])
        wd_jja = metpy.calc.wind_direction(u = seas_vals['x wind component (with respect to grid)']['JJA'], v = seas_vals['y wind component (with respect to grid)']['JJA'])
        wd_son = metpy.calc.wind_direction(u = seas_vals['x wind component (with respect to grid)']['SON'], v = seas_vals['y wind component (with respect to grid)']['SON'])
        wd_ann = metpy.calc.wind_direction(u=seas_vals['x wind component (with respect to grid)']['ANN'], v=seas_vals['y wind component (with respect to grid)']['ANN'])
        seas_vals['WD'] = [wd_djf.magnitude, wd_mam.magnitude, wd_jja.magnitude, wd_son.magnitude, wd_ann.magnitude]
        seas_means['WD'] = metpy.calc.wind_direction(u = seas_means['x wind component (with respect to grid)'].values, v = seas_means['y wind component (with respect to grid)'].values)
        seas_means = seas_means.transpose()
        print(seas_means)
        seas_means.to_csv('/data/mac/ellgil82/hindcast/output/'+ each_year +'_seas_means.csv')
        #save indivdual dataframes for each year
        # calculate mean for whole period
        #DJF_means[each_var] = DJF_means[each_var] / len(year_list)
    return seas_means, seas_vals

def calc_bias(trimmed_vars, AWS_total, station, daily, foehn):
    # Calculate bias of time series
    # Forecast error
    vars_yr = trimmed_vars
    if foehn == 'yes':
        foehn_df = pd.read_csv(filepath + 'daily_foehn_frequency_all_stations.csv')  # turn this into AWS 14/15/18 average, then diagnose when foehn is shown at one or more
        foehn_subset = {}
        foehn_df = foehn_df.iloc[vars_yr['start']:vars_yr['end']]
        for v in ['WD', 'HS', 'Tair', 'Ts', 'LWdown', 'HL', 'Emelt', 'SWdown', 'LWup', 'SWnet', 'SWup', 'RH',
                  'FF_10m', 'sfc_P', 'MSLP', 'Etot', 'LWnet', 'u', 'v']:
            sta_ts = np.mean(vars_yr[v][:, lat_dict[station_dict[station]] - 1:lat_dict[station_dict[station]] + 1,
                             lon_dict[station_dict[station]] - 1:lon_dict[station_dict[station]] + 1].data,
                             axis=(1, 2))  # calculate mean of variable at the correct station
            sta_ts[foehn_df[station_dict[station]] == 0] = np.nan  # mask values where foehn conditions at that station are not simulated
            foehn_subset[v] = iris.cube.Cube(data=sta_ts)
        foehn_subset['Tmax'] = vars_yr['Tmax']
        foehn_subset['Tmin'] = vars_yr['Tmin']
        AWS_masked = AWS_total[:sta_ts.shape[0]].copy()
        AWS_masked.values[foehn_df[station_dict[station]] == 0] = np.nan
        AWS_total = AWS_masked
        vars_yr = foehn_subset
    if daily == 'yes':
        AWS_var = AWS_total.groupby(AWS_total.index).mean()
    elif daily == 'no':
        AWS_var = AWS_total[::3]
    length = min(trimmed_vars['Tair'].shape[0], AWS_var['Tair_2m'].shape[0])
    if station == 'AWS15_hourly_2009-2014.csv':
        surf_met_obs = [AWS_var['Tair_2m'][:length], AWS_var['Tair_max'][:length], AWS_var['Tair_min'][:length], AWS_var['RH'][:length],
                        AWS_var['FF_10m'][:length], AWS_var['P'][:length], AWS_var['u'][:length], AWS_var['v'][:length],
                        AWS_var['SWin'][:length],  AWS_var['LWin'][:length], AWS_var['SWnet'][:length],
                        AWS_var['LWnet'][:length]]
        surf_mod = [np.nanmean(vars_yr['Tair'].data[:length,
                            lat_dict[station_dict[station]] - 1:lat_dict[station_dict[station]] + 1,
                            lon_dict[station_dict[station]] - 1:lon_dict[station_dict[station]] + 1], axis=(1, 2)),
                    mins[station_dict[station]],
                    maxes[station_dict[station]],
                    np.nanmean(vars_yr['RH'].data[:length,
                            lat_dict[station_dict[station]] - 1:lat_dict[station_dict[station]] + 1,
                            lon_dict[station_dict[station]] - 1:lon_dict[station_dict[station]] + 1], axis=(1, 2)),
                    np.nanmean(vars_yr['FF_10m'].data[:length,
                            lat_dict[station_dict[station]] - 1:lat_dict[station_dict[station]] + 1,
                            lon_dict[station_dict[station]] - 1:lon_dict[station_dict[station]] + 1], axis=(1, 2)),
                    np.nanmean(vars_yr['sfc_P'].data[:length,
                            lat_dict[station_dict[station]] - 1:lat_dict[station_dict[station]] + 1,
                            lon_dict[station_dict[station]] - 1:lon_dict[station_dict[station]] + 1], axis=(1, 2)),
                    np.nanmean(vars_yr['u'].data[:length,
                            lat_dict[station_dict[station]] - 1:lat_dict[station_dict[station]] + 1,
                            lon_dict[station_dict[station]] - 1:lon_dict[station_dict[station]] + 1], axis=(1, 2)),
                    np.nanmean(vars_yr['v'].data[:length,
                            lat_dict[station_dict[station]] - 1:lat_dict[station_dict[station]] + 1,
                            lon_dict[station_dict[station]] - 1:lon_dict[station_dict[station]] + 1], axis=(1, 2)),
                    np.nanmean(vars_yr['SWdown'].data[:length,
                            lat_dict[station_dict[station]] - 1:lat_dict[station_dict[station]] + 1,
                            lon_dict[station_dict[station]] - 1:lon_dict[station_dict[station]] + 1], axis=(1, 2)),
                    np.nanmean(vars_yr['LWdown'].data[:length,
                            lat_dict[station_dict[station]] - 1:lat_dict[station_dict[station]] + 1,
                            lon_dict[station_dict[station]] - 1:lon_dict[station_dict[station]] + 1], axis=(1, 2)),
                    np.nanmean(vars_yr['SWnet'].data[:length,
                            lat_dict[station_dict[station]] - 1:lat_dict[station_dict[station]] + 1,
                            lon_dict[station_dict[station]] - 1:lon_dict[station_dict[station]] + 1], axis=(1, 2)),
                    np.nanmean(vars_yr['LWnet'].data[:length,
                            lat_dict[station_dict[station]] - 1:lat_dict[station_dict[station]] + 1,
                            lon_dict[station_dict[station]] - 1:lon_dict[station_dict[station]] + 1], axis=(1, 2))]
        idx = ['Tair', 'Tmin', 'Tmax', 'RH', 'FF', 'P', 'u', 'v', 'Swdown', 'LWdown', 'SWnet', 'LWnet']
    else:
        surf_met_obs = [AWS_var['Tsobs'][:length], AWS_var['Tair_2m'][:length], AWS_var['Tair_min'], AWS_var['Tair_max'], AWS_var['RH'][:length], AWS_var['FF_10m'][:length], AWS_var['pres'][:length], AWS_var['u'][:length], AWS_var['v'][:length], AWS_var['SWin_corr'][:length],
                        AWS_var['LWin'][:length], AWS_var['SWnet_corr'][:length], AWS_var['LWnet_corr'][:length], AWS_var['Hsen'][:length], AWS_var['Hlat'][:length], AWS_var['E'][:length], AWS_var['melt_energy'][:length]]
        surf_mod = [np.nanmean(vars_yr['Ts'].data[:length, lat_dict[station_dict[station]]-1:lat_dict[station_dict[station]]+1, lon_dict[station_dict[station]]-1:lon_dict[station_dict[station]]+1], axis = (1,2)),
                    np.nanmean(vars_yr['Tair'].data[:length, lat_dict[station_dict[station]]-1:lat_dict[station_dict[station]]+1, lon_dict[station_dict[station]]-1:lon_dict[station_dict[station]]+1],axis = (1,2)),
                    mins[station_dict[station]],
                    maxes[station_dict[station]],
                    np.nanmean(vars_yr['RH'].data[:length, lat_dict[station_dict[station]]-1:lat_dict[station_dict[station]]+1, lon_dict[station_dict[station]]-1:lon_dict[station_dict[station]]+1], axis = (1,2)),
                    np.nanmean(vars_yr['FF_10m'].data[:length, lat_dict[station_dict[station]]-1:lat_dict[station_dict[station]]+1, lon_dict[station_dict[station]]-1:lon_dict[station_dict[station]]+1], axis = (1,2)),
                    np.nanmean(vars_yr['sfc_P'].data[:length, lat_dict[station_dict[station]]-1:lat_dict[station_dict[station]]+1, lon_dict[station_dict[station]]-1:lon_dict[station_dict[station]]+1], axis = (1,2)),
                    np.nanmean(vars_yr['u'].data[:length, lat_dict[station_dict[station]]-1:lat_dict[station_dict[station]]+1, lon_dict[station_dict[station]]-1:lon_dict[station_dict[station]]+1], axis = (1,2)),
                    np.nanmean(vars_yr['v'].data[:length, lat_dict[station_dict[station]] - 1:lat_dict[station_dict[station]] + 1, lon_dict[station_dict[station]] - 1:lon_dict[station_dict[station]] + 1], axis=(1, 2)),
                    np.nanmean(vars_yr['SWdown'].data[:length, lat_dict[station_dict[station]]-1:lat_dict[station_dict[station]]+1, lon_dict[station_dict[station]]-1:lon_dict[station_dict[station]]+1], axis = (1,2)),
                    np.nanmean(vars_yr['LWdown'].data[:length, lat_dict[station_dict[station]]-1:lat_dict[station_dict[station]]+1, lon_dict[station_dict[station]]-1:lon_dict[station_dict[station]]+1], axis = (1,2)),
                    np.nanmean(vars_yr['SWnet'].data[:length, lat_dict[station_dict[station]]-1:lat_dict[station_dict[station]]+1, lon_dict[station_dict[station]]-1:lon_dict[station_dict[station]]+1], axis = (1,2)),
                    np.nanmean(vars_yr['LWnet'].data[:length, lat_dict[station_dict[station]]-1:lat_dict[station_dict[station]]+1, lon_dict[station_dict[station]]-1:lon_dict[station_dict[station]]+1], axis = (1,2)),
                    (np.nanmean(vars_yr['HS'].data[:length, lat_dict[station_dict[station]]-1:lat_dict[station_dict[station]]+1, lon_dict[station_dict[station]]-1:lon_dict[station_dict[station]]+1], axis = (1,2))*-1.),
                    (np.nanmean(vars_yr['HL'].data[:length, lat_dict[station_dict[station]]-1:lat_dict[station_dict[station]]+1, lon_dict[station_dict[station]]-1:lon_dict[station_dict[station]]+1], axis = (1,2))*-1.),
                    np.nanmean(vars_yr['Etot'].data[:length, lat_dict[station_dict[station]] - 1:lat_dict[station_dict[station]] + 1,lon_dict[station_dict[station]] - 1:lon_dict[station_dict[station]] + 1], axis=(1, 2)),
                    np.nanmean(vars_yr['Emelt'].data[:length, lat_dict[station_dict[station]]-1:lat_dict[station_dict[station]]+1, lon_dict[station_dict[station]]-1:lon_dict[station_dict[station]]+1], axis = (1,2))]
        idx = ['Ts', 'Tair', 'Tmin', 'Tmax', 'RH', 'FF', 'P', 'u', 'v', 'Swdown', 'LWdown', 'SWnet', 'LWnet', 'HS', 'HL', 'Etot', 'Emelt']
        melt_nonzero = np.copy(surf_met_obs[-1])
        melt_nonzero[melt_nonzero == 0.] = np.nan
        AWS_nonz_mn = np.nanmean(melt_nonzero)
        mod_melt_nonzero = np.copy(surf_mod[-2])
        mod_melt_nonzero[mod_melt_nonzero == 0.] = np.nan
        mod_nonz_mn = np.nanmean(mod_melt_nonzero)
        calc_mod_melt_nonzero = np.copy(surf_mod[-1])
        calc_mod_melt_nonzero[calc_mod_melt_nonzero == 0.] = np.nan
        calc_mod_nonz_mn = np.nanmean(calc_mod_melt_nonzero)
        nonz_bias = np.nanmean(mod_melt_nonzero - melt_nonzero)
    mean_obs = []
    mean_mod = []
    bias = []
    errors = []
    r2s = []
    rmses = []
    for i in np.arange(len(surf_met_obs)):
        b = surf_mod[i] - surf_met_obs[i]
        errors.append(b)
        mean_obs.append(np.mean(surf_met_obs[i]))
        mean_mod.append(np.mean(surf_mod[i]))
        bias.append(mean_mod[i] - mean_obs[i])
        slope, intercept, r2, p, sterr = scipy.stats.linregress(surf_met_obs[i], surf_mod[i])
        r2s.append(r2)
        mse = mean_squared_error(y_true = surf_met_obs[i], y_pred = surf_mod[i])
        rmse = np.sqrt(mse)
        rmses.append(rmse)
    df = pd.DataFrame(index = idx)
    df['obs mean'] = pd.Series(mean_obs, index = idx)
    df['mod mean'] = pd.Series(mean_mod, index = idx)
    df['bias'] =pd.Series(bias, index=idx)
    df['rmse'] = pd.Series(rmses, index = idx)
    df['% RMSE'] = ( df['rmse']/df['obs mean'] ) * 100
    df['correl'] = pd.Series(r2s, index = idx)
    for i in range(len(surf_mod)):
        slope, intercept, r2, p, sterr = scipy.stats.linregress(surf_met_obs[i], surf_mod[i])
        print(idx[i])
        print('\nr2 = %s\n' % r2)
    if daily == 'yes':
        df.to_csv(filepath + trimmed_vars['year'] + '_' + station_dict[station] + '_bias_RMSE_daily_mean.csv') # change to be station-specific
    elif daily == 'no':
        df.to_csv(filepath + trimmed_vars['year'] + '_' + station_dict[station] + '_bias_RMSE.csv')
    print(df)

#calc_bias(srs_14_trimmed, ANN_14, station = 'AWS14_SEB_2009-2017_norp.csv', daily = 'yes', foehn='yes')
#ANN_15 = ANN_15.interpolate('linear', limit_direction = 'both')
#calc_bias(srs_15_trimmed, ANN_15, station = 'AWS15_hourly_2009-2014.csv', daily = 'yes')
#calc_bias(srs_17_trimmed, ANN_17, station = 'AWS17_SEB_2011-2015_norp.csv', daily = 'yes')
#ANN_18 = ANN_18.interpolate('linear', limit_direction = 'both')
#calc_bias(srs_18_trimmed, ANN_18, station = 'AWS18_SEB_2014-2017_norp.csv', daily = 'yes')

#inlet_df = pd.concat([AWS18_df, AWS17_df])
#inlet_df = inlet_df.groupby(inlet_df.index).mean()

#calc_bias(srs_14_trimmed, ANN_14, station = 'AWS14_SEB_2009-2017_norp.csv', daily = 'no')
#ANN_15 = ANN_15.interpolate('linear', limit_direction = 'both')
#calc_bias(srs_15_trimmed, ANN_15, station = 'AWS15_hourly_2009-2014.csv', daily = 'no')
#calc_bias(srs_17_trimmed, ANN_17, station = 'AWS17_SEB_2011-2015_norp.csv', daily = 'no')
#ANN_18 = ANN_18.interpolate('linear', limit_direction = 'both')
#calc_bias(srs_18_trimmed, ANN_18, station = 'AWS18_SEB_2014-2017_norp.csv', daily = 'no')

def calc_seas_bias(vars_yr, station, daily, foehn, load_again):
    # reload AWS data
    os.chdir(filepath)
    AWS_var = load_all_AWS(station, daily)
    length = min(vars_yr['Tair'].shape[0], AWS_var['Tair_2m'].shape[0])
    idx = ['Ts', 'Tair', 'Tmin', 'Tmax', 'RH', 'FF', 'P', 'u', 'v', 'Swdown', 'SWup', 'SWnet', 'LWdown', 'LWup',
           'LWnet', 'HS', 'HL', 'Etot', 'Emelt', 'datetime']
    if load_again == 'yes':
        if station == 'AWS15_hourly_2009-2014.csv':
            surf_met_obs = [AWS_var['Tair_2m'][:length].values, AWS_var['Tair_min'][:length].values, AWS_var['Tair_max'][:length].values,
                            AWS_var['RH'][:length].values, AWS_var['FF_10m'][:length].values, AWS_var['P'][:length].values,
                            AWS_var['u'][:length].values, AWS_var['v'][:length].values, AWS_var['SWin'][:length].values,
                            AWS_var['SWout'][:length].values* -1., AWS_var['SWnet'][:length].values, AWS_var['LWin'][:length].values,
                            AWS_var['LWout'][:length].values* -1.,  AWS_var['LWnet'][:length].values, AWS_var.index.values]
            surf_mod = [np.mean(vars_yr['Tair'].data[:length, lat_dict[station_dict[station]] - 1:lat_dict[station_dict[station]] + 1,
                        lon_dict[station_dict[station]] - 1:lon_dict[station_dict[station]] + 1], axis=(1, 2)),
            mins[station_dict[station]],
            maxes[station_dict[station]],
            np.mean(vars_yr['RH'].data[:length, lat_dict[station_dict[station]] - 1:lat_dict[station_dict[station]] + 1,
                    lon_dict[station_dict[station]] - 1:lon_dict[station_dict[station]] + 1], axis=(1, 2)),
            np.mean(
                vars_yr['FF_10m'].data[:length, lat_dict[station_dict[station]] - 1:lat_dict[station_dict[station]] + 1,
                lon_dict[station_dict[station]] - 1:lon_dict[station_dict[station]] + 1], axis=(1, 2)),
            np.mean(
                vars_yr['sfc_P'].data[:length, lat_dict[station_dict[station]] - 1:lat_dict[station_dict[station]] + 1,
                lon_dict[station_dict[station]] - 1:lon_dict[station_dict[station]] + 1], axis=(1, 2)),
            np.mean(vars_yr['u'].data[:length, lat_dict[station_dict[station]] - 1:lat_dict[station_dict[station]] + 1,
                    lon_dict[station_dict[station]] - 1:lon_dict[station_dict[station]] + 1], axis=(1, 2)),
            np.mean(vars_yr['v'].data[:length, lat_dict[station_dict[station]] - 1:lat_dict[station_dict[station]] + 1,
                    lon_dict[station_dict[station]] - 1:lon_dict[station_dict[station]] + 1], axis=(1, 2)),
            np.mean(vars_yr['SWdown'].data[:length,
                lat_dict[station_dict[station]] - 1:lat_dict[station_dict[station]] + 1,
                lon_dict[station_dict[station]] - 1:lon_dict[station_dict[station]] + 1], axis=(1, 2)),
            np.mean( vars_yr['SWup'].data[:length,
                lat_dict[station_dict[station]] - 1:lat_dict[station_dict[station]] + 1,
                lon_dict[station_dict[station]] - 1:lon_dict[station_dict[station]] + 1], axis=(1, 2)),
            np.mean(vars_yr['SWnet'].data[:length,
                lat_dict[station_dict[station]] - 1:lat_dict[station_dict[station]] + 1,
                lon_dict[station_dict[station]] - 1:lon_dict[station_dict[station]] + 1], axis=(1, 2)),
            np.mean( vars_yr['LWdown'].data[:length,
                lat_dict[station_dict[station]] - 1:lat_dict[station_dict[station]] + 1,
                lon_dict[station_dict[station]] - 1:lon_dict[station_dict[station]] + 1], axis=(1, 2)),
            np.mean(vars_yr['LWup'].data[:length,
                lat_dict[station_dict[station]] - 1:lat_dict[station_dict[station]] + 1,
                lon_dict[station_dict[station]] - 1:lon_dict[station_dict[station]] + 1], axis=(1, 2)),
            np.mean(vars_yr['LWnet'].data[:length,
                lat_dict[station_dict[station]] - 1:lat_dict[station_dict[station]] + 1,
                lon_dict[station_dict[station]] - 1:lon_dict[station_dict[station]] + 1], axis=(1, 2)),
            vars_yr['Timesrs']]
            idx = ['Tair', 'Tmin', 'Tmax', 'RH', 'FF', 'P', 'u', 'v', 'Swdown', 'SWup', 'SWnet', 'LWdown', 'LWup', 'LWnet','datetime']
        else:
            surf_met_obs = [AWS_var['Tsobs'][:length].values, AWS_var['Tair_2m'][:length].values, AWS_var['Tair_min'][:length].values, AWS_var['Tair_max'][:length].values,
                            AWS_var['RH'][:length].values, AWS_var['FF_10m'][:length].values, AWS_var['pres'][:length].values, AWS_var['u'][:length].values,
                            AWS_var['v'][:length].values, AWS_var['SWin_corr'][:length].values, AWS_var['SWout'][:length].values* -1., AWS_var['SWnet_corr'][:length].values,
                            AWS_var['LWin'][:length].values, AWS_var['LWout_corr'][:length].values* -1., AWS_var['LWnet_corr'][:length].values,
                            AWS_var['Hsen'][:length].values, AWS_var['Hlat'][:length].values, AWS_var['E'][:length].values,
                            AWS_var['melt_energy'][:length].values, AWS_var.index.values]
            surf_mod = [np.mean(vars_yr['Ts'].data[:length, lat_dict[station_dict[station]] - 1:lat_dict[station_dict[station]] + 1,
                                lon_dict[station_dict[station]] - 1:lon_dict[station_dict[station]] + 1], axis=(1, 2)),
                        np.mean(vars_yr['Tair'].data[:length, lat_dict[station_dict[station]] - 1:lat_dict[station_dict[station]] + 1,
                            lon_dict[station_dict[station]] - 1:lon_dict[station_dict[station]] + 1], axis=(1, 2)),
                        mins[station_dict[station]],maxes[station_dict[station]],
                        np.mean(vars_yr['RH'].data[:length, lat_dict[station_dict[station]] - 1:lat_dict[station_dict[station]] + 1,
                                lon_dict[station_dict[station]] - 1:lon_dict[station_dict[station]] + 1], axis=(1, 2)),
                        np.mean(vars_yr['FF_10m'].data[:length, lat_dict[station_dict[station]] - 1:lat_dict[station_dict[station]] + 1,
                            lon_dict[station_dict[station]] - 1:lon_dict[station_dict[station]] + 1], axis=(1, 2)),
                        np.mean(vars_yr['sfc_P'].data[:length, lat_dict[station_dict[station]] - 1:lat_dict[station_dict[station]] + 1,
                            lon_dict[station_dict[station]] - 1:lon_dict[station_dict[station]] + 1], axis=(1, 2)),
                        np.mean(vars_yr['u'].data[:length, lat_dict[station_dict[station]] - 1:lat_dict[station_dict[station]] + 1,
                                lon_dict[station_dict[station]] - 1:lon_dict[station_dict[station]] + 1], axis=(1, 2)),
                        np.mean(vars_yr['v'].data[:length, lat_dict[station_dict[station]] - 1:lat_dict[station_dict[station]] + 1,
                                lon_dict[station_dict[station]] - 1:lon_dict[station_dict[station]] + 1], axis=(1, 2)),
                        np.mean( vars_yr['SWdown'].data[:length, lat_dict[station_dict[station]] - 1:lat_dict[station_dict[station]] + 1,
                            lon_dict[station_dict[station]] - 1:lon_dict[station_dict[station]] + 1], axis=(1, 2)),
                        np.mean(vars_yr['SWup'].data[:length,
                            lat_dict[station_dict[station]] - 1:lat_dict[station_dict[station]] + 1,
                            lon_dict[station_dict[station]] - 1:lon_dict[station_dict[station]] + 1], axis=(1, 2)),
                        np.mean(vars_yr['SWnet'].data[:length, lat_dict[station_dict[station]] - 1:lat_dict[station_dict[station]] + 1,
                            lon_dict[station_dict[station]] - 1:lon_dict[station_dict[station]] + 1], axis=(1, 2)),
                        np.mean( vars_yr['LWdown'].data[:length,lat_dict[station_dict[station]] - 1:lat_dict[station_dict[station]] + 1,
                            lon_dict[station_dict[station]] - 1:lon_dict[station_dict[station]] + 1], axis=(1, 2)),
                        np.mean( vars_yr['LWup'].data[:length,  (lat_dict[station_dict[station]] - 1):(lat_dict[station_dict[station]] + 1),
                            (lon_dict[station_dict[station]] - 1):(lon_dict[station_dict[station]] + 1)], axis=(1, 2)),
                        np.mean(vars_yr['LWnet'].data[:length, lat_dict[station_dict[station]] - 1:lat_dict[station_dict[station]] + 1,
                            lon_dict[station_dict[station]] - 1:lon_dict[station_dict[station]] + 1], axis=(1, 2)),
                        (np.mean(vars_yr['HS'].data[:length, lat_dict[station_dict[station]] - 1:lat_dict[station_dict[station]] + 1,
                            lon_dict[station_dict[station]] - 1:lon_dict[station_dict[station]] + 1], axis=(1, 2))),
                        (np.mean(vars_yr['HL'].data[:length, lat_dict[station_dict[station]] - 1:lat_dict[station_dict[station]] + 1,
                            lon_dict[station_dict[station]] - 1:lon_dict[station_dict[station]] + 1], axis=(1, 2))),
                        np.mean( vars_yr['Etot'].data[:length, lat_dict[station_dict[station]] - 1:lat_dict[station_dict[station]] + 1,
                            lon_dict[station_dict[station]] - 1:lon_dict[station_dict[station]] + 1], axis=(1, 2)),
                        np.mean( vars_yr['Emelt'].data[:length, lat_dict[station_dict[station]] - 1:lat_dict[station_dict[station]] + 1,
                            lon_dict[station_dict[station]] - 1:lon_dict[station_dict[station]] + 1], axis=(1, 2)), vars_yr['Timesrs']]
        if foehn=='yes':
            foehn_str = 'foehn'
            foehn_df = pd.read_csv(filepath + 'daily_foehn_frequency_all_stations.csv')  # turn this into AWS 14/15/18 average, then diagnose when foehn is shown at one or more
            foehn_df = foehn_df.iloc[vars_yr['start']:vars_yr['end']]
            foehn_df['AWS17'] = foehn_df['AWS18'].copy() #place-holder til I add 17 in
            for f in range(len(surf_mod)-1):
                try:
                    surf_met_obs[f][foehn_df[station_dict[station]]==0]=np.nan
                    surf_mod[f][foehn_df[station_dict[station]]==0]=np.nan # mask values where foehn conditions at that station are not simulated
                except pd.core.indexing.IndexingError:
                    surf_met_obs[f][foehn_df[station_dict[station]] == 0] = np.nan
                    surf_mod[f].values[foehn_df[station_dict[station]] == 0] = np.nan
                except:
                    print("Not THAT one, you silly goose")
        else:
            foehn_str = 'non-foehn'
        obs_df = pd.DataFrame(surf_met_obs[:length], index = idx)
        mod_df = pd.DataFrame(surf_mod, index = idx)
        obs_df.to_csv(filepath + 'Surface_observed_time_series_'  + foehn_str + '_' + station_dict[station] + '.csv')
        mod_df.to_csv(filepath + 'Surface_modelled_time_series_' + foehn_str + '_' +  station_dict[station] + '.csv')
    else:
        if foehn=='yes':
            foehn_str = 'foehn'
        else:
            foehn_str = 'non-foehn'
        obs_df = pd.read_csv(filepath + 'Surface_observed_time_series_' + foehn_str + '_' + station_dict[station] + '.csv',)
        mod_df = pd.read_csv(filepath + 'Surface_modelled_time_series_' + foehn_str + '_' + station_dict[station] + '.csv')
        mod_df.index=idx
        obs_df.index=idx
    mod_df=mod_df.transpose()
    obs_df = obs_df.transpose()[:length]
    obs_df['datetime'] = mod_df['datetime']
    mod_DJF = mod_df.loc[
        (mod_df['datetime'].dt.month == 1) | (mod_df['datetime'].dt.month == 2) | (mod_df['datetime'].dt.month == 12)]
    obs_DJF = obs_df.loc[
        (obs_df['datetime'].dt.month == 1) | (obs_df['datetime'].dt.month == 2) | (obs_df['datetime'].dt.month == 12)]
    mod_MAM = mod_df.loc[
        (mod_df['datetime'].dt.month == 3) | (mod_df['datetime'].dt.month == 4) | (mod_df['datetime'].dt.month == 5)]
    obs_MAM = obs_df.loc[
        (obs_df['datetime'].dt.month == 3) | (obs_df['datetime'].dt.month == 4) | (obs_df['datetime'].dt.month == 5)]
    mod_JJA = mod_df.loc[
        (mod_df['datetime'].dt.month == 6) | (mod_df['datetime'].dt.month == 7) | (mod_df['datetime'].dt.month == 8)]
    obs_JJA = obs_df.loc[
        (obs_df['datetime'].dt.month == 6) | (obs_df['datetime'].dt.month == 7) | (obs_df['datetime'].dt.month == 8)]
    mod_SON = mod_df.loc[
        (mod_df['datetime'].dt.month == 9) | (mod_df['datetime'].dt.month == 10) | (mod_df['datetime'].dt.month == 11)]
    obs_SON = obs_df.loc[
        (obs_df['datetime'].dt.month == 9) | (obs_df['datetime'].dt.month == 10) | (obs_df['datetime'].dt.month == 11)]
    for obs_df, mod_df, seas_name in zip([obs_DJF, obs_MAM, obs_JJA, obs_SON], [mod_DJF, mod_MAM, mod_JJA, mod_SON], ['DJF', 'MAM', 'JJA', 'SON']):
        a = obs_df[:length].drop('datetime', axis = 1)
        b = mod_df[:length].drop('datetime', axis = 1)
        bias = b - a
        sterr = []
        r = []
        p = []
        rmses = []
        a = a.replace(pd.NaT, np.NaN)
        b = b.replace(pd.NaT, np.NaN)
        shp = min(a.shape[0], b.shape[0])
        mask =  ~np.isnan(a['u'][:shp].values.tolist()) # mask nans if using foehn mask
        for vars in idx[:-1]:
            slope, intercept, r_val, p_val, sterr_val = scipy.stats.linregress(a[vars].values[:shp][mask].tolist(), b[vars][:shp].values[mask].tolist()) #np.array(a[vars].values)[mask], np.array(b[vars].values)[mask]) #tolist()
            r.append(r_val)
            p.append(p_val)
            sterr.append(sterr_val)
            mse = mean_squared_error(y_true=a[vars].values[:shp][mask].tolist(), y_pred=b[vars].values[:shp][mask].tolist())
            rmse = np.sqrt(mse)
            rmses.append(rmse)
        stats_df = pd.DataFrame()
        stats_df['bias'] = pd.Series(bias.mean())
        stats_df['r'] = pd.Series(r, index = idx[:-1])
        stats_df['p'] = pd.Series(p, index = idx[:-1])
        stats_df['sterr'] = pd.Series(sterr, index = idx[:-1])
        stats_df['rmse'] = pd.Series(rmses, index = idx[:-1])
        if daily == 'yes':
            stats_df.to_csv(filepath +  seas_name + '_' + station_dict[station] + '_' + foehn_str  + '_seasonal_validation_daily.csv')
            obs_df.to_csv(filepath +  seas_name + '_' + station_dict[station] + '_' + foehn_str + '_observed_time_srs_daily.csv')
            mod_df.to_csv(filepath +  seas_name + '_' + station_dict[station] + '_' + foehn_str + '_modelled_time_srs_daily.csv')
        else:
            stats_df.to_csv(filepath + seas_name + '_' + station_dict[station] + '_' + foehn_str + '_' + '_seasonal_validation.csv')
            obs_df.to_csv(filepath +  seas_name + '_' + station_dict[station] + '_' + foehn_str + '_observed_time_srs.csv')
            mod_df.to_csv(filepath +  seas_name + '_' + station_dict[station] + '_' + foehn_str + '_modelled_time_srs.csv')
    return obs_df, mod_df

obs_df14_foehn, mod_df14_foehn = calc_seas_bias(srs_14_trimmed, station='AWS14_SEB_2009-2017_norp.csv', daily = 'yes', foehn='yes', load_again = 'yes')
obs_df15_foehn, mod_df15_foehn = calc_seas_bias(srs_15_trimmed, station='AWS15_hourly_2009-2014.csv', daily = 'yes', foehn='yes', load_again = 'yes')
obs_df17_foehn, mod_df17_foehn = calc_seas_bias(srs_17_trimmed, station='AWS17_SEB_2011-2015_norp.csv', daily = 'yes', foehn='yes', load_again = 'yes')
obs_df18_foehn, mod_df18_foehn = calc_seas_bias(srs_18_trimmed, station='AWS18_SEB_2014-2017_norp.csv', daily = 'yes', foehn='yes', load_again = 'yes')

print('\nValidating seasonally\n')
Tmax14, Tmin14 = calc_minmax('AWS14_SEB_2009-2017_norp.csv', srs_14_trimmed, daily = 'yes')
Tmax15, Tmin15 = calc_minmax('AWS15_hourly_2009-2014.csv', srs_15_trimmed, daily='yes')
Tmax18, Tmin18 = calc_minmax('AWS18_SEB_2014-2017_norp.csv', srs_18_trimmed, daily='yes')
Tmax17, Tmin17 = calc_minmax('AWS17_SEB_2011-2015_norp.csv', srs_17_trimmed, daily='yes')

mins = {'AWS14': Tmin14,
            'AWS15': Tmin15,
            'AWS17': Tmin17,
            'AWS18': Tmin18}

maxes = {'AWS14': Tmax14,
            'AWS15': Tmax15,
            'AWS17': Tmax17,
            'AWS18': Tmax18}

obs_df14, mod_df14 = calc_seas_bias(srs_14_trimmed, station='AWS14_SEB_2009-2017_norp.csv', daily = 'yes', foehn='no', load_again='yes')
obs_df15, mod_df15 = calc_seas_bias(srs_15_trimmed, station='AWS15_hourly_2009-2014.csv', daily = 'yes', foehn='no', load_again = 'yes')
obs_df17, mod_df17 = calc_seas_bias(srs_17_trimmed, station='AWS17_SEB_2011-2015_norp.csv', daily = 'yes', foehn='no', load_again = 'yes')
obs_df18, mod_df18 = calc_seas_bias(srs_18_trimmed, station='AWS18_SEB_2014-2017_norp.csv', daily = 'yes', foehn='no', load_again = 'yes')

month_num_to_season =   { 1:'DJF',  2:'DJF',
                          3:'MAM',  4:'MAM',  5:'MAM',
                          6:'JJA',  7:'JJA',  8:'JJA',
                          9:'SON', 10:'SON', 11:'SON',
                         12:'DJF'}

grouped =  bias.groupby(lambda x: month_num_to_season.get(x.month))

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


    percentiles = []
    for each_var in [T_surf, T_air, RH, sp_srs]:
        p95 = np.percentile(each_var, 95, axis = (1,2))
        p5 = np.percentile(each_var, 5, axis = (1,2))
        percentiles.append(p5)
        percentiles.append(p95)

def calc_melt(AWS_vars, model_vars):
    Lf = 334000  # J kg-1
    rho_H2O = 999.7  # kg m-3
    melt_m_per_3hr_obs = (AWS_vars['melt_energy'][::3] / (Lf * rho_H2O))  # in mm per hr min
    melt_m_per_s_obs = melt_m_per_3hr_obs * (3600 * 3)  # multiply by (60 seconds * 60 mins) to get flux per second
    total_melt_mmwe = melt_m_per_s_obs * 1000
    obs_total_melt_cmv = np.cumsum(total_melt_mmwe, axis=0)[-1]
    melt_m_per_hr_mod = (model_vars['Emelt'] / (Lf * rho_H2O))  # in mm per 30 min
    melt_m_per_s_mod = melt_m_per_hr_mod * (3600 * 3)  # multiply by (60 seconds * 30 mins) to get flux per second
    total_melt_mmwe = melt_m_per_s_mod * 1000
    mod_total_melt_cmv = np.cumsum(total_melt_mmwe, axis=0)[-1]
    print(obs_total_melt_cmv, mod_total_melt_cmv)
    return obs_total_melt_cmv, mod_total_melt_cmv

#obs_total_melt_cmv, mod_total_melt_cmv = calc_melt(AWS_vars = AWS14_SEB, model_vars = vars_2011)

def calc_nonzero_melt(year, station, var_yr):
    AWS_var, DJF, MAM, JJA, SON = load_AWS(year = year, station = station)
    AWS_melt = AWS_var['melt_energy']
    melt_nonzero = np.copy(AWS_melt)
    melt_nonzero[melt_nonzero == 0.] = np.nan
    AWS_nonz_mn = np.nanmean(melt_nonzero)
    mod_melt_nonzero = np.copy(np.mean(var_yr['Emelt'].data[:,lat_dict[station_dict[station]]-1:lat_dict[station_dict[station]]+1, lon_dict[station_dict[station]]-1:lon_dict[station_dict[station]]+1], axis = (1,2)))
    mod_melt_nonzero[mod_melt_nonzero == 0.] = np.nan
    mod_nonz_mn = np.nanmean(mod_melt_nonzero)
    nonz_bias = np.nanmean(mod_melt_nonzero - melt_nonzero[::3])
    return AWS_nonz_mn, mod_nonz_mn, nonz_bias


def calc_melt_days(melt_var):
    melt = np.copy(melt_var)
    out = melt.reshape(-1, 8, melt.shape[1], melt.shape[2]).sum(1)
    melt_days = np.count_nonzero(out, axis = 0)
    return melt_days

#melt_days_calc = calc_melt_days(vars_2012['Emelt_calc'].data)
#melt_days = calc_melt_days(vars_2012['Emelt'].data)

#np.mean(melt_days_calc)
#np.mean(melt_days)

def calc_melt_duration(melt_var):
    melt = np.copy(melt_var)
    melt_periods = np.count_nonzero(melt, axis = 0)
    melt_periods = melt_periods*3. # multiply by 3 to get number of hours per year (3-hourly data)
    return melt_periods

#melt_dur_calc = calc_melt_duration(vars_2011['Emelt_calc'].data)
#melt_dur = calc_melt_duration(vars_2011['Emelt'].data)

#np.mean(melt_dur_calc)
#np.mean(melt_dur)

## ---------------------------------------- PLOTTING ROUTINES ------------------------------------------------- ##

## Set up plotting options
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Segoe UI', 'Helvetica', 'Liberation sans', 'Tahoma', 'DejaVu Sans', 'Verdana']

from windrose import WindroseAxes
import matplotlib.cm as cm

#AWS_dict = {AWS14_SEB: 'AWS14',
#            AWS15: 'AWS15',
#            AWS17: 'AWS17',
#            AWS18_SEB: 'AWS18'}

#yr_dict = {vars_2014: '2014',
#           vars_2012: '2012',
#           vars_2016: '2016',
#           vars_2017: '2017'}

def wind_rose(year, station):
    vars_yr = load_vars(year, mn = 'no')
    AWS_var = load_AWS(station, year)
    fig = plt.figure(figsize = (16,8))
    rect = [0.05, 0.1, 0.45, 0.6]
    wa = WindroseAxes(fig, rect)
    fig.add_axes(wa)
    # define data limits
    max_mod = max(np.mean(vars_yr['FF_10m'][:,lat_index14-1:lat_index14+1, lon_index14-1:lon_index14+1].data, axis = (1,2)))
    max_obs = max(AWS_var['FF'])
    wa.set_title('Observed', fontsize = 28, color = 'dimgrey', pad = 50)
    wa.axes.spines['polar'].set_visible(False)
    wa.tick_params(axis='both', which='both', labelsize=24, tick1On=False, tick2On=False, labelcolor='dimgrey', pad=10)
    wa.bar(AWS_var['WD'], AWS_var['FF'], bins = np.arange(0, max(max_mod, max_obs),4), cmap = plt.get_cmap('viridis'), normed = True,opening=0.8, edgecolor='white')
    wa.set_yticklabels([])
    rect = [0.55, 0.1, 0.45, 0.6]
    wa = WindroseAxes(fig, rect)
    fig.add_axes(wa)
    wa.set_title('Modelled', fontsize=28, color='dimgrey', pad = 50)
    wa.bar(np.mean(vars_yr['WD'][:,lat_index14-1:lat_index14+1, lon_index14-1:lon_index14+1].data, axis = (1,2)),
           np.mean(vars_yr['FF_10m'][:,lat_index14-1:lat_index14+1, lon_index14-1:lon_index14+1].data, axis = (1,2)),
           bins = np.arange(0, max(max_mod, max_obs),4), cmap = plt.get_cmap('viridis'),  normed = True, opening=0.8, edgecolor='white')
    lgd = wa.set_legend( bbox_to_anchor=(-0.5, 0.9))
    frame = lgd.get_frame()
    frame.set_facecolor('white')
    for ln in lgd.get_texts():
        plt.setp(ln, color='dimgrey', fontsize = 18)
    lgd.get_frame().set_linewidth(0.0)
    wa.axes.spines['polar'].set_visible(False)
    wa.set_yticklabels([])
    wa.tick_params(axis='both', which='both', labelsize=24, tick1On=False, tick2On=False, labelcolor='dimgrey', pad=10)
    plt.savefig('/users/ellgil82/figures/Hindcast/validation/wind_rose_' + station + '_' + year + '.png')
    plt.savefig('/users/ellgil82/figures/Hindcast/validation/wind_rose_' + station + '_' + year + '.eps')
    plt.show()

#wind_rose(year = '2011', station = 'AWS14')

def plot_diurnal(var, domain):
    orog = iris.load_cube('OFCAP_orog.nc', 'surface_altitude')
    lsm = iris.load_cube('OFCAP_lsm.nc', 'land_binary_mask')
    orog = orog[0,0,:,:]
    lsm = lsm[0,0,:,:]
    # Load diurnal cycles in
    if var == 'SEB':
        diur_SWdown = iris.load_cube('diurnal_SWdown.nc', 'surface_downwelling_shortwave_flux_in_air')
        diur_SWup = iris.load_cube('diurnal_SWup.nc', 'Net short wave radiation flux')
        diur_SWnet = iris.load_cube('diurnal_SWnet.nc', 'Net short wave radiation flux')
        diur_LWnet = iris.load_cube('diurnal_LWnet.nc', 'surface_net_downward_longwave_flux')
        diur_LWdown = iris.load_cube('diurnal_LWdown.nc', 'IR down')
        diur_LWup = iris.load_cube('diurnal_LWup.nc', 'surface_net_downward_longwave_flux')
        diur_HL =iris.load_cube('diurnal_HL.nc', 'Latent heat flux')
        diur_HS = iris.load_cube('diurnal_HS.nc', 'surface_upward_sensible_heat_flux')
        vars_yr = {'SWdown': diur_SWdown, 'SWnet': diur_SWnet, 'SWup': diur_SWup, 'LWdown': diur_LWdown, 'LWnet': diur_LWnet, 'LWup': diur_LWup, 'HL': diur_HL, 'HS':  diur_HS}
        colour_dict = {'SWdown': '#6fb0d2', 'SWnet': '#6fb0d2', 'SWup': '#6fb0d2', 'LWdown': '#86ad63', 'LWnet': '#86ad63', 'LWup': '#86ad63','HL': '#33a02c', 'HS': '#1f78b4'}
        UTC_3 = pd.DataFrame()
        for x in vars_yr:
            UTC_3[x] = np.concatenate((np.mean(vars_yr[x][5:, 0, 199:201, 199:201].data, axis=(1, 2)),
                                       np.mean(vars_yr[x][:5, 0, 199:201, 199:201].data, axis=(1, 2))), axis=0)
    elif var == 'met':
        diur_Ts = iris.load_cube('diurnal_Ts.nc','surface_temperature')
        diur_Tair = iris.load_cube('diurnal_Tair.nc', 'air_temperature')
        for i in [diur_Tair, diur_Ts]:
            i.convert_units('celsius')
        diur_u = iris.load_cube('diurnal_u.nc', 'eastward_wind')
        diur_v = iris.load_cube('diurnal_v.nc', 'northward_wind')
        diur_v = diur_v[:,:,1::]
        real_lon, real_lat = rotate_data(diur_v, 2, 3)
        real_lon, real_lat = rotate_data(diur_u, 2, 3)
        diur_q = iris.load_cube('diurnal_q.nc', 'specific_humidity')
        vars_yr = {'Ts': diur_Ts, 'Tair': diur_Tair, 'u': diur_u, 'v': diur_v,'q': diur_q}
        colour_dict = {'Ts': '#dd1c77', 'Tair': '#91003f', 'ff': '#238b45', 'q': '#2171b5'}
        if domain == 'ice shelf' or domain == 'Larsen C':
            # Create Larsen mask
            Larsen_mask = np.zeros((400, 400))
            lsm_subset = lsm.data[35:260, 90:230]
            Larsen_mask[35:260, 90:230] = lsm_subset
            Larsen_mask[orog.data > 25] = 0
            Larsen_mask = np.logical_not(Larsen_mask)
            UTC_3 = pd.DataFrame()
            for x in vars_yr:
                UTC_3[x] = np.ma.concatenate((np.ma.masked_array(vars_yr[x][5:, 0, :, :].data, mask=np.broadcast_to(Larsen_mask, vars_yr[x][5:, 0, :, :].shape)).mean(axis=(1, 2)),
                                              np.ma.masked_array(vars_yr[x][:5, 0, :, :].data, mask=np.broadcast_to(Larsen_mask, vars_yr[x][:5, 0, :, :].shape)).mean(axis=(1, 2))), axis=0)
        elif domain == 'AWS 14' or domain == 'AWS14':
            UTC_3 = pd.DataFrame()
            diur_ff = iris.cube.Cube(data = np.sqrt(vars_yr['v'].data**2)+(vars_yr['u'].data**2))
            UTC_3['ff'] = np.concatenate((np.mean(diur_ff[5:, 0, 199:201, 199:201].data, axis=(1, 2)),
                                       np.mean(diur_ff[:5, 0, 199:201, 199:201].data, axis=(1, 2))), axis=0)
            for x in vars_yr:
                UTC_3[x] = np.concatenate((np.mean(vars_yr[x][5:, 0, 199:201, 199:201], axis=(1, 2)),
                                           np.mean(vars_yr[x][:5, 0, 199:201, 199:201], axis=(1, 2))), axis=0)
        elif domain == 'AWS 15' or domain == 'AWS15':
            UTC_3 = pd.DataFrame()
            for x in vars_yr:
                UTC_3[x] = np.concatenate((np.mean(vars_yr[x][5:, 0, 161:163, 182:184].data, axis=(1, 2)),
                                           np.mean(vars_yr[x][:5, 0, 161:163, 182:184].data, axis=(1, 2))), axis=0)
            diur_ff = iris.cube.Cube(data=np.sqrt(vars_yr['v'].data ** 2) + (vars_yr['u'].data ** 2))
            UTC_3['ff'] = np.concatenate((np.mean(diur_ff[5:, 0, 161:163, 182:184].data, axis=(1, 2)),
                                       np.mean(diur_ff[:5, 0,  161:163, 182:184].data, axis=(1, 2))), axis=0)
    ## Set up plotting options
    if var == 'SEB':
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.spines['top'].set_visible(False)
        plt.setp(ax.spines.values(), linewidth=3, color='dimgrey')
        ax.spines['right'].set_visible(False)
        [l.set_visible(False) for (w, l) in enumerate(ax.xaxis.get_ticklabels()) if w % 2 != 0]
        ax.set_ylabel('Mean energy \nflux (W m$^{-2}$', fontname='SegoeUI semibold', color='dimgrey', rotation=0,fontsize=28, labelpad=75)
        for x in vars_yr:
            ax.plot(UTC_3[x].data, color=colour_dict[x], lw=2)
        plt.savefig('/users/ellgil82/figures/Cloud data/OFCAP_period/OFCAP_diurnal_SEB.png', transparent = True)
        plt.savefig('/users/ellgil82/figures/Cloud data/OFCAP_period/OFCAP_diurnal_SEB.eps', transparent=True)
    elif var == 'met':
        ## Set up plotting options
        fig, ax = plt.subplots(2, 2, sharex = True, figsize=(10, 10))
        ax = ax.flatten()
        plot = 0
        for axs in ax:
            axs.spines['top'].set_visible(False)
            plt.setp(axs.spines.values(), linewidth=3, color='dimgrey')
            axs.spines['right'].set_visible(False)
            [l.set_visible(False) for (w, l) in enumerate(axs.xaxis.get_ticklabels()) if w % 2 != 0]
        for x in vars_yr:
            ax[plot].plot(UTC_3[x], color = colour_dict[x], lw = 2)
            ax[plot].set_ylabel(x, fontname='SegoeUI semibold', color='dimgrey', rotation=0, fontsize=28, labelpad=75)
            ax[plot].tick_params(axis='both', which='both', labelsize=24, tick1On=False, tick2On=False, labelcolor='dimgrey', pad=10)
            ax[plot].set_xlabel('Time', fontname='SegoeUI semibold', color='dimgrey', fontsize=28, labelpad=35)
            plot = plot+1
    plt.show()


#wind_rose(vars_2016, AWS14_SEB)

def seas_wind_rose(year, station):
    seas_means, seas_vals = seas_mean([year], station)
    fig = plt.figure(figsize=(16, 16))
    rect = [0.05, 0.55, 0.35, 0.35]
    wa = WindroseAxes(fig, rect)
    fig.add_axes(wa)
    # define data limits
    max_mod = max(seas_vals['wind_speed']['ANN'])
    wa.set_title('DJF', fontsize=28, color='dimgrey', pad=50)
    wa.axes.spines['polar'].set_visible(False)
    wa.tick_params(axis='both', which='both', labelsize=24, tick1On=False, tick2On=False, labelcolor='dimgrey', pad=10)
    wa.bar(seas_vals['WD']['DJF'], seas_vals['wind_speed']['DJF'], bins=np.arange(0, max_mod, 4), cmap=plt.get_cmap('viridis'), normed=True, opening=0.8, edgecolor='white')
    wa.set_yticklabels([])
    rect = [0.55, 0.55, 0.35, 0.35]
    wa = WindroseAxes(fig, rect)
    fig.add_axes(wa)
    wa.set_title('MAM', fontsize=28, color='dimgrey', pad=50)
    wa.axes.spines['polar'].set_visible(False)
    wa.tick_params(axis='both', which='both', labelsize=24, tick1On=False, tick2On=False, labelcolor='dimgrey', pad=10)
    wa.bar(seas_vals['WD']['MAM'], seas_vals['wind_speed']['MAM'], bins=np.arange(0, max_mod, 4), cmap=plt.get_cmap('viridis'), normed=True, opening=0.8, edgecolor='white')
    wa.set_yticklabels([])
    rect = [0.05, 0.05, 0.35, 0.35]
    wa = WindroseAxes(fig, rect)
    fig.add_axes(wa)
    wa.set_title('JJA', fontsize=28, color='dimgrey', pad=50)
    wa.axes.spines['polar'].set_visible(False)
    wa.tick_params(axis='both', which='both', labelsize=24, tick1On=False, tick2On=False, labelcolor='dimgrey', pad=10)
    wa.bar(seas_vals['WD']['JJA'], seas_vals['wind_speed']['JJA'], bins=np.arange(0, max_mod, 4), cmap=plt.get_cmap('viridis'), normed=True, opening=0.8, edgecolor='white')
    wa.set_yticklabels([])
    rect = [0.55, 0.05,0.35, 0.35]
    wa = WindroseAxes(fig, rect)
    fig.add_axes(wa)
    wa.set_title('SON', fontsize=28, color='dimgrey', pad=50)
    wa.axes.spines['polar'].set_visible(False)
    wa.tick_params(axis='both', which='both', labelsize=24, tick1On=False, tick2On=False, labelcolor='dimgrey', pad=10)
    wa.bar(seas_vals['WD']['SON'], seas_vals['wind_speed']['SON'], bins=np.arange(0, max_mod, 4), cmap=plt.get_cmap('viridis'), normed=True, opening=0.8, edgecolor='white')
    wa.set_yticklabels([])
    lgd = wa.set_legend(bbox_to_anchor=(-0.45, 1.))
    frame = lgd.get_frame()
    frame.set_facecolor('white')
    for ln in lgd.get_texts():
        plt.setp(ln, color='dimgrey', fontsize=24)
    lgd.get_frame().set_linewidth(0.0)
    plt.savefig('/users/ellgil82/figures/Hindcast/wind_rose_' + station + '_' + year + '_seasons.png')
    plt.savefig('/users/ellgil82/figures/Hindcast/wind_rose_' + station + '_' + year + '_seasons.eps')
    plt.show()

#seas_wind_rose('2012', 'AWS17')
#seas_wind_rose('2011', 'AWS14')

def SEB_plot():
    fig, ax = plt.subplots(2,2,sharex= True, sharey = True, figsize=(22, 12))
    ax = ax.flatten()
    col_dict = {'0.5 km': '#33a02c', '1.5 km': '#f68080', '4.0 km': '#1f78b4'}
    lab_dict = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g', 7: 'h', 8: 'i', 9: 'j', 10: 'k', 11: 'l' }
    days = mdates.DayLocator(interval=1)
    dayfmt = mdates.DateFormatter('%d %b')
    plot = 0
    for r in ['1.5 km']: # can increase the number of res
        for j, k in zip(['SWin', 'LWin', 'Hsen', 'Hlat'], ['SW_d', 'LW_d', 'SH', 'LH']):
            limits = {'SWin': (-200,400), 'LWin': (-200,400), 'Tair_2m': (-25, 15), 'Tsurf': (-25, 15)}
            titles = {'SWin': 'Downwelling \nShortwave \nRadiation \n(W m$^{-2}$)', 'LWin': 'Downwelling \nLongwave \nRadiation \n(W m$^{-2}$)',
                      'Hsen': 'Sensible \nheat (W m$^{-2}$)', 'Hlat': 'Latent \nheat (W m$^{-2}$)'}
            ax2 = ax[plot].twiny()
            obs = ax[plot].plot(AWS_var['datetime'], AWS_var[j], color='k', linewidth=2.5, label="Cabinet Inlet AWS", zorder = 3)
            ax2.plot(loc_dict[location][0]['Time_srs'], loc_dict[location][0][k], linewidth=2.5, color=col_dict[r], label='*%(r)s UM output for Cabinet Inlet' % locals(), zorder = 5)
            ax2.plot(SEB_4p0['Time_srs'], SEB_4p0[k], linewidth=2.5, color='#1f78b4', label='*%(r)s UM output for Cabinet Inlet' % locals(), zorder=4)
            ax2.axis('off')
            ax2.set_xlim(loc_dict[location][0]['Time_srs'][1], loc_dict[location][0]['Time_srs'][-1])
            ax2.tick_params(axis='both', tick1On = False, tick2On = False)
            ax2.set_ylim([-200,400])
            ax[plot].set_ylabel(titles[j], rotation=0, fontsize=24, color='dimgrey', labelpad=80)
            lab = ax[plot].text(0.08, 0.85, zorder = 6, transform = ax[plot].transAxes, s=lab_dict[plot], fontsize=32, fontweight='bold', color='dimgrey')
            plot = plot + 1
    for axs in [ax[0], ax[2]]:
        axs.yaxis.set_label_coords(-0.3, 0.5)
        axs.spines['right'].set_visible(False)
    for axs in [ax[1], ax[3]]:
        axs.yaxis.set_label_coords(1.3, 0.5)
        axs.yaxis.set_ticks_position('right')
        axs.tick_params(axis='y', tick1On = False)
        axs.spines['left'].set_visible(False)
    for axs in [ax[2], ax[3]]:
        plt.setp(axs.get_yticklabels()[-2], visible=False)
        axs.xaxis.set_major_formatter(dayfmt)
        #plt.setp(axs.get_xticklabels()[])
    for axs in ax:
        axs.spines['top'].set_visible(False)
        plt.setp(axs.spines.values(), linewidth=2, color='dimgrey', )
        axs.set_xlim(AWS_var['datetime'][1], AWS_var['datetime'][-1])
        axs.tick_params(axis='both', which='both', labelsize=24, tick1On=False, tick2On=False, labelcolor='dimgrey', pad=10)
        [l.set_visible(False) for (w, l) in enumerate(axs.yaxis.get_ticklabels()) if w % 2 != 0]
        [l.set_visible(False) for (w, l) in enumerate(axs.xaxis.get_ticklabels()) if w % 2 != 0]
    ax[0].tick_params(axis='both', which='both', labelsize=24, tick1On = False, tick2On = False)
    #ax[3].fill_between(loc_dict[location][0]['Time_srs'], loc_dict[location][0]['percentiles'][2], loc_dict[location][0]['percentiles'][3], facecolor=col_dict[r], alpha=0.4, zorder=2)
    #ax[2].fill_between(loc_dict[location][0]['Time_srs'], loc_dict[location][0]['percentiles'][0], loc_dict[location][0]['percentiles'][1], facecolor=col_dict[r], alpha=0.4, zorder=2)
    #ax[1].fill_between(loc_dict[location][0]['Time_srs'], loc_dict[location][0]['percentiles'][8], loc_dict[location][0]['percentiles'][9], facecolor=col_dict[r], alpha=0.4, zorder=2)
    #ax[0].fill_between(loc_dict[location][0]['Time_srs'], loc_dict[location][0]['percentiles'][4], loc_dict[location][0]['percentiles'][5], facecolor=col_dict[r], alpha=0.4, zorder=2)
    #Legend
    lns = [Line2D([0],[0], color='k', linewidth = 2.5)]
    labs = ['Observations from Cabinet Inlet']
    for r in ['1.5 km', '4.0 km']:
        lns.append(Line2D([0],[0], color=col_dict[r], linewidth = 2.5))
        labs.append(r[0]+'.'+r[2]+' km UM output for Cabinet Inlet')#'1.5 km UM output for Cabinet Inlet')#
    lgd = ax[1].legend(lns, labs, bbox_to_anchor=(0.55, 1.1), loc=2, fontsize=20)
    frame = lgd.get_frame()
    frame.set_facecolor('white')
    for ln in lgd.get_texts():
        plt.setp(ln, color='dimgrey')
    lgd.get_frame().set_linewidth(0.0)
    plt.subplots_adjust(wspace = 0.05, hspace = 0.05, top = 0.95, right = 0.8, left = 0.2, bottom = 0.08)
    plt.savefig('/users/ellgil82/figures/Wintertime melt/Re-runs/SEB_'+case_study+'_both_res_no_range.png', transparent = True)
    plt.savefig('/users/ellgil82/figures/Wintertime melt/Re-runs/SEB_'+case_study+'_both_res_no_range.eps', transparent = True)
    plt.savefig('/users/ellgil82/figures/Wintertime melt/Re-runs/SEB_'+case_study+'_both_res_no_range.pdf', transparent = True)
    #plt.show()

#SEB_plot()

# Calculate daily mean data of AWS 14 and 15, then AWS 17 and 18
#for station, AWS_var in zip(['AWS14', 'AWS15', 'AWS17', 'AWS18'], [ANN_14, ANN_15, ANN_17, ANN_18]):
#    AWS_var.to_csv(filepath+station+'_daymn.csv')

def calc_monthly_data():
    # Load observed series at each station and turn into DataFrame
    AWS14_daymn = pd.read_csv('AWS14_daymn.csv')
    AWS15_daymn = pd.read_csv('AWS15_daymn.csv')
    AWS17_daymn = pd.read_csv('AWS17_daymn.csv')
    AWS18_daymn = pd.read_csv('AWS18_daymn.csv')
    #for i in [AWS14_daymn, AWS15_daymn, AWS17_daymn, AWS18_daymn]:
    #    i.index = i['Date']
    #AWS14_df = pd.DataFrame()
    #AWS15_df = pd.DataFrame()
    #AWS17_df = pd.DataFrame()
    #AWS18_df = pd.DataFrame()
    # Create dataframe
    #for j in ['Tair', 'Ts', 'FF_10m',  'u', 'v',  'sfc_P', 'RH', 'SWdown', 'SWup', 'SWnet', 'LWdown', 'LWup', 'LWnet', 'HS', 'HL', 'Etot', 'Emelt']:
    #    try:
    #        AWS14_df[j] = np.mean(full_srs[j][:,lat_index14 - 1:lat_index14 + 1, lon_index14 - 1:lon_index14 + 1].data, axis=(1, 2))
    #        AWS15_df[j] = np.mean(full_srs[j][:,lat_index15 - 1:lat_index15 + 1, lon_index15 - 1:lon_index15 + 1].data, axis=(1, 2))
    #        AWS17_df[j] = np.mean(full_srs[j][:,lat_index17 - 1:lat_index17 + 1, lon_index17 - 1:lon_index17 + 1].data, axis=(1, 2))
    #        AWS18_df[j] = np.mean(full_srs[j][:,lat_index18 - 1:lat_index18 + 1, lon_index18 - 1:lon_index18 + 1].data, axis=(1, 2))
    #    except AttributeError:
    #        AWS14_df[j] = np.mean(full_srs[j][:, lat_index14 - 1:lat_index14 + 1, lon_index14 - 1:lon_index14 + 1], axis=(1, 2))
    #        AWS15_df[j] = np.mean(full_srs[j][:, lat_index15 - 1:lat_index15 + 1, lon_index15 - 1:lon_index15 + 1], axis=(1, 2))
    #        AWS17_df[j] = np.mean(full_srs[j][:, lat_index17 - 1:lat_index17 + 1, lon_index17 - 1:lon_index17 + 1], axis=(1, 2))
    #        AWS18_df[j] = np.mean(full_srs[j][:, lat_index18 - 1:lat_index18 + 1, lon_index18 - 1:lon_index18 + 1], axis=(1, 2))
    # Reset index
    #for k in [AWS14_df, AWS15_df, AWS17_df, AWS18_df]:
    #    k.index = pd.date_range(datetime(1997,12,31,23,59,59),datetime(2017,12,31,23,59,59), freq ='D')
    #    k.resample('M').mean()
    #inlet_df = pd.concat([AWS18_df, AWS17_df])
    #inlet_df = inlet_df.groupby(inlet_df.index).mean()
    #iceshelf_df = pd.concat([AWS14_df, AWS15_df])
    #iceshelf_df = iceshelf_df.groupby(iceshelf_df.index).mean()
    #for g in [inlet_df, iceshelf_df]:
    #    g['HS'] = f['HS'] * -1.
    #    g['HL'] = f['HL'] * -1.
    #iceshelf_df.to_csv(filepath + 'Modelled_ice_shelf_daily_mean_series.csv')
    #inlet_df.to_csv(filepath + 'Modelled_inlet_daily_mean_series.csv')
    inlet_df = pd.read_csv(filepath + 'Modelled_inlet_daily_mean_series.csv', index_col = 0)
    iceshelf_df = pd.read_csv(filepath + 'Modelled_ice_shelf_daily_mean_series.csv', index_col = 0)
    for df in [inlet_df, iceshelf_df]:
        df['Datetime'] = pd.date_range(datetime(1997,12,31,0,0,0),datetime(2017,12,31,23,59,59), freq = 'D')
        df.index  = df['Datetime']
    iceshelf_monmn = iceshelf_df.resample('M').mean()
    iceshelf_monmax = iceshelf_df.resample('M').quantile(0.95)
    iceshelf_monmin = iceshelf_df.resample('M').quantile(0.05)
    iceshelf_monmn.to_csv(filepath + 'Modelled_ice_shelf_monthly_mean_series.csv')
    iceshelf_monmax.to_csv(filepath + 'Modelled_ice_shelf_monthly_95_series.csv')
    iceshelf_monmin.to_csv(filepath + 'Modelled_ice_shelf_monthly_5_series.csv')
    inlet_monmn = inlet_df.resample('M').mean()
    inlet_monmax = inlet_df.resample('M').quantile(0.95)
    inlet_monmin = inlet_df.resample('M').quantile(0.05)
    inlet_monmn.to_csv(filepath + 'Modelled_inlet_monthly_mean_series.csv')
    inlet_monmax.to_csv(filepath + 'Modelled_inlet_monthly_95_series.csv')
    inlet_monmin.to_csv(filepath + 'Modelled_inlet_monthly_5_series.csv')
    # Repeat for observations
    iceshelf_obs = pd.concat([ANN_14, ANN_15])
    inlet_obs = pd.concat([ANN_17, ANN_18])
    iceshelf_monmn_obs = iceshelf_obs.resample('M').mean()
    iceshelf_monmin_obs = iceshelf_obs.resample('M').quantile(0.05)
    iceshelf_monmax_obs = iceshelf_obs.resample('M').quantile(0.95)
    inlet_monmn_obs = inlet_obs.resample('M').mean()
    inlet_monmin_obs = inlet_obs.resample('M').quantile(0.05)
    inlet_monmax_obs = inlet_obs.resample('M').quantile(0.95)
    inlet_monmn_obs.to_csv(filepath +  'Observed_inlet_monthly_mean_series.csv')
    inlet_monmax_obs.to_csv(filepath + 'Observed_inlet_monthly_95_series.csv')
    inlet_monmin_obs.to_csv(filepath + 'Observed_inlet_monthly_5_series.csv')
    iceshelf_monmn_obs.to_csv(filepath +  'Observed_ice_shelf_monthly_mean_series.csv')
    iceshelf_monmax_obs.to_csv(filepath + 'Observed_ice_shelf_monthly_95_series.csv')
    iceshelf_monmin_obs.to_csv(filepath + 'Observed_ice_shelf_monthly_5_series.csv')
    table_obs = pd.DataFrame(index = ['Inlet Mean', 'Inlet 5th', 'Inlet 95th','Ice shelf Mean', 'Ice shelf 5th', 'Ice shelf 95th' ])
    for j in ['Tair_2m',  'Tsobs',  'FF_10m',  'u', 'v', 'pres', 'RH', 'SWin_corr', 'SWout', 'SWnet_corr', 'LWin', 'LWout_corr', 'LWnet_corr', 'Hsen', 'Hlat', 'E', 'melt_energy']:
        table_obs[j] = pd.Series([inlet_monmn_obs.mean()[j],inlet_monmn_obs.quantile(0.05)[j], inlet_monmn_obs.quantile(0.95)[j], iceshelf_monmn_obs.mean()[j], iceshelf_monmn_obs.quantile(0.05)[j],
                                  iceshelf_monmn_obs.quantile(0.95)[j]], index = ['Inlet Mean', 'Inlet 5th', 'Inlet 95th','Ice shelf Mean', 'Ice shelf 5th', 'Ice shelf 95th' ])
    table_obs = table_obs.transpose()
    table_obs.to_csv(filepath + 'Observed_monthly_stats.csv')
    table_mod = pd.DataFrame(index = ['Inlet Mean', 'Inlet 5th', 'Inlet 95th','Ice shelf Mean', 'Ice shelf 5th', 'Ice shelf 95th' ])
    for j in ['Tair',  'Ts',  'FF_10m', 'u', 'v', 'sfc_P', 'RH', 'SWdown','SWup', 'SWnet', 'LWdown', 'LWup', 'LWnet', 'HS', 'HL', 'Etot', 'Emelt']:
        table_mod[j] = pd.Series([inlet_monmn.mean()[j],inlet_monmn.quantile(0.05)[j], inlet_monmn.quantile(0.95)[j], iceshelf_monmn.mean()[j],
                                  iceshelf_monmn.quantile(0.05)[j], iceshelf_monmn.quantile(0.95)[j]], index = ['Inlet Mean', 'Inlet 5th', 'Inlet 95th','Ice shelf Mean', 'Ice shelf 5th', 'Ice shelf 95th' ])
    table_mod = table_mod.transpose()
    table_mod.to_csv(filepath + 'Modelled_monthly_stats.csv')

calc_monthly_data()

def surf_plot(inlet_df, iceshelf_df, minmax, which_vars):
    fig, ax = plt.subplots(2,2,sharex= True, figsize=(22, 12))
    ax = ax.flatten()
    lab_dict = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g', 7: 'h', 8: 'i', 9: 'j', 10: 'k', 11: 'l' }
    days = mdates.MonthLocator(interval=1)
    dayfmt = mdates.DateFormatter('%m')
    plot = 0
    translation = {'SWdown': 'SWin_corr', 'LWdown': 'LWin', 'HS':'Hsen', 'HL': 'Hlat', 'RH': 'RH', 'FF_10m':'FF_10m', 'Tair': 'Tair_2m', 'Ts': 'Tsobs'}
    # Plot each variable in turn
    if which_vars == 'surf_met':
        var_list = ['RH', 'FF_10m', 'Tair', 'Ts']
        limits = {'RH': (40,100), 'FF_10m': (0,25), 'Tair': (-40, 5), 'Ts': (-40, 5)}
        titles = {'RH': 'Relative \nhumidity (%)',
                  'FF_10m': 'Wind speed \n(m s$^{-1}$)',
                  'Tair': '2 m air \ntemperature ($^{\circ}$C)',
                  'Ts': 'Surface \ntemperature ($^{\circ}$C)'}
    elif which_vars == 'SEB':
        var_list = ['SWdown', 'LWdown', 'HS', 'HL']
        limits = {'SWdown': (0, 600), 'LWdown': (100, 350), 'HS': (-50,150), 'HL': (-100,50)}
        titles = {'SWdown': 'Downwelling \nShortwave \nRadiation \n(W m$^{-2}$)',
                  'LWdown': 'Downwelling \nLongwave \nRadiation \n(W m$^{-2}$)',
                  'HS': 'Sensible heat \nflux (W m$^{-2}$)',
                  'HL': 'Latent heat \nflux (W m$^{-2}$)'}
    for j in var_list:
        inlet = ax[plot].plot(inlet_df.index, inlet_df[j], color='#cc4c02', alpha = 0.75,  label = 'Inlet stations')
        iceshelf = ax[plot].plot(iceshelf_df.index, iceshelf_df[j], color='#045a8d', alpha = 0.75, linewidth=2.5, label = 'Ice shelf stations')
        #ax2 = plt.twiny(ax[plot])
        inlet_obs = ax[plot].plot(inlet_monmn_obs.index, inlet_monmn_obs[translation[j]], color='#803001', marker = 'x', linewidth=2.5, label = 'Inlet stations, obs')
        iceshelf_obs = ax[plot].plot(iceshelf_monmn_obs.index, iceshelf_monmn_obs[translation[j]], color='#022b43', marker = 'x', linewidth=2.5, label = 'Ice shelf stations, obs')
        ax[plot].set_xlim(inlet_df.index[0], inlet_df.index[-1])
        ax[plot].set_ylim(limits[j])
        #ax2.axis('off')
        ax[plot].set_xlim(inlet_df.index[0], inlet_df.index[-1])
        ax[plot].set_ylim(limits[j])#[floor(np.floor(np.min(AWS_var[j])),5),ceil(np.ceil( np.max(AWS_var[j])),5)])
        ax[plot].set_ylabel(titles[j], rotation=0, fontsize=24, color = 'dimgrey', labelpad = 80)
        ax[plot].tick_params(axis='both', which='both', labelsize=24, tick1On = False, tick2On = False)
        lab = ax[plot].text(0.08, 0.85, zorder = 100, transform = ax[plot].transAxes, s=lab_dict[plot], fontsize=32, fontweight='bold', color='dimgrey')
        if minmax == 'yes' or minmax == True:
            ax[plot].fill_between(inlet_df.index, inlet_monmin[j], inlet_monmax[j], color='#cc4c02', alpha = 0.3)
            ax[plot].fill_between(iceshelf_df.index, iceshelf_monmin[j], iceshelf_monmax[j], color='#045a8d', alpha=0.3)
        plot = plot + 1
    for axs in [ax[0], ax[2]]:
        axs.yaxis.set_label_coords(-0.3, 0.5)
        axs.spines['right'].set_visible(False)
    for axs in [ax[1], ax[3]]:
        axs.yaxis.set_label_coords(1.27, 0.5)
        axs.yaxis.set_ticks_position('right')
        axs.tick_params(axis='y', tick1On = False)
        axs.spines['left'].set_visible(False)
    for axs in [ax[0], ax[1]]:
        [l.set_visible(False) for (w, l) in enumerate(axs.yaxis.get_ticklabels()) if w % 2 != 0]
    for axs in ax:
        axs.spines['top'].set_visible(False)
        plt.setp(axs.spines.values(), linewidth=2, color='dimgrey', )
        axs.set_xlim(inlet_df.index[0], inlet_df.index.values[-1])
        axs.tick_params(axis='both', which='both', labelsize=24, tick1On=False, tick2On=False, labelcolor='dimgrey', pad=10)
        [l.set_visible(False) for (w, l) in enumerate(axs.xaxis.get_ticklabels()) if w % 2 != 0]
        #plt.xticks([inlet_df.index[0], vars_yr[k].coord('time').points[-1]])
    # Legend
    lgd = ax[1].legend(bbox_to_anchor=(0.3, 1.1), loc=2, fontsize=20)
    frame = lgd.get_frame()
    frame.set_facecolor('white')
    for ln in lgd.get_texts():
        plt.setp(ln, color='dimgrey')
    lgd.get_frame().set_linewidth(0.0)
    plt.subplots_adjust(wspace = 0.05, hspace = 0.05, top = 0.95, right = 0.84, left = 0.17, bottom = 0.08)
    if minmax == True:
        plt.savefig('/gws/nopw/j04/bas_climate/users/ellgil82/hindcast/figures/' + which_vars + '_inlet_v_ice_shelf_monmn.png')
        plt.savefig('/gws/nopw/j04/bas_climate/users/ellgil82/hindcast/figures/' + which_vars + '_inlet_v_ice_shelf_monmn.eps')
        plt.savefig('/gws/nopw/j04/bas_climate/users/ellgil82/hindcast/figures/' + which_vars + '_inlet_v_ice_shelf_monmn.pdf')
    else:
        plt.savefig('/gws/nopw/j04/bas_climate/users/ellgil82/hindcast/figures/' + which_vars + '_inlet_v_ice_shelf_no_range_monmn.png')
        plt.savefig('/gws/nopw/j04/bas_climate/users/ellgil82/hindcast/figures/' + which_vars + '_inlet_v_ice_shelf_no_range_monmn.eps')
        plt.savefig('/gws/nopw/j04/bas_climate/users/ellgil82/hindcast/figures/' + which_vars + '_inlet_v_ice_shelf_no_range_monmn.pdf')
    plt.show()

#surf_plot(inlet_monmn, iceshelf_monmn, minmax = True, which_vars = 'surf_met')
#surf_plot(inlet_monmn, iceshelf_monmn, minmax = True, which_vars = 'SEB')


#fig, ax = plt.subplots(figsize=(18, 8))
#inlet = ax.plot(inlet_df.index, inlet_df[j], color='#cc4c02', linewidth=2.5, label = 'Inlet stations')
#ceshelf = ax.plot(iceshelf_df.index, iceshelf_df.melt, color='#045a8d', linewidth=2.5, label = 'Ice shelf stations')

def melt_plot(AWS_var, vars_yr, station):
    fig, ax = plt.subplots(figsize = (18,8))
    ax2 = ax.twiny()
    ax2.plot(AWS_var.index, AWS_var['melt_energy'], lw=2, color='k', label = 'Observed $E_{melt}$', zorder = 1)
    ax.plot(AWS_var.index[:-1], vars_yr['Emelt'][:, 0, lat_dict[station], lon_dict[station]].data, lw = 2, color = '#f68080', label = 'Modelled $E_{melt}$', zorder = 5)#color = '#f68080',
    ax2.set_xlim(AWS_var.index[0], AWS_var.index[-1])
    ax.set_xlim(AWS_var.index[0], AWS_var.index[-1])
    days = mdates.DayLocator(interval=1)
    dayfmt = mdates.DateFormatter('%b %Y')
    ax.set_ylim(0, 100)
    ax.set_ylabel('$E_{melt}$ \n(W m$^{-2}$)', rotation = 0, fontsize = 36, labelpad = 100, color = 'dimgrey')
    ax2.axis('off')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.setp(ax.spines.values(), linewidth=2, color='dimgrey', )
    ax.tick_params(axis='both', which='both', labelsize=36, length = 5, width = 2, color = 'dimgrey', labelcolor='dimgrey', pad=10)
    #[l.set_visible(False) for (w, l) in enumerate(ax.yaxis.get_ticklabels()) if w % 2 != 0]
    [l.set_visible(False) for (w, l) in enumerate(ax.xaxis.get_ticklabels()) if w % 2 != 0]
    ax.xaxis.set_major_formatter(dayfmt)
    #xticks = ax.set_xticklabels(['2014', 'Jan 2012', '2014', 'Jan 2012', 'Jan 2015', '2015'])
    yticks = ax.set_yticks([0,50,100])
    #Legend
    lns = [Line2D([0],[0], color='k', linewidth = 2.5),
           Line2D([0],[0], color =   '#f68080', linewidth = 2.5)]
    #       Line2D([0],[0], color =  '#33a02c', linewidth = 2.5)]          #Line2D([0],[0], color = '#f68080', linewidth = 2.5)]
    labs = ['Observed $E_{melt}$', 'Modelled $E_{melt}$']#['Observed melt flux', 'ctrl run', 'BL run']#
    lgd = ax2.legend(lns, labs, bbox_to_anchor=(0.55, 1.1), loc=2, fontsize=28)
    frame = lgd.get_frame()
    frame.set_facecolor('white')
    for ln in lgd.get_texts():
        plt.setp(ln, color='dimgrey')
    lgd.get_frame().set_linewidth(0.0)
    plt.subplots_adjust(left = 0.22, right = 0.95)
    #plt.savefig('/users/ellgil82/figures/Cloud data/OFCAP_period/OFCAP_melt_BL.png', transparent = True)
    plt.savefig('/gws/nopw/j04/bas_climate/users/ellgil82/hindcast/figures/Melt_time_series_obs_v_mod_+ ' + station + '.png', transparent = True)
    plt.savefig('/gws/nopw/j04/bas_climate/users/ellgil82/hindcast/figures/Melt_time_series_obs_v_mod_+ ' + station + '.eps', transparent=True)
    plt.show()

#melt_plot(AWS_var = ANN_18, vars_yr = srs_18_trimmed, station = 'AWS18')

def melt_map(vars_yr, AWS14_var, AWS17_var, calc, which):
    fig, axs  = plt.subplots(1,1, figsize = (10, 10))#, figsize=(20, 12), frameon=False)
    lab_dict = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g', 7: 'h', 8: 'i', 9: 'j', 10: 'k', 11: 'l'} 
    # Set up plot
    axs.spines['right'].set_visible(False)
    axs.spines['left'].set_visible(False)
    axs.spines['top'].set_visible(False)
    axs.spines['bottom'].set_visible(False)
    # Plot LSM and orography
    axs.contour(vars_yr['lon'], vars_yr['lat'], LSM.data, colors='#535454', linewidths=2.5,zorder=5)  # , transform=RH.coord('grid_longitude').coord_system.as_cartopy_projection())
    axs.contour(vars_yr['lon'], vars_yr['lat'],  orog.data, colors='#535454', levels=[50], linewidths=2.5, zorder=6)
    lsm_masked = np.ma.masked_where(LSM.data == 1, LSM.data)
    orog_masked = np.ma.masked_where(orog.data < 50, orog.data)
    # Mask orography above 15 m
    axs.contourf(vars_yr['lon'], vars_yr['lat'], orog_masked, colors = 'w', zorder = 3)
    # Make the sea blue
    axs.contourf(vars_yr['lon'], vars_yr['lat'],  lsm_masked, colors='#a6cee3', zorder=2)
    # Sort out ticks
    axs.tick_params(which='both', axis='both', labelsize=34, labelcolor='dimgrey', pad=15, size=0, tick1On=False, tick2On=False)
    PlotLonMin = np.min(vars_yr['lon'])
    PlotLonMax = np.max(vars_yr['lon'])
    PlotLatMin = np.min(vars_yr['lat'])
    PlotLatMax = np.max(vars_yr['lat'])
    XTicks = np.linspace(PlotLonMin, PlotLonMax, 3)
    XTickLabels = [None] * len(XTicks)
    for i, XTick in enumerate(XTicks):
        if XTick < 0:
            XTickLabels[i] = '{:.0f}{:s}'.format(np.abs(XTick), '$^{\circ}$W')
        else:
            XTickLabels[i] = '{:.0f}{:s}'.format(np.abs(XTick), '$^{\circ}$E')
    plt.sca(axs)
    plt.xticks(XTicks, XTickLabels)
    axs.set_xlim(PlotLonMin, PlotLonMax)
    YTicks = np.linspace(PlotLatMin, PlotLatMax, 4)
    YTickLabels = [None] * len(YTicks)
    for i, YTick in enumerate(YTicks):
        if YTick < 0:
            YTickLabels[i] = '{:.0f}{:s}'.format(np.abs(YTick), '$^{\circ}$S')
        else:
            YTickLabels[i] = '{:.0f}{:s}'.format(np.abs(YTick), '$^{\circ}$N')
    plt.sca(axs)
    plt.yticks(YTicks, YTickLabels)
    axs.set_ylim(PlotLatMin, PlotLatMax)
    # Add plot labels
    #lab = axs.text(-80, -61.5, zorder=10,  s=lab_dict[a], fontsize=32, fontweight='bold', color='dimgrey')
    # Calculate observed melt
    AWS14_var['melt_energy'][AWS14_var['melt_energy'] == 0] = np.nan
    melt_obs14 = np.nanmean(AWS14_var['melt_energy'])
    AWS17_var['melt_energy'][AWS17_var['melt_energy'] == 0] = np.nan
    melt_obs17 = np.nanmean(AWS17_var['melt_energy'])
    x, y = np.meshgrid(vars_yr['lon'], vars_yr['lat'])
    if which == 'days':
        if calc == 'no' or calc == False:
            melt_spatial = np.copy(vars_yr['Emelt'].data)
            melt_spatial[melt_spatial == 0] = np.nan
            melt_spatial = np.nanmean(melt_spatial, axis = 0)
            melt_spatial[(LSM.data == 0)] = 0
            #melt_obs = ((AWS_var['melt_energy']/ (Lf * rho_H2O))*10800)*1000
            # Plot model melt rates
            c = axs.pcolormesh(x, y, melt_spatial, cmap='viridis', vmin=0, vmax=100, zorder=1)
        elif calc == 'yes' or calc == True:
            melt_days_calc = calc_melt_days(vars_2012['Emelt_calc'].data)
            c = axs.pcolormesh(x, y, melt_days_calc, cmap='viridis', vmin=0, vmax=100, zorder=1)
        CBarXTicks = [0, 100]
    elif which == 'duration':
        if calc == 'no' or calc == False:
            melt_dur = calc_melt_duration(vars_2012['Emelt'].data)
            #melt_obs = ((AWS_var['melt_energy']/ (Lf * rho_H2O))*10800)*1000
            # Plot model melt rates
            c = axs.pcolormesh(x, y, melt_dur, cmap='viridis', vmin=0, vmax=500, zorder=1)
        elif calc == 'yes' or calc == True:
            melt_dur_calc = calc_melt_duration(vars_2012['Emelt_calc'].data)
            c = axs.pcolormesh(x, y, melt_dur_calc, cmap='viridis', vmin=0, vmax=500, zorder=1)
        CBarXTicks = [0, 500]
    # Plot observed melt rate at AWS 14
    # Hacky fix to plot melt at right colour
    axs.scatter(-67.01,-61.50, c = melt_obs14, s = 100,marker='o', edgecolors = 'w',vmin=0, vmax=100, zorder = 100, cmap = matplotlib.cm.viridis )
    axs.scatter(-65.93, -61.85, c = melt_obs17, s = 100,marker='o', edgecolors = 'w',vmin=0, vmax=100, zorder = 100, cmap = matplotlib.cm.viridis )
    #axs.plot(-67.01,-61.50,  markerfacecolor='#3d4d8a', markersize=15, marker='o', markeredgecolor='w', zorder=100)#
    # Add colourbar
    CBAxes = fig.add_axes([0.25, 0.15, 0.5, 0.04])
    CBar = plt.colorbar(c, cax=CBAxes, orientation='horizontal', ticks=CBarXTicks)  #
    if which == 'days':
        CBar.set_label('Number of melt days per year', fontsize=34, labelpad=10, color='dimgrey')
    elif which == 'duration':
        CBar.set_label('Melt duration (hours per year)', fontsize=34, labelpad=10, color='dimgrey')
    CBar.solids.set_edgecolor("face")
    CBar.outline.set_edgecolor('dimgrey')
    CBar.ax.tick_params(which='both', axis='both', labelsize=34, labelcolor='dimgrey', pad=10, size=0, tick1On=False, tick2On=False)
    CBar.outline.set_linewidth(2)
    # Sort out plot and save
    plt.subplots_adjust(bottom = 0.3, top = 0.95, wspace = 0.25, hspace = 0.25)
    if calc == 'no' or calc == False:
        plt.savefig('/users/ellgil82/figures/Hindcast/SMB/'+ vars_yr['year']+'_melt_map.png', transparent = True)
        plt.savefig('/users/ellgil82/figures/Hindcast/SMB/'+ vars_yr['year']+'_melt_map.eps', transparent=True)
    elif calc == 'yes' or calc == True:
        plt.savefig('/users/ellgil82/figures/Hindcast/SMB/'+ vars_yr['year']+'_melt_map_calc.png', transparent = True)
        plt.savefig('/users/ellgil82/figures/Hindcast/SMB/'+ vars_yr['year']+'_melt_map_calc.eps', transparent=True)
    plt.show()

#melt_map(vars_yr = vars_2012, AWS14_var = AWS14_SEB, AWS17_var = AWS17_SEB, calc = True, which = 'days')
#melt_map(vars_yr = vars_2012, AWS14_var = AWS14_SEB, AWS17_var = AWS17_SEB, calc = False, which = 'days')
#melt_map(vars_yr = vars_2011, AWS14_var = AWS14_SEB, AWS17_var = AWS17_SEB, calc = True, which = 'duration')
#melt_map(vars_yr = vars_2011, AWS14_var = AWS14_SEB, AWS17_var = AWS17_SEB, calc = False, which = 'duration')
#melt_map(vars_yr = vars_2014, AWS14_var = AWS14_SEB, AWS17_var = AWS17_SEB, calc = False, which = 'duration')


def validation_series(which_vars, location, minmax):
    fig, ax = plt.subplots(2, 2, sharex=True, figsize=(22, 12))
    ax = ax.flatten()
    lab_dict = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g', 7: 'h', 8: 'i', 9: 'j', 10: 'k', 11: 'l'}
    days = mdates.MonthLocator(interval=1)
    dayfmt = mdates.DateFormatter('%m')
    loc_dict = {'AWS14': (srs_14_trimmed, ANN_14), 'AWS15': (srs_15_trimmed, ANN_15), 'AWS17': (srs_17_trimmed, ANN_17), 'AWS18': (srs_18_trimmed, ANN_18)}
    plot = 0
    # Plot each variable in turn
    if which_vars == 'surf_met':
        var_list = ['RH', 'FF_10m', 'Tair', 'Ts']
        obs_name = {'RH': 'RH', 'FF_10m': 'FF_10m', 'Tair': 'Tair_2m', 'Ts': 'Tsobs'}
        limits = {'RH': (40, 100), 'FF_10m': (0, 25), 'Tair': (-40, 5), 'Ts': (-40, 5)}
        titles = {'RH': 'Relative \nhumidity (%)',
                  'FF_10m': 'Wind speed \n(m s$^{-1}$)',
                  'Tair': '2 m air \ntemperature ($^{\circ}$C)',
                  'Ts': 'Surface \ntemperature ($^{\circ}$C)'}
    elif which_vars == 'SEB':
        var_list = ['SWdown', 'LWdown', 'HS', 'HL']
        obs_name = {'SWdown':'SWin_corr', 'LWdown': 'LWin', 'HS': 'Hsen', 'HL': 'Hlat'}
        limits = {'SWdown': (0, 600), 'LWdown': (100, 350), 'HS': (-50, 150), 'HL': (-100, 50)}
        titles = {'SWdown': 'Downwelling \nShortwave \nRadiation \n(W m$^{-2}$)',
                  'LWdown': 'Downwelling \nLongwave \nRadiation \n(W m$^{-2}$)',
                  'HS': 'Sensible heat \nflux (W m$^{-2}$)',
                  'HL': 'Latent heat \nflux (W m$^{-2}$)'}
    for j in var_list:
        mod = ax[plot].plot(loc_dict[location][1].index[:-1], loc_dict[location][0][j][:, lat_dict[location], lon_dict[location]].data, color='#cc4c02', linewidth=2.5, label='Model')
        obs = ax[plot].plot(loc_dict[location][1].index[:-1], loc_dict[location][1][obs_name[j]][:-1], color='k', linewidth=2.5, label='Observations')
        plt.sca(ax[plot])
        plt.xlim(loc_dict[location][1].index[0], loc_dict[location][1].index[-1])
        plt.ylim(limits[j])  # [floor(np.floor(np.min(AWS_var[j])),5),ceil(np.ceil( np.max(AWS_var[j])),5)])
        ax[plot].set_ylabel(titles[j], rotation=0, fontsize=24, color='dimgrey', labelpad=80)
        ax[plot].tick_params(axis='both', which='both', labelsize=24, tick1On=False, tick2On=False)
        lab = ax[plot].text(0.08, 0.85, zorder=100, transform=ax[plot].transAxes, s=lab_dict[plot], fontsize=32,
                            fontweight='bold', color='dimgrey')
        if minmax == 'yes' or minmax == True:
            ax[plot].fill_between(loc_dict[location][0]['Timesrs'], inlet_monmin[j], inlet_monmax[j], color='#cc4c02', alpha=0.4)
        plot = plot + 1
    for axs in [ax[0], ax[2]]:
        axs.yaxis.set_label_coords(-0.3, 0.5)
        axs.spines['right'].set_visible(False)
    for axs in [ax[1], ax[3]]:
        axs.yaxis.set_label_coords(1.27, 0.5)
        axs.yaxis.set_ticks_position('right')
        axs.tick_params(axis='y', tick1On=False)
        axs.spines['left'].set_visible(False)
    for axs in [ax[0], ax[1]]:
        [l.set_visible(False) for (w, l) in enumerate(axs.yaxis.get_ticklabels()) if w % 2 != 0]
    for axs in ax:
        axs.spines['top'].set_visible(False)
        plt.setp(axs.spines.values(), linewidth=2, color='dimgrey', )
        #axs.set_xlim(inlet_df.index[0], inlet_df.index.values[-1])
        axs.tick_params(axis='both', which='both', labelsize=24, tick1On=False, tick2On=False, labelcolor='dimgrey',
                        pad=10)
        [l.set_visible(False) for (w, l) in enumerate(axs.xaxis.get_ticklabels()) if w % 2 != 0]
        # plt.xticks([inlet_df.index[0], vars_yr[k].coord('time').points[-1]])
    # Legend
    lgd = ax[1].legend(bbox_to_anchor=(0.55, 1.1), loc=2, fontsize=20)
    frame = lgd.get_frame()
    frame.set_facecolor('white')
    for ln in lgd.get_texts():
        plt.setp(ln, color='dimgrey')
    lgd.get_frame().set_linewidth(0.0)
    plt.subplots_adjust(wspace=0.05, hspace=0.05, top=0.95, right=0.84, left=0.17, bottom=0.08)
    if minmax == True:
        plt.savefig('/gws/nopw/j04/bas_climate/users/ellgil82/hindcast/figures/' + which_vars + '_' + location + '_validation_time_srs_daymn.png')
        plt.savefig('/gws/nopw/j04/bas_climate/users/ellgil82/hindcast/figures/' + which_vars + '_' + location + '_validation_time_srs_daymn.eps')
        plt.savefig('/gws/nopw/j04/bas_climate/users/ellgil82/hindcast/figures/' + which_vars + '_' + location + '_validation_time_srs_daymn.pdf')
    else:
        plt.savefig('/gws/nopw/j04/bas_climate/users/ellgil82/hindcast/figures/' + which_vars + '_' + location + '_validation_time_srs_no_range_daymn.png')
        plt.savefig('/gws/nopw/j04/bas_climate/users/ellgil82/hindcast/figures/' + which_vars + '_' + location + '_validation_time_srs_no_range_daymn.eps')
        plt.savefig('/gws/nopw/j04/bas_climate/users/ellgil82/hindcast/figures/' + which_vars + '_' + location + '_validation_time_srs_no_range_daymn.pdf')
    #plt.show()


def correl_plot(location):
    loc_dict = {'AWS14': (srs_14_trimmed, ANN_14), 'AWS15': (srs_15_trimmed, ANN_15), 'AWS17': (srs_17_trimmed, ANN_17),
                'AWS18': (srs_18_trimmed, ANN_18)}
    R_net = loc_dict[location][0]['SWnet'].data + loc_dict[location][0]['LWnet'].data
    fig, ax = plt.subplots(4,2, figsize = (16,28))
    ax2 = ax[:,1]
    ax = ax.flatten()
    ax2.flatten()
    lab_dict = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g', 7: 'h'}
    limits = {2: (40, 100), 3: (0, 25), 1: (-40, 5), 0: (-40, 5), 4: (0, 400), 5: (100, 350), 6: (-100,100), 7: (0, 100)}
    for axs in ax:
        axs.spines['top'].set_visible(False)
        axs.spines['right'].set_visible(False)
        plt.setp(axs.spines.values(), linewidth=2, color='dimgrey', )
        axs.set(adjustable='box-forced', aspect='equal')
        axs.tick_params(axis='both', which='both', labelsize=24, tick1On=False, tick2On=False, labelcolor='dimgrey', pad=10)
        axs.yaxis.set_label_coords(-0.4, 0.5)
    for axs in ax2:
        axs.spines['left'].set_visible(False)
        axs.spines['right'].set_visible(True)
        axs.yaxis.set_label_position('right')
        axs.yaxis.set_ticks_position('right')
        axs.tick_params(axis='both', which='both', labelsize=24, tick1On=False, tick2On=False, labelcolor='dimgrey', pad=10)
        axs.yaxis.set_label_coords(1.45, 0.57)
    plot = 0
    min_length = min(loc_dict[location][0]['Tair'].shape[0], loc_dict[location][1]['Tair_2m'].shape[0])
    surf_met_mod = [loc_dict[location][0]['Ts'].data, loc_dict[location][0]['Tair'].data, loc_dict[location][0]['RH'].data,
                    loc_dict[location][0]['FF_10m'].data, loc_dict[location][0]['SWdown'].data, loc_dict[location][0]['LWdown'].data, R_net, loc_dict[location][0]['Emelt'].data ]
    surf_met_obs = [loc_dict[location][1]['Tsobs'], loc_dict[location][1]['Tair_2m'], loc_dict[location][1]['RH'], loc_dict[location][1]['FF_10m'], loc_dict[location][1]['SWin_corr'],
                    loc_dict[location][1]['LWin'], loc_dict[location][1]['Rnet_corr'], loc_dict[location][1]['melt_energy']]
    titles = ['$T_S$ \n($^{\circ}$C)', '$T_{air}$ \n($^{\circ}$C)', '\nRelative Humidity \n(%)', '\nWind speed \n(m s$^{-1}$)', '$SW_\downarrow$ \n(W m$^{-2}$)',  '$LW_\downarrow$ \n(W m$^{-2}$)', '$R_{net}$ \n(W m$^{-2}$)', 'E$_{melt}$ \n(W m$^{-2}$)']
    from itertools import chain
    for i in range(len(surf_met_mod)):
        slope, intercept, r2, p, sterr = scipy.stats.linregress(surf_met_obs[i][:min_length], surf_met_mod[i][:min_length, lat_dict[location], lon_dict[location]])
        if p <= 0.01:
            ax[plot].text(0.9, 0.15, horizontalalignment='right', verticalalignment='top',
                          s='r$^{2}$ = %s' % np.round(r2, decimals=2), fontweight = 'bold', transform=ax[plot].transAxes, size=24,
                          color='dimgrey')
        else:
            ax[plot].text(0.9, 0.15, horizontalalignment='right', verticalalignment='top',
                          s='r$^{2}$ = %s' % np.round(r2, decimals=2), transform=ax[plot].transAxes, size=24,
                          color='dimgrey')
        ax[plot].scatter(surf_met_obs[i][:min_length], surf_met_mod[i][:min_length, lat_dict[location], lon_dict[location]], color = '#f68080', s = 50)
        ax[plot].set_xlim(limits[i][0], limits[i][1])
        ax[plot].set_ylim(limits[i][0], limits[i][1])
        ax[plot].plot(ax[plot].get_xlim(), ax[plot].get_ylim(), ls="--", c = 'k', alpha = 0.8)
        ax[plot].set_xticks([limits[i][0], limits[i][1]])
        ax[plot].set_yticks([limits[i][0], limits[i][1]])
        ax[plot].set_xlabel('Observed %s' % titles[i], size = 24, color = 'dimgrey', rotation = 0, labelpad = 10)
        ax[plot].set_ylabel('Modelled %s' % titles[i], size = 24, color = 'dimgrey', rotation =0, labelpad= 80)
        lab = ax[plot].text(0.1, 0.85, transform = ax[plot].transAxes, s=lab_dict[plot], fontsize=32, fontweight='bold', color='dimgrey')
        plot = plot +1
    plt.subplots_adjust(top = 0.98, hspace = 0.1, bottom = 0.05, wspace = 0.15, left = 0.2, right = 0.8)
    plt.savefig('/gws/nopw/j04/bas_climate/users/ellgil82/hindcast/figures/' + location + '_validation_correlations.png', transparent=True )
    plt.savefig('/gws/nopw/j04/bas_climate/users/ellgil82/hindcast/figures/' + location + '_validation_correlations.eps', transparent=True)
    plt.savefig('/gws/nopw/j04/bas_climate/users/ellgil82/hindcast/figures/' + location + '_validation_correlations.pdf', transparent=True)


for sta in ['AWS14',  'AWS17', 'AWS18']:
    correl_plot(location = sta)
    #for w in ['SEB', 'surf_met']:
    #    validation_series(which_vars = w, location = sta, minmax = 'no')


plt.show()

# Subset for foehn conditions only
station = 'AWS14'
foehn_df = pd.read_csv(filepath + 'daily_foehn_frequency_all_stations.csv')  # turn this into AWS 14/15/18 average, then diagnose when foehn is shown at one or more
'''
full_srs = load_vars('1998-2017', mn = 'yes')
full_srs['Etot'] = iris.cube.Cube(data = (full_srs['LWnet'].data + full_srs['SWnet'].data + full_srs['HL'].data + full_srs['HS'].data))
foehn_subset = {}
for v in full_srs.keys():
    sta_ts = np.mean(full_srs[v].data[:, lat_dict[station_dict[station]] - 1:lat_dict[station_dict[station]] + 1,
            lon_dict[station_dict[station]] - 1:lon_dict[station_dict[station]] + 1], axis=(1, 2)) # calculate mean of variable at the correct station
    sta_ts[foehn_df[station]==0]=np.nan # mask values where foehn conditions at that station are not simulated
    foehn_subset[v] = sta_ts

run validation

ANN, DJF, MAM, JJA, SON = load_all_AWS(station, daily)
AWS_masked = ANN.copy()
AWS_masked[foehn[station]==0] = np.nan #mask obs where foehn not occurring

def calc_seas_bias_foehn(vars_yr, AWS_var, station, daily):
    os.chdir(filepath)
    length = min(vars_yr['Tair'].shape[0], AWS_var['Tair_2m'].shape[0])
    if station == 'AWS15_hourly_2009-2014.csv':
        surf_met_obs = [AWS_var['Tair_2m'][:length].values, AWS_var['Tair_min'][:length].values, AWS_var['Tair_max'][:length].values,
                        AWS_var['RH'][:length].values, AWS_var['FF_10m'][:length].values, AWS_var['P'][:length].values,
                        AWS_var['u'][:length].values, AWS_var['v'][:length].values, AWS_var['SWin'][:length].values,
                        AWS_var['SWout'][:length].values* -1., AWS_var['SWnet'][:length].values, AWS_var['LWin'][:length].values,
                        AWS_var['LWout'][:length].values* -1.,  AWS_var['LWnet'][:length].values, AWS_var.index.values]
        surf_mod = [np.mean(vars_yr['Tair'].data[:length, lat_dict[station_dict[station]] - 1:lat_dict[station_dict[station]] + 1,
                    lon_dict[station_dict[station]] - 1:lon_dict[station_dict[station]] + 1], axis=(1, 2)),
            mins[station_dict[station]],
            maxes[station_dict[station]],
            np.mean(vars_yr['RH'].data[:length, lat_dict[station_dict[station]] - 1:lat_dict[station_dict[station]] + 1,
                    lon_dict[station_dict[station]] - 1:lon_dict[station_dict[station]] + 1], axis=(1, 2)),
            np.mean(
                vars_yr['FF_10m'].data[:length, lat_dict[station_dict[station]] - 1:lat_dict[station_dict[station]] + 1,
                lon_dict[station_dict[station]] - 1:lon_dict[station_dict[station]] + 1], axis=(1, 2)),
            np.mean(
                vars_yr['sfc_P'].data[:length, lat_dict[station_dict[station]] - 1:lat_dict[station_dict[station]] + 1,
                lon_dict[station_dict[station]] - 1:lon_dict[station_dict[station]] + 1], axis=(1, 2)),
            np.mean(vars_yr['u'].data[:length, lat_dict[station_dict[station]] - 1:lat_dict[station_dict[station]] + 1,
                    lon_dict[station_dict[station]] - 1:lon_dict[station_dict[station]] + 1], axis=(1, 2)),
            np.mean(vars_yr['v'].data[:length, lat_dict[station_dict[station]] - 1:lat_dict[station_dict[station]] + 1,
                    lon_dict[station_dict[station]] - 1:lon_dict[station_dict[station]] + 1], axis=(1, 2)),
            np.mean(
                vars_yr['SWdown'].data[:length,
                lat_dict[station_dict[station]] - 1:lat_dict[station_dict[station]] + 1,
                lon_dict[station_dict[station]] - 1:lon_dict[station_dict[station]] + 1], axis=(1, 2)),
            np.mean(
                vars_yr['SWup'].data[:length,
                lat_dict[station_dict[station]] - 1:lat_dict[station_dict[station]] + 1,
                lon_dict[station_dict[station]] - 1:lon_dict[station_dict[station]] + 1], axis=(1, 2)),
            np.mean(
                vars_yr['SWnet'].data[:length,
                lat_dict[station_dict[station]] - 1:lat_dict[station_dict[station]] + 1,
                lon_dict[station_dict[station]] - 1:lon_dict[station_dict[station]] + 1], axis=(1, 2)),
            np.mean(
                vars_yr['LWdown'].data[:length,
                lat_dict[station_dict[station]] - 1:lat_dict[station_dict[station]] + 1,
                lon_dict[station_dict[station]] - 1:lon_dict[station_dict[station]] + 1], axis=(1, 2)),
            np.mean(
                vars_yr['LWup'][:length,
                lat_dict[station_dict[station]] - 1:lat_dict[station_dict[station]] + 1,
                lon_dict[station_dict[station]] - 1:lon_dict[station_dict[station]] + 1], axis=(1, 2)),
            np.mean(
                vars_yr['LWnet'].data[:length,
                lat_dict[station_dict[station]] - 1:lat_dict[station_dict[station]] + 1,
                lon_dict[station_dict[station]] - 1:lon_dict[station_dict[station]] + 1], axis=(1, 2)),
            vars_yr['Timesrs']]
        idx = ['Tair', 'Tmin', 'Tmax', 'RH', 'FF', 'P', 'u', 'v', 'Swdown', 'SWup', 'SWnet', 'LWdown', 'LWup', 'LWnet', 'datetime']
    else:
        surf_met_obs = [AWS_var['Tsobs'][:length].values, AWS_var['Tair_2m'][:length].values, AWS_var['Tair_min'][:length].values, AWS_var['Tair_max'][:length].values,
                        AWS_var['RH'][:length].values, AWS_var['FF_10m'][:length].values, AWS_var['pres'][:length].values, AWS_var['u'][:length].values,
                        AWS_var['v'][:length].values, AWS_var['SWin_corr'][:length].values, AWS_var['SWout'][:length].values* -1., AWS_var['SWnet_corr'][:length].values,
                        AWS_var['LWin'][:length].values, AWS_var['LWout_corr'][:length].values* -1., AWS_var['LWnet_corr'][:length].values,
                        AWS_var['Hsen'][:length].values, AWS_var['Hlat'][:length].values, AWS_var['E'][:length].values,
                        AWS_var['melt_energy'][:length].values, AWS_var.index.values]
        surf_mod = [np.mean(vars_yr['Ts'].data[:length, lat_dict[station_dict[station]] - 1:lat_dict[station_dict[station]] + 1,
                            lon_dict[station_dict[station]] - 1:lon_dict[station_dict[station]] + 1], axis=(1, 2)),
                    np.mean(vars_yr['Tair'].data[:length, lat_dict[station_dict[station]] - 1:lat_dict[station_dict[station]] + 1,
                        lon_dict[station_dict[station]] - 1:lon_dict[station_dict[station]] + 1], axis=(1, 2)),
                    mins[station_dict[station]],
                    maxes[station_dict[station]],
                    np.mean(vars_yr['RH'].data[:length, lat_dict[station_dict[station]] - 1:lat_dict[station_dict[station]] + 1,
                            lon_dict[station_dict[station]] - 1:lon_dict[station_dict[station]] + 1], axis=(1, 2)),
                    np.mean(
                        vars_yr['FF_10m'].data[:length, lat_dict[station_dict[station]] - 1:lat_dict[station_dict[station]] + 1,
                        lon_dict[station_dict[station]] - 1:lon_dict[station_dict[station]] + 1], axis=(1, 2)),
                    np.mean(
                        vars_yr['sfc_P'].data[:length, lat_dict[station_dict[station]] - 1:lat_dict[station_dict[station]] + 1,
                        lon_dict[station_dict[station]] - 1:lon_dict[station_dict[station]] + 1], axis=(1, 2)),
                    np.mean(vars_yr['u'].data[:length, lat_dict[station_dict[station]] - 1:lat_dict[station_dict[station]] + 1,
                            lon_dict[station_dict[station]] - 1:lon_dict[station_dict[station]] + 1], axis=(1, 2)),
                    np.mean(vars_yr['v'].data[:length, lat_dict[station_dict[station]] - 1:lat_dict[station_dict[station]] + 1,
                            lon_dict[station_dict[station]] - 1:lon_dict[station_dict[station]] + 1], axis=(1, 2)),
                    np.mean(
                        vars_yr['SWdown'].data[:length, lat_dict[station_dict[station]] - 1:lat_dict[station_dict[station]] + 1,
                        lon_dict[station_dict[station]] - 1:lon_dict[station_dict[station]] + 1], axis=(1, 2)),
                    np.mean(
                        vars_yr['SWup'].data[:length,
                        lat_dict[station_dict[station]] - 1:lat_dict[station_dict[station]] + 1,
                        lon_dict[station_dict[station]] - 1:lon_dict[station_dict[station]] + 1], axis=(1, 2)),
                    np.mean(
                        vars_yr['SWnet'].data[:length, lat_dict[station_dict[station]] - 1:lat_dict[station_dict[station]] + 1,
                        lon_dict[station_dict[station]] - 1:lon_dict[station_dict[station]] + 1], axis=(1, 2)),
                    np.mean(
                        vars_yr['LWdown'].data[:length,
                        lat_dict[station_dict[station]] - 1:lat_dict[station_dict[station]] + 1,
                        lon_dict[station_dict[station]] - 1:lon_dict[station_dict[station]] + 1], axis=(1, 2)),
                    np.mean(
                        vars_yr['LWup'][:length,  (lat_dict[station_dict[station]] - 1):(lat_dict[station_dict[station]] + 1),
                        (lon_dict[station_dict[station]] - 1):(lon_dict[station_dict[station]] + 1)], axis=(1, 2)),
                    np.mean(
                        vars_yr['LWnet'].data[:length, lat_dict[station_dict[station]] - 1:lat_dict[station_dict[station]] + 1,
                        lon_dict[station_dict[station]] - 1:lon_dict[station_dict[station]] + 1], axis=(1, 2)),
                    (np.mean(
                        vars_yr['HS'].data[:length, lat_dict[station_dict[station]] - 1:lat_dict[station_dict[station]] + 1,
                        lon_dict[station_dict[station]] - 1:lon_dict[station_dict[station]] + 1], axis=(1, 2))),
                    (np.mean(
                        vars_yr['HL'].data[:length, lat_dict[station_dict[station]] - 1:lat_dict[station_dict[station]] + 1,
                        lon_dict[station_dict[station]] - 1:lon_dict[station_dict[station]] + 1], axis=(1, 2))),
                    np.mean(
                        vars_yr['Etot'].data[:length, lat_dict[station_dict[station]] - 1:lat_dict[station_dict[station]] + 1,
                        lon_dict[station_dict[station]] - 1:lon_dict[station_dict[station]] + 1], axis=(1, 2)),
                    np.mean(
                        vars_yr['Emelt'].data[:length, lat_dict[station_dict[station]] - 1:lat_dict[station_dict[station]] + 1,
                        lon_dict[station_dict[station]] - 1:lon_dict[station_dict[station]] + 1], axis=(1, 2)),
                    vars_yr['Timesrs']]
        idx = ['Ts', 'Tair', 'Tmin', 'Tmax', 'RH', 'FF', 'P', 'u', 'v', 'Swdown', 'SWup',  'SWnet', 'LWdown', 'LWup', 'LWnet', 'HS', 'HL', 'Etot', 'Emelt', 'datetime']
    # load time series into dateframe
    obs_df = pd.DataFrame(surf_met_obs)
    mod_df = pd.DataFrame(surf_mod)
    #mod_df = mod_df[:, :surf_met_obs[0].shape[0]]
    obs_df.index = idx
    mod_df.index = idx
    obs_df = obs_df.transpose()
    mod_df = mod_df.transpose()
    # index by datetime
    mod_df.index = vars_yr['Timesrs']
    obs_df.index = AWS_var.index
    months_obs = [g for n, g in obs_df.groupby(pd.TimeGrouper('M'))]
    months_mod = [g for n, g in mod_df.groupby(pd.TimeGrouper('M'))]
    obs_seas = pd.Series()
    mod_seas = pd.Series()
    jan = np.arange(0, 240, 12)
    feb = np.arange(1, 240, 12)
    mar = np.arange(2, 240, 12)
    apr = np.arange(3, 240, 12)
    may = np.arange(4, 240, 12)
    jun = np.arange(5, 240, 12)
    jul = np.arange(6, 240, 12)
    aug = np.arange(7, 240, 12)
    sep = np.arange(8, 240, 12)
    oct = np.arange(9, 240, 12)
    nov = np.arange(10, 240, 12)
    dec = np.arange(11, 240, 12)
    # group into seasons
    for yr in range(20):
        obs_DJF = pd.concat((obs_seas, months_obs[dec[yr]], months_obs[jan[yr]], months_obs[feb[yr]]))
        obs_MAM = pd.concat((obs_seas, months_obs[mar[yr]], months_obs[apr[yr]], months_obs[may[yr]]))
        obs_JJA = pd.concat((obs_seas, months_obs[jun[yr]], months_obs[jul[yr]], months_obs[aug[yr]]))
        obs_SON = pd.concat((obs_seas, months_obs[sep[yr]], months_obs[oct[yr]], months_obs[nov[yr]]))
        mod_DJF = pd.concat((mod_seas, months_mod[dec[yr]], months_mod[jan[yr]], months_mod[feb[yr]]))
        mod_MAM = pd.concat((mod_seas, months_mod[mar[yr]], months_mod[apr[yr]], months_mod[may[yr]]))
        mod_JJA = pd.concat((mod_seas, months_mod[jun[yr]], months_mod[jul[yr]], months_mod[aug[yr]]))
        mod_SON = pd.concat((mod_seas, months_mod[sep[yr]], months_mod[oct[yr]], months_mod[nov[yr]]))
    # run validation on each season in turn
    seas_names = ['DJF', 'MAM', 'JJA', 'SON']
    iteration = 0
    for a, b in zip([obs_DJF, obs_MAM, obs_JJA, obs_SON], [mod_DJF, mod_MAM, mod_JJA, mod_SON]):
        seas_bias = b-a
        sterr = []
        r = []
        p = []
        rmses = []
        for vars in idx[:-1]:
            slope, intercept, r_val, p_val, sterr_val = scipy.stats.linregress(a[vars].values.tolist(), b[vars].values.tolist())
            r.append(r_val)
            p.append(p_val)
            sterr.append(sterr_val)
            mse = mean_squared_error(y_true=a[vars], y_pred=b[vars])
            rmse = np.sqrt(mse)
            rmses.append(rmse)
        stats_df = pd.DataFrame()
        stats_df['bias'] = pd.Series(seas_bias.mean())
        stats_df['r'] = pd.Series(r, index = idx[:-1])
        stats_df['p'] = pd.Series(p, index = idx[:-1])
        stats_df['sterr'] = pd.Series(sterr, index = idx[:-1])
        stats_df['rmse'] = pd.Series(rmses, index = idx[:-1])
        if daily == 'yes':
            stats_df.to_csv(filepath + vars_yr['year'] + '_' + station_dict[station] + '_' + seas_names[iteration]  + '_seasonal_validation_daily_foehn.csv')
            obs_df.to_csv(filepath + vars_yr['year'] + '_' + station_dict[station] + '_observed_time_srs_daily_foehn.csv')
            mod_df.to_csv(filepath + vars_yr['year'] + '_' + station_dict[station] + '_modelled_time_srs_daily_foehn.csv')
        else:
            stats_df.to_csv(filepath + vars_yr['year'] + '_' + station_dict[station] + '_' + seas_names[iteration]  + '_seasonal_validation_foehn.csv')
            obs_df.to_csv(filepath + vars_yr['year'] + '_' + station_dict[station] + '_observed_time_srs_foehn.csv')
            mod_df.to_csv(filepath + vars_yr['year'] + '_' + station_dict[station] + '_modelled_time_srs_foehn.csv')
        iteration = iteration + 1
    return obs_df, mod_df
'''