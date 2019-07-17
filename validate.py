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
        melt = iris.load_cube(filepath  + year + '_land_snow_melt_flux.nc', 'Snow melt heating flux')
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
    Emelt_calc = np.copy(Etot)
    Emelt_calc[Ts.data < -0.025] = 0
    for turb in [HS, HL]:
        turb.data = 0-turb.data
    Emelt_calc = iris.cube.Cube(Emelt_calc)
    Etot = iris.cube.Cube(Etot)
    vars_yr = {'Tair': Tair[:,0,:,:], 'Ts': Ts[:,0,:,:], 'MSLP': MSLP[:,0,:,:], 'sfc_P': sfc_P[:,0,:,:], 'FF_10m': FF_10m[:,0,:,:],
               'RH': RH[:,0,:,:], 'WD': WD[:,0,:,:], 'LWnet': LWnet[:,0,:,:], 'SWnet': SWnet[:,0,:,:], 'SWdown': SWdown[:,0,:,:],
               'LWdown': LWdown[:,0,:,:], 'HL': HL[:,0,:,:], 'HS': HS[:,0,:,:], 'Etot': Etot[:,0,:,:], 'Emelt': melt[:,0,:,:],
               'lon': real_lon, 'lat': real_lat, 'year': year, 'Emelt_calc': Emelt_calc[:,0,:,:]}
    return vars_yr

vars_2012 = load_vars('2012', mn = 'no')
vars_2014 = load_vars('2014', mn = 'no')
vars_2011 = load_vars('2011', mn = 'yes')
#vars_2016 = load_vars('2016')
#vars_2017 = load_vars('2017')

try:
    LSM = iris.load_cube(filepath+'new_mask.nc', 'LAND MASK (No halo) (LAND=TRUE)')
    orog = iris.load_cube(filepath+'orog.nc', 'surface_altitude')
    orog = orog[0,0,:,:]
    LSM = LSM[0,0,:,:]
except iris.exceptions.ConstraintMismatchError:
    print('Files not found')

def load_AWS(station, year):
    ## --------------------------------------------- SET UP VARIABLES ------------------------------------------------##
    ## Load data from AWS 14 and AWS 15 for January 2011
    print('\nDayum grrrl, you got a sweet AWS...')
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
    AWS_srs['time'] = 24*(AWS_srs['Time'] - AWS_srs['day'])
    case = AWS_srs.loc[year+'-01-01':year +'-12-31'] #'2015-01-01':'2015-12-31'
    print '\nconverting times...'
    # Convert times so that they can be plotted
    time_list = []
    for i in case['time']:
        hrs = int(i)                 # will now be 1 (hour)
        mins = int((i-hrs)*60)       # will now be 4 minutes
        secs = int(0 - hrs*60*60 + mins*60) # will now be 30
        j = datetime.time(hour = hrs, minute=mins)
        time_list.append(j)
    case['Time'] = time_list
    case['datetime'] = case.apply(lambda r : pd.datetime.combine(r['Date'],r['Time']),1)
    case['E'] = case['LWnet_corr'].values + case['SWnet_corr'].values + case['Hlat'].values + case['Hsen'].values - case['Gs'].values
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
        os.chdir('/group_workspaces/jasmin4/bas_climate/users/ellgil82/OFCAP/proc_data/')
    elif host == 'bsl':
        os.chdir('/data/mac/ellgil82/hindcast/output/')
    return case, DJF, MAM, JJA, SON

AWS14_SEB, DJF_14, MAM_14, JJA_14, SON_14 = load_AWS('AWS14_SEB_2009-2017_norp', '2011')
#AWS15_SEB = load_AWS('AWS15_hourly_2009-2014.csv', '2014')
AWS17_SEB, DJF_17, MAM_17, JJA_17, SON_17 = load_AWS('AWS17_SEB_2011-2015_norp', '2011')

# Find locations of AWSs
lon_index14, lat_index14 = find_gridbox(-67.01, -61.03, vars_2012['lat'], vars_2012['lon'])
lon_index15, lat_index15 = find_gridbox(-67.34, -62.09, vars_2012['lat'], vars_2012['lon'])
lon_index17, lat_index17 = find_gridbox(-65.93, -61.85, vars_2012['lat'], vars_2012['lon'])
lon_index18, lat_index18 = find_gridbox(-66.48272, -63.37105, vars_2012['lat'], vars_2012['lon'])

file_dict = {'surface_temperature': '_Ts.nc',
             'air_temperature': '_Tair_1p5m.nc',
             'wind_speed': '_FF_10m.nc',
             'x wind component (with respect to grid)': '_u_10m.nc',
             'y wind component (with respect to grid)': '_v_10m.nc',
             'relative_humidity': '_RH_1p5m.nc',
             'surface_air_pressure': '_sfc_P.nc',
             'air_pressure_at_sea_level': '_MSLP.nc',
             'surface_net_downward_longwave_flux': '_surface_LW_net.nc',
             'Net short wave radiation flux': '_surface_SW_net.nc',
             'IR down': '_surface_LW_down.nc',
             'surface_downwelling_shortwave_flux_in_air': '_surface_SW_down.nc',
             'Latent heat flux': '_latent_heat.nc',
             'surface_upward_sensible_heat_flux': '_sensible_heat.nc'}

lat_dict = {'AWS14': lat_index14,
            'AWS15': lat_index15,
            'AWS17': lat_index17,
            'AWS18': lat_index18}

lon_dict = {'AWS14': lon_index14,
            'AWS15': lon_index15,
            'AWS17': lon_index17,
            'AWS18': lon_index18}

station_dict = {'AWS14_SEB_2009-2017_norp': 'AWS14',
             'AWS15_hourly_2009-2014.csv': 'AWS15',
              'AWS17_SEB_2011-2015_norp': 'AWS17'  }

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

seas_means, seas_vals = seas_mean(['2011'], 'AWS17')
seas_means, seas_vals = seas_mean(['2011'], 'AWS14')

def calc_bias(year, station):
    # Calculate bias of time series
    # Forecast error
    AWS_var, DJF, MAM, JA, SON = load_AWS(station, year)
    AWS_var = AWS_var[::3]
    vars_yr = load_vars(year, mn = 'no')
    surf_met_obs = [AWS_var['Tsobs'], AWS_var['Tair_2m'], AWS_var['RH'], AWS_var['FF_10m'], AWS_var['pres'], AWS_var['WD'], AWS_var['SWin_corr'], AWS_var['LWin'], AWS_var['SWnet_corr'], AWS_var['LWnet_corr'],
                    AWS_var['Hsen'], AWS_var['Hlat'], AWS_var['melt_energy'],AWS_var['melt_energy']]
    surf_mod = [np.mean(vars_yr['Ts'].data[:, lat_dict[station_dict[station]]-1:lat_dict[station_dict[station]]+1, lon_dict[station_dict[station]]-1:lon_dict[station_dict[station]]+1], axis = (1,2)),
                np.mean(vars_yr['Tair'].data[:, lat_dict[station_dict[station]]-1:lat_dict[station_dict[station]]+1, lon_dict[station_dict[station]]-1:lon_dict[station_dict[station]]+1],axis = (1,2)),
                np.mean(vars_yr['RH'].data[:, lat_dict[station_dict[station]]-1:lat_dict[station_dict[station]]+1, lon_dict[station_dict[station]]-1:lon_dict[station_dict[station]]+1], axis = (1,2)),
                np.mean(vars_yr['FF_10m'].data[:, lat_dict[station_dict[station]]-1:lat_dict[station_dict[station]]+1, lon_dict[station_dict[station]]-1:lon_dict[station_dict[station]]+1], axis = (1,2)),
                np.mean(vars_yr['sfc_P'].data[:, lat_dict[station_dict[station]]-1:lat_dict[station_dict[station]]+1, lon_dict[station_dict[station]]-1:lon_dict[station_dict[station]]+1], axis = (1,2)),
                np.mean(vars_yr['WD'].data[:, lat_dict[station_dict[station]]-1:lat_dict[station_dict[station]]+1, lon_dict[station_dict[station]]-1:lon_dict[station_dict[station]]+1], axis = (1,2)),
                np.mean(vars_yr['SWdown'].data[:, lat_dict[station_dict[station]]-1:lat_dict[station_dict[station]]+1, lon_dict[station_dict[station]]-1:lon_dict[station_dict[station]]+1], axis = (1,2)),
                np.mean(vars_yr['LWdown'].data[:, lat_dict[station_dict[station]]-1:lat_dict[station_dict[station]]+1, lon_dict[station_dict[station]]-1:lon_dict[station_dict[station]]+1], axis = (1,2)),
                np.mean(vars_yr['SWnet'].data[:, lat_dict[station_dict[station]]-1:lat_dict[station_dict[station]]+1, lon_dict[station_dict[station]]-1:lon_dict[station_dict[station]]+1], axis = (1,2)),
                np.mean(vars_yr['LWnet'].data[:, lat_dict[station_dict[station]]-1:lat_dict[station_dict[station]]+1, lon_dict[station_dict[station]]-1:lon_dict[station_dict[station]]+1], axis = (1,2)),
                np.mean(vars_yr['HS'].data[:, lat_dict[station_dict[station]]-1:lat_dict[station_dict[station]]+1, lon_dict[station_dict[station]]-1:lon_dict[station_dict[station]]+1], axis = (1,2)),
                np.mean(vars_yr['HL'].data[:, lat_dict[station_dict[station]]-1:lat_dict[station_dict[station]]+1, lon_dict[station_dict[station]]-1:lon_dict[station_dict[station]]+1], axis = (1,2)),
                np.mean(vars_yr['Emelt'].data[:, lat_dict[station_dict[station]]-1:lat_dict[station_dict[station]]+1, lon_dict[station_dict[station]]-1:lon_dict[station_dict[station]]+1], axis = (1,2)),
                np.mean(vars_yr['Emelt_calc'].data[:, lat_dict[station_dict[station]]-1:lat_dict[station_dict[station]]+1, lon_dict[station_dict[station]]-1:lon_dict[station_dict[station]]+1], axis = (1,2))]
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
        idx = ['Ts', 'Tair', 'RH', 'FF', 'P', 'WD', 'Swdown', 'LWdown', 'SWnet', 'LWnet', 'HS', 'HL', 'Emelt', 'Emelt']
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
    print('RMSE/bias = \n\n\n')
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
    print(' observed mean: \n%s\n' % AWS_nonz_mn)
    print(' model mean: \n%s\n' % mod_nonz_mn)
    print(' calculated model mean: \n%s\n' % calc_mod_nonz_mn)
    df.to_csv('/data/mac/ellgil82/hindcast/'+ year + '_' + station_dict[station] + '_bias_RMSE.csv') # change to be station-specific
    print(df)

calc_bias('2011', station = 'AWS14_SEB_2009-2017_norp')
calc_bias('2011', station = 'AWS14_SEB_2009-2017_norp')
#calc_bias('2014', station = 'AWS17_SEB_2011-2015_norp')
#calc_bias('2012', station = 'AWS17_SEB_2011-2015_norp')

def seas_bias(year, station):
    # then for each variable
    # Calculate bias of time series
    # Forecast error
    # Load AWS data, inc seasons
    AWS_var, AWS_DJF, AWS_MAM, AWS_JJA, AWS_SON = load_AWS(station, year)
    AWS_var = AWS_var[::3]
    AWS_DJF = AWS_DJF[::3]
    AWS_MAM = AWS_MAM[::3]
    AWS_JJA = AWS_JJA[::3]
    AWS_SON = AWS_SON[::3]
    for each_seas, each_mod_seas in zip([AWS_DJF, AWS_MAM, AWS_JJA, AWS_SON], ['DJF', 'MAM', 'JJA', 'SON']):
        surf_met_obs = [each_seas['Tsobs'], each_seas['Tair_2m'],each_seas['RH'], each_seas['FF_10m'], each_seas['pres'],
                        each_seas['WD'], each_seas['SWin_corr'], each_seas['LWin'], each_seas['SWnet_corr'],each_seas['LWnet_corr'],
                        each_seas['Hsen'], each_seas['Hlat'], each_seas['melt_energy'], each_seas['melt_energy']]
        # Load in model cubes
        os.chdir(filepath + year + '/')
        cube_list = iris.load(each_mod_seas + '_' + year + '*.nc')
        seas_cubes = []
        for i in range(len(cube_list)):
            cube_list[i] = cube_list[i][:, 0, lat_dict[station_dict[station]] - 1:lat_dict[station_dict[station]] + 1,
                           lon_dict[station_dict[station]] - 1:lon_dict[station_dict[station]] + 1]
            if cube_list[i].long_name == 'Pressure':
                cube_list[i].convert_units('hPa')
            elif cube_list[i].long_name == 'Temperature T':
                cube_list[i].convert_units('celsius')
            elif cube_list[i].long_name == 'Sensible heat flux' or cube_list[i].long_name == 'Latent heat flux':
                cube_list[i].data = 0 - cube_list[i].data
            seas_cubes.append(np.mean(cube_list[i].data, axis=(1, 2)))
        WD =  metpy.calc.wind_direction(u =seas_cubes[5], v = seas_cubes[6])
        Etot = seas_cubes[4]+seas_cubes[13]+seas_cubes[15]+seas_cubes[0]
        Emelt_calc = np.copy(Etot)
        Emelt_calc[seas_cubes[14] < -0.025] = 0
        seas_cubes.append(WD.magnitude)
        seas_cubes.append(Emelt_calc)
        surf_mod = [seas_cubes[14], seas_cubes[8], seas_cubes[10], seas_cubes[16], seas_cubes[11], seas_cubes[17], seas_cubes[12], seas_cubes[3], seas_cubes[4], seas_cubes[13], seas_cubes[15], seas_cubes[0], seas_cubes[2], seas_cubes[18]]
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
            mse = mean_squared_error(y_true=surf_met_obs[i], y_pred=surf_mod[i])
            rmse = np.sqrt(mse)
            rmses.append(rmse)
            idx = ['Ts', 'Tair', 'RH', 'FF', 'P', 'WD', 'Swdown', 'LWdown', 'SWnet', 'LWnet', 'HS', 'HL', 'Emelt', 'Emelt_calc']
        df = pd.DataFrame(index=idx)
        df['obs mean'] = pd.Series(mean_obs, index=idx)
        df['mod mean'] = pd.Series(mean_mod, index=idx)
        df['bias'] = pd.Series(bias, index=idx)
        df['rmse'] = pd.Series(rmses, index=idx)
        df['% RMSE'] = (df['rmse'] / df['obs mean']) * 100
        df['correl'] = pd.Series(r2s, index=idx)
        #for i in range(len(surf_mod)):
            #slope, intercept, r2, p, sterr = scipy.stats.linregress(surf_met_obs[i], surf_mod[i])
            #print(idx[i])
            #print('\nr2 = %s\n' % r2)
        print('RMSE/bias = \n\n\n')
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
        print(each_mod_seas + ' observed mean: \n%s\n' % AWS_nonz_mn)
        print(each_mod_seas + ' model mean: \n%s\n' % mod_nonz_mn)
        print(each_mod_seas + ' calculated model mean: \n%s\n' % calc_mod_nonz_mn)
        #df.to_csv('/data/mac/ellgil82/hindcast/' + year + '_' + each_mod_seas + '_' + station_dict[station] + '_bias_RMSE.csv')  # change to be station-specific
        print(df)

seas_bias(year = '2011', station = 'AWS14_SEB_2009-2017_norp')
#seas_bias(year = '2012', station = 'AWS17_SEB_2011-2015_norp')

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
    print obs_total_melt_cmv, mod_total_melt_cmv
    return obs_total_melt_cmv, mod_total_melt_cmv

obs_total_melt_cmv, mod_total_melt_cmv = calc_melt(AWS_vars = AWS14_SEB, model_vars = vars_2011)

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

melt_days_calc = calc_melt_days(vars_2012['Emelt_calc'].data)
melt_days = calc_melt_days(vars_2012['Emelt'].data)

np.mean(melt_days_calc)
np.mean(melt_days)

def calc_melt_duration(melt_var):
    melt = np.copy(melt_var)
    melt_periods = np.count_nonzero(melt, axis = 0)
    melt_periods = melt_periods*3. # multiply by 3 to get number of hours per year (3-hourly data)
    return melt_periods

melt_dur_calc = calc_melt_duration(vars_2011['Emelt_calc'].data)
melt_dur = calc_melt_duration(vars_2011['Emelt'].data)

np.mean(melt_dur_calc)
np.mean(melt_dur)

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

wind_rose(year = '2011', station = 'AWS14')

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
seas_wind_rose('2011', 'AWS14')

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
            ax2.plot(SEB_1p5['Time_srs'], SEB_1p5[k], linewidth=2.5, color=col_dict[r], label='*%(r)s UM output for Cabinet Inlet' % locals(), zorder = 5)
            ax2.plot(SEB_4p0['Time_srs'], SEB_4p0[k], linewidth=2.5, color='#1f78b4', label='*%(r)s UM output for Cabinet Inlet' % locals(), zorder=4)
            ax2.axis('off')
            ax2.set_xlim(SEB_1p5['Time_srs'][1], SEB_1p5['Time_srs'][-1])
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
    #ax[3].fill_between(SEB_1p5['Time_srs'], SEB_1p5['percentiles'][2], SEB_1p5['percentiles'][3], facecolor=col_dict[r], alpha=0.4, zorder=2)
    #ax[2].fill_between(SEB_1p5['Time_srs'], SEB_1p5['percentiles'][0], SEB_1p5['percentiles'][1], facecolor=col_dict[r], alpha=0.4, zorder=2)
    #ax[1].fill_between(SEB_1p5['Time_srs'], SEB_1p5['percentiles'][8], SEB_1p5['percentiles'][9], facecolor=col_dict[r], alpha=0.4, zorder=2)
    #ax[0].fill_between(SEB_1p5['Time_srs'], SEB_1p5['percentiles'][4], SEB_1p5['percentiles'][5], facecolor=col_dict[r], alpha=0.4, zorder=2)
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

def surf_plot(vars_yr, AWS_var, station):
    fig, ax = plt.subplots(2,2,sharex= True, figsize=(22, 12))
    ax = ax.flatten()
    col_dict = {'0.5 km': '#33a02c', '1.5 km': '#f68080', '4.0 km': '#1f78b4'}
    lab_dict = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g', 7: 'h', 8: 'i', 9: 'j', 10: 'k', 11: 'l' }
    days = mdates.MonthLocator(interval=1)
    dayfmt = mdates.DateFormatter('%m')
    plot = 0
    AWS_daymn = AWS_var.groupby(['day']).mean()
    AWS_daymn.to_csv('/data/clivarm/wip/ellgil82/AWS/'+station+'_'+ vars_yr['year']+'_daymn.csv')
    obs_var = pd.read_csv('/data/clivarm/wip/ellgil82/AWS/'+station+'_'+ vars_yr['year']+'_daymn.csv', header = 0)
    # Plot each variable in turn for 1.5 km resolution
    for r in ['4.0 km']: # can increase the number of res, '4.0 km'
        for j, k in zip(['RH', 'FF_10m', 'Tair_2m', 'Tsobs'], ['RH', 'FF_10m', 'Tair', 'Ts']):
            limits = {'RH': (40,100), 'FF_10m': (0,25), 'Tair_2m': (-40, 5), 'Tsobs': (-40, 5)}
            titles = {'RH': 'Relative \nhumidity (%)',
                      'FF_10m': 'Wind speed \n(m s$^{-1}$)',
                      'Tair_2m': '2 m air \ntemperature ($^{\circ}$C)',
                      'Tsobs': 'Surface \ntemperature ($^{\circ}$C)'}
            obs = ax[plot].plot(obs_var.index, obs_var[j], color='k', linewidth=2.5, label="Observations at AWS 14")
            ax2 = ax[plot].twiny()
            ax2.plot(vars_yr[k].coord('time').points, np.mean(vars_yr[k].data[:, lat_index14-1:lat_index14+1,lon_index14-1:lon_index14+1], axis = (1,2)), linewidth=2.5, color=col_dict[r], label='*%(r)s UM output for AWS 14' % locals(), zorder = 5)
            ax2.axis('off')
            ax2.set_xlim(vars_yr[k].coord('time').points[0], vars_yr[k].coord('time').points[-1])
            ax[plot].set_xlim(obs_var.index[0], obs_var.index[-1])
            ax2.tick_params(axis='both', tick1On = False, tick2On = False)
            ax2.set_ylim(limits[j])
            ax[plot].set_ylim(limits[j])#[floor(np.floor(np.min(AWS_var[j])),5),ceil(np.ceil( np.max(AWS_var[j])),5)])
            ax[plot].set_ylabel(titles[j], rotation=0, fontsize=24, color = 'dimgrey', labelpad = 80)
            ax[plot].tick_params(axis='both', which='both', labelsize=24, tick1On = False, tick2On = False)
            lab = ax[plot].text(0.08, 0.85, zorder = 100, transform = ax[plot].transAxes, s=lab_dict[plot], fontsize=32, fontweight='bold', color='dimgrey')
            plot = plot + 1
        for axs in [ax[0], ax[2]]:
            axs.yaxis.set_label_coords(-0.3, 0.5)
            axs.spines['right'].set_visible(False)
        for axs in [ax[1], ax[3]]:
            axs.yaxis.set_label_coords(1.27, 0.5)
            axs.yaxis.set_ticks_position('right')
            axs.tick_params(axis='y', tick1On = False)
            axs.spines['left'].set_visible(False)
        for axs in [ax[2], ax[3]]:
            axs.set_xlabel('Day of year')
        for axs in [ax[0], ax[1]]:
            [l.set_visible(False) for (w, l) in enumerate(axs.yaxis.get_ticklabels()) if w % 2 != 0]
        for axs in ax:
            axs.spines['top'].set_visible(False)
            plt.setp(axs.spines.values(), linewidth=2, color='dimgrey', )
            axs.set_xlim(obs_var.index[0], obs_var.index.values[-1])
            axs.tick_params(axis='both', which='both', labelsize=24, tick1On=False, tick2On=False, labelcolor='dimgrey', pad=10)
            #[l.set_visible(False) for (w, l) in enumerate(axs.xaxis.get_ticklabels()) if w % 2 != 0]
            plt.xticks([vars_yr[k].coord('time').points[0], vars_yr[k].coord('time').points[-1]])
    # Legend
    lns = [Line2D([0],[0], color='k', linewidth = 2.5)]
    labs = ['Observations from AWS 14']
    for r in ['4.0 km']:#
        lns.append(Line2D([0],[0], color=col_dict[r], linewidth = 2.5))
        labs.append(r[0]+'.'+r[2]+' km UM output at AWS 14')#('1.5 km output for Cabinet Inlet')
    lgd = ax[1].legend(lns, labs, bbox_to_anchor=(0.55, 1.1), loc=2, fontsize=20)
    frame = lgd.get_frame()
    frame.set_facecolor('white')
    for ln in lgd.get_texts():
        plt.setp(ln, color='dimgrey')
    lgd.get_frame().set_linewidth(0.0)
    plt.subplots_adjust(wspace = 0.05, hspace = 0.05, top = 0.95, right = 0.85, left = 0.16, bottom = 0.08)
    plt.savefig('/users/ellgil82/figures/Hindcast/Validation/surface_met_'+station+'_'+ vars_yr['year'] + '_no_range_daymn.png', transparent = True)
    plt.savefig('/users/ellgil82/figures/Hindcast/Validation/surface_met_'+station+'_'+ vars_yr['year'] + '_no_range_daymn.eps', transparent = True)
    plt.savefig('/users/ellgil82/figures/Hindcast/Validation/surface_met_'+station+'_'+ vars_yr['year'] + '_no_range_daymn.pdf', transparent = True)
    plt.show()

surf_plot(vars_2011, AWS14_SEB, station = 'AWS14')
#surf_plot(vars_2012, AWS17_SEB, station = 'AWS17')


def melt_plot(AWS_var, vars_yr, station):
    fig, ax = plt.subplots(figsize = (18,8))
    ax2 = ax.twiny()
    ax.plot(AWS_var['datetime'][::3], vars_yr['Emelt'][:, lat_dict[station], lon_dict[station]].data, lw = 2, color = '#1f78b4', label = 'Modelled $E_{melt}$', zorder = 1)#color = '#f68080',
    ax2.plot(AWS_var['datetime'], AWS_var['melt_energy'], lw=2, color='k', label = 'Observed $E_{melt}$', zorder = 2)
    ax2.set_xlim(AWS_var['datetime'][0], AWS_var['datetime'][-1])
    ax.set_xlim(AWS_var['datetime'][0], AWS_var['datetime'][-1])
    days = mdates.DayLocator(interval=1)
    dayfmt = mdates.DateFormatter('%d %b')
    ax.set_ylim(0, 150)
    ax.set_ylabel('$E_{melt}$ \n(W m$^{-2}$)', rotation = 0, fontsize = 36, labelpad = 100, color = 'dimgrey')
    ax2.axis('off')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.setp(ax.spines.values(), linewidth=2, color='dimgrey', )
    ax.tick_params(axis='both', which='both', labelsize=36, tick1On=False, tick2On=False, labelcolor='dimgrey', pad=10)
    [l.set_visible(False) for (w, l) in enumerate(ax.yaxis.get_ticklabels()) if w % 2 != 0]
    [l.set_visible(False) for (w, l) in enumerate(ax.xaxis.get_ticklabels()) if w % 2 != 0]
    ax.xaxis.set_major_formatter(dayfmt)
    #Legend
    lns = [Line2D([0],[0], color='k', linewidth = 2.5),
           Line2D([0],[0], color =  '#1f78b4', linewidth = 2.5)]
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
    if host == 'bsl':
        plt.savefig('/users/ellgil82/figures/Hindcast/2011_melt.png', transparent = True)
        plt.savefig('/users/ellgil82/figures/Hindcast/2011_melt.eps', transparent=True)
    plt.show()

melt_plot(AWS_var = AWS14_SEB, vars_yr = vars_2011, station = 'AWS14')

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
melt_map(vars_yr = vars_2011, AWS14_var = AWS14_SEB, AWS17_var = AWS17_SEB, calc = True, which = 'duration')
melt_map(vars_yr = vars_2011, AWS14_var = AWS14_SEB, AWS17_var = AWS17_SEB, calc = False, which = 'duration')
#melt_map(vars_yr = vars_2014, AWS14_var = AWS14_SEB, AWS17_var = AWS17_SEB, calc = False, which = 'duration')