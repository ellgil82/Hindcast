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
    filepath = '/data/mac/ellgil82/hindcast/'

## Load data
def load_vars(year, mn):
    if mn == 'yes':
        Tair = iris.load_cube( filepath + 'output/'+year+'_Tair_1p5m_daymn.nc', 'air_temperature')
        Ts = iris.load_cube( filepath + 'output/'+year+'_Ts_daymn.nc', 'surface_temperature')
        MSLP = iris.load_cube( filepath + 'output/'+year+'_MSLP_daymn.nc', 'air_pressure_at_sea_level')
        sfc_P = iris.load_cube(filepath + 'output/' + year + '_sfc_P_daymn.nc', 'surface_air_pressure')
        FF_10m = iris.load_cube( filepath + 'output/'+year+'_FF_10m_daymn.nc', 'wind_speed')
        RH = iris.load_cube(filepath + 'output/' + year + '_RH_1p5m_daymn.nc', 'relative_humidity')
        u = iris.load_cube(filepath + 'output/' + year + '_u_10m_daymn.nc', 'x wind component (with respect to grid)')
        v = iris.load_cube(filepath + 'output/' + year + '_v_10m_daymn.nc', 'y wind component (with respect to grid)')
        LWnet = iris.load_cube( filepath + 'output/'+year+'_surface_LW_net_daymn.nc', 'surface_net_downward_longwave_flux')
        SWnet = iris.load_cube(filepath + 'output/' + year + '_surface_SW_net_daymn.nc','Net short wave radiation flux')
        LWdown = iris.load_cube( filepath + 'output/'+year+'_surface_LW_down_daymn.nc', 'IR down')
        SWdown = iris.load_cube( filepath + 'output/'+year+'_surface_SW_down_daymn.nc', 'surface_downwelling_shortwave_flux_in_air')
        HL = iris.load_cube( filepath + 'output/'+year+'_latent_heat_daymn.nc', 'Latent heat flux')
        HS = iris.load_cube( filepath + 'output/'+year+'_sensible_heat_daymn.nc', 'surface_upward_sensible_heat_flux')
    else:
        Tair = iris.load_cube( filepath + 'output/'+year+'_Tair_1p5m.nc', 'air_temperature')
        Ts = iris.load_cube( filepath + 'output/'+year+'_Ts.nc', 'surface_temperature')
        MSLP = iris.load_cube( filepath + 'output/'+year+'_MSLP.nc', 'air_pressure_at_sea_level')
        sfc_P = iris.load_cube(filepath + 'output/' + year + '_sfc_P.nc', 'surface_air_pressure')
        FF_10m = iris.load_cube( filepath + 'output/'+year+'_FF_10m.nc', 'wind_speed')
        RH = iris.load_cube(filepath + 'output/' + year + '_RH_1p5m.nc', 'relative_humidity')
        u = iris.load_cube(filepath + 'output/' + year + '_u_10m.nc', 'x wind component (with respect to grid)')
        v = iris.load_cube(filepath + 'output/' + year + '_v_10m.nc', 'y wind component (with respect to grid)')
        LWnet = iris.load_cube( filepath + 'output/'+year+'_surface_LW_net.nc', 'surface_net_downward_longwave_flux')
        SWnet = iris.load_cube(filepath + 'output/' + year + '_surface_SW_net.nc','Net short wave radiation flux')
        LWdown = iris.load_cube( filepath + 'output/'+year+'_surface_LW_down.nc', 'IR down')
        SWdown = iris.load_cube( filepath + 'output/'+year+'_surface_SW_down.nc', 'surface_downwelling_shortwave_flux_in_air')
        HL = iris.load_cube( filepath + 'output/'+year+'_latent_heat.nc', 'Latent heat flux')
        HS = iris.load_cube( filepath + 'output/'+year+'_sensible_heat.nc', 'surface_upward_sensible_heat_flux')
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
    Emelt = np.copy(Etot)
    Emelt[Ts.data < -0.025] = 0
    for turb in [HS, HL]:
        turb = 0-turb.data
        turb = iris.cube.Cube(turb)
    Emelt = iris.cube.Cube(Emelt)
    Etot = iris.cube.Cube(Etot)
    vars_yr = {'Tair': Tair[:,0,:,:], 'Ts': Ts[:,0,:,:], 'MSLP': MSLP[:,0,:,:], 'sfc_P': sfc_P[:,0,:,:], 'FF_10m': FF_10m[:,0,:,:],
               'RH': RH[:,0,:,:], 'WD': WD[:,0,:,:], 'LWnet': LWnet[:,0,:,:], 'SWnet': SWnet[:,0,:,:], 'SWdown': SWdown[:,0,:,:],
               'LWdown': LWdown[:,0,:,:], 'HL': HL[:,0,:,:], 'HS': HS[:,0,:,:], 'Etot': Etot[:,0,:,:], 'Emelt': Emelt[:,0,:,:],
               'lon': real_lon, 'lat': real_lat, 'year': year}
    return vars_yr

vars_2012 = load_vars('2012', mn = 'no')
#vars_2014 = load_vars('2014', mn = 'no')
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
    if host == 'jasmin':
        os.chdir('/group_workspaces/jasmin4/bas_climate/users/ellgil82/OFCAP/proc_data/')
    elif host == 'bsl':
        os.chdir('/data/mac/ellgil82/cloud_data/um/vn11_test_runs/Jan_2011/proc_data/')
    return case

AWS14_SEB = load_AWS('AWS14_SEB_2009-2017_norp', '2014')
#AWS15_SEB = load_AWS('AWS15_hourly_2009-2014.csv', '2014')
AWS17_SEB = load_AWS('AWS17_SEB_2011-2015_norp', '2014')

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
             }

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
            ANN = iris.load_cube(filepath + 'output/' + each_year + file_dict[each_var], each_var)
            DJF = iris.load_cube(filepath + 'output/' + each_year + '/DJF_' + each_year +  file_dict[each_var], each_var)
            MAM = iris.load_cube(filepath + 'output/' + each_year + '/MAM_' + each_year + file_dict[each_var], each_var)
            JJA = iris.load_cube(filepath + 'output/' + each_year + '/JJA_' + each_year + file_dict[each_var], each_var)
            SON = iris.load_cube(filepath + 'output/' + each_year + '/SON_' + each_year +  file_dict[each_var], each_var)
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
        seas_means.to_csv('/data/mac/ellgil82/hindcast/output/'+ each_year +'seas_means.csv')
        #save indivdual dataframes for each year
        # calculate mean for whole period
        #DJF_means[each_var] = DJF_means[each_var] / len(year_list)
    return seas_means, seas_vals

#seas_means, seas_vals = seas_mean(['2014'], 'AWS17')

def calc_bias(year, station):
    # Calculate bias of time series
    # Forecast error
    AWS_var = load_AWS(station, year)
    AWS_var = AWS_var[::3]
    vars_yr = load_vars(year, mn = 'no')
    surf_met_obs = [AWS_var['Tsobs'], AWS_var['Tair_2m'], AWS_var['RH'], AWS_var['FF_10m'], AWS_var['pres'], AWS_var['WD'], AWS_var['SWin_corr'], AWS_var['LWin'], AWS_var['SWnet_corr'], AWS_var['LWnet_corr'],
                    AWS_var['Hsen'], AWS_var['Hlat'], AWS_var['melt_energy']]
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
                np.mean(vars_yr['Emelt'].data[:, lat_dict[station_dict[station]]-1:lat_dict[station_dict[station]]+1, lon_dict[station_dict[station]]-1:lon_dict[station_dict[station]]+1], axis = (1,2))]
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
        idx = ['Ts', 'Tair', 'RH', 'FF', 'P', 'WD', 'Swdown', 'LWdown', 'SWnet', 'LWnet', 'HS', 'HL', 'Emelt']
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
    df.to_csv('/data/mac/ellgil82/hindcast/'+ year + '_' + station_dict[station] + '_bias_RMSE.csv') # change to be station-specific
    print(df)

#calc_bias('2014', station = 'AWS14_SEB_2009-2017_norp')
calc_bias('2012', station = 'AWS14_SEB_2009-2017_norp')
#calc_bias('2014', station = 'AWS17_SEB_2011-2015_norp')
calc_bias('2012', station = 'AWS17_SEB_2011-2015_norp')

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
    plt.savefig('/users/ellgil82/figures/Hindcast/wind_rose_' + station + '_' + year + '.png')
    plt.savefig('/users/ellgil82/figures/Hindcast/wind_rose_' + station + '_' + year + '.eps')
    plt.show()

wind_rose(year = '2014', station = 'AWS17')

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

seas_wind_rose('2014', 'AWS17')

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

def surf_plot(vars_yr, AWS_var):
    fig, ax = plt.subplots(2,2,sharex= True, figsize=(22, 12))
    ax = ax.flatten()
    col_dict = {'0.5 km': '#33a02c', '1.5 km': '#f68080', '4.0 km': '#1f78b4'}
    lab_dict = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g', 7: 'h', 8: 'i', 9: 'j', 10: 'k', 11: 'l' }
    days = mdates.DayLocator(interval=1)
    dayfmt = mdates.DateFormatter('%d %b')
    plot = 0
    AWS_daymn = AWS_var.groupby('day').mean()
    # Plot each variable in turn for 1.5 km resolution
    for r in ['4.0 km']: # can increase the number of res, '4.0 km'
        for j, k in zip(['RH', 'FF_10m', 'Tair_2m', 'Tsobs'], ['RH', 'FF_10m', 'Tair', 'Ts']):
            limits = {'RH': (40,100), 'FF_10m': (0,25), 'Tair_2m': (-40, 5), 'Tsobs': (-40, 5)}
            titles = {'RH': 'Relative \nhumidity (%)',
                      'FF_10m': 'Wind speed \n(m s$^{-1}$)',
                      'Tair_2m': '2 m air \ntemperature ($^{\circ}$C)',
                      'Tsobs': 'Surface \ntemperature ($^{\circ}$C)'}
            obs = ax[plot].plot(vars_yr[k].coord('time').points[:365], AWS_daymn[j], color='k', linewidth=2.5, label="Observations at AWS 14")
            ax2 = ax[plot].twiny()
            ax2.plot(vars_yr[k].coord('time').points, np.mean(vars_yr[k].data[:, lat_index14-1:lat_index14+1,lon_index14-1:lon_index14+1], axis = (1,2)), linewidth=2.5, color=col_dict[r], label='*%(r)s UM output for AWS 14' % locals(), zorder = 5)
            ax2.axis('off')
            ax2.set_xlim(vars_yr[k].coord('time').points[0], vars_yr[k].coord('time').points[-1])
            ax[plot].set_xlim(vars_yr[k].coord('time').points[0], vars_yr[k].coord('time').points[-1])
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
            #plt.setp(axs.get_yticklabels()[-2], visible=False)
            axs.xaxis.set_major_formatter(dayfmt)
            #plt.setp(axs.get_xticklabels()[])
        for axs in [ax[0], ax[1]]:
            [l.set_visible(False) for (w, l) in enumerate(axs.yaxis.get_ticklabels()) if w % 2 != 0]
        for axs in ax:
            axs.spines['top'].set_visible(False)
            plt.setp(axs.spines.values(), linewidth=2, color='dimgrey', )
            axs.set_xlim(AWS_var['datetime'].values[0], AWS_var['datetime'].values[-1])
            axs.tick_params(axis='both', which='both', labelsize=24, tick1On=False, tick2On=False, labelcolor='dimgrey', pad=10)
            [l.set_visible(False) for (w, l) in enumerate(axs.xaxis.get_ticklabels()) if w % 2 != 0]
        #plt.setp(ax[3].get_yticklabels()[-1], visible=False)
        #plt.setp(ax[3].get_yticklabels()[-5], visible=False)
        #ax[3].fill_between(surf_1p5['Time_srs'], surf_1p5['percentiles'][0], surf_1p5['percentiles'][1], facecolor=col_dict[r], alpha=0.4)
        #ax[2].fill_between(surf_1p5['Time_srs'], surf_1p5['percentiles'][2], surf_1p5['percentiles'][3], facecolor=col_dict[r], alpha=0.4)
        #ax[0].fill_between(surf_1p5['Time_srs'], surf_1p5['percentiles'][4], surf_1p5['percentiles'][5], facecolor=col_dict[r], alpha=0.4)
        #ax[1].fill_between(surf_1p5['Time_srs'], surf_1p5['percentiles'][6], surf_1p5['percentiles'][7], facecolor=col_dict[r], alpha=0.4)
    # Legend
    lns = [Line2D([0],[0], color='k', linewidth = 2.5)]
    labs = ['Observations from AWS 14']
    for r in ['4.0 km']:#
        lns.append(Line2D([0],[0], color=col_dict[r], linewidth = 2.5))
        labs.append(r[0]+'.'+r[2]+' km UM output for Cabinet Inlet')#('1.5 km output for Cabinet Inlet')
    lgd = ax[1].legend(lns, labs, bbox_to_anchor=(0.55, 1.1), loc=2, fontsize=20)
    frame = lgd.get_frame()
    frame.set_facecolor('white')
    for ln in lgd.get_texts():
        plt.setp(ln, color='dimgrey')
    lgd.get_frame().set_linewidth(0.0)
    plt.subplots_adjust(wspace = 0.05, hspace = 0.05, top = 0.95, right = 0.85, left = 0.16, bottom = 0.08)
    plt.savefig('/users/ellgil82/figures/Hindcast/Validation/surface_met_'+station_dict[station]+'_no_range_daymn.png', transparent = True)
    plt.savefig('/users/ellgil82/figures/Hindcast/Validation/surface_met_'+station_dict[station]+'_no_range_daymn.eps', transparent = True)
    plt.savefig('/users/ellgil82/figures/Hindcast/Validation/surface_met_'+station_dict[station]+'_no_range_daymn.pdf', transparent = True)
    plt.show()

surf_plot(vars_2014, AWS14_SEB)
#surf_plot(vars_2012, AWS14_SEB)