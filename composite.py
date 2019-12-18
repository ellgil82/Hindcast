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

from rotate_data import rotate_data
from sklearn.preprocessing import normalize
from divg_temp_colourmap import shiftedColorMap
import datetime
from datetime import datetime, time
from scipy import stats

# Set up filepath
if host == 'jasmin':
    filepath = '/gws/nopw/j04/bas_climate/users/ellgil82/hindcast/output/alloutput/'
elif host == 'bsl':
    filepath = '/data/mac/ellgil82/hindcast/output/'

unit_dict = {'MSLP': 'hPa',
             'T': 'celsius',
             'Tair_1p5m': 'celsius',
             'Tair_1p5m_daymax': 'celsius',
             'Ts': 'celsius',
             'u_10m': 'm s-1',
             'v_10m': 'm s-1',
             'u': 'm s-1',
             'v': 'm s-1',
             'SIC' : 'unknown',
             'cl_frac': 'unknown',
             'latent_heat': 'W m-2',
'surface_LW_net' : 'W m-2',
'surface_SW_net' : 'W m-2',
'surface_LW_down' : 'W m-2',
'surface_SW_down' : 'W m-2',
'E_tot': 'W m-2',
'land_snow_melt_flux': 'W m-2',
'sensible_heat': 'W m-2',
             'land_snow_melt_amnt': 'kg m-2',
             'total_column_liquid': 'g kg-1',
'total_column_ice': 'g kg-1',
'total_column_vapour': 'g kg-1',
             'u_wind_P_levs': 'm s-1',
             'u_wind_full_profile': 'm s-1'
             }

print('CAHHHMMNN AHHHNNN!\n\n')

## Load data
def load_vars(year, var_name, seas): # seas should be in the format '???_'
    #full_tsrs = pd.date_range(datetime(int(year),1,1,0,0,0),datetime(int(year),12,31,23,59,59), freq = 'D')
    # Load daily means
    if var_name == 'Tair_1p5m_daymax':
        var = iris.load_cube(filepath + year + '_' + seas  + var_name + '.nc')
    else:
        var = iris.load_cube(filepath + year + '_' + seas + var_name + '_daymn.nc')
    if var_name == 'sensible_heat' or var_name == 'latent_heat':
        var = iris.analysis.maths.multiply(var, -1.)
    elif var_name == 'land_snow_melt_amnt':
        var = iris.analysis.maths.multiply(var, 108.)
    for i in [var]:
        i.convert_units(unit_dict[var_name])
    if var_name == 'v_10m' or var_name == 'FF_10m' or var_name == 'u_prof' or var_name == 'u_Plev':
        var = var[:,:,1:,:]
    return var #var[start_idx[0][0]:end_idx[0][0],0,:,:], clim[start_idx[0][0]:end_idx[0][0],0,:,:], anom

def create_clim(var_name, seas): # seas should be in the format '???_'
    import calendar
    # Load climatology
    clim = iris.load_cube(filepath + seas + var_name + '_climatology.nc')
    for i in [clim]:
        i.convert_units(unit_dict[var_name])
    if seas == 'DJF':
        clim = clim[:89]
        clim_total = clim.data[:89]
    else:
        clim = clim[:365]
        clim_total = clim.data[:365]
    for i in range(1998,2017):
        if calendar.isleap(i) == True: # if leap year, repeat final day of climatology
            print(str(i) + ' is leap')
            clim_total = np.concatenate((clim_total, clim.data), axis = 0)
            clim_total = np.concatenate((clim_total, clim.data[-1:]), axis = 0)
        else:
            clim_total = np.concatenate((clim_total, clim.data), axis = 0 )
    if var_name == 'v_10m' or var_name == 'FF_10m' or var_name == 'u_prof' or var_name == 'u_Plev':
        clim_total = clim_total[:,:,1:,:]
    if var_name == 'sensible_heat' or var_name == 'latent_heat':
        clim_total = clim_total * -1.
    return iris.cube.Cube(clim_total)

print('Loading synoptic meteorology... \n\n')
clims = {}
anoms = {}
var_dict = {}
for i in [ 'Tair_1p5m', 'Tair_1p5m_daymax', 'MSLP', 'u_10m', 'v_10m', 'SIC']:#
    var_dict[i] = load_vars('1998-2017', i)
    clims[i] = create_clim(i)
    anoms[i] = iris.cube.Cube(data = (var_dict[i].data[:7305] - clims[i].data))

try:
    LSM = iris.load_cube(filepath+'new_mask.nc')
    orog = iris.load_cube(filepath+'orog.nc')
    orog = orog[0,0,:,:]
    lsm = LSM[0,0,:,:]
    for i in [orog, lsm]:
        real_lon, real_lat = rotate_data(i, np.ndim(i)-2, np.ndim(i)-1)
except iris.exceptions.ConstraintMismatchError:
    print('Files not found')

def plot_synop_composite(cf_var, c_var, u_var, v_var, regime):
    #cf_var = np.mean(cf_var.data[:, :, :], axis=0)
    #c_var = np.mean(c_var.data[:, :, :], axis=0)
    #u_var = np.mean(u_var.data[:, :, :], axis=0)
    #v_var = np.mean(v_var.data[:, 1:, :], axis=0)
    fig = plt.figure(frameon=False, figsize=(10, 13))  # !!change figure dimensions when you have a larger model domain
    fig.patch.set_visible(False)
    ax = fig.add_subplot(111)#, projection=ccrs.PlateCarree())
    plt.axis = 'off'
    PlotLonMin = np.min(real_lon)
    PlotLonMax = np.max(real_lon)
    PlotLatMin = np.min(real_lat)
    PlotLatMax = np.max(real_lat)
    XTicks = np.linspace(PlotLonMin, PlotLonMax, 4)
    XTickLabels = [None] * len(XTicks)
    for i, XTick in enumerate(XTicks):
        if XTick < 0:
            XTickLabels[i] = '{:.0f}{:s}'.format(np.abs(XTick), '$^{\circ}$W')
        else:
            XTickLabels[i] = '{:.0f}{:s}'.format(np.abs(XTick), '$^{\circ}$E')#
    plt.xticks(XTicks, XTickLabels)
    ax.set_xlim(PlotLonMin, PlotLonMax)
    ax.tick_params(which='both', pad=10, labelsize = 34, color = 'dimgrey')
    YTicks = np.linspace(PlotLatMin, PlotLatMax, 4)
    YTickLabels = [None] * len(YTicks)
    for i, YTick in enumerate(YTicks):
        if YTick < 0:
            YTickLabels[i] = '{:.0f}{:s}'.format(np.abs(YTick), '$^{\circ}$S')
        else:
            YTickLabels[i] = '{:.0f}{:s}'.format(np.abs(YTick), '$^{\circ}$N')
    plt.yticks(YTicks, YTickLabels)
    ax.set_ylim(PlotLatMin, PlotLatMax)
    ax.tick_params(which='both', axis='both', labelsize=34, labelcolor='dimgrey', pad=10, size=0, tick1On=False,
                   tick2On=False)
    bwr_zero = shiftedColorMap(cmap=matplotlib.cm.bwr, min_val=-20., max_val=5., name='bwr_zero', var=cf_var.data, start=0.15,
                               stop=0.85) #-6, 3
    xlon, ylat = np.meshgrid(real_lon, real_lat)
    c = ax.pcolormesh(xlon, ylat, cf_var, cmap=bwr_zero,vmin=-20., vmax=5., zorder=1)  # latlon=True, transform=ccrs.PlateCarree(),
    cs = ax.contour(xlon, ylat, np.ma.masked_where(lsm.data == 1, c_var), latlon=True, colors='k', zorder=4)
    ax.clabel(cs, inline=True, fontsize = 24, inline_spacing = 30, fmt =  '%1.0f')
    coast = ax.contour(xlon, ylat, lsm.data, levels=[0], colors='#222222', lw=2, latlon=True, zorder=2)
    topog = ax.contour(xlon, ylat, orog.data, levels=[50], colors='#222222', linewidth=1.5, latlon=True, zorder=3)
    CBarXTicks = [-20, -10, 0,  5]  # CLevs[np.arange(0,len(CLevs),int(np.ceil(len(CLevs)/5.)))]
    CBAxes = fig.add_axes([0.22, 0.2, 0.6, 0.025])
    CBar = plt.colorbar(c, cax=CBAxes, orientation='horizontal', ticks=CBarXTicks, extend = 'both')  #
    CBar.set_label('Mean daily maximum 1.5 m \nair temperature ($^{\circ}$C)', fontsize=34, labelpad=10, color='dimgrey')
    CBar.solids.set_edgecolor("face")
    CBar.outline.set_edgecolor('dimgrey')
    CBar.ax.tick_params(which='both', axis='both', labelsize=34, labelcolor='dimgrey', pad=10, size=0, tick1On=False,
                        tick2On=False)
    CBar.outline.set_linewidth(2)
    x, y = np.meshgrid(real_lon, real_lat)
    q = ax.quiver(x[::20, ::20], y[::20, ::20], u_var[::20, ::20], v_var[::20, ::20], pivot='middle', scale=100,
                  zorder=5)
    plt.quiverkey(q, 0.25, 0.9, 10, r'$10$ $m$ $s^{-1}$', labelpos='N', color='dimgrey', labelcolor='dimgrey',
                  fontproperties={'size': '32', 'weight': 'bold'},
                  coordinates='figure', )
    plt.subplots_adjust(left = 0.2, bottom = 0.33, right = 0.85)
    plt.savefig('/gws/nopw/j04/bas_climate/users/ellgil82/hindcast/figures/composite_' + regime + '.png')
    plt.savefig('/gws/nopw/j04/bas_climate/users/ellgil82/hindcast/figures/composite_' + regime + '.eps')
    #plt.show()

#Plot mean conditions during MAM 2016
def find_case(var, seas):
    if var == 'land_snow_melt_amnt':
        cube = iris.load_cube(filepath + seas + '_' + var + '_daysum.nc')
        cube = iris.analysis.maths.multiply(cube, 108.) # 10800 s in 3 hrs / 100 s in a model timestep = 108 ==> melt amount per output timestep
    else:
        cube = iris.load_cube(filepath + seas +'_' + var + '_daymn.nc')
    cube.convert_units(unit_dict[var])
    clim = iris.load_cube(filepath + var + '_climatology.nc')
    clim.convert_units(unit_dict[var])
    if var == 'land_snow_melt_amnt':
        clim = iris.analysis.maths.multiply(clim, 108.)
    if var == 'v_10m':
        cube = cube[:,:,1:,:]
        clim = clim[:,:,1:,:]
    if var == 'sensible_heat' or var == 'latent_heat':
        # FLip turbulent fluxes to match convention (positive towards surface)
        cube = iris.analysis.maths.multiply(cube, -1.)
    clim = clim[60:152]
    anom = cube.data - clim.data
    if var == 'land_snow_melt_amnt':
        return np.cumsum(cube[:,0,:,:].data, axis = 0), np.cumsum(clim[:,0,:,:].data, axis = 0), np.cumsum(anom[:,0,:,:], axis=0)
    else:
        return np.mean(cube[:,0,:,:].data, axis = 0), clim[:,0,:,:], np.mean(anom[:,0,:,:], axis=0)

#MAM_16_Tmax = iris.load_cube(filepath + 'MAM_2016_Tair_1p5m_daymax.nc')
#MAM_16_Tmax.convert_units('celsius')
#MAM_16_Tmax_anom = MAM_16_Tmax.data - Tmax_clim[60:152].data
#MAM_16_u, MAM_16_clim_u, MAM_16_uanom = find_case('u_10m', 'MAM_2016')
#MAM_16_v, MAM_16_clim_v, MAM_16_vanom = find_case('v_10m', 'MAM_2016')
#MAM_16_MSLP, MAM_16_clim_MSLP, MAM_16_MSLPanom = find_case('MSLP', 'MAM_2016')

#Plot met conditions
#plot_synop_composite(np.mean(MAM_16_Tmax_anom[:,0,:,:], axis = 0), MAM_16_MSLP, MAM_16_u, MAM_16_v, regime = 'MAM_2016_mean_cond_Tmax')


#MAM_16_HL, MAM_16_clim_HL, MAM_16_HLanom = find_case('latent_heat', 'MAM_2016')
#MAM_16_HS, MAM_16_clim_HS, MAM_16_HSanom = find_case('sensible_heat', 'MAM_2016')
#
##MAM_16_melt, MAM_16_clim_melt, MAM_16_meltanom = find_case('land_snow_melt_flux', 'MAM_2016')
#MAM_16_melt, MAM_16_clim_melt, MAM_16_meltanom = find_case('land_snow_melt_amnt', 'MAM_2016')
#MAM_16_melt, MAM_16_clim_melt, MAM_16_meltanom = MAM_16_melt[-1], MAM_16_clim_melt[-1], MAM_16_meltanom[-1]
#MAM_16_SWnet, MAM_16_clim_SWnet, MAM_16_SWnetanom = find_case('surface_SW_net', 'MAM_2016')
#MAM_16_LWnet, MAM_16_clim_LWnet, MAM_16_LWnetanom = find_case('surface_LW_net', 'MAM_2016')
#
#MAM_16_E_tot = MAM_16_SWnet + MAM_16_LWnet + MAM_16_HL + MAM_16_HS
#MAM_16_clim_E_tot = iris.load_cube(filepath + 'E_tot_climatology.nc')
#MAM_16_E_totanom = MAM_16_E_tot - MAM_16_clim_E_tot.data
#
def compare_melt():
    totm1516 = iris.load_cube(filepath + 'total_melt_amnt_during_15-16_melt_season.nc')
    totm1516 = totm1516.data*108
    totm1516 = np.copy(totm1516)
    Larsen_mask = np.zeros((220, 220))
    lsm_subset = lsm.data[:150, 90:160]
    Larsen_mask[:150, 90:160] = lsm_subset
    Larsen_mask[orog[:, :].data > 2000] = 0
    Larsen_mask = np.logical_not(Larsen_mask)
    totm1516[Larsen_mask == 1] = np.nan
    melt_seas_mn = np.nanmean(totm1516)
    melt_seas_tot = np.nansum(totm1516)
    totm = np.copy(MAM_16_melt)
    totm[Larsen_mask == 1] = np.nan
    MAM16_mn = np.nanmean(totm)
    MAM16_tot = np.nansum(totm)
    return melt_seas_mn, melt_seas_tot, MAM16_mn, MAM16_tot, totm1516, totm

#melt_seas_mn, melt_seas_tot, MAM16_mn, MAM16_tot, totm1516, totm = compare_melt()


def calc_total_melt_LC(melt_var):
    totm = np.copy(melt_var)
    totm[lsm == 0.] = np.nan
    # calculate total melt in one gridbox
    melt_tot_per_gridbox = np.zeros(totm.shape)
    for i in range(220):
        for j in range(220):
            melt_tot_per_gridbox[i, j] = totm[i, j] * (4000 * 4000)  # total kg per gridbox
    melt_tot = np.nansum(melt_tot_per_gridbox)
    return melt_tot/10**12

#calc_total_melt_LC(totm1516)

# Identify patterns

print('Loading observed indices... \n\n')
var_dict['u_Plev'] = load_vars('1998-2017', 'u_wind_P_levs')
var_dict['u_prof']  = load_vars('1998-2017', 'u_wind_full_profile')
SAM_full = pd.read_csv(filepath + 'Daily_mean_SAM_index_1998-2017.csv', usecols = ['SAM'], dtype = np.float64, header = 0, na_values = '*******')
SAM_full.index = pd.date_range('1998-01-01', '2017-12-31', freq = 'D')
SAM_full = SAM_full.values[:,0]
ENSO_full = iris.load_cube(filepath + 'inino34_daily.nc')
ENSO_full = pd.DataFrame(data = ENSO_full[6117:-731].data) # subset to 3 months before 1998-01-01 to 2017-12-31
ENSO_full = ENSO_full.rolling(window = 90).mean() # first 90 days will be nans
ENSO_full = ENSO_full[90:].values
ENSO_full = ENSO_full[:7305,0]

var_dict['SAM'] = SAM_full
var_dict['ENSO'] = ENSO_full

#try:
#    for i in ['MSLP', 'T', 'u', 'v', 'SIC']:
#        real_lon, real_lat = rotate_data(full_srs[i], np.ndim(full_srs[i])-2, np.ndim(full_srs[i])-1)
        #real_lon, real_lat = rotate_data(clim_srs[i], np.ndim(clim_srs[i]) - 2, np.ndim(clim_srs[i]) - 1)
        #clim_srs[i].convert_units(unit_dict[i])
#        full_srs[i].convert_units(unit_dict[i])
#except:
#    for i in ['T', 'u', 'v', 'SIC']:
#        real_lon, real_lat = rotate_data(full_srs[i], np.ndim(full_srs[i])-2, np.ndim(full_srs[i])-1)
        #real_lon, real_lat = rotate_data(clim_srs[i], np.ndim(clim_srs[i]) - 2, np.ndim(clim_srs[i]) - 1)
        #clim_srs[i].convert_units(unit_dict[i])
#        full_srs[i].convert_units(unit_dict[i])

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

print('Calculating Froude number... \n\n')
Fr, h_hat = Froude_number(np.mean(var_dict['u_prof'].data[:, 4:23, 75:175, 4:42], axis = (1,2,3)))
var_dict['Fr'] = Fr[:7305]

def apply_composite_mask(regime, var):
    regime_mask = np.zeros(var.shape[0])
    if regime == 'barrier':
        indicator_var = var_dict['v_10m'][:,0,:,:]
        threshold = 5.
        region = ((4,110), (150,-4))
    if regime == 'cross-peninsula':
        indicator_var = var_dict['u_Plev'][:,3,:,:]
        threshold = 2.
        region = ((75,175), (4,42))#up to 75:175 (y), 4:42 (x)
    if regime == 'flow-over':
        indicator_var = var_dict['Fr']
        threshold = 0.5 # Froude number threshold
    if regime == 'blocked':
        indicator_var = var_dict['Fr']
        threshold = 0.5 # Froude number threshold 'flow-over' conditions (Orr et al., 2008; van Lipzig et al., 2008)
    if regime == 'SIC_Weddell_L' :
        indicator_var = var_dict['SIC'][:,0,:,:].data
        threshold = 0.5
        region = ((4, 150),(150, -4)) # Weddell Sea region = x: 150:-4, y: 4:150
    elif regime == 'SIC_Weddell_H':
        indicator_var = var_dict['SIC'][:, 0, :, :]
        threshold = 0.85
        region = ((4, 150), (150, -4))  # Weddell Sea region = x: 150:-4, y: 4:150
    if regime == 'ASL':
        indicator_var = anoms['MSLP'][:,0,:,:]
        threshold = -5.
        region = ((4,110), (4, 75)) # ASL region = x: 4:75, y: 4:110
    if regime == 'SAM+':
        indicator_var = var_dict['SAM']
        threshold = 1.36 # plus one standard deviation
    if regime == 'SAM-':
        indicator_var = var_dict['SAM']
        threshold = -1.36 # minus one standard deviation
    if regime == 'ENSO+':
        indicator_var = var_dict['ENSO']
        threshold = 0.5
    if regime == 'ENSO-':
        indicator_var = var_dict['ENSO']
        threshold = -0.5
    if regime == 'cloudy':
        indicator_var = var_dict['mean_cloud']
        threshold = 0.75
    if regime == 'clear':
        indicator_var = var_dict['mean_cloud']
        threshold = 0.31
    if regime == 'melt75':
        indicator_var = var_dict['Larsen_melt']
        threshold = np.nanpercentile(indicator_var, q = 75)
    if regime == 'melt25':
        indicator_var = var_dict['Larsen_melt']
        threshold = np.nanpercentile(indicator_var, q = 25)
    if regime == 'LWP75':
        indicator_var = var_dict['Larsen_LWP']
        threshold = np.nanpercentile(indicator_var, q = 75)
    if regime == 'LWP25':
        indicator_var = var_dict['Larsen_LWP']
        threshold = np.nanpercentile(indicator_var, q = 25)
    for each_day in range(var.shape[0]):
        if regime == 'SIC_Weddell_L' or regime == 'ASL':
            if np.mean(indicator_var[each_day, region[0][0]:region[0][1], region[1][0]:region[1][1]].data) <= threshold:
                regime_mask[each_day] = 1.
            else:
                regime_mask[each_day] = 0.
        elif regime == 'SIC_Weddell_H':
            if np.mean(indicator_var[each_day, region[0][0]:region[0][1], region[1][0]:region[1][1]].data) >= threshold:
                regime_mask[each_day] = 1.
            else:
                regime_mask[each_day] = 0.
        elif regime == 'SAM+' or regime == 'ENSO+' or regime == 'flow-over' or regime == 'cloudy' or regime == 'melt75' or regime == 'LWP75':
            regime_mask[indicator_var < threshold] = 0.
            regime_mask[indicator_var > threshold] = 1.
        elif regime == 'SAM-' or regime == 'ENSO-' or regime == 'blocked' or regime == 'clear' or regime == 'melt25' or regime == 'LWP25':
            regime_mask[indicator_var <  threshold] = 1.
            regime_mask[indicator_var >  threshold] = 0.
        else:
            if np.mean(indicator_var[each_day, region[0][0]:region[0][1], region[1][0]:region[1][1]].data) >= threshold: # if mean v wind > 2.0 m s-1 within barrier jet region (excluding model domain blending halo)
                regime_mask[each_day] = 1.
            else:
                regime_mask[each_day] = 0.
        # Apply mask to variable requested
    composite_array = np.zeros((220,220))
    for each_day in range(var.shape[0]):
        if regime_mask[each_day] == 1.:
            composite_array= composite_array + var[each_day,:,:]
        else:
            composite_array = composite_array
    composite_array = composite_array/np.count_nonzero(regime_mask)
    return composite_array, regime_mask

# Load SEB and cloud terms
for i in ['land_snow_melt_amnt','cl_frac', 'land_snow_melt_flux', 'surface_SW_net', 'surface_LW_net', 'latent_heat', 'sensible_heat', 'E_tot', 'total_column_liquid']:
    var_dict[i] = load_vars('1998-2017', i)
    clims[i] = create_clim(i)
    anoms[i] = iris.cube.Cube(data=(var_dict[i].data[:7305] - clims[i].data))

var_dict['mean_cloud'] = np.mean(var_dict['cl_frac'][:7305, 0, 40:140, 85:155].data, axis = (1,2))
var_dict['Larsen_LWP'] = np.mean(var_dict['total_column_liquid'][:7305, 0, 40:140, 85:155].data, axis = (1,2))

def standardise_anomaly(var, anom, clim):
    ''' Standardises anomalies by dividing daily mean values by the standard deviation of the daily climatology value for the entire period.

    i.e. std_anom = anom/std(clim)

    after Hart & Grumm (2001) Monthly Weather Review 129 (9) 2426-2442.

    Inputs:

        - var: daily mean time series (3D)

        - anom: time series of daily mean anomalies (3D)

        - clim: daily mean climatology (3D)

    Outputs:

        - std_anom: standardised anomaly

    '''
    std = np.std(clim, axis = 0)
    std_anom = np.zeros(var.shape)
    for each_ts in range(var.shape[0]):
        std_anom[each_ts, :,:] = anom[each_ts, :,:]/std
    return std_anom

def normalise_anomaly(anom, scale):
    '''Normalises anomalies by using the scikit.learn.preprocessing module.

    Inputs:

        - anom: time series of daily mean anomalies (3D)

    Outputs:

        - norm_anom: normalised anomaly

    '''
    from sklearn.preprocessing import minmax_scale
    shape = anom.shape
    anom = anom.reshape(shape[0], int(shape[1]*shape[2]))
    norm_anom = minmax_scale(anom, feature_range=scale, axis = 0)
    norm_anom = norm_anom.reshape(shape)
    return norm_anom

# Normalise anomalies
print('Normalising anomalies... \n\n')

#cl_anom_norm = normalise_anomaly(anoms['cl_frac'].data[:,0,:,:])
#SW_anom_norm = normalise_anomaly(anoms['surface_SW_net'][:,0,:,:].data, scale = (-1,1))
#LW_anom_norm = normalise_anomaly(anoms['surface_LW_net'].data[:,0,:,:], scale = (-1,1))
#HL_anom_norm = normalise_anomaly( anoms['latent_heat'].data[:,0,:,:], scale = (-1,1))
#HS_anom_norm = normalise_anomaly(anoms['sensible_heat'].data[:,0,:,:], scale = (-1,1))
melt_land = np.copy(anoms['land_snow_melt_flux'].data[:,0,:,:]) # remove sea points
melt_land[melt_land > 800] = np.nan
#melt_land[melt_land > 75] = 75
#melt_anom_norm = normalise_anomaly(melt_land, scale = (-1,1))
#E_anom_norm = normalise_anomaly( anoms['E_tot'].data[:,0,:,:], scale = (-1,1))
var_dict['Larsen_melt'] = np.mean(var_dict['land_snow_melt_amnt'][:7305, 0,40:140, 85:155].data, axis = (1,2))
var_dict['Larsen_melt'][var_dict['Larsen_melt'] == 0] = np.nan

def apply_Larsen_mask(var):
    # Make ice shelf mask
    Larsen_mask = np.zeros((220, 220))
    lsm_subset = surf['lsm'].data[:150, 90:160]
    Larsen_mask[:150, 90:160] = lsm_subset
    Larsen_mask[surf['orog'].data > 100] = 0
    Larsen_mask = np.logical_not(Larsen_mask)
    # Apply mask to variable requested
    var_masked = np.ma.masked_array(var, mask=np.broadcast_to(Larsen_mask, var.shape))
    return var_masked, Larsen_mask

#for i in var_dict.keys():
#    real_lon, real_lat = rotate_data(var_dict[i], np.ndim(var_dict[i]) - 2, np.ndim(var_dict[i]) - 1)

def plot_SEB_composites(var1, var2, var3, var4, var5, var6, regime):
    fig, axs = plt.subplots(3,2, frameon=False, figsize=(11, 18))
    axs = axs.flatten()
    fig.patch.set_visible(False)
    var_list = [var1[:7305], var2[:7305], var3[:7305], var4[:7305], var5[:7305], var6[:7305]]
    for i, j in zip(axs, ['SW$_{net}$', 'LW$_{net}$', 'H$_{L}$','H$_{S}$',  'E$_{tot}$', 'E$_{melt}$',]):
        i.set_title(j, color = 'dimgrey', fontsize = 34)
    plt.axis = 'off'
    lims = {'SW$_{net}$': (-100, 100),
             'LW$_{net}$': (-100, 100),
            'H$_{L}$': (-100,50),
            'H$_{S}$': (-100, 100),
            'E$_{tot}$': (-50,50),
            'E$_{melt}$': (-5,5)
    }
    for ax in [axs[1], axs[3], axs[5]]:
        ax.yaxis.tick_right()
    for ax, cf_var in zip(axs[:-1], var_list[:-1]):
        ax.axis = 'off'
        ax.tick_params(which='both', axis='both', labelsize=24, labelcolor='dimgrey', pad=10, size=0, tick1On=False,
                       tick2On=False)
        PlotLonMin = np.min(real_lon)
        PlotLonMax = np.max(real_lon)
        PlotLatMin = np.min(real_lat)
        PlotLatMax = np.max(real_lat)
        XTicks = np.linspace(PlotLonMin, PlotLonMax, 3)
        XTickLabels = [None] * len(XTicks)
        for i, XTick in enumerate(XTicks):
            if XTick < 0:
                XTickLabels[i] = '{:.0f}{:s}'.format(np.abs(XTick), '$^{\circ}$W')
            else:
                XTickLabels[i] = '{:.0f}{:s}'.format(np.abs(XTick), '$^{\circ}$E')#
        plt.sca(ax)
        plt.xticks(XTicks, XTickLabels)
        ax.set_xlim(PlotLonMin, PlotLonMax)
        ax.tick_params(which='both', pad=10, labelsize = 24, color = 'dimgrey')
        YTicks = np.linspace(PlotLatMin, PlotLatMax, 4)
        YTickLabels = [None] * len(YTicks)
        for i, YTick in enumerate(YTicks):
            if YTick < 0:
                YTickLabels[i] = '{:.0f}{:s}'.format(np.abs(YTick), '$^{\circ}$S')
            else:
                YTickLabels[i] = '{:.0f}{:s}'.format(np.abs(YTick), '$^{\circ}$N')
        plt.yticks(YTicks, YTickLabels)
        ax.set_ylim(PlotLatMin, PlotLatMax)
        ax.tick_params(which='both', axis='both', labelsize=24, labelcolor='dimgrey', pad=10, size=0, tick1On=False,tick2On=False)
        xlon, ylat = np.meshgrid(real_lon, real_lat)
        #cf_var, regime_mask = apply_composite_mask(regime, var)
        bwr_zero = shiftedColorMap(cmap=matplotlib.cm.bwr, min_val=lims[j][0], max_val=lims[j][1], name='bwr_zero', var=cf_var.data, start=0.15, stop=0.85)
        c = ax.pcolormesh(xlon, ylat, cf_var, cmap=bwr_zero, vmin=lims[j][0], vmax=lims[j][1], zorder=1)  # latlon=True, transform=ccrs.PlateCarree(),
        plt.sca(ax)
        CBarXTicks = np.linspace(lims[j][0], lims[j][1], num = 3)
        CBar = plt.colorbar(c, orientation='horizontal', extend='both', ticks=CBarXTicks)
        CBar.solids.set_edgecolor("face")
        CBar.outline.set_edgecolor('dimgrey')
        CBar.ax.tick_params(which='both', axis='both', labelsize=34, labelcolor='dimgrey', pad=10, size=0, tick1On=False,tick2On=False)
        CBar.outline.set_linewidth(2)
        coast = ax.contour(xlon, ylat, lsm.data, levels=[0], colors='#222222', lw=2, latlon=True, zorder=2)
        topog = ax.contour(xlon, ylat, orog.data, levels=[50], colors='#222222', linewidth=1.5, latlon=True, zorder=3)
    bwr_zero = shiftedColorMap(cmap=matplotlib.cm.bwr, min_val=-1., max_val=0., name='bwr_zero', var=var6.data,
                               start=0.15, stop=0.85)
    axs[5].pcolormesh(xlon,ylat, var6, vmin =-1, vmax = 0, cmap = bwr_zero)
    coast = axs[5].contour(xlon, ylat, lsm.data, levels=[0], colors='#222222', lw=2, latlon=True, zorder=2)
    topog = axs[5].contour(xlon, ylat, orog.data, levels=[50], colors='#222222', linewidth=1.5, latlon=True, zorder=3)
    #CBarXTicks = [-1, 0, 1]  # CLevs[np.arange(0,len(CLevs),int(np.ceil(len(CLevs)/5.)))]
    #CBAxes = fig.add_axes([0.2, 0.15, 0.6, 0.015])
    #CBar = plt.colorbar(c, cax=CBAxes, orientation='horizontal', extend='both', ticks=CBarXTicks)
    #CBar.set_label('Normalised flux anomaly (W m$^{-2}$)', fontsize=34, labelpad=10, color='dimgrey')
    #CBar.solids.set_edgecolor("face")
    #CBar.outline.set_edgecolor('dimgrey')
    #CBar.ax.tick_params(which='both', axis='both', labelsize=34, labelcolor='dimgrey', pad=10, size=0, tick1On=False,
    #                    tick2On=False)
    plt.subplots_adjust(bottom = 0.22, top = 0.9, hspace = 0.3, wspace = 0.25, right = 0.87)
    plt.savefig('/gws/nopw/j04/bas_climate/users/ellgil82/hindcast/figures/SEB_composite_subplot_' + regime + '.png')
    plt.savefig('/gws/nopw/j04/bas_climate/users/ellgil82/hindcast/figures/SEB_composite_subplot_' + regime + '.eps')
    #plt.show()

#plot_SEB_composites( SW_anom_norm, LW_anom_norm, HL_anom_norm, HS_anom_norm, E_anom_norm, melt_anom_norm , 'barrier')

print('Plotting some met composites for... \n\n')

#masks = {}
#for regime in  [ 'melt75', 'melt25','LWP25', 'LWP75']:#
#    print(regime + '\n\n')
#    c_var, regime_mask  = apply_composite_mask(regime, var_dict['MSLP'][:-1,0,:,:].data)
#    cf_var, regime_mask  = apply_composite_mask(regime, anoms['Tair_1p5m_daymax'][:,0,:,:].data) # try this with T anomalies instead
#    u_var, regime_mask  = apply_composite_mask(regime, var_dict['u_10m'][:-1,0,:,:].data)
#    v_var, regime_mask  = apply_composite_mask(regime, var_dict['v_10m'][:-1,0,:,:].data)
#    plot_synop_composite(cf_var, c_var, u_var, v_var, regime)
    #plot_SEB_composites( SW_anom_norm, LW_anom_norm, HL_anom_norm, HS_anom_norm, E_anom_norm, melt_anom_norm , regime)


def plot_melt(cf_var, regime):
    fig = plt.figure(frameon=False, figsize=(10, 11))  # !!change figure dimensions when you have a larger model domain
    fig.patch.set_visible(False)
    ax = fig.add_subplot(111)#, projection=ccrs.PlateCarree())
    plt.axis = 'off'
    plt.setp(ax.spines.values(), linewidth=0, color='dimgrey')
    PlotLonMin = np.min(real_lon)
    PlotLonMax = np.max(real_lon)
    PlotLatMin = np.min(real_lat)
    PlotLatMax = np.max(real_lat)
    XTicks = np.linspace(PlotLonMin, PlotLonMax, 4)
    XTickLabels = [None] * len(XTicks)
    for i, XTick in enumerate(XTicks):
        if XTick < 0:
            XTickLabels[i] = '{:.0f}{:s}'.format(np.abs(XTick), '$^{\circ}$W')
        else:
            XTickLabels[i] = '{:.0f}{:s}'.format(np.abs(XTick), '$^{\circ}$E')#
    plt.xticks(XTicks, XTickLabels)
    ax.set_xlim(PlotLonMin, PlotLonMax)
    ax.tick_params(which='both', pad=10, labelsize = 34, color = 'dimgrey')
    YTicks = np.linspace(PlotLatMin, PlotLatMax, 4)
    YTickLabels = [None] * len(YTicks)
    for i, YTick in enumerate(YTicks):
        if YTick < 0:
            YTickLabels[i] = '{:.0f}{:s}'.format(np.abs(YTick), '$^{\circ}$S')
        else:
            YTickLabels[i] = '{:.0f}{:s}'.format(np.abs(YTick), '$^{\circ}$N')
    plt.yticks(YTicks, YTickLabels)
    ax.set_ylim(PlotLatMin, PlotLatMax)
    ax.tick_params(which='both', axis='both', labelsize=34, labelcolor='dimgrey', pad=10, size=0, tick1On=False,
                   tick2On=False)
    bwr_zero = shiftedColorMap(cmap=matplotlib.cm.bwr, min_val=-3., max_val=3., name='bwr_zero', var=cf_var.data, start=0.15, stop=0.85) #-6, 3
    xlon, ylat = np.meshgrid(real_lon, real_lat)
    c = ax.pcolormesh(xlon, ylat, cf_var, cmap = bwr_zero, vmin = -3., vmax = 3.)
    #shaded = np.zeros((220,220))
    #shaded[60:130,:] = 1.
    #ax.contour(xlon, ylat, shaded, levels= [1.], colors='dimgrey', linewidths = 4, latlon=True)
    #ax.text(0., 1.1, zorder=6, transform=ax.transAxes, s='b', fontsize=32, fontweight='bold', color='dimgrey')
    coast = ax.contour(xlon, ylat, lsm.data, levels=[0], colors='#222222', lw=2, latlon=True, zorder=2)
    topog = ax.contour(xlon, ylat, orog.data, levels=[50], colors='#222222', linewidth=1.5, latlon=True, zorder=3)
    CBarXTicks = [-3,   0, 3,]  # CLevs[np.arange(0,len(CLevs),int(np.ceil(len(CLevs)/5.)))]
    CBAxes = fig.add_axes([0.25, 0.15, 0.6, 0.03])
    CBar = plt.colorbar(c, cax=CBAxes, orientation='horizontal', ticks=CBarXTicks, extend = 'both')  #
    CBar.set_label('Mean E$_{melt}$ anomaly (W m$^{-2}$)', fontsize=34, labelpad=10, color='dimgrey')
    CBar.solids.set_edgecolor("face")
    CBar.outline.set_edgecolor('dimgrey')
    CBar.ax.tick_params(which='both', axis='both', labelsize=34, labelcolor='dimgrey', pad=10, size=0, tick1On=False,
                        tick2On=False)
    CBar.outline.set_linewidth(2)
    plt.subplots_adjust(left = 0.2, bottom = 0.3, right = 0.85)
    plt.savefig('/gws/nopw/j04/bas_climate/users/ellgil82/hindcast/figures/Melt_composite_' + regime + '.png', transparent = True)
    plt.savefig('/gws/nopw/j04/bas_climate/users/ellgil82/hindcast/figures/Melt_composite_' + regime + '.eps', transparent = True)
    plt.show()

#    masks[regime] = regime_mask


#plot_melt(HL_anom_norm, regime + '_HL')

def melt_transect():
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.setp(ax.spines.values(), linewidth=3, color='dimgrey')
    ax.plot(np.max(MAM_16_melt[60:130,:],axis = 0), color = 'red', lw = 2.5)
    ax.set_ylabel('\n\n\n\nMax E$_{melt}$  \n(W m$^{-2}$)', fontname='SegoeUI semibold', color='dimgrey', rotation=0,
                  fontsize=20, labelpad=50)
    ax.yaxis.set_label_coords(-0.2,0.4)
    ax.set_xlim(0,220)
    ax.text(-0, 1.1, zorder=6, transform=ax.transAxes, s='b', fontsize=32, fontweight='bold',
            color='dimgrey')
    plt.tick_params(axis='x', which='both', labelbottom= False, tick1On=False, tick2On=False, labelcolor='dimgrey', pad=5)
    plt.tick_params(axis='y', which='both', labelsize=24, tick1On=False, tick2On=False, labelcolor='dimgrey', pad=5)
    plt.subplots_adjust(left = 0.2,  right = 0.85, top = 0.85)
    plt.savefig('/gws/nopw/j04/bas_climate/users/ellgil82/hindcast/figures/melt_transect_MAM_16.eps')
    plt.savefig('/gws/nopw/j04/bas_climate/users/ellgil82/hindcast/figures/melt_transect_MAM_16.png')
    plt.show()

#melt_transect()

print(' Calculating melt during each regime')
Larsen_melt = var_dict['land_snow_melt_amnt'].data[:7305,0,40:140, 85:155]
totm = np.copy(Larsen_melt)
# calculate total melt in one gridbox
melt_tot_per_gridbox = np.zeros((7305, 100,70))
for t in range(7305):
    totm[lsm[40:140, 85:155] == 0.] = np.nan
    for i in range(100):
        for j in range(70):
            melt_tot_per_gridbox[t, i, j] = totm[t, i, j] * (4000 * 4000)  # total kg per gridbox

melt_tot = np.nansum(np.ma.masked_greater(melt_tot_per_gridbox, 3.20000006e+25), axis = 0) # remove sea values
melt_tot = np.nansum(melt_tot)/10**12 # integrated ice shelf melt amount (in Gt!)

melt_list = []
regime_list = []

for regime in ['blocked', 'flow-over', 'cloudy', 'clear']:#, 'melt25', 'melt75', 'LWP25', 'LWP75', 'SAM+', 'SAM-', 'ENSO+', 'ENSO-', 'ASL', 'barrier', ]: # 'SIC_Weddell_L', 'SIC_Weddell_H',
    print('\n\nPlotting synoptic composites during ' + regime + '...\n\n')
    #c_var, regime_mask = apply_composite_mask(regime, var_dict['MSLP'][:7305, 0, :, :].data)
    #cf_var, regime_mask = apply_composite_mask(regime, anoms['Tair_1p5m_daymax'][:7305, 0, :, :].data)  # try this with Tmax anomalies instead
    #u_var, regime_mask = apply_composite_mask(regime, var_dict['u_10m'][:7305, 0, :, :].data)
    #v_var, regime_mask = apply_composite_mask(regime, var_dict['v_10m'][:7305, 0, :, :].data)
    #plot_synop_composite(cf_var, c_var, u_var, v_var, regime)
    print('\n\nPlotting SEB composites during ' + regime + '...\n\n')
    SW_masked, regime_mask = apply_composite_mask(regime, anoms['surface_SW_net'])
    LW_masked, regime_mask = apply_composite_mask(regime, anoms['surface_LW_net'])
    HL_masked, regime_mask = apply_composite_mask(regime, anoms['latent_heat'])
    HS_masked, regime_mask = apply_composite_mask(regime, anoms['sensible_heat'])
    Etot_masked, regime_mask = apply_composite_mask(regime, anoms['Etot'])
    melt_masked, regime_mask = apply_composite_mask(regime, anoms['land_snow_melt_flux'])
    plot_SEB_composites(SW_masked, LW_masked, HL_masked, HS_masked, Etot_masked, melt_masked, regime)
    melt_masked, regime_mask = apply_composite_mask(regime, anoms['land_snow_melt_flux'].data[:7305,0,:,:])
    plot_melt(melt_masked, regime)
    regime_freq = (np.float(np.count_nonzero(regime_mask))/ 7305.)*100
    melt_regime = np.copy(melt_tot_per_gridbox) # copy total
    melt_regime[regime_mask == 0] = np.nan # apply mask
    melt_tot_regime = np.nansum(melt_regime, axis=0) # sum over time
    melt_tot_regime = np.ma.masked_greater(melt_tot_regime, 3e+20) # mask sea values
    melt_tot_regime = np.nansum(melt_tot_regime)/10**12 # sum across entire ice shelf, and return in Gt meltwater
    melt_tot_regime_pct = (melt_tot_regime/melt_tot)*100 # find as percentage of melt
    print(regime + ' associated with ' +  str(melt_tot_regime_pct) + ' % of melting during hindcast\n\n')
    print(regime + ' occurs ' + str(regime_freq) + ' % of the time during hindcast\n\n')
    melt_list.append(melt_tot_regime_pct)
    regime_list.append(regime_freq)

df = pd.DataFrame(index = ['blocked', 'flow-over', 'cloudy', 'clear', 'melt25', 'melt75','LWP25', 'LWP75',  'SAM+', 'SAM-', 'ENSO+', 'ENSO-', 'ASL', 'barrier'])
df['melt_pct'] = pd.Series(melt_list, index = ['blocked', 'flow-over', 'cloudy', 'clear', 'melt25', 'melt75','LWP25', 'LWP75',  'SAM+', 'SAM-', 'ENSO+', 'ENSO-', 'ASL', 'barrier'])
df['regime_freq'] = pd.Series(regime_list, index = ['blocked', 'flow-over', 'cloudy', 'clear', 'melt25', 'melt75', 'LWP25', 'LWP75', 'SAM+', 'SAM-', 'ENSO+', 'ENSO-', 'ASL', 'barrier'])
df.to_csv(filepath + 'regime_melt_freq.csv')

print(regime_freq)

print(regime)

def plot_melt_composites():
    fig, axs = plt.subplots(4,2, frameon=False, figsize=(11, 18))
    axs = axs.flatten()
    fig.patch.set_visible(False)
    reg_list = ['barrier', 'ASL', 'SAM+', 'SAM-', 'ENSO+', 'ENSO-' 'blocked', 'flow-over']
    for ax in [axs[1], axs[3], axs[5]]:
        ax.yaxis.tick_right()
    for i in range(6):
        melt, reg_mask = apply_composite_mask(reg_list[i], anoms['land_snow_melt_flux'].data[:7305,0,:,:])
        ax = axs[i]
        ax.set_title(reg_list[i], color = 'dimgrey', fontsize = 34)
        ax.axis = 'off'
        ax.tick_params(which='both', axis='both', labelsize=24, labelcolor='dimgrey', pad=10, size=0, tick1On=False, tick2On=False)
        PlotLonMin = np.min(real_lon)
        PlotLonMax = np.max(real_lon)
        PlotLatMin = np.min(real_lat)
        PlotLatMax = np.max(real_lat)
        XTicks = np.linspace(PlotLonMin, PlotLonMax, 3)
        XTickLabels = [None] * len(XTicks)
        for i, XTick in enumerate(XTicks):
            if XTick < 0:
                XTickLabels[i] = '{:.0f}{:s}'.format(np.abs(XTick), '$^{\circ}$W')
            else:
                XTickLabels[i] = '{:.0f}{:s}'.format(np.abs(XTick), '$^{\circ}$E')#
        plt.sca(ax)
        plt.xticks(XTicks, XTickLabels)
        ax.set_xlim(PlotLonMin, PlotLonMax)
        ax.tick_params(which='both', pad=10, labelsize = 24, color = 'dimgrey')
        YTicks = np.linspace(PlotLatMin, PlotLatMax, 4)
        YTickLabels = [None] * len(YTicks)
        for i, YTick in enumerate(YTicks):
            if YTick < 0:
                YTickLabels[i] = '{:.0f}{:s}'.format(np.abs(YTick), '$^{\circ}$S')
            else:
                YTickLabels[i] = '{:.0f}{:s}'.format(np.abs(YTick), '$^{\circ}$N')
        plt.yticks(YTicks, YTickLabels)
        ax.set_ylim(PlotLatMin, PlotLatMax)
        ax.tick_params(which='both', axis='both', labelsize=24, labelcolor='dimgrey', pad=10, size=0, tick1On=False,
                       tick2On=False)
       # bwr_zero = shiftedColorMap(cmap=matplotlib.cm.bwr, min_val=-6., max_val=3., name='bwr_zero', var=cf_var.data, start=0.15, stop=0.85)
        xlon, ylat = np.meshgrid(real_lon, real_lat)
        c = ax.pcolormesh(xlon, ylat, melt, vmin = -3, vmax = 3, cmap = 'bwr')#, cmap=bwr_zero, vmin=-6., vmax=3., zorder=1)  # latlon=True, transform=ccrs.PlateCarree(),
        coast = ax.contour(xlon, ylat, lsm.data, levels=[0], colors='#222222', lw=2, latlon=True, zorder=2)
        topog = ax.contour(xlon, ylat, orog.data, levels=[50], colors='#222222', linewidth=1.5, latlon=True, zorder=3)
    CBarXTicks = [-3, 0, 3]  # CLevs[np.arange(0,len(CLevs),int(np.ceil(len(CLevs)/5.)))]
    CBAxes = fig.add_axes([0.25, 0.15, 0.55, 0.02])
    CBar = plt.colorbar(c, cax=CBAxes, orientation='horizontal', extend='both', ticks=CBarXTicks)
    CBar.set_label('Melt flux anomaly (W m$^{-2}$)', fontsize=34, labelpad=10, color='dimgrey')
    CBar.solids.set_edgecolor("face")
    CBar.outline.set_edgecolor('dimgrey')
    CBar.ax.tick_params(which='both', axis='both', labelsize=34, labelcolor='dimgrey', pad=10, size=0, tick1On=False, tick2On=False)
    CBar.outline.set_linewidth(2)
    plt.subplots_adjust(bottom = 0.22, top = 0.9, hspace = 0.3, wspace = 0.25, right = 0.87)
    plt.savefig('/gws/nopw/j04/bas_climate/users/ellgil82/hindcast/figures/Melt_flux_anomalies_all_regimes.png')
    plt.savefig('/gws/nopw/j04/bas_climate/users/ellgil82/hindcast/figures/Melt_flux_anomalies_all_regimes.eps')
    plt.show()

plot_melt_composites()

## Approach 1: Define specific large-scale regimes to evaluate:
##
## e.g. Southerly barrier wind
## When wind directions in the lower right hand side of the domain (define gridboxes) is southerly and temperatures are
## colder than average (?)
##
## e.g. Deep ASL
## when MSLP in a specific region is < threshold (?) -> or could use GLM output
##
## e.g. Cross-Peninsula flow
## when wind at one Rossby Radius of deformation away is westerly (u > 2.0 m s-1)
##
## Approach 2: Composite large-scale meteorological variables during high melt years
## e.g. 2000/01, 2003/04, 2006/07, 2008/09
##
## Look at time series of melt amount to determine whether it was early/late/all season melt, i.e. was it one or two
## significant melt events, or the cumulative effect of many smaller ones?
##
## --> are the patterns in melt the same if we look at melt duration rather than melt amount?
##
## Composite daily mean anomalies from daily mean climatology in variables like atmospheric transmissivity/
## deseasonalised SW(?), air temperature, MSLP, winds, SEB components.

    # plot "inlet" mean
    # plot "ice shelf" mean
    # also plot map of these two sub-regions (i.e. mask the ice shelf out, then cut it down the middle vertically)

def test_rels(x_var, y_var):
    x = load_vars('1998-2017', x_var)
    y = load_vars('1998-2017', y_var)
    x_clim = create_clim(x_var)
    y_clim = create_clim(y_var)
    x_clim, mask = apply_Larsen_mask(x_clim.data)
    y_clim, mask = apply_Larsen_mask(y_clim.data)
    x, mask = apply_Larsen_mask(x.data)
    y, mask = apply_Larsen_mask(y.data)
    x_anom = np.mean(x, axis = (1,2,3))[:7305] - np.mean(x_clim, axis = (1,2,3))[:7305]
    y_anom = np.mean(y, axis = (1,2,3))[:7305] - np.mean(y_clim, axis = (1,2,3))[:7305]
    slope, intercept, r_value, p_value, std_err = stats.linregress(x_anom, y_anom)
    if p_value <= 0.05:
        print('Correlation coefficient between ' + x_var + ' and ' + y_var + ' anomalies:\n\n' + str(r_value) + '\n\n')
    else:
        print('No significant correlation between ' + x_var + ' and ' + y_var + ' anomalies.\n\n')
    slope, intercept, r_value, p_value, std_err = stats.linregress(np.mean(x, axis = (1,2,3))[:7305], np.mean(y, axis = (1,2,3))[:7305])
    if p_value <= 0.05:
        print('Correlation coefficient between ' + x_var + ' and ' + y_var + ' mean series:\n\n' + str(r_value)  + '\n\n')
    else:
        print('No significant correlation between ' + x_var + ' and ' + y_var + ' mean series.\n\n')
    return x, y, x_anom, y_anom, x_clim, y_clim, r_value, p_value, std_err

#melt, SWnet, melt_anom, SW_anom, melt_clim, SW_clim, r_value, p_value, std_err = test_rels('land_snow_melt_flux', 'surface_SW_net')
#melt, LWnet, melt_anom, LW_anom, melt_clim, LW_clim, r_value, p_value, std_err = test_rels('land_snow_melt_flux', 'surface_LW_net')
#melt, SWdown, melt_anom, SWd_anom, melt_clim, SWd_clim, r_value, p_value, std_err = test_rels('land_snow_melt_flux', 'surface_SW_down')
#melt, LWdown, melt_anom, LWd_anom, melt_clim, LWd_clim, r_value, p_value, std_err = test_rels('land_snow_melt_flux', 'surface_LW_down')
#melt, cl, melt_anom, cl_anom, melt_clim, cl_clim, r_value, p_value, std_err = test_rels('land_snow_melt_flux', 'cl_frac')
#melt, LWP, melt_anom, LWP_anom, melt_clim, LWP_clim, r_value, p_value, std_err = test_rels('land_snow_melt_flux', 'total_column_liquid')
#melt, IWP, melt_anom, IWP_anom, melt_clim, IWP_clim, r_value, p_value, std_err = test_rels('land_snow_melt_flux', 'total_column_ice')


def calculate_pvalues(df):
    from scipy.stats import pearsonr
    df = df.dropna()._get_numeric_data()
    dfcols = pd.DataFrame(columns=df.columns)
    pvalues = dfcols.transpose().join(dfcols, how='outer')
    rvalues = dfcols.transpose().join(dfcols, how='outer')
    for r in df.columns:
        for c in df.columns:
            pvalues[r][c] = round(pearsonr(df[r], df[c])[1], 4)
            rvalues[r][c] = round(pearsonr(df[r], df[c])[0], 4)
    rvalues[pvalues>0.01] = 0
    return pvalues, rvalues

def calc_seas_corrs():
    # reload AWS data
    os.chdir(filepath)
    var_list = []
    clim_list = []
    anom_list = []
    idx = ['surface_SW_net', 'surface_LW_net', 'surface_SW_down', 'surface_LW_down', 'latent_heat', 'sensible_heat', 'land_snow_melt_flux', 'cl_frac', 'total_column_liquid', 'total_column_ice']
    for i in idx:
        var = load_vars('1998-2017', i)
        clim = create_clim(i)
        var_masked = var[:,0,40:140, 85:155].data
        clim_masked= clim[:,0,40:140, 85:155].data
        anom = np.mean(var_masked, axis=(1, 2))[:7305] - np.mean(clim_masked, axis=(1, 2))[:7305]
        var_list.append(np.mean(var_masked, axis=(1, 2))[:7305])
        clim_list.append(np.mean(clim_masked, axis=(1, 2))[:7305])
        anom_list.append(anom)
    # load time series into dateframe
    var_df = pd.DataFrame(var_list)
    clim_df = pd.DataFrame(clim_list)
    anom_df = pd.DataFrame(anom_list)
    var_df.index = idx
    anom_df.index = idx
    clim_df.index = idx
    # index by datetime
    var_df = var_df.transpose()
    anom_df = anom_df.transpose()
    clim_df = clim_df.transpose()
    var_df['Time'] = pd.date_range(datetime(1998,1,1,0,0,0),datetime(2017,12,31,23,59,59), freq = 'D')
    var_df.index = var_df['Time']
    anom_df.index = var_df['Time']
    clim_df.index = var_df['Time']
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
    months = [g for n, g in var_df.groupby(pd.TimeGrouper('M'))]
    var_DJF = pd.DataFrame()
    var_MAM = pd.DataFrame()
    var_JJA = pd.DataFrame()
    var_SON = pd.DataFrame()
    anom_DJF = pd.DataFrame()
    anom_MAM = pd.DataFrame()
    anom_JJA = pd.DataFrame()
    anom_SON = pd.DataFrame()
    clim_DJF = pd.DataFrame()
    clim_MAM = pd.DataFrame()
    clim_JJA = pd.DataFrame()
    clim_SON = pd.DataFrame()
    for yr in range(20):
        var_DJF = pd.concat((var_DJF, months[dec[yr]], months[jan[yr]], months[feb[yr]]))
        var_MAM = pd.concat((var_MAM, months[mar[yr]], months[apr[yr]], months[may[yr]]))
        var_JJA = pd.concat((var_JJA, months[jun[yr]], months[jul[yr]], months[aug[yr]]))
        var_SON = pd.concat((var_SON, months[sep[yr]], months[oct[yr]], months[nov[yr]]))
    months = [g for n, g in anom_df.groupby(pd.TimeGrouper('M'))]
    for yr in range(20):
        anom_DJF = pd.concat((months[dec[yr]], months[jan[yr]], months[feb[yr]]))
        anom_MAM = pd.concat((months[mar[yr]], months[apr[yr]], months[may[yr]]))
        anom_JJA = pd.concat((months[jun[yr]], months[jul[yr]], months[aug[yr]]))
        anom_SON = pd.concat((months[sep[yr]], months[oct[yr]], months[nov[yr]]))
    months = [g for n, g in clim_df.groupby(pd.TimeGrouper('M'))]
    for yr in range(20):
        clim_DJF = pd.concat((months[dec[yr]], months[jan[yr]], months[feb[yr]]))
        clim_MAM = pd.concat((months[mar[yr]], months[apr[yr]], months[may[yr]]))
        clim_JJA = pd.concat((months[jun[yr]], months[jul[yr]], months[aug[yr]]))
        clim_SON = pd.concat((months[sep[yr]], months[oct[yr]], months[nov[yr]]))
    # run validation on each season in turn
    seas_names = ['DJF', 'MAM', 'JJA', 'SON']
    iteration = 0
    for a in [var_DJF, var_MAM, var_JJA, var_SON]:
        pvalues, rvalues = calculate_pvalues(a)
        rvalues.to_csv(filepath + '1998-2017_ice_shelf_integrated_' + seas_names[iteration]  + '_seasonal_correlations_r_full_srs_daily.csv')
        pvalues.to_csv(filepath + '1998-2017_ice_shelf_integrated_' + seas_names[iteration]  + '_seasonal_correlations_p_full_srs_daily.csv')
        a.to_csv(filepath + '1998-2017_ice_shelf_integrated_' + seas_names[iteration] + '_seasonal_full_srs_daily.csv')
        iteration = iteration + 1
    iteration = 0
    for a in [anom_DJF, anom_MAM, anom_JJA, anom_SON]:
        pvalues, rvalues = calculate_pvalues(a)
        rvalues.to_csv(filepath + '1998-2017_ice_shelf_integrated_' + seas_names[iteration] + '_seasonal_correlations_r_anomalies_daily.csv')
        pvalues.to_csv(filepath + '1998-2017_ice_shelf_integrated_' + seas_names[iteration] + '_seasonal_correlations_p_anomalies_daily.csv')
        a.to_csv(filepath + '1998-2017_ice_shelf_integrated_' + seas_names[iteration] + '_seasonal_anomalies_daily.csv')
        iteration = iteration + 1
    iteration = 0
    for a in [clim_DJF, clim_MAM, clim_JJA, clim_SON]:
        pvalues, rvalues = calculate_pvalues(a)
        rvalues.to_csv(filepath + '1998-2017_ice_shelf_integrated_' + seas_names[iteration] + '_seasonal_correlations_r_climatology_daily.csv')
        pvalues.to_csv(filepath + '1998-2017_ice_shelf_integrated_' + seas_names[iteration] + '_seasonal_correlations_p_climatology_daily.csv')
        a.to_csv(filepath + '1998-2017_ice_shelf_integrated_' + seas_names[iteration] + '_seasonal_climatologies_daily.csv')
        iteration = iteration + 1
    var_df.to_csv(filepath + '1998-2017_ice_shelf_integrated_full_time_srs_daily.csv')
    anom_df.to_csv(filepath + '1998-2017_ice_shelf_integrated_anomalies_daily.csv')
    clim_df.to_csv(filepath + '1998-2017_ice_shelf_integrated_climatologies_daily.csv')
    return var_df, anom_df, clim_df

print('\nCalculating seasonal correlations\n')

#var_df, anom_df, clim_df = calc_seas_corrs()

# Do seasonal compositing

def seas_composite():
    for seas in ['DJF', 'MAM', 'JJA', 'SON']:
        var_dict = {}
        clims = {}
        anoms = {}
        for i in [ 'Tair_1p5m', 'Tair_1p5m_daymax', 'MSLP', 'u_10m', 'v_10m', 'SIC', 'land_snow_melt_amnt','cl_frac', 'u_prof',
                   'u_Plev', 'land_snow_melt_flux', 'surface_SW_net', 'surface_LW_net', 'latent_heat', 'sensible_heat', 'E_tot']:#
            var_dict[i] = load_vars('1998-2017_' + seas, i)
            clims[i] = create_clim(seas + '_' + i)
            anoms[i] = iris.cube.Cube(data=(var_dict[i].data - clims[i].data))
        # Normalise anomalies
        print('Normalising anomalies... \n\n')
        cl_anom_norm = normalise_anomaly(anoms['cl_frac'].data)
        SW_anom_norm = normalise_anomaly(anoms['surface_SW_net'].data)
        LW_anom_norm = normalise_anomaly(anoms['surface_LW_net'].data)
        HL_anom_norm = normalise_anomaly(anoms['latent_heat'].data)
        HS_anom_norm = normalise_anomaly(anoms['sensible_heat'].data)
        melt_anom_norm = normalise_anomaly(anoms['land_snow_melt_flux'].data)
        E_anom_norm = normalise_anomaly(anoms['E_tot'].data)
        var_dict['Larsen_melt'] = np.mean(var_dict['land_snow_melt_amnt'][:, 40:140, 85:155].data, axis=(1, 2, 3))
        var_dict['Fr'] = Fr
        var_dict['mean_cloud'] = np.mean(var_dict['cl_frac'][:, 0, 40:140, 85:155].data, axis=(1, 2, 3))
        # Calculate months
        SAM_full = pd.read_csv(filepath + 'Daily_mean_SAM_index_1998-2017.csv', usecols=['SAM'], dtype=np.float64, header=0, na_values='*******')
        SAM_full.index = pd.date_range('1998-01-01', '2017-12-31', freq='D')
        months = [g for n, g in SAM_full.groupby(pd.TimeGrouper('M'))]
        month_ENSO = [g for n, g in ENSO_full.groupby(pd.TimeGrouper('M'))]
        ENSO_full = iris.load_cube(filepath + 'inino34_daily.nc')
        ENSO_full = pd.DataFrame(data=ENSO_full[6117:-732].data)  # subset to 3 months before 1998-01-01 to 2017-12-31
        ENSO_full = ENSO_full.rolling(window=90).mean()  # first 90 days will be nans
        ENSO_full = ENSO_full[90:]#.values
        ENSO_full.index = pd.date_range('1998-01-01', '2017-12-31', freq='D')
        #ENSO_full = ENSO_full[:7305, 0]
        ENSO_seas = pd.Series()
        SAM_seas = pd.Series()
        jan = np.arange(0,240,12)
        feb = np.arange(1,240,12)
        mar = np.arange(2,240,12)
        apr = np.arange(3,240,12)
        may = np.arange(4,240,12)
        jun = np.arange(5,240,12)
        jul = np.arange(6,240,12)
        aug = np.arange(7,240,12)
        sep = np.arange(8,240,12)
        oct = np.arange(9,240,12)
        nov = np.arange(10,240,12)
        dec = np.arange(11, 240, 12)
        for yr in range(20):
            if seas == 'DJF':
                SAM_seas = pd.concat((SAM_seas, months[dec[yr]], months[jan[yr]],months[feb[yr]] ))
                ENSO_seas =  pd.concat((ENSO_seas, month_ENSO[dec[yr]], month_ENSO[jan[yr]],month_ENSO[feb[yr]] ))
            elif seas == 'MAM':
                SAM_seas = pd.concat((months[mar[yr]], months[apr[yr]], months[may[yr]]))
                ENSO_seas = pd.concat((ENSO_seas, month_ENSO[mar[yr]], month_ENSO[apr[yr]], month_ENSO[may[yr]]))
            elif seas == 'JJA':
                SAM_seas = pd.concat((months[jun[yr]], months[jul[yr]], months[aug[yr]]))
                ENSO_seas = pd.concat((ENSO_seas, month_ENSO[jun[yr]], month_ENSO[jul[yr]], month_ENSO[aug[yr]]))
            elif seas == 'SON':
                SAM_seas = pd.concat((months[sep[yr]], months[oct[yr]], months[nov[yr]]))
                ENSO_seas = pd.concat((ENSO_seas, month_ENSO[sep[yr]], month_ENSO[oct[yr]], month_ENSO[nov[yr]]))
        SAM_seas = SAM_seas.values[:, 1]
        ENSO_seas = ENSO_seas.values[:,0]
        var_dict['SAM'] = SAM_seas
        var_dict['ENSO'] = ENSO_seas
        for regime in ['blocked', 'flow-over', 'cloudy', 'clear', 'melt25', 'melt75', 'SAM+', 'SAM-', 'ENSO+', 'ENSO-', 'ASL', 'barrier']:
            c_var, regime_mask = apply_composite_mask(regime, var_dict['MSLP'][:-1, 0, :, :].data)
            cf_var, regime_mask = apply_composite_mask(regime, anoms['Tair_1p5m_daymax'].data)  # try this with T anomalies instead
            u_var, regime_mask = apply_composite_mask(regime, var_dict['u'][:-1, 0, :, :].data)
            v_var, regime_mask = apply_composite_mask(regime, var_dict['v'][:-1, 0, :, :].data)
            plot_synop_composite(cf_var, c_var, u_var, v_var, regime)
            #plot_SEB_composites(SW_anom_norm, LW_anom_norm, HL_anom_norm, HS_anom_norm, E_anom_norm, melt_anom_norm, regime)
        plot_melt_composites()

seas_composite()

