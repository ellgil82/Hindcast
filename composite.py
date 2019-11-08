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
import cartopy.crs as ccrs
from divg_temp_colourmap import shiftedColorMap
import time
from sklearn.metrics import mean_squared_error
import datetime
import metpy
import metpy.calc
#from cftime import datetime
from datetime import datetime, time
from mpl_toolkits.basemap import Basemap

# Set up filepath
if host == 'jasmin':
    filepath = '/gws/nopw/j04/bas_climate/users/ellgil82/hindcast/output/alloutput/'
elif host == 'bsl':
    filepath = '/data/mac/ellgil82/hindcast/output/'

unit_dict = {'MSLP': 'hPa',
             'T': 'celsius',
             'Tair_1p5m': 'celsius',
             'Ts': 'celsius',
             'u_10m': 'm s-1',
             'v_10m': 'm s-1',
             'u': 'm s-1',
             'v': 'm s-1',
             'SIC' : 'unknown'
             }

## Load data
def load_vars(year, var_name, date_range):
    times = pd.date_range(date_range[0], date_range[1], freq = 'D')
    full_tsrs = pd.date_range(datetime(int(year),1,1,0,0,0),datetime(int(year),12,31,23,59,59), freq = 'D')
    start_idx = np.where(full_tsrs == times[0])
    end_idx = np.where(full_tsrs == times[-1])
    # Load daily means
    var = iris.load_cube(filepath + year + '_' + var_name + '_daymn.nc')
    # Load climatology
    clim = iris.load_cube( filepath +  var_name + '_climatology.nc')
    clim = clim[:365]
    for i in [var, clim]:
        i.convert_units(unit_dict[var_name])
    # Calculate anomaly
    anom = var.data - clim.data
    #anom = anom[start_idx[0][0]:end_idx[0][0],0,:,:]
    return var, clim, anom #var[start_idx[0][0]:end_idx[0][0],0,:,:], clim[start_idx[0][0]:end_idx[0][0],0,:,:], anom

date_range = ('2011-01-01', '2011-12-31') # date range must be in YYYY-MM-DD format

MSLP, Pclim, Panom = load_vars('2011', 'MSLP', date_range )
T, Tclim, Tanom = load_vars('2011', 'Tair_1p5m', date_range )
u, uclim, uanom = load_vars('2011', 'u_10m', date_range )
v, vclim, vanom = load_vars('2011', 'v_10m', date_range )
vclim = vclim[:,:,1:,:]
SIC, SICclim, SICanom = load_vars('2011', 'SIC', date_range )

def create_clim(var):
    import calendar
    clim_total = var.data
    for i in range(1998,2017):
        if calendar.isleap(i) == True: # if leap year, repeat final day of climatology
            clim_total = np.concatenate((clim_total, var.data), axis = 0)
            clim_total = np.concatenate((clim_total, var.data[-1:]), axis = 0)
        else:
            clim_total = np.concatenate((clim_total, var.data), axis = 0 )
    return iris.cube.Cube(clim_total)

Tclim_tot = create_clim(Tclim)
Pclim_tot = create_clim(Pclim)
uclim_tot = create_clim(uclim)
vclim_tot = create_clim(vclim)
SICclim_tot = create_clim(SICclim)

clim_srs = {'MSLP': Pclim_tot,
            'v': vclim_tot,
            'u': uclim_tot,
            'T': Tclim_tot,
            'SIC': SICclim_tot}

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
    fig = plt.figure(frameon=False, figsize=(10, 12))  # !!change figure dimensions when you have a larger model domain
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
    bwr_zero = shiftedColorMap(cmap=matplotlib.cm.bwr, min_val=-6., max_val=5., name='bwr_zero', var=cf_var.data, start=0.,
                               stop=1.)
    xlon, ylat = np.meshgrid(real_lon, real_lat)
    c = ax.pcolormesh(xlon, ylat, cf_var, cmap=bwr_zero,vmin=-6., vmax=5., zorder=1)  # latlon=True, transform=ccrs.PlateCarree(),
    cs = ax.contour(xlon, ylat, np.ma.masked_where(lsm.data == 1, c_var), latlon=True, colors='k', zorder=4)
    ax.clabel(cs, inline=True, fontsize = 24, inline_spacing = 30, fmt =  '%1.0f')
    coast = ax.contour(xlon, ylat, lsm.data, levels=[0], colors='#222222', lw=2, latlon=True, zorder=2)
    topog = ax.contour(xlon, ylat, orog.data, levels=[50], colors='#222222', linewidth=1.5, latlon=True, zorder=3)
    CBarXTicks = [-20, -10, -5,  0,  5,  10]  # CLevs[np.arange(0,len(CLevs),int(np.ceil(len(CLevs)/5.)))]
    CBAxes = fig.add_axes([0.25, 0.15, 0.6, 0.03])
    CBar = plt.colorbar(c, cax=CBAxes, orientation='horizontal', ticks=CBarXTicks, extend = 'both')  #
    CBar.set_label('1.5 m air temperature ($^{\circ}$C)', fontsize=34, labelpad=10, color='dimgrey')
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
    plt.subplots_adjust(left = 0.2, bottom = 0.3, right = 0.85)
    plt.savefig('/gws/nopw/j04/bas_climate/users/ellgil82/hindcast/figures/composite_' + regime + '.png')
    plt.savefig('/gws/nopw/j04/bas_climate/users/ellgil82/hindcast/figures/composite_' + regime + '.eps')
    plt.show()

# Identify patterns
MSLP_full = iris.load_cube(filepath + '1998-2017_MSLP_daymn.nc')
v_full =  iris.load_cube(filepath + '1998-2017_v_10m_daymn.nc')
v_full = v_full[:,:,1:,:]
T_full = iris.load_cube(filepath + '1998-2017_Tair_1p5m_daymn.nc')
u_full =  iris.load_cube(filepath + '1998-2017_u_10m_daymn.nc')
u_Plev = iris.load_cube(filepath + '1998-2017_u_wind_P_levs_daymn.nc')
SIC_full = iris.load_cube(filepath + '1998-2017_SIC_daymn.nc')
SAM_full = pd.read_csv(filepath + 'Daily_mean_SAM_index_1998-2017.csv', usecols = ['SAM'], dtype = np.float64, header = 0, na_values = '*******')
SAM_full.index = pd.date_range('1998-01-01', '2017-12-31', freq = 'D')
ENSO_full = iris.load_cube(filepath + 'inino34_daily.nc')
ENSO_full = pd.DataFrame(data = ENSO_full[6117:-731].data) # subset to 3 months before 1998-01-01 to 2017-12-31
ENSO_full = ENSO_full.rolling(window = 90).mean() # first 90 days will be nans
ENSO_full = ENSO_full[90:].values
ENSO_full = ENSO_full[:,0]

full_srs = {'MSLP': MSLP_full,
            'v': v_full,
            'u': u_full,
            'T': T_full,
            'u_Plev': u_Plev,
            'SIC': SIC_full,
            'SAM': SAM_full,
            'ENSO': ENSO_full}

try:
    for i in ['MSLP', 'T', 'u', 'v', 'SIC']:
        real_lon, real_lat = rotate_data(full_srs[i], np.ndim(full_srs[i])-2, np.ndim(full_srs[i])-1)
        #real_lon, real_lat = rotate_data(clim_srs[i], np.ndim(clim_srs[i]) - 2, np.ndim(clim_srs[i]) - 1)
        #clim_srs[i].convert_units(unit_dict[i])
        full_srs[i].convert_units(unit_dict[i])
except:
    for i in ['T', 'u', 'v', 'SIC']:
        real_lon, real_lat = rotate_data(full_srs[i], np.ndim(full_srs[i])-2, np.ndim(full_srs[i])-1)
        #real_lon, real_lat = rotate_data(clim_srs[i], np.ndim(clim_srs[i]) - 2, np.ndim(clim_srs[i]) - 1)
        #clim_srs[i].convert_units(unit_dict[i])
        full_srs[i].convert_units(unit_dict[i])


# Find anomalies of full series
MSLP_anom = full_srs['MSLP'][:7305, 0, :,:].data - Pclim_tot[:,0,:,:].data
v_anom = full_srs['v'][:7305,0,:,:].data - vclim_tot[:,0,:,:].data
u_anom = full_srs['u'][:7305,0,:,:].data - uclim_tot[:,0,:,:].data
T_anom = full_srs['T'][:7305,0,:,:].data - Tclim_tot[:,0,:,:].data
SIC_anom =full_srs['SIC'][:7305,0,:,:].data - SICclim_tot[:,0,:,:].data

anom_srs = {'MSLP': MSLP_anom,
            'v': v_anom,
            'u': u_anom,
            'T': T_anom,
            'SIC': SIC_anom}

def apply_composite_mask(regime, var):
    regime_mask = np.zeros(var.shape[0])
    if regime == 'barrier':
        indicator_var = v_full[:,0,:,:]
        threshold = 5.
        region = ((4,110), (150,-4))
    if regime == 'cross-peninsula':
        indicator_var = u_Plev[:,3,:,:]
        threshold = 2.
        region = ((75,175), (4,42))#up to 75:175 (y), 4:42 (x)
    #if regime == 'ASL':
        # will we see the effects of this in the domain?
    if regime == 'SIC_Weddell_L':
        indicator_var = SIC_full[:,0,:,:]
        threshold = 0.5
        region = ((4, 150),(150, -4)) # Weddell Sea region = x: 150:-4, y: 4:150
    if regime == 'SAM+':
        indicator_var = SAM_full['SAM'].values
        threshold = 1.36 # plus one standard deviation
    if regime == 'SAM-':
        indicator_var = SAM_full['SAM'].values
        threshold = -1.36 # minus one standard deviation
    if regime == 'ENSO+':
        indicator_var = ENSO_full
        threshold = 0.5
    if regime == 'ENSO-':
        indicator_var = ENSO_full
        threshold = -0.5
    for each_day in range(var.shape[0]):
        if regime == 'SIC_Weddell_L':
            if np.mean(indicator_var[each_day, region[0][0]:region[0][1], region[1][0]:region[1][1]].data) <= threshold:
                regime_mask[each_day] = 1.
            else:
                regime_mask[each_day] = 0.
        elif regime == 'SAM+' or regime == 'ENSO+':
            regime_mask[indicator_var < threshold] = 0.
            regime_mask[indicator_var > threshold] = 1.
        elif regime == 'SAM-' or regime == 'ENSO-':
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

regime = ''

for regime in  [ 'ENSO+', 'ENSO-']:#'cross-peninsula','barrier',  'SAM+', 'SAM-',
    c_var, regime_mask  = apply_composite_mask(regime, full_srs['MSLP'][:-1,0,:,:].data)
    cf_var, regime_mask  = apply_composite_mask(regime, T_anom) # try this with T anomalies instead
    u_var, regime_mask  = apply_composite_mask(regime, full_srs['u'][:-1,0,:,:].data)
    v_var, regime_mask  = apply_composite_mask(regime, full_srs['v'][:-1,0,:,:].data)
    plot_synop_composite(cf_var, c_var, u_var, v_var, regime)


#for regime in ['SIC_Weddell_L', 'barrier', 'cross-peninsula', 'SAM+', 'SAM-', 'ENSO+', 'ENSO-']:
#    c_var, regime_mask  = apply_composite_mask(regime, MSLP_full[:-1,0,:,:].data)
#    cf_var, regime_mask  = apply_composite_mask(regime, T_anom) # try this with T anomalies instead
#    u_var, regime_mask  = apply_composite_mask(regime, u_full[:-1,0,:,:].data)
#    v_var, regime_mask  = apply_composite_mask(regime, v_full[:-1,0,1:,:].data)
#    plot_synop_composite(cf_var, c_var, u_var, v_var, regime)



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

def apply_Larsen_mask(var):
    # Make ice shelf mask
    Larsen_mask = np.zeros((220, 220))
    lsm_subset = all_vars['lsm'].data[:150, 90:160]
    Larsen_mask[:150, 90:160] = lsm_subset
    Larsen_mask[all_vars['orog'].data > 100] = 0
    Larsen_mask = np.logical_not(Larsen_mask)
    # Apply mask to variable requested
    var_masked = np.ma.masked_array(var, mask=np.broadcast_to(Larsen_mask, var.shape)).mean(axis=(1, 2))
    return var_masked, Larsen_mask


def plot_melt():
    # plot whole ice shelf integrated mean melt series
    # Create Larsen mask

    ice_shelf_integrated_srs, Larsen_mask = apply_Larsen_mask(melt_amnt)

    # plot "inlet" mean
    # plot "ice shelf" mean
    # also plot map of these two sub-regions (i.e. mask the ice shelf out, then cut it down the middle vertically)