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
elif host == 'pc':
    sys.path.append('C:/Users/Ella/OneDrive - University of Reading/Scripts/Tools/')

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
elif host == 'pc':
    filepath = '/Users/Ella/Downloads/'

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
             'foehn_cond': '1',
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
        try:
            i.convert_units(unit_dict[var_name])
        except iris.exceptions.UnitConversionError:
            i.units = unit_dict[var_name]
    if var_name == 'v_10m' or var_name == 'FF_10m' or var_name == 'u_prof' or var_name == 'u_Plev':
        var = var[:,:,1:,:]
    return var #var[start_idx[0][0]:end_idx[0][0],0,:,:], clim[start_idx[0][0]:end_idx[0][0],0,:,:], anom

def create_clim(var_name, seas): # seas should be in the format '???_'
    import calendar
    # Load climatology
    clim = iris.load_cube(filepath + seas + var_name + '_climatology.nc')
    for i in [clim]:
        try:
            i.convert_units(unit_dict[var_name])
        except iris.exceptions.UnitConversionError:
            i.units = unit_dict[var_name]
    if seas == 'DJF_':
        clim = clim[:90]
        clim_total = clim.data[:90]
    else:
        clim = clim[:365]
        clim_total = clim.data[:365]
    for i in range(1998,2017):
        if calendar.isleap(i) == True: # if leap year, repeat final day of climatology
            #print(str(i) + ' is leap')
            clim_total = np.concatenate((clim_total, clim.data), axis = 0)
            clim_total = np.concatenate((clim_total, clim.data[-1:]), axis = 0)
        else:
            clim_total = np.concatenate((clim_total, clim.data), axis = 0 )
    if var_name == 'v_10m' or var_name == 'FF_10m' or var_name == 'u_prof' or var_name == 'u_Plev':
        clim_total = clim_total[:,:,1:,:]
    if var_name == 'sensible_heat' or var_name == 'latent_heat':
        clim_total = clim_total * -1.
    return iris.cube.Cube(clim_total)

try:
    LSM = iris.load_cube(filepath+'new_mask.nc')
    orog = iris.load_cube(filepath+'orog.nc')
    orog = orog[0,0,:,:]
    lsm = LSM[0,0,:,:]
    for i in [orog, lsm]:
        real_lon, real_lat = rotate_data(i, np.ndim(i)-2, np.ndim(i)-1)
except OSError:
    LSM = iris.load_cube(filepath + 'MetUM_v11p1_Antarctic_Peninsula_4km_19980101-20171231_land_binary_mask.nc')
    orog = iris.load_cube(filepath + 'MetUM_v11p1_Antarctic_Peninsula_4km_19980101-20171231_surface_altitude.nc')
    orog = orog[0, 0, :, :]
    lsm = LSM[0, 0, :, :]
    for i in [orog, lsm]:
        real_lon, real_lat = rotate_data(i, np.ndim(i) - 2, np.ndim(i) - 1)
except iris.exceptions.ConstraintMismatchError:
    print('Files not found')

def plot_synop_composite(cf_var, c_var, u_var, v_var, regime, seas):
    fig = plt.figure(frameon=False, figsize=(5, 6))  # !!change figure dimensions when you have a larger model domain (10,13)
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
    bwr_zero = shiftedColorMap(cmap=matplotlib.cm.bwr, min_val=-3., max_val=10., name='bwr_zero', var=cf_var.data, start=0.15,stop=0.85) #-20, 5
    xlon, ylat = np.meshgrid(real_lon, real_lat)
    Larsen_box = np.zeros((220,220))
    Larsen_box[40:135, 90:155] = 1.
    c = ax.pcolormesh(xlon, ylat,  np.ma.masked_where(Larsen_box == 0, cf_var), cmap=bwr_zero, vmin=-3., vmax=10., zorder=1)  # latlon=True, transform=ccrs.PlateCarree(),
    cs = ax.contour(xlon, ylat, np.ma.masked_where(lsm.data == 1, c_var), latlon=True, colors='k', zorder=4)
    ax.clabel(cs, inline=True, fontsize = 20, inline_spacing = 30, fmt =  '%1.0f')
    coast = ax.contour(xlon, ylat, lsm.data, levels=[0], colors='#222222', lw=2, latlon=True, zorder=2)
    topog = ax.contour(xlon, ylat, orog.data, levels=[50], colors='#222222', linewidth=1.5, latlon=True, zorder=3)
    CBarXTicks = [-3, 0, 5, 10]#[-20, -10, 0,  5]  # CLevs[np.arange(0,len(CLevs),int(np.ceil(len(CLevs)/5.)))]
    CBAxes = fig.add_axes([0.22, 0.2, 0.6, 0.025])
    CBar = plt.colorbar(c, cax=CBAxes, orientation='horizontal', ticks=CBarXTicks, extend = 'both')  #
    CBar.set_label('Mean daily maximum 1.5 m \nair temperature anomaly ($^{\circ}$C)', fontsize=20, labelpad=10, color='dimgrey')
    CBar.solids.set_edgecolor("face")
    CBar.outline.set_edgecolor('dimgrey')
    CBar.ax.tick_params(which='both', axis='both', labelsize=24, labelcolor='dimgrey', pad=10, size=0, tick1On=False,
                        tick2On=False)
    CBar.outline.set_linewidth(2)
    x, y = np.meshgrid(real_lon, real_lat)
    q = ax.quiver(x[::20, ::20], y[::20, ::20], u_var[::20, ::20], v_var[::20, ::20], pivot='middle', scale=100,
                  zorder=5)
    plt.quiverkey(q, 0.25, 0.9, 10, r'$10$ $m$ $s^{-1}$', labelpos='N', color='dimgrey', labelcolor='dimgrey',
                  fontproperties={'size': '20', 'weight': 'bold'},
                  coordinates='figure', )
    plt.subplots_adjust(left = 0.2, bottom = 0.33, top = 0.85, right = 0.85)
    if host == 'jasmin':
        plt.savefig('/gws/nopw/j04/bas_climate/users/ellgil82/hindcast/figures/composite_' + regime + '_' + seas + '.png')
        plt.savefig('/gws/nopw/j04/bas_climate/users/ellgil82/hindcast/figures/composite_' + regime + '_' + seas + '.eps')
    elif host == 'bsl':
        plt.savefig('/users/ellgil82/figures/Hindcast/Drivers/Seas/composite_' + regime + '_' + seas + '.png')
        plt.savefig('/users/ellgil82/figures/Hindcast/Drivers/Seas/composite_' + regime + '_' + seas + '.eps')
    elif host == 'pc':
        plt.savefig(filepath + '../OneDrive - University of Reading/Hindcast paper/composite_' + regime + '_' + seas + '.eps',
            transparent=True)
        plt.savefig(filepath + '../OneDrive - University of Reading/Hindcast paper/composite_' + regime + '_' + seas + '.png',
            transparent=True)
    plt.show()

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
        return np.mean(cube[50:,0,:,:].data, axis = 0), clim[50:,0,:,:], np.mean(anom[50:,0,:,:], axis=0)


seas_lens = {'DJF': 1805,
            'MAM': 1840,
             'JJA': 1840,
             'SON': 1820,
             'ANN': 7305}

MAM_dict = {}
MAM_clim = {}
MAM_anom = {}

#for i in ['Ts', 'land_snow_melt_flux', 'Tair_1p5m_daymax', 'MSLP', 'u_10m', 'v_10m',  'surface_SW_down',  'surface_SW_net',
#          'surface_LW_net', 'surface_LW_down', 'latent_heat', 'sensible_heat', 'E_tot']:#
#    MAM_dict[i] = load_vars('1998-2017', i, seas='MAM_')
#    MAM_clim[i] = create_clim(i, seas='MAM_')
#    MAM_anom[i] = iris.cube.Cube(data=(MAM_dict[i][:seas_lens['MAM']].data - MAM_clim[i][:seas_lens['MAM']].data))

# Extract only MAM 2016, from day 50 onwards
#for v in MAM_dict.keys():
#    MAM_dict[v] = MAM_dict[v][1701:1748,0,:,:] # April-May 2016
#    MAM_anom[v] = MAM_anom[v][1701:1748,0,:,:]
#    MAM_clim[v] = MAM_clim[v][1701:1748,0,:,:]

def MAM_SEB():
    fig, ax = plt.subplots( figsize=(16,9))
    colour_dict = {'SWdown': ('#6fb0d2', 'o'), 'surface_SW_net': ('#b733ff', 'o'), 'SWup': ('#6fb0d2', 'o'),
                   'LWdown': ('#86ad63','X'), 'surface_LW_net': ('#de2d26','X'), 'LWup': ('#86ad63','X'),
                   'latent_heat': ('#33aeff', '*'), 'sensible_heat': ('#ff6500', '^'), 'E_tot': ('#222222', 's'),
                   'land_snow_melt_flux': ('#f68080', 'P')}
    leg_dict = {'surface_SW_net': 'SW$_{net}$', 'surface_LW_net': 'LW$_{net}$', 'latent_heat': 'H$_{L}$',
                'sensible_heat': 'H$_{S}$', 'E_tot': 'E$_{tot}$', 'land_snow_melt_flux': 'E$_{melt}$'}
    for k in ['surface_SW_net', 'surface_LW_net', 'latent_heat', 'sensible_heat', 'E_tot', 'land_snow_melt_flux']:
        plt.plot(MAM_dict[k][1701:1748,0,40:135, 90:155].data.mean(axis=(1,2)), color = colour_dict[k][0], markersize=10, label=leg_dict[k], lw=4, marker=colour_dict[k][1])
    ax.set_xlim(0, 46)
    ax.set_ylim(-75, 120)
    ax.set_xticks(ticks=[0, 15, 30, 46])
    ax.set_xticklabels(['15 Apr', '1 May', '15 May', '31 May'])
    ax.spines['top'].set_visible(False)
    plt.setp(ax.spines.values(), linewidth=2, color='dimgrey', )
    ax.tick_params(axis='both', which='major', length=8, width=2, color='dimgrey', labelsize=24, tick1On=True,
                   tick2On=False, labelcolor='dimgrey', pad=10)
    ax.tick_params(axis='both', which='minor', length=0, width=0, color='dimgrey', labelsize=24, tick1On=True,
                   tick2On=False, labelcolor='dimgrey', pad=10)
    ax.set_ylabel('Energy flux\n(W m$^{-2}$)', fontsize=24, color='dimgrey', rotation=0)
    ax.yaxis.set_label_coords(-0.15, .5)
    foehn_crit = np.zeros(46)
    idx = np.where((MAM_dict['Tair_1p5m_daymax'][1701:1747, 0, 40:135, 90:155].data.max(axis = (1,2)) >= 0) & (MAM_dict['u_10m'][1701:1747, 0, 40:135, 90:155].data.mean(axis = (1,2)) > 5.))
    foehn_crit[idx] = 1.
    ax.fill_between(x=range(46), y1=-75, y2=120, where=foehn_crit >= 1, facecolor ='lightgrey', zorder = 1)
    ax.vlines(x = 11, ymin=-75, ymax=120, colors = 'lightgrey', zorder = 1, lw=20)
    # Legend
    lgd = ax.legend(bbox_to_anchor=(0.1, 1), markerscale=2, ncol=2, loc=2, fontsize=20)
    frame = lgd.get_frame()
    frame.set_facecolor('white')
    for ln in lgd.get_texts():
        plt.setp(ln, color='dimgrey')
    for line in lgd.get_lines():
        line.set_linewidth(4.0)
    lgd.get_frame().set_linewidth(0.0)
    plt.subplots_adjust(wspace=0.15, hspace=0.05, top=0.92, right=0.9, left=0.18, bottom=0.08)
    plt.savefig('/gws/nopw/j04/bas_climate/users/ellgil82/hindcast/figures/MAM_2016_SEB.png')
    plt.show()

#MAM_SEB()
#plt.plot(MAM_dict['Tair_1p5m_daymax'][1701:1747, 0, 40:135, 90:155].data.mean(axis = (1,2)))

#Plot met conditions
#plot_synop_composite(np.mean(MAM_anom['Tair_1p5m_daymax'].data, axis = 0), np.mean(MAM_dict['MSLP'].data, axis = 0),
#                     np.mean(MAM_dict['u_10m'].data, axis = 0), np.mean(MAM_dict['v_10m'].data, axis=0),
#                     regime = 'MAM_2016_mean_cond_Tmax_anom_Apr_onwards', seas = '')

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

def apply_composite_mask(regime, var, var_dict, anoms):
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
        region = ((75, 175), (4, 42))
        threshold = 0.5 # Froude number threshold
    if regime == 'blocked':
        indicator_var = var_dict['Fr']
        region = ((75, 175), (4, 42))
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
        indicator_var1 = var_dict['ASL_P'] #anoms['MSLP'][:, 0, :, :].data
        indicator_var2 = var_dict['ASL_lat']
        threshold1 = -10.1666259765625 # -5
        threshold2 = -70.0
        #region = ((4,110), (4, 75)) # ASL region = x: 4:75, y: 4:110
    if regime == 'SAM+':
        indicator_var = var_dict['SAM']
        threshold = 1.36 # plus one standard deviation
    if regime == 'SAM-':
        indicator_var = var_dict['SAM']
        threshold = -1.36 # minus one standard deviation
    if regime == 'ENSO-': # El Nino = warm Pacific SSTs, = SOI -v
        indicator_var = var_dict['ENSO']
        threshold = 0.5
    if regime == 'ENSO+': # La Nina = cool Pacific SSTs = SOI +v
        indicator_var = var_dict['ENSO']
        threshold = -0.5
    if regime == 'foehn':
        indicator_var = var_dict['foehn']
        threshold = 6.0
    if regime == 'cloudy':
        indicator_var = var_dict['mean_cloud']
        threshold = 0.75
    if regime == 'clear':
        indicator_var = var_dict['mean_cloud']
        threshold = 0.31
    if regime == 'sunny':
        indicator_var = np.copy(var_dict['Larsen_SW'])
        threshold = np.percentile(indicator_var, q = 75)
    if regime == 'sunny_non_foehn':
        indicator_var1 = np.copy(var_dict['Larsen_SW'])
        indicator_var2 = var_dict['foehn']
        threshold1 = np.percentile(indicator_var1, q = 75)
        threshold2 = 2.
    if regime == 'sunny_foehn':
        indicator_var1 = np.copy(var_dict['Larsen_SW'])
        indicator_var2 = var_dict['foehn']
        threshold1 = np.percentile(indicator_var1, q = 75)
        threshold2 = 6.
    if regime == 'non_sunny_foehn':
        indicator_var1 = np.copy(var_dict['Larsen_SW'])
        indicator_var2 = var_dict['foehn']
        threshold1 = np.percentile(indicator_var1, q = 25)
        threshold2 = 6.
    if regime == 'melt75':
        indicator_var = np.copy(var_dict['Larsen_melt'])
        indicator_var[indicator_var == 0.0] = np.nan
        threshold = np.nanpercentile(indicator_var, q = 75)
    if regime == 'melt25':
        indicator_var = np.copy(var_dict['Larsen_melt'])
        indicator_var[indicator_var == 0.0] = np.nan
        threshold = np.nanpercentile(indicator_var, q = 25)
    if regime == 'LWP75':
        indicator_var = var_dict['Larsen_LWP']
        threshold = np.nanpercentile(indicator_var, q = 75)
    if regime == 'LWP25':
        indicator_var = var_dict['Larsen_LWP']
        threshold = np.nanpercentile(indicator_var, q = 25)
    for each_day in range(var.shape[0]):
        if regime == 'SIC_Weddell_L':
            if np.mean(indicator_var[each_day, region[0][0]:region[0][1], region[1][0]:region[1][1]].data) <= threshold:
                regime_mask[each_day] = 1.
            else:
                regime_mask[each_day] = 0.
        elif regime == 'SIC_Weddell_H':
            if np.mean(indicator_var[each_day, region[0][0]:region[0][1], region[1][0]:region[1][1]].data) >= threshold:
                regime_mask[each_day] = 1.
            else:
                regime_mask[each_day] = 0.
        elif regime == 'SAM+' or regime == 'ENSO-' or regime == 'flow-over' or regime == 'cloudy' or regime == 'sunny' or regime == 'melt75' or regime == 'LWP75' or regime == 'foehn':
            regime_mask[indicator_var < threshold] = 0.
            regime_mask[indicator_var > threshold] = 1.
        elif regime == 'SAM-' or regime == 'ENSO+' or regime == 'blocked' or regime == 'clear' or regime == 'melt25' or regime == 'LWP25':
            regime_mask[indicator_var < threshold] = 1.
            regime_mask[indicator_var > threshold] = 0.
        elif regime == 'sunny_foehn':
            regime_mask[indicator_var1 < threshold1] = 0.
            regime_mask[(indicator_var1 > threshold1) & (indicator_var2 > threshold2)] = 1.
        elif regime == 'sunny_non_foehn':
            regime_mask[indicator_var1 < threshold1] = 0.
            regime_mask[(indicator_var1 > threshold1) & (indicator_var2 < threshold2)] = 1.
        elif regime == 'non_sunny_foehn':
            regime_mask[indicator_var1 > threshold1] = 0.
            regime_mask[(indicator_var1 < threshold1) & (indicator_var2 > threshold2)] = 1.
        elif regime == 'ASL':
            regime_mask[indicator_var1 > threshold1] = 0.
            regime_mask[(indicator_var1 < threshold1) & (indicator_var2 > threshold2)] = 1.
        else:
            if np.mean(indicator_var[each_day, region[0][0]:region[0][1], region[1][0]:region[1][
                1]].data) >= threshold:  # if mean v wind > 2.0 m s-1 within barrier jet region (excluding model domain blending halo)
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

def plot_SEB_composites(var1, var2, var3, var4, var5, var6, regime, seas):
    fig, axs = plt.subplots(3,2, frameon=False, figsize=(11, 18))
    axs = axs.flatten()
    fig.patch.set_visible(False)
    var_list = [var1[:7305], var2[:7305], var3[:7305], var4[:7305], var5[:7305], var6[:7305]]
    for i, j in zip(axs, ['SW$_{net}$', 'LW$_{net}$', 'H$_{L}$','H$_{S}$',  'E$_{tot}$', 'E$_{melt}$',]):
        i.set_title(j, color = 'dimgrey', fontsize = 34, pad = 20)
        plt.axis = 'off'
    lims = {'SW$_{net}$': (-50, 50),
             'LW$_{net}$': (-50, 50),
            'H$_{L}$': (-50,50),
            'H$_{S}$': (-100, 100),
            'E$_{tot}$': (-50,50),
            'E$_{melt}$': (-25,25)
    }
    for ax in [axs[1], axs[3], axs[5]]:
        ax.yaxis.tick_right()
    for ax, cf_var, j in zip(axs, var_list, ['SW$_{net}$', 'LW$_{net}$', 'H$_{L}$','H$_{S}$',  'E$_{tot}$', 'E$_{melt}$',]):
        ax.axis = 'off'
        ax.tick_params(which='both', axis='both', labelsize=24, labelcolor='dimgrey', pad=10, size=0, tick1On=False, tick2On=False)
        PlotLonMin = np.min(real_lon[90:155])
        PlotLonMax = np.max(real_lon[90:155])
        PlotLatMin = np.min(real_lat[40:135])
        PlotLatMax = np.max(real_lat[40:135])
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
        xlon, ylat = np.meshgrid(real_lon[90:155], real_lat[40:135])
        #cf_var, regime_mask = apply_composite_mask(regime, var)
        Larsen_box = np.zeros((220, 220))
        Larsen_box[40:135, 90:155] = 1.
        #bwr_zero = shiftedColorMap(cmap=matplotlib.cm.bwr, min_val=lims[j][0], max_val=lims[j][1], name='bwr_zero', var=cf_var[40:135, 90:155], start=0.15, stop=0.85)
        c = ax.pcolormesh(xlon, ylat, np.ma.masked_where(lsm[40:135, 90:155].data==0, cf_var[40:135, 90:155]), cmap='bwr', vmin=lims[j][0], vmax=lims[j][1], zorder=1)  # latlon=True, transform=ccrs.PlateCarree(),
        plt.sca(ax)
        CBarXTicks = np.linspace(lims[j][0], lims[j][1], num = 3)
        CBar = plt.colorbar(c, orientation='horizontal', extend='both', ticks=CBarXTicks)
        CBar.solids.set_edgecolor("face")
        CBar.outline.set_edgecolor('dimgrey')
        CBar.ax.tick_params(which='both', axis='both', labelsize=24, labelcolor='dimgrey', pad=10, size=0, tick1On=False,tick2On=False)
        CBar.outline.set_linewidth(2)
        coast = ax.contour(xlon, ylat, lsm[40:135, 90:155].data, levels=[0], colors='#222222', lw=2, latlon=True, zorder=2)
        topog = ax.contour(xlon, ylat, orog[40:135, 90:155].data, levels=[50], colors='#222222', linewidth=1.5, latlon=True, zorder=3)
        topog = ax.contour(xlon, ylat, orog[40:135, 90:155].data, levels=[500, 1000, 1500, 2000], colors='#A4A3A3', linewidth=0.25,
                           latlon=True, zorder=3)
    plt.subplots_adjust(bottom = 0.06, top = 0.94, hspace = 0.3, wspace = 0.25, right = 0.87)
    plt.savefig('/gws/nopw/j04/bas_climate/users/ellgil82/hindcast/figures/SEB_composite_subplot_' + regime + '_' + seas + '.png')
    plt.savefig('/gws/nopw/j04/bas_climate/users/ellgil82/hindcast/figures/SEB_composite_subplot_' + regime + '_' + seas + '.eps')
    plt.show()

#plot_SEB_composites(np.mean(MAM_anom['surface_SW_net'].data, axis = 0), np.mean(MAM_anom['surface_LW_net'].data, axis = 0),
#                    np.mean(MAM_anom['latent_heat'].data, axis = 0),  np.mean(MAM_anom['sensible_heat'].data, axis = 0),
#                    np.mean(MAM_anom['E_tot'].data, axis = 0),  np.mean(MAM_anom['land_snow_melt_flux'].data, axis = 0),
#                    regime = 'MAM_16_mean_SEB_Apr_onwards', seas='')

print('Plotting some met composites for... \n\n')

#melt_an = load_vars('1998-2017', var_name = 'land_snow_melt_amnt', seas = '')
#melt_JJA = load_vars('1998-2017', var_name = 'land_snow_melt_amnt', seas = 'JJA_')
#melt_SON = load_vars('1998-2017', var_name = 'land_snow_melt_amnt', seas = 'SON_')
#melt_DJF = load_vars('1998-2017', var_name = 'land_snow_melt_amnt', seas = 'DJF_')
#melt_MAM = load_vars('1998-2017', var_name = 'land_snow_melt_amnt', seas = 'MAM_')

#melt_cumsum = np.cumsum(np.mean(melt_an[:, 0, 40:135, 90:155].data, axis=(1, 2)))


#(np.cumsum(np.mean(melt_JJA[:, 0, 40:135, 90:155].data, axis=(1, 2)))[-1]/melt_cumsum[-1])*100
#(np.cumsum(np.mean(melt_SON[:, 0, 40:135, 90:155].data, axis=(1, 2)))[-1]/melt_cumsum[-1])*100
#(np.cumsum(np.mean(melt_DJF[:, 0, 40:135, 90:155].data, axis=(1, 2)))[-1]/melt_cumsum[-1])*100
#(np.cumsum(np.mean(melt_MAM[:, 0, 40:135, 90:155].data, axis=(1, 2)))[-1]/melt_cumsum[-1])*100


def plot_melt(cf_var, regime, seas):
    fig = plt.figure(frameon=False, figsize=(10, 11))  # !!change figure dimensions when you have a larger model domain
    fig.patch.set_visible(False)
    ax = fig.add_subplot(111)#, projection=ccrs.PlateCarree())
    plt.axis = 'off'
    plt.setp(ax.spines.values(), linewidth=0, color='dimgrey')
    PlotLonMin = np.min(real_lon[90:155])
    PlotLonMax = np.max(real_lon[90:155])
    PlotLatMin = np.min(real_lat[40:135])
    PlotLatMax = np.max(real_lat[40:135])
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
    bwr_zero = shiftedColorMap(cmap=matplotlib.cm.bwr, min_val=-5., max_val=5., name='bwr_zero', var=cf_var[40:135, 90:155].data, start=0.15, stop=0.85) #-6, 3
    xlon, ylat = np.meshgrid(real_lon[90:155], real_lat[40:135])
    #Larsen_box = np.ones((95,65))#
    #Larsen_box[40:135, 90:155] = 1.
    c = ax.pcolormesh(xlon, ylat,  cf_var[40:135, 90:155], cmap = bwr_zero, vmin = -5., vmax = 5.)#np.ma.masked_where((Larsen_box == 0.),
    #shaded = np.zeros((220,220))
    #shaded[60:130,:] = 1.
    #ax.contour(xlon, ylat, shaded, levels= [1.], colors='dimgrey', linewidths = 4, latlon=True)
    #ax.text(0., 1.1, zorder=6, transform=ax.transAxes, s='b', fontsize=32, fontweight='bold', color='dimgrey')
    coast = ax.contour(xlon, ylat, lsm[40:135, 90:155].data, levels=[0], colors='#222222', lw=2, latlon=True, zorder=2)
    topog = ax.contour(xlon, ylat, orog[40:135, 90:155].data, levels=[50], colors='#222222', linewidth=1.5, latlon=True, zorder=3)
    CBarXTicks = [-5,   0, 5,]  # CLevs[np.arange(0,len(CLevs),int(np.ceil(len(CLevs)/5.)))]
    CBAxes = fig.add_axes([0.25, 0.15, 0.6, 0.03])
    CBar = plt.colorbar(c, cax=CBAxes, orientation='horizontal', ticks=CBarXTicks, extend = 'both')  #
    CBar.set_label('Mean E$_{melt}$ anomaly (W m$^{-2}$)', fontsize=34, labelpad=10, color='dimgrey')
    CBar.solids.set_edgecolor("face")
    CBar.outline.set_edgecolor('dimgrey')
    CBar.ax.tick_params(which='both', axis='both', labelsize=34, labelcolor='dimgrey', pad=10, size=0, tick1On=False, tick2On=False)
    CBar.outline.set_linewidth(2)
    plt.subplots_adjust(left = 0.2, bottom = 0.3, right = 0.85)
    if host == 'jasmin':
        plt.savefig('/gws/nopw/j04/bas_climate/users/ellgil82/hindcast/figures/Melt_composite_' + regime + '_' + seas + '.png', transparent = True)
        plt.savefig('/gws/nopw/j04/bas_climate/users/ellgil82/hindcast/figures/Melt_composite_' + regime + '_' + seas + '.eps', transparent = True)
    elif host =='pc':
        plt.savefig(filepath + '../OneDrive - University of Reading/Hindcast paper/Melt_composite_' + regime + '_' + seas + '.eps', transparent = True)
        plt.savefig(filepath + '../OneDrive - University of Reading/Hindcast paper/Melt_composite_' + regime + '_' + seas + '.png', transparent=True)
    plt.show()

#    masks[regime] = regime_mask

#plot_melt(MAM_16_Emeltanom, 'MAM_Emelt_anom_Apr_onwards' , '')

def melt_transect():
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.setp(ax.spines.values(), linewidth=3, color='dimgrey')
    ax.plot(np.max(MAM_16_Emelt[60:130,:],axis = 0), color = 'red', lw = 2.5)
    ax.set_ylabel('\n\n\n\nMax E$_{melt}$  \n(W m$^{-2}$)', fontname='SegoeUI semibold', color='dimgrey', rotation=0,
                  fontsize=20, labelpad=50)
    ax.yaxis.set_label_coords(-0.2,0.4)
    ax.set_xlim(0,220)
    ax.text(-0, 1.1, zorder=6, transform=ax.transAxes, s='a', fontsize=32, fontweight='bold',
            color='dimgrey')
    plt.tick_params(axis='x', which='both', labelbottom= False, tick1On=False, tick2On=False, labelcolor='dimgrey', pad=5)
    plt.tick_params(axis='y', which='both', labelsize=24, tick1On=False, tick2On=False, labelcolor='dimgrey', pad=5)
    plt.subplots_adjust(left = 0.2,  right = 0.85, top = 0.85)
    plt.savefig('/gws/nopw/j04/bas_climate/users/ellgil82/hindcast/figures/melt_transect_MAM_16_Apr_onwards.eps')
    plt.savefig('/gws/nopw/j04/bas_climate/users/ellgil82/hindcast/figures/melt_transect_MAM_16_Apr_onwards.png')
    plt.show()

#melt_transect()

#print(' Calculating melt during each regime')
#Larsen_melt = var_dict['land_snow_melt_amnt'].data[:7305,0,40:135, 90:155]
#totm = np.copy(Larsen_melt)
## calculate total melt in one gridbox
#melt_tot_per_gridbox = np.zeros((7305, 100,70))
#for t in range(7305):
#    totm[lsm[40:135, 90:155] == 0.] = np.nan
#    for i in range(100):
#        for j in range(70):
#            melt_tot_per_gridbox[t, i, j] = totm[t, i, j] * (4000 * 4000)  # total kg per gridbox
#
#melt_tot = np.nansum(np.ma.masked_greater(melt_tot_per_gridbox, 3.20000006e+25), axis = 0) # remove sea values
#melt_tot = np.nansum(melt_tot)/10**12 # integrated ice shelf melt amount (in Gt!)

def run_composites():
    melt_list = []
    regime_list = []
    for regime in ['blocked', 'flow-over', 'cloudy', 'clear', 'melt25', 'melt75', 'LWP25', 'LWP75', 'SAM+', 'SAM-', 'ENSO+', 'ENSO-', 'ASL', 'barrier', ]: # 'SIC_Weddell_L', 'SIC_Weddell_H',
        print('\n\nPlotting synoptic composites during ' + regime + '...\n\n')
        c_var, regime_mask = apply_composite_mask(regime, var_dict['MSLP'][:7305, 0, :, :].data)
        cf_var, regime_mask = apply_composite_mask(regime, anoms['Tair_1p5m_daymax'][:7305, 0, :, :].data)  # try this with Tmax anomalies instead
        u_var, regime_mask = apply_composite_mask(regime, var_dict['u_10m'][:7305, 0, :, :].data)
        v_var, regime_mask = apply_composite_mask(regime, var_dict['v_10m'][:7305, 0, :, :].data)
        plot_synop_composite(cf_var, c_var, u_var, v_var, regime, seas = 'ANN')
        print('\n\nPlotting SEB composites during ' + regime + '...\n\n')
        SW_masked, regime_mask = apply_composite_mask(regime, anoms['surface_SW_net'].data, var_dict, anoms)
        LW_masked, regime_mask = apply_composite_mask(regime, anoms['surface_LW_net'].data, var_dict, anoms)
        HL_masked, regime_mask = apply_composite_mask(regime, anoms['latent_heat'].data, var_dict, anoms)
        HS_masked, regime_mask = apply_composite_mask(regime, anoms['sensible_heat'].data, var_dict, anoms)
        Etot_masked, regime_mask = apply_composite_mask(regime, anoms['Etot'].data, var_dict, anoms)
        melt_masked, regime_mask = apply_composite_mask(regime, anoms['land_snow_melt_flux'][:7305,0,:,:].data, var_dict, anoms)
        plot_SEB_composites(SW_masked, LW_masked, HL_masked, HS_masked, Etot_masked, melt_masked, regime, seas = 'ANN')
        plot_melt(melt_masked, regime, seas = 'ANN')
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
    df.to_csv(filepath + 'regime_melt_freq_ANN.csv')
    print(regime_freq)
    print(regime)

#run_composites()

def plot_melt_composites(seas, anoms, var_dict):
    fig, axs = plt.subplots(4,2, frameon=False, figsize=(11, 18))
    axs = axs.flatten()
    fig.patch.set_visible(False)
    lab_dict = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g', 7: 'h'}
    reg_list = [ 'SAM-', 'SAM+', 'ENSO+', 'ENSO-', 'blocked', 'flow-over', 'barrier', 'ASL', ]
    for ax in [axs[1], axs[3], axs[5], axs[7]]:
        ax.yaxis.tick_right()
        ax.axis('off')
    for i in range(8):
        melt, reg_mask = apply_composite_mask(reg_list[i], anoms['land_snow_melt_flux'].data[:seas_lens[seas]-1,0,:,:], var_dict, anoms)
        ax = axs[i]
        ax.text(0., 1.1, zorder=6, transform=ax.transAxes, s=lab_dict[i], fontsize=32, fontweight='bold', color='dimgrey')
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
        YTicks = np.linspace(PlotLatMin, PlotLatMax, 3)
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
        Larsen_box = np.zeros((220,220))
        Larsen_box[40:135, 90:155] = 1.
        c = ax.pcolormesh(xlon, ylat, np.ma.masked_where((Larsen_box ==0.), melt), vmin = -3, vmax = 3, cmap = 'bwr')#, cmap=bwr_zero, vmin=-6., vmax=3., zorder=1)  # latlon=True, transform=ccrs.PlateCarree(),
        coast = ax.contour(xlon, ylat, lsm.data, levels=[0], colors='#222222', lw=2, latlon=True, zorder=2)
        topog = ax.contour(xlon, ylat, orog.data, levels=[50], colors='#222222', linewidth=1.5, latlon=True, zorder=3)
    CBarXTicks = [-3, 0, 3]  # CLevs[np.arange(0,len(CLevs),int(np.ceil(len(CLevs)/5.)))]
    CBAxes = fig.add_axes([0.25, 0.15, 0.55, 0.02])
    CBar = plt.colorbar(c, cax=CBAxes, orientation='horizontal', extend='both', ticks=CBarXTicks)
    CBar.set_label('Mean E$_{melt}$ anomaly (W m$^{-2}$)', fontsize=34, labelpad=10, color='dimgrey')
    CBar.solids.set_edgecolor("face")
    CBar.outline.set_edgecolor('dimgrey')
    CBar.ax.tick_params(which='both', axis='both', labelsize=34, labelcolor='dimgrey', pad=10, size=0, tick1On=False, tick2On=False)
    CBar.outline.set_linewidth(2)
    plt.subplots_adjust(bottom = 0.22, top = 0.9, hspace = 0.3, wspace = 0.25, right = 0.87)
    plt.savefig('/gws/nopw/j04/bas_climate/users/ellgil82/hindcast/figures/Melt_flux_anomalies_all_regimes_' + seas + '.png')
    plt.savefig('/gws/nopw/j04/bas_climate/users/ellgil82/hindcast/figures/Melt_flux_anomalies_all_regimes_' + seas + '.eps')
    #plt.show()

#plot_melt_composites(seas = 'MAM', anoms=anoms, var_dict = var_dict)

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
        var_masked = var[:,0,40:135, 90:155].data
        clim_masked= clim[:,0,40:135, 90:155].data
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
    months = [g for n, g in var_df.groupby(pd.Grouper('M'))]
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
    months = [g for n, g in anom_df.groupby(pd.Grouper('M'))]
    for yr in range(20):
        anom_DJF = pd.concat((months[dec[yr]], months[jan[yr]], months[feb[yr]]))
        anom_MAM = pd.concat((months[mar[yr]], months[apr[yr]], months[may[yr]]))
        anom_JJA = pd.concat((months[jun[yr]], months[jul[yr]], months[aug[yr]]))
        anom_SON = pd.concat((months[sep[yr]], months[oct[yr]], months[nov[yr]]))
    months = [g for n, g in clim_df.groupby(pd.Grouper('M'))]
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

seas_lens = {'DJF': 1805,
             'MAM': 1840,
             'JJA': 1840,
             'SON': 1820,
             'ANN': 7305}

def grouper(iterable, seas):
    if seas == 'DJF':
        chunks = np.split(iterable, [90, 180, 271, 361, 451, 541, 632, 722, 812, 902, 993, 1083, 1173, 1263, 1354, 1444, 1534, 1624, 1715], axis = 0)
    else:
        chunks = np.split(iterable, range(0, seas_lens[seas], chunk_lens[seas]), axis = 0)
    return chunks

chunk_lens = {'SON': 91,
              'MAM': 92,
              'JJA': 92,
              'DJF': 90}

def seas_composite():
    for seas in ['MAM','JJA','SON','DJF']:#, 'JJA']:
        var_dict = {}
        clims = {}
        anoms = {}
        seas_lens = {'DJF': 1805,
                     'MAM': 1840,
                     'JJA': 1840,
                         'SON': 1820}
        for i in ['land_snow_melt_flux',  'Tair_1p5m_daymax', 'MSLP', 'u_10m', 'v_10m', 'land_snow_melt_amnt', 'surface_SW_down','cl_frac', 'total_column_liquid', ]:#  'land_snow_melt_amnt','land_snow_melt_flux', 'surface_SW_down']:#, 'SIC', 'surface_SW_net', 'surface_LW_net', 'latent_heat', 'sensible_heat', 'E_tot']:#
            print('\nLoading ' + seas + ' ' +  i)
            var_dict[i] = load_vars('1998-2017', i, seas = seas + '_')
            clims[i] = create_clim(i, seas = seas + '_')
            anoms[i] = iris.cube.Cube(data=(var_dict[i][:seas_lens[seas]].data - clims[i][:seas_lens[seas]].data))
        #for i in ['u_wind_full_profile', 'u_wind_P_levs',]:
        #    print('\nLoading ' + seas + ' ' + i)
        #    var_dict[i] = load_vars('1998-2017', i, seas=seas + '_')
        # Normalise anomalies
        print('Normalising anomalies... \n\n')
        var_dict['Larsen_melt'] = np.mean(var_dict['land_snow_melt_amnt'][:seas_lens[seas]-1, 0, 40:135, 90:155].data, axis=(1, 2))
        var_dict['Larsen_SW']= np.mean(var_dict['surface_SW_down'][:seas_lens[seas]-1, 0, 40:135, 90:155].data, axis=(1, 2))
        print('Calculating Froude number... \n\n')
        #Fr, h_hat = Froude_number(np.mean(var_dict['u_wind_full_profile'].data[:seas_lens[seas]-1, 4:23, 75:175, 4:42], axis=(1, 2, 3)))
        #Fr[np.mean(var_dict['u_wind_full_profile'].data[:seas_lens[seas]-1, 4:23, 75:175, 4:42], axis=(1, 2, 3)) < 2.0] = np.nan
        #var_dict['Fr'] = Fr
        var_dict['Larsen_LWP'] = np.mean(var_dict['total_column_liquid'][:seas_lens[seas]-1, 0, 40:135, 90:155].data, axis = (1,2))
        var_dict['mean_cloud'] = np.mean(var_dict['cl_frac'][:seas_lens[seas] - 1, 0, 40:135, 90:155].data, axis=(1, 2))
        foehn_df = pd.read_csv(filepath + 'daily_foehn_frequency_all_stations.csv') # turn this into AWS 14/15/18 average, then diagnose when foehn is shown at one or more
        foehn_sum = foehn_df['sum_foehn']
        foehn_sum.index = pd.date_range('1998-01-01', '2017-12-31', freq='D')
        #var_dict['foehn'] = np.mean(var_dict['foehn_cond'][:seas_lens[seas] - 1, 0, 40:135, 90:155].data, axis=(1, 2))
        # Calculate months
        SAM_full = pd.read_csv(filepath + 'Daily_mean_SAM_index_1998-2017.csv', usecols=['SAM'], dtype=np.float64, header=0, na_values='*******')
        SAM_full.index = pd.date_range('1998-01-01', '2017-12-31', freq='D')
        months = [g for n, g in SAM_full.groupby(pd.Grouper(freq = 'M'))]
        ENSO_full = iris.load_cube(filepath + 'inino34_daily.nc')
        ENSO_full = pd.DataFrame(data=ENSO_full[6117:13512].data)  # subset to 3 months before 1998-01-01 to 2017-12-31
        ENSO_full = ENSO_full.rolling(window=90).mean()  # first 90 days will be nans
        ENSO_full = ENSO_full[90:]  # .values
        ENSO_full.index = pd.date_range('1998-01-01', '2017-12-31', freq='D')
        ASL_full = pd.read_csv(filepath + 'ASL_index_daily.csv', header=0, usecols=['RelCenPres', 'lat'])
        ASL_full.index=pd.date_range('1998-01-01', '2017-12-31', freq='D')
        month_ASL_P = [g for n, g in ASL_full['RelCenPres'].groupby(pd.Grouper(freq = 'M'))]
        month_ASL_lat = [g for n, g in ASL_full['lat'].groupby(pd.Grouper(freq='M'))]
        month_ENSO = [g for n, g in ENSO_full.groupby(pd.Grouper(freq = 'M'))]
        month_foehn = [g for n, g in foehn_sum.groupby(pd.Grouper(freq = 'M'))]
        ENSO_full = ENSO_full[:7305]#, 0]
        ENSO_seas = pd.Series()
        SAM_seas = pd.Series()
        foehn_seas = pd.Series()
        ASL_Pseas = pd.Series()
        ASL_latseas = pd.Series()
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
                foehn_seas = pd.concat((foehn_seas, month_foehn[dec[yr]], month_foehn[jan[yr]], month_foehn[feb[yr]]))
                ASL_Pseas = pd.concat((ASL_Pseas, month_ASL_P[dec[yr]], month_ASL_P[jan[yr]], month_ASL_P[feb[yr]]))
                ASL_latseas = pd.concat((ASL_latseas, month_ASL_lat[dec[yr]], month_ASL_lat[jan[yr]], month_ASL_lat[feb[yr]]))
            elif seas == 'MAM':
                SAM_seas = pd.concat((SAM_seas,months[mar[yr]], months[apr[yr]], months[may[yr]]))
                ENSO_seas = pd.concat((ENSO_seas, month_ENSO[mar[yr]], month_ENSO[apr[yr]], month_ENSO[may[yr]]))
                foehn_seas = pd.concat((foehn_seas, month_foehn[mar[yr]], month_foehn[apr[yr]], month_foehn[may[yr]]))
                ASL_Pseas = pd.concat((ASL_Pseas, month_ASL_P[mar[yr]], month_ASL_P[apr[yr]], month_ASL_P[may[yr]]))
                ASL_latseas = pd.concat((ASL_latseas, month_ASL_lat[mar[yr]], month_ASL_lat[apr[yr]], month_ASL_lat[may[yr]]))
            elif seas == 'JJA':
                SAM_seas = pd.concat((SAM_seas,months[jun[yr]], months[jul[yr]], months[aug[yr]]))
                ENSO_seas = pd.concat((ENSO_seas, month_ENSO[jun[yr]], month_ENSO[jul[yr]], month_ENSO[aug[yr]]))
                foehn_seas = pd.concat((foehn_seas, month_foehn[jun[yr]], month_foehn[jul[yr]], month_foehn[aug[yr]]))
                ASL_Pseas = pd.concat((ASL_Pseas, month_ASL_P[jun[yr]], month_ASL_P[jul[yr]], month_ASL_P[aug[yr]]))
                ASL_latseas = pd.concat((ASL_latseas, month_ASL_lat[jun[yr]], month_ASL_lat[jul[yr]], month_ASL_lat[aug[yr]]))
            elif seas == 'SON':
                SAM_seas = pd.concat((SAM_seas,months[sep[yr]], months[oct[yr]], months[nov[yr]]))
                ENSO_seas = pd.concat((ENSO_seas, month_ENSO[sep[yr]], month_ENSO[oct[yr]], month_ENSO[nov[yr]]))
                foehn_seas = pd.concat((foehn_seas, month_foehn[sep[yr]], month_foehn[oct[yr]], month_foehn[nov[yr]]))
                ASL_Pseas = pd.concat((ASL_Pseas, month_ASL_P[sep[yr]], month_ASL_P[oct[yr]], month_ASL_P[nov[yr]]))
                ASL_latseas = pd.concat( (ASL_latseas, month_ASL_lat[sep[yr]], month_ASL_lat[oct[yr]], month_ASL_lat[nov[yr]]))
        SAM_seas = SAM_seas.values[:seas_lens[seas]-1, 1]
        ENSO_seas = ENSO_seas.values[:seas_lens[seas]-1,0]
        foehn_seas = foehn_seas.values[:seas_lens[seas]-1]
        ASL_Pseas = ASL_Pseas.values[:seas_lens[seas]-1]
        ASL_latseas = ASL_latseas.values[:seas_lens[seas] - 1]
        var_dict['SAM'] = SAM_seas
        var_dict['ENSO'] = ENSO_seas
        var_dict['foehn'] = foehn_seas
        var_dict['ASL_P'] = ASL_Pseas
        var_dict['ASL_lat'] = ASL_latseas
        print(' Calculating melt during each regime')
        Larsen_melt = var_dict['land_snow_melt_amnt'].data[:seas_lens[seas]-1, 0, 40:135, 90:155]
        totm = np.copy(Larsen_melt)
        # calculate total melt in one gridbox
        melt_tot_per_gridbox = np.zeros(Larsen_melt.shape)
        for t in range(Larsen_melt.shape[0]):
            totm[lsm[40:135, 90:155] == 0.] = np.nan
            for i in range(95):
                for j in range(65):
                    melt_tot_per_gridbox[t, i, j] = totm[t, i, j] * (4000 * 4000)  # total kg per gridbox
        melt_tot = np.nansum(np.ma.masked_greater(melt_tot_per_gridbox, 3.20000006e+25), axis=0)  # remove sea values
        melt_tot = np.nansum(melt_tot) / 10 ** 12  # integrated ice shelf melt amount (in Gt!)
        melt_list = []
        regime_list = []
        melt_sd_list = []
        regime_sd_list = []
        regime_masks = pd.DataFrame()
        for regime in [ 'ASL','SAM+', 'SAM-', 'ENSO+', 'ENSO-',  'cloudy', 'sunny', 'foehn', 'sunny_foehn', 'sunny_non_foehn', 'non_sunny_foehn','barrier', 'melt25' ,  'melt75', 'cloudy', 'clear' , 'LWP25', 'LWP75' ]: #,  'melt25', 'LWP75', 'LWP25',,'melt75' , 'ENSO+', 'blocked', 'LWP25', 'LWP75','cloudy', 'clear' 'SIC_Weddell_L', 'SIC_Weddell_H',
            print('\n\nPlotting synoptic composites during ' + regime + '...\n\n')
            c_var, regime_mask = apply_composite_mask(regime, var_dict['MSLP'][:seas_lens[seas]-1, 0, :, :].data, var_dict, anoms)
            cf_var, regime_mask = apply_composite_mask(regime, anoms['Tair_1p5m_daymax'][:seas_lens[seas]-1, 0, :, :].data, var_dict, anoms)# try this with Tmax anomalies instead
            u_var, regime_mask = apply_composite_mask(regime, var_dict['u_10m'][:seas_lens[seas]-1, 0, :, :].data, var_dict, anoms)
            v_var, regime_mask = apply_composite_mask(regime, var_dict['v_10m'][:seas_lens[seas]-1, 0, :, :].data, var_dict, anoms)
            plot_synop_composite(cf_var, c_var, u_var, v_var, regime, seas = seas)
            print('\n\nPlotting SEB composites during ' + regime + '...\n\n')
            #SW_masked, regime_mask = apply_composite_mask(regime, anoms['surface_SW_net'].data[:seas_lens[seas]-1, 0, :, :], var_dict, anoms)
            #LW_masked, regime_mask = apply_composite_mask(regime, anoms['surface_LW_net'].data[:seas_lens[seas]-1, 0, :, :], var_dict, anoms)
            #HL_masked, regime_mask = apply_composite_mask(regime, anoms['latent_heat'].data[:seas_lens[seas]-1, 0, :, :], var_dict, anoms)
            #HS_masked, regime_mask = apply_composite_mask(regime, anoms['sensible_heat'].data[:seas_lens[seas]-1, 0, :, :], var_dict, anoms)
            #Etot_masked, regime_mask = apply_composite_mask(regime, anoms['E_tot'].data[:seas_lens[seas]-1, 0, :, :], var_dict, anoms)
            melt_masked, regime_mask = apply_composite_mask(regime, anoms['land_snow_melt_flux'].data[:seas_lens[seas]-1, 0, :, :], var_dict, anoms)
            regime_masks[regime] = regime_mask
            #plot_SEB_composites(SW_masked, LW_masked, HL_masked, HS_masked, Etot_masked, melt_masked, regime, seas = seas)
            print('\n\nPlotting melt...')
            plot_melt(melt_masked, regime, seas = seas)
            print('\n\nDoing some maths...')
            regime_freq = (np.float(np.count_nonzero(regime_mask)) / float(var_dict['v_10m'][:-1].shape[0])) * 100
            melt_regime = np.copy(melt_tot_per_gridbox[:seas_lens[seas]-1])  # copy total
            melt_regime[regime_mask == 0] = np.nan  # apply mask
            melt_tot_regime = np.nansum(melt_regime, axis=0)  # sum over time
            melt_tot_regime = np.ma.masked_greater(melt_tot_regime, 3e+20)  # mask sea values
            melt_tot_regime = np.nansum( melt_tot_regime) / 10 ** 12  # sum across entire ice shelf, and return in Gt meltwater
            melt_tot_regime_pct = (melt_tot_regime / melt_tot) * 100  # find as percentage of melt
            print(regime + ' associated with ' + str(melt_tot_regime_pct) + ' % of melting during ' + seas + ' in hindcast\n\n')
            print(regime + ' occurs ' + str(regime_freq) + ' % of the time during ' + seas + ' in hindcast\n\n')
            seas_chunk = grouper(regime_mask, seas)
            seas_melt_regime = np.ma.masked_greater(melt_regime, 3e+20)  # mask sea values
            seas_melt_regime = np.nansum(seas_melt_regime, axis=(1, 2)) / 10 ** 12
            # find annual melt percentage stats
            melt_chunk = grouper(seas_melt_regime, seas)
            seas_tot_melt = np.ma.masked_greater(melt_tot_per_gridbox, 3e+20)  # mask sea values
            seas_tot_melt = np.nansum(seas_tot_melt, axis=(1, 2)) / 10 ** 12
            tot_melt_chunk = grouper(seas_tot_melt, seas)
            regime_seas_melt_tots = []
            for c in melt_chunk[1:]:
                tot = np.sum(c)
                regime_seas_melt_tots = np.append(regime_seas_melt_tots, tot)
            seas_melt_tots = []
            for c in tot_melt_chunk[1:]:
                tot = np.sum(c)
                seas_melt_tots = np.append(seas_melt_tots, tot)
            seas_regime_freq = []
            for c in seas_chunk[1:]:
                freq = (np.float(np.count_nonzero(c)) / np.float(c.shape[0])) * 100
                seas_regime_freq = np.append(seas_regime_freq, freq)
            regime_seas_tot_melt_pct = (regime_seas_melt_tots / seas_melt_tots) * 100
            regime_seas_tot_melt_pct = np.nan_to_num(regime_seas_tot_melt_pct)
            melt_sd = np.std(regime_seas_tot_melt_pct)  # in %
            regime_sd = np.std(seas_regime_freq)  # in %
            melt_list.append(melt_tot_regime_pct)
            regime_list.append(regime_freq)
            regime_sd_list.append(regime_sd)
            melt_sd_list.append(melt_sd)
            print(melt_list)
            print(melt_sd)
            print(regime_freq)
            print(regime_sd)
        regime_masks.to_csv(filepath + 'Regime_masks_' + seas + '.csv')
        df = pd.DataFrame(index=[ 'ASL','SAM+', 'SAM-', 'ENSO+', 'ENSO-',  'cloudy', 'sunny', 'foehn', 'sunny_foehn', 'sunny_non_foehn', 'non_sunny_foehn','barrier',  'melt25', 'melt75', 'cloudy', 'clear', 'LWP25', 'LWP75' ])
        df['melt_pct'] = pd.Series(melt_list, index=[ 'ASL','SAM+', 'SAM-', 'ENSO+', 'ENSO-',  'cloudy', 'sunny', 'foehn', 'sunny_foehn', 'sunny_non_foehn', 'non_sunny_foehn','barrier',  'melt25', 'melt75', 'cloudy', 'clear', 'LWP25', 'LWP75'])
        df['regime_freq'] = pd.Series(regime_list, index=[ 'ASL','SAM+', 'SAM-', 'ENSO+', 'ENSO-',  'cloudy', 'sunny', 'foehn', 'sunny_foehn', 'sunny_non_foehn', 'non_sunny_foehn','barrier', 'melt25', 'melt75', 'cloudy', 'clear', 'LWP25', 'LWP75'])
        df['melt_pct_sd'] = pd.Series(melt_sd_list, index=['ASL','SAM+', 'SAM-', 'ENSO+', 'ENSO-',  'cloudy', 'sunny', 'foehn', 'sunny_foehn', 'sunny_non_foehn', 'non_sunny_foehn','barrier', 'melt25', 'melt75', 'cloudy', 'clear'  , 'LWP25', 'LWP75'])
        df['regime_freq_sd'] = pd.Series(regime_sd_list, index=[ 'ASL','SAM+', 'SAM-', 'ENSO+', 'ENSO-',  'cloudy', 'sunny', 'foehn', 'sunny_foehn', 'sunny_non_foehn', 'non_sunny_foehn','barrier', 'melt25', 'melt75', 'cloudy', 'clear' ,   'LWP25', 'LWP75'])
        df.to_csv(filepath + seas + '_regime_melt_freq.csv')
        #plot_melt_composites(seas, anoms = anoms, var_dict = var_dict)

seas_composite()

seas = 'JJA'
df_overlap = pd.read_csv(filepath + 'Regime_masks_' +  seas + '.csv', index_col = 0)
df_overlap['ENSO_foehn'] = df_overlap['ENSO-'] + df_overlap['foehn']
df_overlap['ENSO_foehn'][df_overlap['ENSO_foehn']<2] =  0.
df_overlap['ENSO_foehn'][df_overlap['ENSO_foehn']==2] = 1.
np.count_nonzero(df_overlap['ENSO_foehn']) / float(np.count_nonzero(df_overlap['foehn']))
np.count_nonzero(df_overlap['ENSO_foehn']) / float(np.count_nonzero(df_overlap['ENSO-']))
df_overlap['sunny_foehn'] = df_overlap['sunny'] + df_overlap['foehn']
df_overlap['sunny_foehn'][df_overlap['sunny_foehn']<2] =  0.
df_overlap['sunny_foehn'][df_overlap['sunny_foehn']==2] = 1.
np.count_nonzero(df_overlap['sunny_foehn']) / float(np.count_nonzero(df_overlap['sunny']))
df_overlap['foehn'].corr(df_overlap['sunny'])
df_overlap['sunny'].corr(df_overlap['melt75'])
df_overlap['foehn'].corr(df_overlap['melt75'])


df_overlap['sunny_melt'] = df_overlap['sunny'] + df_overlap['melt75']
df_overlap['sunny_melt'][df_overlap['sunny_melt']<2] =  0.
df_overlap['sunny_melt'][df_overlap['sunny_melt']==2] = 1.
np.count_nonzero(df_overlap['sunny_melt']) / float(np.count_nonzero(df_overlap['melt75']))
df_overlap['foehn_melt'] = df_overlap['foehn'] + df_overlap['melt75']
df_overlap['foehn_melt'][df_overlap['foehn_melt']<2] =  0.
df_overlap['foehn_melt'][df_overlap['foehn_melt']==2] = 1.
np.count_nonzero(df_overlap['foehn_melt']) / float(np.count_nonzero(df_overlap['melt75']))




df_overlap['foehn_clearance'] = df_overlap['clear'] + df_overlap['foehn']
df_overlap['foehn_clearance'][df_overlap['foehn_clearance']<2] =  0.
df_overlap['foehn_clearance'][df_overlap['foehn_clearance']==2] = 1.
np.count_nonzero(df_overlap['foehn_clearance']) / float(np.count_nonzero(df_overlap['foehn']))


df_overlap['SAM_ENSO'] = df_overlap['SAM+'] + df_overlap['ENSO-']
df_overlap['SAM_ENSO'][df_overlap['SAM_ENSO']<2] = 0.
df_overlap['SAM_ENSO'][df_overlap['SAM_ENSO']==2] = 1.
df_overlap['SAM_ENSO-'] = df_overlap['SAM-'] + df_overlap['ENSO+']
df_overlap['SAM_ENSO-'][df_overlap['SAM_ENSO-']<2] = 0.
df_overlap['SAM_ENSO-'][df_overlap['SAM_ENSO-']==2] = 1.
np.count_nonzero(df_overlap['SAM_ENSO'])/float(len(df_overlap)) + np.count_nonzero(df_overlap['SAM_ENSO-']) / float(len(df_overlap))

