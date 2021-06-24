# Define where the script is running
host = 'pc'

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
import cf_units
from divg_temp_colourmap import shiftedColorMap
import time
import copy
from sklearn.metrics import mean_squared_error
import datetime
import cftime
import metpy
import metpy.calc
import glob
from scipy import stats

# Set up filepath
if host == 'jasmin':
    filepath = '/gws/nopw/j04/bas_climate/users/ellgil82/hindcast/output/alloutput/'
elif host == 'bsl':
    filepath = '/data/mac/ellgil82/hindcast/output/'
elif host == 'pc':
    filepath = 'OneDrive - University of Reading/Hindcast paper/ERA5_monthly_'

LWnet = iris.load(filepath + '*.nc', 'surface_net_upward_longwave_flux')
SWnet = iris.load(filepath + '*.nc', 'surface_net_downward_shortwave_flux')
HL = iris.load(filepath + '*.nc', 'surface_upward_latent_heat_flux')
HS = iris.load(filepath + '*.nc', 'surface_upward_sensible_heat_flux')
melt_amnt = iris.load(filepath + '*.nc', 'Snowmelt')
albedo = iris.load(filepath + '*.nc', 'Snow albedo')
Ts = iris.load(filepath + '*.nc', 'Skin temperature')
lsm = iris.load_cube('Downloads/ERA5_orog_lsm.nc', 'land_binary_mask')
orog = iris.load_cube('Downloads/ERA5_orog_lsm.nc', 'geopotential')
orog = orog / 9.80665  # in m

means = {}
for cubelist, cubename in zip([LWnet, SWnet, HS, HL, melt_amnt, albedo, Ts],['LWnet', 'SWnet', 'HS', 'HL', 'melt_amnt', 'albedo', 'Ts']):
    means[cubename] = []
    for cube in cubelist:
        if cubelist == LWnet or cubelist == SWnet or cubelist == HS or cubelist == HL:
            cube = cube/86400
        lsm_3d = np.broadcast_to(lsm.data, cube.shape)
        cube_masked = np.ma.masked_where(lsm_3d < 0.9, cube.data)
        means[cubename] = cube_masked.mean()


def load_var(seas, time):
    # Load variables
    LWnet = iris.load_cube(filepath + seas + '_' + time + '.nc', 'surface_net_upward_longwave_flux')
    SWnet = iris.load_cube(filepath + seas + '_'+  time +'.nc', 'surface_net_downward_shortwave_flux')
    HL = iris.load_cube(filepath + seas + '_'+ time + '.nc', 'surface_upward_latent_heat_flux')
    HS = iris.load_cube(filepath + seas + '_'+ time + '.nc', 'surface_upward_sensible_heat_flux')
    melt_amnt = iris.load_cube(filepath + seas + '_'+ time + '.nc', 'Snowmelt')
    albedo = iris.load_cube(filepath + seas + '_'+  time +'.nc', 'Snow albedo')
    Ts = iris.load_cube(filepath + seas + '_'+ time + '.nc', 'Skin temperature')
    # Load ERA 5 data
    lsm = iris.load_cube('Downloads/ERA5_orog_lsm.nc', 'land_binary_mask')
    orog = iris.load_cube('Downloads/ERA5_orog_lsm.nc', 'geopotential')
    orog = orog / 9.80665  # in m
    # Create standardised time units
   # t = [cftime.datetime(0, 0, 0, 0), cftime.datetime(3, 0, 0, 0), cftime.datetime(6, 0, 0, 0), cftime.datetime(9, 0, 0, 0),
   #      cftime.datetime(12, 0, 0, 0), cftime.datetime(15, 0, 0, 0), cftime.datetime(18, 0, 0, 0), cftime.datetime(21, 0, 0, 0)]
   # t_num = [0,3,6,9,12,15,18,21]
   # new_time = iris.coords.AuxCoord(t, long_name='time', standard_name='time', units=cf_units.Unit('hours since 1970-01-01 00:00:00', calendar='standard'))
   # T_dim = iris.coords.DimCoord(t_num, long_name='time', standard_name='time', units=cf_units.Unit('hours since 1970-01-01 00:00:00', calendar='standard'))
    LWnet = LWnet / 86400
    SWnet = SWnet / 86400
    HL = HL / 86400
    HS = HS / 86400
    #Calculate Etot
    Etot = iris.cube.Cube(data = SWnet.data - LWnet.data - HL.data - HS.data,  long_name = 'Total energy flux', var_name = 'Etot', units = SWnet.units)
    #melt_flux = iris.cube.Cube(data=np.copy(Etot.data), long_name=' melt flux', var_name='Emelt', units=SWnet.units)
    #melt_flux.data[Ts.data <= 273] = 0.
    #for n in range(2):
    #    Etot.add_dim_coord(SWnet.dim_coords[n],n+1)
    #    melt_flux.add_dim_coord(SWnet.dim_coords[n],n+1)
    #Etot.add_aux_coord(SWnet.aux_coords[0], 0)
    #melt_flux.add_aux_coord(SWnet.aux_coords[0], 0)
    # Flip direction of turbulent fluxes to match convention (positive towards surface)
    #HS = iris.analysis.maths.multiply(HS, -1.)
    #HL = iris.analysis.maths.multiply(HL, -1.)
    #LWnet = iris.analysis.maths.multiply(LWnet, -1.)
    seas_SEB = { 'LWnet': LWnet,  'SWnet': SWnet,
                 'HL': HL,  'HS': HS, 'melt_amnt': melt_amnt, 'Ts':Ts,
                'Etot': Etot,  'albedo': albedo}#'melt_flux': melt_flux,
    lsm_3d = np.broadcast_to(lsm.data, SWnet.shape)
    for v in ['SWnet', 'LWnet', 'HS', 'HL', 'Etot', 'albedo', 'Ts']:#'melt_flux',
        seas_SEB[v].data = np.ma.masked_where(lsm_3d < 0.9, seas_SEB[v].data)
    m = copy.deepcopy(seas_SEB['Etot'].data)
    seas_SEB['melt_flux'] = np.ma.masked_where(lsm_3d < 0.9, m.data)
    return seas_SEB

DJF_00Z = load_var('DJF', '00Z')
DJF_03Z = load_var('DJF', '03Z')
DJF_06Z = load_var('DJF', '06Z')
DJF_09Z = load_var('DJF', '09Z')
DJF_12Z = load_var('DJF', '12Z')
DJF_15Z = load_var('DJF', '15Z')
DJF_18Z = load_var('DJF', '18Z')
DJF_21Z = load_var('DJF', '21Z')

DJF_inlet = pd.DataFrame(columns = time_dict.keys(), index = ['00Z', '03Z','06Z', '09Z', '12Z', '15Z', '18Z', '21Z'])
DJF_ice_shelf = pd.DataFrame(columns = time_dict.keys(), index = ['00Z', '03Z','06Z', '09Z', '12Z', '15Z', '18Z', '21Z'])

for v in DJF_00Z.keys():
    v_list = []
    h_list = []
    for time_dict in [ DJF_06Z, DJF_09Z, DJF_12Z, DJF_15Z, DJF_18Z, DJF_21Z, DJF_00Z, DJF_03Z]: # UTC - 3
        # find values for points close to steep orography
        f = np.ma.masked_where((np.broadcast_to(orog.data, time_dict[v].data.shape) > 250.) & (np.broadcast_to(orog.data, time_dict[v].data.shape) > 100.), time_dict[v].data)
        g = np.ma.masked_where(np.broadcast_to(orog[0].data < 100., time_dict[v].data.shape), time_dict[v].data)
        v_list.append(f.mean())
        h_list.append(g.mean())
    DJF_inlet[v] = pd.Series(v_list,  index = ['00Z', '03Z','06Z', '09Z', '12Z', '15Z', '18Z', '21Z'])
    DJF_ice_shelf[v] = pd.Series(h_list,  index = ['00Z', '03Z','06Z', '09Z', '12Z', '15Z', '18Z', '21Z'])

DJF_inlet['melt_flux'].values[DJF_inlet['melt_flux'].values < 0] = 0.
DJF_ice_shelf['melt_flux'].values[DJF_ice_shelf['melt_flux'].values < 0] = 0.


JJA_00Z = load_var('JJA', '00Z')
JJA_03Z = load_var('JJA', '03Z')
JJA_06Z = load_var('JJA', '06Z')
JJA_09Z = load_var('JJA', '09Z')
JJA_12Z = load_var('JJA', '12Z')
JJA_15Z = load_var('JJA', '15Z')
JJA_18Z = load_var('JJA', '18Z')
JJA_21Z = load_var('JJA', '21Z')

JJA_inlet = pd.DataFrame(columns = time_dict.keys(), index = ['00Z', '03Z','06Z', '09Z', '12Z', '15Z', '18Z', '21Z'])
JJA_ice_shelf = pd.DataFrame(columns = time_dict.keys(), index = ['00Z', '03Z','06Z', '09Z', '12Z', '15Z', '18Z', '21Z'])

for v in JJA_00Z.keys():
    v_list = []
    h_list = []
    for time_dict in [ JJA_06Z, JJA_09Z, JJA_12Z, JJA_15Z, JJA_18Z, JJA_21Z, JJA_00Z, JJA_03Z]: # UTC - 3
        # find values for points close to steep orography
        f = np.ma.masked_where((np.broadcast_to(orog.data, time_dict[v].data.shape) > 250.) & (np.broadcast_to(orog.data, time_dict[v].data.shape) > 100.), time_dict[v].data)
        g = np.ma.masked_where(np.broadcast_to(orog[0].data < 100., time_dict[v].data.shape), time_dict[v].data)
        v_list.append(f.mean())
        h_list.append(g.mean())
    JJA_inlet[v] = pd.Series(v_list,  index = ['00Z', '03Z','06Z', '09Z', '12Z', '15Z', '18Z', '21Z'])
    JJA_ice_shelf[v] = pd.Series(h_list,  index = ['00Z', '03Z','06Z', '09Z', '12Z', '15Z', '18Z', '21Z'])

JJA_inlet['melt_flux'].values[JJA_inlet['melt_flux'].values < 0] = 0.
JJA_ice_shelf['melt_flux'].values[JJA_ice_shelf['melt_flux'].values < 0] = 0.

MAM_00Z = load_var('MAM', '00Z')
MAM_03Z = load_var('MAM', '03Z')
MAM_06Z = load_var('MAM', '06Z')
MAM_09Z = load_var('MAM', '09Z')
MAM_12Z = load_var('MAM', '12Z')
MAM_15Z = load_var('MAM', '15Z')
MAM_18Z = load_var('MAM', '18Z')
MAM_21Z = load_var('MAM', '21Z')

MAM_inlet = pd.DataFrame(columns = time_dict.keys(), index = ['00Z', '03Z','06Z', '09Z', '12Z', '15Z', '18Z', '21Z'])
MAM_ice_shelf = pd.DataFrame(columns = time_dict.keys(), index = ['00Z', '03Z','06Z', '09Z', '12Z', '15Z', '18Z', '21Z'])

for v in MAM_00Z.keys():
    v_list = []
    h_list = []
    for time_dict in [ MAM_06Z, MAM_09Z, MAM_12Z, MAM_15Z, MAM_18Z, MAM_21Z, MAM_00Z, MAM_03Z]: # UTC - 3
        # find values for points close to steep orography
        f = np.ma.masked_where((np.broadcast_to(orog.data, time_dict[v].data.shape) > 250.) & (np.broadcast_to(orog.data, time_dict[v].data.shape) > 100.), time_dict[v].data)
        g = np.ma.masked_where(np.broadcast_to(orog[0].data < 100., time_dict[v].data.shape), time_dict[v].data)
        v_list.append(f.mean())
        h_list.append(g.mean())
    MAM_inlet[v] = pd.Series(v_list,  index = ['00Z', '03Z','06Z', '09Z', '12Z', '15Z', '18Z', '21Z'])
    MAM_ice_shelf[v] = pd.Series(h_list,  index = ['00Z', '03Z','06Z', '09Z', '12Z', '15Z', '18Z', '21Z'])

MAM_inlet['melt_flux'].values[MAM_inlet['melt_flux'].values < 0] = 0.
MAM_ice_shelf['melt_flux'].values[MAM_ice_shelf['melt_flux'].values < 0] = 0.
s

DJF_all_times = {}
for v in DJF_00Z.keys():
    DJF_all_times[v] = (np.mean(DJF_00Z[v].data, axis = 0) + np.mean(DJF_03Z[v].data, axis = 0)  + np.mean(DJF_06Z[v].data, axis = 0) + np.mean(DJF_09Z[v].data, axis = 0)  + np.mean(DJF_12Z[v].data, axis = 0)  +np.mean(DJF_15Z[v].data, axis = 0)  + np.mean(DJF_18Z[v].data, axis = 0)  + np.mean(DJF_21Z[v].data, axis = 0) )/8.

m = copy.deepcopy(DJF_all_times['Etot'])
m[DJF_all_times['Ts'].data < 273] = 0.

def plot_SEB_composites(var1, var2, var3, var4, var5, var6, regime, seas):
    fig, axs = plt.subplots(3,2, frameon=False, figsize=(11, 18))
    axs = axs.flatten()
    fig.patch.set_visible(False)
    var_list = [var1[:7305], var2[:7305], var3[:7305], var4[:7305], var5[:7305], var6[:7305]]
    for i, j in zip(axs, ['SW$_{net}$', 'LW$_{net}$', 'H$_{L}$','H$_{S}$',  'E$_{tot}$', 'E$_{melt}$',]):
        i.set_title(j, color = 'dimgrey', fontsize = 34, pad = 20)
        plt.axis = 'off'
    lims = {'SW$_{net}$': (-10, 10),
             'LW$_{net}$': (-10, 10),
            'H$_{L}$': (-5,5),
            'H$_{S}$': (-5, 5),
            'E$_{tot}$': (-10,10),
            'E$_{melt}$': (-5, 5)
    }
    for ax in [axs[1], axs[3], axs[5]]:
        ax.yaxis.tick_right()
    for ax, cf_var, j in zip(axs, var_list, ['SW$_{net}$', 'LW$_{net}$', 'H$_{L}$','H$_{S}$',  'E$_{tot}$', 'E$_{melt}$',]):
        ax.axis = 'off'
        ax.tick_params(which='both', axis='both', labelsize=24, labelcolor='dimgrey', pad=10, size=0, tick1On=False, tick2On=False)
        PlotLonMin = np.min(lsm.coord('longitude').points)
        PlotLonMax = np.max(lsm.coord('longitude').points)
        PlotLatMin = np.min(lsm.coord('latitude').points)
        PlotLatMax = np.max(lsm.coord('latitude').points)
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
        xlon, ylat = np.meshgrid(lsm.coord('longitude').points, lsm.coord('latitude').points)
        #cf_var, regime_mask = apply_composite_mask(regime, var)
        #bwr_zero = shiftedColorMap(cmap=matplotlib.cm.bwr, min_val=lims[j][0], max_val=lims[j][1], name='bwr_zero', var=cf_var[40:135, 90:155], start=0.15, stop=0.85)
        c = ax.pcolormesh(xlon, ylat, np.ma.masked_where(lsm[0].data<0.9, cf_var), cmap='bwr', vmin=lims[j][0], vmax=lims[j][1], zorder=1)  # latlon=True, transform=ccrs.PlateCarree(),
        plt.sca(ax)
        CBarXTicks = np.linspace(lims[j][0], lims[j][1], num = 3)
        CBar = plt.colorbar(c, orientation='horizontal', extend='both', ticks=CBarXTicks)
        CBar.solids.set_edgecolor("face")
        CBar.outline.set_edgecolor('dimgrey')
        CBar.ax.tick_params(which='both', axis='both', labelsize=24, labelcolor='dimgrey', pad=10, size=0, tick1On=False,tick2On=False)
        CBar.outline.set_linewidth(2)
        coast = ax.contour(xlon, ylat, lsm[0].data, levels=[0.9], colors='#222222', lw=2, latlon=True, zorder=2)
        topog = ax.contour(xlon, ylat, orog[0].data, levels=[500], colors='#222222', linewidth=1.5, latlon=True, zorder=3)
        topog = ax.contour(xlon, ylat, orog[0].data, levels=[500, 1000, 1500, 2000], colors='#A4A3A3', linewidth=0.25,
                           latlon=True, zorder=3)
    plt.subplots_adjust(bottom = 0.06, top = 0.94, hspace = 0.3, wspace = 0.25, right = 0.87)
    plt.savefig('OneDrive - University of Reading/Hindcast paper/SEB_composite_subplot_' + regime + '_' + seas + '.png')
    plt.savefig('OneDrive - University of Reading/Hindcast paper/SEB_composite_subplot_' + regime + '_' + seas + '.eps')
    plt.show()


plot_SEB_composites(DJF_all_times['SWnet'], DJF_all_times['LWnet'], DJF_all_times['HL'],DJF_all_times['HS'],DJF_all_times['Etot'],
                    DJF_all_times['melt_flux'],regime = 'ERA5_DJF', seas='DJF')
