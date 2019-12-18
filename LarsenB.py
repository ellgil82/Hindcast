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
from sklearn.preprocessing import normalize
import datetime
import metpy
import metpy.calc
from scipy import stats

# Set up filepath
if host == 'jasmin':
    filepath = '/gws/nopw/j04/bas_climate/users/ellgil82/hindcast/output/LarsenB/'
elif host == 'bsl':
    filepath = '/data/mac/ellgil82/hindcast/output/'

## Load data
def load_vars():
    try:
        melt_flux = iris.load_cube(filepath +  'LarsenB_land_snow_melt_flux.nc', 'Snow melt heating flux')  # W m-2
        melt_amnt = iris.load_cube(filepath + 'LarsenB_land_snow_melt_amnt.nc', 'Snowmelt')  # kg m-2 TS-1 (TS = 100 s)
        melt_amnt = iris.analysis.maths.multiply(melt_amnt, 108.) # 10800 s in 3 hrs / 100 s in a model timestep = 108 ==> melt amount per output timestep
        HL = iris.load_cube(filepath + 'LarsenB_latent_heat.nc')
        HS = iris.load_cube(filepath + 'LarsenB_sensible_heat.nc')
        HL = iris.analysis.maths.multiply(HL, -1.)
        HS = iris.analysis.maths.multiply(HS, -1.)
        SWnet = iris.load_cube(filepath + 'LarsenB_surface_SW_net.nc')
        LWnet = iris.load_cube(filepath + 'LarsenB_surface_LW_net.nc')
        Etot = HL.data+HS.data+SWnet.data+LWnet.data
        Ts = iris.load_cube(filepath + 'LarsenB_Ts.nc')
        Tair = iris.load_cube(filepath + 'LarsenB_Tair_1p5m.nc')
        Tair.convert_units('celsius')
        Ts.convert_units('celsius')
        u = iris.load_cube(filepath + 'LarsenB_u_10m.nc')
        v = iris.load_cube(filepath + 'LarsenB_v_10m.nc')
        v = v[:,:,1:,:]
        MSLP = iris.load_cube(filepath + 'LarsenB_MSLP.nc')
        MSLP.convert_units('hPa')
        orog = iris.load_cube(filepath + 'orog.nc')
        orog = orog[0, 0, :, :]
        LSM = iris.load_cube(filepath + 'lsm.nc')
        lsm = LSM[0, 0, :, :]
    except iris.exceptions.ConstraintMismatchError:
        print('Files not found')
    var_list = [SWnet, LWnet, HL, HS, melt_amnt, melt_flux, Etot, lsm, orog, Ts, u, v, MSLP]
    for i in var_list:
        real_lon, real_lat = rotate_data(i, np.ndim(i)-2, np.ndim(i)-1)
    vars_yr = {'melt_flux': melt_flux[:,0,:,:],  'melt_amnt': melt_amnt[:,0,:,:], 'HL': HL[:,0,:,:], 'HS': HS[:,0,:,:],
               'Etot': Etot[:,0,:,:], 'Ts': Ts[:,0,:,:],'Tair': Tair[:,0,:,:], 'u': u[:,0,:,:], 'v': v[:,0,:,:], 'MSLP': MSLP[:,0,:,:],
               'LWnet': LWnet[:,0,:,:], 'SWnet': SWnet[:,0,:,:], 'orog': orog, 'lsm': lsm,'lon': real_lon, 'lat': real_lat}
    return vars_yr

var_dict = load_vars()

## Larsen B domain: var[:, 130:175, 120:170]
LarsenB_melt = np.sum(var_dict['melt_amnt'][:, 130:175, 120:170].data, axis =0)
totm_LB = np.sum(LarsenB_melt)*(4000*4000)/10**12



def totm_map(vars_yr):
    fig, ax = plt.subplots(figsize=(8, 8))
    CbAx = fig.add_axes([0.27, 0.2, 0.5, 0.025])
    ax.axis('off')
    c = ax.pcolormesh(np.sum(vars_yr['melt_amnt'].data, axis = 0), vmin = 0,vmax = 600)
    xticks = [0,300,600]
    cb_lab = 'Cumulative snow \nmelt amount (mm w.e.)'
    ax.contour(vars_yr['lsm'].data, colors='#222222')
    ax.contour(vars_yr['orog'].data, colors='#222222', levels=[50], width = 2)
    cb = plt.colorbar(c, orientation='horizontal', cax=CbAx,  extend = "max", ticks=xticks,)
    cb.outline.set_edgecolor('dimgrey')
    cb.ax.tick_params(which='both', axis='both', labelsize=24, labelcolor='dimgrey', pad=10, size=0, tick1On=False,tick2On=False)
    cb.outline.set_linewidth(2)
    cb.ax.xaxis.set_ticks_position('bottom')
    cb.set_label(cb_lab, fontsize=24, color='dimgrey', labelpad=20)
    plt.subplots_adjust(left=0.1, right=0.9, top=0.92, bottom=0.32, hspace=0.3, wspace=0.05)
    plt.savefig('/gws/nopw/j04/bas_climate/users/ellgil82/hindcast/figures/melt_cumulative_LarsenB.png', transparent=True)
    plt.savefig('/gws/nopw/j04/bas_climate/users/ellgil82/hindcast/figures/melt_cumulative_LarsenB.eps', transparent=True)
    plt.show()

totm_map(var_dict)

def plot_SEB_composites():
    fig, axs = plt.subplots(2,2, frameon=False, figsize=(11, 14))
    axs = axs.flatten()
    fig.patch.set_visible(False)
    var_list = [var_dict['HL'], var_dict['HS'], var_dict['Etot'], var_dict['melt_flux']]#var_dict['SWnet'], var_dict['LWnet'],
    for i, j in zip(axs, [ 'H$_{L}$','H$_{S}$',  'E$_{tot}$', 'E$_{melt}$',]):#'SW$_{net}$', 'LW$_{net}$',
        i.set_title(j, color = 'dimgrey', fontsize = 34)
    plt.axis = 'off'
    for ax in [axs[1], axs[3]]:#, axs[5]]:
        ax.yaxis.tick_right()
    for ax, var in zip(axs, var_list):
        ax.axis = 'off'
        ax.tick_params(which='both', axis='both', labelsize=24, labelcolor='dimgrey', pad=10, size=0, tick1On=False, tick2On=False)
        PlotLonMin = np.min(var_dict['lon'][120:170])
        PlotLonMax = np.max(var_dict['lon'][120:170])
        PlotLatMin = np.min(var_dict['lat'][130:175])
        PlotLatMax = np.max(var_dict['lat'][130:175])
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
        xlon, ylat = np.meshgrid(var_dict['lon'][120:170], var_dict['lat'][130:175])
        cf_var = np.mean(var.data[565:1450,130:175, 120:170], axis = (0))
        #cf_var = normalize(cf_var)
        #bwr_zero = shiftedColorMap(cmap=matplotlib.cm.bwr, min_val=-50, max_val=50, name='bwr_zero', var=cf_var.data, start=0.15, stop=0.85)
        c = ax.pcolormesh(xlon, ylat, cf_var, vmin = -50, vmax =50, cmap='bwr')#, vmin=-6., vmax=3., zorder=1)  # latlon=True, transform=ccrs.PlateCarree(),
        coast = ax.contour(xlon, ylat, var_dict['lsm'].data[130:175, 120:170], levels=[0], colors='#222222', lw=2, latlon=True, zorder=2)
        topog = ax.contour(xlon, ylat, var_dict['orog'].data[130:175, 120:170], levels=[50], colors='#222222', linewidth=1.5, latlon=True, zorder=3)
    CBarXTicks = [-50, 0, 50]  # CLevs[np.arange(0,len(CLevs),int(np.ceil(len(CLevs)/5.)))]
    CBAxes = fig.add_axes([0.2, 0.15, 0.6, 0.015])
    CBar = plt.colorbar(c, cax=CBAxes, orientation='horizontal', extend='both', ticks=CBarXTicks)
    CBar.set_label('Mean flux (W m$^{-2}$)', fontsize=34, labelpad=10, color='dimgrey')
    CBar.solids.set_edgecolor("face")
    CBar.outline.set_edgecolor('dimgrey')
    CBar.ax.tick_params(which='both', axis='both', labelsize=34, labelcolor='dimgrey', pad=10, size=0, tick1On=False, tick2On=False)
    CBar.outline.set_linewidth(2)
    plt.subplots_adjust(bottom = 0.22, top = 0.9, hspace = 0.3, wspace = 0.25, right = 0.87)
    plt.savefig('/gws/nopw/j04/bas_climate/users/ellgil82/hindcast/figures/SEB_subplot_LarsenB_zoomed_norad.png')
    plt.savefig('/gws/nopw/j04/bas_climate/users/ellgil82/hindcast/figures/SEB_subplot_LarsenB_zoomed_norad.eps')
    plt.show()

plot_SEB_composites()


# Find daily means, maxes, climatologies and anomalies
Ts_clim = iris.load_cube('/gws/nopw/j04/bas_climate/users/ellgil82/hindcast/output/alloutput/Ts_climatology.nc')
Ts_clim.convert_units('celsius')
Ts_clim = np.concatenate((Ts_clim[244:].data, Ts_clim[:90].data), axis = 0)
Tair_clim = iris.load_cube('/gws/nopw/j04/bas_climate/users/ellgil82/hindcast/output/alloutput/Tair_1p5m_climatology.nc')
Tair_clim.convert_units('celsius')
Tair_clim = np.concatenate((Tair_clim[244:].data, Tair_clim[:90].data), axis = 0)
import iris.coord_categorisation as cat
cat.add_day_of_year(var_dict['Ts'], 't', name='day')
cat.add_day_of_year(var_dict['Tair'], 't', name='day')
Ts_daymax = var_dict['Ts'].aggregated_by('day', iris.analysis.MAX)
Ts_daily = var_dict['Ts'].aggregated_by('day', iris.analysis.MEAN)
Ts_anom = Ts_daily - Ts_clim[:,0,:,:]
Tair_daymax = var_dict['Tair'].aggregated_by('day', iris.analysis.MAX)
Tair_daily = var_dict['Tair'].aggregated_by('day', iris.analysis.MEAN)
Tair_anom = Tair_daily - Tair_clim[:,0,:,:]


def plot_synop_composite(cf_var, c_var, u_var, v_var):
    fig = plt.figure(frameon=False, figsize=(10, 13))  # !!change figure dimensions when you have a larger model domain
    fig.patch.set_visible(False)
    ax = fig.add_subplot(111)#, projection=ccrs.PlateCarree())
    plt.axis = 'off'
    PlotLonMin = np.min(var_dict['lon'][120:170])
    PlotLonMax = np.max(var_dict['lon'][120:170])
    PlotLatMin = np.min(var_dict['lat'][130:175])
    PlotLatMax = np.max(var_dict['lat'][130:175])
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
    bwr_zero = shiftedColorMap(cmap=matplotlib.cm.bwr, min_val=-15., max_val=2., name='bwr_zero', var=cf_var.data, start=0.15,
                               stop=0.85) #-6, 3
    xlon, ylat = np.meshgrid(var_dict['lon'][120:170], var_dict['lat'][130:175])
    c = ax.pcolormesh(xlon, ylat, cf_var, cmap=bwr_zero,vmin=-15., vmax=2., zorder=1)  # latlon=True, transform=ccrs.PlateCarree(),
    cs = ax.contour(xlon, ylat, np.ma.masked_where(var_dict['lsm'].data[130:175, 120:170] == 1, c_var), latlon=True, colors='k', zorder=4)
    ax.clabel(cs, inline=True, fontsize = 24, inline_spacing = 30, fmt =  '%1.0f')
    coast = ax.contour(xlon, ylat, var_dict['lsm'].data[130:175, 120:170], levels=[0], colors='#222222', lw=2, latlon=True, zorder=2)
    topog = ax.contour(xlon, ylat, var_dict['orog'].data[130:175, 120:170], levels=[50], colors='#222222', linewidth=1.5, latlon=True, zorder=3)
    CBarXTicks = [-15,  0,  2]  # CLevs[np.arange(0,len(CLevs),int(np.ceil(len(CLevs)/5.)))]
    CBAxes = fig.add_axes([0.22, 0.2, 0.6, 0.025])
    CBar = plt.colorbar(c, cax=CBAxes, orientation='horizontal', ticks=CBarXTicks, extend = 'both')  #
    CBar.set_label('Mean daily maximum 1.5 m \nair temperature ($^{\circ}$C)', fontsize=34, labelpad=10, color='dimgrey')
    CBar.solids.set_edgecolor("face")
    CBar.outline.set_edgecolor('dimgrey')
    CBar.ax.tick_params(which='both', axis='both', labelsize=34, labelcolor='dimgrey', pad=10, size=0, tick1On=False,
                        tick2On=False)
    CBar.outline.set_linewidth(2)
    x, y = np.meshgrid(var_dict['lon'], var_dict['lat'])
    q = ax.quiver(x[::8, ::8], y[::8, ::8], u_var[::8, ::8], v_var[::8, ::8], pivot='middle', scale=50,
                  zorder=5, width = 0.006)
    plt.quiverkey(q, 0.25, 0.9, 5, r'$5$ $m$ $s^{-1}$', labelpos='N', color='dimgrey', labelcolor='dimgrey',
                  fontproperties={'size': '32', 'weight': 'bold'},
                  coordinates='figure', )
    plt.subplots_adjust(left = 0.2, bottom = 0.33, right = 0.85)
    plt.savefig('/gws/nopw/j04/bas_climate/users/ellgil82/hindcast/figures/synoptic_cond_Tair_max_LarsenB_melt_period_zoomed.png')
    plt.savefig('/gws/nopw/j04/bas_climate/users/ellgil82/hindcast/figures/synoptic_cond_Tair_max_LarsenB_melt_period_zoomed.eps')
    plt.show()

plot_synop_composite(cf_var = Tair_daymax[40:150,130:175, 120:170 ].collapsed('t', iris.analysis.MEAN).data,
                     c_var = var_dict['MSLP'][40:150,130:175, 120:170].collapsed('t', iris.analysis.MEAN).data,
                     u_var = var_dict['u'][40:150,130:175, 120:170].collapsed('t', iris.analysis.MEAN).data,
                     v_var = var_dict['v'][40:150,130:175, 120:170].collapsed('t', iris.analysis.MEAN).data)


def run_corr(year_vars, xvar, yvar):
    unmasked_idx = np.where(year_vars['lsm'][110:, 110:200].data == 1)
    r = np.zeros(xvar.shape)
    p = np.zeros(xvar.shape)
    err = np.zeros(xvar.shape)
    for x, y in zip(unmasked_idx[0],unmasked_idx[1]):
        if x > 0. or y > 0.:
            slope, intercept, r_value, p_value, std_err = stats.linregress(xvar[:,x, y], yvar[:,x, y])
            r[x,y] = r_value
            p[x,y] = p_value
            err[x,y] = std_err
        r2 = r**2
    return r, r2, p, err

def correlation_maps(xvar, yvar):
    r, r2, p, err = run_corr(var_dict, xvar=var_dict[xvar][110:,110:200].data, yvar=var_dict[yvar][110:,110:200].data)
    #r = np.ma.masked_where(var_dict['lsm'].data == 0, r)
    sig = np.ma.masked_where(var_dict['lsm'][110:,110:].data == 0, p)
    sig = np.ma.masked_greater(sig, 0.01)
    mean_r_composite = r
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.axis('off')
    # Make masked areas white, instead of colour used for zero in colour map
    ax.contourf(sig.mask, cmap='Greys_r')
    # Plot coastline
    ax.contour(var_dict['lsm'][110:,110:200].data, colors='#222222', lw=2)
    # Plot correlations
    c = ax.pcolormesh( mean_r_composite, cmap=matplotlib.cm.Spectral_r, vmin=-0.1, vmax=.5)
    # Plot 50 m orography contour on top
    ax.contour(var_dict['orog'][110:,110:200].data, colors='#222222', levels=[50])
    # Overlay stippling to indicate signficance
    ax.contourf(sig, hatches='...', alpha = 0.0)
    # Set up colourbar
    cb = plt.colorbar(c, orientation='horizontal', extend ='both')#, ticks=[-1, 0, 1])
    cb.solids.set_edgecolor("face")
    cb.outline.set_edgecolor('dimgrey')
    cb.ax.tick_params(which='both', axis='both', labelsize=24, labelcolor='dimgrey', pad=10, size=0, tick1On=False,
                      tick2On=False)
    cb.outline.set_linewidth(2)
    cb.ax.xaxis.set_ticks_position('bottom')
    cb.set_label('Correlation coefficient', fontsize=24, color='dimgrey', labelpad=30)
    # Save figure
    if host == 'bsl':
        plt.savefig('/users/ellgil82/figures/Hindcast/SMB/' + xvar + '_v_' + yvar + '_composite.png', transparent=True)
        plt.savefig('/users/ellgil82/figures/Hindcast/SMB/' + xvar + '_v_' + yvar + '_composite.eps', transparent=True)
    elif host == 'jasmin':
        plt.savefig('/gws/nopw/j04/bas_climate/users/ellgil82/hindcast/figures/' + xvar + '_v_' + yvar + '_LarsenB.png', transparent=True)
        plt.savefig('/gws/nopw/j04/bas_climate/users/ellgil82/hindcast/figures/' + xvar + '_v_' + yvar + '_LarsenB.eps', transparent=True)
    plt.show()

correlation_maps(xvar='melt_flux', yvar='HS')
correlation_maps(xvar='melt_flux', yvar='HL')

melt_LB = np.sum(var_dict['melt_amnt'][:, 130:175, 120:170].data, axis = 0)