# Define where the script is running
host = 'bsl'

# Import packages
import iris
import iris.plot as iplt
import sys
import numpy as np
if host == 'jasmin':
    sys.path.append('/gws/nopw/j04/bas_climate/users/ellgil82/scripts/Tools/')
elif host == 'bsl':
    sys.path.append('/users/ellgil82/scripts/Tools/')

from find_gridbox import find_gridbox
from rotate_data import rotate_data
from divg_temp_colourmap import shiftedColorMap
import matplotlib.pyplot as plt
import matplotlib
import scipy
from matplotlib.lines import Line2D

# Set up filepath
if host == 'jasmin':
    filepath = '/gws/nopw/j04/bas_climate/users/ellgil82/hindcast/output/test_run/'
elif host == 'bsl':
    filepath = '/data/mac/ellgil82/hindcast/'

## Check that sea ice and SSTs are inherited from ERA-Interim --> GLM and then GLM --> LAM
## Load ERA-Interim data
# Load sea ice and chop it into matching area/times

date = '20080101'
if date[4:] == '0101':
    ERA_time = 1
elif date[4:] == '0207':
    ERA_time = 75
elif date[4:] == '0130':
    ERA_time = 59

ERA_SIC = iris.load_cube(filepath+ str(int(date[:4])-1) + '1231_12-' + date[:4] + '1231_12_sic_um_grid_glm_n512.nc', 'iceconc')
ERA_SIC = ERA_SIC[ERA_time,0,82:116,807:867]

print ERA_SIC

# Load SSTs and chop it into matching area/times

ERA_SST = iris.load_cube(filepath+ str(int(date[:4])-1) + '1231_12-' + date[:4] + '1231_12_sst_um_grid_glm_n512.nc', 'temp')
ERA_SST = ERA_SST[ERA_time,0,82:116,807:867]
ERA_SST.convert_units('celsius')

## Load GLM data

glm_SIC = iris.load_cube(filepath+date+'T0000Z_glm_pa000.pp', 'sea_ice_area_fraction')
glm_SIC = glm_SIC[0,82:116,807:867]

glm_SST = iris.load_cube(filepath+ date+'T0000Z_glm_pa000.pp', 'surface_temperature')
glm_SST = glm_SST[0,82:116,807:867]
glm_SST.convert_units('celsius')
glm_SST.data = np.ma.array(glm_SST.data, mask = ERA_SST.data.mask)

glm_LSM = iris.load_cube(filepath+date+'T0000Z_glm_pa000.pp', 'land_binary_mask')
glm_LSM = glm_LSM[82:116, 807:867]

# return lats/lons
glm_lat = glm_SIC.coord('latitude').points
glm_lon = glm_SIC.coord('longitude').points

## Load LAM data

LAM_SIC = iris.load_cube(filepath+date+'T0000Z_Peninsula_4km_test_inheritance_pa000.pp', 'sea_ice_area_fraction')
LAM_Ts = iris.load_cube(filepath+date+'T0000Z_Peninsula_4km_test_inheritance_pa000.pp', 'surface_temperature')
LAM_SIC = LAM_SIC[0,:,:]
try:
    LAM_LSM = iris.load_cube(filepath+'new_mask.nc', 'land_binary_mask')
    LAM_LSM = LAM_LSM[0, 0, :, :]
except iris.exceptions.ConstraintMismatchError:
    LAM_LSM = iris.load_cube(filepath+'new_mask.nc', 'LAND MASK (No halo) (LAND=TRUE)')
    LAM_LSM = LAM_LSM[0,0,:,:]

LAM_SST = LAM_Ts[0,:,:]

## Rotate projection
for var in [LAM_SIC, LAM_SST, LAM_LSM]:
    real_lon, real_lat = rotate_data(var, 0, 1)

# Mask surface temperature data over land
LAM_SST.data = np.ma.masked_where(LAM_LSM.data == 1, LAM_SST.data )
LAM_SST.convert_units('celsius')

# Compare re-gridding methods
# Remove coord_system from glm data before regridding
#glm_SST.coord(axis='y').coord_system = None
#glm_SST.coord(axis='x').coord_system = None
#glm_SIC.coord(axis='x').coord_system = None
#glm_SIC.coord(axis='y').coord_system = None

# Up- and down-sample data, respectively
#upsamp_glm_SST = glm_SST.regrid(LAM_SST, iris.analysis.Linear())
#downsamp_LAM_SST = LAM_SST.regrid(glm_SST, iris.analysis.Linear())


# Plot for comparison
def regrid_plot():
    fig, ax = plt.subplots(1,2, figsize = (12,6))
    ax.flatten()
    us = ax[0].pcolormesh(upsamp_glm_SST.data, vmin = -8, vmax = 8)
    ax[0].set_title('GLM regridded onto LAM grid')
    ds = ax[1].pcolormesh(downsamp_LAM_SST.data,vmin = -8, vmax = 8)
    ax[1].set_title('LAM regridded onto GLM grid')
    CBAxes = fig.add_axes([0.25, 0.1, 0.5, 0.03])
    plt.colorbar(ds, orientation='horizontal', cax = CBAxes)
    plt.subplots_adjust(bottom = 0.18)
    plt.savefig(filepath+date+'regrid_difs.png')
    plt.show()

# Upsampling is better; upsample ERA and glm data 
#upsamp_ERA_SST = ERA_SST.regrid(LAM_SST, iris.analysis.Linear())
#upsamp_ERA_SIC = ERA_SIC.regrid(LAM_SIC, iris.analysis.Linear())
#upsamp_glm_SIC = glm_SIC.regrid(LAM_SIC, iris.analysis.Linear())

## Test whether ERA/GLM are the same

#glm_minus_ERA_SIC = upsamp_glm_SIC.data - upsamp_ERA_SIC.data
#LAM_minus_ERA_SIC = LAM_SIC.data - upsamp_ERA_SIC.data
#LAM_minus_glm_SIC = LAM_SIC.data - upsamp_glm_SIC.data

#glm_minus_ERA_SST = upsamp_glm_SST.data - upsamp_ERA_SST.data
#LAM_minus_ERA_SST = LAM_SST.data - upsamp_ERA_SST.data
#LAM_minus_glm_SST = LAM_SST.data - upsamp_glm_SST.data


# Express difference as percentage of range
#SST_range = np.max(np.maximum(glm_SST.data, ERA_SST.data))-np.min(np.minimum(glm_SST.data, ERA_SST.data))
#SST_glm_ERA_pct = (glm_minus_ERA_SST/SST_range)*100

#SIC_range = np.max(np.maximum(glm_SIC.data, ERA_SIC.data))-np.min(np.minimum(glm_SIC.data, ERA_SIC.data))
#SIC_glm_ERA_pct = (glm_minus_ERA_SIC/SIC_range)*100

#np.mean(glm_minus_ERA_SIC)
#np.mean(glm_minus_ERA_SST)

## Compare with climatological SSTs and SIC
# Load climatological variables
clim_SST = iris.load_cube(filepath + 'clim_sst.nc', 'SURFACE TEMPERATURE AFTER TIMESTEP')
clim_SST.convert_units('celsius')
Jan_SST = clim_SST[0,0,82:116, 807:867]
Feb_SST = clim_SST[1,0, 82:116, 807:867]
clim_SIC = iris.load_cube(filepath + 'clim_sic.nc', 'FRAC OF SEA ICE IN SEA AFTER TSTEP')
Jan_SIC = clim_SIC[0,0,82:116, 807:867]
Feb_SIC = clim_SIC[1,0,82:116, 807:867]


## Set up plotting options
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Segoe UI', 'Helvetica', 'Liberation sans', 'Tahoma', 'DejaVu Sans',
                               'Verdana']

def plot_SIC_SST():
    fig, ax = plt.subplots(1,2, figsize = (16,12))
    ax.flatten()
    case_list = ['SST', 'SIC']
    lab_dict = {0: 'a', 1: 'b'}
    for a in [0,1]:
        ax[a].spines['right'].set_visible(False)
        ax[a].spines['left'].set_visible(False)
        ax[a].spines['top'].set_visible(False)
        ax[a].spines['bottom'].set_visible(False)
        PlotLonMin = np.min(glm_lon)
        PlotLonMax = np.max(glm_lon)
        PlotLatMin = np.min(glm_lat)
        PlotLatMax = np.max(glm_lat)
        XTicks = np.linspace(PlotLonMin, PlotLonMax, 3)
        XTickLabels = [None] * len(XTicks)
        for i, XTick in enumerate(XTicks):
            if XTick < 0:
                XTickLabels[i] = '{:.0f}{:s}'.format(np.abs(XTick), '$^{\circ}$W')
            else:
                XTickLabels[i] = '{:.0f}{:s}'.format(np.abs(XTick), '$^{\circ}$E')
        plt.sca(ax[a])
        plt.xticks(XTicks, XTickLabels)
        ax[a].set_xlim(PlotLonMin, PlotLonMax)
        ax[a].tick_params(which='both', pad=20, labelsize = 34, color = 'dimgrey')
        YTicks = np.linspace(PlotLatMin, PlotLatMax, 4)
        YTickLabels = [None] * len(YTicks)
        for i, YTick in enumerate(YTicks):
            if YTick < 0:
                YTickLabels[i] = '{:.0f}{:s}'.format(np.abs(YTick), '$^{\circ}$S')
            else:
                YTickLabels[i] = '{:.0f}{:s}'.format(np.abs(YTick), '$^{\circ}$N')
        plt.sca(ax[a])
        plt.yticks(YTicks, YTickLabels)
        ax[a].set_ylim(PlotLatMin, PlotLatMax)
        ax[a].tick_params(which='both', axis='both', labelsize=34, labelcolor='dimgrey', pad=20, size=0, tick1On=False, tick2On=False)
        ax[a].set_ylim(PlotLatMin, PlotLatMax)
        lab = ax[a].text(-80, -61.5, zorder=100,  s=lab_dict[a], fontsize=32, fontweight='bold', color='dimgrey')
        ax[a].set_title(case_list[a], fontsize=40, color='dimgrey')
    #bwr_zero = shiftedColorMap(cmap=matplotlib.cm.bwr_r, min_val=-5, max_val=40, name='bwr_zero', var=SST_glm_ERA_pct, start = 0., stop = 1.)
    SST = ax[0].pcolormesh(LAM_LSM.coord('longitude').points, LAM_LSM.coord('latitude').points, SST_glm_ERA_pct, cmap = 'viridis_r')#, vmin = -40, vmax = 1)
    SIC = ax[1].pcolormesh(LAM_LSM.coord('longitude').points, LAM_LSM.coord('latitude').points, SIC_glm_ERA_pct, cmap = 'viridis_r')#, vmin = 0., vmax = 1.)
    ax[1].yaxis.tick_right()
    CBarXTicks = [-40, -20, -10, 0, 10]  # CLevs[np.arange(0,len(CLevs),int(np.ceil(len(CLevs)/5.)))]
    CBAxes = fig.add_axes([0.25, 0.15, 0.5, 0.03])
    CBar = plt.colorbar(SIC, cax=CBAxes, orientation='horizontal', ticks=CBarXTicks)  #
    CBar.set_label('Difference glm - ERA-Interim (%)', fontsize=34, labelpad=10, color='dimgrey')
    CBar.solids.set_edgecolor("face")
    CBar.outline.set_edgecolor('dimgrey')
    CBar.ax.tick_params(which='both', axis='both', labelsize=34, labelcolor='dimgrey', pad=10, size=0, tick1On=False, tick2On=False)
    CBar.outline.set_linewidth(2)
    plt.subplots_adjust(left = 0.12, right = 0.88, top = 0.9, bottom = 0.28, hspace = 0.1)
    #plt.savefig(filepath+date+'_SIC_SST_inheritance_glm_ERA.png')
    #plt.savefig(filepath+date+'_SIC_SST_inheritance_glm_ERA.eps')
    plt.show()

#plot_SIC_SST()

def subplots(var):
    fig, ax = plt.subplots(1,3, figsize = (22,8))
    ax.flatten()
    case_list = ['ERA-Interim', 'GLM', 'LAM']
    lab_dict = {0: 'a', 1: 'b', 2: 'c'}
    for a in [0, 1, 2]:
        ax[a].spines['right'].set_visible(False)
        ax[a].spines['left'].set_visible(False)
        ax[a].spines['top'].set_visible(False)
        ax[a].spines['bottom'].set_visible(False)
        PlotLonMin = np.min(LAM_SST.coord('longitude').points)
        PlotLonMax = np.max(LAM_SST.coord('longitude').points)
        PlotLatMin = np.min(LAM_SST.coord('latitude').points)
        PlotLatMax = np.max(LAM_SST.coord('latitude').points)
        XTicks = np.linspace(PlotLonMin, PlotLonMax, 3)
        XTickLabels = [None] * len(XTicks)
        for i, XTick in enumerate(XTicks):
            if XTick < 0:
                XTickLabels[i] = '{:.0f}{:s}'.format(np.abs(XTick), '$^{\circ}$W')
            else:
                XTickLabels[i] = '{:.0f}{:s}'.format(np.abs(XTick), '$^{\circ}$E')
        plt.sca(ax[a])
        plt.xticks(XTicks, XTickLabels)
        ax[a].set_xlim(PlotLonMin, PlotLonMax)
        ax[a].tick_params(which='both', pad=20, labelsize = 34, color = 'dimgrey')
        YTicks = np.linspace(PlotLatMin, PlotLatMax, 4)
        YTickLabels = [None] * len(YTicks)
        for i, YTick in enumerate(YTicks):
            if YTick < 0:
                YTickLabels[i] = '{:.0f}{:s}'.format(np.abs(YTick), '$^{\circ}$S')
            else:
                YTickLabels[i] = '{:.0f}{:s}'.format(np.abs(YTick), '$^{\circ}$N')
        plt.sca(ax[a])
        plt.yticks(YTicks, YTickLabels)
        ax[a].set_ylim(PlotLatMin, PlotLatMax)
        ax[a].tick_params(which='both', axis='both', labelsize=34, labelcolor='dimgrey', pad=20, size=0, tick1On=False, tick2On=False)
        ax[a].set_ylim(PlotLatMin, PlotLatMax)
        lab = ax[a].text(-80, -61.5, zorder=100,  s=lab_dict[a], fontsize=32, fontweight='bold', color='dimgrey')
        ax[a].set_title(case_list[a], fontsize=40, color='dimgrey')
        ax[a].contour(LAM_SIC.coord('longitude').points, LAM_SIC.coord('latitude').points, LAM_LSM.data, colors = 'dimgrey', linewidth = 2)
    if var == 'SIC':
        ERA = ax[0].pcolormesh(LAM_SIC.coord('longitude').points, LAM_SIC.coord('latitude').points, upsamp_ERA_SIC.data, cmap = 'Blues_r', vmin = 0., vmax = 1.)
        GLM = ax[1].pcolormesh(LAM_SIC.coord('longitude').points, LAM_SIC.coord('latitude').points,upsamp_glm_SIC.data, cmap = 'Blues_r', vmin = -0., vmax = 1.)
        LAM = ax[2].pcolormesh(LAM_SIC.coord('longitude').points, LAM_SIC.coord('latitude').points, LAM_SIC.data, cmap = 'Blues_r', vmin = 0., vmax = 1.)
        CBarXTicks = [ 0., 0.5, 1.]  # CLevs[np.arange(0,len(CLevs),int(np.ceil(len(CLevs)/5.)))]
    elif var == 'SST':
        ERA = ax[0].pcolormesh(LAM_SST.coord('longitude').points, LAM_SST.coord('latitude').points, upsamp_ERA_SST.data, cmap = 'Blues_r', vmin = -2., vmax = 3.)
        GLM = ax[1].pcolormesh(LAM_SST.coord('longitude').points, LAM_SST.coord('latitude').points,upsamp_glm_SST.data, cmap = 'Blues_r', vmin = -2., vmax = 3.)
        LAM = ax[2].pcolormesh(LAM_SST.coord('longitude').points, LAM_SST.coord('latitude').points, LAM_SST.data, cmap = 'Blues_r', vmin = -2., vmax = 3.)
        CBarXTicks = [-2, 0, 2]
    ax[2].yaxis.tick_right()
    plt.setp(ax[1].get_yticklabels(), visible=False)
    CBAxes = fig.add_axes([0.4, 0.15, 0.2, 0.03])
    CBar = plt.colorbar(LAM, cax=CBAxes, orientation='horizontal', ticks=CBarXTicks)  #
    CBar.solids.set_edgecolor("face")
    CBar.outline.set_edgecolor('dimgrey')
    CBar.ax.tick_params(which='both', axis='both', labelsize=34, labelcolor='dimgrey', pad=10, size=0, tick1On=False, tick2On=False)
    CBar.outline.set_linewidth(2)
    plt.subplots_adjust(left = 0.08, right = 0.92, top = 0.9, bottom = 0.28, wspace = 0.23, hspace = 0.18)
    if var == 'SIC':
        CBar.set_label('SIC', fontsize=34, labelpad=10, color='dimgrey')
        plt.savefig('/group_workspaces/jasmin4/bas_climate/users/ellgil82/hindcast/output/' + date + '_SIC_inheritance_glm_ERA.png')
        plt.savefig('/group_workspaces/jasmin4/bas_climate/users/ellgil82/hindcast/output/' + date + '_SIC_inheritance_glm_ERA.eps')
    elif var == 'SST':
        CBar.set_label('SST ($^{\circ}$C)', fontsize=34, labelpad=10, color='dimgrey')
        plt.savefig(filepath+date+'_clim_SST_inheritance_glm_ERA.png')
        plt.savefig(filepath+date+'_clim_SST_inheritance_glm_ERA.eps')
    plt.show()

#subplots('SIC')
#subplots('SST')

def subplots_simple(var, clim, savefig):
    fig, ax = plt.subplots(1,3, figsize = (22,8))
    ax.flatten()
    if clim == 'True' or clim == 'yes':
        case_list = ['Climatology', 'GLM', 'LAM']
    else:
        case_list = ['ERA-Interim', 'GLM', 'LAM']
    lab_dict = {0: 'a', 1: 'b', 2: 'c'}
    for a in [0, 1, 2]:
        ax[a].spines['right'].set_visible(False)
        ax[a].spines['left'].set_visible(False)
        ax[a].spines['top'].set_visible(False)
        ax[a].spines['bottom'].set_visible(False)
        ax[a].tick_params(which = 'both', tick1On = 'False', tick2On = 'False')
        ax[a].axis('off')
        plt.setp(ax[a].get_yticklabels(), visible=False)
        plt.setp(ax[a].get_xticklabels(), visible = False)
        lab = ax[a].text(0.1, 0.85, transform = ax[a].transAxes, zorder=100, s=lab_dict[a], fontsize=32, fontweight='bold', color='dimgrey')
        ax[a].set_title(case_list[a], fontsize=40, color='dimgrey')
    if var == 'SST':
        bwr_zero = shiftedColorMap(cmap=matplotlib.cm.bwr, min_val=-2.5, max_val=3.5, name='bwr_zero', var='s', start=0., stop=1.)
        if clim == 'True' or clim == 'yes':
            if date == '20080101' or date == '20080130':
                clim = ax[0].pcolormesh(Jan_SST.data, cmap=bwr_zero, vmin= -2.5, vmax = 3.5)
            elif date == '20080207':
                clim = ax[0].pcolormesh(Feb_SST.data, cmap=bwr_zero, vmin= -2.5, vmax = 3.5)
        else:
            ERA = ax[0].pcolormesh(ERA_SST.data, cmap=bwr_zero,vmin= -2.5, vmax = 3.5)
        lsm0 = ax[0].contour(glm_LSM.data, colors='dimgrey', linewidths=3, levels=[0])
        GLM = ax[1].pcolormesh(glm_SST.data, cmap = bwr_zero, vmin= -2.5, vmax = 3.5)
        lsm1 = ax[1].contour(glm_LSM.data, colors='dimgrey', linewidths = 3, levels = [0])
        LAM = ax[2].pcolormesh(LAM_SST.data, cmap =bwr_zero, vmin= -2.5, vmax = 3.5)
        lsm2 = ax[2].contour(LAM_LSM.data, colors='dimgrey',  linewidths = 3, levels = [0])
        CBarXTicks = [-2, 0, 2]
    elif var == 'SIC':
        if clim == 'True' or clim == 'yes':
            if date == '20080101' or date == '20080130':
                clim = ax[0].pcolormesh(Jan_SIC.data, cmap='Blues_r', vmin= 0., vmax = 1.)
            elif date == '20080207':
                clim = ax[0].pcolormesh(Feb_SIC.data, cmap='Blues_r', vmin= 0., vmax = 1.)
        else:
            ERA = ax[0].pcolormesh(ERA_SIC.data, cmap = 'Blues_r', vmin = 0., vmax = 1.)
        lsm0 = ax[0].contour(glm_LSM.data, colors='dimgrey', linewidths=3, levels=[0])
        GLM = ax[1].pcolormesh(glm_SIC.data, cmap = 'Blues_r', vmin = -0., vmax = 1.)
        lsm1 = ax[1].contour(glm_LSM.data, colors = 'dimgrey',  linewidths = 3, levels = [0])
        LAM = ax[2].pcolormesh(LAM_SIC.data, cmap = 'Blues_r', vmin = 0., vmax = 1.)
        lsm2 = ax[2].contour(LAM_LSM.data, colors = 'dimgrey',  linewidths = 3, levels = [0])
        CBarXTicks = [ 0., 0.5, 1.]  # CLevs[np.arange(0,len(CLevs),int(np.ceil(len(CLevs)/5.)))]
    ax[2].yaxis.tick_right()
    CBAxes = fig.add_axes([0.4, 0.15, 0.2, 0.03])
    CBar = plt.colorbar(LAM, cax=CBAxes, orientation='horizontal', ticks=CBarXTicks)  #
    CBar.solids.set_edgecolor("face")
    CBar.outline.set_edgecolor('dimgrey')
    CBar.ax.tick_params(which='both', axis='both', labelsize=34, labelcolor='dimgrey', pad=10, size=0, tick1On=False, tick2On=False)
    CBar.outline.set_linewidth(2)
    plt.subplots_adjust(left = 0.08, right = 0.92, top = 0.9, bottom = 0.28, wspace = 0.23, hspace = 0.18)
    if savefig == 'yes' or savefig == True:
        if var == 'SIC':
            CBar.set_label('Sea ice fraction', fontsize=34, labelpad=10, color='dimgrey')
            if clim == 'yes':
                plt.savefig(filepath + date + '_clim_SIC_inheritance_glm_ERA_native.png')
                plt.savefig(filepath + date + '_clim_SIC_inheritance_glm_ERA_native.eps')
            else:
                plt.savefig(filepath + date + '_SIC_inheritance_glm_ERA_native.png')
                plt.savefig(filepath + date + '_SIC_inheritance_glm_ERA_native.eps')
        elif var == 'SST':
            CBar.set_label('SST ($^{\circ}$C)', fontsize=34, labelpad=10, color='dimgrey')
            if clim == 'yes':
                plt.savefig(filepath+date+'_clim_SST_inheritance_glm_ERA_native.png')
                plt.savefig(filepath+date+'_clim_SST_inheritance_glm_ERA_native.eps')
            else:
                plt.savefig(filepath + date + '_SST_inheritance_glm_ERA_native.png')
                plt.savefig(filepath + date + '_SST_inheritance_glm_ERA_native.eps')
    plt.show()

subplots_simple('SIC', clim = 'no', savefig = 'yes')
subplots_simple('SST', clim = 'no',  savefig = 'yes')

def mask_difs():
    difs = glm_SIC.data-ERA_SIC.data
    difs[glm_SIC==0] = 0
    difs_m = np.ma.masked_equal(difs, 0)
    difs_m = np.ma.masked_greater_equal(difs, -0.001)
    fig, ax = plt.subplots(1,1, figsize = (12,9))
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(which = 'both', tick1On = 'False', tick2On = 'False')
    ax.axis('off')
    plt.setp(ax.get_yticklabels(), visible=False)
    plt.setp(ax.get_xticklabels(), visible = False)
    bwr_zero = shiftedColorMap(cmap=matplotlib.cm.bwr, min_val=-2.5, max_val=3.5, name='bwr_zero', var='s', start=0., stop=1.)
    GLM = ax.contourf(glm_SST.data, cmap=bwr_zero, vmin=-2.5, vmax=3.5)
    difs = ax.contourf(difs_m, cmap = 'viridis')
    lsm1 = ax.contour(glm_LSM.data, colors='dimgrey', linewidths=3, levels=[0])
    thresh1 = ax.contour(glm_SIC.data, levels =[0.1], linewidths=5, colors = 'r',  zorder = 9, )
    thresh2 = ax.contour(ERA_SIC.data, levels= [0] , linewidths=5,colors = 'green')
    CBarXTicks = [-2, 0, 2]
    SST_axes = fig.add_axes([0.1, 0.1, 0.03, 0.7]) #left, bottom, width, height
    dif_axes = fig.add_axes([0.82, 0.1, 0.03, 0.7])
    CBar1 = plt.colorbar(GLM, cax=SST_axes, orientation='vertical', ticks=CBarXTicks)  #
    ax.text(-0.25, 1.1, s = 'SST ($^{\circ}$C)', transform =  ax.transAxes, rotation = 0, color = 'dimgrey', fontsize = '34')
    CBar2 = plt.colorbar(difs, cax = dif_axes, orientation='vertical', ticks = [0, -0.05, -0.1])
    ax.text(0.98, 1.1, s = 'glm SIC - \nERA SIC', transform =  ax.transAxes, rotation = 0, color = 'dimgrey', fontsize = '34')
    for CBar in [CBar1, CBar2]:
        CBar.solids.set_edgecolor("face")
        CBar.outline.set_edgecolor('dimgrey')
        CBar.ax.tick_params(which='both', axis='both', labelsize=34, labelcolor='dimgrey', pad=10, size=0, tick1On=False, tick2On=False)
        CBar.outline.set_linewidth(2)
    lns = [Line2D([0], [0],  color='red',  linewidth=5),
           Line2D([0], [0], color='green',  linewidth=5)]
    labs = ['glm SIC = 0.1','ERA SIC edge']#  '                      ','                      '
    lgd = plt.legend(lns, labs, ncol=1, bbox_to_anchor=(-6,1.2), borderaxespad=0.,  prop={'size': 24})
    for ln in lgd.get_texts():
        plt.setp(ln, color='dimgrey')
    lgd.get_frame().set_linewidth(0.0)
    plt.subplots_adjust(left = 0.2, bottom = 0.1, right = 0.78, top = 0.8)
    plt.show()