'''# Diagnose foehn from 3-hourly data

# How to diagnose foehn?
- options (Dan Bannister's thesis):
a) surface met-based, i.e. dT > 2 K AND increased wind speed, decreased RH and wind direction cross-AP for 6+ hours
b) isentrope-based, i.e if dZ isentrope > 1000 m for 6+ hours 
c) Froude number/h-hat based, i.e. if Fr > 0.9 for 6+ hours

Only calculate if wind direction within 30 deg region that makes it 'cross-peninsula'.

1. Select time periods when wind direction within range for 6+ hours in a row
2. At each timestep, determine whether conditions are met (for whichever method is being used) for 2+ timesteps in a row.


Foehn stats:
- number of events
- timing of events (so preserve times - use pandas)
- length of events
- strength of events

'''
foehn_stats = []


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
	mean_u = np.mean(u_wind)
	N = 0.01 # s-1 = Brunt-Vaisala frequency
	h = 2000 # m = height of AP mountains
	Fr = mean_u/(N*h)
	h_hat = (N*h)/mean_u
	return Fr, h_hat

def uv_to_ws(u_wind,v_wind):
	'''Calculate wind speed from u and v wind components

		Inputs:

			- u_wind: zonal component of wind (m s-1), where positive = westerly

			- v_wind: meridional component of wind (m s-1), where positive = southerly

		Outputs:

			- ws: wind speed (m s-1)

		'''
	ws = np.sqrt(np.square(u_wind) + np.square(v_wind))
	return ws

def uv_to_wd(u_wind,v_wind):
	'''Calculate wind direction from u and v wind components

		Inputs:

			- u_wind: zonal component of wind (m s-1), where positive = westerly

			- v_wind: meridional component of wind (m s-1), where positive = southerly

		Outputs:

			- wd: wind direction (in degrees), where North = 0

		'''
	wd = 270 - ((math.atan2(u_wind, v_wind)) * (180 / math.pi)) % 360
	return wd

# Determine whether undisturbed upstream conditions meet criteria for foehn
if wd >= 240 and wd <=300 and ws >= 4.0:

# Isentrope method
for i in timestep:
	Z1 = theta[timestep, lev_2000, lat_range, lon]
	transect_lats, transect_lons = [lat_range, lon_range] # define ~40 km transect on the Eastern side, i.e. over Larsen, over which to measure Z3
	# Find indices where theta is equal to theta at Z1 over the 40 km transect
	theta_idx = np.where(np.isclose(Z1, theta[i, :, lat_range, lon_range]))
	# Find minimum height over these indices
	Z2 = min(theta.coord('Height')[theta_idx]) # find model level
	Z3 = Z1 - Z2
	if Z3 >= 1000 for 2+ timesteps, then = foehn





between each 3-hourly timestep,
if:
1) mean flow is cross-peninsula and westerly (directional and speed conditions)
	2a) Fr > 0.9 (Dan Bannister) is above threshold / OR:
	2b) surface met changes sufficiently
foehn_stats.append(1)
else:
	foehn_stats.append(0)

	Should end up with a list of len-1 time series full of 1s and 0s (i.e. foehn event or not)

	# Calculate total frequency of foehn in time series
	# Calculate mean number of foehn in each year
	# Plot total number of foehn in each year (bar chart)
	# Calculate and plot Mean number of foehn in each season
	# Plot mean number of foehn per month, whole time series

	Correlations between:

	- foehn events + turbulent fluxes (seasonal/overall) - sim. 1st melt chapter
	- foehn events + melt rates (seasonal/overall) - sim. 1st melt chapter
	- foehn events and increased SWdown / decreased cloud cover (cloud clearance)
