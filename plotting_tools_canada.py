# -*- coding: utf-8 -*-
# Name: Asli Bese
# Scinet username: tmp_abese

import numpy as np   
import matplotlib.pyplot as plt  
import matplotlib.path as mpath    
import matplotlib.ticker as mticker
from matplotlib import cm
from matplotlib.colors import ListedColormap
import cartopy.feature as cfeature
import cartopy.crs as ccrs # crs: coordinate reference system
from cartopy.util import add_cyclic_point

# function to create a customized background plot 
def plot_background(ax, lower_boundary):

	# set the background colour of the plot area to white 
	ax.patch.set_facecolor('w')
	# change the border of the plot to black 
	ax.spines['geo'].set_edgecolor('k')

	# set the geographical extent of the plot to cover lon from -180 to 180
	# and lat from 50 to 90
	# plate carrée map projection is an equidistant cylindrical projection with the standard parallel located at the equator
	ax.set_extent([-180, 180, 50, 90], crs=ccrs.PlateCarree())
	# add coastline features with a scale of '110m'; zorder=10 ensures that coastlines are drawn above other layers
	ax.add_feature(cfeature.COASTLINE.with_scale('110m'), zorder=10)

	# add gridlines to show the Arctic boundary
	gl = ax.gridlines(linewidth=2, color='k', alpha=0.5, linestyle='--')
	gl.n_steps = 100 
	# set the Arctic lower boundary based on the argument
	gl.ylocator = mticker.FixedLocator([float(lower_boundary)])
	gl.xlocator = mticker.FixedLocator([])
	
	theta = np.linspace(0, 2*np.pi, 100)
	center, radius = [0.5, 0.5], 0.5

	# construct a 'Path' object with the vertices of the circle set this path as the boundary of the plot
	# to focus on the Arctic region
	verts = np.vstack([np.sin(theta), np.cos(theta)]).T
	circle = mpath.Path(verts * radius + center)

	ax.set_boundary(circle, transform=ax.transAxes)
	return ax


# function to plot the Arctic temperature trend as a polar map
def plot_multipanel_ArcticTrend(datasets, lower_boundary, subplot_labels):
	n_datasets = len(datasets)
	fig, axs = plt.subplots(1, n_datasets, figsize=(5*n_datasets, 5), subplot_kw={'projection': ccrs.NorthPolarStereo()}, squeeze=False)
	axs = axs.flatten()
	
	# set up colour range and levels for the countour plot (min, max values of the temperature anomaly trends to display)
	cmin, cmax, incr = -1.5, 1.5, 0.25 
	levels = np.arange(cmin, cmax + incr, incr) 
	cmap = 'RdYlBu_r'

	for i, ds in enumerate(datasets):
		ax = axs[i]
		data = ds['slope'] * 10  # 'slope' is per year; convert to per decade
		data_cyclic, lon_cyclic = add_cyclic_point(data, coord=ds['lon']) #  use add_cyclic_point to avoid a gap at the longtitude seam

		# create filled contour plot
		contour = ax.contourf(lon_cyclic, ds['lat'], data_cyclic, levels=levels, cmap=cmap, extend='both', transform=ccrs.PlateCarree())
		ax.set_title(subplot_labels[i], fontsize=14)
		plot_background(ax, lower_boundary) #  call the plot_background() function to create a background plot 
			
	# add a shared colourbar 
	fig.subplots_adjust(bottom=0.25)
	cbar_ax = fig.add_axes([0.32, 0.1, 0.4, 0.05]) # [left, bottom, width, height]
	cbar = fig.colorbar(contour, cax=cbar_ax, orientation='horizontal')
	cbar.ax.tick_params(labelsize=14)
	cbar.set_label('Temperature trend [°C per decade]', fontsize=16)
	labels = np.arange(cmin, cmax + incr, incr * 3)
	cbar.set_ticks(labels)



# function to plot the Arctic Amplification (AA) as a polar map
def plot_multipanel_AA(datasets, lower_boundary, subplot_labels):
	n_datasets = len(datasets)
	fig, axs = plt.subplots(1, n_datasets, figsize=(5*n_datasets, 5), subplot_kw={'projection': ccrs.NorthPolarStereo()}, squeeze=False)
	axs = axs.flatten()

	# generate base colourmaps
	Reds = cm.get_cmap('Reds', 12) # make 12 discrete shades of red 
	Blues = cm.get_cmap('Blues', 10) # make 10 discrete shades of blue
	
	# modify colourmap colours
	newcolours = Reds(np.linspace(0,1,12)) # array created from 'Reds'
	newcolours[0,:] = Blues(0.7) # replace the first index with dark blue shade: AA = 0-0.5
	newcolours[1,:] = Blues(0.2) # replace the second index with lighter blue shade: AA = 0.5-1

	# loop to create a stepped effect in the colour transition for AA values larger than 1
	for i in range(2,12,2):
		newcolours[i] = newcolours[i+1, :]
	
	newcolours = np.vstack((newcolours, newcolours[-1], newcolours[-1])) # duplicate the last colour twice (highest AA values)
	newcolours[-2:, :3] = newcolours[-2:, :3] * 0.4 # set the last two colours as distinct colours by reducing the brigthness

	new_cmap = ListedColormap(newcolours) # create a new colourmap from 'newcolours'

	# set out-of-range values below and above the colourmap's range using shades from the 'Blues' and 'Greys' colourmaps
	new_cmap.set_under(cm.get_cmap('Blues')(0.99), 1.0) 
	new_cmap.set_over(cm.get_cmap('Greys')(0.75), 1.0)

	# display AA values from 0 to 7 with 0.5 increments
	cmin, cmax, incr = 0, 7, 0.25
	levels = np.arange(cmin, cmax + incr, incr)

	for i, ds in enumerate(datasets):
		ax = axs[i]
		data = ds['amplification']  
		data_cyclic, lon_cyclic = add_cyclic_point(data, coord=ds['lon'])

		contour = ax.contourf(lon_cyclic, ds['lat'], data_cyclic, levels=levels, cmap=new_cmap, extend='both', transform=ccrs.PlateCarree())		
		ax.set_title(subplot_labels[i], fontsize=14)
		plot_background(ax, lower_boundary)	

	# add a shared colorbar
	fig.subplots_adjust(bottom=0.25)
	cbar_ax = fig.add_axes([0.32, 0.1, 0.4, 0.05]) # [left, bottom, width, height]
	cbar = fig.colorbar(contour, cax=cbar_ax, orientation='horizontal')
	cbar.ax.tick_params(labelsize=14)
	cbar.set_label('Local amplification', fontsize=16)
	labels = np.arange(cmin, cmax + incr, 1)
	cbar.set_ticks(labels)


