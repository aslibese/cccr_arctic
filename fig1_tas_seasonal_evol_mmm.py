#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on 06 Oct 2024

This script generates a plot for the seasonal evolution of the multi-model mean temperature anomaly in the Canadian Arctic region.

Code adapted from: https://doi.org/10.5281/zenodo.6965473

author: @aslibese
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from pathlib import Path

home = Path("~").expanduser() 
in_file = home.joinpath("data_dir", "CMIP6/cccr_arctic/tas_cmip6_month_v3.nc")
ds = xr.open_dataset(in_file)

MMM_tarctic = ds['MMM_tarctic']  # dimensions: (month, year)

# rearrange months to start from June and end in May
month_indices = [5, 6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4]  # June to May (0-based indexing)
month_names = ['Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May']

# reorder the data
MMM_tarctic_reordered = MMM_tarctic.isel(month=month_indices)
# create new month coordinate from 1 to 12
MMM_tarctic_reordered = MMM_tarctic_reordered.assign_coords(month=np.arange(1, 13))

year_values = MMM_tarctic_reordered['year'].values


# plot
fig, ax = plt.subplots(figsize=(12, 8))

levels = np.arange(0, 8.5, 0.5) # define levels for contour

cf = ax.contourf(year_values, MMM_tarctic_reordered['month'], MMM_tarctic_reordered, levels=levels, cmap='Reds', extend='both')

ax.set_yticks(np.arange(1, 13))
ax.set_yticklabels(month_names)

# compute the month of maximum value for each year
max_month_indices = MMM_tarctic_reordered.argmax(dim='month') + 1  # months from 1 to 12

# select the years every 5 years between 1990 and 2085
# 1990 is the first year where the running mean doesn’t overlap with the reference period
# 2085 is the last year to compute a 30-year running mean because we need data from 2070 to 2100 to compute the mean for 2085
dot_start_year = 1990
dot_end_year = 2085
dot_years = np.arange(dot_start_year, dot_end_year + 1, 5)

# get the max months for the selected years
max_months_for_dots = max_month_indices.sel(year=dot_years)

# plot the dots
ax.plot(max_months_for_dots['year'], max_months_for_dots, color='grey', linestyle='None', marker='o', markersize=7.5)

ax.set_ylim(0.8, 12.2)
ax.set_xlim(1966, 2085)

ax.tick_params(axis='both', which='major', labelsize=12)
ax.set_xticks(np.arange(1975, 2076, 25))

# colourbar
cbar = fig.colorbar(cf, ax=ax, orientation='vertical', pad=0.02, ticks=levels)
cbar.set_label('Temperature Anomaly (K)', fontsize=14)
cbar.ax.tick_params(labelsize=12)

plt.tight_layout()

output_file = home.joinpath("my_dir", "cccr_arctic/figures/tas_seasonal_evol_MMM.png")
plt.savefig(output_file)
print(f'Plot saved to: {output_file}')

