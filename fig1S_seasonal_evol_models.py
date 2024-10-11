#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on 06 Oct 2024

This script generates a plot for the seasonal evolution of the temperature anomaly in the Canadian Arctic region for each CMIP6 and the multi-model mean.

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
tarctic_moving_diff = ds['tarctic_moving_diff']  # (model, month, year)
models = ds['model'].values  

month_indices = [5, 6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4]  # June to May (0-based indexing)
month_names = ['Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May']

# reorder MMM data
MMM_tarctic_reordered = MMM_tarctic.isel(month=month_indices)
MMM_tarctic_reordered = MMM_tarctic_reordered.assign_coords(month=np.arange(1, 13))

# reorder model data
tarctic_moving_diff_reordered = tarctic_moving_diff.isel(month=month_indices)
tarctic_moving_diff_reordered = tarctic_moving_diff_reordered.assign_coords(month=np.arange(1, 13))



# plot settings
n_models = len(models)
n_rows = 5  
n_cols = 5
fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 16), sharex=True, sharey=True)
axes = axes.flatten()

levels = np.arange(0, 8.5, 0.5)
cmap = 'Reds'

# select the years every 5 years between 1990 and 2085 for the dots
dot_start_year = 1990
dot_end_year = 2085
dot_years = np.arange(dot_start_year, dot_end_year + 1, 5)


# plot each model's data
for i, model in enumerate(models):
    ax = axes[i]
    model_data = tarctic_moving_diff_reordered.sel(model=model)
    year_values = model_data['year'].values
    cf = ax.contourf(year_values, model_data['month'], model_data, levels=levels, cmap=cmap, extend='both')

    # month of maximum value for each year
    max_month_indices = model_data.where(~np.isnan(model_data), other=-np.inf).argmax(dim='month') + 1  
    max_months_for_dots = max_month_indices.sel(year=dot_years)
    ax.plot(max_months_for_dots['year'], max_months_for_dots, color='grey', linestyle='None', marker='o', markersize=7.5)

    ax.set_yticks(np.arange(1, 13))
    ax.set_yticklabels(month_names)
    ax.set_title(f'{model}', fontsize=10)


# plot MMM in the last subplot
ax = axes[23]
cf = ax.contourf(year_values, MMM_tarctic_reordered['month'], MMM_tarctic_reordered, levels=levels, cmap=cmap, extend='both')
# plot the dots
max_month_indices_MMM = MMM_tarctic_reordered.argmax(dim='month') + 1  # months from 1 to 12
max_months_for_dots_MMM = max_month_indices_MMM.sel(year=dot_years)
ax.plot(max_months_for_dots_MMM['year'], max_months_for_dots_MMM, color='grey', linestyle='None', marker='o', markersize=7.5)

ax.set_yticks(np.arange(1, 13))
ax.set_yticklabels(month_names)
ax.set_title('Multi-Model Mean', fontsize=10)


# remove the extra subplot 
for j in range(len(models) + 1, len(axes)):
    fig.delaxes(axes[j])

# shared axis settings
for ax in axes[:24]:
    ax.set_xlim(1966, 2085)
    ax.tick_params(axis='both', which='major', labelsize=8)
    ax.set_xticks(np.arange(1975, 2076, 25))

# colourbar
fig.subplots_adjust(right=0.92)
cbar_ax = fig.add_axes([0.93, 0.15, 0.02, 0.7])
cbar = fig.colorbar(cf, cax=cbar_ax, orientation='vertical', ticks=levels)
cbar.set_label('Temperature Anomaly (K)', fontsize=12)
cbar.ax.tick_params(labelsize=10)

plt.tight_layout(rect=[0, 0, 0.90, 1])  # leave space for colorbar

output_file = home.joinpath("my_dir", "cccr_arctic/figures/tas_seasonal_evol_models.png")
plt.savefig(output_file)
print(f'Plot saved to: {output_file}')
