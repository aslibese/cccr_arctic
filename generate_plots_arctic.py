#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on 04 Oct 2024

This script computes the Arctic temperature trend and amplification for the periods 1950-2023 and 1979-2023 using the average of four observational datasets
and creates a time series plot and polar trend and amplification plots.

"""

from util_arctic import fix_coords, weighted_avg, perform_linear_regression, apply_grid_cell_regression
from util_plots import plot_average_ArcticTrend, plot_average_AA

import numpy as np  
import xarray as xr 
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import cartopy.crs as ccrs # crs: coordinate reference system
from pathlib import Path


def main():
    home = Path("~").expanduser()
    obs_ave_filepath = home.joinpath("my_dir", "cccr/obs_ave.nc")
    
    ds = xr.open_dataset(obs_ave_filepath)
    ds = fix_coords(ds)

    # define boundaries for Canada
    canada_south = 41.68  
    canada_north = 83.11  
    canada_west = 360.0-141.00  
    canada_east = 360.0-52.62 

    # define boundaries for the Canadian Arctic
    ca_arc_south = 66.5
    ca_arc_north = 83.11
    ca_arc_west = 360.0-141.0
    ca_arc_east = 360.0-52.62

    ann_mean_temp_anom = ds['temperature'].groupby('time.year').mean('time')

    """
    Compute the weighted average of temperature anomalies for the regions and the warming trend
    """
    # compute the weighted average of temperate anomalies for the regions
    temp_canada = weighted_avg(ann_mean_temp_anom, lat_bound_s=canada_south, lat_bound_n=canada_north, lon_bound_w=canada_west, lon_bound_e=canada_east)
    temp_ca_arc = weighted_avg(ann_mean_temp_anom, lat_bound_s=ca_arc_south, lat_bound_n=ca_arc_north, lon_bound_w=ca_arc_west, lon_bound_e=ca_arc_east)
    temp_glob = weighted_avg(ann_mean_temp_anom)

    # compute the warming trend for 1979-2023
    canada_trend, canada_intercept, _ = perform_linear_regression(temp_canada.sel(time=slice(1979, 2023)).values)
    y_fit_canada = np.arange(len(temp_canada.sel(time=slice(1979, 2023)))) * canada_trend + canada_intercept

    ca_arc_trend, ca_arc_intercept, _ = perform_linear_regression(temp_ca_arc.sel(time=slice(1979, 2023)).values)
    y_fit_ca_arc = np.arange(len(temp_ca_arc.sel(time=slice(1979, 2023)))) * ca_arc_trend + ca_arc_intercept

    glob_trend, glob_intercept, _ = perform_linear_regression(temp_glob.sel(time=slice(1979, 2023)).values) 
    y_fit_glob = np.arange(len(temp_glob.sel(time=slice(1979, 2023)))) * glob_trend + glob_intercept

    print(f"Canada warming trend for 1979-2023: {np.round(canada_trend, 3)}")
    print(f"Canadian Arctic warming trend for 1979-2023: {np.round(ca_arc_trend, 3)}")
    print(f"Global warming trend for 1979-2023: {np.round(glob_trend, 3)}")


    """
    Figure1: time series plot of temperature anomalies for the global, Canada, and Canadian Arctic regions
    """
    plt.figure(figsize=(10, 6))

    colours = {'canada': 'red', 'ca_arc': 'blue', 'global': 'gray'}

    plt.plot(temp_canada.year, temp_ca_arc, colours['ca_arc'], alpha=1)
    plt.plot(temp_canada.sel(time=slice(1979, 2023)).year, y_fit_canada, colours['ca_arc'], alpha=1)

    plt.plot(temp_ca_arc.year, temp_ca_arc, colours['canada'], alpha=1)
    plt.plot(temp_ca_arc.sel(time=slice(1979, 2023)).year, y_fit_ca_arc, colours['canada'], alpha=1)

    plt.plot(temp_glob.year, temp_glob, colours['global'], alpha=0.5)
    plt.plot(temp_glob.sel(time=slice(1979, 2023)).year, y_fit_glob, colours['global'], alpha=0.5)

    plt.ylabel('Temperature Anomaly [°C]', fontsize=16)
    plt.legend(loc='upper left', fontsize=14)
    plt.grid(True, linestyle='-') 
    plt.xticks(np.arange(1950, 2021, 10))
    plt.tick_params(axis='both', labelsize=16)
    plt.xlim(1950, 2025)

    output_dir = home.joinpath("my_dir", "cccr/figures")
    plt.savefig(f'{output_dir}/obs_time_series.png', dpi=300)


    """
    Compute Arctic temperature trend and amplification for each grid cell
    """
    # compute the slope (trend) and p-value for each grid cell for 1950-2023
    trend_1950, p_value_1950 = apply_grid_cell_regression(ann_mean_temp_anom.sel(time=slice(1950, 2023)))
    # mask trend values where p_value is not statistically significant
    trend_1950_masked = trend_1950.where(p_value_1950 < 0.05, np.nan)

    # repeat for 1979-2023
    trend_1979, p_value_1979 = apply_grid_cell_regression(ann_mean_temp_anom.sel(time=slice(1979, 2023)))
    trend_1979_masked = trend_1979.where(p_value_1979 < 0.05, np.nan)

    # global trend for 1950-2023
    glob_trend_1950, _, _ = perform_linear_regression(temp_glob.sel(time=slice(1979, 2023)).values) 

    # compute amplification for each grid cell
    amplification_1950 = trend_1950 / glob_trend_1950
    amplification_1979 = trend_1979 / glob_trend 

    """
    Figure2: Polar map of Arctic temperature trend and amplification 
    """

    fig2 = plt.subplots(figsize=(10, 6))
    gs = GridSpec(2, 2, figure=fig2, hspace=0.3, wspace=0.3)

    # 1950-2023 temp trend (top left)
    ax1 = fig2.add_subplot(gs[0, 0], projection=ccrs.NorthPolarStereo(central_longitude=-100))
    plot_average_ArcticTrend(trend_1950_masked, lower_boundary=66.5, ax=ax1)

    # 1950-2023 amplification (top right)
    ax2 = fig2.add_subplot(gs[0, 1], projection=ccrs.NorthPolarStereo(central_longitude=-100))
    plot_average_AA(amplification_1950, lower_boundary=66.5, ax=ax2)

    # 1979-2023 temp trend (bottom left)
    ax3 = fig2.add_subplot(gs[1, 0], projection=ccrs.NorthPolarStereo(central_longitude=-100))
    plot_average_ArcticTrend(trend_1979_masked, lower_boundary=66.5, ax=ax3)

    # 1979-2023 amplification (bottom right)
    ax4 = fig2.add_subplot(gs[1, 1], projection=ccrs.NorthPolarStereo(central_longitude=-100))
    plot_average_AA(amplification_1979, lower_boundary=66.5, ax=ax4)


    ax1.text(0, 1.1, 'a)', transform=ax1.transAxes, fontsize=16, fontweight='bold', va='top', ha='right')
    ax2.text(0, 1.1, 'b)', transform=ax2.transAxes, fontsize=16, fontweight='bold', va='top', ha='right')
    ax3.text(0, 1.1, 'c)', transform=ax3.transAxes, fontsize=16, fontweight='bold', va='top', ha='right')
    ax4.text(0, 1.1, 'd)', transform=ax4.transAxes, fontsize=16, fontweight='bold', va='top', ha='right')

    plt.savefig(f'{output_dir}/obs_polar_maps.png', dpi=300)


if __name__ == "__main__":
    main()