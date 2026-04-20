"""
ERA5 Wind Data Interface Module for SAR-based Wake Detection

This module provides utilities for loading and processing ERA5 reanalysis
wind data (10 m wind components) for use in the R2G2 wake detection framework.

Main functionalities include:
1. Loading ERA5 datasets in NetCDF (.nc) or GRIB (.grib) format.
2. Extracting zonal (u10) and meridional (v10) wind components.
3. Computing wind speed magnitude and wind direction.
4. Providing spatial and temporal averaging of wind fields.

The derived wind direction is used to constrain the wake propagation
direction and define the search region in the wake detection algorithm,
while wind speed is used for wake–upstream comparison.

Notes
-----
- Wind direction is computed from ERA5 u and v components using an arctangent
  formulation and converted to degrees.
- Longitude is assumed to follow [0, 360] convention.
- This module relies on xarray and dask for efficient large-scale data handling.

Dependencies
------------
- xarray
- dask
- numpy

Part of:
R2G2 (Region-constrained + Region-growing) wake detection framework.
"""

import xarray as xr
import os
import dask.array as da
import math


class ERA5WindSpeed(object):
    """
    ERA5 wind data handler for retrieving wind speed and direction.

    This class loads ERA5 reanalysis data (NetCDF or GRIB format) and provides
    methods to compute wind speed and direction at specified locations and times.
    """

    def __init__(self, file_path, file_name):
        """
        Parameters
        ----------
        file_path : str
            Directory containing the ERA5 data file.
        file_name : str
            Name of the ERA5 file ('.nc' or '.grib').
        """
        self.path = file_path
        self.name = file_name

        # Load dataset
        self.load_data()

        # Extract variables
        self.u10n = self.dataset.u10n   # zonal wind (m/s)
        self.v10n = self.dataset.v10n   # meridional wind (m/s)
        self.longitude = self.dataset.longitude.data
        self.latitude = self.dataset.latitude.data
        self.time = self.dataset.time

    def load_data(self):
        """
        Load ERA5 dataset based on file format.

        Supported formats:
        - NetCDF (.nc)
        - GRIB (.grib)
        """
        if self.name.split('.')[-1] == 'grib':
            self.dataset = xr.open_dataset(
                os.path.join(self.path, self.name),
                engine='cfgrib'
            )
        elif self.name.split('.')[-1] == 'nc':
            self.dataset = xr.open_dataset(
                os.path.join(self.path, self.name)
            )
        else:
            raise RuntimeError(
                'The current version does not support the %s format.'
                % (self.name.split('.')[-1])
            )

    def get_speed_direction(self, time=None, latitude=(-90.0, 90.0), longitude=(0.0, 360.0)):
        """
        Compute wind speed and direction at a given time and location.

        Parameters
        ----------
        time : str or tuple, optional
            Target time (ISO format). If None, the first available time is used.
        latitude : float or tuple
            Latitude or latitude range.
        longitude : float or tuple
            Longitude or longitude range.

        Returns
        -------
        speed : xarray.DataArray
            Wind speed (m/s).
        direction : xarray.DataArray
            Wind direction (degrees, meteorological convention).
        """
        if time is None:
            time = (self.time[0], self.time[1])

        # Select data at specified time and location
        u10n_target = self.u10n.sel(time=time, latitude=latitude, longitude=longitude)
        v10n_target = self.v10n.sel(time=time, latitude=latitude, longitude=longitude)

        # Compute wind direction (degrees)
        radians = xr.apply_ufunc(da.arctan2, u10n_target, v10n_target, dask='allowed')
        direction = radians * 180 / math.pi + 180

        # Compute wind speed (magnitude)
        speed = xr.apply_ufunc(
            da.sqrt,
            u10n_target ** 2 + v10n_target ** 2,
            dask='allowed'
        )

        return speed, direction

    def get_average_speed(self, time=None, latitude=(-90.0, 90.0), longitude=(0.0, 360.0)):
        """
        Compute time-averaged wind components and speed over a region.

        Parameters
        ----------
        time : tuple, optional
            Time range (start, end).
        latitude : tuple
            Latitude range (min, max).
        longitude : tuple
            Longitude range (min, max).

        Returns
        -------
        u_mean : xarray.DataArray
            Mean zonal wind (m/s).
        v_mean : xarray.DataArray
            Mean meridional wind (m/s).
        speed_mean : xarray.DataArray
            Mean wind speed (m/s).
        """
        if time is None:
            time = (self.time[0], self.time[1])

        # Select data within spatial and temporal ranges
        u10n_target = self.u10n.sel(
            time=slice(time[0], time[1]),
            latitude=slice(latitude[1], latitude[0]),
            longitude=slice(longitude[0], longitude[1])
        )
        v10n_target = self.v10n.sel(
            time=slice(time[0], time[1]),
            latitude=slice(latitude[1], latitude[0]),
            longitude=slice(longitude[0], longitude[1])
        )

        # Compute wind speed magnitude
        w10n_target = xr.apply_ufunc(
            da.sqrt,
            u10n_target ** 2 + v10n_target ** 2,
            dask='allowed'
        )

        return (
            u10n_target.mean(dim='time'),
            v10n_target.mean(dim='time'),
            w10n_target.mean(dim='time')
        )