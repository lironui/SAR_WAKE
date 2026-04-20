"""
Main Processing Pipeline for SAR-based Wind Farm Wake Detection

This script implements the full R2G2 wake detection workflow for Sentinel-1 SAR
wind-field products. The processing pipeline includes:

1. Reading Sentinel-1-derived wind speed data.
2. Locating offshore wind farms from a configuration file.
3. Estimating wind direction from ERA5 reanalysis data.
4. Constructing wind farm masks using gradient-based turbine localization.
5. Building downstream wake search regions and upstream reference regions.
6. Extracting wake areas using the constrained region-growing algorithm.
7. Exporting wake masks, visual products, and scene-level statistics.

The script also supports multiprocessing for batch processing of multiple SAR scenes.

Dependencies
------------
- numpy
- opencv-python
- matplotlib
- Pillow
- esa_snappy
- region_grow
- search_range
- wind_direction.wind_direction
- yml_io

Part of
-------
R2G2 (Region-constrained + Region-growing) wake detection framework.
"""

import gc
import os
import csv
import math
import traceback
import multiprocessing
from multiprocessing import Pool

import cv2
import numpy as np
import matplotlib.pyplot as plt
from numpy import array

# Custom modules
from region_grow import Point, region_grow
from wind_direction.wind_direction import ERA5WindSpeed
from esa_snappy import ProductIO, GeoPos, ProductData
from search_range import SearchList, UpstreamList
from yml_io import YamlHandler


# =============================================================================
# Global Configuration
# =============================================================================
ERA5_BASE_PATH = "/media/u2370656/LaCie1/ERA5"
WAKE_EFFECT_BASE = "/media/u2370656/LaCie1/wake_effect"
WIND_SPEED_BASE = "/media/u2370656/LaCie1/wind_speed"
WAKE_VISUAL_BASE = "/media/u2370656/LaCie1/wake_visual"
WAKE_MASK_BASE = "/media/u2370656/LaCie1/wake_mask"

DEFAULT_OUTPUT_CSV = "wake_statistics_50.csv"

# Recommended to keep this lower than 32 for SNAP/Java memory stability
DEFAULT_NUM_PROCESSES = 8


class SentinelWakeDetection(object):
    """
    Main class for wake detection from a single Sentinel-1 scene.

    Parameters
    ----------
    input_path : str
        Path to the input Sentinel-1 product.
    output_path : str
        Path to the output product.
    wind_farm_name : str
        Wind farm identifier in the YAML configuration file.
    search_range : tuple, optional
        Downstream wake search range (length, width).
    upstream_range : tuple, optional
        Upstream reference search range (length, width).
    farm_radius : int, optional
        Search radius around the wind farm center for turbine mask generation.
    farm_search_range : int, optional
        Pixel neighborhood size for turbine checking.
    shift_search_range : int, optional
        Pixel neighborhood size for seed-point shifting.
    """

    def __init__(
        self,
        input_path,
        output_path,
        wind_farm_name,
        search_range=(200, 75),
        upstream_range=(30, 50),
        farm_radius=50,
        farm_search_range=1,
        shift_search_range=1,
    ):
        self.sentinel_1 = ProductIO.readProduct(input_path)
        self.wind_farm_name = wind_farm_name
        self.search_range = search_range
        self.output_path = output_path
        self.upstream_range = upstream_range
        self.farm_radius = farm_radius
        self.farm_search_range = farm_search_range
        self.shift_search_range = shift_search_range

        # Thresholds computed during processing
        self.gradient_threshold = None
        self.speed_threshold = None

    # -------------------------------------------------------------------------
    # Basic utilities
    # -------------------------------------------------------------------------
    def locate_wind_farm(self):
        """
        Read wind farm geographic coordinates from YAML configuration.

        Returns
        -------
        list
            [latitude, longitude]
        """
        wind_farm_yml = YamlHandler("wind_farms.yml").read_yaml()["wind_farms"]
        target_farm = wind_farm_yml[self.wind_farm_name]
        latitude = target_farm["coordinates"]["latitude"]
        longitude = target_farm["coordinates"]["longitude"]
        return [latitude, longitude]

    def geo_to_pixel(self, latitude, longitude):
        """
        Convert geographic coordinates to pixel coordinates in the Sentinel-1 scene.

        Parameters
        ----------
        latitude : float
            Geographic latitude.
        longitude : float
            Geographic longitude.

        Returns
        -------
        tuple
            (row, col) pixel coordinates
        """
        geo_position = GeoPos(latitude, longitude)
        pixel_position = self.sentinel_1.getSceneGeoCoding().getPixelPos(geo_position, None)
        return round(pixel_position.getY()), round(pixel_position.getX())

    def farm_location_check(self):
        """
        Check whether the target wind farm is inside the current SAR scene.

        Returns
        -------
        bool
            True if the farm is inside the scene, otherwise False.
        """
        latitude, longitude = self.locate_wind_farm()
        geo_position = GeoPos(latitude, longitude)
        pixel_position = self.sentinel_1.getSceneGeoCoding().getPixelPos(geo_position, None)
        return not math.isnan(pixel_position.getX())

    # -------------------------------------------------------------------------
    # SAR bands and image preparation
    # -------------------------------------------------------------------------
    def add_band(self, wake_mask, upstream_mask, wind_farm_mask):
        """
        Write wake detection results into the output product.

        Encoding
        --------
        1 : wake region
        2 : upstream reference region
        3 : wind farm region
        """
        sigma_band = self.sentinel_1.getBand("Sigma0_VV")
        sigma_band.readRasterDataFully()

        sar_pixel_ori = sigma_band.getRasterData().getElems()
        sar_pixel = array(sar_pixel_ori).reshape(
            sigma_band.getRasterWidth(), sigma_band.getRasterHeight()
        )

        wake_pixel = array(
            wake_mask + upstream_mask * 2 + wind_farm_mask * 3
        ).reshape(sigma_band.getRasterWidth(), sigma_band.getRasterHeight())

        wake_band = self.sentinel_1.getBand("wake_effect")
        wake_band.writePixels(0, 0, wake_pixel.shape[0], wake_pixel.shape[1], wake_pixel)
        sigma_band.writePixels(0, 0, sigma_band.getRasterWidth(), sigma_band.getRasterHeight(), sar_pixel)

    def get_pixel(self):
        """
        Read the SAR-derived wind speed band and convert it to a 2D NumPy array.

        Returns
        -------
        ndarray
            Wind speed image.
        """
        wind_band = self.sentinel_1.getBand("wind_speed")
        wind_band.readRasterDataFully()

        wind_pixel_ori = wind_band.getRasterData().getElems()
        wind_pixel = array(wind_pixel_ori).reshape(
            wind_band.getRasterWidth(), wind_band.getRasterHeight()
        )
        wind_band.writePixels(0, 0, wind_band.getRasterWidth(), wind_band.getRasterHeight(), wind_pixel)

        # Match original orientation used in downstream processing
        wind_pixel_img = array(wind_pixel_ori).reshape(wind_pixel.shape[1], wind_pixel.shape[0])
        return wind_pixel_img

    def calculate_gradient(self, wind_pixel_img):
        """
        Compute the image gradient using the Prewitt operator.

        Parameters
        ----------
        wind_pixel_img : ndarray
            Input wind speed image.

        Returns
        -------
        ndarray
            Gradient magnitude image.
        """
        max_value = wind_pixel_img.max()
        if max_value <= 0:
            return np.zeros_like(wind_pixel_img, dtype=np.uint8)

        kernelx = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=int)
        kernely = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=int)

        normalized = (wind_pixel_img * (255 / max_value)).astype(np.uint8)
        x_grad = cv2.filter2D(normalized, cv2.CV_16S, kernelx)
        y_grad = cv2.filter2D(normalized, cv2.CV_16S, kernely)

        abs_x = cv2.convertScaleAbs(x_grad)
        abs_y = cv2.convertScaleAbs(y_grad)
        prewitt = cv2.addWeighted(abs_x, 0.5, abs_y, 0.5, 0)

        return prewitt

    # -------------------------------------------------------------------------
    # ERA5 wind data
    # -------------------------------------------------------------------------
    def get_era_data(self, wind_farm_location):
        """
        Retrieve ERA5 wind speed and direction for the SAR acquisition time.

        Parameters
        ----------
        wind_farm_location : list
            [latitude, longitude]

        Returns
        -------
        tuple
            speed, direction, date_str, time_str
        """
        latitude, longitude = wind_farm_location

        product_name = self.sentinel_1.getName()
        date, time = product_name.split("_")[4].split("T")

        data_name = f"wind_speed_{date[0:4]}_{date[4:6]}.nc"
        wind_product = ERA5WindSpeed(ERA5_BASE_PATH, data_name)

        wind_time = f"{date[0:4]}-{date[4:6]}-{date[6:8]}T{time[0:2]}:00:00.000000000"
        longitude = np.mod(longitude, 360)

        speed, direction = wind_product.get_speed_direction(
            wind_time,
            math.floor(latitude * 4) / 4,
            math.floor(longitude * 4) / 4,
        )

        return (
            speed,
            direction,
            f"{date[0:4]}.{date[4:6]}.{date[6:8]}",
            f"{time[0:2]}:{time[2:4]}:{time[4:6]}",
        )

    # -------------------------------------------------------------------------
    # Wind farm mask generation
    # -------------------------------------------------------------------------
    def determine_turbine(self, position, gradient, search_range, gradient_threshold):
        """
        Determine whether turbine-like pixels exist in a local neighborhood.

        Parameters
        ----------
        position : list
            [row, col] position.
        gradient : ndarray
            Gradient image.
        search_range : list
            Pixel offsets for neighborhood search.
        gradient_threshold : float
            Threshold for turbine identification.

        Returns
        -------
        bool
            True if a turbine-like signature is detected.
        """
        farm_x, farm_y = position
        search_area = [[farm_x + i, farm_y + j] for i in search_range for j in search_range]
        gradient_shape = gradient.shape

        for element in search_area:
            if element[0] in range(0, gradient_shape[0]) and element[1] in range(0, gradient_shape[1]):
                if gradient[element[0], element[1]] > gradient_threshold:
                    return True
        return False

    def generate_wind_farm_mask(self, wind_farm_location, wind_pixel_img, gradient, search_range=1, show_mask=False):
        """
        Generate a binary mask of the wind farm area by region expansion around
        the known farm location.

        Parameters
        ----------
        wind_farm_location : list
            [latitude, longitude]
        wind_pixel_img : ndarray
            Wind speed image.
        gradient : ndarray
            Gradient image.
        search_range : int, optional
            Neighborhood radius for turbine checking.
        show_mask : bool, optional
            Whether to display the mask.

        Returns
        -------
        ndarray
            Binary wind farm mask.
        """
        connects = [(0, -1), (1, 0), (0, 1), (-1, 0)]
        wind_farm_mask = np.zeros(wind_pixel_img.shape)
        wind_farm_list = []

        search_range = [i for i in range(-search_range, search_range + 1)]
        latitude, longitude = wind_farm_location

        farm_x, farm_y = self.geo_to_pixel(latitude, longitude)

        farm_x = min(farm_x, wind_pixel_img.shape[0] - 1)
        farm_y = min(farm_y, wind_pixel_img.shape[1] - 1)

        wind_farm_mask[farm_x, farm_y] = 1
        wind_farm_list.append((farm_x, farm_y))

        while wind_farm_list:
            x, y = wind_farm_list.pop(0)
            for dx, dy in connects:
                new_x, new_y = x + dx, y + dy

                if new_x in range(0, wind_pixel_img.shape[0]) and new_y in range(0, wind_pixel_img.shape[1]):
                    distance = np.sqrt((new_x - farm_x) ** 2 + (new_y - farm_y) ** 2)
                    if distance < self.farm_radius:
                        if (
                            self.determine_turbine(
                                [new_x, new_y],
                                gradient,
                                search_range,
                                gradient_threshold=self.gradient_threshold,
                            )
                            and wind_farm_mask[new_x, new_y] != 1
                        ):
                            wind_farm_list.append((new_x, new_y))
                            wind_farm_mask[new_x, new_y] = 1

        if show_mask:
            plt.imshow(wind_farm_mask)
            plt.show()

        return wind_farm_mask

    # -------------------------------------------------------------------------
    # Pixel shifting for wake/upstream seed construction
    # -------------------------------------------------------------------------
    @staticmethod
    def safe_slope_from_direction(direction_deg, eps=1e-6):
        """
        Convert wind direction to line slope while avoiding division by zero.

        Returns
        -------
        float
            Slope value used for search-region construction.
        """
        sin_val = math.sin(math.radians(direction_deg))
        cos_val = math.cos(math.radians(direction_deg))

        if abs(sin_val) < eps:
            sin_val = eps if sin_val >= 0 else -eps

        return cos_val / sin_val

    def shift_location_until_outside_turbine(
        self,
        wind_farm_location,
        direction,
        gradient,
        shift_step,
        search_range,
    ):
        """
        Shift a geographic point along the wind direction until it leaves the
        turbine neighborhood.

        Parameters
        ----------
        wind_farm_location : list
            [latitude, longitude]
        direction : float
            Wind direction in degrees.
        gradient : ndarray
            Gradient image.
        shift_step : int
            Step size for shifting.
        search_range : int
            Neighborhood radius for turbine checking.

        Returns
        -------
        tuple
            ([farm_x, farm_y], [seed_x, seed_y], shifted_latitude, shifted_longitude)
        """
        latitude, longitude = wind_farm_location
        pixel_shift = [0.5 / 111, 0.5 / (111 * abs(math.cos(math.radians(latitude))))]

        farm_x, farm_y = self.geo_to_pixel(latitude, longitude)
        seed_x, seed_y = farm_x, farm_y

        search_range = [i for i in range(-search_range, search_range + 1)]

        turbine_neighbor = True
        while turbine_neighbor:
            coordinate_shift = [i * shift_step for i in pixel_shift]
            latitude = latitude - math.cos(math.radians(direction)) * coordinate_shift[0]
            longitude = longitude - math.sin(math.radians(direction)) * coordinate_shift[1]

            seed_x, seed_y = self.geo_to_pixel(latitude, longitude)

            if not self.determine_turbine([seed_x, seed_y], gradient, search_range, self.gradient_threshold):
                turbine_neighbor = False

        return [farm_x, farm_y], [seed_x, seed_y], latitude, longitude

    def farm_location_shift(self, wind_farm_location, direction, gradient, shift_step=3, search_range=2):
        """
        Shift from farm center to a downstream seed point for wake detection.

        Returns
        -------
        tuple
            farm_pixel, seed_pixel, wake_direction
        """
        farm_pixel, seed_pixel, _, _ = self.shift_location_until_outside_turbine(
            wind_farm_location,
            direction,
            gradient,
            shift_step=shift_step,
            search_range=search_range,
        )

        farm_x, farm_y = farm_pixel
        seed_x, seed_y = seed_pixel
        shift_direction = 1 if seed_y > farm_y else -1

        return [farm_x, farm_y], [seed_x, seed_y], shift_direction

    def upstream_location_shift(self, wind_farm_location, direction, gradient, shift_step=-2, search_range=1):
        """
        Shift from farm center to an upstream seed point.

        Returns
        -------
        list
            [row, col] of upstream seed point
        """
        _, seed_pixel, _, _ = self.shift_location_until_outside_turbine(
            wind_farm_location,
            direction,
            gradient,
            shift_step=shift_step,
            search_range=search_range,
        )
        return seed_pixel

    # -------------------------------------------------------------------------
    # Search region generation
    # -------------------------------------------------------------------------
    def generate_search_list(
        self,
        farm_location,
        seed_location,
        direction,
        wind_farm_mask,
        image,
        wake_direction,
        search_range=(300, 50),
        show_mask=False,
    ):
        """
        Generate downstream candidate search pixels and region-growing seeds.

        Returns
        -------
        tuple
            search_list, search_mask, seeds
        """
        farm_x, farm_y = farm_location[0], farm_location[1]
        seed_x, seed_y = seed_location[0], seed_location[1]

        seeds = [Point(seed_x, seed_y)]
        for offset in (1, 3, 5):
            seeds.append(Point(seed_x + offset, seed_y + offset))
            seeds.append(Point(seed_x - offset, seed_y - offset))
            seeds.append(Point(seed_x - offset, seed_y + offset))
            seeds.append(Point(seed_x + offset, seed_y - offset))
            seeds.append(Point(seed_x, seed_y + offset))
            seeds.append(Point(seed_x + offset, seed_y))
            seeds.append(Point(seed_x, seed_y - offset))
            seeds.append(Point(seed_x - offset, seed_y))

        slope = self.safe_slope_from_direction(direction)

        search_object = SearchList(
            slope,
            farm_y,
            farm_x,
            wind_farm_mask,
            wake_direction,
            image=image,
            search_range=search_range,
        )
        search_list, search_mask = search_object.get_list()

        if show_mask:
            plt.imshow(search_mask)
            plt.show()

        return search_list, search_mask, seeds

    def generate_upstream_list(
        self,
        direction,
        farm_location,
        gradient_threshold,
        gradient,
        image,
        wind_farm_mask,
        wake_speed,
        upstream_direction,
        show_mask=False,
    ):
        """
        Generate upstream reference region and compute its mean wind speed.

        Returns
        -------
        tuple
            upstream_mask, upstream_speed
        """
        farm_x, farm_y = farm_location[0], farm_location[1]
        wake_speed = 5 if wake_speed == 0 else wake_speed

        slope = self.safe_slope_from_direction(direction)

        upstream_object = UpstreamList(
            slope,
            farm_y,
            farm_x,
            gradient,
            gradient_threshold,
            image,
            wind_farm_mask,
            wake_speed,
            upstream_direction,
            search_range=self.upstream_range,
        )

        upstream_list, upstream_mask, upstream_speed = upstream_object.get_list()

        if show_mask:
            plt.imshow(upstream_mask, cmap="gray")
            plt.show()

        return upstream_mask, upstream_speed

    # -------------------------------------------------------------------------
    # Main single-scene processing
    # -------------------------------------------------------------------------
    def process(self, show=False):
        """
        Process a single Sentinel-1 scene and extract wake information.

        Parameters
        ----------
        show : bool, optional
            Whether to display intermediate masks.

        Returns
        -------
        tuple or bool
            Scene-level wake statistics and masks if successful, otherwise False.
        """
        if not self.farm_location_check():
            return False

        # Prepare output band
        self.sentinel_1.addBand("wake_effect", ProductData.TYPE_FLOAT32)
        writer = ProductIO.getProductWriter("BEAM-DIMAP")
        self.sentinel_1.setProductWriter(writer)
        self.sentinel_1.writeHeader(self.output_path)

        # Read image and compute gradient
        wind_pixel_img = self.get_pixel()
        gradient = self.calculate_gradient(wind_pixel_img)

        gradient_mask = np.where((0.1 < gradient) & (gradient < 50))
        if gradient_mask[0].shape[0] == 0:
            return False

        self.gradient_threshold = gradient[gradient_mask].sum() / gradient_mask[0].shape[0] * 2
        self.speed_threshold = wind_pixel_img[gradient_mask].sum() / gradient_mask[0].shape[0]

        # Get farm location and external wind direction
        wind_farm_location = self.locate_wind_farm()
        speed, direction, date, time = self.get_era_data(wind_farm_location)

        # Some xarray scalars may need explicit conversion
        try:
            direction = float(direction.values)
        except Exception:
            direction = float(direction)

        # Build wind farm mask
        wind_farm_mask = self.generate_wind_farm_mask(
            wind_farm_location,
            wind_pixel_img,
            gradient,
            search_range=self.farm_search_range,
            show_mask=show,
        )

        # Downstream seed and search area
        farm_pixel, seed_pixel, wake_direction = self.farm_location_shift(
            wind_farm_location,
            direction,
            gradient,
            search_range=self.shift_search_range,
        )

        search_list, search_mask, seeds = self.generate_search_list(
            farm_pixel,
            seed_pixel,
            direction,
            wind_farm_mask,
            wind_pixel_img,
            wake_direction,
            search_range=self.search_range,
            show_mask=show,
        )

        # Wake extraction
        wake_mask, wake_mean_speed, external_box = region_grow(
            wind_pixel_img,
            seeds,
            [0.6, 1.15],
            search_list,
            self.gradient_threshold,
            gradient=gradient,
            p=1,
            show_mask=show,
        )

        # Upstream reference region
        upstream_pixel = self.upstream_location_shift(
            wind_farm_location,
            direction,
            gradient,
            search_range=self.shift_search_range,
        )

        upstream_mask, upstream_speed = self.generate_upstream_list(
            direction,
            upstream_pixel,
            self.gradient_threshold,
            gradient,
            wind_pixel_img,
            wind_farm_mask,
            wake_mean_speed,
            -wake_direction,
            show_mask=show,
        )

        upstream_pixel_num = np.where(upstream_mask != 0)[0].shape[0]

        # Write output product
        self.add_band(wake_mask, upstream_mask, wind_farm_mask)
        self.sentinel_1.closeIO()

        return (
            wake_mean_speed,
            np.where(wake_mask != 0)[0].shape[0],
            upstream_speed,
            upstream_pixel_num,
            wind_pixel_img,
            wind_farm_mask,
            wake_mask,
            upstream_mask,
            date,
            time,
        )


# =============================================================================
# Visualization utility
# =============================================================================
def contrast_stretch(img_gray, f1, f2, g1, g2):
    """
    Perform piecewise linear contrast stretching.

    Parameters
    ----------
    img_gray : ndarray
        Grayscale image.
    f1, f2 : float
        Input intensity breakpoints.
    g1, g2 : float
        Output intensity breakpoints.

    Returns
    -------
    ndarray
        Contrast-enhanced image.
    """
    img = np.zeros_like(img_gray)

    cond1 = np.logical_and(img_gray > 0, img_gray <= f1)
    cond2 = np.logical_and(img_gray > f1, img_gray <= f2)
    cond3 = np.logical_and(img_gray > f2, img_gray <= 255)

    img[cond1] = g1 / f1 * img_gray[cond1]
    img[cond2] = (g2 - g1) / (f2 - f1) * (img_gray[cond2] - f1) + g1
    img[cond3] = (255 - g2) / (255 - f2) * (img_gray[cond3] - f2) + g2

    return img


# =============================================================================
# Multiprocessing worker
# =============================================================================
def process_one_scene(task):
    """
    Worker function for processing a single SAR scene.

    Parameters
    ----------
    task : tuple
        Task configuration for one SAR scene.

    Returns
    -------
    dict or None
        Scene-level statistics if successful, otherwise None.
    """
    (
        year,
        wind_farm,
        search_range,
        upstream_range,
        farm_radius,
        farm_search_range,
        shift_search_range,
        input_path,
        output_path,
        mask_basic,
        visual_basic,
        folder,
    ) = task

    wake_detection_instance = None

    try:
        gc.enable()

        wake_detection_instance = SentinelWakeDetection(
            input_path,
            output_path,
            wind_farm,
            search_range,
            upstream_range,
            farm_radius,
            farm_search_range,
            shift_search_range,
        )

        result_data = wake_detection_instance.process()

        if not result_data:
            return None

        (
            wake_mean_speed,
            wake_pixel_num,
            upstream_speed,
            upstream_pixel_num,
            wind_pixel_img,
            wind_farm_mask,
            wake_mask,
            upstream_mask,
            date,
            time,
        ) = result_data

        # RGB mask visualization
        mask_rgb = np.zeros((wind_pixel_img.shape[0], wind_pixel_img.shape[1], 3), dtype=np.uint8)
        mask_rgb[np.where(wind_farm_mask == 1)] = [204, 255, 0]   # farm
        mask_rgb[np.where(wake_mask == 1)] = [238, 189, 215]      # wake
        mask_rgb[np.where(upstream_mask == 1)] = [240, 176, 0]    # upstream

        os.makedirs(mask_basic, exist_ok=True)
        os.makedirs(visual_basic, exist_ok=True)

        cv2.imwrite(os.path.join(mask_basic, folder + ".png"), mask_rgb)

        # Contrast-enhanced grayscale visualization
        mask = cv2.normalize(
            wind_pixel_img + wind_farm_mask + wake_mask * 2 + upstream_mask * 3,
            None,
            0,
            255,
            cv2.NORM_MINMAX,
        ).astype(np.uint8)

        mask = contrast_stretch(mask, 5, 50, 1, 150)
        cv2.imwrite(os.path.join(visual_basic, folder + ".png"), mask)

        return {
            "wind_farm_name": wind_farm,
            "year": year,
            "date": date,
            "time": time,
            "wake_speed": wake_mean_speed,
            "wake_pixel": wake_pixel_num,
            "upstream_speed": upstream_speed,
            "upstream_pixel": upstream_pixel_num,
            "sar_name": folder,
        }

    except Exception:
        print(f"Error processing {year} / {folder}:")
        traceback.print_exc()
        return None

    finally:
        if wake_detection_instance:
            try:
                if hasattr(wake_detection_instance, "sentinel_1"):
                    wake_detection_instance.sentinel_1.dispose()
            except Exception:
                pass

        del wake_detection_instance
        gc.collect()


# =============================================================================
# Batch-task generation
# =============================================================================
def build_tasks(farms):
    """
    Build processing task list for all wind farms and years.

    Parameters
    ----------
    farms : list of dict
        Wind farm configuration list.

    Returns
    -------
    list
        Task list for multiprocessing.
    """
    tasks = []

    for f in farms:
        search_range = (100, f["sr"])
        upstream_range = (20, f["ur"])
        farm_search_range = 2

        wind_farm = f["wf"]
        years = f["yrs"]
        farm_radius = f["rad"]
        shift_search_range = f["step"]

        print(
            f"processing: {wind_farm}, "
            f"Search: {search_range}, Upstream: {upstream_range}, File: {DEFAULT_OUTPUT_CSV}"
        )

        for year in years:
            output_basic = os.path.join(f"{WAKE_EFFECT_BASE}/archive{year}", wind_farm)
            input_basic = os.path.join(f"{WIND_SPEED_BASE}/archive{year}", wind_farm)
            visual_basic = os.path.join(f"{WAKE_VISUAL_BASE}/archive{year}", wind_farm)
            mask_basic = os.path.join(f"{WAKE_MASK_BASE}/archive{year}", wind_farm)

            for p in [output_basic, visual_basic, mask_basic]:
                os.makedirs(p, exist_ok=True)

            if not os.path.exists(input_basic):
                print(f"Skipping {input_basic}, path not found.")
                continue

            folders = [folder for folder in os.listdir(input_basic) if folder.endswith(".dim")]

            for folder in folders:
                input_path = os.path.join(input_basic, folder)
                output_path = os.path.join(output_basic, folder)

                tasks.append(
                    (
                        year,
                        wind_farm,
                        search_range,
                        upstream_range,
                        farm_radius,
                        farm_search_range,
                        shift_search_range,
                        input_path,
                        output_path,
                        mask_basic,
                        visual_basic,
                        folder,
                    )
                )

    print(f"Total scenes to process: {len(tasks)}")
    return tasks


def initialize_csv(file_path, header):
    """
    Initialize output CSV file and write header if needed.
    """
    file_exists = os.path.isfile(file_path)
    with open(file_path, "a+", encoding="utf-8", newline="") as f:
        dict_writer = csv.DictWriter(f, header)
        if not file_exists or os.stat(file_path).st_size == 0:
            dict_writer.writeheader()


def run_batch_processing(tasks, file_path, header, processes=DEFAULT_NUM_PROCESSES):
    """
    Run multiprocessing batch processing and write results to CSV.

    Parameters
    ----------
    tasks : list
        Task list.
    file_path : str
        Output CSV file path.
    header : list
        CSV header.
    processes : int, optional
        Number of worker processes.
    """
    if not tasks:
        return

    with open(file_path, "a+", encoding="utf-8", newline="") as f:
        dict_writer = csv.DictWriter(f, header)

        with Pool(processes=processes) as pool:
            iterator = pool.imap_unordered(process_one_scene, tasks, chunksize=1)

            for result in iterator:
                if result is not None:
                    dict_writer.writerow(result)
                    f.flush()


# =============================================================================
# Main execution
# =============================================================================
if __name__ == "__main__":
    farms = [
        {"wf": "wf2", "yrs": ["2020", "2021", "2022"], "sr": 75, "ur": 75, "rad": 50, "step": 1},
        {"wf": "wf4", "yrs": ["2022"], "sr": 50, "ur": 50, "rad": 35, "step": 1},
        {"wf": "wf7", "yrs": ["2021", "2022"], "sr": 35, "ur": 35, "rad": 25, "step": 1},
        {"wf": "wf8", "yrs": ["2020", "2021", "2022"], "sr": 30, "ur": 30, "rad": 20, "step": 1},
        {"wf": "wf9", "yrs": ["2020", "2021", "2022"], "sr": 25, "ur": 25, "rad": 15, "step": 1},
        {"wf": "wf10", "yrs": ["2022"], "sr": 50, "ur": 50, "rad": 35, "step": 2},
        {"wf": "wf11", "yrs": ["2020", "2021", "2022"], "sr": 30, "ur": 30, "rad": 20, "step": 2},
        {"wf": "wf12", "yrs": ["2020", "2021", "2022"], "sr": 40, "ur": 40, "rad": 30, "step": 1},
        {"wf": "wf13", "yrs": ["2020", "2021", "2022"], "sr": 40, "ur": 40, "rad": 30, "step": 1},
        {"wf": "wf14", "yrs": ["2020", "2021", "2022"], "sr": 25, "ur": 25, "rad": 15, "step": 1},
        {"wf": "wf15", "yrs": ["2020", "2021", "2022"], "sr": 25, "ur": 25, "rad": 15, "step": 1},
        {"wf": "wf16", "yrs": ["2020", "2021", "2022"], "sr": 30, "ur": 30, "rad": 20, "step": 1},
        {"wf": "wf17", "yrs": ["2022"], "sr": 35, "ur": 35, "rad": 25, "step": 1},
        {"wf": "wf19", "yrs": ["2020", "2021", "2022"], "sr": 30, "ur": 30, "rad": 20, "step": 1},
        {"wf": "wf20", "yrs": ["2020", "2021", "2022"], "sr": 35, "ur": 35, "rad": 25, "step": 1},
        {"wf": "wf21", "yrs": ["2020", "2021", "2022"], "sr": 35, "ur": 35, "rad": 25, "step": 1},
        {"wf": "wf23", "yrs": ["2020", "2021", "2022"], "sr": 25, "ur": 25, "rad": 15, "step": 1},
        {"wf": "wf24", "yrs": ["2020", "2021", "2022"], "sr": 35, "ur": 35, "rad": 25, "step": 1},
        {"wf": "wf25", "yrs": ["2020", "2021", "2022"], "sr": 35, "ur": 35, "rad": 25, "step": 1},
        {"wf": "wf28", "yrs": ["2020", "2021", "2022"], "sr": 25, "ur": 25, "rad": 15, "step": 1},
        {"wf": "wf29", "yrs": ["2020", "2021", "2022"], "sr": 35, "ur": 35, "rad": 25, "step": 1},
        {"wf": "wf30", "yrs": ["2021", "2022"], "sr": 35, "ur": 35, "rad": 25, "step": 2},
        {"wf": "wf31", "yrs": ["2022"], "sr": 40, "ur": 40, "rad": 30, "step": 1},
        {"wf": "wf34", "yrs": ["2022"], "sr": 25, "ur": 25, "rad": 15, "step": 1},
        {"wf": "wf35", "yrs": ["2022"], "sr": 25, "ur": 25, "rad": 15, "step": 1},
        {"wf": "wf37", "yrs": ["2020", "2021", "2022"], "sr": 30, "ur": 30, "rad": 20, "step": 1},
        {"wf": "wf38", "yrs": ["2020", "2021", "2022"], "sr": 25, "ur": 25, "rad": 15, "step": 1},
        {"wf": "wf47", "yrs": ["2020", "2021", "2022"], "sr": 25, "ur": 25, "rad": 15, "step": 1},
        {"wf": "wf50", "yrs": ["2020", "2021", "2022"], "sr": 25, "ur": 25, "rad": 15, "step": 1},
        {"wf": "wf51", "yrs": ["2020", "2021", "2022"], "sr": 40, "ur": 40, "rad": 30, "step": 1},
        {"wf": "wf52", "yrs": ["2022"], "sr": 25, "ur": 25, "rad": 15, "step": 1},
        {"wf": "wf53", "yrs": ["2020", "2021", "2022"], "sr": 25, "ur": 25, "rad": 15, "step": 1},
        {"wf": "wf54", "yrs": ["2020", "2021", "2022"], "sr": 40, "ur": 40, "rad": 30, "step": 2},
        {"wf": "wf55", "yrs": ["2020", "2021", "2022"], "sr": 25, "ur": 25, "rad": 15, "step": 1},
        {"wf": "wf56", "yrs": ["2020", "2021", "2022"], "sr": 25, "ur": 25, "rad": 15, "step": 1},
        {"wf": "wf57", "yrs": ["2020", "2021", "2022"], "sr": 35, "ur": 35, "rad": 25, "step": 1},
        {"wf": "wf58", "yrs": ["2021", "2022"], "sr": 25, "ur": 25, "rad": 15, "step": 1},
        {"wf": "wf61", "yrs": ["2022"], "sr": 25, "ur": 25, "rad": 15, "step": 1},
        {"wf": "wf62", "yrs": ["2022"], "sr": 40, "ur": 40, "rad": 30, "step": 2},
    ]

    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    header = [
        "wind_farm_name",
        "year",
        "date",
        "time",
        "wake_speed",
        "wake_pixel",
        "upstream_speed",
        "upstream_pixel",
        "sar_name",
    ]

    tasks = build_tasks(farms)
    initialize_csv(DEFAULT_OUTPUT_CSV, header)
    run_batch_processing(tasks, DEFAULT_OUTPUT_CSV, header, processes=DEFAULT_NUM_PROCESSES)

    print("Done.")