# R2G2: SAR-Based Wind Farm Wake Detection

This repository contains the core implementation of the **R2G2** framework for offshore wind farm wake detection from Sentinel-1 SAR-derived wind fields.

## Overview

The workflow combines:

1. **Region-constrained search**
   - downstream wake search region construction
   - upstream reference region construction
   - wind-farm mask generation

2. **Region-growing wake extraction**
   - seed initialization near the wind farm
   - intensity-ratio-based region growing
   - gradient-based constraint for noise suppression

3. **External wind-direction constraint**
   - ERA5 10 m wind data are used to determine wake propagation direction

The code is designed for batch processing of multiple SAR scenes and exporting:
- wake masks
- upstream reference masks
- visualization products
- scene-level wake statistics

---

## Repository Structure

```text
.
├── wake_test_parallel.py      # Main processing pipeline
├── region_grow.py             # Region-growing wake extraction
├── search_range.py            # Wake and upstream search-region construction
├── point_location.py          # Geometric utility functions
├── wind_direction/
│   └── wind_direction.py      # ERA5 wind speed and direction interface
├── wind_farms.yml             # Wind farm coordinate configuration
├── README.md
└── requirements.txt
````

---

## Main Modules

### `wake_test_parallel.py`

Main batch-processing script for Sentinel-1 scenes.
This script:

* reads SAR-derived wind speed products,
* locates wind farms,
* obtains ERA5 wind direction,
* constructs wake and upstream search regions,
* applies the region-growing algorithm,
* exports masks and statistics.

### `region_grow.py`

Implements the constrained region-growing algorithm for wake extraction.

### `search_range.py`

Defines:

* the downstream wake search region, and
* the upstream reference region

using geometric constraints and wind direction.

### `point_location.py`

Provides geometric helper functions for point-in-polygon tests.

### `wind_direction/wind_direction.py`

Loads ERA5 wind data and computes wind speed and direction from 10 m wind components.

---

## Data Requirements

This code requires the following input data:

### 1. Sentinel-1 SAR-derived wind-speed products

The input products are expected to contain at least:

* `wind_speed`
* `Sigma0_VV`

and be readable through `esa_snappy`.

### 2. ERA5 reanalysis data

ERA5 wind data are used to estimate wind direction and speed at the wind farm location and SAR acquisition time.

Supported formats:

* `.nc`
* `.grib`

### 3. Wind farm location file

The YAML file (`wind_farms.yml`) is required to provide the geographic coordinates of each wind farm.

Example structure:

```yaml
wind_farms:
  wf1:
    coordinates:
      latitude: 54.123
      longitude: 3.456
```

---

## Dependencies

Main Python dependencies include:

* `numpy`
* `opencv-python`
* `matplotlib`
* `Pillow`
* `xarray`
* `dask`
* `cfgrib`

In addition, this project depends on:

* **SNAP / esa_snappy** for reading and writing Sentinel-1 products
* local ERA5 data files
* wind farm configuration files

Install Python dependencies with:

```bash
pip install -r requirements.txt
```

---

## Notes on `esa_snappy`

`esa_snappy` is **not installed via pip** in the standard way and usually requires a local installation of ESA SNAP.

Please install SNAP first and configure the Python interface according to the official SNAP documentation.

---

## Usage

Run the main processing script:

```bash
python wake_test_parallel.py
```

Before running, update the following paths in the script as needed:

* ERA5 data directory
* Sentinel-1 input directory
* output directories for wake masks, visualizations, and statistics

---

## Output

The workflow generates:

### 1. Wake-effect products

A new output band named `wake_effect` is written to the output Sentinel-1 product.

Mask encoding:

* `1` = wake region
* `2` = upstream reference region
* `3` = wind farm region

### 2. Visualization images

* RGB mask overlays
* contrast-enhanced grayscale images

### 3. CSV statistics

A CSV file containing scene-level results, including:

* wind farm name
* acquisition date and time
* wake-region mean wind speed
* wake pixel count
* upstream-region mean wind speed
* upstream pixel count
* SAR product name

---

## Reproducibility

This repository provides the core implementation of the R2G2 algorithm, including:

* wake search-region construction,
* upstream reference-region construction,
* gradient-based turbine localization,
* region-growing wake extraction,
* and scene-level processing workflow.

To ensure reproducibility, users must prepare:

* SAR-derived wind-speed products,
* ERA5 wind data,
* wind farm coordinate configuration,
* and a properly configured SNAP Python interface.

---

## Citation

If you use this code in academic work, please cite the corresponding paper.



