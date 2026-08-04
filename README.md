# SAR_WAKE: Long-Range Offshore Wind Farm Wake Detection from Sentinel-1 SAR

[![Project Website](https://img.shields.io/badge/Project-Website-44e1d0?style=for-the-badge)](https://lironui.github.io/SAR_WAKE/)
[![Paper](https://img.shields.io/badge/Paper-Communications%20Engineering-ff775f?style=for-the-badge)](https://doi.org/10.1038/s44172-026-00684-7)
[![GitHub](https://img.shields.io/badge/Code-SAR__WAKE-07171a?style=for-the-badge&logo=github)](https://github.com/lironui/SAR_WAKE)
[![EMail](https://img.shields.io/badge/EMail__WAKE-07171a?style=for-the-badge&logo=email)](lironui@outlook.com)

> **Explore the bilingual project website:**  
> **https://lironui.github.io/SAR_WAKE/**

This repository provides the code, scene lists, metadata, and selected visualizations accompanying the paper:

**Long-range near-surface wake signatures of offshore wind farm clusters revealed by satellite observations**  
Rui Li, Jincheng Zhang, and Xiaowei Zhao  
*Communications Engineering*, **5**, 144 (2026)  
[https://doi.org/10.1038/s44172-026-00684-7](https://doi.org/10.1038/s44172-026-00684-7)

## Highlights

- Analysis of **7,122 Sentinel-1A/B SAR scenes** acquired from 2020 to 2022.
- Coverage of **more than 60 offshore wind farms** across Europe and Asia.
- Observation-based evidence of **near-surface wake signatures extending beyond 100 km** under favorable conditions.
- A mean 10 m wind-speed deficit of **0.990 m/s**, equivalent to **12.4%** of the upstream wind speed.
- **3,789 wake-deficit cases** identified from 5,920 valid quantitative samples.
- Examples of inter-farm and cross-border wake interactions in densely developed offshore regions.
- An open collection of representative SAR wind-field visualizations and scene metadata.

> [!IMPORTANT]
> All reported wake metrics are derived from SAR-retrieved **10 m neutral wind speeds**. They describe near-surface wake signatures and should not be directly extrapolated to turbine hub height, rotor-layer flow, or project-level power losses without additional modeling or in-situ observations.

## Project Website

The project website presents the paper findings, methodology, original figures, and selected SAR scenes in an interactive format.

- **English:** [https://lironui.github.io/SAR_WAKE/](https://lironui.github.io/SAR_WAKE/)
- **中文:** [https://lironui.github.io/SAR_WAKE/zh-CN.html](https://lironui.github.io/SAR_WAKE/zh-CN.html)

## Method Overview

The processing workflow combines Sentinel-1 SAR preprocessing, CMOD5.N wind-speed retrieval, ERA5 wind-direction information, and the proposed **Restricted-Region Gradient-Guided Growing (R²G³)** method.

1. **SAR preprocessing**
   - thermal noise removal;
   - orbit correction;
   - radiometric calibration;
   - speckle filtering;
   - land, turbine, ship, and other bright-target masking;
   - multi-looking to a final resolution of approximately 500 m.

2. **Near-surface wind retrieval**
   - Sentinel-1 azimuth and incidence angles;
   - ERA5 hourly wind direction;
   - CMOD5.N geophysical model function;
   - retrieval of neutral wind speed at 10 m height.

3. **R²G³ wake extraction**
   - region-constrained upstream and downstream searches;
   - wind-farm mask construction;
   - gradient-guided turbine and wake-boundary localization;
   - seed initialization and constrained region growing;
   - extraction of upstream reference and downstream wake regions.

4. **Scene-level statistics**
   - upstream and wake-region mean wind speeds;
   - wake and reference pixel counts;
   - wind-speed deficit and percentage reduction;
   - visualization and metadata export.

## Repository Structure

```text
SAR_WAKE/
├── Highligh_Scene/
│   ├── metadata/                  # Sentinel-1 scene metadata
│   └── visualization/             # Selected SAR-derived wind-field images
├── sar_list_farm/                 # SAR-to-wind-farm mapping lists
├── assets/                        # Project website figures and gallery images
├── .github/workflows/             # GitHub Pages deployment workflow
├── index.html                     # English project website
├── zh-CN.html                     # Chinese project website
├── styles.css                     # Website styling
├── script.js                      # Gallery, filtering, and lightbox behavior
├── wake_detection.py              # Main batch-processing pipeline
├── region_grow.py                 # Constrained region-growing algorithm
├── search_range.py                # Upstream/downstream search regions
├── wind_direction.py              # ERA5 wind speed and direction interface
├── wind_farms.yml                 # Wind-farm coordinates and configuration
├── sar_list.txt                   # Sentinel-1 scene list
├── requirements.txt
└── README.md
```

## Highlight Scenes

`Highligh_Scene/` contains a curated set of representative Sentinel-1 wake scenes:

- `visualization/` provides contrast-enhanced SAR-derived wind-field images;
- `metadata/` provides the associated platform, acquisition time, orbit, pass direction, and scene coordinates.

The dark elongated features in these visualizations generally represent lower retrieved near-surface wind speeds. Interpretation should consider imaging artifacts, meteorological variability, retrieval uncertainty, and wind-farm operating conditions.

## Data Requirements

### Sentinel-1 wind-speed products

Input products should contain at least:

- `wind_speed`
- `Sigma0_VV`

The current pipeline expects products readable through ESA SNAP and `esa_snappy`.

### ERA5 reanalysis

ERA5 hourly data are used to estimate wind speed and direction at the wind-farm location and SAR acquisition time.

Supported local formats include:

- NetCDF (`.nc`)
- GRIB (`.grib`)

### Wind-farm configuration

`wind_farms.yml` stores wind-farm locations and geometric configuration used during batch processing.

### SAR-to-farm mapping

`sar_list_farm/` associates individual SAR scenes with the wind farms covered by each acquisition.

## Installation

Create a Python environment and install the listed dependencies:

```bash
git clone https://github.com/lironui/SAR_WAKE.git
cd SAR_WAKE
pip install -r requirements.txt
```

The main Python dependencies include:

- NumPy
- OpenCV
- Matplotlib
- Pillow
- Xarray
- Dask
- cfgrib

### ESA SNAP

`esa_snappy` is not installed through the standard PyPI workflow. Install [ESA SNAP](https://step.esa.int/main/download/snap-download/) and configure its Python interface before running the processing pipeline.

## Usage

Run the main processing script:

```bash
python wake_detection.py
```

Before processing, update the local paths for:

- Sentinel-1 input products;
- ERA5 data;
- wind-farm configuration;
- output products, visualizations, and statistics.

## Outputs

The workflow can generate:

### Wake-effect products

A `wake_effect` band is written with the following mask encoding:

| Value | Region |
|---:|---|
| `1` | Wake-affected region |
| `2` | Upstream reference region |
| `3` | Wind-farm region |

### Visualizations

- RGB mask overlays;
- contrast-enhanced grayscale wind fields;
- selected inter-farm wake scenes.

### Scene-level statistics

- wind-farm name;
- acquisition date and time;
- upstream mean wind speed;
- wake-region mean wind speed;
- upstream and wake pixel counts;
- SAR product identifier.

## Data Availability

- Sentinel-1 SAR: [Copernicus Data Space Ecosystem](https://browser.dataspace.copernicus.eu/)
- ERA5: [ECMWF Reanalysis v5](https://www.ecmwf.int/en/forecasts/dataset/ecmwf-reanalysis-v5)
- Buoy observations: [NOAA National Data Buoy Center](https://www.ndbc.noaa.gov/)
- SAR scene lists and selected visualizations: this repository

## Citation

If this repository, dataset, or project website contributes to your work, please cite:

```bibtex
@article{li2026longrange,
  title   = {Long-range near-surface wake signatures of offshore wind farm clusters revealed by satellite observations},
  author  = {Li, Rui and Zhang, Jincheng and Zhao, Xiaowei},
  journal = {Communications Engineering},
  volume  = {5},
  pages   = {144},
  year    = {2026},
  doi     = {10.1038/s44172-026-00684-7}
}
```

## Acknowledgements

Sentinel-1 source imagery is provided by the European Space Agency through the Copernicus programme. ERA5 data are provided by ECMWF, and buoy observations are provided by the NOAA National Data Buoy Center.

## Contact

For questions about the paper or repository, please open a GitHub issue or contact the corresponding author listed in the publication.

