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
