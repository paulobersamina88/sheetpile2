# Sheet Pile Design Streamlit App

A preliminary design and teaching app for steel sheet pile walls.

## Included features
- lateral soil pressure using Rankine active/passive pressure
- surcharge loading
- optional seismic increment using simplified Mononobe-Okabe style logic
- cantilever embedment length screening
- shear and bending extraction at dredge line / cantilever bottom
- sheet pile section selection from editable CSV
- simplified AISC axial compression + bending interaction check
- downloadable CSV summary

## Important limitations
This package is a **screening / educational tool** and is not a substitute for:
- project geotechnical recommendations
- groundwater and seepage checks
- deflection/serviceability checks
- corrosion/interlock/constructability checks
- full code-based design by a licensed engineer

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Edit the section library
Modify:
`data/sample_sheet_piles.csv`

Recommended columns:
- `section`
- `type`
- `A_mm2`
- `Sx_mm3`
- `Zx_mm3`
- `Ix_mm4`
- `rx_mm`
- `ry_mm`
- `Fy_MPa`
- `weight_kg_per_m`

## Design philosophy used in the app
- IBC-style retaining wall lateral load workflow
- AISC 360-22 style steel capacity screening
- optional NSCP-compatible input workflow depending on project seismic parameters

Always verify the final design using the governing local code and the manufacturer's certified section properties.
