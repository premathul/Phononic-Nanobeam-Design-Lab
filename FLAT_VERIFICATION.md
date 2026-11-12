# Flat-folder verification — 12 November 2026

- All 496 pre-reorganization files preserved byte-for-byte; full path/hash map in flat_manifest.json.
- Complete before-flatten ZIP verified against every original file.
- Zero subdirectories in delivered main folder.
- All Python filenames end _AP_pnc.py.
- 300 original-source hashes and 180 archived artifacts verified.
- 333 saved harmonic residuals and 60 cavity eigen-residuals passed existing arithmetic/numerical gates.
- 14 field/mode NPZ files checked for finite arrays and valid mesh dimensions.
- Six original launcher safety tests passed.
- Flat adapter restored the final previously completed smoke run, resumed it, skipped the hash-verified complete stage, and collected results back into a flat folder successfully.
- New-launch dry-run passed without creating output.
- No full physics campaign rerun. Numerical model and existing failed physical-validation gates unchanged.

An older smoke run correctly failed resume because its source differs from the current template. The adapter test used the final verified smoke run. An initial workbook-reader interpreter lacked openpyxl; the successful check used the existing anaconda3 base interpreter (openpyxl 3.1.2) with the existing qtcad mechanics interpreter. No packages were installed.
