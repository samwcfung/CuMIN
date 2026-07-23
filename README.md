# CuMIN

**Cu**rated ROIs for **Min**ian — a modular pipeline for extracting and analysing calcium
fluorescence traces from manually curated ROIs in spinal dorsal horn slice recordings.

CuMIN takes a `.tif` timelapse plus a matching ImageJ/FIJI `RoiSet.zip`, applies
photobleaching correction, background removal and denoising, extracts per-ROI ΔF/F traces,
detects events, and produces per-slice, per-mouse and per-pain-model summaries.

---

## Requirements

Python 3.10–3.11. No GPU required.

```bash
git clone https://github.com/samwcfung/CuMIN.git
cd CuMIN
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

For the interactive notebook, install into the **same** environment that
launches JupyterLab:

```bash
pip install -r requirements-notebook.txt
python -m ipykernel install --user --name cuminv3 --display-name "CuMIN"
```

Or with conda:

```bash
conda env create -f environment.yml
conda activate cumin
```

### Optional extras

Everything in `requirements-optional.txt` is genuinely optional — each import is guarded and
the pipeline warns and continues without them.

| Extra | Needed when | Install |
|---|---|---|
| `cupy_cuda11x` | `preprocessing.use_gpu: true` | requires an NVIDIA GPU + CUDA 11.x |
| `normcorre` | `motion_correction.enabled: true` | `pip install normcorre` |

Both are off by default in `config/config.yaml`.

---

## Input data

> **This is the most common source of silent failure.** If filenames don't parse, the pipeline
> logs `No matched file pairs found`, writes an empty summary, and exits *successfully*.

Place `.tif` recordings and their ROI `.zip` files in one directory (subdirectories are
searched recursively). Filenames must encode four metadata fields, underscore-separated:

```
<MouseID><n>_<date>_<ipsi|contra><n>_<condition>um
```

Example of a valid pair:

```
data/
├── CFA1_7.23.20_ipsi1_10um.tif
└── RoiSet_CFA1_7.23.20_ipsi1_10um.zip
```

| Field | Pattern | Examples |
|---|---|---|
| Mouse ID / pain model | letters + digits | `CFA1`, `SNI4`, `PTOA2` |
| Date | `d.d.yy` or `d-d-yyyy` | `7.23.20`, `10-2-19` |
| Slice | `ipsi` or `contra` + optional digits | `ipsi1`, `contra2` |
| Condition | digits + `um` | `0um`, `10um`, `25um` |

The pain model is derived from the leading letters of the mouse ID (`CFA1` → `CFA`) and is
used to group the final summaries.

**Tolerated variations.** Matching runs in two passes — exact stem match first, then metadata
match — so these prefixes and suffixes are stripped automatically:

- Prefixes: `RoiSet_`, `ROIset_`, `ROIs_`, `ROI_` (case-insensitive)
- Suffixes: `_cor`, `_corrected`, `_uncor`

Files pair when mouse ID, date, slice type, slice number **and** condition all agree.

**Condition must be configured.** `0um`, `10um` and `25um` have condition-specific baseline
and analysis windows in `config.yaml`. Any other condition silently falls back to the defaults
(`baseline_frames: [0, 200]`, `analysis_frames: [230, 580]`). Add a block under
`analysis.condition_specific` before using a new concentration.

---

## Usage

```bash
python pipeline.py \
  --input_dir  /path/to/data \
  --output_dir /path/to/results \
  --config     config/config.yaml \
  --mode       all
```

> Pass `--config config/config.yaml` explicitly. The flag defaults to `config.yaml` in the
> working directory, which does not exist in a fresh clone.

| Flag | Default | Description |
|---|---|---|
| `--input_dir` | *(required)* | Directory containing `.tif` and ROI `.zip` files |
| `--output_dir` | *(required)* | Created if absent |
| `--config` | `config.yaml` | Path to the YAML config |
| `--mode` | `all` | `all`, `preprocess`, `extract`, or `analyze` |
| `--max_workers` | all CPUs | Parallel worker processes |
| `--disable_advanced` | off | Skip correlation/clustering/population analysis |

### Modes

| Mode | Motion corr. + photobleaching | ROI extraction + background subtraction | Event detection, QC, summaries |
|---|---|---|---|
| `preprocess` | ✓ | — | — |
| `extract` | — | ✓ | — |
| `analyze` | — | ✓ | ✓ |
| `all` | ✓ | ✓ | ✓ |

### Memory

Slices are processed in parallel via `ProcessPoolExecutor`, and each worker holds a full
image stack. On a large dataset, `--max_workers` defaulting to the CPU count will exhaust
RAM. Start conservatively (`--max_workers 4`) and scale up. `roi_processing.optimization`
in the config also offers `spatial_downsample` and `use_float16` to cut footprint.

---

## Outputs

```
results/
├── logs/
│   └── main_pipeline.log              # top-level run log
├── pipeline_summary.json              # run metadata, mode, per-slice status
├── <slice_name>/                      # one directory per matched pair
│   ├── intermediate_traces/           # per-stage traces (CSV by default)
│   ├── <slice_name>_metrics.xlsx      # per-ROI metrics for this slice
│   ├── <slice_name>_corrected.h5      # only if save_corrected_data: true
│   ├── pnr_diagnostic_info.json       # PNR-based ROI refinement diagnostics
│   ├── pnr_visualization.png
│   └── *.png                          # ROI maps, trace overlays, QC plots
├── <mouse_id>_summary.xlsx            # per-mouse metrics
└── <pain_model>_combined_summary.xlsx # pooled per pain model
```

Large intermediates (`save_corrected_data`, `save_motion_data`, `save_masks_for_cnmf`) are
disabled by default to keep output size manageable. Enable them in the config when debugging
a specific slice.

---

## Configuration

`config/config.yaml` is the single source of truth for all parameters. Main sections:

| Section | Controls |
|---|---|
| `preprocessing` | Photobleaching correction (polynomial detrend), background removal (`tophat`/`uniform`), denoising (`median`/`gaussian`/`bilateral`/`anisotropic`), GPU toggle |
| `motion_correction` | Rigid/non-rigid NoRMCorre correction — **disabled by default** |
| `roi_processing` | ROI extraction from `.zip`, ROI-specific detrending, PNR-based refinement, background subtraction method, downsampling |
| `analysis` | Frame rate (`1.67` Hz), per-condition baseline/analysis windows, peak detection, spectral features, event detection, QC thresholds |
| `visualization` | Colormaps, which plot types to emit |
| `advanced_analysis` | Correlation, hierarchical clustering, population activity |

Parameters most worth checking before a first run: `analysis.frame_rate`,
`analysis.condition_specific.*.baseline_frames` / `analysis_frames`, and
`analysis.peak_detection.prominence`.

---

## Configuration integrity

`config/config.yaml` is the single source of truth: every parameter the
pipeline uses comes from it, and the pipeline refuses to start if it does not.

Three mechanisms enforce this (`modules/config_schema.py`):

1. **Schema validation** runs in `load_config()` before any processing. Missing
   or mistyped keys abort the run with a list of what is wrong. Set
   `CUMIN_CONFIG_STRICT=0` to downgrade this to warnings.
2. **Fail-loud accessor** wraps the config tree so that any code path reading a
   key the file does not supply logs
   `CONFIG FALLBACK: <key> ... NOT under config control` instead of silently
   using a hardcoded default.
3. **Provenance dump** writes `resolved_config.yaml` and
   `config_provenance.json` into every output directory, recording exactly
   which parameters the run used.

After a run, confirm nothing fell back:

```bash
grep -r "CONFIG FALLBACK" <output_dir>/logs/
```

No output means every parameter came from the config file.

To audit the config against the source at any time:

```bash
python audit_config.py . config/config.yaml
```

This reports keys the code reads that the config does not supply, and keys the
config supplies that no code reads. Treat it as a triage list: the heuristic has
blind spots in both directions, and the runtime check above is authoritative.

## Defining an active ROI

An ROI is called active when it clears an absolute dF/F floor and, optionally,
an SNR floor measured against its own baseline noise. The second gate exists
because a fixed dF/F cut judges a quiet ROI and a noisy one by the same
yardstick, which lets noise excursions in poor ROIs count as responses.

```yaml
analysis:
  condition_specific:
    "25um":
      baseline_frames: [0, 200]
      analysis_frames: [230, 580]
      active_threshold: 0.02        # absolute dF/F floor
      active_metric: "max_df_f"
      noise_criterion:
        enabled: true
        method: "mad"               # "mad" (robust) or "std"
        snr_threshold: 3.0          # signal must exceed 3x baseline noise
        require_both: true          # true = absolute AND snr
```

`method: "mad"` (median absolute deviation, scaled by 1.4826) is recommended.
Plain `std` is inflated by any genuine spontaneous transients sitting in the
baseline window, which penalises exactly the ROIs that are most active.

Every decision is recorded per ROI in `<slice>_metrics.xlsx` as `noise_level`,
`activity_snr`, `snr_threshold`, `passes_absolute` and `passes_snr`, so calls
can be re-derived without re-running the pipeline.

Set `noise_criterion.enabled: false` to restore absolute-threshold-only
behaviour.

## Troubleshooting

| Symptom | Cause |
|---|---|
| `No matched file pairs found` | Filenames don't match the convention above. Run with the log at DEBUG to see per-file parse attempts. |
| `FileNotFoundError: config.yaml` | Pass `--config config/config.yaml`. |
| `CuPy not found` warning | Expected without a GPU. Harmless — CPU fallback is used. |
| `NoRMCorre not found` warning | Expected. Harmless unless `motion_correction.enabled: true`. |
| Workers killed / out of memory | Lower `--max_workers`; enable `spatial_downsample` and `use_float16`. |
| Baselines look wrong for a new condition | Add a block under `analysis.condition_specific`. Validation will reject the run if the block is incomplete. |
| `ConfigValidationError` on startup | The config is missing keys the code needs. The message lists them. Fix the config, or set `CUMIN_CONFIG_STRICT=0` to proceed with hardcoded defaults. |
| `CONFIG FALLBACK` warnings in the log | A parameter is not under config control. Add the named key to `config.yaml`. |

---

## Repository layout

```
pipeline.py                  # CLI entry point and per-slice orchestration
config/config.yaml           # all parameters
modules/
├── file_matcher.py          # metadata-based .tif ↔ .zip pairing
├── preprocessing.py         # photobleaching, background, denoising, stripe correction
├── motion_correction.py     # optional NoRMCorre wrapper
├── roi_processing.py        # ROI extraction, background subtraction, PNR refinement
├── analysis.py              # ΔF/F, event detection, QC
├── advanced_analysis.py     # correlation, clustering, population activity
├── visualization.py         # figure generation
├── visualization_helpers.py # interactive (HoloViews/Panel) helpers
├── config_schema.py         # config validation, fail-loud accessor, provenance
└── utils.py                 # logging, per-slice/mouse/model summary export
audit_config.py              # static audit of config keys vs source
notebooks/CUMIN.V6.ipynb     # interactive exploration
```

---

## Citation

If you use CuMIN, please cite:

> *(add manuscript reference once published)*

## License

*(add license — currently unspecified)*
