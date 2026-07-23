"""Strict configuration handling for the CuMIN pipeline.

Problem this solves
-------------------
The pipeline reads parameters with ``config.get("key", <hardcoded default>)``.
When ``key`` is absent from config.yaml the hardcoded default silently wins, so
the pipeline runs with parameters that differ from the ones the config file
appears to specify. Four such mismatches existed in the original code, e.g.
evoked peak detection ran at prominence=0.02 / height=0.01 regardless of the
``analysis.peak_detection`` values.

Three mechanisms, layered
-------------------------
1. ``StrictDict``   - wraps every dict in the config tree. Any ``.get()`` for a
                      missing key is recorded and logged. In strict mode it
                      raises instead. No call sites need to change.
2. ``validate_config`` - checks the config against ``SCHEMA`` at load time,
                      before any compute happens. Reports missing required keys,
                      wrong types, and keys the code never reads.
3. ``dump_resolved_config`` - writes the fully resolved config plus a report of
                      every default that fired into the run's output directory,
                      so each result set carries a record of what actually ran.

Usage
-----
    from modules.config_schema import (
        wrap_strict, validate_config, dump_resolved_config, get_fallback_report,
    )

    config = yaml.safe_load(open(path))
    validate_config(config, logger, strict=True)
    config = wrap_strict(config, strict=False)
    ...
    dump_resolved_config(config, output_dir, logger)
"""

from __future__ import annotations

import json
import logging
import os
import threading
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


# --------------------------------------------------------------------------
# 1. Fail-loud accessor
# --------------------------------------------------------------------------

_fallbacks: dict[str, dict] = {}
_fallback_lock = threading.Lock()


class MissingConfigKey(KeyError):
    """Raised in strict mode when the config does not supply a requested key."""


class StrictDict(dict):
    """A dict that reports whenever ``.get()`` falls back to a default.

    Nested dicts returned by ``.get()``/``[]`` are themselves StrictDicts, so a
    sub-dict handed to a downstream function stays strict.

    ``_path`` is the dotted location in the config tree, used in messages.
    ``_strict`` raises instead of warning.
    """

    __slots__ = ("_path", "_strict")

    def __init__(self, *args, _path: str = "", _strict: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self._path = _path
        self._strict = _strict

    # -- pickling (required: config crosses process boundaries via
    #    ProcessPoolExecutor, which uses spawn on Windows) -------------------
    def __reduce__(self):
        return (_rebuild_strictdict, (dict(self), self._path, self._strict))

    def _child_path(self, key: Any) -> str:
        return f"{self._path}.{key}" if self._path else str(key)

    def _wrap_child(self, key: Any, value: Any) -> Any:
        return _wrap(value, self._child_path(key), self._strict)

    def get(self, key, default=None):
        if key not in self:
            _record_fallback(self._child_path(key), default, self._strict)
            # Wrap the default too, so `config.get("missing", {}).get("x")`
            # is still tracked rather than silently hitting a bare dict.
            return _wrap(default, self._child_path(key), self._strict)
        return self._wrap_child(key, super().__getitem__(key))

    def __getitem__(self, key):
        value = super().__getitem__(key)
        return self._wrap_child(key, value)


def _rebuild_strictdict(data, path, strict):
    return StrictDict(data, _path=path, _strict=strict)


class StrictList(list):
    """List whose dict elements stay strict (config lists of dicts)."""

    __slots__ = ()


def _wrap(value: Any, path: str, strict: bool) -> Any:
    if isinstance(value, StrictDict):
        return value
    if isinstance(value, dict):
        return StrictDict(value, _path=path, _strict=strict)
    if isinstance(value, list):
        return [_wrap(v, f"{path}[{i}]", strict) for i, v in enumerate(value)]
    return value


def wrap_strict(config: dict, strict: bool = False) -> StrictDict:
    """Recursively wrap a config tree so missing-key reads are reported.

    strict=False -> log a warning and use the default (recommended default;
                    keeps existing runs working while surfacing every gap)
    strict=True  -> raise MissingConfigKey on the first missing key
    """
    return _wrap(config, "", strict)


def _record_fallback(path: str, default: Any, strict: bool) -> None:
    with _fallback_lock:
        entry = _fallbacks.setdefault(path, {"default": repr(default), "count": 0})
        entry["count"] += 1
        first_time = entry["count"] == 1

    if strict:
        raise MissingConfigKey(
            f"config key '{path}' is not supplied by the config file "
            f"(code would silently fall back to {default!r}). "
            f"Add it to config.yaml or run with strict=False."
        )

    if first_time:
        logging.getLogger(__name__).warning(
            "CONFIG FALLBACK: '%s' missing from config file; using hardcoded "
            "default %r. This parameter is NOT under config control.",
            path,
            default,
        )


def get_fallback_report() -> dict:
    """Every config key that fell back to a hardcoded default this run."""
    with _fallback_lock:
        return {k: dict(v) for k, v in _fallbacks.items()}


def reset_fallback_report() -> None:
    with _fallback_lock:
        _fallbacks.clear()


# --------------------------------------------------------------------------
# 2. Schema
# --------------------------------------------------------------------------
# Dotted path -> (type or tuple of types, description)
#
# These are the keys the code actually reads. Keys marked ORPHANED were read by
# the code but absent from the original config.yaml, meaning a hardcoded default
# was in force. Keys marked DEAD were supplied by config.yaml but read by no
# code; they are listed in DEAD_KEYS below rather than here.

NUM = (int, float)

SCHEMA: dict[str, tuple] = {
    # ---- preprocessing -------------------------------------------------
    "preprocessing.correction_method": (str, "Photobleaching correction method"),
    "preprocessing.polynomial_order": (int, "Detrend polynomial degree"),
    "preprocessing.smoothing_sigma": (NUM, "Gaussian smoothing sigma"),
    "preprocessing.generate_plot": (bool, "Emit correction verification plot"),
    "preprocessing.save_corrected_data": (bool, "Save corrected H5 (large)"),
    "preprocessing.save_intermediate_traces": (bool, "Save per-stage traces"),
    "preprocessing.trace_save_format": (str, "csv or h5"),
    "preprocessing.background_removal.enabled": (bool, ""),
    "preprocessing.background_removal.method": (str, "uniform or tophat"),
    "preprocessing.background_removal.window_size": (int, ""),
    "preprocessing.denoise.enabled": (bool, ""),
    "preprocessing.denoise.method": (str, "gaussian/median/bilateral/anisotropic"),
    "preprocessing.denoise.params": (dict, "Method-specific denoise params"),

    # ---- motion correction ---------------------------------------------
    "motion_correction.enabled": (bool, ""),
    "motion_correction.method": (str, "rigid or nonrigid"),
    "motion_correction.max_shift": (int, ""),
    "motion_correction.apply_before_photobleach": (bool, ""),

    # ---- roi processing -------------------------------------------------
    "roi_processing.steps.extract_rois": (bool, ""),
    "roi_processing.steps.roi_specific_detrend": (bool, "Gate 1 of 2"),
    "roi_processing.steps.save_masks": (bool, ""),
    "roi_processing.steps.refine_rois": (bool, ""),
    "roi_processing.steps.refine_with_pnr": (bool, ""),
    "roi_processing.steps.subtract_background": (bool, ""),
    "roi_processing.save_intermediate_traces": (bool, ""),
    "roi_processing.trace_save_format": (str, ""),
    # ORPHANED: gate 2 of 2 for ROI detrending. Without this block the step
    # never ran even with steps.roi_specific_detrend: true.
    "roi_processing.roi_detrend.enabled": (bool, "Gate 2 of 2 for ROI detrend"),
    "roi_processing.roi_detrend.polynomial_degree": (int, "ROI detrend degree"),
    # ORPHANED: subdirectory name for intermediate traces
    "roi_processing.intermediate_trace_dir": (str, "Intermediate trace subdir"),
    "roi_processing.pnr_refinement.noise_freq_cutoff": (NUM, ""),
    "roi_processing.pnr_refinement.min_pnr": (NUM, "Min peak-to-noise to keep ROI"),
    "roi_processing.pnr_refinement.percentile_threshold": (NUM, ""),
    "roi_processing.pnr_refinement.trace_smoothing": (int, ""),
    "roi_processing.pnr_refinement.auto_determine": (bool, ""),
    "roi_processing.pnr_refinement.generate_plots": (bool, ""),
    "roi_processing.background.method": (str, ""),
    "roi_processing.background.min_background_area": (int, ""),
    "roi_processing.background.background_dilation": (int, ""),
    "roi_processing.background.periphery_size": (int, ""),
    "roi_processing.background.percentile": (NUM, ""),
    "roi_processing.background.median_filter_size": (int, ""),
    "roi_processing.background.dilation_size": (int, ""),
    "roi_processing.background.save_intermediate_traces": (bool, ""),
    "roi_processing.background.correct_slope": (bool, ""),
    "roi_processing.background.slope_window": (list, ""),
    "roi_processing.background.peak_prominence": (NUM, ""),

    # ---- analysis -------------------------------------------------------
    "analysis.frame_rate": (NUM, "Acquisition frame rate (Hz)"),
    "analysis.use_preprocessed_data": (bool, ""),
    "analysis.baseline_frames": (list, "Default baseline window"),
    "analysis.analysis_frames": (list, "Default analysis window"),
    "analysis.active_threshold": (NUM, "Default activity threshold"),
    "analysis.baseline_method": (str, "percentile/mean/min"),
    "analysis.baseline_percentile": (NUM, ""),
    "analysis.baseline_n_frames": (int, ""),
    "analysis.calculate_spectral_features": (bool, ""),
    "analysis.noise_criterion.enabled": (bool, "Global noise-gate default"),
    "analysis.noise_criterion.method": (str, "mad or std"),
    "analysis.noise_criterion.snr_threshold": (NUM, ""),
    "analysis.noise_criterion.require_both": (bool, ""),

    # peak detection (regular path)
    "analysis.peak_detection.prominence": (NUM, ""),
    "analysis.peak_detection.width": (NUM, ""),
    "analysis.peak_detection.distance": (NUM, ""),
    "analysis.peak_detection.height": (NUM, "Min dF/F for a peak"),
    "analysis.peak_detection.rel_height": (NUM, ""),
    "analysis.peak_detection.edge_detection": (bool, ""),
    "analysis.peak_detection.edge_threshold": (NUM, ""),
    "analysis.peak_detection.edge_peak_frames": (int, ""),
    "analysis.peak_detection.edge_rise_threshold": (NUM, ""),

    # ORPHANED: evoked path read these three and found none of them, so it ran
    # at prominence=0.02 / width=1 / height=0.01 on every slice.
    "analysis.evoked_detection.evoked_peak_prominence": (NUM, "Evoked prominence"),
    "analysis.evoked_detection.evoked_peak_width": (NUM, "Evoked min width"),
    "analysis.evoked_detection.evoked_peak_height": (NUM, "Evoked min dF/F"),

    "analysis.spontaneous_activity.prominence": (NUM, ""),
    "analysis.spontaneous_activity.width": (NUM, ""),

    # ORPHANED: perform_qc_checks reads only these three. The four keys the
    # original config supplied (min_snr, max_baseline_var, min_event_count,
    # max_motion_correlation) were read by nothing.
    "analysis.qc_thresholds.min_variance": (NUM, "Min trace variance"),
    "analysis.qc_thresholds.max_jump": (NUM, "Max frame-to-frame jump"),
    "analysis.qc_thresholds.max_drift": (NUM, "Max baseline drift"),

    # ---- visualization ---------------------------------------------------
    "visualization.roi_color_map": (str, ""),
    "visualization.save_individual_plots": (bool, ""),
    "visualization.plot_types": (list, ""),

    # ---- advanced analysis ------------------------------------------------
    "advanced_analysis.enabled": (bool, "Read by pipeline.py"),
    # ORPHANED: AdvancedFluorescenceAnalysis.__init__ reads these; none existed,
    # so n_clusters=3, distance_threshold=50, n_bins=20 were hardcoded and the
    # `methods:` list was inert.
    "advanced_analysis.ml_enabled": (bool, ""),
    "advanced_analysis.correlation_enabled": (bool, ""),
    "advanced_analysis.emd_enabled": (bool, ""),
    "advanced_analysis.ml.n_clusters": (int, "k for k-means over ROIs"),
    "advanced_analysis.ml.feature_selection": (list, ""),
    "advanced_analysis.correlation.distance_threshold": (NUM, ""),
    "advanced_analysis.correlation.method": (str, "pearson/spearman"),
    "advanced_analysis.emd.features": (list, ""),
    "advanced_analysis.emd.n_bins": (int, ""),
}

# Per-condition blocks under analysis.condition_specific.<condition>
CONDITION_SCHEMA: dict[str, tuple] = {
    "baseline_frames": (list, "Baseline window [start, end]"),
    "analysis_frames": (list, "Analysis window [start, end]"),
    "active_threshold": (NUM, "Absolute dF/F floor for calling an ROI active"),
    "active_metric": (str, "max_df_f / spont_peak_frequency"),
}

# Optional per-condition noise-aware activity gate. Validated only when the
# block is present, so conditions without it (e.g. 0um) stay valid.
NOISE_CRITERION_SCHEMA: dict[str, tuple] = {
    "enabled": (bool, "Apply the SNR gate in addition to the absolute floor"),
    "method": (str, "mad (robust) or std"),
    "snr_threshold": (NUM, "Signal must exceed this multiple of baseline noise"),
    "require_both": (bool, "true = absolute AND snr; false = OR"),
}

# Supplied by config.yaml but read by no code. Kept for documentation; the
# validator reports them as informational rather than as errors.
DEAD_KEYS: set[str] = {
    "analysis.evoked_detection.threshold",
    "analysis.evoked_detection.min_duration",
    "analysis.qc_thresholds.min_snr",
    "analysis.qc_thresholds.max_baseline_var",
    "analysis.qc_thresholds.min_event_count",
    "analysis.qc_thresholds.max_motion_correlation",
    "advanced_analysis.methods",
    "advanced_analysis.save_detailed_results",
    "analysis.enhanced_metrics",
    "analysis.event_detection",
    "preprocessing.apply_cnmf",
    "preprocessing.use_gpu",
    "preprocessing.roi_detrend_degree",
    "preprocessing.spatial_hp_sigma",
    "roi_processing.optimization",
    "roi_processing.steps.save_masks_for_cnmf",
    "motion_correction.save_motion_data",
    "visualization.trace_color",
    "visualization.event_color",
    "visualization.save_summary_plots",
}


# --------------------------------------------------------------------------
# 3. Validation
# --------------------------------------------------------------------------

def _lookup(config: dict, dotted: str):
    node = config
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return None, False
        node = node[part]
    return node, True


def _type_ok(value, expected) -> bool:
    if expected is bool:
        return isinstance(value, bool)
    if expected is int:
        return isinstance(value, int) and not isinstance(value, bool)
    if isinstance(expected, tuple):
        return isinstance(value, expected) and not isinstance(value, bool)
    return isinstance(value, expected)


class ConfigValidationError(Exception):
    pass


def validate_config(config: dict, logger=None, strict: bool = True) -> dict:
    """Validate config against SCHEMA before any processing begins.

    Returns a report dict. Raises ConfigValidationError in strict mode when
    required keys are missing or mistyped.
    """
    log = logger or logging.getLogger(__name__)

    missing, wrong_type, dead_present = [], [], []

    for dotted, (expected, _desc) in SCHEMA.items():
        value, found = _lookup(config, dotted)
        if not found:
            missing.append(dotted)
        elif not _type_ok(value, expected):
            names = (
                "/".join(t.__name__ for t in expected)
                if isinstance(expected, tuple)
                else expected.__name__
            )
            wrong_type.append(f"{dotted}: expected {names}, got {type(value).__name__}")

    # per-condition blocks
    conditions, found = _lookup(config, "analysis.condition_specific")
    if found and isinstance(conditions, dict):
        for cond, block in conditions.items():
            if not isinstance(block, dict):
                wrong_type.append(f"analysis.condition_specific.{cond}: expected dict")
                continue
            for key, (expected, _d) in CONDITION_SCHEMA.items():
                if key not in block:
                    missing.append(f"analysis.condition_specific.{cond}.{key}")
                elif not _type_ok(block[key], expected):
                    wrong_type.append(
                        f"analysis.condition_specific.{cond}.{key}: bad type"
                    )
            # noise_criterion is optional, but if present must be complete
            nc = block.get("noise_criterion")
            if nc is not None:
                base = f"analysis.condition_specific.{cond}.noise_criterion"
                if not isinstance(nc, dict):
                    wrong_type.append(f"{base}: expected dict")
                else:
                    for key, (expected, _d) in NOISE_CRITERION_SCHEMA.items():
                        if key not in nc:
                            missing.append(f"{base}.{key}")
                        elif not _type_ok(nc[key], expected):
                            wrong_type.append(f"{base}.{key}: bad type")
                    if nc.get("method") not in (None, "mad", "std"):
                        wrong_type.append(
                            f"{base}.method: expected 'mad' or 'std', "
                            f"got {nc.get('method')!r}"
                        )

    for dotted in sorted(DEAD_KEYS):
        _, found = _lookup(config, dotted)
        if found:
            dead_present.append(dotted)

    if missing:
        log.error("CONFIG VALIDATION: %d required key(s) missing:", len(missing))
        for k in missing:
            default_note = ""
            log.error("    missing: %s%s", k, default_note)
    if wrong_type:
        log.error("CONFIG VALIDATION: %d key(s) with wrong type:", len(wrong_type))
        for k in wrong_type:
            log.error("    %s", k)
    if dead_present:
        log.info(
            "CONFIG VALIDATION: %d key(s) present but read by no code "
            "(harmless, but they do not affect the run):",
            len(dead_present),
        )
        for k in dead_present:
            log.info("    inert: %s", k)

    if not missing and not wrong_type:
        log.info(
            "CONFIG VALIDATION: OK - all %d schema keys present and well-typed.",
            len(SCHEMA),
        )

    report = {
        "missing": missing,
        "wrong_type": wrong_type,
        "inert": dead_present,
        "schema_key_count": len(SCHEMA),
    }

    if strict and (missing or wrong_type):
        raise ConfigValidationError(
            f"Config validation failed: {len(missing)} missing, "
            f"{len(wrong_type)} mistyped. See log above. "
            f"Run with strict=False to proceed using hardcoded defaults."
        )
    return report


# --------------------------------------------------------------------------
# 4. Provenance dump
# --------------------------------------------------------------------------

def _plain(obj):
    if isinstance(obj, dict):
        return {k: _plain(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_plain(v) for v in obj]
    return obj


def dump_resolved_config(config, output_dir, logger=None, validation_report=None):
    """Write the fully resolved config and a fallback report into output_dir.

    Produces:
      resolved_config.yaml  - exactly the parameters this run used
      config_provenance.json - validation result + any hardcoded defaults that
                               fired, so the run is self-documenting
    """
    log = logger or logging.getLogger(__name__)
    try:
        os.makedirs(output_dir, exist_ok=True)
        plain = _plain(config)

        if yaml is not None:
            path = os.path.join(output_dir, "resolved_config.yaml")
            with open(path, "w", encoding="utf-8") as f:
                yaml.safe_dump(plain, f, default_flow_style=False, sort_keys=False)
            log.info("Wrote resolved config to %s", path)

        fallbacks = get_fallback_report()
        prov = {
            "validation": validation_report or {},
            "hardcoded_defaults_used_main_process": fallbacks,
            "main_process_fully_config_driven": len(fallbacks) == 0,
            "note": (
                "Fallback tracking is per-process. Slices are processed in "
                "worker processes, whose fallbacks are logged as 'CONFIG "
                "FALLBACK' warnings in logs/ rather than aggregated here. "
                "Grep the logs for CONFIG FALLBACK to see all of them."
            ),
        }
        ppath = os.path.join(output_dir, "config_provenance.json")
        with open(ppath, "w", encoding="utf-8") as f:
            json.dump(prov, f, indent=2, default=str)

        if fallbacks:
            log.warning(
                "%d config key(s) fell back to hardcoded defaults this run - "
                "see %s",
                len(fallbacks),
                ppath,
            )
        else:
            log.info(
                "No config fallbacks in the main process. Check logs/ for "
                "'CONFIG FALLBACK' warnings from worker processes."
            )
        return ppath
    except Exception as e:  # never let provenance break a run
        log.warning("Could not write config provenance: %s", e)
        return None
