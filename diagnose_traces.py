"""Diagnose ROI traces that go strongly negative (dF/F < -1).

Why this matters
----------------
dF/F = (F - F0) / F0. A value of -1.0 means F reached zero; anything below -1
means F went negative, which is physically impossible for fluorescence. So a
trace reaching -1.5 is not reporting biology -- some processing stage has
driven the signal below zero.

The pipeline saves the trace at every stage under
<slice_dir>/intermediate_traces/. This script walks those stages for a given
ROI and reports where the signal first goes negative, which distinguishes:

  * BACKGROUND OVER-SUBTRACTION - the background trace rises above the ROI
    trace, so `roi - background` goes negative (roi_processing.py:811 performs
    this subtraction with no floor).
  * SIGN INVERSION - the final trace is anti-correlated with the raw trace,
    i.e. peaks became troughs.
  * SLOPE-CORRECTION OVERSHOOT - the linear trend fitted on slope_window is
    extrapolated across the whole recording; if the true bleaching is
    exponential, the extrapolation misfits badly late in the trace.

Usage
-----
    python diagnose_traces.py <slice_output_dir>
    python diagnose_traces.py <slice_output_dir> --roi 1 --roi 3
    python diagnose_traces.py <slice_output_dir> --plot
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# NOTE: "0_raw_traces.csv" is written by analysis.py from `fluorescence_data`,
# which is what analysis RECEIVES -- i.e. already background-subtracted. It is
# NOT the raw signal despite the name. "1_extracted_raw_traces.csv" (written by
# roi_processing.py) is the true pre-processing trace.
STAGES = [
    ("1_extracted_raw_traces.csv", "TRUE raw (ROI pixel means)"),
    ("0_raw_traces.csv", "analysis input (already bg-subtracted; misnamed)"),
    ("2_roi_detrended_traces.csv", "after ROI-specific detrend"),
    ("3_roi_traces_before_bg.csv", "ROI, before background subtraction"),
    ("3_background_traces.csv", "background trace"),
    ("3_roi_traces_slope_corrected.csv", "ROI, slope corrected"),
    ("3_background_traces_slope_corrected.csv", "background, slope corrected"),
    ("4_after_background_subtraction.csv", "after background subtraction"),
    ("2_df_f_traces.csv", "final dF/F"),
]


def load_stage(traces_dir: Path, filename: str):
    path = traces_dir / filename
    if not path.exists():
        return None
    df = pd.read_csv(path, index_col=0)
    return df.to_numpy(dtype=float)


def describe(arr, roi_idx):
    if arr is None or roi_idx >= arr.shape[0]:
        return None
    t = arr[roi_idx]
    return {
        "min": float(np.nanmin(t)),
        "max": float(np.nanmax(t)),
        "mean": float(np.nanmean(t)),
        "n_negative": int(np.sum(t < 0)),
        "frac_negative": float(np.mean(t < 0)),
        "argmin": int(np.nanargmin(t)),
        "trace": t,
    }


def diagnose_roi(traces_dir: Path, roi_idx: int, verbose=True):
    print("=" * 78)
    print(f"ROI {roi_idx + 1}")
    print("=" * 78)

    stages = {}
    for fname, label in STAGES:
        arr = load_stage(traces_dir, fname)
        d = describe(arr, roi_idx)
        if d is None:
            continue
        stages[fname] = (label, d)

    if not stages:
        print("  No intermediate traces found. Enable in config:")
        print("    roi_processing.save_intermediate_traces: true")
        return {}

    print(f"  {'stage':<42s} {'min':>10s} {'max':>10s} {'<0 frames':>10s}")
    print("  " + "-" * 74)
    first_negative = None
    for fname, (label, d) in stages.items():
        flag = ""
        if d["n_negative"] > 0:
            flag = "  <-- NEGATIVE"
            if first_negative is None:
                first_negative = (fname, label, d)
        print(
            f"  {label:<42s} {d['min']:>10.3f} {d['max']:>10.3f} "
            f"{d['n_negative']:>10d}{flag}"
        )

    print()

    # --- background over-subtraction check -----------------------------
    roi_pre = stages.get("3_roi_traces_slope_corrected.csv") or stages.get(
        "3_roi_traces_before_bg.csv"
    )
    bg = stages.get("3_background_traces_slope_corrected.csv") or stages.get(
        "3_background_traces.csv"
    )
    if roi_pre and bg:
        r, b = roi_pre[1]["trace"], bg[1]["trace"]
        n = min(len(r), len(b))
        r, b = r[:n], b[:n]
        exceed = b > r
        if exceed.any():
            first = int(np.argmax(exceed))
            worst = int(np.argmax(b - r))
            print("  BACKGROUND OVER-SUBTRACTION")
            print(
                f"    background exceeds ROI in {exceed.sum()} / {n} frames "
                f"({100*exceed.mean():.1f}%)"
            )
            print(f"    first at frame {first}, worst at frame {worst}")
            print(
                f"    at worst: ROI={r[worst]:.1f}, background={b[worst]:.1f}, "
                f"difference={r[worst]-b[worst]:.1f}"
            )
            print(
                "    -> `roi - background` (roi_processing.py:811) has no floor, "
                "so this drives F negative."
            )
        else:
            print("  Background never exceeds ROI: over-subtraction NOT the cause.")
        print()

    # --- sign inversion check ------------------------------------------
    raw = stages.get("1_extracted_raw_traces.csv")  # true raw only
    final = stages.get("2_df_f_traces.csv")
    if raw and final:
        a, c = raw[1]["trace"], final[1]["trace"]
        n = min(len(a), len(c))
        if n > 10:
            r = float(np.corrcoef(a[:n], c[:n])[0, 1])
            print(f"  SIGN CHECK: corr(raw, final dF/F) = {r:+.3f}")
            if r < -0.5:
                print("    -> strongly ANTI-correlated: the trace is INVERTED.")
            elif r > 0.5:
                print("    -> positively correlated: NOT inverted.")
            else:
                print(
                    "    -> weak correlation: processing has substantially "
                    "reshaped the trace (not a simple inversion)."
                )
        print()

    # --- dF/F sanity ----------------------------------------------------
    if final:
        d = final[1]
        below = int(np.sum(d["trace"] < -1.0))
        if below:
            print(
                f"  IMPOSSIBLE VALUES: dF/F < -1 in {below} frames "
                f"(min {d['min']:.2f}). F was negative there."
            )
        else:
            print("  dF/F stays above -1: physically plausible.")
    print()
    return stages


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("slice_dir", type=Path, help="Output dir for one slice")
    ap.add_argument(
        "--roi", type=int, action="append",
        help="1-based ROI number; repeatable. Default: scan all and report worst.",
    )
    ap.add_argument("--plot", action="store_true", help="Save a stage plot per ROI")
    args = ap.parse_args()

    traces_dir = args.slice_dir / "intermediate_traces"
    if not traces_dir.is_dir():
        sys.exit(f"No intermediate_traces/ under {args.slice_dir}")

    final = load_stage(traces_dir, "2_df_f_traces.csv")
    if final is None:
        sys.exit("No 2_df_f_traces.csv found.")

    if args.roi:
        rois = [r - 1 for r in args.roi]
    else:
        mins = np.nanmin(final, axis=1)
        bad = np.where(mins < -1.0)[0]
        print(f"\n{len(bad)} of {final.shape[0]} ROIs have dF/F < -1 "
              f"(physically impossible).\n")
        rois = list(bad[np.argsort(mins[bad])][:5]) if len(bad) else [
            int(np.argmin(mins))
        ]

    for roi_idx in rois:
        stages = diagnose_roi(traces_dir, roi_idx)
        if args.plot and stages:
            try:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt

                fig, axes = plt.subplots(
                    len(stages), 1, figsize=(11, 2.0 * len(stages)), sharex=True
                )
                if len(stages) == 1:
                    axes = [axes]
                for ax, (fname, (label, d)) in zip(axes, stages.items()):
                    ax.plot(d["trace"], lw=1.0)
                    ax.axhline(0, color="grey", ls="--", lw=0.8)
                    ax.set_ylabel(label, fontsize=7)
                    ax.tick_params(labelsize=7)
                axes[-1].set_xlabel("Frame")
                fig.suptitle(f"ROI {roi_idx+1} - stage by stage", fontsize=11)
                fig.tight_layout()
                out = args.slice_dir / f"diagnose_roi_{roi_idx+1:03d}.png"
                fig.savefig(out, dpi=110)
                plt.close(fig)
                print(f"  plot saved: {out}\n")
            except Exception as e:
                print(f"  (plot failed: {e})")


if __name__ == "__main__":
    main()
