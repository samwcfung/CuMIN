# utils.py

import os
import sys
import logging
import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime
import openpyxl # Ensure this is imported if not already
import xlsxwriter # Ensure this is imported if not already

# --- Define Consistent Column Orders ---

# Define the desired order for the main 'Summary' sheet in mouse/pain model files
# Define the desired order for the main 'Summary' sheet in mouse/pain model files
MOUSE_SUMMARY_COLUMN_ORDER = [
    "Slice Name",
    "Mouse ID",
    "Pain Model",
    "Condition",
    "Slice Type",
    "Slice Number",
    "Total ROIs",
    "Active ROIs",
    "Active ROI %",
    "Mean Distance to Lamina",
    
    # === Core Signal Metrics ===
    "Mean Peak Amplitude", "Std Peak Amplitude",
    "Mean Max Amplitude", "Std Max Amplitude",
    "Mean Mean Amplitude", "Std Mean Amplitude",
    "Mean Mean dF/F", "Std Mean dF/F",
    "Mean Max dF/F", "Std Max dF/F",
    "Mean Peak SNR", "Std Peak SNR",
    
    # === Enhanced Spontaneous Event Shape Metrics ===
    "Mean Peak Width", "Std Peak Width",
    "Mean Rise Time", "Std Rise Time",
    "Mean Decay Time", "Std Decay Time",
    "Mean Peak Asymmetry", "Std Peak Asymmetry",
    "Mean Area Under Curve", "Std Area Under Curve",
    "Mean Max Rise Slope", "Std Max Rise Slope",
    "Mean Avg Rise Slope", "Std Avg Rise Slope",
    "Mean Time Of Max Rise", "Std Time Of Max Rise",
    "Mean Overall Max Slope", "Std Overall Max Slope",
    "Mean Overall Avg Slope", "Std Overall Avg Slope",
    
    # === Temporal Pattern Metrics ===
    "Mean Peak Frequency", "Std Peak Frequency",
    "Mean Peak Count", "Std Peak Count",
    "Mean Spont Peak Frequency", "Std Spont Peak Frequency",
    "Mean Mean Iei", "Std Mean Iei",
    "Mean Cv Iei", "Std Cv Iei",
    "Mean Amplitude Cv", "Std Amplitude Cv",
    
    # === Signal Quality Metrics ===
    "Mean Baseline Variability", "Std Baseline Variability",
    
    # === Evoked Response Metrics ===
    "Mean Evoked Max Value", "Std Evoked Max Value",
    "Mean Evoked Rise Time", "Std Evoked Rise Time",
    "Mean Evoked Half Width", "Std Evoked Half Width",
    "Mean Evoked Duration", "Std Evoked Duration",
    "Mean Evoked Latency", "Std Evoked Latency",
    "Mean Evoked Time To Peak", "Std Evoked Time To Peak",
    "Mean Evoked Max Rise Slope", "Std Evoked Max Rise Slope",
    "Mean Evoked Avg Rise Slope", "Std Evoked Avg Rise Slope",
    "Mean Evoked Time Of Max Rise", "Std Evoked Time Of Max Rise",
    "Mean Evoked Decay Slope", "Std Evoked Decay Slope",
    "Mean Evoked Area", "Std Evoked Area",
    "Mean Evoked Reliability", "Std Evoked Reliability",
]

# Define the desired order for the 'PCA Metrics' sheet
PCA_METRICS_COLUMN_ORDER = [
    "Slice Name",
    "Mouse ID",
    "Pain Model",
    "Condition",
    "Slice Type",
    "Slice Number",
    "Active ROIs",
    
    # === Signal Quality ===
    "Mean Peak SNR",
    "Mean Peak Baseline Variability",
    
    # === Enhanced Peak Shape/Timing Metrics for Spontaneous Events ===
    "Mean Peak Amplitude",
    "Mean Peak Max Amplitude",
    "Mean Peak Mean Amplitude",
    "Mean Peak Width",
    "Mean Rise Time",
    "Mean Decay Time",
    "Mean Peak Asymmetry",
    "Mean Area Under Curve",
    "Mean Max Rise Slope",
    "Mean Avg Rise Slope",
    "Mean Time Of Max Rise",
    "Mean Overall Max Slope",
    "Mean Overall Avg Slope",
    
    # === Temporal Pattern Metrics ===
    "Mean Peak Frequency",
    "Mean Peak Count",
    "Mean Spont Peak Frequency",
    "Mean Mean Iei",
    "Mean Cv Iei",
    "Mean Amplitude Cv",
    
    # === Evoked Response Shape/Timing ===
    "Mean Evoked Max Value",
    "Mean Evoked Rise Time",
    "Mean Evoked Half Width",
    "Mean Evoked Duration",
    "Mean Evoked Latency",
    "Mean Evoked Time To Peak",
    "Mean Evoked Max Rise Slope",
    "Mean Evoked Avg Rise Slope",
    "Mean Evoked Time Of Max Rise",
    "Mean Evoked Decay Slope",
    "Mean Evoked Area",
    "Mean Evoked Reliability",
    
    # === Spectral Features ===
    "Mean Dominant Frequency",
    "Mean Spectral Centroid",
    "Mean Spectral Entropy",
    "Mean Power Ultra Low",
    "Mean Power Low",
    "Mean Power Mid",
    "Mean Power High",
    
    # === Baseline vs Response Spectral Features (for evoked conditions) ===
    "Mean Baseline Dominant Frequency",
    "Mean Baseline Spectral Centroid",
    "Mean Baseline Spectral Entropy",
    "Mean Baseline Power Ultra Low",
    "Mean Response Dominant Frequency",
    "Mean Response Spectral Centroid",
    "Mean Response Spectral Entropy",
    "Mean Response Power Ultra Low",
]


def setup_logging(log_file=None, process_id=None):
    # ... (setup_logging function remains the same) ...
    """
    Setup logging configuration.

    Parameters
    ----------
    log_file : str, optional
        Path to save log file, or None for console only
    process_id : int, optional
        Process ID for parallel processing

    Returns
    -------
    logging.Logger
        Configured logger object
    """
    # Create logger
    log_name = f"pipeline_process_{process_id}" if process_id is not None else "pipeline"
    logger = logging.getLogger(log_name)

    # Prevent duplicate handlers if called multiple times
    if logger.hasHandlers():
        logger.handlers.clear()

    logger.setLevel(logging.DEBUG) # Capture all levels

    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Create console handler (INFO level)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Create file handler (DEBUG level) if log_file is provided
    if log_file:
        try:
            # Create directory if it doesn't exist
            log_dir = os.path.dirname(log_file)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)

            file_handler = logging.FileHandler(log_file, mode='a') # Append mode
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except Exception as e:
            logging.error(f"Failed to set up file logging to {log_file}: {e}", exc_info=True)

    logger.propagate = False # Prevent messages from propagating to the root logger

    return logger


def save_slice_data(slice_name, metrics_df, output_dir, logger):
    # ... (save_slice_data function remains the same) ...
    """
    Save slice data (metrics) to CSV and Excel.

    Parameters
    ----------
    slice_name : str
        Name of the slice (used for filename).
    metrics_df : pandas.DataFrame
        DataFrame containing metrics for each ROI for this slice.
    output_dir : str
        Directory for the specific slice's output.
    logger : logging.Logger
        Logger object.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Created slice output directory: {output_dir}")

    # --- Save metrics ---
    # Prefer CSV for robustness and speed, Excel as secondary
    csv_path = os.path.join(output_dir, f"{slice_name}_metrics.csv")
    excel_path = os.path.join(output_dir, f"{slice_name}_metrics.xlsx")

    try:
        metrics_df.to_csv(csv_path, index=False)
        logger.debug(f"Saved slice metrics to CSV: {csv_path}")
    except Exception as e:
        logger.error(f"Failed to save slice metrics to CSV {csv_path}: {e}", exc_info=True)

    try:
        # Use openpyxl engine explicitly for better compatibility if needed
        metrics_df.to_excel(excel_path, index=False, engine='openpyxl')
        logger.debug(f"Saved slice metrics to Excel: {excel_path}")
    except Exception as e:
        logger.error(f"Failed to save slice metrics to Excel {excel_path}: {e}", exc_info=True)

    # JSON saving can be added back if needed, but ensure robust serialization
    # json_path = os.path.join(output_dir, f"{slice_name}_metrics.json")
    # try:
    #     metrics_dict = metrics_df.to_dict(orient='records')
    #     # Custom encoder or manual conversion might be needed for numpy types
    #     import json
    #     with open(json_path, 'w') as f:
    #         json.dump(metrics_dict, f, indent=2, default=lambda x: int(x) if isinstance(x, np.integer) else float(x) if isinstance(x, np.floating) else x.tolist() if isinstance(x, np.ndarray) else None)
    #     logger.debug(f"Saved slice metrics to JSON: {json_path}")
    # except Exception as e:
    #     logger.error(f"Failed to save slice metrics to JSON {json_path}: {e}", exc_info=True)


def _create_summary_rows(result, logger):
    """Helper function to create summary rows for a single slice result."""
    slice_name = result["slice_name"]
    # Use the CSV file path preferentially if it exists
    metrics_source = result.get("metrics_csv_file", result.get("metrics_file"))
    if not metrics_source or not os.path.exists(metrics_source):
         logger.error(f"Metrics file/CSV not found or not provided for slice {slice_name}. Cannot generate summary row.")
         return None, None # Indicate failure

    metadata = result["metadata"]
    pain_model = metadata.get("pain_model", "unknown")
    mouse_id = metadata.get("mouse_id", "unknown")

    try:
        # Prefer reading CSV as it's generally faster and less prone to type issues
        if metrics_source.endswith(".csv"):
            metrics_df = pd.read_csv(metrics_source)
        else: # Fallback to Excel
            metrics_df = pd.read_excel(metrics_source)

        if metrics_df.empty:
            logger.warning(f"Metrics data is empty for slice {slice_name}.")
            # Still create a row, but most values will be 0 or NaN
            n_rois = 0
        else:
             n_rois = len(metrics_df)

        # --- Create Slice Summary Row ---
        summary_row = {
            "Slice Name": slice_name,
            "Mouse ID": mouse_id,
            "Pain Model": pain_model,
            "Condition": metadata.get("condition", "unknown"),
            "Slice Type": metadata.get("slice_type", "unknown"),
            "Slice Number": metadata.get("slice_number", "1"),
            "Total ROIs": n_rois,
            "Active ROIs": metrics_df["is_active"].sum() if "is_active" in metrics_df.columns and n_rois > 0 else 0,
            "Active ROI %": (metrics_df["is_active"].sum() / n_rois * 100) if "is_active" in metrics_df.columns and n_rois > 0 else 0,
            # Add other basic stats using .get() for safety
            "Mean Distance to Lamina": metrics_df["distance_to_lamina"].mean() if "distance_to_lamina" in metrics_df.columns and n_rois > 0 else np.nan,
        }

        # --- Add Mean/Std Dev for ALL numeric metrics found ---
        # This dynamically adapts to the metrics actually present
        pca_metric_row_data = {} # Store data for the PCA sheet separately

        if n_rois > 0:
            numeric_cols = metrics_df.select_dtypes(include=np.number).columns
            for col in numeric_cols:
                # Skip identifiers or obviously non-metric columns if needed
                if col in ['roi_id', 'is_active', 'slice_number', 'distance_to_lamina', 'Total ROIs', 'Active ROIs']:
                    continue

                # Format display name (e.g., "peak_mean_amplitude" -> "Peak Mean Amplitude")
                display_name_parts = col.split('_')
                # Capitalize parts intelligently (e.g., df -> dF/F) - simple title case for now
                display_name = " ".join(part.capitalize() for part in display_name_parts)
                # Specific replacements if needed:
                display_name = display_name.replace("Df F", "dF/F").replace("Iei", "IEI").replace("Cv", "CV").replace("Snr", "SNR")

                mean_val = metrics_df[col].mean()
                std_val = metrics_df[col].std() if n_rois > 1 else 0 # Std dev is 0 for 1 ROI

                # Add to the main summary row
                summary_row[f"Mean {display_name}"] = mean_val
                summary_row[f"Std {display_name}"] = std_val

                # Also store the mean value for the PCA metrics row
                pca_metric_row_data[f"Mean {display_name}"] = mean_val


        # --- Create PCA Metrics Row ---
        # Start with the same basic info
        pca_row = summary_row.copy()
        # Remove Std columns from the PCA row (usually only means are used for PCA input)
        pca_row = {k: v for k, v in pca_row.items() if not k.startswith("Std ")}
        # Update/overwrite with the mean values we stored
        pca_row.update(pca_metric_row_data)


        return summary_row, pca_row

    except FileNotFoundError:
         logger.error(f"Metrics file not found when generating summary: {metrics_source}")
         return None, None
    except Exception as e:
        logger.error(f"Error processing metrics file {metrics_source} for summary: {e}", exc_info=True)
        return None, None


def save_mouse_summary(mouse_id, slice_results, output_dir, logger):
    """
    Create a summary Excel file for a mouse with consistent header order.

    Parameters
    ----------
    mouse_id : str
        Mouse identifier.
    slice_results : list
        List of slice result dictionaries for this mouse.
    output_dir : str
        Base output directory (summary will be saved here).
    logger : logging.Logger
        Logger object.
    """
    logger.info(f"Creating summary for mouse {mouse_id} with consistent headers.")

    summary_path = os.path.join(output_dir, f"{mouse_id}_summary.xlsx")

    all_slice_summary_data = []
    all_slice_pca_data = []

    # Process each slice result for this mouse
    for result in slice_results:
        if "error" in result:
            logger.warning(f"Skipping slice {result.get('slice_name', 'N/A')} for mouse {mouse_id} due to processing error: {result['error']}")
            continue

        summary_row, pca_row = _create_summary_rows(result, logger)
        if summary_row and pca_row:
            all_slice_summary_data.append(summary_row)
            all_slice_pca_data.append(pca_row)


    if not all_slice_summary_data:
        logger.warning(f"No valid slice data found to create summary for mouse {mouse_id}.")
        return

    # Create DataFrames from the collected rows
    summary_df = pd.DataFrame(all_slice_summary_data)
    pca_summary_df = pd.DataFrame(all_slice_pca_data)

    # --- Enforce Consistent Column Order ---
    # Get all columns present in the generated summary DF
    current_summary_cols = summary_df.columns.tolist()
    # Create the final order: start with the predefined ones, add any others found at the end
    final_summary_order = [col for col in MOUSE_SUMMARY_COLUMN_ORDER if col in current_summary_cols]
    final_summary_order.extend([col for col in current_summary_cols if col not in final_summary_order])
    summary_df = summary_df[final_summary_order] # Reorder

    # Repeat for PCA metrics sheet
    current_pca_cols = pca_summary_df.columns.tolist()
    final_pca_order = [col for col in PCA_METRICS_COLUMN_ORDER if col in current_pca_cols]
    final_pca_order.extend([col for col in current_pca_cols if col not in final_pca_order])
    pca_summary_df = pca_summary_df[final_pca_order] # Reorder


    # --- Write to Excel ---
    try:
        with pd.ExcelWriter(summary_path, engine='xlsxwriter') as writer:
            # Write Summary sheet
            summary_df.to_excel(writer, sheet_name="Summary", index=False)
            logger.debug(f"Writing 'Summary' sheet for {mouse_id}")

            # Write PCA Metrics sheet
            pca_summary_df.to_excel(writer, sheet_name="PCA Metrics", index=False)
            logger.debug(f"Writing 'PCA Metrics' sheet for {mouse_id}")

            # Add Slice Details Sheets (Optional but potentially useful)
            # Consider adding this if needed - writes raw metrics per slice
            # for result in slice_results:
            #     if "error" not in result and result.get("metrics_file"):
            #         try:
            #             slice_name = result["slice_name"]
            #             if result["metrics_file"].endswith(".csv"):
            #                  metrics_df = pd.read_csv(result["metrics_file"])
            #             else:
            #                  metrics_df = pd.read_excel(result["metrics_file"])
            #             # Create a safe sheet name
            #             sheet_name = re.sub(r'[\\/*?:\[\]]', '_', slice_name)[:31] # Max 31 chars, no invalid chars
            #             metrics_df.to_excel(writer, sheet_name=sheet_name, index=False)
            #         except Exception as e_sheet:
            #             logger.warning(f"Could not add sheet for slice {slice_name} to mouse summary: {e_sheet}")


            # --- Formatting ---
            workbook = writer.book
            pct_format = workbook.add_format({'num_format': '0.00%'}) # Format as percentage

            # Format Summary sheet
            summary_sheet = writer.sheets.get("Summary")
            if summary_sheet:
                 # Find 'Active ROI %' column index (case-insensitive)
                try:
                    active_roi_pct_col_idx = final_summary_order.index("Active ROI %")
                    # Apply format (A=0, B=1, ...)
                    summary_sheet.set_column(active_roi_pct_col_idx, active_roi_pct_col_idx, 15, pct_format)
                except ValueError:
                    logger.debug("'Active ROI %' column not found for formatting in Summary sheet.")
                 # Auto-adjust column widths (optional, can be slow)
                 # for i, col in enumerate(final_summary_order):
                 #    summary_sheet.set_column(i, i, max(len(col), summary_df[col].astype(str).str.len().max()) + 1)


            # Format PCA Metrics sheet (optional)
            pca_sheet = writer.sheets.get("PCA Metrics")
            # Add formatting if needed (e.g., adjusting widths)

        logger.info(f"Saved mouse summary with consistent headers to {summary_path}")

    except Exception as e:
        logger.error(f"Failed to write mouse summary Excel file {summary_path}: {e}", exc_info=True)


# --- NEW FUNCTION for Combined Pain Model Summary ---
def save_pain_model_summary(pain_model, all_results_for_model, output_dir, logger):
    """
    Create a combined Excel file for a specific pain model, aggregating
    ROI-level data and slice-level summaries.

    Parameters
    ----------
    pain_model : str
        Pain model identifier (e.g., 'SNI', 'CFA').
    all_results_for_model : list
        List of *all* slice result dictionaries for this pain model.
    output_dir : str
        Base output directory (combined file will be saved here).
    logger : logging.Logger
        Logger object.
    """
    logger.info(f"Creating combined summary for pain model '{pain_model}'. Processing {len(all_results_for_model)} slices.")

    combined_path = os.path.join(output_dir, f"{pain_model}_combined_summary.xlsx")

    # Data storage for ROI-level sheets
    roi_data_by_condition = {
        "0um": [],
        "10um": [],
        "25um": [],
        "unknown": [] # Store data with unknown conditions separately or merge later
    }

    # Data storage for slice-level summary sheets
    all_slice_summary_data = []
    all_slice_pca_data = []

    # --- Iterate through all slices for this pain model ---
    for result in all_results_for_model:
        if "error" in result:
            logger.warning(f"Skipping slice {result.get('slice_name', 'N/A')} for pain model {pain_model} due to processing error: {result['error']}")
            continue

        slice_name = result["slice_name"]
        metadata = result["metadata"]
        condition = metadata.get("condition", "unknown").lower() # Normalize condition
        mouse_id = metadata.get("mouse_id", "unknown")

        # --- Append to Slice-Level Summaries ---
        summary_row, pca_row = _create_summary_rows(result, logger)
        if summary_row and pca_row:
            all_slice_summary_data.append(summary_row)
            all_slice_pca_data.append(pca_row)
        else:
            logger.warning(f"Could not generate summary rows for slice {slice_name}, skipping its contribution to slice summaries.")


        # --- Append to ROI-Level Data ---
        # Use the CSV file path preferentially if it exists
        metrics_source = result.get("metrics_csv_file", result.get("metrics_file"))
        if not metrics_source or not os.path.exists(metrics_source):
             logger.warning(f"Metrics file/CSV not found for slice {slice_name}. Skipping its ROIs for combined sheets.")
             continue

        try:
            # Prefer reading CSV
            if metrics_source.endswith(".csv"):
                 metrics_df = pd.read_csv(metrics_source)
            else: # Fallback to Excel
                 metrics_df = pd.read_excel(metrics_source)

            if not metrics_df.empty:
                # Add identifying information to each ROI row
                metrics_df["slice_name"] = slice_name
                metrics_df["mouse_id"] = mouse_id
                metrics_df["pain_model"] = pain_model # Already known, but good for completeness
                # Condition, slice_type, slice_number should already be in metrics_df from pipeline

                # Append to the correct condition list
                target_condition_key = condition if condition in roi_data_by_condition else "unknown"
                roi_data_by_condition[target_condition_key].append(metrics_df)
                logger.debug(f"Added {len(metrics_df)} ROIs from {slice_name} (Condition: {condition}) to combined data.")
            else:
                logger.debug(f"Metrics DataFrame for {slice_name} is empty, no ROIs added.")

        except Exception as e:
            logger.error(f"Error reading metrics file {metrics_source} for ROI aggregation: {e}", exc_info=True)


    # --- Create Final DataFrames and Write to Excel ---
    if not all_slice_summary_data and not any(roi_data_by_condition.values()):
        logger.warning(f"No valid data found for pain model '{pain_model}'. Skipping combined file generation.")
        return

    try:
        with pd.ExcelWriter(combined_path, engine='xlsxwriter') as writer:

            # --- Write ROI-Level Sheets ---
            logger.info(f"Writing ROI-level sheets for {pain_model}...")
            # Define a comprehensive column order for ROI sheets
            # Start with identifiers, then include ALL possible metric columns
            # Get all unique columns across all conditions first
            all_roi_cols = set(["slice_name", "mouse_id", "pain_model", "condition", "slice_type", "slice_number", "roi_id"])
            for cond_key, df_list in roi_data_by_condition.items():
                if df_list:
                    for df in df_list:
                        all_roi_cols.update(df.columns)

            # Define preferred order (customize as needed) - FIXED: Split into two steps
            # Step 1: Define the base order with basic columns and categorized columns
            preferred_roi_order = [
                "pain_model", "mouse_id", "slice_name", "condition", "slice_type", "slice_number", "roi_id",
                "distance_to_lamina", "is_active", "baseline_fluorescence", "mean_df_f", "max_df_f", "std_df_f",
                # Add all peak_, spont_, evoked_, spectral_ columns found
                # Sort them alphabetically within categories for some consistency
            ]
            
            # Add categorized columns
            preferred_roi_order.extend(sorted([col for col in all_roi_cols if col.startswith("peak_")]))
            preferred_roi_order.extend(sorted([col for col in all_roi_cols if col.startswith("spont_")]))
            preferred_roi_order.extend(sorted([col for col in all_roi_cols if col.startswith("evoked_")]))
            preferred_roi_order.extend(sorted([col for col in all_roi_cols if col.startswith("baseline_") and "spectral" in col]))
            preferred_roi_order.extend(sorted([col for col in all_roi_cols if col.startswith("response_") and "spectral" in col]))
            
            # Step 2: Add any remaining columns that weren't included above
            remaining_cols = sorted([col for col in all_roi_cols 
                                   if col not in preferred_roi_order 
                                   and not any(col.startswith(p) for p in ["peak_","spont_","evoked_","baseline_","response_"])])
            preferred_roi_order.extend(remaining_cols)

            for condition_key, roi_list in roi_data_by_condition.items():
                sheet_name = condition_key # Use '0um', '10um', '25um', 'unknown'
                if roi_list:
                    logger.debug(f"Concatenating {len(roi_list)} DataFrames for condition '{condition_key}'...")
                    combined_roi_df = pd.concat(roi_list, ignore_index=True, sort=False) # Sort=False preserves original order before reindexing

                    # Ensure all potential columns are present, filled with NaN if missing
                    final_roi_cols_present = [col for col in preferred_roi_order if col in combined_roi_df.columns]
                    final_roi_cols_present.extend([col for col in combined_roi_df.columns if col not in final_roi_cols_present]) # Add any unexpected columns at the end

                    combined_roi_df = combined_roi_df.reindex(columns=final_roi_cols_present)

                    logger.debug(f"Writing sheet '{sheet_name}' with {len(combined_roi_df)} ROIs and {len(combined_roi_df.columns)} columns.")
                    combined_roi_df.to_excel(writer, sheet_name=sheet_name, index=False)
                else:
                    logger.info(f"No ROI data found for condition '{condition_key}' in pain model '{pain_model}'. Skipping sheet '{sheet_name}'.")


            # --- Write Slice-Level Summary Sheets ---
            logger.info(f"Writing slice-level summary sheets for {pain_model}...")
            if all_slice_summary_data:
                 slice_summary_df = pd.DataFrame(all_slice_summary_data)
                 slice_pca_df = pd.DataFrame(all_slice_pca_data)

                 # Enforce Consistent Column Order (using the same lists as mouse summary)
                 current_summary_cols = slice_summary_df.columns.tolist()
                 final_summary_order = [col for col in MOUSE_SUMMARY_COLUMN_ORDER if col in current_summary_cols]
                 final_summary_order.extend([col for col in current_summary_cols if col not in final_summary_order])
                 slice_summary_df = slice_summary_df[final_summary_order]

                 current_pca_cols = slice_pca_df.columns.tolist()
                 final_pca_order = [col for col in PCA_METRICS_COLUMN_ORDER if col in current_pca_cols]
                 final_pca_order.extend([col for col in current_pca_cols if col not in final_pca_order])
                 slice_pca_df = slice_pca_df[final_pca_order]

                 # Write sheets
                 slice_summary_df.to_excel(writer, sheet_name="Slice_summary", index=False)
                 slice_pca_df.to_excel(writer, sheet_name="Slice_PCA_metrics", index=False)

                 # --- Formatting --- Apply formatting similar to mouse summary
                 workbook = writer.book
                 pct_format = workbook.add_format({'num_format': '0.00%'})
                 summary_sheet = writer.sheets.get("Slice_summary")
                 if summary_sheet:
                     try:
                         active_roi_pct_col_idx = final_summary_order.index("Active ROI %")
                         summary_sheet.set_column(active_roi_pct_col_idx, active_roi_pct_col_idx, 15, pct_format)
                     except ValueError:
                         logger.debug("'Active ROI %' column not found for formatting in Slice_summary sheet.")

            else:
                 logger.warning(f"No slice summary data generated for pain model '{pain_model}'. Skipping Slice_summary/Slice_PCA_metrics sheets.")


        logger.info(f"Successfully saved combined summary for pain model '{pain_model}' to {combined_path}")

    except Exception as e:
        logger.error(f"Failed to write combined pain model summary Excel file {combined_path}: {e}", exc_info=True)


def calculate_lamina_distance(roi_masks, image_shape):
    # ... (calculate_lamina_distance function remains the same) ...
    """
    Calculate distance from each ROI center to the top of the image (lamina border).

    Parameters
    ----------
    roi_masks : list or numpy.ndarray
        List of 2D boolean masks or a 3D numpy array (n_masks, height, width).
    image_shape : tuple
        Tuple of (height, width) defining the image dimensions.

    Returns
    -------
    list
        List of distances (in pixels) from ROI center to top of image.
    """
    distances = []
    n_masks = len(roi_masks) if isinstance(roi_masks, list) else roi_masks.shape[0]

    for i in range(n_masks):
        mask = roi_masks[i]
        # Find ROI center coordinates
        y_indices, x_indices = np.where(mask)
        if len(y_indices) > 0:
            # Use median for potentially more robust center for irregular shapes
            center_y = int(np.median(y_indices))
            # Distance to top (y=0) is simply the y-coordinate
            distance = center_y
        else:
            # Handle empty mask case
            distance = np.nan # Use NaN for empty masks
            logging.warning(f"Empty mask found at index {i} when calculating distance to lamina.")

        distances.append(distance)

    return distances


def save_requirements_txt(output_dir):
    # ... (save_requirements_txt function remains the same) ...
    """
    Generate a requirements.txt file with common dependencies.

    Parameters
    ----------
    output_dir : str
        Directory to save the requirements file.
    """
    requirements = [
        "numpy>=1.21.0",        # Updated common versions
        "pandas>=1.3.0",
        "scipy>=1.7.0",
        "matplotlib>=3.4.0",
        "PyYAML>=6.0",
        "tifffile>=2021.11.2",
        "h5py>=3.6.0",
        "openpyxl>=3.0.9",      # For reading/writing .xlsx
        "xlsxwriter>=3.0.2",    # For writing .xlsx with formatting
        "tqdm>=4.62.0",         # Progress bars
        "scikit-image>=0.18.0", # Often used for image operations (check if needed by background/denoise)
        "roifile>=2.0.0",       # If using roifile for reading ImageJ ROIs
        # Add other essential libraries used across modules (e.g., seaborn for viz)
        "seaborn>=0.11.2",
    ]

    # Optional dependencies (consider checking if HAS_MOTION_CORRECTION etc.)
    optional_requirements = [
        "# Optional dependencies:",
        "# opencv-python>=4.5.5 # Required by some preprocessing/motion correction",
        "# pynrrd>=0.4.2 # If using NRRD format",
        "# caiman>=1.9.8 # If using CaImAn / CNMF / CNMF-E",
        "# normcorre>=1.5.1 # If using NoRMCorre motion correction",
    ]

    req_path = os.path.join(output_dir, "requirements.txt")
    try:
        with open(req_path, "w") as f:
            f.write("# Essential Dependencies\n")
            f.write("\n".join(requirements))
            f.write("\n\n")
            f.write("\n".join(optional_requirements))
        logging.info(f"Generated requirements file: {req_path}")
    except Exception as e:
        logging.error(f"Failed to write requirements file {req_path}: {e}")