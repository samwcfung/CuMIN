#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File Matcher Module (Robust Metadata Matching)
---------------------------------------------
Matches .tif video files with their corresponding .zip ROI files using
metadata extraction to handle inconsistent naming conventions, including
variable prefixes (e.g., RoiSet_) and suffixes (e.g., _cor).
"""

import os
import re
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Optional

# If using this standalone, configure basic logging
# logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)

def extract_metadata_from_filename(filename_stem: str, logger: logging.Logger) -> Optional[Dict[str, str]]:
    """
    Extract metadata from custom filename format, handling potential prefixes/suffixes.
    Searches for key patterns rather than relying solely on underscore splitting order.

    Expected pattern elements (order may vary slightly after prefix/suffix):
    - Mouse ID/Pain Model (e.g., CFA1, SNI4) - Often starts with letters, ends with digits.
    - Date (e.g., 7.23.20, 10-2-19) - Contains numbers and separators.
    - Slice Type/Number (e.g., ipsi1, contra2) - 'ipsi' or 'contra' followed by optional digits.
    - Condition (e.g., 0um, 10um, 25um) - Digits followed by 'um'.

    Parameters
    ----------
    filename_stem : str
        The filename stem (without extension) to analyze.
    logger : logging.Logger
        Logger object for logging warnings/debug info.

    Returns
    -------
    Optional[Dict[str, str]]
        Dictionary of extracted metadata, or None if essential components cannot be parsed.
    """
    logger.debug(f"--- Starting metadata extraction for stem: '{filename_stem}' ---")
    if not filename_stem:
        logger.warning("Received empty filename stem.")
        return None

    original_stem = filename_stem
    stem_to_parse = filename_stem

    # --- 1. Define patterns for different metadata components (same as before) ---
    patterns = {
        "condition": re.compile(r'^(\d+)um$', re.IGNORECASE),
        "slice_info": re.compile(r'^(ipsi|contra)(\d*)$', re.IGNORECASE),
        "date": re.compile(r'^\d{1,2}[.-]\d{1,2}[.-](?:\d{2}|\d{4})$'),
        "mouse_id_model": re.compile(r'^([A-Za-z]+)(\d*)$'),
    }

    # --- 2. Define Prefixes/Suffixes MORE EXPLICITLY ---
    # List known prefixes (case-insensitive). Add variations as needed.
    known_prefixes = ["roiset_", "Roiset_", "RoiSet_", "ROIset_", "RoiSET_", "ROISet_", "ROISET_", "ROIs_", "ROI_"]
    # List known suffixes (case-insensitive). Add variations as needed.
    known_suffixes = ["_cor", "_corrected", "_uncor"] # Add _uncor based on logs

    # --- 3. Clean the stem - Revised Logic ---
    # Remove Prefix (check from longest to shortest if overlaps are possible)
    prefix_removed = False
    for prefix in sorted(known_prefixes, key=len, reverse=True):
        # Use startswith for simple, case-insensitive check
        if stem_to_parse.lower().startswith(prefix.lower()):
            # Slice the string to remove the prefix
            original_length = len(stem_to_parse)
            stem_to_parse = stem_to_parse[len(prefix):]
            logger.debug(f"Applied prefix removal ('{prefix}'): '{original_stem}' -> '{stem_to_parse}'")
            prefix_removed = True
            break # Remove only the first matching prefix
    if not prefix_removed:
        logger.debug("No known prefix found.")

    # Remove Suffix (check from longest to shortest)
    suffix_removed = False
    for suffix in sorted(known_suffixes, key=len, reverse=True):
         # Use endswith for simple, case-insensitive check
        if stem_to_parse.lower().endswith(suffix.lower()):
            # Slice the string to remove the suffix
            original_value = stem_to_parse
            stem_to_parse = stem_to_parse[:-len(suffix)]
            logger.debug(f"Applied suffix removal ('{suffix}'): '{original_value}' -> '{stem_to_parse}'")
            suffix_removed = True
            break # Remove only the first matching suffix
    if not suffix_removed:
        logger.debug("No known suffix found.")


    logger.debug(f"Stem after cleaning: '{stem_to_parse}'")

    # --- 4. Split cleaned stem and attempt parsing ---
    parts = [p for p in stem_to_parse.split('_') if p]
    logger.debug(f"Parts after splitting cleaned stem: {parts}")

    metadata: Dict[str, Optional[str]] = {
        "mouse_id": None, "date": None, "pain_model": None,
        "slice_type": None, "slice_number": None, "condition": None,
        "original_stem": original_stem
    }
    used_part_indices = set()

    # --- 5. Parse by matching patterns (most distinct first - same logic as before) ---

    # 5a. Condition
    condition_found = False
    for i, part in enumerate(parts):
        if i in used_part_indices: continue
        match = patterns["condition"].match(part)
        if match:
            metadata["condition"] = f"{match.group(1)}um"
            logger.debug(f"Matched Condition: '{metadata['condition']}' from part '{part}' at index {i}")
            used_part_indices.add(i)
            condition_found = True
            break
    if not condition_found: logger.debug(f"Condition pattern ('##um') not found in remaining parts: {[p for i,p in enumerate(parts) if i not in used_part_indices]}")


    # 5b. Slice Info
    slice_found = False
    for i, part in enumerate(parts):
        if i in used_part_indices: continue
        match = patterns["slice_info"].match(part)
        if match:
            metadata["slice_type"] = match.group(1).capitalize()
            metadata["slice_number"] = match.group(2) or "1"
            logger.debug(f"Matched Slice Info: '{metadata['slice_type']}{metadata['slice_number']}' from part '{part}' at index {i}")
            used_part_indices.add(i)
            slice_found = True
            break
    if not slice_found: logger.debug(f"Slice pattern ('ipsi#', 'contra#') not found in remaining parts: {[p for i,p in enumerate(parts) if i not in used_part_indices]}")

    # 5c. Date
    date_found = False
    for i, part in enumerate(parts):
        if i in used_part_indices: continue
        match = patterns["date"].match(part)
        if match:
            metadata["date"] = part
            logger.debug(f"Matched Date: '{metadata['date']}' from part '{part}' at index {i}")
            used_part_indices.add(i)
            date_found = True
            break
    if not date_found: logger.debug(f"Date pattern (e.g., 'd.d.d') not found in remaining parts: {[p for i,p in enumerate(parts) if i not in used_part_indices]}")

    # --- 6. Infer Mouse ID / Model from remaining parts (same logic as before) ---
    remaining_parts_info = [(i, part) for i, part in enumerate(parts) if i not in used_part_indices]
    logger.debug(f"Remaining parts for Mouse ID/Model inference: {remaining_parts_info}")

    if remaining_parts_info:
        # Assume the first remaining part is the Mouse ID/Model component
        # Check if there are multiple significant remaining parts - could indicate parsing issue
        if len(remaining_parts_info) > 1:
             logger.warning(f"Multiple significant parts remain for Mouse ID inference in '{original_stem}': {[p for _,p in remaining_parts_info]}. Using the first: '{remaining_parts_info[0][1]}'.")

        mouse_part_index, mouse_part_value = remaining_parts_info[0]
        logger.debug(f"Attempting Mouse ID/Model parse on first remaining part: '{mouse_part_value}'")
        used_part_indices.add(mouse_part_index)

        match = patterns["mouse_id_model"].match(mouse_part_value)
        if match:
            metadata["pain_model"] = match.group(1).upper()
            mouse_number = match.group(2) or "1"
            metadata["mouse_id"] = f"{metadata['pain_model']}{mouse_number}"
            logger.debug(f"Parsed Mouse ID/Model: '{metadata['mouse_id']}' / '{metadata['pain_model']}'")
        else:
            metadata["mouse_id"] = mouse_part_value
            guessed_model = ''.join(filter(str.isalpha, mouse_part_value)).upper()
            metadata["pain_model"] = guessed_model if guessed_model else "UNKNOWN"
            logger.debug(f"Could not parse Mouse ID structure, using full part '{mouse_part_value}' as ID. Guessed Model: '{metadata['pain_model']}'")

        # Log any *other* remaining parts
        final_remaining_indices = [idx for idx, _ in remaining_parts_info[1:]]
        if final_remaining_indices:
             logger.warning(f"Potentially unused parts remain after parsing '{original_stem}': {[parts[i] for i in final_remaining_indices]}")

    else:
        logger.warning(f"No remaining parts found to infer Mouse ID/Model for '{original_stem}'")


    # --- 7. Final Validation (same logic as before) ---
    essential_keys = ["mouse_id", "date", "slice_type", "slice_number", "condition"]
    parsed_values = {k: v for k, v in metadata.items() if v is not None and k != 'original_stem'}
    missing_essential = [key for key in essential_keys if key not in parsed_values]

    if missing_essential:
         logger.warning(f"Failed to parse essential metadata for '{original_stem}'. Missing: {missing_essential}. Parsed: {parsed_values}. Returning None.")
         return None

    logger.info(f"Successfully parsed metadata for '{original_stem}': {parsed_values}")
    final_metadata: Dict[str, str] = {k: v for k, v in metadata.items() if v is not None}
    return final_metadata


def match_tif_and_roi_files(input_dir: str, logger: logging.Logger) -> List[Tuple[str, str]]:
    """
    Match .tif video files with their corresponding .zip ROI files using robust metadata comparison.

    Parameters
    ----------
    input_dir : str
        Directory containing .tif and .zip files (searches recursively).
    logger : logging.Logger
        Logger object.

    Returns
    -------
    List[Tuple[str, str]]
        List of tuples containing matched (absolute_tif_path, absolute_roi_path).
    """
    input_path = Path(input_dir)
    if not input_path.is_dir():
        logger.error(f"Input directory not found or is not a directory: {input_dir}")
        return []

    # Find all .tif and .zip files recursively
    tif_files: List[Path] = sorted(list(input_path.glob("**/*.tif")))
    zip_files: List[Path] = sorted(list(input_path.glob("**/*.zip")))

    logger.info(f"Found {len(tif_files)} .tif files and {len(zip_files)} .zip files in {input_dir} and subdirectories.")

    if not tif_files or not zip_files:
        logger.warning("No TIF or no ZIP files found, cannot perform matching.")
        return []

    matched_pairs: List[Tuple[str, str]] = []
    unmatched_tifs: List[Path] = []
    used_zip_files: set[Path] = set() # Keep track of zip files already matched

    # --- Pass 1: Try Exact Stem Matching ---
    logger.info("Attempting Pass 1: Exact stem matching...")
    # Create a lookup map for zip stems to their paths
    zip_stem_map: Dict[str, Path] = {zip_file.stem: zip_file for zip_file in zip_files}
    tif_files_remaining: List[Path] = [] # TIFs to process in Pass 2

    for tif_file in tif_files:
        tif_stem = tif_file.stem
        if tif_stem in zip_stem_map:
            zip_file = zip_stem_map[tif_stem]
            if zip_file not in used_zip_files:
                logger.debug(f"Exact match: '{tif_file.name}' -> '{zip_file.name}'")
                matched_pairs.append((str(tif_file.resolve()), str(zip_file.resolve())))
                used_zip_files.add(zip_file)
            else:
                logger.warning(f"Exact match TIF '{tif_file.name}' found, but corresponding ZIP '{zip_file.name}' was already used. Skipping exact match for this TIF.")
                tif_files_remaining.append(tif_file) # Try metadata matching later
        else:
            tif_files_remaining.append(tif_file) # No exact stem match, try metadata

    logger.info(f"Pass 1 complete. {len(matched_pairs)} pairs found via exact matching. {len(tif_files_remaining)} TIFs remain for metadata matching.")

    # --- Pass 2: Metadata Matching for Remaining TIFs ---
    if not tif_files_remaining:
        logger.info("All TIF files were matched exactly.")
        return matched_pairs

    logger.info("Attempting Pass 2: Metadata-based matching...")

    # Pre-parse metadata for available (unused) zip files for efficiency
    available_zip_files: List[Path] = [zf for zf in zip_files if zf not in used_zip_files]
    zip_metadata_cache: Dict[Path, Dict[str, str]] = {}
    for zip_file in available_zip_files:
        zip_stem = zip_file.stem
        meta = extract_metadata_from_filename(zip_stem, logger)
        if meta: # Only cache if metadata could be parsed reliably
            zip_metadata_cache[zip_file] = meta
        else:
            logger.warning(f"Could not parse reliable metadata for ZIP: {zip_file.name}. It will not be used for metadata matching.")

    # Define key fields that MUST match for a pair to be considered valid.
    # Adjust these based on the minimum unique identifiers in your filenames.
    required_match_keys = ["mouse_id", "date", "slice_type", "slice_number", "condition"]
    logger.info(f"Metadata matching requires agreement on keys: {required_match_keys}")

    for tif_file in tif_files_remaining:
        tif_stem = tif_file.stem
        logger.debug(f"Attempting metadata match for TIF: {tif_file.name}")
        tif_meta = extract_metadata_from_filename(tif_stem, logger)

        if not tif_meta:
            logger.warning(f"Could not parse reliable metadata for TIF: {tif_file.name}. Marking as unmatched.")
            unmatched_tifs.append(tif_file)
            continue

        # Find potential zip matches based on metadata
        current_potential_matches: List[Path] = []
        for zip_file_candidate, zip_meta in zip_metadata_cache.items():
             # Compare metadata dictionaries based on required keys
             is_metadata_match = True
             for key in required_match_keys:
                 # Both must have the key, and the values must be identical
                 if not (tif_meta.get(key) and zip_meta.get(key) and tif_meta[key] == zip_meta[key]):
                     is_metadata_match = False
                     # logger.debug(f"  Metadata mismatch on key '{key}': TIF='{tif_meta.get(key)}' vs ZIP='{zip_meta.get(key)}' for {zip_file_candidate.name}")
                     break # Mismatch on a required key

             if is_metadata_match:
                 logger.debug(f"  Potential metadata match found with ZIP: {zip_file_candidate.name}")
                 current_potential_matches.append(zip_file_candidate)

        # Handle results of metadata matching for this TIF
        if len(current_potential_matches) == 1:
            # Unique metadata match found
            zip_match = current_potential_matches[0]
            logger.info(f"Metadata match: '{tif_file.name}' -> '{zip_match.name}'")
            matched_pairs.append((str(tif_file.resolve()), str(zip_match.resolve())))
            used_zip_files.add(zip_match)
            # Remove the matched zip from the cache to prevent re-matching
            del zip_metadata_cache[zip_match]
        elif len(current_potential_matches) > 1:
            # Ambiguous match! Multiple available ZIPs fit the TIF metadata.
            logger.warning(f"AMBIGUOUS metadata match for TIF: '{tif_file.name}'. Multiple available ZIP files match:")
            for zp in current_potential_matches:
                logger.warning(f"  - {zp.name} (Metadata: {zip_metadata_cache.get(zp)})")
            logger.warning(f"  TIF '{tif_file.name}' will be treated as unmatched due to ambiguity.")
            unmatched_tifs.append(tif_file) # Treat as unmatched
        else:
            # No metadata match found among available ZIPs
            logger.info(f"No metadata match found for TIF: {tif_file.name}")
            unmatched_tifs.append(tif_file)


    # --- Final Logging ---
    logger.info(f"Completed matching. Total matched pairs: {len(matched_pairs)}")
    if unmatched_tifs:
        logger.warning(f"Could not find unique/unambiguous matches for {len(unmatched_tifs)} .tif files:")
        # Log paths relative to input_dir for potentially cleaner output
        try:
            unmatched_relative = [f.relative_to(input_path) for f in unmatched_tifs]
        except ValueError: # Handle case where files might be outside input_path unexpectedly
            unmatched_relative = unmatched_tifs

        for unmatched_path in unmatched_relative[:15]: # Log first few/many
            logger.warning(f"  - {unmatched_path}")
        if len(unmatched_tifs) > 15:
            logger.warning(f"  - ... and {len(unmatched_tifs) - 15} more.")
    # Log unused ZIPs as well, as this might indicate problems
    unused_zip_files = [zf for zf in zip_files if zf not in used_zip_files]
    if unused_zip_files:
        logger.warning(f"Found {len(unused_zip_files)} .zip files that were not matched to any TIF:")
        try:
            unused_relative = [f.relative_to(input_path) for f in unused_zip_files]
        except ValueError:
            unused_relative = unused_zip_files
        for unused_path in unused_relative[:15]:
            logger.warning(f"  - {unused_path}")
        if len(unused_zip_files) > 15:
            logger.warning(f"  - ... and {len(unused_zip_files) - 15} more.")


    if not unmatched_tifs and not unused_zip_files:
        logger.info("All found TIF files were successfully matched with unique ZIP files.")

    return matched_pairs


def group_files_by_mouse(file_pairs: List[Tuple[str, str]], logger: logging.Logger) -> Dict[str, List[Tuple[str, str]]]:
    """
    Group matched file pairs by mouse ID extracted from the TIF filename.

    Parameters
    ----------
    file_pairs : List[Tuple[str, str]]
        List of matched (absolute_tif_path, absolute_roi_path) pairs.
    logger : logging.Logger
        Logger object.

    Returns
    -------
    Dict[str, List[Tuple[str, str]]]
        Dictionary mapping mouse IDs to lists of file pairs.
        Uses "UNKNOWN_MOUSE" if mouse ID cannot be determined.
    """
    mouse_groups: Dict[str, List[Tuple[str, str]]] = {}

    for tif_path_str, roi_path_str in file_pairs:
        tif_path = Path(tif_path_str)
        tif_stem = tif_path.stem
        # Attempt to extract metadata just to get the mouse_id
        # Use a temporary logger or suppress warnings if desired during grouping
        metadata = extract_metadata_from_filename(tif_stem, logger)

        mouse_id = "UNKNOWN_MOUSE" # Default group
        if metadata and metadata.get("mouse_id"):
             mouse_id = metadata["mouse_id"]
        else:
             logger.warning(f"Could not determine mouse_id for TIF '{tif_path.name}' during grouping. Assigning to '{mouse_id}'.")

        if mouse_id not in mouse_groups:
            mouse_groups[mouse_id] = []

        mouse_groups[mouse_id].append((tif_path_str, roi_path_str)) # Store original strings

    logger.info(f"Grouped {len(file_pairs)} file pairs into {len(mouse_groups)} mouse groups.")
    # Log group sizes for quick overview
    for mid, pairs in mouse_groups.items():
        logger.debug(f"  Group '{mid}': {len(pairs)} pairs")

    return mouse_groups

# Example Usage Block (for testing purposes)
if __name__ == '__main__':
    # Setup basic logging for testing this module directly
    test_logger = logging.getLogger("MatcherTest")
    test_logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    if not test_logger.handlers: # Avoid adding handlers multiple times if re-run in interactive session
        test_logger.addHandler(handler)

    # Create a temporary directory structure for testing
    base_test_dir = Path("./temp_matching_test_area")
    base_test_dir.mkdir(exist_ok=True)
    test_subdir = base_test_dir / "subdir"
    test_subdir.mkdir(exist_ok=True)

    test_logger.info(f"Creating test files in {base_test_dir.resolve()}")

    files_to_create = {
        base_test_dir: [
            "CFA1_7.23.20_ipsi1_0um.tif", "ROIset_CFA1_7.23.20_ipsi1_0um.zip", # Metadata match with prefix
            "CFA1_7.23.20_ipsi2_0um.tif", "CFA1_7.23.20_ipsi2_0um.zip",       # Exact match
            "CFA2_7.24.20_contra1_10um_cor.tif", "CFA2_7.24.20_contra1_10um.zip", # Metadata match with suffix
            "SNI4_10.2.19_ipsi2_0um_cor.tif", "RoiSet_OA3_10.2.19_ipsi2_0um_cor.zip", # Mismatch example from user log
            "SNI4_10.2.19_ipsi2_0um_cor_Correct.zip", # The (presumably) correct zip for SNI4
            "AMBIG1_1.1.21_ipsi1_0um.tif", "AMBIG1_1.1.21_ipsi1_0um_roi1.zip", "AMBIG1_1.1.21_ipsi1_0um_roi2.zip", # Ambiguous ZIPs
            "ExactMatch_Test.tif", "ExactMatch_Test.zip",                     # Simple exact match
            "BadNameTif.tif", "BadNameZip.zip",                               # Will fail parsing
            "Partial_1.2.22_ipsi3_25um.tif", # Missing Mouse ID part
            "MouseOnly_CFA5.tif", "MouseOnly_CFA5.zip", # Missing other parts
            "Unmatched_Tif_1.tif",
            "Unused_Zip_1.zip",
        ],
        test_subdir: [
             "Subdir_Exact_Match.tif", "Subdir_Exact_Match.zip",
             "Subdir_Meta_Match_1.1.23_contra1_10um.tif", "ROI_Subdir_Meta_Match_1.1.23_contra1_10um.zip",
        ]
    }

    for dir_path, filenames in files_to_create.items():
        for fname in filenames:
            (dir_path / fname).touch() # Create empty files

    # --- Run Matching ---
    test_logger.info("\n" + "="*20 + " RUNNING MATCHER " + "="*20)
    matched = match_tif_and_roi_files(str(base_test_dir), test_logger)
    test_logger.info("\n--- Matched Pairs Found ---")
    if matched:
        for tif, roi in matched:
            # Show relative paths for clarity if possible
            try:
                rel_tif = Path(tif).relative_to(base_test_dir)
                rel_roi = Path(roi).relative_to(base_test_dir)
                test_logger.info(f"  TIF: {rel_tif} -> ROI: {rel_roi}")
            except ValueError:
                 test_logger.info(f"  TIF: {tif} -> ROI: {roi}") # Fallback to absolute
    else:
        test_logger.info("  No pairs matched.")

    # --- Run Grouping ---
    test_logger.info("\n" + "="*20 + " RUNNING GROUPING " + "="*20)
    grouped = group_files_by_mouse(matched, test_logger)
    test_logger.info("\n--- Files Grouped by Mouse ---")
    if grouped:
        for mouse, pairs in grouped.items():
            test_logger.info(f" Mouse Group: '{mouse}' ({len(pairs)} pairs)")
            for tif, roi in pairs:
                 try:
                     rel_tif = Path(tif).relative_to(base_test_dir)
                     rel_roi = Path(roi).relative_to(base_test_dir)
                     test_logger.info(f"    - TIF: {rel_tif} / ROI: {rel_roi}")
                 except ValueError:
                     test_logger.info(f"    - TIF: {tif} / ROI: {roi}")
    else:
        test_logger.info("  No groups formed (no matches found).")

    # --- Clean up ---
    test_logger.info("\n" + "="*20 + " CLEANUP " + "="*20)
    # Optional: Uncomment to automatically remove test files
    # import shutil
    # try:
    #     shutil.rmtree(base_test_dir)
    #     test_logger.info(f"Removed temporary test directory: {base_test_dir}")
    # except OSError as e:
    #     test_logger.error(f"Error removing test directory {base_test_dir}: {e}")