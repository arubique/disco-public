import pandas as pd
from IPython.display import display
import sys
import os
import numpy as np
import json
import argparse
from tqdm import tqdm


# local imports
sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    ),
)
from plots import (
    RESULTS_FOLDER,
    MAX_TABLE_SIZE,
    make_table_avg,
    make_perf_table,
)
from utils import load_pickle
from utils_for_notebooks import merge_methods

sys.path.pop(0)


NUM_ANCHORS = 100


def prepare_mae(mae_value, multiplier=1):
    if mae_value is None or mae_value == float("nan"):
        return float("nan")
    elif isinstance(mae_value, str):  # when mean +- std
        return mae_value
    else:
        return round(mae_value * multiplier, 2)


def prepare_rank(rank_value):
    if rank_value is None or rank_value == float("nan"):
        return float("nan")
    elif isinstance(rank_value, str):  # when mean +- std
        return rank_value
    else:
        return round(rank_value, 3)


def make_table_1(data_for_table_1):
    if len(data_for_table_1) == 2:
        print("DEBUG: debug mode, length of data_for_table_1 is 2")
        debug = True
    else:
        assert (
            len(data_for_table_1) == 8
        )  # mae and rank for mmlu, hellaswag, winogrande and arc [BENCHMARK]
        debug = False

    rows = []

    # Extract data using a more maintainable approach
    benchmark_names = ["mmlu", "hellaswag", "winogrande", "arc"]
    data_types = ["maes", "ranks"]

    # Initialize data storage
    benchmark_data = {}
    num_anchors_list = []

    # Extract MMLU data (always present)
    mmlu_maes, num_anchors_mmlu_maes = data_for_table_1[0]
    mmlu_ranks, num_anchors_mmlu_ranks = data_for_table_1[1]
    benchmark_data["mmlu"] = {"maes": mmlu_maes, "ranks": mmlu_ranks}
    num_anchors_list.extend([num_anchors_mmlu_maes, num_anchors_mmlu_ranks])
    num_anchors = num_anchors_mmlu_maes

    # Extract other benchmark data if not in debug mode
    if not debug:
        for i, benchmark in enumerate(benchmark_names[1:], start=2):
            maes_data, num_anchors_maes = data_for_table_1[i * 2 - 2]
            ranks_data, num_anchors_ranks = data_for_table_1[i * 2 - 1]
            benchmark_data[benchmark] = {"maes": maes_data, "ranks": ranks_data}
            num_anchors_list.extend([num_anchors_maes, num_anchors_ranks])

        # Verify all num_anchors values are the same
        assert all(na == num_anchors for na in num_anchors_list)

        # Handle None values by creating NaN-filled copies
        for benchmark in benchmark_names[1:]:  # Skip mmlu as it's the reference
            for data_type in data_types:
                if benchmark_data[benchmark][data_type] is None:
                    # Create a copy of the reference data and fill with NaN
                    benchmark_data[benchmark][data_type] = benchmark_data[
                        "mmlu"
                    ][data_type].copy()
                    benchmark_data[benchmark][data_type].loc[:, :] = float(
                        "nan"
                    )

    # Create convenience variables for backward compatibility
    if not debug:
        hellaswag_maes = benchmark_data["hellaswag"]["maes"]
        hellaswag_ranks = benchmark_data["hellaswag"]["ranks"]
        winogrande_maes = benchmark_data["winogrande"]["maes"]
        winogrande_ranks = benchmark_data["winogrande"]["ranks"]
        arc_maes = benchmark_data["arc"]["maes"]
        arc_ranks = benchmark_data["arc"]["ranks"]

    # [DUPLICATION]
    # [BENCHMARK]
    rows.append(
        [  # headers
            "Approach",
            "Condensation",  # type
            "Condensation",  # num_anchors
            "Prediction",  # type
            "MMLU",  # mae
            "MMLU",  # rank
            "hellaswag",  # mae
            "hellaswag",  # rank
            "winogrande",  # mae
            "winogrande",  # rank
            "arc",  # mae
            "arc",  # rank
        ][: 6 if debug else 12]
    )
    rows.append(
        [
            "",
            "type",  # type
            "num_anchors",  # num_anchors
            "type",  # type
            "mae",  # mae
            "rank",  # rank
            "mae",  # mae
            "rank",  # rank
            "mae",  # mae
            "rank",  # rank
            "mae",  # mae
            "rank",  # rank
        ][: 6 if debug else 12]
    )
    rows.append(
        [  # RANDOM direct eval
            "Baseline",
            "Random",
            num_anchors,
            "Eval",
            prepare_mae(mmlu_maes.loc["random"]["naive"], multiplier=100),
            prepare_rank(mmlu_ranks.loc["random"]["naive"]),
            None
            if debug
            else prepare_mae(
                hellaswag_maes.loc["random"]["naive"], multiplier=100
            ),
            None
            if debug
            else prepare_rank(hellaswag_ranks.loc["random"]["naive"]),
            None
            if debug
            else prepare_mae(
                winogrande_maes.loc["random"]["naive"], multiplier=100
            ),
            None
            if debug
            else prepare_rank(winogrande_ranks.loc["random"]["naive"]),
            None
            if debug
            else prepare_mae(arc_maes.loc["random"]["naive"], multiplier=100),
            None if debug else prepare_rank(arc_ranks.loc["random"]["naive"]),
        ][: 6 if debug else 12]
    )
    # tinyBenchmarks
    rows.append(
        [  # Random gp-IRT
            "tinyBenchmarks",
            "Random",
            num_anchors,
            "gp-IRT",
            prepare_mae(mmlu_maes.loc["random"]["gpirt"], multiplier=100),
            prepare_rank(mmlu_ranks.loc["random"]["gpirt"]),
            None
            if debug
            else prepare_mae(
                hellaswag_maes.loc["random"]["gpirt"], multiplier=100
            ),
            None
            if debug
            else prepare_rank(hellaswag_ranks.loc["random"]["gpirt"]),
            None
            if debug
            else prepare_mae(
                winogrande_maes.loc["random"]["gpirt"], multiplier=100
            ),
            None
            if debug
            else prepare_rank(winogrande_ranks.loc["random"]["gpirt"]),
            None
            if debug
            else prepare_mae(arc_maes.loc["random"]["gpirt"], multiplier=100),
            None if debug else prepare_rank(arc_ranks.loc["random"]["gpirt"]),
        ][: 6 if debug else 12]
    )
    rows.append(
        [  # anchor-IRT gp-IRT
            "tinyBenchmarks",
            "anchor-IRT",
            num_anchors,
            "gp-IRT",
            prepare_mae(mmlu_maes.loc["anchor-irt"]["gpirt"], multiplier=100),
            prepare_rank(mmlu_ranks.loc["anchor-irt"]["gpirt"]),
            None
            if debug
            else prepare_mae(
                hellaswag_maes.loc["anchor-irt"]["gpirt"], multiplier=100
            ),
            None
            if debug
            else prepare_rank(hellaswag_ranks.loc["anchor-irt"]["gpirt"]),
            None
            if debug
            else prepare_mae(
                winogrande_maes.loc["anchor-irt"]["gpirt"], multiplier=100
            ),
            None
            if debug
            else prepare_rank(winogrande_ranks.loc["anchor-irt"]["gpirt"]),
            None
            if debug
            else prepare_mae(
                arc_maes.loc["anchor-irt"]["gpirt"], multiplier=100
            ),
            None
            if debug
            else prepare_rank(arc_ranks.loc["anchor-irt"]["gpirt"]),
        ][: 6 if debug else 12]
    )
    rows.append(
        [  # anchor-correctness gp-IRT
            "tinyBenchmarks",
            "anchor-correctness",
            num_anchors,
            "gp-IRT",
            prepare_mae(mmlu_maes.loc["anchor"]["gpirt"], multiplier=100),
            prepare_rank(mmlu_ranks.loc["anchor"]["gpirt"]),
            None
            if debug
            else prepare_mae(
                hellaswag_maes.loc["anchor"]["gpirt"], multiplier=100
            ),
            None
            if debug
            else prepare_rank(hellaswag_ranks.loc["anchor"]["gpirt"]),
            None
            if debug
            else prepare_mae(
                winogrande_maes.loc["anchor"]["gpirt"], multiplier=100
            ),
            None
            if debug
            else prepare_rank(winogrande_ranks.loc["anchor"]["gpirt"]),
            None
            if debug
            else prepare_mae(arc_maes.loc["anchor"]["gpirt"], multiplier=100),
            None if debug else prepare_rank(arc_ranks.loc["anchor"]["gpirt"]),
        ][: 6 if debug else 12]
    )
    rows.append(
        [  # Random KNN
            "Baseline",
            "Random",
            num_anchors,
            "kNN",
            prepare_mae(mmlu_maes.loc["random"]["KNN"], multiplier=100),
            prepare_rank(mmlu_ranks.loc["random"]["KNN"]),
            None
            if debug
            else prepare_mae(
                hellaswag_maes.loc["random"]["KNN"], multiplier=100
            ),
            None
            if debug
            else prepare_rank(hellaswag_ranks.loc["random"]["KNN"]),
            None
            if debug
            else prepare_mae(
                winogrande_maes.loc["random"]["KNN"], multiplier=100
            ),
            None
            if debug
            else prepare_rank(winogrande_ranks.loc["random"]["KNN"]),
            None
            if debug
            else prepare_mae(arc_maes.loc["random"]["KNN"], multiplier=100),
            None if debug else prepare_rank(arc_ranks.loc["random"]["KNN"]),
        ][: 6 if debug else 12]
    )
    rows.append(
        [  # Random fit
            "Baseline",
            "Random",
            num_anchors,
            "fit",
            prepare_mae(mmlu_maes.loc["random"]["fit"], multiplier=100),
            prepare_rank(mmlu_ranks.loc["random"]["fit"]),
            None
            if debug
            else prepare_mae(
                hellaswag_maes.loc["random"]["fit"], multiplier=100
            ),
            None
            if debug
            else prepare_rank(hellaswag_ranks.loc["random"]["fit"]),
            None
            if debug
            else prepare_mae(
                winogrande_maes.loc["random"]["fit"], multiplier=100
            ),
            None
            if debug
            else prepare_rank(winogrande_ranks.loc["random"]["fit"]),
            None
            if debug
            else prepare_mae(arc_maes.loc["random"]["fit"], multiplier=100),
            None if debug else prepare_rank(arc_ranks.loc["random"]["fit"]),
        ][: 6 if debug else 12]
    )
    rows.append(
        [
            "DISCO (ours)",
            "High PDS",
            num_anchors,
            "kNN",
            prepare_mae(mmlu_maes.loc["highest"]["KNN"], multiplier=100),
            prepare_rank(mmlu_ranks.loc["highest"]["KNN"]),
            None
            if debug
            else prepare_mae(
                hellaswag_maes.loc["highest"]["KNN"], multiplier=100
            ),
            None
            if debug
            else prepare_rank(hellaswag_ranks.loc["highest"]["KNN"]),
            None
            if debug
            else prepare_mae(
                winogrande_maes.loc["highest"]["KNN"], multiplier=100
            ),
            None
            if debug
            else prepare_rank(winogrande_ranks.loc["highest"]["KNN"]),
            None
            if debug
            else prepare_mae(arc_maes.loc["highest"]["KNN"], multiplier=100),
            None if debug else prepare_rank(arc_ranks.loc["highest"]["KNN"]),
        ][: 6 if debug else 12]
    )
    rows.append(
        [
            "DISCO (ours)",
            "High PDS",
            num_anchors,
            "fit",
            prepare_mae(mmlu_maes.loc["highest"]["fit"], multiplier=100),
            prepare_rank(mmlu_ranks.loc["highest"]["fit"]),
            None
            if debug
            else prepare_mae(
                hellaswag_maes.loc["highest"]["fit"], multiplier=100
            ),
            None
            if debug
            else prepare_rank(hellaswag_ranks.loc["highest"]["fit"]),
            None
            if debug
            else prepare_mae(
                winogrande_maes.loc["highest"]["fit"], multiplier=100
            ),
            None
            if debug
            else prepare_rank(winogrande_ranks.loc["highest"]["fit"]),
            None
            if debug
            else prepare_mae(arc_maes.loc["highest"]["fit"], multiplier=100),
            None if debug else prepare_rank(arc_ranks.loc["highest"]["fit"]),
        ][: 6 if debug else 12]
    )

    df = pd.DataFrame(rows)

    latex_str = make_table_1_latex(df, debug=debug)

    return df, latex_str


def make_table_1_latex(df, debug=False):
    # Add column headers
    df.columns = [
        "Approach",
        "Type",
        "# Samples",
        "Type",
        "MAE",
        "Rank",
        "MAE",
        "Rank",
        "MAE",
        "Rank",
        "MAE",
        "Rank",
    ][
        : 6 if debug else 12
    ]  # [BENCHMARK]

    # Create LaTeX table content
    latex_str = "\\begin{table}[H]\n"
    latex_str += "\\centering\n\\small\n"
    if debug:
        latex_str += "\\begin{tabular}{c|cc|c|cc}\n"  # [BENCHMARK]
    else:
        latex_str += "\\begin{tabular}{c|cc|c|cc|cc|cc|cc}\n"  # [BENCHMARK]
    latex_str += "\\toprule\n"
    if debug:
        latex_str += "\\multicolumn{1}{c}{\\textbf{Approach}}&\\multicolumn{2}{c}{\\textbf{Condensation}} & \\multicolumn{1}{c}{\\textbf{Prediction}} & \\multicolumn{2}{c}{\\textbf{MMLU}} \\\\\n"  # [BENCHMARK]
    else:
        latex_str += "\\multicolumn{1}{c}{\\textbf{Approach}}&\\multicolumn{2}{c}{\\textbf{Condensation}} & \\multicolumn{1}{c}{\\textbf{Prediction}} & \\multicolumn{2}{c}{\\textbf{MMLU}}& \\multicolumn{2}{c}{\\textbf{hellaswag}}& \\multicolumn{2}{c}{\\textbf{winogrande}}& \\multicolumn{2}{c}{\\textbf{arc}} \\\\\n"  # [BENCHMARK]

    if debug:
        latex_str += (
            "&Type & \\# \\negthinspace Samples & Type & {MAE}  &Rank \\\\\n"
        )
    else:
        latex_str += "&Type & \\# \\negthinspace Samples & Type & {MAE}  &Rank& {MAE}  &Rank& {MAE}  &Rank& {MAE}  &Rank \\\\\n"
    latex_str += "\\toprule\n"

    # Process each row
    current_approach = ""
    for _, row in df.iterrows():
        if row["Approach"] == "Approach" or row["Approach"] == "":
            continue
        if row["Approach"] == current_approach:
            approach_str = ""
        else:
            approach_str = row["Approach"]
            current_approach = row["Approach"]

            # Add midrule before new approach except for first one
            if approach_str != "Baseline":
                latex_str += "\\midrule\n"

        def get_value(row, key, index):
            if isinstance(row[key], str):
                return row[key]
            elif isinstance(row[key], float):
                return row[key]
            else:
                return row[key].values[index]

        # Format numbers
        # [BENCHMARK]
        mae_mmlu_value = get_value(row, "MAE", 0)
        rank_mmlu_value = get_value(row, "Rank", 0)
        mae_mmlu = (
            "-"
            # if pd.isna(row["MAE"].values[0])
            # else f"{float(row['MAE'].values[0]):.2f}"
            if pd.isna(mae_mmlu_value)
            # else f"{float(mae_mmlu_value):.2f}"
            else prepare_mae(mae_mmlu_value)
        )
        rank_mmlu = (
            "-"
            if pd.isna(rank_mmlu_value)
            # else f"{float(rank_mmlu_value):.3f}"
            else prepare_rank(rank_mmlu_value)
        )
        if not debug:
            mae_hellaswag_value = get_value(row, "MAE", 1)
            rank_hellaswag_value = get_value(row, "Rank", 1)
            mae_winogrande_value = get_value(row, "MAE", 2)
            rank_winogrande_value = get_value(row, "Rank", 2)
            mae_arc_value = get_value(row, "MAE", 3)
            rank_arc_value = get_value(row, "Rank", 3)

            mae_hellaswag = (
                "-"
                if pd.isna(mae_hellaswag_value)
                # else f"{float(mae_hellaswag_value):.2f}"
                else prepare_mae(mae_hellaswag_value)
            )
            rank_hellaswag = (
                "-"
                if pd.isna(rank_hellaswag_value)
                # else f"{float(rank_hellaswag_value):.3f}"
                else prepare_rank(rank_hellaswag_value)
            )
            mae_winogrande = (
                "-"
                if pd.isna(mae_winogrande_value)
                # else f"{float(mae_winogrande_value):.2f}"
                else prepare_mae(mae_winogrande_value)
            )
            rank_winogrande = (
                "-"
                if pd.isna(rank_winogrande_value)
                # else f"{float(rank_winogrande_value):.3f}"
                else prepare_rank(rank_winogrande_value)
            )
            mae_arc = (
                "-"
                if pd.isna(mae_arc_value)
                # else f"{float(mae_arc_value):.2f}"
                else prepare_mae(mae_arc_value)
            )
            rank_arc = (
                "-"
                if pd.isna(rank_arc_value)
                # else f"{float(rank_arc_value):.3f}"
                else prepare_rank(rank_arc_value)
            )

        # Bold best results
        if approach_str == "DISCO (ours)" and row["Type"].values[1] == "linear":
            mae_mmlu = f"\\textbf{{{mae_mmlu}}}"
            rank_mmlu = f"\\textbf{{{rank_mmlu}}}"
        # [BENCHMARK]
        if debug:
            latex_str += f"{approach_str}&{row['Type'].values[0]} & {row['# Samples']} & {row['Type'].values[1]} & {mae_mmlu} &{rank_mmlu} \\\\\n"
        else:
            latex_str += f"{approach_str}&{row['Type'].values[0]} & {row['# Samples']} & {row['Type'].values[1]} & {mae_mmlu} &{rank_mmlu} & {mae_hellaswag} &{rank_hellaswag} & {mae_winogrande} &{rank_winogrande} & {mae_arc} &{rank_arc} \\\\\n"

    latex_str += "\\bottomrule\n"
    latex_str += "\\end{tabular}\n"
    latex_str += "\\vspace{1em}\n"
    latex_str += "\\caption{Mean Absolute Error (MAE) and Pearson Rank Correlation (Rank) for different sampling and prediction strategies. For question answering task on MMLU, Hellaswag, Winogrande and Arc datasets."
    latex_str += "\\label{tab:language-main}\n"
    latex_str += "\\end{table}"

    # Store LaTeX code in DataFrame metadata
    df.attrs["latex_table"] = latex_str
    return latex_str


def extract_data_for_table_1_v2(
    source_df, num_anchors, lower_better, key="PDS type", target_df=None
):
    # Default target to source for verification mode
    if target_df is None:
        target_df = source_df

    # Slice to the specific anchors level
    df_s = source_df[num_anchors]
    df_t = target_df[num_anchors]

    # if key == "stratified":
    #     df_s = df_s[df_s["PDS type"] == "highest"]
    #     df_t = df_t[df_t["PDS type"] == "highest"]

    # Split by key presence
    nan_rows_s = df_s[df_s[key].isna()]
    non_nan_rows_s = df_s[df_s[key].notna()]

    nan_rows_t = df_t[df_t[key].isna()]
    non_nan_rows_t = df_t[df_t[key].notna()]

    # Aggregated per-group values (min/max over rows) on SOURCE
    if lower_better:
        grouped_non_nan_s = non_nan_rows_s.groupby(
            key, as_index=(key == "PDS type")
        ).min()
    else:
        grouped_non_nan_s = non_nan_rows_s.groupby(
            key, as_index=(key == "PDS type")
        ).max()
    grouped_df_source = pd.concat([grouped_non_nan_s, nan_rows_s])

    # For coordinate selection, get the index (row label) per group and per column on SOURCE
    if lower_better:
        selected_rows_source = non_nan_rows_s.groupby(
            key, as_index=(key == "PDS type")
        ).idxmin()
    else:
        selected_rows_source = non_nan_rows_s.groupby(
            key, as_index=(key == "PDS type")
        ).idxmax()

    # Construct TARGET aggregated dataframe by taking values at selected coordinates
    meta_cols = {key, "stratified", "#guiding_models", "cirt", "pirt"}
    model_cols = [
        c
        for c in selected_rows_source.columns
        if c in non_nan_rows_t.columns and c not in meta_cols
    ]

    grouped_non_nan_t = pd.DataFrame(index=selected_rows_source.index)
    for col in model_cols:
        idx_labels = selected_rows_source[col]
        grouped_non_nan_t[col] = (
            non_nan_rows_t[col].reindex(idx_labels).to_numpy()
        )

    # Copy over metadata columns from SOURCE aggregated groups to maintain shape and info
    for c in meta_cols:
        if c in grouped_non_nan_s.columns:
            grouped_non_nan_t[c] = grouped_non_nan_s[c]

    grouped_df_target = pd.concat([grouped_non_nan_t, nan_rows_t])

    # Drop specified columns (apply same rule to both, using SOURCE columns as reference)
    to_drop = [
        "MLP3_e700_lr0.001",
        "Ridge_10",
        "Lasso_e-4",
        "GradientBoostingRegressor_200",
    ]
    to_drop = [col for col in to_drop if col in grouped_df_source.columns]
    grouped_df_source = grouped_df_source.drop(columns=to_drop)
    grouped_df_target = grouped_df_target.drop(
        columns=[c for c in to_drop if c in grouped_df_target.columns]
    )

    # Columns to consider for the row-wise min/max to produce 'fit'
    min_cols = [
        "MLP3_e700_lr0.001",
        "Ridge_10",
        "Lasso_e-4",
        "RandomForestRegressor_100",
        "GradientBoostingRegressor_200",
    ]
    min_cols = [col for col in min_cols if col in grouped_df_source.columns]

    # Row-wise argmin/argmax on SOURCE across min_cols to select the column per row
    if len(min_cols) > 0:
        # Get the subset of data for min/max operations
        data_subset = grouped_df_source[min_cols]

        # # Check for rows with all NaN values and handle them
        # all_nan_rows = data_subset.isna().all(axis=1)

        # Handle all-NaN rows before idxmin/idxmax to avoid FutureWarning
        # all_nan_rows = data_subset.isna().all(axis=1)
        # fill_value = float('inf') if lower_better else float('-inf')
        # data_subset = data_subset.fillna(fill_value).infer_objects(copy=False)
        # Handle all-NaN rows before idxmin/idxmax to avoid FutureWarning

        # Create a copy to avoid modifying the original data
        data_subset_clean = data_subset.copy()

        # Fill NaN values with infinity (or -infinity) based on whether we want min or max
        # Use numpy operations to avoid pandas downcasting warnings
        fill_value = np.inf if lower_better else -np.inf
        for col in data_subset_clean.columns:
            # Use numpy where to avoid fillna() downcasting warnings
            mask = data_subset_clean[col].isna()
            data_subset_clean[col] = np.where(
                mask, fill_value, data_subset_clean[col]
            )

        # For rows that were all NaN, set their values back to NaN

        # FutureWarning: The behavior of DataFrame.idxmin with all-NA values, or any-NA and skipna=False, is deprecated. In a future version this will raise ValueError
        if lower_better:
            # Use skipna=True to avoid the warning, then handle NaN results
            best_col_per_row = data_subset_clean.idxmin(axis=1, skipna=True)
        else:
            # Use skipna=True to avoid the warning, then handle NaN results
            best_col_per_row = data_subset_clean.idxmax(axis=1, skipna=True)

        # For rows where all values are NaN, idxmin/idxmax will return NaN
        # We can either drop these rows or handle them as needed
        # For now, we'll keep the NaN values and let the downstream code handle them

        # Extract corresponding TARGET values from the selected column
        # Use safe lookup without deprecated DataFrame.lookup
        # Align rows
        grouped_df_target_aligned = grouped_df_target.reindex(
            best_col_per_row.index
        )
        # Get integer column positions for each chosen column
        col_positions = grouped_df_target_aligned.columns.get_indexer(
            best_col_per_row
        )
        row_positions = np.arange(len(grouped_df_target_aligned))
        fit_values = grouped_df_target_aligned.to_numpy()[
            row_positions, col_positions
        ]

        # Build final output from TARGET values: drop min_cols and add 'fit'
        grouped_df_out = grouped_df_target.drop(
            columns=[c for c in min_cols if c in grouped_df_target.columns]
        )
        grouped_df_out["fit"] = fit_values
    else:
        grouped_df_out = grouped_df_target.copy()

    # Drop metadata columns that are not meaningful after grouping
    for cols_to_drop in ["stratified", "#guiding_models", "cirt", "pirt"]:
        if cols_to_drop == key:
            continue
        if cols_to_drop in grouped_df_out.columns:
            grouped_df_out = grouped_df_out.drop(cols_to_drop, axis=1)

    # Verification: when target is source, results should match original function
    if target_df is source_df:
        legacy_df, _ = extract_data_for_table_1(
            source_df, num_anchors, lower_better, key=key
        )
        # Align columns order for comparison
        common_cols = [
            c for c in legacy_df.columns if c in grouped_df_out.columns
        ]
        assert legacy_df[common_cols].equals(
            grouped_df_out[common_cols]
        ), "v2 output does not match legacy when target==source"

    return grouped_df_out, num_anchors


def extract_data_for_table_1(
    source_df, num_anchors, lower_better, key="PDS type"
):
    # Group by PDS type and calculate best for each group
    df = source_df[num_anchors]

    # Keep rows with NaN PDS type and group the rest
    nan_rows = df[df[key].isna()]
    non_nan_rows = df[df[key].notna()]
    # get best across all PDS types
    if lower_better:
        grouped_non_nan = non_nan_rows.groupby(
            key, as_index=(key == "PDS type")
        ).min()
    else:
        grouped_non_nan = non_nan_rows.groupby(
            key, as_index=(key == "PDS type")
        ).max()
    grouped_df_source = pd.concat([grouped_non_nan, nan_rows])

    # to_drop = ["GradientBoostingRegressor_200"]
    # to_drop = ['MLP3_e700_lr0.001', 'Ridge_10', 'Lasso_e-4', 'RandomForestRegressor_100']
    to_drop = [
        "MLP3_e700_lr0.001",
        "Ridge_10",
        "Lasso_e-4",
        "GradientBoostingRegressor_200",
    ]  # keep only Random Forest
    to_drop = [col for col in to_drop if col in grouped_df_source.columns]

    grouped_df_source = grouped_df_source.drop(columns=to_drop)

    # Get the columns to find minimum across
    min_cols = [
        "MLP3_e700_lr0.001",
        "Ridge_10",
        "Lasso_e-4",
        "RandomForestRegressor_100",
        "GradientBoostingRegressor_200",
    ]
    min_cols = [col for col in min_cols if col in grouped_df_source.columns]

    # Find minimum value across specified columns and store in new 'linear' column
    if lower_better:
        grouped_df_source["fit"] = grouped_df_source[min_cols].min(axis=1)
    else:
        grouped_df_source["fit"] = grouped_df_source[min_cols].max(axis=1)

    # Drop the original columns
    grouped_df = grouped_df_source.drop(columns=min_cols)

    # Drop the stratified and #guiding_models columns since they're no longer meaningful after grouping
    for cols_to_drop in ["stratified", "#guiding_models", "cirt", "pirt"]:
        if cols_to_drop == key:
            continue
        if cols_to_drop in grouped_df.columns:
            grouped_df = grouped_df.drop(cols_to_drop, axis=1)

    return grouped_df, num_anchors


def make_df_with_results(table_avg, table_std, bench, split, extract_std=False):
    cur_methods_for_table = table_avg[bench][split].keys()

    df = make_perf_table(
        table_avg[bench][split],
        table_std[bench][split],
        methods=cur_methods_for_table,
        extract_std=extract_std,
    )

    pd.set_option("display.max_rows", MAX_TABLE_SIZE)
    pd.set_option("display.max_columns", MAX_TABLE_SIZE)
    pd.set_option("display.max_colwidth", MAX_TABLE_SIZE)
    for num_samples in df.keys():
        # print("#anchor_points:", num_samples)
        # Reorder columns to put guiding models, PDS type, and stratified first
        cols = df[num_samples].columns.tolist()
        first_cols = ["#guiding_models", "PDS type", "stratified"]
        other_cols = [col for col in cols if col not in first_cols]
        df[num_samples] = df[num_samples][first_cols + other_cols]

        # Replace all values in #guiding_models column with 382
        df[num_samples].loc[
            df[num_samples]["#guiding_models"] == "all", "#guiding_models"
        ] = 382

        # Sort rows by #guiding_models
        df[num_samples] = df[num_samples].sort_values(
            ["PDS type", "stratified", "#guiding_models"]
        )

        # print(df[num_samples])

    # df[max(list(df.keys()))].to_csv(results_table_path)
    return df


def process_table_data(
    bench,
    split,
    filename_suffix,
    data,
    scenarios_to_skip,
    ordered,
    agg_type,
    table_avg_base,
    table_std_base,
    model_perf_base,
    std_across_models=False,
):
    current_table_avg, current_table_std, current_model_perf = make_table_avg(
        bench,
        split,
        filename_suffix,
        data,
        scenarios_to_skip=scenarios_to_skip,
        ordered=ordered,
        return_perf_table=True,
        agg_type=agg_type,
        std_across_models=std_across_models,
    )
    table_avg_base = merge_methods(table_avg_base, current_table_avg)
    table_std_base = merge_methods(table_std_base, current_table_std)
    model_perf_base = merge_methods(model_perf_base, current_model_perf)
    return table_avg_base, table_std_base, model_perf_base


def extract_results_needed_for_tables(
    results_suffixes, return_only_df=False, target_df_dict=None
):
    def make_df_key(bench, split, agg_type):
        return f"{bench}_{split}_{agg_type}"

    df_ablation = None
    df_num_anchors = None
    df_iid = None

    if target_df_dict is None:
        target_df_ablation = None
        target_df_num_anchors = None
        target_df_iid = None
    else:
        target_df_ablation = target_df_dict["ablation"]
        target_df_num_anchors = target_df_dict["num_anchors"]
        target_df_iid = target_df_dict["iid"]

    scenarios_to_skip = []
    table_1_data = []
    table_1_data_iid = []
    figure_n_anchors_data = {}
    umap_pca_dfs = {}  # different number of PCA/UMAP dimensions
    prediction_strategy_dfs = (
        {}
    )  # different prediction strategies (linear, mlps, RandomForests etc.)
    num_models_df = None  # different number of source models
    ablation_strat = None  # stratified / non-stratified

    table_avg_dict = {}
    table_std_dict = {}
    model_perf_dict = {}

    # iterate over benchmarks or ablation names
    for bench, per_bench in tqdm(results_suffixes.items()):
        if bench not in table_avg_dict:
            table_avg_dict[bench] = {}
            table_std_dict[bench] = {}
            model_perf_dict[bench] = {}
        # ordered = bench in ["mmlu_fields", "hellaswag", "num_models", "umap_pca"]

        # ordered means chronological split
        ordered = True
        # iterate over aggregation types
        for agg_type in ["mae", "rank"]:
            if agg_type not in table_avg_dict[bench]:
                table_avg_dict[bench][agg_type] = {}
                table_std_dict[bench][agg_type] = {}
                model_perf_dict[bench][agg_type] = {}

            # when bench means ablation name (num_models, umap_pca, prediction_strategy)
            if bench in ["num_models", "umap_pca", "prediction_strategy"]:
                if agg_type == "mae" and bench not in [
                    "umap_pca",
                    "prediction_strategy",
                ]:
                    continue

                # all ablations are done on mmlu_fields and noniid (chornological) split
                split = "noniid"
                real_bench = "mmlu_fields"

                factor_list = []
                for factor, filename_suffix in per_bench.items():
                    table_avg_base = None
                    table_std_base = None
                    model_perf_base = None

                    if factor not in table_avg_dict[bench]:
                        table_avg_dict[bench][factor] = {}
                        table_std_dict[bench][factor] = {}
                        model_perf_dict[bench][factor] = {}

                    results_path = f"{RESULTS_FOLDER}/accs_{real_bench}_split-{split}_iterations-5{filename_suffix}.pickle"

                    data = load_pickle(results_path)

                    (
                        table_avg_base,
                        table_std_base,
                        model_perf_base,
                    ) = process_table_data(
                        real_bench,
                        split,
                        filename_suffix,
                        data,
                        scenarios_to_skip,
                        ordered,
                        agg_type,
                        table_avg_base,
                        table_std_base,
                        model_perf_base,
                        std_across_models=False,
                    )

                    table_avg_dict[bench][agg_type][factor] = table_avg_base
                    table_std_dict[bench][agg_type][factor] = table_std_base
                    model_perf_dict[bench][agg_type][factor] = model_perf_base

                    if df_ablation is None:
                        df_ablation = {}
                    df_ablation[
                        make_df_key(bench, split, agg_type)
                    ] = make_df_with_results(
                        table_avg_base,
                        table_std_base,
                        real_bench,
                        split,
                        extract_std=return_only_df,
                    )
                    if not return_only_df:
                        # ablations for prediction_strategy - linear models, mlps, RandomForests etc.
                        if bench == "prediction_strategy":
                            filtered_df = df_ablation[
                                make_df_key(bench, split, agg_type)
                            ][NUM_ANCHORS]

                            filtered_df = filtered_df[
                                filtered_df["PDS type"] == "highest"
                            ]
                            if factor == "other":
                                if agg_type == "rank":
                                    filtered_df = (
                                        filtered_df.loc[
                                            filtered_df[
                                                "RandomForestRegressor_100"
                                            ].idxmax()
                                        ]
                                        .to_frame()
                                        .T
                                    )
                                else:
                                    filtered_df = (
                                        filtered_df.loc[
                                            filtered_df[
                                                "RandomForestRegressor_100"
                                            ].idxmin()
                                        ]
                                        .to_frame()
                                        .T
                                    )

                                filtered_df.drop(
                                    columns=["MLP3_e700_lr0.001"], inplace=True
                                )
                            else:
                                assert factor in ["linear", "mlps"]
                                if agg_type == "rank":
                                    filtered_df = pd.DataFrame(
                                        [filtered_df.max()], index=["highest"]
                                    )
                                else:
                                    filtered_df = pd.DataFrame(
                                        [filtered_df.min()], index=["highest"]
                                    )

                                if factor == "linear":
                                    filtered_df = filtered_df[
                                        ["LinearRegression"]
                                    ]
                                else:
                                    assert factor == "mlps"
                                    filtered_df = filtered_df[
                                        [
                                            "MLP2_e200_lr0.001",
                                            "MLP3_e700_lr0.001",
                                        ]
                                    ]

                            cols_to_drop = [
                                "PDS type",
                                "stratified",
                                "#guiding_models",
                            ]
                            cols_to_drop = [
                                col
                                for col in cols_to_drop
                                if col in filtered_df.columns
                            ]
                            filtered_df.drop(columns=cols_to_drop, inplace=True)
                            factor_list.append(filtered_df)
                        # ablations not for prediction_strategy
                        else:
                            grouped_df, _ = extract_data_for_table_1_v2(
                                df_ablation[
                                    make_df_key(bench, split, agg_type)
                                ],
                                num_anchors=NUM_ANCHORS,
                                lower_better=(agg_type == "mae"),
                                target_df=(
                                    target_df_ablation[
                                        make_df_key(bench, split, agg_type)
                                    ]
                                    if target_df_ablation is not None
                                    else None
                                ),
                            )
                            grouped_df = grouped_df[["fit"]].rename(
                                columns={"fit": f"fit-{factor}"}
                            )
                            factor_list.append(grouped_df)

                if len(factor_list) > 0:
                    factor_df = pd.concat(factor_list, axis=1)

                    if bench == "num_models":
                        num_models_df = factor_df

                    if bench == "prediction_strategy":
                        prediction_strategy_dfs[agg_type] = factor_df

                    if bench == "umap_pca":
                        umap_pca_dfs[agg_type] = factor_df

            # when bench means benchmark name (mmlu, hellaswag, etc.)
            # also includes ablations for num_models and stratification
            else:
                for split, per_split in per_bench.items():
                    if split not in table_avg_dict[bench][agg_type]:
                        table_avg_dict[bench][agg_type][split] = {}
                        table_std_dict[bench][agg_type][split] = {}
                        model_perf_dict[bench][agg_type][split] = {}

                    table_avg_base = None
                    table_std_base = None
                    model_perf_base = None
                    for method in ["ours", "irt"]:
                        filename_suffix = per_split[method]
                        # real_bench is a keyword used for making paths, when debug real_bench can be copied from another benchmark
                        bench_key = bench

                        results_path = f"{RESULTS_FOLDER}/accs_{bench_key}_split-{split}_iterations-5{filename_suffix}.pickle"
                        data = load_pickle(results_path)

                        (
                            table_avg_base,
                            table_std_base,
                            model_perf_base,
                        ) = process_table_data(
                            bench_key,
                            split,
                            filename_suffix,
                            data,
                            scenarios_to_skip,
                            ordered,
                            agg_type,
                            table_avg_base,
                            table_std_base,
                            model_perf_base,
                            std_across_models=False,
                        )

                    table_avg_dict[bench][agg_type][split] = table_avg_base
                    table_std_dict[bench][agg_type][split] = table_std_base
                    model_perf_dict[bench][agg_type][split] = model_perf_base

                    if split == "noniid":
                        if df_num_anchors is None:
                            df_num_anchors = {}
                        cur_df_num_anchors = make_df_with_results(
                            table_avg_base,
                            table_std_base,
                            bench_key,
                            split,
                            extract_std=return_only_df,
                        )
                        df_num_anchors[
                            make_df_key(bench, split, agg_type)
                        ] = cur_df_num_anchors
                        if not return_only_df:
                            table_1_data.append(
                                extract_data_for_table_1_v2(
                                    cur_df_num_anchors,
                                    num_anchors=NUM_ANCHORS,
                                    lower_better=(agg_type == "mae"),
                                    target_df=(
                                        target_df_num_anchors[
                                            make_df_key(bench, split, agg_type)
                                        ]
                                        if target_df_num_anchors is not None
                                        else None
                                    ),
                                )
                            )
                            for num_anchors in cur_df_num_anchors.keys():
                                if num_anchors not in figure_n_anchors_data:
                                    figure_n_anchors_data[num_anchors] = []
                                figure_n_anchors_data[num_anchors].append(
                                    extract_data_for_table_1_v2(
                                        cur_df_num_anchors,
                                        num_anchors=num_anchors,
                                        lower_better=(agg_type == "mae"),
                                        target_df=(
                                            target_df_num_anchors[
                                                make_df_key(
                                                    bench, split, agg_type
                                                )
                                            ]
                                            if target_df_num_anchors is not None
                                            else None
                                        ),
                                    )
                                )
                            if agg_type == "rank" and bench == "mmlu_fields":
                                ablation_strat, _ = extract_data_for_table_1_v2(
                                    cur_df_num_anchors,
                                    num_anchors=NUM_ANCHORS,
                                    lower_better=(agg_type == "mae"),
                                    key="stratified",
                                    target_df=(
                                        target_df_num_anchors[
                                            make_df_key(bench, split, agg_type)
                                        ]
                                        if target_df_num_anchors is not None
                                        else None
                                    ),
                                )
                    elif split == "iid":
                        if df_iid is None:
                            df_iid = {}
                        cur_df_iid = make_df_with_results(
                            table_avg_base,
                            table_std_base,
                            bench_key,
                            split,
                            extract_std=return_only_df,
                        )
                        df_iid[make_df_key(bench, split, agg_type)] = cur_df_iid
                        if not return_only_df:
                            table_1_data_iid.append(
                                extract_data_for_table_1_v2(
                                    cur_df_iid,
                                    num_anchors=NUM_ANCHORS,
                                    lower_better=(agg_type == "mae"),
                                    target_df=(
                                        target_df_iid[
                                            make_df_key(bench, split, agg_type)
                                        ]
                                        if target_df_iid is not None
                                        else None
                                    ),
                                )
                            )
    if return_only_df:
        return {
            "ablation": df_ablation,
            "num_anchors": df_num_anchors,
            "iid": df_iid,
        }
    else:
        return (
            scenarios_to_skip,
            table_1_data,
            table_1_data_iid,
            figure_n_anchors_data,
            umap_pca_dfs,
            prediction_strategy_dfs,
            ablation_strat,
            num_models_df,
            model_perf_dict,
        )


def main():
    # load needed results from JSON config
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source_config_path",
        type=str,
        required=True,
        help="Path to config file with result suffixes",
    )
    parser.add_argument(
        "--target_config_path",
        type=str,
        # required=True,
        default=None,
        help="Path to config file with result suffixes",
    )
    args = parser.parse_args()

    with open(args.source_config_path, "r") as f:
        source_results_suffixes = json.load(
            f
        )  # source = we select hyperparams based on them

    if args.target_config_path is not None:
        with open(args.target_config_path, "r") as f:
            target_results_suffixes = json.load(
                f
            )  # target = we extract data from them by using found hyperparams

    # target_df_dict contains dataframes with results from target experiments
    # It has 3 keys:
    # "ablation": Contains dataframes for ablation studies (e.g. num_models, umap_pca, prediction_strategy)
    # "num_anchors": Contains dataframes for experiments with different numbers of anchor points
    # "iid": Contains dataframes for IID split experiments
    # Each value is a dataframe containing metrics like MAE and rank for different methods
    # Each key in target_df_dict ("ablation", "num_anchors", "iid") contains a dict
    # with keys of form "{bench}_{split}_{agg_type}"
    # where:
    # - bench is the benchmark name (e.g. "mmlu_fields", "hellaswag")
    # - split is the data split type (e.g. "noniid", "iid")
    # - agg_type is the aggregation type ("mae" or "rank")
    # For example: "mmlu_fields_noniid_mae" or "hellaswag_iid_rank"
    if args.target_config_path is not None:
        target_df_dict = extract_results_needed_for_tables(
            target_results_suffixes, return_only_df=True
        )
    else:
        target_df_dict = None

    (
        scenarios_to_skip,
        table_1_data,
        table_1_data_iid,
        figure_n_anchors_data,
        umap_pca_dfs,
        prediction_strategy_dfs,
        ablation_strat,
        num_models_df,
        model_perf_dict,
    ) = extract_results_needed_for_tables(
        source_results_suffixes,
        return_only_df=False,
        target_df_dict=target_df_dict,
    )

    table_1, latex_str = make_table_1(table_1_data)
    display(table_1)
    print(latex_str)


if __name__ == "__main__":
    main()
