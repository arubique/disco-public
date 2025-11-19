import pandas as pd
import numpy as np
import re
from io import StringIO

raw_text = """

Seed = 0

         Approach                Type     # Samples        Type   MAE   Rank  \
0        Approach        Condensation  Condensation  Prediction  Mmlu   Mmlu
1                                type   num_anchors        type   mae   rank
2        Baseline              Random           100        Eval  3.32  0.903
3  tinyBenchmarks              Random           100      gp-IRT  2.57  0.912
4  tinyBenchmarks          anchor-IRT           100      gp-IRT  2.69   0.93
5  tinyBenchmarks  anchor-correctness           100      gp-IRT   1.9  0.918
6        Baseline              Random           100         kNN  1.91    0.9
7        Baseline              Random           100         fit  1.67  0.942
8    DISCO (ours)            High PDS           100         kNN  1.23  0.963
9    DISCO (ours)            High PDS           100         fit  1.07  0.978

         MAE       Rank         MAE        Rank   MAE   Rank
0  Hellaswag  Hellaswag  Winogrande  Winogrande   Arc    Arc
1        mae       rank         mae        rank   mae   rank
2       2.83      0.821        3.62       0.802  2.62  0.882
3        1.7      0.858        2.32       0.885   2.1  0.912
4        2.1      0.828        3.88       0.721  4.71  0.788
5       1.34      0.947        1.77        0.89  1.96  0.949
6       1.57      0.885        1.68       0.911   2.4  0.898
7       1.33      0.937        1.27       0.932  1.69  0.931
8       1.33      0.934        1.19       0.942  2.06   0.93
9       1.11      0.971        1.02       0.956  1.35  0.961



seed = 1

         Approach                Type     # Samples        Type   MAE   Rank  \
0        Approach        Condensation  Condensation  Prediction  Mmlu   Mmlu
1                                type   num_anchors        type   mae   rank
2        Baseline              Random           100        Eval  3.37  0.919
3  tinyBenchmarks              Random           100      gp-IRT  2.61  0.925
4  tinyBenchmarks          anchor-IRT           100      gp-IRT  3.08  0.915
5  tinyBenchmarks  anchor-correctness           100      gp-IRT  2.13   0.92
6        Baseline              Random           100         kNN  1.87  0.905
7        Baseline              Random           100         fit  1.69  0.943
8    DISCO (ours)            High PDS           100         kNN  1.34  0.958
9    DISCO (ours)            High PDS           100         fit  1.28  0.976

         MAE       Rank         MAE        Rank   MAE   Rank
0  Hellaswag  Hellaswag  Winogrande  Winogrande   Arc    Arc
1        mae       rank         mae        rank   mae   rank
2       2.85      0.848        3.59       0.847  2.61  0.903
3       1.69      0.863        1.68        0.94  2.13   0.93
4       2.14      0.868        2.15        0.85  4.14  0.832
5        1.4      0.959        2.14       0.925  2.55  0.948
6       1.55      0.897        1.59       0.926  2.32  0.914
7       1.33      0.941         1.3       0.938  1.72  0.943
8       1.06      0.947        1.05       0.976  1.81  0.957
9        0.9       0.98        0.91       0.968  1.44  0.974


seed = 2

         Approach                Type     # Samples        Type   MAE   Rank  \
0        Approach        Condensation  Condensation  Prediction  Mmlu   Mmlu
1                                type   num_anchors        type   mae   rank
2        Baseline              Random           100        Eval  3.47  0.916
3  tinyBenchmarks              Random           100      gp-IRT  2.63  0.921
4  tinyBenchmarks          anchor-IRT           100      gp-IRT  2.92  0.921
5  tinyBenchmarks  anchor-correctness           100      gp-IRT  1.88  0.928
6        Baseline              Random           100         kNN  1.72  0.927
7        Baseline              Random           100         fit  1.56  0.947
8    DISCO (ours)            High PDS           100         kNN  1.06   0.98
9    DISCO (ours)            High PDS           100         fit  1.27  0.986

         MAE       Rank         MAE        Rank   MAE   Rank
0  Hellaswag  Hellaswag  Winogrande  Winogrande   Arc    Arc
1        mae       rank         mae        rank   mae   rank
2       2.87       0.84        3.63       0.832  2.55  0.895
3       1.69       0.87        1.73       0.935  2.24  0.922
4       2.17      0.834        2.39       0.864  4.07   0.83
5       1.35       0.96        2.31       0.904  2.33  0.954
6       1.48      0.902        1.58       0.921  2.25  0.904
7       1.24      0.958        1.27       0.941  1.76  0.925
8       1.19      0.965        1.17       0.963   1.9  0.928
9       1.08      0.974        0.91        0.97  1.47  0.966

seed = 3

         Approach                Type     # Samples        Type   MAE   Rank  \
0        Approach        Condensation  Condensation  Prediction  Mmlu   Mmlu
1                                type   num_anchors        type   mae   rank
2        Baseline              Random           100        Eval  3.41  0.921
3  tinyBenchmarks              Random           100      gp-IRT  2.72  0.921
4  tinyBenchmarks          anchor-IRT           100      gp-IRT  2.84  0.941
5  tinyBenchmarks  anchor-correctness           100      gp-IRT  1.98  0.919
6        Baseline              Random           100         kNN   1.9  0.905
7        Baseline              Random           100         fit  1.92   0.93
8    DISCO (ours)            High PDS           100         kNN  1.36  0.978
9    DISCO (ours)            High PDS           100         fit  1.23   0.98

         MAE       Rank         MAE        Rank   MAE   Rank
0  Hellaswag  Hellaswag  Winogrande  Winogrande   Arc    Arc
1        mae       rank         mae        rank   mae   rank
2       2.86      0.836        3.67       0.838  2.57  0.909
3       1.65      0.864        1.73       0.932  2.21  0.918
4        2.2      0.825        2.51       0.861  4.18  0.841
5       1.32      0.957        2.34       0.901  2.34  0.956
6       1.59      0.903        1.63       0.923  2.44  0.896
7       1.29       0.94        1.37       0.921  1.86  0.936
8       1.62      0.959        1.15       0.947  2.07  0.933
9        0.7      0.979        0.94       0.964   1.3  0.975


seed = 4

         Approach                Type     # Samples        Type   MAE   Rank  \
0        Approach        Condensation  Condensation  Prediction  Mmlu   Mmlu
1                                type   num_anchors        type   mae   rank
2        Baseline              Random           100        Eval  3.37    0.9
3  tinyBenchmarks              Random           100      gp-IRT  2.57  0.905
4  tinyBenchmarks          anchor-IRT           100      gp-IRT  2.92  0.896
5  tinyBenchmarks  anchor-correctness           100      gp-IRT  1.98  0.917
6        Baseline              Random           100         kNN  1.74  0.893
7        Baseline              Random           100         fit  1.84  0.918
8    DISCO (ours)            High PDS           100         kNN  1.25  0.964
9    DISCO (ours)            High PDS           100         fit  1.11  0.982

         MAE       Rank         MAE        Rank   MAE   Rank
0  Hellaswag  Hellaswag  Winogrande  Winogrande   Arc    Arc
1        mae       rank         mae        rank   mae   rank
2       2.92      0.817        3.61       0.812  2.56  0.894
3       1.58      0.853        1.64        0.92  2.12  0.905
4       2.18      0.791        2.34       0.849  4.24  0.778
5       1.13       0.96        2.14       0.887  1.72  0.964
6       1.49       0.89        1.44       0.911  2.26  0.888
7       1.12      0.936        1.19       0.921   1.6  0.937
8       1.17       0.88        1.05       0.943  1.97  0.924
9       0.79      0.976        0.92       0.967  1.33  0.976 """

# Split by seed blocks
blocks = re.split(r"[Ss]eed\s*=\s*\d+", raw_text)
seeds = re.findall(r"[Ss]eed\s*=\s*(\d+)", raw_text)

data_rows = []

# process each table block
for seed, block in zip(seeds, blocks[1:]):
    lines = block.strip().splitlines()

    # Find the two parts of the table (split by empty line or continuation)
    # First part: lines with "Approach" header and data rows
    # Second part: continuation with remaining metrics
    first_part_lines = []
    second_part_lines = []
    in_second_part = False

    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Check if this is the start of the second part (has "MAE" and "Rank" but no "Approach")
        if (
            re.search(r"\bMAE\b.*\bRank\b", line)
            and "Approach" not in line
            and not in_second_part
        ):
            # This might be the header of second part, skip it
            in_second_part = True
            continue
        if in_second_part:
            second_part_lines.append(line)
        else:
            first_part_lines.append(line)

    # Parse first part: Approach, Type, # Samples, Type, MAE, Rank (for MMLU)
    first_data = {}  # Use dict keyed by row number
    for line in first_part_lines:
        # Extract data rows (should start with row number 2-9, which are actual data)
        # Skip rows 0-1 which are headers
        match = re.match(r"^(\d+)\s+", line)
        if match:
            row_num = int(match.group(1))
            # Only process data rows (2-9), skip header rows (0-1)
            if row_num >= 2:
                # Split by multiple spaces, but keep row index separate
                parts = re.split(r"\s{2,}", line.strip())
                if (
                    len(parts) >= 7
                ):  # Should have: index, Approach, Type, Samples, Type, MAE, Rank
                    first_data[row_num] = parts[
                        1:7
                    ]  # Skip row index, take 6 columns

    # Parse second part: MAE, Rank, MAE, Rank, MAE, Rank (for Hellaswag, Winogrande, Arc)
    second_data = {}  # Use dict keyed by row number
    for line in second_part_lines:
        # Extract data rows (should start with row number 2-9)
        match = re.match(r"^(\d+)\s+", line)
        if match:
            row_num = int(match.group(1))
            # Only process data rows (2-9), skip header rows (0-1)
            if row_num >= 2:
                parts = re.split(r"\s{2,}", line.strip())
                if (
                    len(parts) >= 7
                ):  # Should have: index, MAE, Rank, MAE, Rank, MAE, Rank
                    second_data[row_num] = parts[
                        1:7
                    ]  # Skip row index, take 6 columns

    # Combine first and second parts for each row, matching by row number
    combined_rows = []
    for row_num in sorted(set(first_data.keys()) & set(second_data.keys())):
        if len(first_data[row_num]) >= 6 and len(second_data[row_num]) >= 6:
            # Combine: first part (6 cols) + second part (6 cols) = 12 cols total
            row = first_data[row_num][:6] + second_data[row_num][:6]
            combined_rows.append(row)

    # Create DataFrame
    if combined_rows:
        df = pd.DataFrame(
            combined_rows,
            columns=[
                "Approach",
                "Type",
                "Samples",
                "PredType",
                "Mmlu_MAE",
                "Mmlu_Rank",
                "Hellaswag_MAE",
                "Hellaswag_Rank",
                "Winogrande_MAE",
                "Winogrande_Rank",
                "Arc_MAE",
                "Arc_Rank",
            ],
        )
        df["seed"] = int(seed)
        data_rows.append(df)

# merge all seeds
df_all = pd.concat(data_rows, ignore_index=True)

# convert numeric columns properly
for c in df_all.columns:
    if c not in ["Approach", "Type", "PredType"]:
        df_all[c] = pd.to_numeric(df_all[c], errors="coerce")

# compute mean & std
summary = df_all.groupby(["Approach", "Type", "PredType"]).agg(["mean", "std"])


# nice formatting: mean ± std
def combine_mean_std(mean, std):
    return f"{mean:.3f} ± {std:.3f}"


formatted = summary.copy()
for col in summary.columns.levels[0]:
    formatted[col, "mean±std"] = summary[col]["mean"].combine(
        summary[col]["std"], combine_mean_std
    )

# Final pretty result
final_table = formatted.xs("mean±std", axis=1, level=1)
# print(final_table)
final_table.to_csv("summary_std_from_tables.csv")
