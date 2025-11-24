import pandas as pd
import numpy as np
import re
from io import StringIO

raw_text = """

Seed = 0

         Approach                Type     # Samples        Type   MAE   Rank  \
0        Approach        Condensation  Condensation  Prediction  Mmlu   Mmlu
1                                type   num_anchors        type   mae   rank
2        Baseline              Random           100        Eval  3.55  0.906
3  tinyBenchmarks              Random           100      gp-IRT  3.88  0.845
4  tinyBenchmarks          anchor-IRT           100      gp-IRT  2.91  0.919
5  tinyBenchmarks  anchor-correctness           100      gp-IRT  2.03  0.885
6        Baseline              Random           100         kNN  2.19  0.883
7        Baseline              Random           100         fit  1.81  0.901
8    DISCO (ours)            High PDS           100         kNN  1.23  0.963
9    DISCO (ours)            High PDS           100         fit  1.12  0.978


0  Hellaswag  Hellaswag  Winogrande  Winogrande   Arc    Arc
1        mae       rank         mae        rank   mae   rank
2       4.03      0.753        3.27        0.56  2.41  0.902
3       2.02      0.783         2.0       0.838   2.3  0.939
4       2.18       0.86        3.29       0.663  4.71  0.794
5       1.45       0.95        1.87       0.916  1.77   0.94
6       1.99      0.883        1.69       0.912  2.23  0.924
7        1.8      0.914         1.3       0.878  1.67  0.944
8       1.33      0.934        1.19       0.942  2.06   0.93
9       1.07      0.972         1.0       0.968  1.32  0.958



seed = 1

         Approach                Type     # Samples        Type   MAE   Rank  \
0        Approach        Condensation  Condensation  Prediction  Mmlu   Mmlu
1                                type   num_anchors        type   mae   rank
2        Baseline              Random           100        Eval  2.59  0.948
3  tinyBenchmarks              Random           100      gp-IRT  2.52   0.96
4  tinyBenchmarks          anchor-IRT           100      gp-IRT  3.61  0.898
5  tinyBenchmarks  anchor-correctness           100      gp-IRT  2.36  0.884
6        Baseline              Random           100         kNN   1.8  0.932
7        Baseline              Random           100         fit  1.71   0.97
8    DISCO (ours)            High PDS           100         kNN  1.34  0.958
9    DISCO (ours)            High PDS           100         fit   1.3  0.971

         MAE       Rank         MAE        Rank   MAE   Rank
0  Hellaswag  Hellaswag  Winogrande  Winogrande   Arc    Arc
1        mae       rank         mae        rank   mae   rank
2       4.14      0.755        3.13       0.696  2.47  0.918
3       2.01      0.781        1.91       0.929  2.33  0.943
4       1.74      0.907        2.08       0.857  4.47  0.815
5       1.51      0.948        2.11       0.934   2.0  0.963
6       1.94      0.896        1.58        0.94  2.09  0.942
7       1.79      0.943        1.31       0.884   1.6  0.957
8       1.06      0.947        1.05       0.976  1.81  0.957
9       0.88      0.982        0.95       0.972  1.44  0.977


seed = 2

         Approach                Type     # Samples        Type   MAE   Rank  \
0        Approach        Condensation  Condensation  Prediction  Mmlu   Mmlu
1                                type   num_anchors        type   mae   rank
2        Baseline              Random           100        Eval  2.34  0.947
3  tinyBenchmarks              Random           100      gp-IRT  2.07  0.937
4  tinyBenchmarks          anchor-IRT           100      gp-IRT  2.92  0.935
5  tinyBenchmarks  anchor-correctness           100      gp-IRT  1.63  0.948
6        Baseline              Random           100         kNN  2.36  0.865
7        Baseline              Random           100         fit  1.73  0.936
8    DISCO (ours)            High PDS           100         kNN  1.06   0.98
9    DISCO (ours)            High PDS           100         fit  1.28  0.985

         MAE       Rank         MAE        Rank   MAE   Rank
0  Hellaswag  Hellaswag  Winogrande  Winogrande   Arc    Arc
1        mae       rank         mae        rank   mae   rank
2       4.19      0.779        3.19       0.606  2.28  0.923
3       1.94      0.815        1.81       0.911  2.39  0.946
4        2.0       0.89        1.94       0.867  4.13  0.844
5       1.47      0.952        2.21       0.919  2.43  0.944
6       1.87      0.898        1.61       0.917  2.05  0.932
7       1.69      0.951        1.35       0.889  1.57  0.948
8       1.19      0.965        1.17       0.963   1.9  0.928
9        1.0      0.971        0.89       0.968  1.52  0.969

seed = 3

0        Approach        Condensation  Condensation  Prediction  Mmlu   Mmlu
1                                type   num_anchors        type   mae   rank
2        Baseline              Random           100        Eval  2.37  0.949
3  tinyBenchmarks              Random           100      gp-IRT  2.11  0.946
4  tinyBenchmarks          anchor-IRT           100      gp-IRT  3.07  0.934
5  tinyBenchmarks  anchor-correctness           100      gp-IRT  1.83  0.943
6        Baseline              Random           100         kNN  1.71  0.962
7        Baseline              Random           100         fit  2.07  0.929
8    DISCO (ours)            High PDS           100         kNN  1.36  0.978
9    DISCO (ours)            High PDS           100         fit  1.23  0.981

         MAE       Rank         MAE        Rank   MAE   Rank
0  Hellaswag  Hellaswag  Winogrande  Winogrande   Arc    Arc
1        mae       rank         mae        rank   mae   rank
2       4.11      0.779        3.15         0.7  2.37  0.927
3       2.01      0.786        1.86       0.917   2.3  0.929
4       2.37      0.805        2.67       0.852  4.24  0.824
5       1.56      0.971         2.4       0.895   2.3  0.958
6       1.99      0.892        1.67       0.928  2.22  0.924
7       1.76      0.926        1.44       0.887  1.72  0.963
8       1.62      0.959        1.15       0.947  2.07  0.933
9       0.68      0.977        0.92       0.969   1.3  0.978

seed = 4

0        Approach        Condensation  Condensation  Prediction  Mmlu   Mmlu
1                                type   num_anchors        type   mae   rank
2        Baseline              Random           100        Eval  4.11  0.888
3  tinyBenchmarks              Random           100      gp-IRT  2.98  0.902
4  tinyBenchmarks          anchor-IRT           100      gp-IRT  2.93   0.84
5  tinyBenchmarks  anchor-correctness           100      gp-IRT  1.92  0.941
6        Baseline              Random           100         kNN   1.7  0.922
7        Baseline              Random           100         fit  2.11   0.89
8    DISCO (ours)            High PDS           100         kNN  1.25  0.964
9    DISCO (ours)            High PDS           100         fit   1.1  0.983

         MAE       Rank         MAE        Rank   MAE   Rank
0  Hellaswag  Hellaswag  Winogrande  Winogrande   Arc    Arc
1        mae       rank         mae        rank   mae   rank
2       4.28      0.746        2.99       0.612  2.35  0.916
3       1.94      0.763        1.88       0.897  2.29  0.916
4       1.93      0.841        2.37       0.875  3.58  0.823
5       0.96      0.967        1.83       0.935  1.69  0.977
6       1.77      0.877        1.36       0.918  1.99  0.918
7       1.63      0.914        1.26       0.888  1.34  0.957
8       1.17       0.88        1.05       0.943  1.97  0.924
9       0.74       0.97        0.92       0.957  1.32  0.977

"""

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
