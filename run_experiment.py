# The code is adapted from the tinyBenchmarks repo: https://github.com/felipemaiapolo/efficbench
import pickle
import copy
import pandas as pd
import argparse
from scipy import stats
import os
import numpy as np

# from experiments import *
from experiments import RANDOM_SEED, evaluate_scenarios
from utils import (
    lb_scenarios,
    # get_lambda,
    # SuppressPrints,
    # sigmoid,
    # item_curve,
    # item_response_function,
    # prepare_data,
    dump_pickle,
    load_pickle,
    alpaca_scenarios,
    icl_templates_scenarios,
    helm_lite_scenarios,
)
from plots import (
    MODEL_OUTPUTS_PATH,
    DATA_FOLDER,
    MAX_TABLE_SIZE,
    winrate,
    benchs,
    splits,
    methods,
    # number_items,
    agg_metric,
    load_scores,
    make_perf_table,
    make_table_avg,
)
from acc import (
    ESTIMATORS,
    FITTING_METHODS,
    BASE_ESTIMATORS,
    BEST_FITTING_METHODS,
    MLP_FITTING_METHODS,
)
from generating_data.utils_for_notebooks import merge_methods
from stnd.utility.utils import apply_random_seed
from stnd.utility.imports import make_from_class_ctor
import json


SCENARIOS_TO_SKIP = ["harness_gsm8k_5"]


def validate_sampling_names(sampling_names_str):
    """
    Validate that the provided comma-separated sampling names contain no duplicates.

    Args:
        sampling_names_str (str): Comma-separated sampling names string from CLI.

    Returns:
        list[str]: Parsed list of unique sampling names.

    Raises:
        ValueError: If duplicate names are detected.
    """
    # Support escaped commas if ever used (mirrors estimators handling)
    tokens = sampling_names_str.replace("__COMMA__", ",").split(",")
    names = [t.strip() for t in tokens if t.strip() != ""]

    seen = set()
    duplicates = set()
    for name in names:
        if name in seen:
            duplicates.add(name)
        else:
            seen.add(name)

    if duplicates:
        # Sort for deterministic message
        dups_sorted = ", ".join(sorted(duplicates))
        raise ValueError(
            f"Duplicate sampling names provided: {dups_sorted}. Remove duplicates and retry."
        )

    return names


def load_estimators_and_fitting_methods(config_path):
    """
    Load estimators and fitting methods from a JSON configuration file.

    Args:
        config_path (str): Path to the JSON configuration file

    Returns:
        tuple: (chosen_estimators, chosen_fitting_methods)
    """
    with open(config_path, "r") as file:
        config = json.load(file)

    chosen_estimators = []
    chosen_fitting_methods = []

    for name, details in config.items():
        if "class_path" in details:
            # This is a fitting method (class + params)
            class_path = details["class_path"]
            params = details.get("params", {})

            # Create a class factory that uses make_from_class_ctor
            class EstimatorFactory:
                def __init__(self, class_path):
                    self.class_path = class_path

                def __call__(self, **kwargs):
                    # Create the class configuration for make_from_class_ctor
                    class_config = {
                        "class": self.class_path,
                        "kwargs": kwargs,  # Include parameters in the config
                    }
                    return make_from_class_ctor(class_config)

            # Create the factory class
            estimator_class = EstimatorFactory(class_path)

            # Add to fitting methods as (name, (class, params)) tuple
            chosen_fitting_methods.append((name, (estimator_class, params)))
            chosen_estimators.append(name)
        else:
            # This is a base estimator (just a name)
            chosen_estimators.append(name)

    return chosen_estimators, chosen_fitting_methods


# [STRUCTURE][separate file]
def get_data(
    bench,
    split,
    text_to_vector=None,
    return_data_path=False,
    subsample_validation=False,
):
    if text_to_vector is not None:
        assert bench in ["helm_lite", "alpaca"]

    data_path = None
    # Loading data
    # [FUNCTION][Load lb, mmlu]
    if bench in ["lb", "mmlu"]:
        # data
        with open("data/lb.pickle", "rb") as handle:
            data = pickle.load(handle)

        # scenarios
        scenarios = (
            {"mmlu": lb_scenarios["mmlu"]} if bench == "mmlu" else lb_scenarios
        )

        # split
        if split == "iid":
            set_of_rows = [list(range(0, len(data["models"]), 4))]
        elif split == "noniid":
            set_of_rows = [
                list(range(int(len(data["models"]) / 4))),
            ]
        elif split == "noniid2":
            set_of_rows = [list(range(200))]
        elif split == "noniid3":
            set_of_rows = [list(range(300))]

        print(len(set_of_rows[0]), len(data["models"]))

    # [FUNCTION][Load helm_lite]
    elif bench == "helm_lite":
        # data
        if text_to_vector is None:
            # with open('data/helm_lite.pickle', 'rb') as handle:
            #     data = pickle.load(handle)
            data_path = "data/helm_lite.pickle"
        elif text_to_vector == "bow":  # bag-of-words
            # with open('./generating_data/download_helm/helm_lite_with_preds_2907_bow.pickle', 'rb') as handle:
            #     data = pickle.load(handle)
            data_path = "./generating_data/download_helm/helm_lite_with_preds_2907_bow.pickle"
        elif text_to_vector == "bge":  # bge
            # with open('./data/helm_lite_with_preds_bge.pickle', 'rb') as handle:
            #     data = pickle.load(handle)
            data_path = "./data/helm_lite_with_preds_bge.pickle"
        else:
            raise NotImplementedError

        with open(data_path, "rb") as handle:
            data = pickle.load(handle)

        # scenarios
        scenarios = helm_lite_scenarios

        # split
        if split == "iid":
            set_of_rows = [
                [0, 11, 22],
                [1, 12, 23],
                [2, 13, 24],
                [3, 14, 25],
                [4, 15, 26],
                [5, 16, 27],
                [6, 17, 28],
                [7, 18, 29],
                [8, 19],
                [9, 20],
                [10, 21],
            ]
        else:
            set_of_rows = [
                [0, 1],  # AI: Yi
                [2, 3, 4],  # AlephAlpha_luminous
                [5, 6],  # ai21_j2
                [7, 8, 9, 10],  # anthropic_claude
                [11, 12],  # cohere
                [13, 14],  # google
                [15, 16, 17, 18],  # llama
                [19, 20],  # mistral ai
                [21, 22, 23, 24, 25],  # openai
                [26, 27],  # TII/UAE
                [28, 29],
            ]  # writer

        print(len(set_of_rows[0]), len(data["models"]))

    # [FUNCTION][Load alpaca]
    elif bench == "alpaca":
        # #data
        # with open('data/alpaca_v2.pickle', 'rb') as handle:
        #     data = pickle.load(handle)

        # data
        if text_to_vector is None:
            # with open('data/alpaca_v2.pickle', 'rb') as handle:
            #     data = pickle.load(handle)
            data_path = "data/alpaca_v2.pickle"
        elif text_to_vector == "bow":  # bag-of-words
            # with open('./data/alpaca_v2_with_preds_bow_29072025.pickle', 'rb') as handle:
            #     data = pickle.load(handle)
            data_path = "./data/alpaca_v2_with_preds_bow_29072025.pickle"
        elif text_to_vector == "bge":  # bge
            # with open('./data/alpaca_v2_with_preds_bge.pickle', 'rb') as handle:
            #     data = pickle.load(handle)
            data_path = "./data/alpaca_v2_with_preds_bge.pickle"
        else:
            raise NotImplementedError

        with open(data_path, "rb") as handle:
            data = pickle.load(handle)

        # scenarios
        scenarios = alpaca_scenarios

        # split
        if split == "iid":
            set_of_rows = [
                list(range(0, len(data["models"]), 4)),
                list(range(1, len(data["models"]) + 1, 4)),
                list(range(2, len(data["models"]) + 2, 4)),
                list(range(3, len(data["models"]) + 3, 4)),
            ]
        elif split == "noniid":
            set_of_rows = [
                list(range(int(len(data["models"]) / 4))),
            ]
        elif split == "noniid2":
            set_of_rows = [list(range(50))]
        elif split == "noniid3":
            set_of_rows = [list(range(75))]

        print(len(set_of_rows[0]), len(data["models"]))

    # Loading data
    # [FUNCTION][Load mmlu]
    elif bench in [
        "mmlu_fields",
        "truthfulqa",
        "winogrande",
        "arc",
        "hellaswag",
    ]:
        model_outputs_path = MODEL_OUTPUTS_PATH

        # data
        with open(model_outputs_path, "rb") as handle:
            data = pickle.load(handle)

        scenario_key = "mmlu" if bench == "mmlu_fields" else bench

        # scenarios
        scenarios = lb_scenarios
        # scenarios = {'mmlu':scenarios['mmlu']}
        scenarios = {scenario_key: scenarios[scenario_key]}

        # split
        if split == "iid":
            k = int(len(data["models"]) / 40)
            set_of_rows = [list(range(0, len(data["models"]), k))]
        else:
            set_of_rows = [list(range(40))]
        print(len(set_of_rows[0]), len(data["models"]))

    # [FUNCTION][Load icl_templates]
    elif bench == "icl_templates":
        # data
        with open("data/icl_templates.pickle", "rb") as handle:
            data = pickle.load(handle)

        # scenarios
        scenarios = icl_templates_scenarios

        # split
        if split == "iid":
            import random

            random.seed(42)  # 0
            list1 = random.sample(
                range(len(data["models"])), int(len(data["models"]) / 2)
            )
            list2 = [i for i in range(len(data["models"])) if i not in list1]
            set_of_rows = [list1, list2]

        elif split == "noniid":  # instruction
            templates = [
                [
                    "GPT_3_style",
                    "MNLI_crowdsource",
                    "always_sometimes_never",
                    "based_on_the_previous_passage",
                    "can_we_infer",
                    "claim_true_false_inconclusive",
                    "consider_always_sometimes_never",
                    "does_it_follow_that",
                ],
                [
                    "does_this_imply",
                    "guaranteed_possible_impossible",
                    "guaranteed_true",
                    "justified_in_saying",
                    "must_be_true",
                    "should_assume",
                    "take_the_following_as_truth",
                ],
            ]
            set_of_rows = [
                [
                    i
                    for i, m in enumerate(data["models"])
                    if np.sum([t in m for t in temp]) > 0
                ]
                for temp in templates
            ]

        elif split == "noniid2":  # size
            sizes = [["65b"]]
            set_of_rows = [
                [
                    i
                    for i, m in enumerate(data["models"])
                    if np.sum([t in m for t in size]) > 0
                ]
                for size in sizes
            ]

        elif split == "noniid3":  # same vs cross instr
            cross = [["same_instr"], ["cross_instr"]]
            set_of_rows = [
                [
                    i
                    for i, m in enumerate(data["models"])
                    if np.sum([t in m for t in cr]) > 0
                ]
                for cr in cross
            ]

        else:
            raise NotImplementedError

        print(len(set_of_rows[0]), len(data["models"]))

    else:
        raise NotImplementedError

    if subsample_validation:
        test_models = set_of_rows[0]
        total_models = len(data["models"])
        train_models = [
            model_idx
            for model_idx in list(range(total_models))
            if model_idx not in test_models
        ]
        if split == "iid":
            val_models = [
                np.random.choice(
                    train_models, size=len(test_models), replace=False
                ).tolist()
            ]
        else:
            assert split == "noniid"
            val_models = train_models[: len(test_models)]
        data["models"] = [
            data["models"][i] for i in train_models
        ]  # remove test models
        set_of_rows = [val_models]

    res = [data, scenarios, set_of_rows]
    if return_data_path:
        res += [data_path]
    return res


def main():
    # [FUNCTION][parsing args]
    # User input
    parser = argparse.ArgumentParser(
        description="Example script with named arguments."
    )

    parser.add_argument(
        "--bench",
        type=str,
        help="Benchmark (helm_lite, lb, mmlu, alpaca, icl_templates)",
        default="lb",
    )
    parser.add_argument(
        "--split", type=str, help="iid/noniid/noniid2/noniid3", default="iid"
    )
    parser.add_argument("--iterations", type=int, help="iterations", default=3)
    parser.add_argument("--device", type=str, help="cpu/cuda", default="cpu")
    parser.add_argument(
        "--num_workers", type=int, help="number of workers", default=12
    )
    parser.add_argument("--skip_irt", action="store_true", help="skip irt")
    parser.add_argument(
        "--cache_path", type=str, help="cache path", default=None
    )
    parser.add_argument(
        "--sampling_names",
        type=str,
        help="sampling names",
        default="random,anchor,anchor-irt",
    )
    parser.add_argument(
        "--filename_suffix", type=str, help="path suffix", default=""
    )
    parser.add_argument(
        "--make_results_table", action="store_true", help="make results table"
    )
    parser.add_argument(
        "--results_table_path",
        type=str,
        help="results table path",
        default=None,
    )
    parser.add_argument(
        "--estimators",
        type=str,
        help="estimators",
        default="naive,pirt,cirt,gpirt",
    )
    parser.add_argument("--pca", type=int, help="pca", default=None)
    parser.add_argument(
        "--n_source_models",
        type=int,
        help="number of source models",
        default=None,
    )
    parser.add_argument(
        "--merge_with_original",
        action="store_true",
        help="merge with original results table",
    )
    parser.add_argument(
        "--number_items",
        type=str,
        help="number of items",
        default="10,30,60,100",
    )
    parser.add_argument(
        "--text_to_vector", type=str, help="text_to_vector", default=None
    )
    parser.add_argument(
        "--disagreement_type", type=str, help="disagreement type", default="pds"
    )
    parser.add_argument(
        "--subsample_validation",
        action="store_true",
        help="subsample validation",
    )

    apply_random_seed(RANDOM_SEED)

    args = parser.parse_args()

    # [FUNCTION][choosing estimators]
    if args.estimators is None or args.estimators == "all":
        chosen_estimators = ESTIMATORS
        chosen_fitting_methods = FITTING_METHODS
    elif args.estimators == "best":
        chosen_fitting_methods = BEST_FITTING_METHODS
        chosen_estimators = BASE_ESTIMATORS + [
            f[0] for f in BEST_FITTING_METHODS
        ]
    elif args.estimators == "mlp":
        chosen_fitting_methods = MLP_FITTING_METHODS
        chosen_estimators = BASE_ESTIMATORS + [
            f[0] for f in MLP_FITTING_METHODS
        ]
    else:
        assert isinstance(args.estimators, str), "estimators must be a string"
        estimators = args.estimators.replace("__COMMA__", ",").split(",")
        if ".json" in args.estimators:
            (
                chosen_estimators,
                chosen_fitting_methods,
            ) = load_estimators_and_fitting_methods(args.estimators)
        else:
            chosen_estimators = [
                e for e in ESTIMATORS if e in estimators or e in BASE_ESTIMATORS
            ]
            chosen_fitting_methods = [
                f for f in FITTING_METHODS if f[0] in estimators
            ]

    if args.results_table_path is None and args.make_results_table:
        if args.cache_path is not None:
            args.results_table_path = args.cache_path.replace(".pickle", ".csv")
        else:
            args.results_table_path = f"default_df_run_experiment.csv"

    bench = args.bench
    split = args.split
    iterations = args.iterations
    device = args.device

    assert bench in [
        "helm_lite",
        "lb",
        "mmlu",
        "alpaca",
        "mmlu_fields",
        "icl_templates",
        "truthfulqa",
        "winogrande",
        "arc",
        "hellaswag",
    ]
    assert split in ["iid", "noniid", "noniid2", "noniid3"]
    assert iterations > 0

    # Defining other parameters
    Ds = [2, 5, 10, 15]
    # Validate and parse sampling names (raise on duplicates)
    sampling_names = validate_sampling_names(args.sampling_names)

    scenario_name = "full"  # we are evaluating all scenarios at once (this is just a nomination)

    data, scenarios, set_of_rows, data_path = get_data(
        bench,
        split,
        text_to_vector=args.text_to_vector,
        return_data_path=True,
        subsample_validation=args.subsample_validation,
    )

    chosen_scenarios = list(scenarios.keys())

    if args.cache_path is not None:
        if os.path.exists(args.cache_path):
            cache = load_pickle(args.cache_path)
        else:
            dirname = os.path.dirname(args.cache_path)
            if dirname != "":
                os.makedirs(dirname, exist_ok=True)
            cache = {"cache_path": args.cache_path}
    else:
        cache = None

    # ## Results
    results_full, accs_full, sampling_time_dic = evaluate_scenarios(
        data,
        scenario_name,
        chosen_scenarios,
        scenarios,
        set_of_rows,
        Ds,
        iterations,
        device,
        bench="irt_" + bench,
        split=split,
        sampling_names=sampling_names,
        num_workers=args.num_workers,
        skip_irt=args.skip_irt,
        cache=cache,
        chosen_estimators=chosen_estimators,
        chosen_fitting_methods=chosen_fitting_methods,
        pca=args.pca,
        n_source_models=args.n_source_models,
        number_items=[int(item) for item in args.number_items.split(",")],
        apply_softmax_to_predictions=(args.text_to_vector is None),
        disagreement_type=args.disagreement_type,
    )

    if args.cache_path is not None:
        dump_pickle(cache, args.cache_path)

    filename_suffix = args.filename_suffix

    # [CLEAN][Use constants instead of hardcoed values]
    results_full_path = f"results/results_{bench}_split-{split}_iterations-{iterations}{filename_suffix}.pickle"
    accs_full_path = f"results/accs_{bench}_split-{split}_iterations-{iterations}{filename_suffix}.pickle"
    samplingtime_full_path = f"results/samplingtime_{bench}_split-{split}_iterations-{iterations}{filename_suffix}.pickle"

    with open(results_full_path, "wb") as handle:
        pickle.dump(results_full, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(accs_full_path, "wb") as handle:
        pickle.dump(accs_full, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(samplingtime_full_path, "wb") as handle:
        pickle.dump(sampling_time_dic, handle, protocol=pickle.HIGHEST_PROTOCOL)

    if args.make_results_table:
        table_avg, table_std = make_table_avg(
            bench,
            split,
            filename_suffix,
            accs_full,
            scenarios_to_skip=SCENARIOS_TO_SKIP,
            num_it=iterations,
            data_path=data_path,
        )
        make_results_table(
            table_avg,
            table_std,
            bench,
            args.results_table_path,
            split,
            merge_with_original=args.merge_with_original,
        )


# TODO(Alex | 31072025): move to plots.py
def make_results_table(
    table_avg,
    table_std,
    bench,
    results_table_path,
    split,
    merge_with_original=True,
):
    # TODO(Alex | 31072025): make constants
    agg = "leaderboard"  # 'leaderboard', 'scenarios'
    results = "acc"  # 'acc', 'rank'

    style = {
        "alpha": 1,
        "markersize": 3,
        "markeredgewidth": 1,
        "elinewidth": 1,
        "capsize": 3,
        "linestyle": "",
    }

    # TODO(Alex | 31072025): remove hardcoded paths
    if merge_with_original:
        # Load table_avg from pickle file
        with open("results/table_avg_original.pickle", "rb") as handle:
            table_avg_original = pickle.load(handle)
        with open("results/table_std_original.pickle", "rb") as handle:
            table_std_original = pickle.load(handle)

        table_avg = merge_methods(table_avg, table_avg_original)
        table_std = merge_methods(table_std, table_std_original)

    if results == "acc":
        ylim = (0, 0.1)
    elif results == "rank":
        if agg_metric == "std":
            ylim = (0, 0.1)
        else:
            ylim = (0.5, 1)
    else:
        raise NotImplementedError

    cur_methods_for_table = table_avg[bench][split].keys()

    df = make_perf_table(
        table_avg[bench][split],
        table_std[bench][split],
        methods=cur_methods_for_table,
    )

    pd.set_option("display.max_rows", MAX_TABLE_SIZE)
    pd.set_option("display.max_columns", MAX_TABLE_SIZE)
    pd.set_option("display.max_colwidth", MAX_TABLE_SIZE)
    for num_samples in df.keys():
        print("#anchor_points:", num_samples)
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

        print(df[num_samples])

    df[max(list(df.keys()))].to_csv(results_table_path)


if __name__ == "__main__":
    main()
