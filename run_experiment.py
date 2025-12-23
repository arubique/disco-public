# The code is adapted from the tinyBenchmarks repo: https://github.com/felipemaiapolo/efficbench
import pickle
import copy
import pandas as pd
import argparse
from scipy import stats
import os
import numpy as np


from experiments import RANDOM_SEED, evaluate_scenarios
from utils import (
    lb_scenarios,
    dump_pickle,
    load_pickle,
)
from plots import (
    MODEL_OUTPUTS_PATH,
    make_table_avg,
    make_results_table,
)
from acc import (
    ESTIMATORS,
    FITTING_METHODS,
    BASE_ESTIMATORS,
    BEST_FITTING_METHODS,
    MLP_FITTING_METHODS,
)
from utils_for_notebooks import merge_methods
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


def load_and_split_model_outputs(
    bench,
    split,
    model_outputs_path,
    text_to_vector=None,
    return_data_path=False,
    subsample_validation=False,
):
    if text_to_vector is not None:
        assert bench in ["helm_lite", "alpaca"]

    # Loading data
    if bench in [
        "mmlu_fields",
        "truthfulqa",
        "winogrande",
        "arc",
        "hellaswag",
    ]:
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
        res += [model_outputs_path]
    return res


def parse_arguments():
    """
    Parse command line arguments for the experiment runner.

    Returns:
        argparse.Namespace: Parsed command line arguments
    """
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
    parser.add_argument(
        "--model_outputs_path",
        type=str,
        help="model outputs path",
        default=MODEL_OUTPUTS_PATH,
    )

    return parser.parse_args()


def choose_estimators(estimators_arg):
    """
    Choose estimators and fitting methods based on the command line argument.

    Args:
        estimators_arg (str): The estimators argument from command line

    Returns:
        tuple: (chosen_estimators, chosen_fitting_methods)
    """
    if estimators_arg is None or estimators_arg == "all":
        chosen_estimators = ESTIMATORS
        chosen_fitting_methods = FITTING_METHODS
    elif estimators_arg == "best":
        chosen_fitting_methods = BEST_FITTING_METHODS
        chosen_estimators = BASE_ESTIMATORS + [
            f[0] for f in BEST_FITTING_METHODS
        ]
    elif estimators_arg == "mlp":
        chosen_fitting_methods = MLP_FITTING_METHODS
        chosen_estimators = BASE_ESTIMATORS + [
            f[0] for f in MLP_FITTING_METHODS
        ]
    else:
        assert isinstance(estimators_arg, str), "estimators must be a string"
        estimators = estimators_arg.replace("__COMMA__", ",").split(",")
        if ".json" in estimators_arg:
            (
                chosen_estimators,
                chosen_fitting_methods,
            ) = load_estimators_and_fitting_methods(estimators_arg)
        else:
            chosen_estimators = [
                e for e in ESTIMATORS if e in estimators or e in BASE_ESTIMATORS
            ]
            chosen_fitting_methods = [
                f for f in FITTING_METHODS if f[0] in estimators
            ]

    return chosen_estimators, chosen_fitting_methods


def main():
    # Parse command line arguments
    args = parse_arguments()

    apply_random_seed(RANDOM_SEED)

    # Choose estimators and fitting methods
    chosen_estimators, chosen_fitting_methods = choose_estimators(
        args.estimators
    )

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

    data, scenarios, set_of_rows, data_path = load_and_split_model_outputs(
        bench,
        split,
        model_outputs_path=args.model_outputs_path,
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

    # Results
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
        print("MAE results:")
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
        )
        print("Rank results:")
        table_avg, table_std = make_table_avg(
            bench,
            split,
            filename_suffix,
            accs_full,
            scenarios_to_skip=SCENARIOS_TO_SKIP,
            num_it=iterations,
            data_path=data_path,
            agg_type="rank",
        )
        make_results_table(
            table_avg,
            table_std,
            bench,
            args.results_table_path,
            split,
        )


if __name__ == "__main__":
    main()
