from stnd.utility.data_utils import make_or_load_from_cache
from stnd.utility.utils import apply_random_seed
import sys
import os
import time
import sklearn
import numpy as np
import pandas as pd
import torch
import argparse
import torch.nn.functional as F

ROOT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.insert(0, ROOT_PATH)
from experiments import (
    RANDOM_SEED,
    # make_train_test_model_embeddings,
    # make_cache_subpath,
    make_disagreement_scores_dict,
    # make_fitted_weights,
    compute_embedding,
    # make_fitted_model,
)
from utils import (
    # lb_scenarios,
    dump_pickle,
    load_pickle,
    prepare_and_split_data,
    create_predictions,
    create_responses,
)
from plots import safe_spearmanr
from selection import sample_by_disagreement, get_random
from run_experiment import load_and_split_model_outputs

# from acc import compute_true_acc, compute_acc_knn
# from scripts.evaluate_mmlu import evaluate_mmlu

sys.path.pop(0)

SAMPLING_ITERATIONS = None
DEFAULT_SCENARIO = "all"


def derive_scenario_metadata(
    source_outputs, chosen_scenarios=[DEFAULT_SCENARIO]
):
    # if chosen_scenarios is None:
    #     chosen_scenarios = [DEFAULT_SCENARIO]

    # scenarios = {chosen_scenarios[0]: [DEFAULT_SCENARIO]}
    # scenarios_position = {DEFAULT_SCENARIO: list(range(scores_train.shape[1]))}
    # subscenarios_position = {DEFAULT_SCENARIO: {DEFAULT_SCENARIO: list(range(scores_train.shape[1]))}}
    # if "Scenarios" in source_outputs:
    assert len(chosen_scenarios) == 1
    main_scenario = chosen_scenarios[0]
    subscenarios_position = {main_scenario: {}}
    scenarios = {main_scenario: []}
    for scenario_name, positions in source_outputs["Scenarios"].items():
        subscenarios_position[main_scenario][scenario_name] = list(positions)
        scenarios[main_scenario].append(scenario_name)
    # scenarios_position = None
    return scenarios, chosen_scenarios, subscenarios_position


def sample_items_v2(
    number_item,
    # iterations,
    sampling_name,
    source_outputs,
    # target_model_count,
    # scenarios_position,
    # balance_weights,
    random_seed,
):
    """
    Convenience wrapper to sample items without using target_outputs.
    Only source_outputs plus metadata are used; target_model_count provides the test-set size.
    """
    predictions_train = source_outputs["predictions"]
    # scores_train = source_outputs["correctness"][:, :, 0]

    (
        scenarios,
        chosen_scenarios,
        subscenarios_position,
    ) = derive_scenario_metadata(source_outputs)
    # # derive scenario metadata from cached outputs
    # chosen_scenarios = [DEFAULT_SCENARIO]
    # scenarios = {DEFAULT_SCENARIO: [DEFAULT_SCENARIO]}
    # # rebuild subscenarios_position from cached Scenarios mapping
    # subscenarios_position = {DEFAULT_SCENARIO: {DEFAULT_SCENARIO: list(range(scores_train.shape[1]))}}
    # if "Scenarios" in source_outputs:
    #     subscenarios_position = {DEFAULT_SCENARIO: {}}
    #     scenarios = {DEFAULT_SCENARIO: []}
    #     for scenario_name, positions in source_outputs["Scenarios"].items():
    #         subscenarios_position[DEFAULT_SCENARIO][scenario_name] = list(positions)
    #         scenarios[DEFAULT_SCENARIO].append(scenario_name)

    # responses_test is only used for shape; fill zeros with correct shape
    # responses_test = np.zeros((target_model_count, scores_train.shape[1]))

    balance_weights = make_balance_weights(
        source_outputs
    )  # needed for stratified sampling

    disagreement_scores_dict = make_disagreement_scores_dict(
        config={
            "sampling_names": [sampling_name],
            "predictions_train": predictions_train,
            "disagreement_type": "pds",
        }
    )

    return sample_items_impl_v2(
        number_item=number_item,
        sampling_name=sampling_name,
        predictions_train=predictions_train,
        # responses_test=responses_test,
        balance_weights=balance_weights,
        disagreement_scores_dict=disagreement_scores_dict,
        chosen_scenarios=chosen_scenarios,
        scenarios=scenarios,
        subscenarios_position=subscenarios_position,
        random_seed=random_seed,
    )[:2]


def sample_items_impl_v2(
    number_item,
    sampling_name,
    predictions_train,
    # responses_test,
    balance_weights,
    disagreement_scores_dict,
    chosen_scenarios,
    scenarios,
    subscenarios_position,
    random_seed,
):
    """
    Simplified sampler for random/disagreement sampling without full scenario metadata.
    Iterations are read from the module-level SAMPLING_ITERATIONS (default=1).
    """
    # iterations = SAMPLING_ITERATIONS or 1

    # item_weights_dic, seen_items_dic, unseen_items_dic = {}, {}, {}
    start_time = time.time()

    # for it in range(iterations):
    if "disagreement" in sampling_name:
        item_weights, seen_items, unseen_items = sample_by_disagreement(
            sampling_name,
            chosen_scenarios,
            scenarios,
            number_item,
            subscenarios_position,
            num_samples_in_test=None,
            predictions_train=predictions_train,
            balance_weights=None,
            disagreement_scores_dict=disagreement_scores_dict,
            random_seed=random_seed,
            high_first=("high" in sampling_name),
        )
    elif sampling_name == "random":
        item_weights, seen_items, unseen_items = get_random(
            chosen_scenarios,
            scenarios,
            number_item,
            subscenarios_position,
            responses_test=None,
            balance_weights=balance_weights,
            random_seed=random_seed,
        )
    else:
        raise NotImplementedError(
            "sample_items_impl_v2 supports only random or disagreement sampling"
        )

    # item_weights_dic[it] = item_weights
    # seen_items_dic[it] = seen_items
    # unseen_items_dic[it] = unseen_items

    # elapsed_time = (time.time() - start_time) / iterations
    elapsed_time = time.time() - start_time
    return item_weights, seen_items, unseen_items, elapsed_time


def _structures_equal(a, b):
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return np.array_equal(a, b)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(
            _structures_equal(x, y) for x, y in zip(a, b)
        )
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(
            _structures_equal(a[k], b[k]) for k in a.keys()
        )
    return a == b


def print_mae_and_rank_corrs(maes_per_method, rank_corrs_per_method):
    for method_name, maes in maes_per_method.items():
        rank_corrs = rank_corrs_per_method[method_name]
        mae_str = f"{maes.mean(axis=1).mean()} +- {maes.std(axis=1).mean()}"

        rank_corrs_str = f"{rank_corrs.mean().mean()} +- {rank_corrs.std()}"
        print(f"{method_name}: {mae_str} | {rank_corrs_str}")


def make_balance_weights(source_outputs):
    scenarios_map = source_outputs["Scenarios"]
    datapoints_map = source_outputs["Datapoints"]
    n_all_datapoints = len(datapoints_map)
    balance_weights = np.ones(n_all_datapoints)
    n_scenarios = len(scenarios_map)
    for scenario in scenarios_map:
        # n = len(scenarios_map[scenario])
        # N = len(scenarios_position[scenario])
        points_per_scenario = [
            datapoints_map[datapoint_key]
            for datapoint_key in scenarios_map[scenario]
        ]
        n_per_scenario = len(points_per_scenario)
        balance_weights[points_per_scenario] = n_all_datapoints / (
            n_per_scenario * n_scenarios
        )

        # for sub in scenarios[scenario]:
        #     n_i = len(subscenarios_position[scenario][sub])
        #     balance_weights[subscenarios_position[scenario][sub]] = N / (
        #         n_sub * n_i
        #     )
    return balance_weights


def compute_true_acc_v2(source_outputs):
    """
    Compute true accuracies using only source_outputs.

    Parameters:
    - source_outputs: Dictionary containing:
        - "correctness": array of shape (n_models, n_questions, 1) with correctness scores
        - "Scenarios": dict mapping scenario names to lists of datapoint indices
        - "Models": dict mapping model names to local model indices

    Returns:
    - array of shape (n_models,) with true accuracies
    """
    # Extract scores from correctness (remove trailing dimension)
    scores = source_outputs["correctness"][
        :, :, 0
    ]  # shape: (n_models, n_questions)

    # Get balance weights
    balance_weights = make_balance_weights(source_outputs)

    # Get model mapping
    models_map = source_outputs["Models"]  # model_name -> local_index

    model_indices = list(models_map.values())
    accs_true = np.zeros(len(model_indices))
    for j in model_indices:
        accs_true[j] = ((balance_weights[None, :] * scores)[j, :]).mean()
    return accs_true


def compute_acc_knn_v2(
    test_model_embedding,
    train_model_embeddings,
    # scenario,
    train_model_true_accs,
):
    """
    Compute the accuracy of the KNN estimator.
    """

    similarities = F.cosine_similarity(
        test_model_embedding, train_model_embeddings, dim=1
    )

    # Get index of most similar embedding
    most_similar_idx = torch.argmax(similarities).item()

    # Return accuracy of most similar example
    return train_model_true_accs[most_similar_idx]


def compute_predicted_accs_v2(
    target_embeddings_v2,
    fitted_weights,
    train_embeddings_v2,
    train_model_true_accs_new,
    # scenario,
    # rows_to_hide,
    target_id_to_model_name,
    sampling_name,
    number_item,
    # iterations,
    fitted_model_type,
    # items_weights,
):
    """
    Compute predicted_accs_new and predicted_accs_knn_new using target_embeddings_v2 and fitted_weights.

    Similar to the computation in lines 656-683, but uses target_embeddings_v2 and train_embeddings_v2
    which are not per-seed (single arrays instead of per-seed dictionaries).

    Parameters:
    - target_embeddings_v2: array of shape (n_target_models, embedding_dim)
    - fitted_weights: dict with structure fitted_weights[sampling_name][number_item][seed][fitted_model_type]
    - train_embeddings_v2: array of shape (n_train_models, embedding_dim)
    - train_model_true_accs_new: dict mapping model indices to scenario accuracies
    - scenario: string scenario name
    - rows_to_hide: list of target model indices
    - sampling_name: string sampling name
    - number_item: int number of items
    - iterations: int number of iterations/seeds
    - fitted_model_type: string fitted model type

    Returns:
    - predicted_accs_new: dict mapping target_model to list of predicted accuracies
    - predicted_accs_knn_new: dict mapping target_model to list of KNN predicted accuracies
    """
    predictors_per_seed = fitted_weights[sampling_name][number_item]
    predicted_accs_new = {}
    predicted_accs_knn_new = {}

    # target_id_to_model_name = {i: model_name for model_name, i in source_outputs["Models"].items()}

    # for seed in range(iterations):
    # fitted_model = predictors_per_seed[seed][fitted_model_type]
    fitted_model = predictors_per_seed[fitted_model_type]
    for target_model_idx in range(target_embeddings_v2.shape[0]):
        test_model_embedding = target_embeddings_v2[target_model_idx]

        # Convert to numpy if it's a torch tensor
        if isinstance(test_model_embedding, torch.Tensor):
            test_model_embedding_np = test_model_embedding.numpy()
        else:
            test_model_embedding_np = test_model_embedding

        # Predict using fitted model
        fitted_acc = fitted_model.predict(
            test_model_embedding_np.reshape(1, -1)
        )[0]

        # Compute KNN accuracy
        # Convert train_embeddings_v2 to torch tensor if needed for compute_acc_knn
        if isinstance(train_embeddings_v2, torch.Tensor):
            train_embeddings_v2_torch = train_embeddings_v2
        else:
            train_embeddings_v2_torch = torch.from_numpy(train_embeddings_v2)

        if isinstance(test_model_embedding, torch.Tensor):
            test_model_embedding_torch = test_model_embedding
        else:
            test_model_embedding_torch = torch.from_numpy(
                test_model_embedding_np
            )

        fitted_acc_knn = compute_acc_knn_v2(
            test_model_embedding=test_model_embedding_torch,
            train_model_embeddings=train_embeddings_v2_torch,
            # scenario=scenario,
            train_model_true_accs=train_model_true_accs_new,
        )

        # target_model = rows_to_hide[target_model_idx]
        target_model = target_id_to_model_name[target_model_idx]
        if target_model not in predicted_accs_new:
            predicted_accs_new[target_model] = []
            predicted_accs_knn_new[target_model] = []
        predicted_accs_new[target_model].append(fitted_acc)
        predicted_accs_knn_new[target_model].append(fitted_acc_knn)

    return predicted_accs_new, predicted_accs_knn_new


def make_fitted_model_v2(config, logger=None):
    (
        builder,
        sampling_name,
        number_item,
        # it,
        cur_train_models_embeddings_np,
        train_model_true_accs_np,
        # fitted_weights,
        model_name,
    ) = (
        config["builder"],
        config["sampling_name"],
        config["number_item"],
        # config["it"],
        config["cur_train_models_embeddings_np"],
        config["train_model_true_accs_np"],
        # config["fitted_weights"],
        config["model_name"],
    )
    # skip_iterations_when_fixed_sampling = config.get(
    #     "skip_iterations_when_fixed_sampling", True
    # )
    builder_func, builder_kwargs = builder
    model = builder_func(**builder_kwargs)
    # if (
    #     skip_iterations_when_fixed_sampling
    #     and sampling_name in ["high-disagreement", "low-disagreement"]
    #     and "@" not in sampling_name
    #     and it > 0
    # ):
    #     # for deterministic sampling fitted model does not change for different runs
    #     fitted_model = deepcopy(
    #         fitted_weights[sampling_name][number_item][0][f"{model_name}"]
    #     )
    # else:
    fitted_model = model.fit(
        cur_train_models_embeddings_np, train_model_true_accs_np
    )
    # when shape like (159, 102400), RandomForest200 can take 5+ min

    # Compute training RMSE
    train_preds = fitted_model.predict(cur_train_models_embeddings_np)
    train_rmse = np.sqrt(np.mean((train_preds - train_model_true_accs_np) ** 2))

    print(
        f"Training RMSE of {model_name} for samplng={sampling_name}, n_anchor={number_item}: {train_rmse}"
    )

    return fitted_model


def make_fitted_weights_v2(config, logger=None):
    (
        sampling_names,
        number_items,
        # iterations,
        train_models_embeddings,
        train_model_true_accs,
        # scenario,
        cache_path,
        chosen_fitting_methods,
    ) = (
        config["sampling_names"],
        config["number_items"],
        # config["iterations"],
        config["train_models_embeddings"],
        config["train_model_true_accs"],
        # config["scenario"],
        config["cache_path"],
        config["chosen_fitting_methods"],
    )
    # skip_iterations_when_fixed_sampling = config.get(
    #     "skip_iterations_when_fixed_sampling", True
    # )

    fitted_weights = {}

    train_model_true_accs_np = np.array(
        [train_model_true_accs[i] for i in range(len(train_model_true_accs))]
    )

    for sampling_name in sampling_names:
        fitted_weights[sampling_name] = {}
        for number_item in number_items:
            fitted_weights[sampling_name][number_item] = {}
            # for it in range(iterations):
            # cur_train_models_embeddings_np = train_models_embeddings[
            #     sampling_name
            # ][number_item][it].numpy()
            cur_train_models_embeddings_np = train_models_embeddings

            # fitted_weights[sampling_name][number_item][it] = {}

            for model_name, builder in chosen_fitting_methods:
                if cache_path is not None:
                    cache_dir = cache_path.split(".")[0] + "_folder"
                else:
                    cache_dir = None
                fitted_model = make_or_load_from_cache(
                    object_name=f"fitted_model_{model_name}_{sampling_name}_{number_item}",
                    object_config={
                        "builder": builder,
                        "sampling_name": sampling_name,
                        "number_item": number_item,
                        # "it": 0,
                        "cur_train_models_embeddings_np": cur_train_models_embeddings_np,
                        "train_model_true_accs_np": train_model_true_accs_np,
                        # "fitted_weights": fitted_weights,
                        "model_name": model_name,
                        # "skip_iterations_when_fixed_sampling": skip_iterations_when_fixed_sampling,
                    },
                    make_func=make_fitted_model_v2,
                    cache_path=cache_dir,
                )

                fitted_weights[sampling_name][number_item][
                    f"{model_name}"
                ] = fitted_model

    return fitted_weights


def main():
    bench = "mmlu_fields"
    # data, scenarios, set_of_rows, data_path = load_and_split_model_outputs(
    #     bench=bench,
    #     split="noniid",
    #     model_outputs_path=MODEL_OUTPUTS_PATH,
    #     text_to_vector=None,
    #     return_data_path=True,
    #     subsample_validation=False,
    # )

    parser = argparse.ArgumentParser(description="2stages_v2.py")
    parser.add_argument(
        "--source_model_outputs_path",
        type=str,
        default=None,
        help="Path to the source model outputs file",
    )
    parser.add_argument(
        "--target_model_outputs_path",
        type=str,
        default=None,
        help="Path to the target model outputs file",
    )
    args = parser.parse_args()

    # cache_dir = os.path.join(ROOT_PATH, "cache")
    # target_cache_path = os.path.join(cache_dir, "target_outputs2.pkl")
    # source_cache_path = os.path.join(cache_dir, "source_outputs2.pkl")
    # (
    #     target_outputs,
    #     source_outputs,
    #     data,
    #     scenarios,
    #     set_of_rows,
    #     data_path,
    #     bench,
    # ) = load_or_make_outputs(
    #     target_cache_path=target_cache_path,
    #     source_cache_path=source_cache_path,
    #     save=True,
    # )

    target_outputs = load_pickle(args.target_model_outputs_path)
    source_outputs = load_pickle(args.source_model_outputs_path)

    # chosen_scenarios = list(scenarios.keys())
    # split_number = 0
    # rows_to_hide = set_of_rows[split_number]

    # (
    #     scores_train,
    #     predictions_train,
    #     predictions_test,
    #     scores_test,
    #     balance_weights,
    #     scenarios_position,
    #     subscenarios_position,
    # ) = prepare_and_split_data(
    #     chosen_scenarios,
    #     scenarios,
    #     data,
    #     rows_to_hide=rows_to_hide,
    #     n_source_models=None,
    # )

    predictions_train = source_outputs["predictions"]

    sampling_names = ["high-disagreement"]
    disagreement_type = "pds"
    disagreement_scores_dict = make_disagreement_scores_dict(
        config={
            "sampling_names": sampling_names,
            "predictions_train": predictions_train,
            "disagreement_type": disagreement_type,
        }
    )

    number_items = [100]
    iterations = 1
    sampling_name = sampling_names[0]
    number_item = number_items[0]
    seen_items_dic = {sampling_name: {}}
    apply_random_seed(RANDOM_SEED)
    # samples = sample_items(
    #     number_item,
    #     iterations,
    #     sampling_name,
    #     chosen_scenarios,
    #     scenarios,
    #     subscenarios_position,
    #     responses_test=scores_test,
    #     scores_train=scores_train,
    #     predictions_train=predictions_train,
    #     scenarios_position=scenarios_position,
    #     A=None,
    #     B=None,
    #     balance_weights=balance_weights,
    #     disagreement_scores_dict=disagreement_scores_dict,
    #     skip_irt=True,
    # )
    # (
    #     item_weights_old,
    #     seen_items_dic[sampling_name][number_item],
    #     _,
    #     _,
    # ) = samples

    # expose iterations to the simplified sampler
    # global SAMPLING_ITERATIONS
    # SAMPLING_ITERATIONS = iterations

    samples_v2 = sample_items_v2(
        number_item=number_item,
        # iterations=iterations,
        sampling_name=sampling_name,
        source_outputs=source_outputs,
        # target_model_count=len(rows_to_hide),
        # chosen_scenarios=chosen_scenarios,
        # scenarios=scenarios,
        # subscenarios_position=subscenarios_position,
        # scenarios_position=scenarios_position,
        # balance_weights=balance_weights,
        random_seed=RANDOM_SEED,
    )
    item_weights_new, anchor_points_new = samples_v2
    samples_v2_random = sample_items_v2(
        number_item=number_item,
        # iterations=iterations,
        sampling_name="random",
        source_outputs=source_outputs,
        # target_model_count=len(rows_to_hide),
        # chosen_scenarios=chosen_scenarios,
        # scenarios=scenarios,
        # subscenarios_position=subscenarios_position,
        # scenarios_position=scenarios_position,
        # balance_weights=balance_weights,
        random_seed=RANDOM_SEED,
    )
    (
        item_weights_new_random,
        anchor_points_new_random_stratified,
    ) = samples_v2_random
    # assert _structures_equal(
    #     item_weights_old[0]["mmlu"], item_weights_new["all"]
    # ), "item_weights differ"
    # assert _structures_equal(
    #     seen_items_dic[sampling_name][number_item][0], anchor_points_new
    # ), "seen_items differ"

    # load embeddings
    # cache_path = ""
    # cache = load_pickle(cache_path)
    # scenario_name = "mmlu"
    # split_number = 0
    # emb_cache_path = (
    #     make_cache_subpath(
    #         cache, scenario_name, split_number, f"embeddings_path"
    #     )
    # )

    pca = 256
    # apply_softmax_to_predictions = True
    # (
    #     train_models_embeddings,
    #     test_models_embeddings,
    # ) = make_train_test_model_embeddings(
    #     emb_config={
    #         "sampling_names": sampling_names,
    #         "number_items": number_items,
    #         "iterations": iterations,
    #         "predictions_train": predictions_train,
    #         "seen_items_dic": seen_items_dic,
    #         "predictions_test": predictions_test,
    #         "seen_items_dic": seen_items_dic,
    #         "pca": pca,
    #         "apply_softmax": apply_softmax_to_predictions,
    #     }
    # )

    predictions_train_v2 = source_outputs["predictions"][
        :, anchor_points_new, :
    ]
    train_embeddings_v2, transform_v2 = compute_embedding(
        predictions_train_v2,
        anchor_indices=None,
        pca=pca,
        transform=None,
        apply_softmax=True,
    )
    # assert np.allclose(
    #     train_embeddings_v2,
    #     train_models_embeddings[sampling_name][number_item][0],
    # ), "train_embeddings_v2 differ"

    predictions_test_v2 = target_outputs["predictions"][:, anchor_points_new, :]
    target_embeddings_v2, _ = compute_embedding(
        predictions_test_v2,
        anchor_indices=None,
        pca=pca,
        transform=transform_v2,
        apply_softmax=True,
    )
    # assert np.allclose(
    #     target_embeddings_v2,
    #     test_models_embeddings[sampling_name][number_item][0],
    # ), "target_embeddings_v2 differ"

    # target_embeddings_v2, _ = compute_embedding(
    #     predictions_test,
    #     seen_items_new,
    #     pca,
    #     transform_v2,
    #     apply_softmax=True,
    # )

    # make_or_load_from_cache(
    #     object_name="train_test_model_embeddings",
    #     object_config={
    #         "sampling_names": sampling_names,
    #         "number_items": number_items,
    #         "iterations": iterations,
    #         "predictions_train": predictions_train,
    #         "seen_items_dic": seen_items_dic,
    #         "predictions_test": predictions_test,
    #         "seen_items_dic": seen_items_dic,
    #         "pca": pca,
    #         "apply_softmax": apply_softmax_to_predictions,
    #     },
    #     make_func=make_train_test_model_embeddings,
    #     cache_path=emb_cache_path,
    # )

    # train_model_indices = list(range(scores_train.shape[0]))
    # train_model_true_accs = compute_true_acc(
    #     scores_train,
    #     balance_weights,  # sample -> sample weight
    #     scenarios_position,  # scenario -> list of sample indices
    #     chosen_scenarios,
    #     train_model_indices,
    #     train_model_indices,  # they are not the global indices, but the contiguous indices of train models after removing test models
    # )

    train_model_true_accs_new = compute_true_acc_v2(
        source_outputs,
    )
    # assert _structures_equal(
    #     train_model_true_accs, train_model_true_accs_new
    # ), "train_model_true_accs differ"

    # "RandomForestRegressor_100": {
    #     "class_path": "sklearn.ensemble.RandomForestRegressor",
    #     "params": {"n_estimators": 100}
    # },

    fitted_model_type = "RandomForestRegressor_100"
    chosen_fitting_methods = [
        (
            fitted_model_type,
            (sklearn.ensemble.RandomForestRegressor, {"n_estimators": 100}),
        )
    ]
    # scenario = bench.split("_")[0]
    fitted_weights = make_fitted_weights_v2(
        config={
            "sampling_names": sampling_names,
            "number_items": number_items,
            # "iterations": iterations,
            "train_models_embeddings": train_embeddings_v2,
            "train_model_true_accs": train_model_true_accs_new,
            # "scenario": scenario,
            "cache_path": "cache/fitted_weights.",  # dot for legacy reasons
            "chosen_fitting_methods": chosen_fitting_methods,
            # "skip_iterations_when_fixed_sampling": False,
        },
        # cache_path="./cache",
        # forward_cache_path=True,
    )

    # make_or_load_from_cache(
    #     object_name="fitted_weights",
    #     object_config={
    #         "sampling_names": sampling_names,
    #         "number_items": number_items,
    #         "iterations": iterations,
    #         "train_models_embeddings": train_models_embeddings,
    #         "train_model_true_accs": train_model_true_accs,
    #         "scenario": bench,
    #         "cache_path": fitted_weights_cache_path,
    #         "chosen_fitting_methods": chosen_fitting_methods,
    #     },
    #     make_func=make_fitted_weights,
    #     cache_path=fitted_weights_cache_path,
    # )

    # predictors_per_seed = fitted_weights[sampling_names[0]][100]
    # target_model_embeddings_per_seed = test_models_embeddings[
    #     sampling_names[0]
    # ][100]
    # predicted_accs = {}
    # predicted_accs_knn = {}
    # for seed in range(iterations):
    #     fitted_model = predictors_per_seed[seed][fitted_model_type]
    #     target_model_embeddings = target_model_embeddings_per_seed[seed]
    #     for target_model_idx in range(target_model_embeddings.shape[0]):
    #         test_model_embedding = target_model_embeddings[target_model_idx]
    #         fitted_acc = fitted_model.predict(
    #             test_model_embedding.numpy().reshape(1, -1)
    #         )[0]
    #         fitted_acc_knn = compute_acc_knn(
    #             test_model_embedding=test_model_embedding,
    #             train_model_embeddings=train_models_embeddings[sampling_name][
    #                 number_item
    #             ][seed],
    #             scenario=scenario,
    #             train_model_true_accs=train_model_true_accs,
    #         )
    #         target_model = rows_to_hide[target_model_idx]
    #         if target_model not in predicted_accs:
    #             predicted_accs[target_model] = []
    #             predicted_accs_knn[target_model] = []
    #         predicted_accs[target_model].append(fitted_acc)
    #         predicted_accs_knn[target_model].append(fitted_acc_knn)

    target_id_to_model_name = {
        i: model_name for model_name, i in target_outputs["Models"].items()
    }
    predicted_accs_new, predicted_accs_knn_new = compute_predicted_accs_v2(
        target_embeddings_v2=target_embeddings_v2,
        fitted_weights=fitted_weights,
        train_embeddings_v2=train_embeddings_v2,
        train_model_true_accs_new=train_model_true_accs_new,
        # scenario=scenario,
        target_id_to_model_name=target_id_to_model_name,
        sampling_name=sampling_name,
        number_item=number_item,
        # iterations=iterations,
        fitted_model_type=fitted_model_type,
        # items_weights=item_weights_new,
    )

    def _compute_naive_accuracies(
        scores,
        item_weights_new,
        item_weights_new_random,
        anchor_points_new,
        anchor_points_new_random_stratified,
        random_anchors,
        target_model_idx,
    ):
        """Compute disagreement, random, and stratified-random naive accuracies for a target model."""
        disagreement_naive_acc = (
            item_weights_new["all"]
            * scores[target_model_idx][anchor_points_new]
        ).sum()
        stratified_random_naive_acc = (
            item_weights_new_random["all"]
            * scores[target_model_idx][anchor_points_new_random_stratified]
        ).sum()
        random_naive_acc = scores[target_model_idx][random_anchors].mean()
        return (
            disagreement_naive_acc,
            random_naive_acc,
            stratified_random_naive_acc,
        )

    # assert _structures_equal(
    #     predicted_accs, predicted_accs_new
    # ), "predicted_accs differ"
    # assert _structures_equal(
    #     predicted_accs_knn, predicted_accs_knn_new
    # ), "predicted_accs_knn differ"

    # scores = load_scores(
    #     bench,
    #     split="noniid",
    #     scenarios_to_skip=[],
    #     ordered=True,
    #     filename_suffix="",
    #     num_it=5,
    #     data_path=None,
    #     filter_indices_by_results=False,
    # )
    # gt_scores = scores[:, rows_to_hide]

    # maes_per_method = {}
    # rank_corrs_per_method = {}
    # predictions = {"fitted": predicted_accs, "knn": predicted_accs_knn}
    # for method_name, accs in predictions.items():
    #     rank_corrs = []
    #     # pred_accs_as_np = np.stack(list(predicted_accs.values()), axis=0)
    #     pred_accs_as_np = np.stack(list(accs.values()), axis=0)
    #     maes = np.abs(pred_accs_as_np - gt_scores[0][:, None])
    #     for i in range(pred_accs_as_np.shape[1]):
    #         rank_corrs.append(
    #             safe_spearmanr(
    #                 pred_accs_as_np[:, i],
    #                 gt_scores[0][:, None],
    #             )
    #         )
    #     maes_per_method[method_name] = maes
    #     rank_corrs_per_method[method_name] = np.array(rank_corrs)

    gt_scores_new = {}
    balance_weights_new = make_balance_weights(target_outputs)
    correctness_new = target_outputs["correctness"][..., 0]
    scores = correctness_new * balance_weights_new

    predicted_accs_disagreement_naive_new = {}
    predicted_accs_random_naive_new = {}
    predicted_accs_stratified_random_naive_new = {}

    predicted_accs_disagreement_naive_corr_new = {}
    predicted_accs_stratified_random_naive_corr_new = {}
    predicted_accs_random_naive_corr_new = {}

    random_anchors = np.random.choice(
        list(range(len(scores[0]))), size=number_item, replace=False
    )
    for target_model_name in target_outputs["Models"].keys():
        target_model_idx = target_outputs["Models"][target_model_name]
        gt_scores_new[target_model_name] = scores[target_model_idx].mean()

        (
            disagreement_naive_acc,
            random_naive_acc,
            stratified_random_naive_acc,
        ) = _compute_naive_accuracies(
            scores=scores,
            item_weights_new=item_weights_new,
            item_weights_new_random=item_weights_new_random,
            anchor_points_new=anchor_points_new,
            anchor_points_new_random_stratified=anchor_points_new_random_stratified,
            random_anchors=random_anchors,
            target_model_idx=target_model_idx,
        )

        predicted_accs_disagreement_naive_new[target_model_name] = [
            disagreement_naive_acc
        ]
        predicted_accs_stratified_random_naive_new[target_model_name] = [
            stratified_random_naive_acc
        ]
        predicted_accs_random_naive_new[target_model_name] = [random_naive_acc]

        (
            disagreement_naive_acc_corr,
            random_naive_acc_corr,
            stratified_random_naive_acc_corr,
        ) = _compute_naive_accuracies(
            scores=correctness_new,
            item_weights_new=item_weights_new,
            item_weights_new_random=item_weights_new_random,
            anchor_points_new=anchor_points_new,
            anchor_points_new_random_stratified=anchor_points_new_random_stratified,
            random_anchors=random_anchors,
            target_model_idx=target_model_idx,
        )

        predicted_accs_disagreement_naive_corr_new[target_model_name] = [
            disagreement_naive_acc_corr
        ]
        predicted_accs_stratified_random_naive_corr_new[target_model_name] = [
            stratified_random_naive_acc_corr
        ]
        predicted_accs_random_naive_corr_new[target_model_name] = [
            random_naive_acc_corr
        ]

    # for target_model_name in target_outputs["Models"].keys():
    #     naive_acc = (
    #         item_weights_new[scenario]
    #         * gt_scores_new[j][selected_indices]
    #     ).sum()
    #     predictions_new["naive"][target_model_name] = naive_acc

    maes_per_method_new = {}
    rank_corrs_per_method_new = {}
    predictions_new = {
        "fitted": predicted_accs_new,
        "knn": predicted_accs_knn_new,
        "disagreement+naive": predicted_accs_disagreement_naive_new,
        "stratified_random+naive": predicted_accs_stratified_random_naive_new,
        "random+naive": predicted_accs_random_naive_new,
        "disagreement+naive+corr": predicted_accs_disagreement_naive_corr_new,
        "stratified_random+naive+corr": predicted_accs_stratified_random_naive_corr_new,
        "random+naive+corr": predicted_accs_random_naive_corr_new,
    }
    gt_scores_new_np = np.stack(
        [
            gt_scores_new[target_model_idx]
            for target_model_idx in target_outputs["Models"].keys()
        ],
        axis=0,
    )
    for method_name, accs in predictions_new.items():
        rank_corrs_new = []
        # pred_accs_as_np = np.stack(list(predicted_accs.values()), axis=0)
        # pred_accs_as_np = np.stack(list(accs.values()), axis=0)
        pred_accs_as_np_new = np.stack(
            list(
                [
                    accs[target_model_name]
                    for target_model_name in target_outputs["Models"].keys()
                ]
            ),
            axis=0,
        )
        maes_new = np.abs(pred_accs_as_np_new - gt_scores_new_np[:, None])
        for i in range(pred_accs_as_np_new.shape[1]):
            rank_corrs_new.append(
                safe_spearmanr(
                    pred_accs_as_np_new[:, i],
                    gt_scores_new_np[:, None],
                )
            )
        maes_per_method_new[method_name] = maes_new
        rank_corrs_per_method_new[method_name] = np.array(rank_corrs_new)

    # assert _structures_equal(
    #     maes_per_method, maes_per_method_new
    # ), "predicted_accs differ"
    # assert _structures_equal(
    #     rank_corrs_per_method, rank_corrs_per_method_new
    # ), "predicted_accs_knn differ"

    # print_mae_and_rank_corrs(maes_per_method, rank_corrs_per_method)
    print_mae_and_rank_corrs(maes_per_method_new, rank_corrs_per_method_new)


if __name__ == "__main__":
    main()
