from stnd.utility.data_utils import make_or_load_from_cache
from stnd.utility.utils import apply_random_seed
import sys
import os
import sklearn
import numpy as np
import pandas as pd

ROOT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.insert(0, ROOT_PATH)
from experiments import (
    RANDOM_SEED,
    make_train_test_model_embeddings,
    make_cache_subpath,
    make_disagreement_scores_dict,
    make_fitted_weights,
)
from utils import (
    lb_scenarios,
    dump_pickle,
    load_pickle,
    prepare_and_split_data,
    create_predictions,
    create_responses,
)
from plots import MODEL_OUTPUTS_PATH, load_scores, safe_spearmanr
from selection import sample_items
from run_experiment import load_and_split_model_outputs
from acc import compute_true_acc, compute_acc_knn
from scripts.evaluate_mmlu import evaluate_mmlu

sys.path.pop(0)


def sample_items_v2(
    number_item,
    iterations,
    sampling_name,
    source_outputs,
    target_model_count,
    chosen_scenarios,
    scenarios,
    subscenarios_position,
    scenarios_position,
    balance_weights,
):
    """
    Convenience wrapper to sample items without using target_outputs.
    Only source_outputs plus metadata are used; target_model_count provides the test-set size.
    """
    predictions_train = source_outputs["predictions"]
    scores_train = source_outputs["correctness"][:, :, 0]

    # responses_test is only used for shape; fill zeros with correct shape
    responses_test = np.zeros((target_model_count, scores_train.shape[1]))

    disagreement_scores_dict = make_disagreement_scores_dict(
        config={
            "sampling_names": [sampling_name],
            "predictions_train": predictions_train,
            "disagreement_type": "pds",
        }
    )

    return sample_items(
        number_item,
        iterations,
        sampling_name,
        chosen_scenarios,
        scenarios,
        subscenarios_position,
        responses_test=responses_test,
        scores_train=scores_train,
        predictions_train=predictions_train,
        scenarios_position=scenarios_position,
        A=None,
        B=None,
        balance_weights=balance_weights,
        disagreement_scores_dict=disagreement_scores_dict,
        skip_irt=True,
    )[:2]


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


def load_or_make_outputs(target_cache_path, source_cache_path, save=False):
    """
    Load target/source outputs from disk if available; otherwise build from data and save.
    """

    if os.path.exists(target_cache_path) and os.path.exists(source_cache_path):
        return (
            load_pickle(target_cache_path),
            load_pickle(source_cache_path),
            None,
            None,
            None,
            None,
        )

    bench = "mmlu_fields"
    data, scenarios, set_of_rows, data_path = load_and_split_model_outputs(
        bench=bench,
        split="noniid",
        model_outputs_path=MODEL_OUTPUTS_PATH,
        text_to_vector=None,
        return_data_path=True,
        subsample_validation=False,
    )

    chosen_scenarios = list(scenarios.keys())
    all_predictions = create_predictions(chosen_scenarios, scenarios, data)
    all_correctness = create_responses(chosen_scenarios, scenarios, data)[
        :, :, None
    ]  # add trailing dim (N, Q, 1)

    def build_outputs_dict(model_indices):
        preds = all_predictions[model_indices]
        corr = all_correctness[model_indices]
        models_map = {
            data["models"][orig_idx]: local_idx
            for local_idx, orig_idx in enumerate(model_indices)
        }
        datapoints_map = {dp_idx: dp_idx for dp_idx in range(preds.shape[1])}
        return {
            "predictions": preds,
            "correctness": corr,
            "Models": models_map,
            "Datapoints": datapoints_map,
        }

    target_model_indices = set_of_rows[0]
    source_model_indices = [
        i
        for i in range(all_predictions.shape[0])
        if i not in target_model_indices
    ]

    target_outputs = build_outputs_dict(target_model_indices)
    source_outputs = build_outputs_dict(source_model_indices)

    if save:
        os.makedirs(os.path.dirname(target_cache_path), exist_ok=True)
        dump_pickle(target_outputs, target_cache_path)
        dump_pickle(source_outputs, source_cache_path)

    return (
        target_outputs,
        source_outputs,
        data,
        scenarios,
        set_of_rows,
        data_path,
        bench,
    )


def main():
    # bench = "mmlu_fields"
    # data, scenarios, set_of_rows, data_path = load_and_split_model_outputs(
    #     bench=bench,
    #     split="noniid",
    #     model_outputs_path=MODEL_OUTPUTS_PATH,
    #     text_to_vector=None,
    #     return_data_path=True,
    #     subsample_validation=False,
    # )

    cache_dir = os.path.join(ROOT_PATH, "cache")
    target_cache_path = os.path.join(cache_dir, "target_outputs2.pkl")
    source_cache_path = os.path.join(cache_dir, "source_outputs2.pkl")
    (
        target_outputs,
        source_outputs,
        data,
        scenarios,
        set_of_rows,
        data_path,
        bench,
    ) = load_or_make_outputs(
        target_cache_path=target_cache_path,
        source_cache_path=source_cache_path,
        save=False,
    )

    chosen_scenarios = list(scenarios.keys())
    split_number = 0
    rows_to_hide = set_of_rows[split_number]

    (
        scores_train,
        predictions_train,
        predictions_test,
        scores_test,
        balance_weights,
        scenarios_position,
        subscenarios_position,
    ) = prepare_and_split_data(
        chosen_scenarios,
        scenarios,
        data,
        rows_to_hide=rows_to_hide,
        n_source_models=None,
    )

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
    samples = sample_items(
        number_item,
        iterations,
        sampling_name,
        chosen_scenarios,
        scenarios,
        subscenarios_position,
        responses_test=scores_test,
        scores_train=scores_train,
        predictions_train=predictions_train,
        scenarios_position=scenarios_position,
        A=None,
        B=None,
        balance_weights=balance_weights,
        disagreement_scores_dict=disagreement_scores_dict,
        skip_irt=True,
    )
    (
        item_weights_old,
        seen_items_dic[sampling_name][number_item],
        _,
        _,
    ) = samples

    samples_v2 = sample_items_v2(
        number_item=number_item,
        iterations=iterations,
        sampling_name=sampling_name,
        source_outputs=source_outputs,
        target_model_count=len(rows_to_hide),
        chosen_scenarios=chosen_scenarios,
        scenarios=scenarios,
        subscenarios_position=subscenarios_position,
        scenarios_position=scenarios_position,
        balance_weights=balance_weights,
    )
    item_weights_new, seen_items_new = samples_v2
    assert _structures_equal(
        item_weights_old, item_weights_new
    ), "item_weights differ"
    assert _structures_equal(
        seen_items_dic[sampling_name][number_item], seen_items_new
    ), "seen_items differ"

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
    apply_softmax_to_predictions = True
    (
        train_models_embeddings,
        test_models_embeddings,
    ) = make_train_test_model_embeddings(
        emb_config={
            "sampling_names": sampling_names,
            "number_items": number_items,
            "iterations": iterations,
            "predictions_train": predictions_train,
            "seen_items_dic": seen_items_dic,
            "predictions_test": predictions_test,
            "seen_items_dic": seen_items_dic,
            "pca": pca,
            "apply_softmax": apply_softmax_to_predictions,
        }
    )

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

    train_model_indices = list(range(scores_train.shape[0]))
    train_model_true_accs = compute_true_acc(
        scores_train,
        balance_weights,  # sample -> sample weight
        scenarios_position,  # scenario -> list of sample indices
        chosen_scenarios,
        train_model_indices,
        train_model_indices,  # they are not the global indices, but the contiguous indices of train models after removing test models
    )

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
    scenario = bench.split("_")[0]
    fitted_weights = make_fitted_weights(
        config={
            "sampling_names": sampling_names,
            "number_items": number_items,
            "iterations": iterations,
            "train_models_embeddings": train_models_embeddings,
            "train_model_true_accs": train_model_true_accs,
            "scenario": scenario,
            "cache_path": "cache/fitted_weights.",  # dot for legacy reasons
            "chosen_fitting_methods": chosen_fitting_methods,
            "skip_iterations_when_fixed_sampling": False,
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

    predictors_per_seed = fitted_weights[sampling_names[0]][100]
    target_model_embeddings_per_seed = test_models_embeddings[
        sampling_names[0]
    ][100]
    predicted_accs = {}
    predicted_accs_knn = {}
    for seed in range(iterations):
        fitted_model = predictors_per_seed[seed][fitted_model_type]
        target_model_embeddings = target_model_embeddings_per_seed[seed]
        for target_model_idx in range(target_model_embeddings.shape[0]):
            test_model_embedding = target_model_embeddings[target_model_idx]
            fitted_acc = fitted_model.predict(
                test_model_embedding.numpy().reshape(1, -1)
            )[0]
            fitted_acc_knn = compute_acc_knn(
                test_model_embedding=test_model_embedding,
                train_model_embeddings=train_models_embeddings[sampling_name][
                    number_item
                ][seed],
                scenario=scenario,
                train_model_true_accs=train_model_true_accs,
            )
            target_model = rows_to_hide[target_model_idx]
            if target_model not in predicted_accs:
                predicted_accs[target_model] = []
                predicted_accs_knn[target_model] = []
            predicted_accs[target_model].append(fitted_acc)
            predicted_accs_knn[target_model].append(fitted_acc_knn)

    scores = load_scores(
        bench,
        split="noniid",
        scenarios_to_skip=[],
        ordered=True,
        filename_suffix="",
        num_it=5,
        data_path=None,
        filter_indices_by_results=False,
    )
    gt_scores = scores[:, rows_to_hide]

    maes_per_method = {}
    rank_corrs_per_method = {}
    predictions = {"fitted": predicted_accs, "knn": predicted_accs_knn}
    for method_name, accs in predictions.items():
        rank_corrs = []
        # pred_accs_as_np = np.stack(list(predicted_accs.values()), axis=0)
        pred_accs_as_np = np.stack(list(accs.values()), axis=0)
        maes = np.abs(pred_accs_as_np - gt_scores[0][:, None])
        for i in range(pred_accs_as_np.shape[1]):
            rank_corrs.append(
                safe_spearmanr(
                    pred_accs_as_np[:, i],
                    gt_scores[0][:, None],
                )
            )
        maes_per_method[method_name] = maes
        rank_corrs_per_method[method_name] = np.array(rank_corrs)

    for method_name, maes in maes_per_method.items():
        rank_corrs = rank_corrs_per_method[method_name]
        mae_str = f"{maes.mean(axis=1).mean()} +- {maes.std(axis=1).mean()}"

        rank_corrs_str = f"{rank_corrs.mean().mean()} +- {rank_corrs.std()}"
        print(f"{method_name}: {mae_str} | {rank_corrs_str}")


if __name__ == "__main__":
    main()
