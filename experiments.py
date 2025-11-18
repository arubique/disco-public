# The code is adapted from the tinyBenchmarks repo: https://github.com/felipemaiapolo/efficbench
from tqdm import tqdm
import pickle
from copy import copy
import multiprocessing as mp
import time
import os
from copy import deepcopy
import umap

# from irt import *
# from selection import *
from irt import (
    create_irt_dataset,
    train_irt_model,
    load_irt_parameters,
    estimate_ability_parameters,
)
from utils import (
    get_lambda,
    load_pickle,
    dump_pickle,
    prepare_and_split_data,
    item_curve,
)
from selection import (
    select_initial_adaptive_items,
    sample_items,
    sample_items_adaptive,
    get_disagreement_scores,
)

# from acc import *
from acc import (
    ESTIMATORS,
    FITTING_METHODS,
    make_method_name,
    calculate_accuracies,
    compute_true_acc,
)

# from utils_from_stuned import make_or_load_from_cache
from stnd.utility.data_utils import make_or_load_from_cache

# from generating_data.utils_for_notebooks import compare_dicts_with_arrays
import numpy as np
import torch
from sklearn.decomposition import PCA
from stnd.utility.utils import optionally_make_parent_dir


RANDOM_SEED = 42


def make_cache_key(scenario_name, split_number, suffix):
    return f"{scenario_name}_{split_number}_{suffix}"


def evaluate_scenarios(
    data,
    scenario_name,
    chosen_scenarios,
    scenarios,
    set_of_rows,
    Ds,
    iterations,
    device,
    bench,
    split,
    sampling_names=["random", "anchor", "anchor-irt"],
    num_workers=1,
    skip_irt=False,
    cache=None,
    chosen_estimators=None,
    chosen_fitting_methods=None,
    pca=None,
    n_source_models=None,
    number_items=[10, 30, 60, 100],
    apply_softmax_to_predictions=True,
    disagreement_type="pds",
    skip_it_fixed_sampling=True,
    vary_selection=False,
    # different_split_per_iteration=False,
    # bootstrap_ratio=0.9,
):
    """
    Evaluates scenarios by training and validating IRT models, then computing accuracies and updating results.

    Parameters:
    - data: A dictionary containing the dataset.
    - scenario_name: The name of the current scenario.
    - chosen_scenarios: A list of scenarios to be considered.
    - scenarios: A dictionary mapping each scenario to its subscenarios.
    - set_of_rows: A set of row indices to hide during training (to simulate missing data).
    - Ds: A list of dimension values to consider for the IRT model.
    - iterations: The number of iterations to perform for random evaluations.
    - device: The computing device ('cpu' or 'gpu') to use for training.

    Returns:
    - A dictionary containing the updated results.
    """

    assert bench in [
        "irt_helm_lite",
        "irt_lb",
        "irt_lb_perf",
        "irt_mmlu",
        "irt_alpaca",
        "irt_mmlu_fields",
        "irt_icl_templates",
        "irt_truthfulqa",
        "irt_winogrande",
        "irt_arc",
        "irt_hellaswag",
    ]
    assert any(
        [
            s
            in [
                "random",
                "anchor",
                "anchor-irt",
                "adaptive",
                "high-disagreement",
                "low-disagreement",
            ]
            or "high-disagreement" in s
            for s in sampling_names
        ]
    )  # [ADD][new sampling]

    # number_items = [10, 30, 60, 100]  # Number of items to consider in evaluations

    if cache is not None:
        optionally_make_parent_dir(os.path.dirname(cache["cache_path"]))

    # cpu = mp.cpu_count()  # Number of available CPU cores
    cpu = num_workers
    epochs = 2000  # Number of epochs for IRT model training (package default is 2000)
    lr = 0.1  # Learning rate for IRT model training (package default is .1)

    # Iterate through each set of rows to hide
    # accs_true = {}  # Initialize a dictionary to hold real accuracies
    out = []  # To store intermediate results

    accs_true_dict = {}
    for split_number, rows_to_hide in enumerate(set_of_rows):
        rows_to_hide_str = (
            ":".join([str(r) for r in rows_to_hide])[:30]
            + ":".join([str(r) for r in rows_to_hide])[-30:]
        )

        print(f"\nEvaluating models {rows_to_hide}")

        # Prepare data and scenarios
        (
            scores_train,
            predictions_train,
            predictions_test,
            scores_test,
            balance_weights,
            scenarios_position,
            subscenarios_position,
            # _,
        ) = prepare_and_split_data(
            chosen_scenarios,
            scenarios,
            data,
            rows_to_hide,
            n_source_models=n_source_models,
            # iterations=None,
            # different_split_per_iteration=different_split_per_iteration,
            # bootstrap_ratio=bootstrap_ratio,
        )

        responses_train = np.zeros(scores_train.shape)
        responses_test = np.zeros(scores_test.shape)

        start_time_pre_sampling_stage = time.time()

        # Threshold responses
        cs = np.linspace(0.01, 0.99, 1000)  # Threshold values to consider
        for scenario in chosen_scenarios:
            ind = scenarios_position[scenario]

            if cache is not None:
                cache_key = make_cache_key(
                    scenario_name,
                    split_number,
                    f"{scenario}_{chosen_scenarios}_c",
                )
            else:
                cache_key = None
            if cache_key is not None and cache_key in cache:
                c = cache[cache_key]
            else:
                # Find the best threshold value that minimizes the difference between mean responses and mean scores
                c = cs[
                    np.argmin(
                        [
                            np.mean(
                                (
                                    np.abs(
                                        (scores_train[:, ind] > c).mean(axis=1)
                                        - scores_train[:, ind].mean(axis=1)
                                    )
                                )
                            )
                            for c in cs
                        ]
                    )
                ]
                # Apply the threshold to train and test responses
                if cache_key is not None:
                    cache[cache_key] = c
                    dump_pickle(cache, cache["cache_path"])
            responses_train[:, ind] = (scores_train[:, ind] > c).astype(int)
            responses_test[:, ind] = (scores_test[:, ind] > c).astype(int)

        # Initialize a dictionary to hold real accuracies
        accs_true = compute_true_acc(
            scores_test,
            balance_weights,
            scenarios_position,
            chosen_scenarios,
            list(range(len(rows_to_hide))),
            rows_to_hide,
        )
        accs_true_dict[split_number] = accs_true

        if skip_irt:
            A, B, Theta, opt_lambds = None, None, None, None
        else:
            # Choosing D through validation
            val_ind = list(
                range(0, responses_train.shape[0], 5)
            )  # list(range(int(responses_train.shape[0]/3)))
            train_ind = [
                i for i in range(responses_train.shape[0]) if i not in val_ind
            ]

            errors = []  # Initialize a list to hold validation errors
            errors2 = []

            if cache is not None:
                cache_key = make_cache_key(scenario_name, split_number, "D")

            if cache_key is not None and cache_key in cache:
                ind_D, D, errors2 = cache[cache_key]
            else:
                print("\ni) choosing optimal D")
                for D in tqdm(Ds):
                    # Train IRT model for the current dimension (D)
                    model_name = f"models/{bench}/split-{split}{split_number}_D-{D}_scenario-{scenario_name}_val/"
                    # Load trained IRT model parameters
                    try:
                        A, B, Theta = load_irt_parameters(model_name)
                    except:
                        # Create IRT dataset for validation and train IRT models
                        dataset_name = f"data/{bench}/split-{split}{split_number}_scenario-{scenario_name}_val.jsonlines"
                        create_irt_dataset(
                            responses_train[train_ind], dataset_name
                        )
                        train_irt_model(
                            dataset_name, model_name, D, lr, epochs, device
                        )
                        A, B, Theta = load_irt_parameters(model_name)
                    # Determine seen and unseen items for validation
                    seen_items = list(range(0, responses_train.shape[1], 2))
                    unseen_items = list(range(1, responses_train.shape[1], 2))
                    # Estimate ability parameters for the validation set
                    print(" - fit. theta in the val set")
                    pool = mp.Pool(cpu)
                    thetas = pool.starmap(
                        estimate_ability_parameters,
                        [
                            (responses_train[val_ind][j], seen_items, A, B)
                            for j in range(len(val_ind))
                        ],
                    )
                    pool.close()
                    pool.join()

                    # Compute validation errors for each scenario and update the errors list (in the end, we give the same weight for all scenarios)
                    errors2.append([])
                    for scenario in chosen_scenarios:
                        ind = [
                            u
                            for u in unseen_items
                            if u in scenarios_position[scenario]
                        ]
                        errors2[-1].append(
                            np.mean(
                                [
                                    abs(
                                        (
                                            balance_weights
                                            * item_curve(thetas[j], A, B)
                                        )[0, ind].mean()
                                        - scores_train[val_ind][j, ind].mean()
                                    )
                                    for j in range(len(val_ind))
                                ]
                            )
                        )
                    errors.append(np.mean(errors2[-1]))
                    print(errors[-1])

                # Choose the simplest model (D) that is not far from the best model based on validation errors
                ind_D = np.argmax(np.array(errors) - np.min(errors) < 0.0025)
                D = Ds[ind_D]
                print("- opt D=", D, "errors=", errors, "\n")

                if cache_key is not None:
                    cache[cache_key] = ind_D, D, errors2
                    dump_pickle(cache, cache["cache_path"])

            # Choosing lambdas (For random G-PIRT)
            print("\nii) choosing optimal lambdas")

            opt_lambds = {
                "random_gpirt": {},
                "anchor_gpirt": {},
                "anchor-irt_gpirt": {},
                "adaptive_gpirt": {},
            }  # Initialize a dictionary to hold optimal lambda values

            vs = {}
            bs = {}
            for i, scenario in enumerate(chosen_scenarios):
                vs[scenario] = np.var(
                    scores_train[:, scenarios_position[scenario]], axis=1
                ).mean()
                bs[scenario] = np.mean(errors2[ind_D][i])

            for scenario in tqdm(chosen_scenarios):
                for key in opt_lambds.keys():
                    opt_lambds[key][scenario] = {}
                    for number_item in number_items:
                        if key == "random_gpirt":
                            opt_lambds[key][scenario][number_item] = get_lambda(
                                bs[scenario], vs[scenario] / number_item
                            )
                        else:
                            opt_lambds[key][scenario][number_item] = get_lambda(
                                bs[scenario], vs[scenario] / (4 * number_item)
                            )

            # Save the final dataset and train the final IRT model
            print("\niii) fitting final IRT model")

            model_name = f"models/{bench}/split-{split}{split_number}_D-validate_scenario-{scenario_name}/"
            # Load the final IRT model
            try:
                A, B, Theta = load_irt_parameters(model_name)
            except:
                dataset_name = f"data/{bench}/split-{split}{split_number}_scenario-{scenario_name}.jsonlines"
                create_irt_dataset(responses_train, dataset_name)
                train_irt_model(dataset_name, model_name, D, lr, epochs, device)
                A, B, Theta = load_irt_parameters(model_name)

        end_time_pre_sampling_stage = time.time()
        elapsed_time_pre_sampling_stage = (
            end_time_pre_sampling_stage - start_time_pre_sampling_stage
        )
        print(
            f"Time taken by pre-sampling stage: {elapsed_time_pre_sampling_stage} seconds"
        )

        print("\niv) sampling")
        start_time_sampling_stage = time.time()
        (
            item_weights_dic,
            seen_items_dic,
            unseen_items_dic,
            sampling_time_dic,
        ) = ({}, {}, {}, {})
        if cache is not None:
            cache_key = make_cache_key(scenario_name, split_number, "sampling")
        else:
            cache_key = None

        disagreement_scores_cache_path = (
            make_cache_subpath(
                cache, scenario_name, split_number, f"disagreement_scores_path"
            )
            if cache_key is not None
            else None
        )
        disagreement_scores_dict = make_or_load_from_cache(
            object_name="disagreement_scores_dict",
            object_config={
                "sampling_names": sampling_names,
                "predictions_train": predictions_train,
                "disagreement_type": disagreement_type,
                "vary_selection": vary_selection,
                "iterations": iterations,
            },
            make_func=make_disagreement_scores_dict,
            cache_path=disagreement_scores_cache_path,
        )

        for sampling_name in tqdm(sampling_names):
            if skip_irt and sampling_name == "anchor-irt":
                continue

            if "adaptive" in sampling_name:
                raise NotImplementedError
                # item_weights_dic[sampling_name] = {number_item: {n_model: [] for n_model in range(responses_test.shape[0])} for number_item in number_items}
                # seen_items_dic[sampling_name] = {number_item: {n_model: [] for n_model in range(responses_test.shape[0])} for number_item in number_items}
                # unseen_items_dic[sampling_name] = {number_item: {n_model: [] for n_model in range(responses_test.shape[0])} for number_item in number_items}
                # sampling_time_dic[sampling_name] = {number_item: [] for number_item in number_items}

                # inital_items = select_initial_adaptive_items(A, B, Theta, D+1, try_size=10000)

                # pool = mp.Pool(cpu)
                # # parallelising models
                # samples = pool.starmap(sample_items_adaptive, [(number_items, iterations, sampling_name, chosen_scenarios, scenarios,
                #                                                 subscenarios_position, responses, scores_train, scenarios_position,
                #                                                 A, B, balance_weights, disagreement_scores_dict, inital_items) for responses in responses_test])
                # pool.close()
                # pool.join()
                # for n_model, samples_model in enumerate(samples): #zip(rows_to_hide, samples):
                #     item_weights_model, seen_items_model, unseen_items_model, sampling_time = samples_model
                #     sampling_time = np.array(sampling_time).mean()
                #     for number_item in number_items:
                #         item_weights_dic[sampling_name][number_item][n_model] = item_weights_model[number_item]
                #         seen_items_dic[sampling_name][number_item][n_model] = seen_items_model[number_item]
                #         unseen_items_dic[sampling_name][number_item][n_model] = unseen_items_model[number_item]
                #         sampling_time_dic[sampling_name][number_item].append(sampling_time)
            else:
                if cache_key is not None and cache_key in cache:
                    # samples, item_weights_dic, seen_items_dic, unseen_items_dic, sampling_time_dic = cache[cache_key]
                    (
                        item_weights_dic,
                        seen_items_dic,
                        unseen_items_dic,
                        sampling_time_dic,
                    ) = cache[cache_key]
                else:
                    # parallelising number of items
                    pool = mp.Pool(cpu)
                    (
                        item_weights_dic[sampling_name],
                        seen_items_dic[sampling_name],
                        unseen_items_dic[sampling_name],
                        sampling_time_dic[sampling_name],
                    ) = ({}, {}, {}, {})
                    samples = pool.starmap(
                        sample_items,
                        [
                            (
                                number_item,
                                iterations,
                                sampling_name,
                                chosen_scenarios,
                                scenarios,
                                subscenarios_position,
                                responses_test,
                                scores_train,
                                predictions_train,
                                scenarios_position,
                                A,
                                B,
                                balance_weights,
                                disagreement_scores_dict,
                                skip_irt,
                            )
                            for number_item in number_items
                        ],
                    )
                    pool.close()
                    pool.join()

                    for i, number_item in enumerate(number_items):
                        (
                            item_weights_dic[sampling_name][number_item],
                            seen_items_dic[sampling_name][number_item],
                            unseen_items_dic[sampling_name][number_item],
                            sampling_time_dic[sampling_name][number_item],
                        ) = samples[i]

        if cache_key is not None:
            cache[cache_key] = (
                # samples,
                item_weights_dic,
                seen_items_dic,
                unseen_items_dic,
                sampling_time_dic,
            )
            dump_pickle(cache, cache["cache_path"])

        # saving points
        if (
            rows_to_hide == set_of_rows[0] and split == "iid"
        ):  # we impose rows_to_hide==set_of_rows[0] because in AlpacaEval we do K-fold CV so there are multiple training/test sets
            dic = {}
            dic["item_weights"] = item_weights_dic
            dic["seen_items"] = seen_items_dic
            dic["scenarios_position"] = scenarios_position
            dic["subscenarios_position"] = subscenarios_position
            dic["opt_lambds"] = opt_lambds
            dic["A"] = A
            dic["B"] = B
            with open(
                f"results/samples_{bench[4:]}_iterations-{iterations}.pickle",
                "wb",
            ) as handle:
                pickle.dump(dic, handle, protocol=pickle.HIGHEST_PROTOCOL)

        end_time_sampling_stage = time.time()
        elapsed_time_pre_sampling_stage = (
            end_time_sampling_stage - start_time_sampling_stage
        )
        print(
            f"Time taken by sampling stage: {elapsed_time_pre_sampling_stage} seconds"
        )

        print("\nv) computing accuracies")
        start_time = time.time()

        train_model_indices = list(range(scores_train.shape[0]))
        train_model_true_accs = compute_true_acc(
            scores_train,
            balance_weights,  # sample -> sample weight
            scenarios_position,  # scenario -> list of sample indices
            chosen_scenarios,
            train_model_indices,
            train_model_indices,  # they are not the global indices, but the contiguous indices of train models after removing test models
        )

        emb_cache_path = (
            make_cache_subpath(
                cache, scenario_name, split_number, f"embeddings_path"
            )
            if cache_key is not None
            else None
        )

        start_time_embedding_stage = time.time()
        (
            train_models_embeddings,
            test_models_embeddings,
        ) = make_or_load_from_cache(
            object_name="train_test_model_embeddings",
            object_config={
                "sampling_names": sampling_names,
                "number_items": number_items,
                "iterations": iterations,
                "predictions_train": predictions_train,
                "seen_items_dic": seen_items_dic,
                "predictions_test": predictions_test,
                "seen_items_dic": seen_items_dic,
                "pca": pca,
                "apply_softmax": apply_softmax_to_predictions,
            },
            make_func=make_train_test_model_embeddings,
            cache_path=emb_cache_path,
        )
        end_time_embedding_stage = time.time()
        elapsed_time_embedding_stage = (
            end_time_embedding_stage - start_time_embedding_stage
        )
        print(
            f"Time taken by embedding stage: {elapsed_time_embedding_stage} seconds"
        )

        start_time_fitted_weights_stage = time.time()
        fitted_weights_cache_path = (
            make_cache_subpath(
                cache, scenario_name, split_number, f"fitted_weights_path"
            )
            if cache_key is not None
            else None
        )

        fitted_weights = make_or_load_from_cache(
            object_name="fitted_weights",
            object_config={
                "sampling_names": sampling_names,
                "number_items": number_items,
                "iterations": iterations,
                "train_models_embeddings": train_models_embeddings,
                "train_model_true_accs": train_model_true_accs,
                "scenario": scenario,
                "cache_path": fitted_weights_cache_path,
                "chosen_fitting_methods": chosen_fitting_methods,
                "skip_iterations_when_fixed_sampling": skip_it_fixed_sampling,
            },
            make_func=make_fitted_weights,
            cache_path=fitted_weights_cache_path,
        )
        end_time_fitted_weights_stage = time.time()
        elapsed_time_fitted_weights_stage = (
            end_time_fitted_weights_stage - start_time_fitted_weights_stage
        )
        print(
            f"Time taken by fitted weights stage: {elapsed_time_fitted_weights_stage} seconds"
        )

        start_time_accuracies_stage = time.time()
        for j in tqdm(range(len(rows_to_hide))):
            out.append(
                calculate_accuracies(
                    j,
                    sampling_names,
                    item_weights_dic,
                    seen_items_dic,
                    unseen_items_dic,
                    A,
                    B,
                    scores_test,
                    scores_train,
                    train_model_true_accs,
                    accs_true,
                    fitted_weights,
                    responses_test,
                    train_models_embeddings,
                    test_models_embeddings,
                    scenarios_position,
                    chosen_scenarios,
                    balance_weights,
                    opt_lambds,
                    rows_to_hide,
                    skip_irt,
                    chosen_estimators,
                )
            )
        end_time_accuracies_stage = time.time()
        elapsed_time_accuracies_stage = (
            end_time_accuracies_stage - start_time_accuracies_stage
        )
        print(
            f"Time taken by accuracies stage: {elapsed_time_accuracies_stage} seconds"
        )
        elapsed_time = np.round(time.time() - start_time)
        print(f" - finished in {elapsed_time} seconds")

    ### Final results
    accs_hat = {}
    results = {}
    for item in out:
        key = list(item.keys())[0]
        accs_hat[key] = item[key]

    # Update results with the mean absolute difference for each approach
    for split_number, rows_to_hide in enumerate(set_of_rows):
        accs_true = accs_true_dict[split_number]
        for j in range(len(rows_to_hide)):
            results[rows_to_hide[j]] = {}
            for number_item in number_items:
                results[rows_to_hide[j]][number_item] = {}
                for sampling_name in sampling_names:
                    if skip_irt and sampling_name == "anchor-irt":
                        continue
                    # for estimators in ['naive', 'pirt', 'cirt', 'gpirt']:
                    for estimator in chosen_estimators:
                        if skip_irt and estimator in ["pirt", "cirt", "gpirt"]:
                            continue
                        method_name = make_method_name(sampling_name, estimator)
                        results[rows_to_hide[j]][number_item][method_name] = {}
                        for scenario in chosen_scenarios:
                            acc_hat = accs_hat[rows_to_hide[j]][number_item][
                                method_name
                            ][scenario]
                            acc_true = accs_true[rows_to_hide[j]][scenario]
                            results[rows_to_hide[j]][number_item][method_name][
                                scenario
                            ] = np.abs(np.array(acc_hat) - acc_true)

    return (
        results,
        accs_hat,
        sampling_time_dic,
    )  # Return the updated results dictionary


def compute_embedding(
    predictions, anchor_indices, pca, transform=None, apply_softmax=True
):
    # if logits are input
    preds_for_anchors = torch.Tensor(predictions)[:, anchor_indices, :]

    if apply_softmax:
        emb_unreduced = preds_for_anchors.softmax(dim=-1)
    else:
        emb_unreduced = preds_for_anchors

    emb_unreduced = emb_unreduced.reshape(emb_unreduced.shape[0], -1)
    # print(emb_unreduced.shape)

    if transform is not None:
        assert pca is not None
    if pca is not None:
        if transform is None:
            if pca < 0:
                transform = umap.UMAP(
                    n_components=(-pca), random_state=RANDOM_SEED
                ).fit(emb_unreduced.numpy())
            else:
                transform = PCA(
                    n_components=pca,
                    svd_solver="full",
                    random_state=RANDOM_SEED,
                ).fit(emb_unreduced.numpy())

        emb = transform.transform(emb_unreduced.numpy())
        emb = torch.Tensor(emb)

    else:
        emb = emb_unreduced
    return emb, transform


def make_train_test_model_embeddings(emb_config, logger=None):
    (
        sampling_names,
        number_items,
        iterations,
        predictions_train,
        seen_items_dic,
        predictions_test,
        seen_items_dic,
        pca,
        apply_softmax,
    ) = (
        emb_config["sampling_names"],
        emb_config["number_items"],
        emb_config["iterations"],
        emb_config["predictions_train"],
        emb_config["seen_items_dic"],
        emb_config["predictions_test"],
        emb_config["seen_items_dic"],
        emb_config["pca"],
        emb_config["apply_softmax"],
    )
    train_models_embeddings = {}
    test_models_embeddings = {}
    for sampling_name in tqdm(sampling_names, desc="Computing embeddings"):
        train_models_embeddings[sampling_name] = {}
        test_models_embeddings[sampling_name] = {}
        for number_item in number_items:
            train_models_embeddings[sampling_name][number_item] = {}
            test_models_embeddings[sampling_name][number_item] = {}
            for it in range(iterations):
                (
                    train_models_embeddings[sampling_name][number_item][it],
                    transform,
                ) = compute_embedding(
                    predictions_train,
                    seen_items_dic[sampling_name][number_item][it],
                    pca,
                    None,
                    apply_softmax=apply_softmax,
                )
                (
                    test_models_embeddings[sampling_name][number_item][it],
                    _,
                ) = compute_embedding(
                    predictions_test,
                    seen_items_dic[sampling_name][number_item][it],
                    pca,
                    transform,
                    apply_softmax=apply_softmax,
                )
    return train_models_embeddings, test_models_embeddings


def make_fitted_weights(config, logger=None):
    (
        sampling_names,
        number_items,
        iterations,
        train_models_embeddings,
        train_model_true_accs,
        scenario,
        cache_path,
        chosen_fitting_methods,
    ) = (
        config["sampling_names"],
        config["number_items"],
        config["iterations"],
        config["train_models_embeddings"],
        config["train_model_true_accs"],
        config["scenario"],
        config["cache_path"],
        config["chosen_fitting_methods"],
    )
    skip_iterations_when_fixed_sampling = config.get(
        "skip_iterations_when_fixed_sampling", True
    )

    fitted_weights = {}

    train_model_true_accs_np = np.array(
        [
            train_model_true_accs[i][scenario]
            for i in range(len(train_model_true_accs))
        ]
    )

    for sampling_name in sampling_names:
        fitted_weights[sampling_name] = {}
        for number_item in number_items:
            fitted_weights[sampling_name][number_item] = {}
            for it in range(iterations):
                cur_train_models_embeddings_np = train_models_embeddings[
                    sampling_name
                ][number_item][it].numpy()

                fitted_weights[sampling_name][number_item][it] = {}

                for model_name, builder in chosen_fitting_methods:
                    if cache_path is not None:
                        cache_dir = cache_path.split(".")[0] + "_folder"
                    else:
                        cache_dir = None
                    fitted_model = make_or_load_from_cache(
                        object_name=f"fitted_model_{model_name}_{sampling_name}_{number_item}_{it}",
                        object_config={
                            "builder": builder,
                            "sampling_name": sampling_name,
                            "number_item": number_item,
                            "it": it,
                            "cur_train_models_embeddings_np": cur_train_models_embeddings_np,
                            "train_model_true_accs_np": train_model_true_accs_np,
                            "fitted_weights": fitted_weights,
                            "model_name": model_name,
                            "skip_iterations_when_fixed_sampling": skip_iterations_when_fixed_sampling,
                        },
                        make_func=make_fitted_model,
                        cache_path=cache_dir,
                    )

                    fitted_weights[sampling_name][number_item][it][
                        f"{model_name}"
                    ] = fitted_model

    return fitted_weights


def make_cache_subpath(cache, scenario_name, split_number, suffix):
    if cache is not None:
        cache_dir = cache["cache_path"].split(".")[0] + "_folder"
        cache_key = make_cache_key(scenario_name, split_number, suffix)
        cache_path = os.path.join(cache_dir, cache_key + ".pkl")
    else:
        cache_path = None
    return cache_path


def make_fitted_model(config, logger=None):
    (
        builder,
        sampling_name,
        number_item,
        it,
        cur_train_models_embeddings_np,
        train_model_true_accs_np,
        fitted_weights,
        model_name,
    ) = (
        config["builder"],
        config["sampling_name"],
        config["number_item"],
        config["it"],
        config["cur_train_models_embeddings_np"],
        config["train_model_true_accs_np"],
        config["fitted_weights"],
        config["model_name"],
    )
    skip_iterations_when_fixed_sampling = config.get(
        "skip_iterations_when_fixed_sampling", True
    )
    builder_func, builder_kwargs = builder
    model = builder_func(**builder_kwargs)
    if (
        skip_iterations_when_fixed_sampling
        and sampling_name in ["high-disagreement", "low-disagreement"]
        and "@" not in sampling_name
        and it > 0
    ):
        # for deterministic sampling fitted model does not change for different runs
        fitted_model = deepcopy(
            fitted_weights[sampling_name][number_item][0][f"{model_name}"]
        )
    else:
        fitted_model = model.fit(
            cur_train_models_embeddings_np, train_model_true_accs_np
        )
        # when shape like (159, 102400), RandomForest200 can take 5+ min

        # Compute training RMSE
        train_preds = fitted_model.predict(cur_train_models_embeddings_np)
        train_rmse = np.sqrt(
            np.mean((train_preds - train_model_true_accs_np) ** 2)
        )

        print(
            f"Training RMSE of {model_name} for samplng={sampling_name}, n_anchor={number_item}, run={it}: {train_rmse}"
        )

    return fitted_model


def make_disagreement_scores_dict(config, logger=None):
    (
        sampling_names,
        predictions_train,
        disagreement_type,
        vary_selection,
        iterations,
    ) = (
        config["sampling_names"],
        config["predictions_train"],
        config["disagreement_type"],
        config["vary_selection"],
        config["iterations"],
    )
    disagreement_scores_dict = {}
    for sampling_name in sampling_names:
        if not "disagreement" in sampling_name:
            continue
        disagreement_key = sampling_name.split("-")[1]
        if "@" in disagreement_key:
            n_guiding_models = int(sampling_name.split("@")[1].split("+")[0])
        else:
            n_guiding_models = None
        if not disagreement_key in disagreement_scores_dict:
            if vary_selection:
                disagreement_scores_dict[disagreement_key] = {}
                for it in range(iterations):
                    disagreement_scores_dict[disagreement_key][
                        it
                    ] = get_disagreement_scores(
                        predictions_train,
                        n_guiding_models,
                        disagreement_type=disagreement_type,
                    )
            else:
                disagreement_scores_dict[
                    disagreement_key
                ] = get_disagreement_scores(
                    predictions_train,
                    n_guiding_models,
                    disagreement_type=disagreement_type,
                )
    return disagreement_scores_dict
