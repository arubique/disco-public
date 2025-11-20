# The code is adapted from the tinyBenchmarks repo: https://github.com/felipemaiapolo/efficbench
from irt import estimate_ability_parameters

# from irt import *
# from utils import *
from utils import (
    item_curve,
    prepare_data,
    prepare_and_split_data,
    create_predictions,
    create_responses,
)
import torch
import torch.nn.functional as F
from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso,
)
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from plots import CONSTANT_ESTIMATORS
import numpy as np
from models import MLPRegressor

BASE_ESTIMATORS = [
    "naive",
    "pirt",
    "cirt",
    "gpirt",
    "mean_train_score",  # [ADD][new estimator]
    "perfect_knn",  # [ADD][new estimator]
    "KNN",  # [ADD][new estimator]
]

BEST_MLP_FITTING_METHODS = [
    (
        "MLP3_e700_lr0.001",
        (
            MLPRegressor,
            {
                "hidden_channels": [128, 128, 1],
                "n_epochs": 700,
                "lr": 0.001,
            },
        ),
    ),
]
MLP_FITTING_METHODS = [
    # MLP 2 layers
    (
        "MLP2_e50_lr0.001",
        (
            MLPRegressor,
            {
                "hidden_channels": [128, 1],
                "n_epochs": 50,
                "lr": 0.001,
            },
        ),
    ),
    (
        "MLP2_e100_lr0.001",
        (
            MLPRegressor,
            {
                "hidden_channels": [128, 1],
                "n_epochs": 100,
                "lr": 0.001,
            },
        ),
    ),
    (
        "MLP2_e200_lr0.001",
        (
            MLPRegressor,
            {
                "hidden_channels": [128, 1],
                "n_epochs": 200,
                "lr": 0.001,
            },
        ),
    ),
    (
        "MLP2_e100_lr0.01",
        (
            MLPRegressor,
            {
                "hidden_channels": [128, 1],
                "n_epochs": 100,
                "lr": 0.01,
            },
        ),
    ),
    (
        "MLP2_e100_lr0.0001",
        (
            MLPRegressor,
            {
                "hidden_channels": [128, 1],
                "n_epochs": 100,
                "lr": 0.0001,
            },
        ),
    ),
    # MLP 3 layers
    (
        "MLP3_e50_lr0.001",
        (
            MLPRegressor,
            {
                "hidden_channels": [128, 128, 1],
                "n_epochs": 50,
                "lr": 0.001,
            },
        ),
    ),
    (
        "MLP3_e100_lr0.001",
        (
            MLPRegressor,
            {
                "hidden_channels": [128, 128, 1],
                "n_epochs": 100,
                "lr": 0.001,
            },
        ),
    ),
    (
        "MLP3_e200_lr0.001",
        (
            MLPRegressor,
            {
                "hidden_channels": [128, 128, 1],
                "n_epochs": 200,
                "lr": 0.001,
            },
        ),
    ),
    (
        "MLP3_e200_lr0.0001",
        (
            MLPRegressor,
            {
                "hidden_channels": [128, 128, 1],
                "n_epochs": 200,
                "lr": 0.0001,
            },
        ),
    ),
    (
        "MLP3_e300_lr0.001",
        (
            MLPRegressor,
            {
                "hidden_channels": [128, 128, 1],
                "n_epochs": 300,
                "lr": 0.001,
            },
        ),
    ),
    (
        "MLP3_e400_lr0.001",
        (
            MLPRegressor,
            {
                "hidden_channels": [128, 128, 1],
                "n_epochs": 400,
                "lr": 0.001,
            },
        ),
    ),
    (
        "MLP3_e500_lr0.001",
        (
            MLPRegressor,
            {
                "hidden_channels": [128, 128, 1],
                "n_epochs": 500,
                "lr": 0.001,
            },
        ),
    ),
    (
        "MLP3_e500_lr0.0001",
        (
            MLPRegressor,
            {
                "hidden_channels": [128, 128, 1],
                "n_epochs": 400,
                "lr": 0.0001,
            },
        ),
    ),
    (
        "MLP3_e600_lr0.001",
        (
            MLPRegressor,
            {
                "hidden_channels": [128, 128, 1],
                "n_epochs": 600,
                "lr": 0.001,
            },
        ),
    ),
    (
        "MLP3_e600_lr0.0001",
        (
            MLPRegressor,
            {
                "hidden_channels": [128, 128, 1],
                "n_epochs": 600,
                "lr": 0.0001,
            },
        ),
    ),
    # ('MLP3_e700_lr0.001', (
    #     MLPRegressor, {
    #         'hidden_channels': [128, 128, 1],
    #         'n_epochs': 700,
    #         'lr': 0.001,
    #     }
    # )),
    (
        "MLP3_e800_lr0.001",
        (
            MLPRegressor,
            {
                "hidden_channels": [128, 128, 1],
                "n_epochs": 800,
                "lr": 0.001,
            },
        ),
    ),
    (
        "MLP3_e100_lr0.01",
        (
            MLPRegressor,
            {
                "hidden_channels": [128, 128, 1],
                "n_epochs": 100,
                "lr": 0.01,
            },
        ),
    ),
    (
        "MLP3_e100_lr0.0001",
        (
            MLPRegressor,
            {
                "hidden_channels": [128, 128, 1],
                "n_epochs": 100,
                "lr": 0.0001,
            },
        ),
    ),
]
MLP_FITTING_METHODS = BEST_MLP_FITTING_METHODS + MLP_FITTING_METHODS

BEST_LINEAR_METHODS = [
    ("Ridge_10", (Ridge, {"alpha": 10})),
    ("Lasso_e-4", (Lasso, {"alpha": 0.0001})),
    (
        "RandomForestRegressor_100",
        (RandomForestRegressor, {"n_estimators": 100}),
    ),
    # ('GradientBoostingRegressor_100', (GradientBoostingRegressor, {'n_estimators': 100})),
    (
        "GradientBoostingRegressor_200",
        (GradientBoostingRegressor, {"n_estimators": 200}),
    ),
]
BEST_FITTING_METHODS = BEST_MLP_FITTING_METHODS + BEST_LINEAR_METHODS

FITTING_METHODS = [
    ("LinearRegression", (LinearRegression, {})),
    ("Ridge_01", (Ridge, {"alpha": 0.1})),
    ("Ridge_1", (Ridge, {"alpha": 1})),
    # ('Ridge_10', (Ridge, {'alpha': 10})), # best ridge
    ("Ridge_100", (Ridge, {"alpha": 100})),
    ("Ridge_1000", (Ridge, {"alpha": 1000})),
    ("Lasso_5e-6", (Lasso, {"alpha": 0.000005})),
    ("Lasso_e-5", (Lasso, {"alpha": 0.00001})),
    # ('Lasso_e-4', (Lasso, {'alpha': 0.0001})), # best lasso
    ("Lasso_e-3", (Lasso, {"alpha": 0.001})),
    ("Lasso_e-2", (Lasso, {"alpha": 0.01})),
    ("RandomForestRegressor_50", (RandomForestRegressor, {"n_estimators": 50})),
    # ('RandomForestRegressor_100', (RandomForestRegressor, {'n_estimators': 100})), # best random forest
    (
        "RandomForestRegressor_200",
        (RandomForestRegressor, {"n_estimators": 200}),
    ),
    (
        "GradientBoostingRegressor_50",
        (GradientBoostingRegressor, {"n_estimators": 50}),
    ),
    # ('GradientBoostingRegressor_100', (GradientBoostingRegressor, {'n_estimators': 100})),
    (
        "GradientBoostingRegressor_100",
        (GradientBoostingRegressor, {"n_estimators": 100}),
    ),
    # ('GradientBoostingRegressor_200', (GradientBoostingRegressor, {'n_estimators': 200})),
]
FITTING_METHODS = BEST_LINEAR_METHODS + MLP_FITTING_METHODS + FITTING_METHODS


# [ADD][new estimator]
ESTIMATORS = []
for model_name, builder in FITTING_METHODS:
    # ESTIMATORS.append(f"fitted-{model_name}")
    ESTIMATORS.append(model_name)
ESTIMATORS = BASE_ESTIMATORS + ESTIMATORS
assert len(ESTIMATORS) == len(set(ESTIMATORS)), "Estimators are not unique"


def find_duplicates(lst):
    # Find and print any duplicate estimators
    seen = set()
    duplicates = []
    for est in lst:
        if est in seen:
            duplicates.append(est)
        else:
            seen.add(est)
    if duplicates:
        print(f"Found duplicate estimators: {duplicates}")


def compute_acc_pirt(
    data_part,
    scenario,
    scenarios_position,
    seen_items,
    unseen_items,
    A,
    B,
    theta,
    balance_weights,
    thresh=None,
):
    """
    Compute the PIRT or CIRT

    Parameters:
    - scenario: The scenario being considered.
    - scenarios_position: A dictionary mapping each scenario to the positions of its items.
    - seen_items: A list of item indices that the subject has been exposed to.
    - unseen_items: A list of item indices that the subject has not been exposed to.
    - A: The discrimination parameter of the item.
    - B: The difficulty parameter of the item.
    - theta: The ability parameter of the subject.
    - balance_weights: balancing weights (mmlu/civil comments).
    - thresh: classification threshold for CIRT (if None, PIRT will be computed).

    Returns:
    - The computed accuracy for the scenario.
    """

    # Determine the weighting parameter
    lambd = len(
        [s for s in seen_items if s in scenarios_position[scenario]]
    ) / len(scenarios_position[scenario])

    # Compute the second part of the accuracy equation based on unseen items (and IRT model)
    D = A.shape[1]  # The number of dimensions in the IRT model
    if thresh == None:
        irt_part = (balance_weights * item_curve(theta.reshape(1, D, 1), A, B))[
            0, [u for u in unseen_items if u in scenarios_position[scenario]]
        ].mean()
    else:
        irt_part = (
            balance_weights
            * (item_curve(theta.reshape(1, D, 1), A, B) >= thresh).astype(float)
        )[
            0, [u for u in unseen_items if u in scenarios_position[scenario]]
        ].mean()

    return lambd * data_part + (1 - lambd) * irt_part


def make_method_name(sampling_name, est):
    if est in CONSTANT_ESTIMATORS:
        return est
    else:
        return sampling_name + "_" + est


def calculate_accuracies(
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
    test_model_true_accs,
    fitted_weights,
    responses_test,
    train_models_embeddings,
    test_models_embeddings,
    scenarios_position,
    chosen_scenarios,
    balance_weights,
    opt_lambds,
    rows_to_hide,
    skip_irt=False,
    chosen_estimators=None,
):
    # number_items = list(item_weights_dic[sampling_names[0]].keys())
    key_from_item_weights_dic = list(item_weights_dic.keys())[0]
    number_items = list(item_weights_dic[key_from_item_weights_dic].keys())

    # shape of scores_train is (n_models, n_qestions)
    mean_train_score = scores_train.mean()

    # Creating output format
    accs = {rows_to_hide[j]: {}}
    for number_item in number_items:
        accs[rows_to_hide[j]][number_item] = {}
        for est in chosen_estimators:
            if skip_irt and est in ["pirt", "cirt", "gpirt"]:
                continue
            for sampling_name in sampling_names:
                if skip_irt and sampling_name == "anchor-irt":
                    continue
                accs[rows_to_hide[j]][number_item][
                    make_method_name(sampling_name, est)
                ] = {}
                for scenario in chosen_scenarios:
                    accs[rows_to_hide[j]][number_item][
                        make_method_name(sampling_name, est)
                    ][scenario] = []

    # Populating output
    for sampling_name in sampling_names:
        if skip_irt and sampling_name == "anchor-irt":
            continue

        for number_item in number_items:
            if "adaptive" in sampling_name:
                raise NotImplementedError
                # # Getting sample for specific model j
                # item_weights, seen_items, unseen_items = item_weights_dic[sampling_name][number_item][j], seen_items_dic[sampling_name][number_item][j], unseen_items_dic[sampling_name][number_item][j]

                # # Estimate ability parameters for the test set (IRT)
                # new_theta = estimate_ability_parameters(responses_test[j], seen_items, A, B)

                # # Update accuracies
                # for scenario in chosen_scenarios:
                #     naive = (item_weights[scenario]*scores_test[j][[s for s in seen_items if s in scenarios_position[scenario]]]).sum()
                #     data_part_pirt = ((balance_weights*scores_test[j])[[s for s in seen_items if s in scenarios_position[scenario]]]).mean()
                #     pirt = compute_acc_pirt(data_part_pirt, scenario, scenarios_position, seen_items, unseen_items, A, B, new_theta, balance_weights, thresh=None)
                #     cirt = compute_acc_pirt(data_part_pirt, scenario, scenarios_position, seen_items, unseen_items, A, B, new_theta, balance_weights, thresh=0.5)
                #     lambd = opt_lambds[sampling_name+'_gpirt'][scenario][number_item]

                #     accs[rows_to_hide[j]][number_item][sampling_name+'_naive'][scenario].append(naive)
                #     accs[rows_to_hide[j]][number_item][sampling_name+'_pirt'][scenario].append(pirt)
                #     accs[rows_to_hide[j]][number_item][sampling_name+'_cirt'][scenario].append(cirt)
                #     accs[rows_to_hide[j]][number_item][sampling_name+'_gpirt'][scenario].append(lambd*naive + (1-lambd)*pirt)
            else:
                iterations = len(
                    list(
                        item_weights_dic[sampling_name][number_items[0]].keys()
                    )
                )
                for it in range(iterations):
                    # print(sampling_name, number_item, it)
                    # Getting sample
                    try:
                        item_weights, seen_items, unseen_items = (
                            item_weights_dic[sampling_name][number_item][it],
                            seen_items_dic[sampling_name][number_item][it],
                            unseen_items_dic[sampling_name][number_item][it],
                        )
                    except:
                        breakpoint()
                    if not skip_irt:
                        # Estimate ability parameters for the test set (IRT)
                        new_theta = estimate_ability_parameters(
                            responses_test[j], seen_items, A, B
                        )

                    # Update accuracies
                    for scenario in chosen_scenarios:
                        # current gt performance: test_model_true_accs[rows_to_hide[j]][scenario]
                        selected_indices = [
                            s
                            for s in seen_items
                            if s in scenarios_position[scenario]
                        ]
                        assert len(selected_indices) > 0, "No items in scenario"

                        if (
                            "naive" in chosen_estimators
                            or "gpirt" in chosen_estimators
                            or "pirt" in chosen_estimators
                        ):
                            naive = (
                                item_weights[scenario]
                                * scores_test[j][selected_indices]
                            ).sum()
                            accs[rows_to_hide[j]][number_item][
                                sampling_name + "_naive"
                            ][scenario].append(naive)

                        if test_models_embeddings is None:
                            pass
                        else:
                            # [ADD][new estimator]
                            if (
                                len(
                                    accs[rows_to_hide[j]][number_item][
                                        "mean_train_score"
                                    ][scenario]
                                )
                                < iterations
                            ):
                                accs[rows_to_hide[j]][number_item][
                                    "mean_train_score"
                                ][scenario].append(
                                    mean_train_score
                                )  # does not depend on sampling type

                            test_model_embedding = test_models_embeddings[
                                sampling_name
                            ][number_item][it][j]

                            if "KNN" in chosen_estimators:
                                # [ADD][new estimator] KNN
                                knn_acc = compute_acc_knn(
                                    test_model_embedding=test_model_embedding,
                                    train_model_embeddings=train_models_embeddings[
                                        sampling_name
                                    ][
                                        number_item
                                    ][
                                        it
                                    ],
                                    scenario=scenario,
                                    train_model_true_accs=train_model_true_accs,
                                )
                                accs[rows_to_hide[j]][number_item][
                                    sampling_name + "_KNN"
                                ][scenario].append(knn_acc)

                            # [ADD][new estimator] fitted weights
                            if (
                                len(
                                    accs[rows_to_hide[j]][number_item][
                                        "perfect_knn"
                                    ][scenario]
                                )
                                < iterations
                            ):
                                perfect_knn_acc = compute_perfect_knn(
                                    train_model_true_accs,
                                    test_model_true_accs[rows_to_hide[j]],
                                    scenario,
                                )
                                accs[rows_to_hide[j]][number_item][
                                    "perfect_knn"
                                ][scenario].append(
                                    perfect_knn_acc
                                )  # does not depend on sampling type

                            # [ADD][new estimator] fitted weights
                            for model_key, fitted_model in fitted_weights[
                                sampling_name
                            ][number_item][it].items():
                                fitted_acc = fitted_model.predict(
                                    test_model_embedding.numpy().reshape(1, -1)
                                )[0]
                                accs[rows_to_hide[j]][number_item][
                                    sampling_name + f"_{model_key}"
                                ][scenario].append(fitted_acc)

                        if not skip_irt:
                            if (
                                "pirt" in chosen_estimators
                                or "cirt" in chosen_estimators
                                or "gpirt" in chosen_estimators
                            ):
                                data_part_pirt = (
                                    (balance_weights * scores_test[j])[
                                        [
                                            s
                                            for s in seen_items
                                            if s in scenarios_position[scenario]
                                        ]
                                    ]
                                ).mean()
                                pirt = compute_acc_pirt(
                                    data_part_pirt,
                                    scenario,
                                    scenarios_position,
                                    seen_items,
                                    unseen_items,
                                    A,
                                    B,
                                    new_theta,
                                    balance_weights,
                                    thresh=None,
                                )
                                cirt = compute_acc_pirt(
                                    data_part_pirt,
                                    scenario,
                                    scenarios_position,
                                    seen_items,
                                    unseen_items,
                                    A,
                                    B,
                                    new_theta,
                                    balance_weights,
                                    thresh=0.5,
                                )

                                lambda_key = sampling_name + "_gpirt"

                                if lambda_key not in opt_lambds:
                                    lambda_key = "anchor-irt_gpirt"

                                lambd = opt_lambds[lambda_key][scenario][
                                    number_item
                                ]

                                accs[rows_to_hide[j]][number_item][
                                    sampling_name + "_pirt"
                                ][scenario].append(pirt)
                                accs[rows_to_hide[j]][number_item][
                                    sampling_name + "_cirt"
                                ][scenario].append(cirt)
                            if "gpirt" in chosen_estimators:
                                accs[rows_to_hide[j]][number_item][
                                    sampling_name + "_gpirt"
                                ][scenario].append(
                                    lambd * naive + (1 - lambd) * pirt
                                )

    # Return output
    return accs


def compute_acc_knn(
    test_model_embedding,
    train_model_embeddings,
    scenario,
    train_model_true_accs,
    k=1,
):
    """
    Compute the accuracy of the KNN estimator.
    """

    similarities = F.cosine_similarity(
        test_model_embedding, train_model_embeddings, dim=1
    )

    # Get index of most similar embedding
    # most_similar_idx = torch.argmax(similarities).item()
    most_similar_idx = torch.topk(similarities, k).indices

    # Return accuracy of most similar example
    # return train_model_true_accs[most_similar_idx][scenario]
    accs = []
    for idx in most_similar_idx:
        accs.append(train_model_true_accs[idx.item()][scenario])
    return np.array(accs).mean()


def compute_perfect_knn(train_model_true_accs, test_model_true_acc, scenario):
    """
    Compute the accuracy of the perfect KNN estimator.
    """
    train_model_true_accs_np = convert_accs_to_numpy(
        train_model_true_accs, scenario
    )
    closest_index = np.argmin(
        np.abs(train_model_true_accs_np - test_model_true_acc[scenario])
    )
    return train_model_true_accs[closest_index][scenario]


def convert_accs_to_numpy(accs, scenario):
    """
    Convert the accuracies to a numpy array.
    """
    return np.array([accs[i][scenario] for i in accs.keys()])


def compute_true_acc(
    scores,
    balance_weights,
    scenarios_position,
    chosen_scenarios,
    model_indices,
    model_keys_dict,
):
    accs_true = {}
    for j in model_indices:
        accs_true[model_keys_dict[j]] = {}
        for scenario in chosen_scenarios:
            accs_true[model_keys_dict[j]][scenario] = (
                (balance_weights[None, :] * scores)[
                    j, scenarios_position[scenario]
                ]
            ).mean()
    return accs_true
