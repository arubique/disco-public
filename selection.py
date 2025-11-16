# The code is adapted from the tinyBenchmarks repo: https://github.com/felipemaiapolo/efficbench
import numpy as np
import random
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import pairwise_distances
from scipy.integrate import nquad, quad
from scipy.stats import norm
from tqdm import tqdm
import time
import torch
import torch.nn.functional as F

# from irt import *
from irt import estimate_ability_parameters

# from utils import *
from utils import (
    item_curve,
    # prepare_data,
    # prepare_and_split_data,
    # create_predictions,
    # create_responses,
)

# from irt import get_irt_params, get_irt_weights
# from utils import prepare_data, prepare_and_split_data, create_predictions, create_responses
from copy import copy


def get_random(
    scenarios_choosen,
    scenarios,
    number_item,
    subscenarios_position,
    responses_test,
    balance_weights,
    random_seed,
):
    """
    Stratified sample items (seen_items). 'unseen_intems' gives the complement.

    Parameters:
    - scenarios_choosen: A list of considered scenarios.
    - scenarios: A dictionary where keys are scenario identifiers and values are lists of subscenarios.
    - number_item: The total number of items to be considered across all chosen scenarios.
    - subscenarios_position: A nested dictionary where the first key is the scenario and the second key is the subscenario,
      and the value is a list of item positions for that subscenario.
    - responses_test: A numpy array of the test subject's responses to all items. (this is only used to get the number of items)

    Returns:
    - seen_items: A list of item indices that the subject has been exposed to.
    - unseen_items: A list of item indices that the subject has not been exposed to.
    """

    random.seed(random_seed)

    def shuffle_list(lista):
        """
        Shuffles a list in place and returns the shuffled list.

        Parameters:
        - lista: The list to be shuffled.

        Returns:
        - A shuffled version of the input list.
        """
        return random.sample(lista, len(lista))

    seen_items = (
        []
    )  # Initialize an empty list to hold the indices of seen items.
    item_weights = {}

    # Iterate through each chosen scenario to determine the seen items.
    for scenario in scenarios_choosen:
        seen_items_scenario = []

        # Allocate the number of items to be seen in each subscenario.
        # (uniformly spread num items to sample across subscenarios)
        number_items_sub = np.zeros(len(scenarios[scenario])).astype(int)
        number_items_sub += number_item // len(scenarios[scenario])
        number_items_sub[: (number_item - number_items_sub.sum())] += 1

        i = 0  # Initialize a counter for the subscenarios.
        # Shuffle the subscenarios and iterate through them to select seen items.
        for sub in shuffle_list(scenarios[scenario]):
            # Randomly select items from the subscenario and add them to the seen items.
            seen_items_scenario += random.sample(
                subscenarios_position[scenario][sub], k=number_items_sub[i]
            )
            i += 1

        item_weights[scenario] = np.ones(number_item) / number_item

        seen_items += seen_items_scenario

    # Determine the unseen items by finding all item indices that are not in the seen items list.
    unseen_items = [
        i for i in range(responses_test.shape[1]) if i not in seen_items
    ]

    return item_weights, seen_items, unseen_items


def stratified_num_items(number_item, scenarios):
    number_items_sub = np.zeros(len(scenarios)).astype(int)
    number_items_sub += number_item // len(scenarios)
    number_items_sub[: (number_item - number_items_sub.sum())] += 1
    return number_items_sub


def are_probs(logits):
    if logits.min() >= 0 and logits.max() <= 1:
        return True
    return False


def get_probs(logits):
    if are_probs(logits):
        probs = logits
    else:
        probs = F.softmax(logits, dim=-1)
    return probs


def pds(logits):
    # input shape: (num_models, num_samples, num_classes)
    # output shape: (num_samples)

    probs = get_probs(logits).clone()

    max_by_model = probs.max(0).values
    sum_over_classes = max_by_model.sum(-1)

    return torch.Tensor(sum_over_classes)


def jsd(logits):
    # input shape: (num_models, num_samples, num_classes)
    # output shape: (num_samples)

    probs = get_probs(logits).clone()

    # Ensure numerical stability and proper normalization
    eps = 1e-12
    probs = probs.clamp(min=eps)
    probs = probs / probs.sum(dim=-1, keepdim=True)

    # Mean distribution across models per sample
    mean_probs = probs.mean(dim=0)
    mean_probs = mean_probs.clamp(min=eps)
    mean_probs = mean_probs / mean_probs.sum(dim=-1, keepdim=True)

    # KL divergence per model and sample: KL(p || m) (summation over classes)
    kl_per_model = (probs * (probs.log() - mean_probs.log())).sum(dim=-1)

    # Jensen-Shannon divergence is the average KL across models
    jsd_per_sample = kl_per_model.mean(dim=0)

    return jsd_per_sample


def get_disagreement_scores(
    predictions_train, n_guiding_models, disagreement_type="pds"
):
    if n_guiding_models is not None:
        guiding_models_indices = np.random.choice(
            predictions_train.shape[0], n_guiding_models, replace=False
        )
        predictions_train = predictions_train[guiding_models_indices, ...]
    if disagreement_type == "pds":
        scores_per_sample = pds(torch.Tensor(predictions_train))
    elif disagreement_type == "jsd":
        scores_per_sample = jsd(torch.Tensor(predictions_train))
    else:
        raise ValueError(f"Disagreement type {disagreement_type} not supported")
    return scores_per_sample.numpy()


def sample_by_disagreement(
    sampling_name,
    scenarios_choosen,
    scenarios,
    number_item,
    subscenarios_position,
    num_samples_in_test,
    predictions_train,
    balance_weights,
    disagreement_scores_dict,
    random_seed,
    high_first=False,
    # n_guiding_models=None
):
    """
    Select items with high disagreement between models.
    """

    seen_items = (
        []
    )  # Initialize an empty list to hold the indices of seen items.
    item_weights = {}

    disagreement_key = sampling_name.split("-")[1]
    disagreement_scores = disagreement_scores_dict[disagreement_key]
    if "+nonstratified" == sampling_name[-14:]:
        stratified = False
    else:
        stratified = True

    if stratified:
        # Iterate through each chosen scenario to determine the seen items.
        for scenario in scenarios_choosen:
            seen_items_scenario = []

            number_items_sub = stratified_num_items(
                number_item, scenarios[scenario]
            )

            # Note: scenario - e.g. mmlu
            # subscenario - e.g. harness_hendrycksTest_sociology_5
            # subscenarios_position - samples' indices that correspond to each subscenario

            i = 0
            for sub in scenarios[scenario]:
                sorted_by_disagreement = sorted(
                    subscenarios_position[scenario][sub],
                    key=lambda x: disagreement_scores[x],
                    reverse=True,
                )

                if high_first:
                    top_by_disagreement = sorted_by_disagreement[
                        : number_items_sub[i]
                    ]
                else:
                    if number_items_sub[i] > 0:
                        top_by_disagreement = sorted_by_disagreement[
                            -number_items_sub[i] :
                        ]
                    else:
                        top_by_disagreement = []
                seen_items_scenario += top_by_disagreement
                i += 1

            # item_weights[scenario] = np.ones(number_item)/number_item

            seen_items += seen_items_scenario
    else:
        sorted_by_disagreement = sorted(
            list(range(len(disagreement_scores))),
            key=lambda x: disagreement_scores[x],
            reverse=True,
        )
        if high_first:
            seen_items = sorted_by_disagreement[:number_item]
        else:
            if number_item > 0:
                seen_items = sorted_by_disagreement[-number_item:]
            else:
                seen_items = []
    for scenario in scenarios_choosen:
        item_weights[scenario] = np.ones(number_item) / number_item

    unseen_items = [
        i for i in range(num_samples_in_test) if i not in seen_items
    ]

    return item_weights, seen_items, unseen_items


def select_initial_adaptive_items(
    A, B, Theta, number_item, try_size=2000, seed=42
):
    random.seed(seed)
    mats = np.stack(
        [np.outer(A[0, :, i], A[0, :, i]) for i in range(A.shape[2])]
    )
    samples = [
        random.sample(range(A.shape[-1]), number_item) for _ in range(try_size)
    ]
    samples_infos = np.stack(
        [
            np.linalg.det(
                np.array(
                    [
                        (p * (1 - p))[:, None, None] * mats[s]
                        for p in item_curve(Theta, A[:, :, s], B[:, :, s])
                    ]
                ).sum(axis=1)
            ).sum()
            for s in samples
        ]
    )
    seen_items = samples[np.argmax(samples_infos)]
    unseen_items = [i for i in range(A.shape[-1]) if i not in seen_items]
    return seen_items, unseen_items, mats


def run_adaptive_selection(
    responses_test,
    initial_items,
    scenarios_choosen,
    scenarios_position,
    A,
    B,
    num_items,
    balance_weights,
    balance=True,
):
    seen_items, all_items, mats = initial_items

    num_items_count = [len(scenarios_choosen) * n for n in num_items]
    max_count, min_count = max(num_items_count), min(num_items_count)
    item_weights, all_seen_items, all_unseen_items = {}, {}, {}

    if (min_count / 3) < len(seen_items):
        seen_items = seen_items[: int(min_count / 3)]

    unseen_items = [i for i in all_items if i not in seen_items]

    # assert len(seen_items) <= target_count
    count = len(seen_items)

    scenario_occurrences = {scenario: 0 for scenario in scenarios_choosen}

    for item in seen_items:
        scenario_of_item = find_scenario_from_position(scenarios_position, item)
        scenario_occurrences[scenario_of_item] += 1

    while True:
        for scenario in scenarios_choosen:
            if count in num_items_count:
                # save intermediate num_items
                current_num_items = int(count / len(scenarios_choosen))

                # item_weights[current_num_items] = get_weighing_adaptive(seen_items, unseen_items, scenarios_position, scenarios_choosen, A, B, balance_weights)
                item_weights[current_num_items] = {
                    scenario: np.array(
                        [occurrences / (count**2)] * occurrences
                    )
                    for scenario, occurrences in scenario_occurrences.items()
                }
                all_seen_items[current_num_items] = copy(seen_items)
                all_unseen_items[current_num_items] = copy(unseen_items)

            if count >= max_count:
                # return if largest num_items is reached
                return item_weights, all_seen_items, all_unseen_items

            (
                seen_items,
                unseen_items,
                scenario_of_item,
            ) = select_next_adaptive_item(
                responses_test,
                seen_items,
                unseen_items,
                scenario,
                scenarios_position,
                A,
                B,
                mats,
                balance,
            )

            scenario_occurrences[scenario_of_item] += 1
            count += 1


def select_next_adaptive_item(
    responses_test,
    seen_items,
    unseen_items,
    scenario,
    scenarios_position,
    A,
    B,
    mats,
    balance,
):
    D = A.shape[1]

    if balance:
        unseen_items_scenario = [
            u for u in unseen_items if u in scenarios_position[scenario]
        ]
    else:
        unseen_items_scenario = unseen_items

    optimal_theta = estimate_ability_parameters(
        responses_test, seen_items, A, B
    )
    P = item_curve(optimal_theta, A, B).squeeze()

    # Compute information matrices for seen and unseen items
    I_seen = ((P * (1 - P))[:, None, None] * mats)[seen_items].sum(axis=0)
    I_unseen = ((P * (1 - P))[:, None, None] * mats)[unseen_items_scenario]

    # Select the next item based on the maximum determinant of information
    next_item = unseen_items_scenario[
        np.argmax(np.linalg.det(I_seen[None, :, :] + I_unseen))
    ]
    seen_items.append(next_item)
    unseen_items.remove(next_item)
    scenario_item = find_scenario_from_position(scenarios_position, next_item)

    return seen_items, unseen_items, scenario_item


def find_scenario_from_position(scenarios_position, position):
    for scenario, positions in scenarios_position.items():
        if position in positions:
            return scenario
    return None


def get_weighing_adaptive(
    seen_items: list,
    unseen_items: list,
    scenarios_position: list,
    chosen_scenarios: list,
    A: np.array,
    B: np.array,
    balance_weights: np.array,
) -> list:
    """
    Gets weights for unseen items based on proximity in IRT parameter space

    """
    by_scenario = True
    weights = {}

    IRT_params = np.concatenate((A, B), axis=1)

    if by_scenario:
        for i, scenario in enumerate(chosen_scenarios):
            # select indices of current scenario
            scenario_idxs = scenarios_position[scenario]

            scenario_seen_items = [
                s for s in seen_items if s in scenarios_position[scenario]
            ]

            weights[scenario] = get_weights(
                IRT_params,
                scenario_seen_items,
                scenario_idxs,
                balance_weights,
                scenario,
                scenarios_position,
            )
    else:
        weights["all"] = get_weights(IRT_params, seen_items, unseen_items)

    return weights


def get_weights(
    IRT_params: np.array,
    seen_items: list,
    all_items: list,
    balance_weights: np.array,
    scenario: str,
    scenarios_position: list,
):
    """
    Assign each item a seen item based on distance in IRT-parameter space.
    Calculate weights for each seen item based on other items assigned to it.
    """

    assert np.mean(balance_weights < 0) == 0
    norm_balance_weights = balance_weights[scenarios_position[scenario]]
    norm_balance_weights /= norm_balance_weights.sum()

    # Prepare parameter space
    params_seen = IRT_params[:, :, seen_items]
    params_all = IRT_params[:, :, all_items]

    # Prepare an array to hold the indices of the closest points
    closest_indices = np.zeros(params_all.shape[2], dtype=int)

    # Iterate over each unseen item
    for i in range(params_all.shape[2]):
        # Calculate the Euclidean distances to all seen items
        distances = np.linalg.norm(
            params_seen - params_all[:, :, i : i + 1], axis=1
        )

        # Find the index of the closest seen item
        closest_indices[i] = np.argmin(distances)

    item_weights = np.array(
        [
            norm_balance_weights[closest_indices == i].sum()
            for i in range(len(seen_items))
        ]
    )

    return item_weights


def get_anchor(
    scores_train,
    chosen_scenarios,
    scenarios_position,
    number_item,
    balance_weights,
    random_seed,
):
    """
    Calculates anchor points, anchor weights, seen and unseen items.

    Parameters:
    scores_train (array): The training scores.
    chosen_scenarios: List of considered scenarios.
    scenarios_position (dict): Positions of the scenarios.
    number_items (list): List containing numbers of clusters/items.

    Returns:
    tuple: A tuple containing anchor points, anchor weights, seen items, and unseen items.
    """
    anchor_points = {}
    anchor_weights = {}
    for scenario in chosen_scenarios:
        anchor_points[scenario] = {}
        anchor_weights[scenario] = {}
        (
            anchor_points[scenario],
            anchor_weights[scenario],
        ) = get_anchor_points_weights(
            scores_train,
            scenarios_position,
            scenario,
            number_item,
            balance_weights,
            random_seed,
        )

    seen_items = [
        list(np.array(scenarios_position[scenario])[anchor_points[scenario]])
        for scenario in chosen_scenarios
    ]
    seen_items = list(np.array(seen_items).reshape(-1))
    unseen_items = [
        i for i in range(scores_train.shape[1]) if i not in seen_items
    ]

    return anchor_points, anchor_weights, seen_items, unseen_items


def get_anchor_points_weights(
    scores_train,
    scenarios_position,
    scenario,
    number_item,
    balance_weights,
    random_seed,
):
    """
    Calculates anchor points and weights using KMeans clustering.

    Parameters:
    scores_train (array): The training scores.
    scenarios_position (list): Positions of the scenarios.
    scenario (int): The specific scenario index.
    number_item (int): Number of clusters/items.

    Returns:
    tuple: A tuple containing the anchor points and anchor weights.
    """
    trials = 5

    assert np.mean(balance_weights < 0) == 0
    norm_balance_weights = balance_weights[scenarios_position[scenario]]
    norm_balance_weights /= norm_balance_weights.sum()

    # Fitting the KMeans model
    X = scores_train[:, scenarios_position[scenario]].T
    kmeans_models = [
        KMeans(
            n_clusters=number_item,
            random_state=1000 * t + random_seed,
            n_init="auto",
        ).fit(X, sample_weight=norm_balance_weights)
        for t in range(trials)
    ]
    kmeans = kmeans_models[np.argmin([m.inertia_ for m in kmeans_models])]

    # Calculating anchor points
    anchor_points = pairwise_distances(
        kmeans.cluster_centers_, X, metric="euclidean"
    ).argmin(axis=1)

    # Calculating anchor weights
    anchor_weights = np.array(
        [
            np.sum(norm_balance_weights[kmeans.labels_ == c])
            for c in range(number_item)
        ]
    )
    assert abs(anchor_weights.sum() - 1) < 1e-5

    return anchor_points, anchor_weights


def sample_items(
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
    skip_irt=False,
):
    assert "adaptive" not in sampling_name

    if skip_irt and sampling_name == "anchor-irt":
        raise NotImplementedError

    item_weights_dic, seen_items_dic, unseen_items_dic = {}, {}, {}
    start_time = time.time()

    for it in range(iterations):
        # if sampling_name == 'high-disagreement':
        if "disagreement" in sampling_name:
            # if '@' in sampling_name:
            #     n_guiding_models = int(sampling_name.split('@')[1])
            # else:
            #     n_guiding_models = None
            item_weights, seen_items, unseen_items = sample_by_disagreement(
                sampling_name,
                chosen_scenarios,
                scenarios,
                number_item,
                subscenarios_position,
                num_samples_in_test=responses_test[it].shape[1],
                predictions_train=predictions_train[it],
                balance_weights=balance_weights[it],
                disagreement_scores_dict=disagreement_scores_dict[it],
                random_seed=it,
                high_first=("high" in sampling_name),
                # n_guiding_models=n_guiding_models
            )
        elif sampling_name == "random":
            item_weights, seen_items, unseen_items = get_random(
                chosen_scenarios,
                scenarios,
                number_item,
                subscenarios_position,
                responses_test,
                balance_weights,
                random_seed=it,
            )

        elif sampling_name == "anchor":
            _, item_weights, seen_items, unseen_items = get_anchor(
                scores_train,
                chosen_scenarios,
                scenarios_position,
                number_item,
                balance_weights,
                random_seed=it,
            )

        elif sampling_name == "anchor-irt":
            A = A.squeeze()
            B = B.squeeze().reshape(A.shape)
            # _, item_weights, seen_items, unseen_items = get_anchor(np.vstack((A.squeeze(), B.reshape((1,-1)))), chosen_scenarios, scenarios_position, number_item, balance_weights, random_seed=it)
            _, item_weights, seen_items, unseen_items = get_anchor(
                np.vstack((A, B)),
                chosen_scenarios,
                scenarios_position,
                number_item,
                balance_weights,
                random_seed=it,
            )

        else:
            raise NotImplementedError

        item_weights_dic[it], seen_items_dic[it], unseen_items_dic[it] = (
            item_weights,
            seen_items,
            unseen_items,
        )

    end_time = time.time()
    elapsed_time = (end_time - start_time) / iterations

    return item_weights_dic, seen_items_dic, unseen_items_dic, elapsed_time


def sample_items_adaptive(
    number_items,
    iterations,
    sampling_name,
    chosen_scenarios,
    scenarios,
    subscenarios_position,
    responses_model,
    scores_train,
    scenarios_position,
    A,
    B,
    balance_weights,
    #  disagreement_scores_dict,
    initial_items=None,
    balance=True,
):
    assert "adaptive" in sampling_name

    # list of different num_items results for one model
    start_time = time.time()
    (
        item_weights_model,
        seen_items_model,
        unseen_items_model,
    ) = run_adaptive_selection(
        responses_model,
        initial_items,
        chosen_scenarios,
        scenarios_position,
        A,
        B,
        number_items,
        balance_weights,
        balance=balance,
    )
    end_time = time.time()
    elapsed_time = end_time - start_time

    return (
        item_weights_model,
        seen_items_model,
        unseen_items_model,
        elapsed_time,
    )
