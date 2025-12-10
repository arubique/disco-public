# The code is adapted from the tinyBenchmarks repo: https://github.com/felipemaiapolo/efficbench
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import sys
import os
import numpy as np
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt
import matplotlib
import h5py


def dict_to_h5(
    data_dict, filename, compression="gzip", max_depth=10, assert_equal=False
):
    """
    Convert a nested dictionary to HDF5 format.

    Args:
        data_dict: Dictionary (possibly nested) to save
        filename: Output HDF5 filename
        compression: Compression algorithm ('gzip', 'lzf', 'szip', or None)
        max_depth: Maximum nesting depth to prevent infinite recursion
    """

    def _save_to_group(group, data, current_depth=0):
        """Recursively save nested dictionary to HDF5 groups"""
        if current_depth > max_depth:
            raise ValueError(f"Maximum nesting depth ({max_depth}) exceeded")

        for key, value in data.items():
            # Escape forward slashes in keys to prevent HDF5 from treating them as group separators
            key_str = str(key).replace("/", "__SLASH__")

            if isinstance(value, dict):
                # Create a subgroup for nested dictionaries
                subgroup = group.create_group(key_str)
                _save_to_group(subgroup, value, current_depth + 1)

            elif isinstance(value, (list, tuple)):
                # Convert lists/tuples to numpy arrays
                original_type = "list" if isinstance(value, list) else "tuple"
                try:
                    arr = np.array(value)
                    # Handle Unicode string arrays by converting to bytes
                    if arr.dtype.kind == "U":  # Unicode string
                        # Convert Unicode array to bytes array
                        max_len = max(len(str(item)) for item in arr.flat)
                        arr_bytes = np.array(
                            [
                                str(item).encode("utf-8").ljust(max_len, b"\0")
                                for item in arr.flat
                            ],
                            dtype=f"S{max_len}",
                        )
                        arr_bytes = arr_bytes.reshape(arr.shape)
                        if arr.size > 1:
                            dataset = group.create_dataset(
                                key_str, data=arr_bytes, compression=compression
                            )
                        else:
                            dataset = group.create_dataset(
                                key_str, data=arr_bytes
                            )
                        # Store original type information
                        dataset.attrs["original_type"] = original_type
                    else:
                        # Only apply compression if array has more than 1 element
                        if arr.size > 1:
                            dataset = group.create_dataset(
                                key_str, data=arr, compression=compression
                            )
                        else:
                            dataset = group.create_dataset(key_str, data=arr)
                        # Store original type information
                        dataset.attrs["original_type"] = original_type
                except (ValueError, TypeError):
                    # If conversion fails, try to handle mixed types
                    arr = np.array(value, dtype=object)
                    if arr.size > 1:
                        dataset = group.create_dataset(
                            key_str, data=arr, compression=compression
                        )
                    else:
                        dataset = group.create_dataset(key_str, data=arr)
                    # Store original type information
                    dataset.attrs["original_type"] = original_type

            elif isinstance(value, np.ndarray):
                # Handle Unicode string arrays by converting to bytes
                if value.dtype.kind == "U":  # Unicode string
                    # Convert Unicode array to bytes array
                    max_len = max(len(str(item)) for item in value.flat)
                    arr_bytes = np.array(
                        [
                            str(item).encode("utf-8").ljust(max_len, b"\0")
                            for item in value.flat
                        ],
                        dtype=f"S{max_len}",
                    )
                    arr_bytes = arr_bytes.reshape(value.shape)
                    if value.size > 1:
                        group.create_dataset(
                            key_str, data=arr_bytes, compression=compression
                        )
                    else:
                        group.create_dataset(key_str, data=arr_bytes)
                else:
                    # Only apply compression if array has more than 1 element
                    if value.size > 1:
                        group.create_dataset(
                            key_str, data=value, compression=compression
                        )
                    else:
                        group.create_dataset(key_str, data=value)

            elif isinstance(value, (int, float)):
                # Store numeric scalars (no compression for scalars)
                group.create_dataset(key_str, data=value)

            elif isinstance(value, str):
                # Store strings (no compression for scalars)
                group.create_dataset(key_str, data=value.encode("utf-8"))

            elif isinstance(value, bool):
                # Store booleans (no compression for scalars)
                group.create_dataset(key_str, data=int(value))

            else:
                # For other types, try to pickle them
                try:
                    pickled_data = pickle.dumps(value)
                    # Convert to fixed-size byte array to handle NULL bytes
                    pickled_array = np.frombuffer(pickled_data, dtype=np.uint8)
                    # Only apply compression if array has more than 1 element
                    if pickled_array.size > 1:
                        group.create_dataset(
                            key_str, data=pickled_array, compression=compression
                        )
                    else:
                        group.create_dataset(key_str, data=pickled_array)
                except Exception as e:
                    print(
                        f"Warning: Could not save key '{key_str}' with value type {type(value)}: {e}"
                    )
                    # Store as string representation as last resort
                    str_data = str(value).encode("utf-8")
                    # Convert to fixed-size byte array to handle NULL bytes
                    str_array = np.frombuffer(str_data, dtype=np.uint8)
                    if str_array.size > 1:
                        group.create_dataset(
                            key_str, data=str_array, compression=compression
                        )
                    else:
                        group.create_dataset(key_str, data=str_array)

    try:
        with h5py.File(filename, "w") as f:
            # Store metadata
            f.attrs["created_by"] = "dict_to_h5"
            f.attrs["max_depth"] = max_depth

            # Save the data
            _save_to_group(f, data_dict)

        print(f"Successfully saved nested dictionary to {filename}")

    except Exception as e:
        print(f"Error saving to HDF5: {e}")
        raise

    if assert_equal:
        loaded_dict = h5_to_dict(filename)
        _assert_dicts_equal(
            data_dict, loaded_dict, "Original", "Loaded from HDF5"
        )


def _assert_dicts_equal(dict1, dict2, name1="Dict1", name2="Dict2", path=""):
    """
    Compare two nested dictionaries and provide detailed information about differences.

    Args:
        dict1, dict2: Dictionaries to compare
        name1, name2: Names to use in error messages
        path: Current path in the nested structure (for error reporting)
    """

    def _get_path_str(current_path, key):
        if current_path:
            return f"{current_path}.{key}"
        return str(key)

    # Check if both are dictionaries
    if not isinstance(dict1, dict) or not isinstance(dict2, dict):
        if dict1 != dict2:
            print(f"MISMATCH at {path}:")
            print(f"  {name1}: {type(dict1)} = {repr(dict1)}")
            print(f"  {name2}: {type(dict2)} = {repr(dict2)}")
            assert False, f"Dictionaries differ at path '{path}'"
        return

    # Check keys
    keys1 = set(dict1.keys())
    keys2 = set(dict2.keys())

    if keys1 != keys2:
        missing_in_2 = keys1 - keys2
        missing_in_1 = keys2 - keys1
        print(f"KEY MISMATCH at {path}:")
        if missing_in_2:
            print(
                f"  Keys in {name1} but not in {name2}: {sorted(missing_in_2)}"
            )
        if missing_in_1:
            print(
                f"  Keys in {name2} but not in {name1}: {sorted(missing_in_1)}"
            )
        assert False, f"Key sets differ at path '{path}'"

    # Recursively compare values
    for key in keys1:
        current_path = _get_path_str(path, key)
        val1 = dict1[key]
        val2 = dict2[key]

        # Handle numpy arrays specially
        if isinstance(val1, np.ndarray) and isinstance(val2, np.ndarray):
            if not np.array_equal(val1, val2):
                print(f"ARRAY MISMATCH at {current_path}:")
                print(f"  {name1} shape: {val1.shape}, dtype: {val1.dtype}")
                print(f"  {name2} shape: {val2.shape}, dtype: {val2.dtype}")
                if val1.shape == val2.shape:
                    diff_mask = val1 != val2
                    if np.any(diff_mask):
                        diff_indices = np.where(diff_mask)
                        print(f"  First few differences:")
                        for i in range(min(5, len(diff_indices[0]))):
                            idx = tuple(arr[i] for arr in diff_indices)
                            print(
                                f"    [{idx}] {name1}: {val1[idx]}, {name2}: {val2[idx]}"
                            )
                assert False, f"Arrays differ at path '{current_path}'"
        elif isinstance(val1, dict) and isinstance(val2, dict):
            # Recursive comparison for nested dicts
            _assert_dicts_equal(val1, val2, name1, name2, current_path)
        else:
            # Check for type mismatch first
            if type(val1) != type(val2):
                print(f"TYPE MISMATCH at {current_path}:")
                print(f"  {name1}: {type(val1)} = {repr(val1)}")
                print(f"  {name2}: {type(val2)} = {repr(val2)}")
                assert (
                    False
                ), f"Types differ at path '{current_path}': {type(val1)} vs {type(val2)}"

            # Special handling for datasets.arrow_dataset.Column objects
            if hasattr(val1, "__class__") and "Column" in str(type(val1)):
                # Compare the underlying data of Column objects
                try:
                    # Try multiple methods to access the underlying data
                    val1_data = None
                    val2_data = None

                    # Method 1: Try to_pandas()
                    if hasattr(val1, "to_pandas") and hasattr(
                        val2, "to_pandas"
                    ):
                        val1_data = val1.to_pandas()
                        val2_data = val2.to_pandas()

                    # Method 2: Try accessing the underlying array directly
                    elif hasattr(val1, "data") and hasattr(val2, "data"):
                        val1_data = val1.data
                        val2_data = val2.data

                    # Method 3: Try converting to list
                    elif hasattr(val1, "tolist") and hasattr(val2, "tolist"):
                        val1_data = val1.tolist()
                        val2_data = val2.tolist()

                    # Method 4: Try accessing the arrow table data
                    elif hasattr(val1, "_data") and hasattr(val2, "_data"):
                        val1_data = val1._data
                        val2_data = val2._data

                    # Compare the data if we found it
                    if val1_data is not None and val2_data is not None:
                        # Handle numpy arrays
                        if isinstance(val1_data, np.ndarray) and isinstance(
                            val2_data, np.ndarray
                        ):
                            if not np.array_equal(val1_data, val2_data):
                                print(
                                    f"COLUMN DATA MISMATCH at {current_path}:"
                                )
                                print(
                                    f"  {name1}: {type(val1)} with different underlying array data"
                                )
                                print(
                                    f"  {name2}: {type(val2)} with different underlying array data"
                                )
                                assert (
                                    False
                                ), f"Column array data differs at path '{current_path}'"
                        # Handle lists
                        elif isinstance(val1_data, list) and isinstance(
                            val2_data, list
                        ):
                            if val1_data != val2_data:
                                print(
                                    f"COLUMN DATA MISMATCH at {current_path}:"
                                )
                                print(
                                    f"  {name1}: {type(val1)} with different underlying list data"
                                )
                                print(
                                    f"  {name2}: {type(val2)} with different underlying list data"
                                )
                                assert (
                                    False
                                ), f"Column list data differs at path '{current_path}'"
                        # Handle pandas DataFrames
                        elif hasattr(val1_data, "equals") and hasattr(
                            val2_data, "equals"
                        ):
                            if not val1_data.equals(val2_data):
                                print(
                                    f"COLUMN DATA MISMATCH at {current_path}:"
                                )
                                print(
                                    f"  {name1}: {type(val1)} with different underlying pandas data"
                                )
                                print(
                                    f"  {name2}: {type(val2)} with different underlying pandas data"
                                )
                                assert (
                                    False
                                ), f"Column pandas data differs at path '{current_path}'"
                        # Direct comparison for other types
                        else:
                            if val1_data != val2_data:
                                print(
                                    f"COLUMN DATA MISMATCH at {current_path}:"
                                )
                                print(
                                    f"  {name1}: {type(val1)} with different underlying data"
                                )
                                print(
                                    f"  {name2}: {type(val2)} with different underlying data"
                                )
                                assert (
                                    False
                                ), f"Column data differs at path '{current_path}'"
                    else:
                        # If we can't access the data, assume they're equal if the string representations match
                        if str(val1) == str(val2):
                            # Skip comparison - assume they're equal
                            pass
                        else:
                            print(f"COLUMN MISMATCH at {current_path}:")
                            print(f"  {name1}: {type(val1)} = {repr(val1)}")
                            print(f"  {name2}: {type(val2)} = {repr(val2)}")
                            assert (
                                False
                            ), f"Column objects differ at path '{current_path}'"
                except Exception as e:
                    print(f"COLUMN COMPARISON ERROR at {current_path}: {e}")
                    print(f"  {name1}: {type(val1)} = {repr(val1)}")
                    print(f"  {name2}: {type(val2)} = {repr(val2)}")
                    assert (
                        False
                    ), f"Cannot compare Column objects at path '{current_path}': {e}"
            else:
                # Direct comparison for other types
                try:
                    if val1 != val2:
                        print(f"VALUE MISMATCH at {current_path}:")
                        print(f"  {name1}: {type(val1)} = {repr(val1)}")
                        print(f"  {name2}: {type(val2)} = {repr(val2)}")
                        assert False, f"Values differ at path '{current_path}'"
                except (ValueError, TypeError) as e:
                    # Handle cases where direct comparison fails
                    print(f"COMPARISON ERROR at {current_path}: {e}")
                    print(f"  {name1}: {type(val1)} = {repr(val1)}")
                    print(f"  {name2}: {type(val2)} = {repr(val2)}")
                    assert (
                        False
                    ), f"Cannot compare values at path '{current_path}': {e}"


def h5_to_dict(filename):
    """
    Load HDF5 file back to nested dictionary.

    Args:
        filename: HDF5 filename to load

    Returns:
        Dictionary with the same nested structure as saved
    """

    def _load_from_group(group):
        """Recursively load HDF5 groups back to dictionary"""
        result = {}

        for key in group.keys():
            # Unescape forward slashes in keys
            original_key = key.replace("__SLASH__", "/")
            item = group[key]

            if isinstance(item, h5py.Group):
                # Recursively load subgroups
                result[original_key] = _load_from_group(item)

            elif isinstance(item, h5py.Dataset):
                # Load dataset
                try:
                    data = item[()]

                    # Handle byte strings (from string data)
                    if isinstance(data, bytes):
                        try:
                            # Try to decode as UTF-8
                            result[original_key] = data.decode("utf-8")
                        except UnicodeDecodeError:
                            # If it's not UTF-8, might be pickled data
                            try:
                                result[original_key] = pickle.loads(data)
                            except:
                                result[original_key] = data

                    # Handle numpy arrays
                    elif isinstance(data, np.ndarray):
                        # Check if this was originally a list/tuple by looking at metadata first
                        if "original_type" in item.attrs:
                            original_type = item.attrs["original_type"]
                            if original_type in ["list", "tuple"]:
                                # Handle byte string arrays (from Unicode strings) first
                                if data.dtype.kind == "S":  # Byte string array
                                    # Convert byte strings back to Unicode
                                    unicode_array = np.array(
                                        [
                                            item.decode("utf-8").rstrip("\0")
                                            for item in data.flat
                                        ]
                                    ).reshape(data.shape)
                                    if original_type == "list":
                                        result[
                                            original_key
                                        ] = unicode_array.tolist()
                                    else:  # tuple
                                        result[original_key] = tuple(
                                            unicode_array.tolist()
                                        )
                                else:
                                    # Convert back to original type
                                    if original_type == "list":
                                        result[original_key] = data.tolist()
                                    else:  # tuple
                                        result[original_key] = tuple(
                                            data.tolist()
                                        )
                            else:
                                result[original_key] = data
                        # Handle byte arrays (from pickled data)
                        elif data.dtype == np.uint8:
                            # Try to unpickle the byte array
                            try:
                                result[original_key] = pickle.loads(
                                    data.tobytes()
                                )
                            except:
                                # If unpickling fails, keep as byte array
                                result[original_key] = data
                        # Handle byte string arrays (from Unicode strings)
                        elif data.dtype.kind == "S":  # Byte string array
                            # Convert byte strings back to Unicode
                            unicode_array = np.array(
                                [
                                    item.decode("utf-8").rstrip("\0")
                                    for item in data.flat
                                ]
                            ).reshape(data.shape)
                            # Check if this was originally a list/tuple
                            if "original_type" in item.attrs:
                                original_type = item.attrs["original_type"]
                                if original_type == "list":
                                    result[
                                        original_key
                                    ] = unicode_array.tolist()
                                elif original_type == "tuple":
                                    result[original_key] = tuple(
                                        unicode_array.tolist()
                                    )
                                else:
                                    result[original_key] = unicode_array
                            else:
                                result[original_key] = unicode_array
                        elif data.dtype == object and len(data) == 1:
                            # Single object array, extract the object
                            result[original_key] = data.item()
                        else:
                            result[original_key] = data

                    # Handle scalars
                    else:
                        result[original_key] = data

                except Exception as e:
                    print(f"Warning: Could not load key '{original_key}': {e}")
                    result[original_key] = None

        return result

    try:
        with h5py.File(filename, "r") as f:
            return _load_from_group(f)

    except Exception as e:
        print(f"Error loading from HDF5: {e}")
        raise


def get_lambda(b, v):
    return (b**2) / (v + (b**2))


class SuppressPrints:

    """
    A context manager to suppress prints to the console, useful for making output cleaner.
    """

    def __enter__(self):
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        sys.stdout = open(os.devnull, "w")
        sys.stderr = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr


def sigmoid(z):
    """
    Compute the sigmoid function for the input z.

    Parameters:
    - z: A numeric value or numpy array.

    Returns:
    - The sigmoid of z.
    """

    return 1 / (1 + np.exp(-z))


def item_curve(theta, a, b):
    """
    Compute the item response curve for given parameters.

    Parameters:
    - theta: The ability parameter of the subject.
    - a: The discrimination parameter of the item.
    - b: The difficulty parameter of the item.

    Returns:
    - The probability of a correct response given the item parameters and subject ability.
    """
    z = np.clip(a * theta - b, -30, 30).sum(axis=1)
    return sigmoid(z)


def item_response_function(xj, theta, a, b):
    """
    Compute the pdf for the Bernoulli distribution of an item response.

    Parameters:
    - xj: The response of the subject (0 or 1).
    - theta: The ability parameter of the subject.
    - a: The discrimination parameter of the item.
    - b: The difficulty parameter of the item.

    Returns:
    - The pdf value for the given response.
    """
    a = np.array([[[a]]]) if type(a) == np.float64 else a
    b = np.array([[[b]]]) if type(b) == np.float64 else b
    p_correct = item_curve(theta, a, b)
    return np.power(p_correct, xj) * np.power(1 - p_correct, 1 - xj)


def prepare_data(chosen_scenarios, scenarios, data):
    """
    Prepare the data by determining the positions of items within each scenario and subscenario.

    Parameters:
    - chosen_scenarios: A list of scenarios to be considered.
    - scenarios: A dictionary mapping each scenario to its subscenarios.
    - data: A dictionary containing correctness data for each subscenario.

    Returns:
    - scenarios_position: A dictionary mapping each scenario to the positions of its items.
    - subscenarios_position: A nested dictionary mapping each scenario and subscenario to the positions of its items.
    """

    i = 0
    subscenarios_position = {}

    # Iterate through each chosen scenario and its subscenarios to record item positions
    for scenario in chosen_scenarios:
        subscenarios_position[scenario] = {}
        for sub in scenarios[scenario]:
            subscenarios_position[scenario][sub] = []
            for j in range(data["data"][sub]["correctness"].shape[0]):
                subscenarios_position[scenario][sub].append(i)
                i += 1

    # Prepare a simplified mapping of scenarios to their item positions
    scenarios_position = {}
    for scenario in chosen_scenarios:
        scenarios_position[scenario] = []
        for key in subscenarios_position[scenario].keys():
            scenarios_position[scenario] += subscenarios_position[scenario][key]
    return scenarios_position, subscenarios_position


def hstack_by_attribute_key(chosen_scenarios, scenarios, data, attribute_key):
    """
    Stack the data by the specified attribute key.

    Parameters:
    - chosen_scenarios: List of scenarios to consider.
    - scenarios: Dictionary mapping scenarios to their subscenarios.
    - data: The data to be used for creating responses and weights.
    - attribute_key: The key to stack the data by.

    Returns:
    - A numpy array of the stacked data.
    """

    transpose_order = (1, 0, 2) if attribute_key == "predictions" else (1, 0)
    predictions = [
        np.vstack(
            [data["data"][sub][attribute_key] for sub in scenarios[scenario]]
        ).transpose(transpose_order)
        for scenario in chosen_scenarios
    ]
    predictions = np.hstack(predictions)
    return predictions


def create_predictions(chosen_scenarios, scenarios, data):
    predictions = hstack_by_attribute_key(
        chosen_scenarios, scenarios, data, "predictions"
    )
    return predictions


def create_responses(chosen_scenarios, scenarios, data):
    """
    Create a matrix of responses for the chosen scenarios.

    Parameters:
    - chosen_scenarios: A list of scenarios to be considered.
    - scenarios: A dictionary mapping each scenario to its subscenarios.
    - data: A dictionary containing correctness data for each subscenario.

    Returns:
    - A numpy array of responses for the chosen scenarios.
    """

    # responses = [np.vstack([data['data'][sub]['correctness'] for sub in scenarios[scenario]]).T for scenario in chosen_scenarios]
    # responses = np.hstack(responses)
    responses = hstack_by_attribute_key(
        chosen_scenarios, scenarios, data, "correctness"
    )
    return responses


def prepare_and_split_data(
    chosen_scenarios, scenarios, data, rows_to_hide, n_source_models=None
):
    """
    Prepares data based on chosen scenarios and splits it into training and testing sets.

    Parameters:
    - chosen_scenarios: List of scenarios to consider.
    - scenarios: Dictionary mapping scenarios to their subscenarios.
    - data: The data to be used for creating responses and weights.
    - rows_to_hide: Indices of rows in the data to be excluded from the training set and used for testing.

    Returns:
    - scores_train: The training set, excluding rows specified by rows_to_hide.
    - scores_test: The testing set, including only rows specified by rows_to_hide.
    - balance_weights: Array of weights used to balance the training data.
    """

    def split_array_in_train_test(array, rows_to_hide):
        train_array = array[
            [i for i in range(array.shape[0]) if i not in rows_to_hide]
        ]
        test_array = array[rows_to_hide]
        return train_array, test_array

    # Prepare data and scenarios
    scenarios_position, subscenarios_position = prepare_data(
        chosen_scenarios, scenarios, data
    )
    scores = create_responses(chosen_scenarios, scenarios, data)
    predictions = create_predictions(chosen_scenarios, scenarios, data)
    # Balance weights
    balance_weights = np.ones(scores.shape[1])
    for scenario in chosen_scenarios:  # list of scnearios, e.g., ["mmlu"]
        N = len(
            scenarios_position[scenario]
        )  # #datapoints in scenario, e.g., 14042
        n_sub = len(scenarios[scenario])  # #sub-scenarios, e.g., 57
        for sub in scenarios[
            scenario
        ]:  # sub-scenario name, e.g., "harness_hendrycksTest_abstract_algebra_5"
            n_i = len(
                subscenarios_position[scenario][sub]
            )  # #datapoints in sub-scenario, e.g., 100
            balance_weights[subscenarios_position[scenario][sub]] = N / (
                n_sub * n_i
            )
    # Create training and test sets by hiding specific rows
    # scores_train = scores[[i for i in range(scores.shape[0]) if i not in rows_to_hide]]
    # scores_test = scores[rows_to_hide]
    scores_train, scores_test = split_array_in_train_test(scores, rows_to_hide)

    predictions_train, predictions_test = split_array_in_train_test(
        predictions, rows_to_hide
    )

    if n_source_models is not None:
        predictions_train = predictions_train[:n_source_models]
        # predictions_test = predictions_test[:n_source_models]
        scores_train = scores_train[:n_source_models]
        # scores_test = scores_test[:n_source_models]

    return (
        scores_train,
        predictions_train,
        predictions_test,
        scores_test,
        balance_weights,
        scenarios_position,
        subscenarios_position,
    )


helm_lite_scenarios = {
    "commonsense:dataset=openbookqa,method=multiple_choice_joint,": [
        "commonsense:dataset=openbookqa,method=multiple_choice_joint,"
    ],
    "gsm:": ["gsm:"],
    "med_qa:": ["med_qa:"],
    "legalbench": [
        "legalbench:subset=abercrombie,",
        "legalbench:subset=corporate_lobbying,",
        "legalbench:subset=function_of_decision_section,",
        "legalbench:subset=proa,",
        "legalbench:subset=international_citizenship_questions,",
    ],
    "math": [
        "math:subject=algebra,level=1,use_official_examples=False,use_chain_of_thought=True,",
        "math:subject=counting_and_probability,level=1,use_official_examples=False,use_chain_of_thought=True,",
        "math:subject=geometry,level=1,use_official_examples=False,use_chain_of_thought=True,",
        "math:subject=intermediate_algebra,level=1,use_official_examples=False,use_chain_of_thought=True,",
        "math:subject=number_theory,level=1,use_official_examples=False,use_chain_of_thought=True,",
        "math:subject=prealgebra,level=1,use_official_examples=False,use_chain_of_thought=True,",
        "math:subject=precalculus,level=1,use_official_examples=False,use_chain_of_thought=True,",
    ],
    "mmlu": [
        "mmlu:subject=abstract_algebra,method=multiple_choice_joint,",
        "mmlu:subject=college_chemistry,method=multiple_choice_joint,",
        "mmlu:subject=computer_security,method=multiple_choice_joint,",
        "mmlu:subject=econometrics,method=multiple_choice_joint,",
        "mmlu:subject=us_foreign_policy,method=multiple_choice_joint,",
    ],
    "narrative_qa:": ["narrative_qa:"],
    "natural_qa:mode=closedbook,": ["natural_qa:mode=closedbook,"],
    "natural_qa:mode=openbook_longans,": ["natural_qa:mode=openbook_longans,"],
    "wmt_14": [
        "wmt_14:language_pair=cs-en,",
        "wmt_14:language_pair=de-en,",
        "wmt_14:language_pair=fr-en,",
        "wmt_14:language_pair=hi-en,",
        "wmt_14:language_pair=ru-en,",
    ],
}

lb_scenarios = {
    "truthfulqa": ["harness_truthfulqa_mc_0"],
    "gsm8k": ["harness_gsm8k_5"],
    "winogrande": ["harness_winogrande_5"],
    "arc": ["harness_arc_challenge_25"],
    "hellaswag": ["harness_hellaswag_10"],
    "mmlu": [
        "harness_hendrycksTest_abstract_algebra_5",
        "harness_hendrycksTest_anatomy_5",
        "harness_hendrycksTest_astronomy_5",
        "harness_hendrycksTest_business_ethics_5",
        "harness_hendrycksTest_clinical_knowledge_5",
        "harness_hendrycksTest_college_biology_5",
        "harness_hendrycksTest_college_chemistry_5",
        "harness_hendrycksTest_college_computer_science_5",
        "harness_hendrycksTest_college_mathematics_5",
        "harness_hendrycksTest_college_medicine_5",
        "harness_hendrycksTest_college_physics_5",
        "harness_hendrycksTest_computer_security_5",
        "harness_hendrycksTest_conceptual_physics_5",
        "harness_hendrycksTest_econometrics_5",
        "harness_hendrycksTest_electrical_engineering_5",
        "harness_hendrycksTest_elementary_mathematics_5",
        "harness_hendrycksTest_formal_logic_5",
        "harness_hendrycksTest_global_facts_5",
        "harness_hendrycksTest_high_school_biology_5",
        "harness_hendrycksTest_high_school_chemistry_5",
        "harness_hendrycksTest_high_school_computer_science_5",
        "harness_hendrycksTest_high_school_european_history_5",
        "harness_hendrycksTest_high_school_geography_5",
        "harness_hendrycksTest_high_school_government_and_politics_5",
        "harness_hendrycksTest_high_school_macroeconomics_5",
        "harness_hendrycksTest_high_school_mathematics_5",
        "harness_hendrycksTest_high_school_microeconomics_5",
        "harness_hendrycksTest_high_school_physics_5",
        "harness_hendrycksTest_high_school_psychology_5",
        "harness_hendrycksTest_high_school_statistics_5",
        "harness_hendrycksTest_high_school_us_history_5",
        "harness_hendrycksTest_high_school_world_history_5",
        "harness_hendrycksTest_human_aging_5",
        "harness_hendrycksTest_human_sexuality_5",
        "harness_hendrycksTest_international_law_5",
        "harness_hendrycksTest_jurisprudence_5",
        "harness_hendrycksTest_logical_fallacies_5",
        "harness_hendrycksTest_machine_learning_5",
        "harness_hendrycksTest_management_5",
        "harness_hendrycksTest_marketing_5",
        "harness_hendrycksTest_medical_genetics_5",
        "harness_hendrycksTest_miscellaneous_5",
        "harness_hendrycksTest_moral_disputes_5",
        "harness_hendrycksTest_moral_scenarios_5",
        "harness_hendrycksTest_nutrition_5",
        "harness_hendrycksTest_philosophy_5",
        "harness_hendrycksTest_prehistory_5",
        "harness_hendrycksTest_professional_accounting_5",
        "harness_hendrycksTest_professional_law_5",
        "harness_hendrycksTest_professional_medicine_5",
        "harness_hendrycksTest_professional_psychology_5",
        "harness_hendrycksTest_public_relations_5",
        "harness_hendrycksTest_security_studies_5",
        "harness_hendrycksTest_sociology_5",
        "harness_hendrycksTest_us_foreign_policy_5",
        "harness_hendrycksTest_virology_5",
        "harness_hendrycksTest_world_religions_5",
    ],
}

alpaca_scenarios = {"alpaca_v2": ["alpaca_v2"]}

icl_templates_scenarios = {"templates": ["templates"]}


def dump_pickle(data, filename):
    with open(filename, "wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(filename):
    with open(filename, "rb") as handle:
        return pickle.load(handle)
