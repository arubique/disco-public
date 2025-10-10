import pickle
import os
import sys
import argparse
import numpy as np
from datetime import datetime
import h5py


ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_PATH)
from plots import DATA_FOLDER
from utils_for_notebooks import (
    MAX_NUM_ANSWERS,
    KEYS_TO_ADD,
    SUB_TO_SKIP,
    parse_df_with_results,
)
from utils import dump_pickle, load_pickle

sys.path.pop(0)


MAX_MODELS = None
# for some scenarios, these models have limited number of responses
MODELS_TO_REMOVE = [
    "open-llm-leaderboard/details_mindy-labs__mindy-7b",
    "open-llm-leaderboard/details_alnrg2arg__test2_3",
    "open-llm-leaderboard/details_NousResearch__Nous-Hermes-2-Mixtral-8x7B-SFT",
    "open-llm-leaderboard/details_internlm__internlm2-20b",
    "open-llm-leaderboard/details_garage-bAInd__Platypus2-70B-instruct",
    "open-llm-leaderboard/details_jondurbin__airoboros-l2-70b-2.2.1",
    "open-llm-leaderboard/details_sethuiyer__distilabled_Chikuma_10.7B",
    "open-llm-leaderboard/details_AIDC-ai-business__Marcoroni-70B-v1",
    "open-llm-leaderboard/details_psmathur__model_009",
    "open-llm-leaderboard/details_fblgit__juanako-7b-UNA",
    "open-llm-leaderboard/details_Brillibits__Instruct_Llama70B_Dolly15k",
    "open-llm-leaderboard/details_monology__openinstruct-mistral-7b",
    "open-llm-leaderboard/details_huggyllama__llama-65b",
    "open-llm-leaderboard/details_tianlinliu0121__zephyr-7b-dpo-full-beta-0.2",
    "open-llm-leaderboard/details_uukuguy__speechless-llama2-13b",
    "open-llm-leaderboard/details_vihangd__smartyplats-7b-v2",
    "open-llm-leaderboard/details_HuggingFaceH4__zephyr-7b-alpha",
    "open-llm-leaderboard/details_abdulrahman-nuzha__finetuned-Mistral-5000-v1.0",
    "open-llm-leaderboard/details_uukuguy__speechless-code-mistral-7b-v1.0",
    "open-llm-leaderboard/details_ajibawa-2023__scarlett-33b",
    "open-llm-leaderboard/details_Aspik101__30B-Lazarus-instruct-PL-lora_unload",
    "open-llm-leaderboard/details_lgaalves__mistral-7b-platypus1k",
    "open-llm-leaderboard/details_Sao10K__Stheno-1.8-L2-13B",
    "open-llm-leaderboard/details_tiiuae__falcon-40b",
    "open-llm-leaderboard/details_Aeala__VicUnlocked-alpaca-30b",
    "open-llm-leaderboard/details_Sao10K__Stheno-v2-Delta-fp16",
    "open-llm-leaderboard/details_sauce1337__BerrySauce-L2-13b",
    "open-llm-leaderboard/details_migtissera__Synthia-13B-v1.2",
    "open-llm-leaderboard/details_meta-llama__Llama-2-13b-hf",
    "open-llm-leaderboard/details_Undi95__U-Amethyst-20B",
    "open-llm-leaderboard/details_uukuguy__speechless-hermes-coig-lite-13b",
    "open-llm-leaderboard/details_speechlessai__speechless-llama2-dolphin-orca-platypus-13b",
    "open-llm-leaderboard/details_KoboldAI__LLaMA2-13B-Holomax",
    "open-llm-leaderboard/details_openaccess-ai-collective__wizard-mega-13b",
    "open-llm-leaderboard/details_yeontaek__llama-2-13b-Beluga-QLoRA",
    "open-llm-leaderboard/details_WizardLM__WizardMath-13B-V1.0",
    "open-llm-leaderboard/details_yeontaek__llama-2-13b-QLoRA",
    "open-llm-leaderboard/details_NobodyExistsOnTheInternet__PuffedConvo13bLoraE4",
    "open-llm-leaderboard/details_circulus__Llama-2-7b-orca-v1",
    "open-llm-leaderboard/details_PocketDoc__Dans-PersonalityEngine-13b",
    "open-llm-leaderboard/details_lvkaokao__llama2-7b-hf-chat-lora-v2",
    "open-llm-leaderboard/details_beaugogh__Llama2-7b-openorca-mc-v1",
    "open-llm-leaderboard/details_pe-nlp__llama-2-13b-vicuna-wizard",
    "open-llm-leaderboard/details_JosephusCheung__Pwen-VL-Chat-20_30",
    "open-llm-leaderboard/details_ziqingyang__chinese-llama-2-13b",
    "open-llm-leaderboard/details_RoversX__llama-2-7b-hf-small-shards-Samantha-V1-SFT",
    "open-llm-leaderboard/details_llm-agents__tora-code-34b-v1.0",
    "open-llm-leaderboard/details_quantumaikr__QuantumLM-7B",
    "open-llm-leaderboard/details_ceadar-ie__FinanceConnect-13B",
    "open-llm-leaderboard/details_ajibawa-2023__scarlett-7b",
    "open-llm-leaderboard/details_AlpinDale__pygmalion-instruct",
    "open-llm-leaderboard/details_microsoft__phi-1_5",
    "open-llm-leaderboard/details_golaxy__gowizardlm",
    "open-llm-leaderboard/details_stabilityai__stablelm-3b-4e1t",
    "open-llm-leaderboard/details_speechlessai__speechless-coding-7b-16k-tora",
    "open-llm-leaderboard/details_AlekseyKorshuk__pygmalion-6b-vicuna-chatml",
    "open-llm-leaderboard/details_codellama__CodeLlama-34b-Python-hf",
    "open-llm-leaderboard/details_GeorgiaTechResearchInstitute__galactica-6.7b-evol-instruct-70k",
    "open-llm-leaderboard/details_HuggingFaceH4__starchat-alpha",
]


def dict_to_h5(data_dict, filename, compression="gzip", max_depth=10):
    """
    Convert a nested dictionary to HDF5 format.

    Args:
        data_dict: Dictionary (possibly nested) to save
        filename: Output HDF5 filename
        compression: Compression algorithm ('gzip', 'lzf', 'szip', or None)
        max_depth: Maximum nesting depth to prevent infinite recursion

    Example:
        nested_dict = {
            'model1': {
                'dataset1': {'predictions': [1, 2, 3], 'accuracy': 0.95},
                'dataset2': {'predictions': [4, 5, 6], 'accuracy': 0.87}
            },
            'model2': {
                'dataset1': {'predictions': [7, 8, 9], 'accuracy': 0.92}
            }
        }
        dict_to_h5(nested_dict, 'output.h5')
    """

    def _save_to_group(group, data, current_depth=0):
        """Recursively save nested dictionary to HDF5 groups"""
        if current_depth > max_depth:
            raise ValueError(f"Maximum nesting depth ({max_depth}) exceeded")

        for key, value in data.items():
            key_str = str(key)

            if isinstance(value, dict):
                # Create a subgroup for nested dictionaries
                subgroup = group.create_group(key_str)
                _save_to_group(subgroup, value, current_depth + 1)

            elif isinstance(value, (list, tuple)):
                # Convert lists/tuples to numpy arrays
                try:
                    arr = np.array(value)
                    group.create_dataset(
                        key_str, data=arr, compression=compression
                    )
                except (ValueError, TypeError):
                    # If conversion fails, try to handle mixed types
                    group.create_dataset(
                        key_str,
                        data=np.array(value, dtype=object),
                        compression=compression,
                    )

            elif isinstance(value, np.ndarray):
                group.create_dataset(
                    key_str, data=value, compression=compression
                )

            elif isinstance(value, (int, float)):
                # Store numeric scalars
                group.create_dataset(
                    key_str, data=value, compression=compression
                )

            elif isinstance(value, str):
                # Store strings
                group.create_dataset(
                    key_str, data=value.encode("utf-8"), compression=compression
                )

            elif isinstance(value, bool):
                # Store booleans
                group.create_dataset(
                    key_str, data=int(value), compression=compression
                )

            else:
                # For other types, try to pickle them
                try:
                    pickled_data = pickle.dumps(value)
                    group.create_dataset(
                        key_str, data=pickled_data, compression=compression
                    )
                except Exception as e:
                    print(
                        f"Warning: Could not save key '{key_str}' with value type {type(value)}: {e}"
                    )
                    # Store as string representation as last resort
                    group.create_dataset(
                        key_str,
                        data=str(value).encode("utf-8"),
                        compression=compression,
                    )

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
            item = group[key]

            if isinstance(item, h5py.Group):
                # Recursively load subgroups
                result[key] = _load_from_group(item)

            elif isinstance(item, h5py.Dataset):
                # Load dataset
                try:
                    data = item[()]

                    # Handle byte strings (from string data)
                    if isinstance(data, bytes):
                        try:
                            # Try to decode as UTF-8
                            result[key] = data.decode("utf-8")
                        except UnicodeDecodeError:
                            # If it's not UTF-8, might be pickled data
                            try:
                                result[key] = pickle.loads(data)
                            except:
                                result[key] = data

                    # Handle numpy arrays
                    elif isinstance(data, np.ndarray):
                        if data.dtype == object and len(data) == 1:
                            # Single object array, extract the object
                            result[key] = data.item()
                        else:
                            result[key] = data

                    # Handle scalars
                    else:
                        result[key] = data

                except Exception as e:
                    print(f"Warning: Could not load key '{key}': {e}")
                    result[key] = None

        return result

    try:
        with h5py.File(filename, "r") as f:
            return _load_from_group(f)

    except Exception as e:
        print(f"Error loading from HDF5: {e}")
        raise


# adapted from: https://github.com/felipemaiapolo/efficbench/blob/master/generating_data/download-openllmleaderboard/process_lb_data.ipynb
def process_extended_data(df, max_models=MAX_MODELS):
    models = list(df.keys())

    for models_to_remove in MODELS_TO_REMOVE:
        models.remove(models_to_remove)
    print(len(models))
    # 393

    dates = np.array(
        [
            datetime.strptime(
                df[model]["harness_gsm8k_5"]["dates"][0][:10], "%Y_%m_%d"
            )
            for model in models
        ]
    )

    order = np.argsort(dates)[::-1]

    data, max_answers_dict = parse_df_with_results(
        df,
        models,
        # order=None,
        order=order,
        sub_to_skip=SUB_TO_SKIP,
        max_num_answers=MAX_NUM_ANSWERS,
        keys_to_add=KEYS_TO_ADD,
        max_models=max_models,
    )

    cnt = 0
    for sub in data["data"].keys():
        for key in KEYS_TO_ADD:
            data["data"][sub][key] = data["data"][sub][key][:, order]
            if cnt < 10:
                print(data["data"][sub][key].shape)
            cnt += 1

    return data, df


# adapted from: https://github.com/felipemaiapolo/efficbench/blob/master/generating_data/download-openllmleaderboard/process_mmlu_fields_data.ipynb
def process_data(extended_data, df, max_models=MAX_MODELS):
    def prune_data(data, data_lb, sub, delete_ind, keyword):
        # hstack mmlu_fields data with lb data and remove
        if delete_ind is not None and len(delete_ind) > 0:
            data["data"][sub][keyword] = np.hstack(
                (
                    data["data"][sub][keyword],
                    np.delete(
                        data_lb["data"][sub][keyword], delete_ind, axis=1
                    ),
                )
            )

    data_lb = extended_data

    models = list(df.keys())

    # remove empty model lines
    for key1 in df.keys():
        for key2 in df[key1].keys():
            if df[key1][key2] == None:
                try:
                    models.remove(key1)
                except:
                    pass

    data, max_answers_dict = parse_df_with_results(
        df, models, order=None, max_models=max_models
    )

    mmlu_sub = [
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
    ]

    delete_ind = -np.sort(
        [
            -i
            for i in range(len(data_lb["models"]))
            if data_lb["models"][i] in data["models"]
        ]
    )

    for sub in mmlu_sub + [
        "harness_arc_challenge_25",
        "harness_hellaswag_10",
        "harness_truthfulqa_mc_0",
        "harness_winogrande_5",
    ]:
        print(data["data"][sub]["correctness"].shape)
        prune_data(data, data_lb, sub, delete_ind, "correctness")
        prune_data(data, data_lb, sub, delete_ind, "predictions")
        print(data["data"][sub]["correctness"].shape)

    data["models"] += [
        data_lb["models"][i]
        for i in range(len(data_lb["models"]))
        if i not in delete_ind
    ]

    return data, df


# adapted from: https://github.com/felipemaiapolo/efficbench/blob/c9df1547a460d5f1d2ad1ad88b6a117979858fbc/generating_data/download-openllmleaderboard/process_lb_data.ipynb
def sort_models_by_dates(data, extended_raw_df, raw_df):
    # def check_order(order, dates, idx=None):
    def check_order(data, extended_raw_df, raw_df, idx=None):
        dates_list = []
        for model_name in data["models"]:
            if model_name in extended_raw_df.keys():
                df = extended_raw_df
                # [0][:10]
            else:
                assert model_name in raw_df.keys()
                df = raw_df
            date = df[model_name]["harness_gsm8k_5"]["dates"][0][:10]
            dates_list.append(datetime.strptime(date, "%Y_%m_%d"))
            # break
        dates = np.array(dates_list)
        # print(dates)
        order = np.argsort(dates)[::-1]
        print("Newest first:")
        print(order)
        print(f"{order[0]}:", dates[order[0]])
        print(f"{order[-1]}:", dates[order[-1]])
        if idx is not None:
            print(f"{idx}:", dates[idx])
        return order

    order = check_order(data, extended_raw_df, raw_df)

    data_ordered = data
    data_ordered["models"] = [data["models"][i] for i in order]
    for sub in data_ordered["data"].keys():
        for key in data_ordered["data"][sub].keys():
            print(data_ordered["data"][sub][key].shape)
            if key == "correctness":
                data_ordered["data"][sub][key] = data_ordered["data"][sub][key][
                    :, order
                ]
            else:
                assert key == "predictions"
                data_ordered["data"][sub][key] = data_ordered["data"][sub][key][
                    :, order, :
                ]
            print(data_ordered["data"][sub][key].shape)

    check_order(data_ordered, extended_raw_df, raw_df)
    return data_ordered


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lb_raw_extended_path",
        type=str,
        default=os.path.join(DATA_FOLDER, "lb_raw_extended.pickle"),
        help="Path to the raw leaderboard extended data pickle file",
    )
    parser.add_argument(
        "--lb_raw_path",
        type=str,
        default=os.path.join(DATA_FOLDER, "lb_raw.pickle"),
        help="Path to the raw leaderboard data pickle file",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=os.path.join(DATA_FOLDER, "model_outputs.pickle"),
        help="Path to the model outputs pickle file",
    )
    args = parser.parse_args()

    print("Loading raw extended data (can take up to 4 minutes)...")
    extended_raw_df = load_pickle(args.lb_raw_extended_path)
    print("Loading raw data (can take up to 30 seconds)...")
    raw_df = load_pickle(args.lb_raw_path)

    extended_data, extended_raw_df = process_extended_data(
        extended_raw_df, max_models=MAX_MODELS
    )
    data, raw_df = process_data(extended_data, raw_df, max_models=MAX_MODELS)

    data_ordered = sort_models_by_dates(data, extended_raw_df, raw_df)

    dump_pickle(data_ordered, args.output_path)


if __name__ == "__main__":
    main()
