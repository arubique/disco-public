# The code is adapted from the tinyBenchmarks repo: https://github.com/felipemaiapolo/efficbench
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
import pickle
import os
import argparse
import requests
import json
import sys


sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
# from utils import dict_to_h5
from utils import dump_pickle

sys.path.pop(0)

CACHE_DIR = "./cache_dir"


MODELS_NAMES = [
    "logicker/SkkuDataScienceGlobal-10.7b",
    "yujinpy/Sakura-SOLRCA-Math-Instruct-DPO-v2",
    "kyujinpy/Sakura-SOLRCA-Math-Instruct-DPO-v1",
    "fblgit/UNA-POLAR-10.7B-InstructMath-v2",
    "abacusai/MetaMath-bagel-34b-v0.2-c1500",
    "Q-bert/MetaMath-Cybertron-Starling",
    "meta-math/MetaMath-70B-V1.0",
    "meta-math/MetaMath-Mistral-7B",
    "SanjiWatsuki/neural-chat-7b-v3-3-wizardmath-dare-me",
    "ed001/datascience-coder-6.7b",
    "abacusai/Fewshot-Metamath-OrcaVicuna-Mistral",
    "meta-math/MetaMath-70B-V1.0",
    "WizardLM/WizardMath-70B-V1.0",
    "WizardLM/WizardMath-13B-V1.0",
    "WizardLM/WizardMath-7B-V1.0",
    "SanjiWatsuki/neural-chat-7b-v3-3-wizardmath-dare-me",
    "meta-math/MetaMath-Llemma-7B",
    "rameshm/llama-2-13b-mathgpt-v4",
    "rombodawg/Everyone-Coder-4x7b-Base",
    "qblocks/mistral_7b_DolphinCoder",
    "FelixChao/vicuna-33b-coder",
    "rombodawg/LosslessMegaCoder-llama2-13b-mini",
    "defog/sqlcoder-34b-alpha",
    "WizardLM/WizardCoder-Python-34B-V1.0",
    "OpenBuddy/openbuddy-deepseekcoder-33b-v16.1-32k",
    "mrm8488/llama-2-coder-7b",
    "jondurbin/airocoder-34b-2.1",
    "openchat/opencoderplus",
    "bigcode/starcoderplus",
    "qblocks/falcon_7b_DolphinCoder",
    "deepseek-ai/deepseek-coder-6.7b-instruct",
    "ed001/datascience-coder-6.7b",
    "glaiveai/glaive-coder-7b",
    "uukuguy/speechless-coder-ds-6.7b",
    "WizardLM/WizardCoder-Python-7B-V1.0",
    "WizardLM/WizardCoder-15B-V1.0",
    "LoupGarou/WizardCoder-Guanaco-15B-V1.1",
    "GeorgiaTechResearchInstitute/starcoder-gpteacher-code-instruct",
    "deepseek-ai/deepseek-coder-1.3b-instruct",
    "uukuguy/speechless-coder-ds-1.3b",
    "bigcode/tiny_starcoder_py",
    "Deci/DeciCoder-1b",
    "KevinNi/mistral-class-bio-tutor",
    "FelixChao/vicuna-7B-chemical",
    "AdaptLLM/finance-chat",
    "ceadar-ie/FinanceConnect-13B",
    "Harshvir/Llama-2-7B-physics",
    "FelixChao/vicuna-7B-physics",
    "lgaalves/gpt-2-xl_camel-ai-physics",
    "AdaptLLM/law-chat",
]


SCENARIOS = [
    "harness_arc_challenge_25",
    "harness_hellaswag_10",
    #'harness_hendrycksTest_5',
    "harness_truthfulqa_mc_0",
    "harness_winogrande_5",
    "harness_gsm8k_5",
]

MMLU_SUBSCENARIOS = [
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


LB_SAVEPATH = "data/leaderboard_fields_raw_22042025.pickle"

EXTRA_KEYS = [
    # 'full_prompt',
    "example",
    "predictions",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lb_type", type=str, default="mmlu_fields")
    parser.add_argument("--lb_savepath", type=str, default=LB_SAVEPATH)
    parser.add_argument("--save_only_once", action="store_true")
    args = parser.parse_args()

    lb_savepath = args.lb_savepath

    # if args.lb_type == "helm_lite":
    #     print("Downloading HELM Lite")
    #     download_helm_lite(args.lb_savepath)
    #     return
    if args.lb_type == "mmlu_fields":
        model_names = MODELS_NAMES
    else:
        df = pd.read_csv(
            # "./generating_data/download-openllmleaderboard/open-llm-leaderboard.csv"
            "benchmark_csvs/open-llm-leaderboard.csv"
        )
        df = df.loc[df.MMLU > 30]
        model_names = list(df.Model)
        print(len(model_names))  # 2288
        model_names = [model_names[i] for i in range(0, len(model_names), 5)]
        print(len(model_names))  # 458

    models = []
    for m in model_names:
        creator, model = tuple(m.split("/"))
        models.append(
            "open-llm-leaderboard/details_{:}__{:}".format(creator, model)
        )

    data = {}
    # for model in tqdm(models):
    #     data[model] = {}
    #     for s in SCENARIOS + MMLU_SUBSCENARIOS:
    #         data[model][s] = {}
    #         data[model][s]['correctness'] = None
    #         data[model][s]['dates'] = None

    os.makedirs(CACHE_DIR, exist_ok=True)
    skipped = 0
    log = []
    for model in tqdm(models):
        skipped_aux = 0

        for s in SCENARIOS + MMLU_SUBSCENARIOS:
            if "arc" in s:
                metric = "acc_norm"
            elif "hellaswag" in s:
                metric = "acc_norm"
            elif "truthfulqa" in s:
                metric = "mc2"
            else:
                metric = "acc"

            try:
                if model not in data:
                    data[
                        model
                    ] = (
                        {}
                    )  # TODO(Alex 24.04.2025): check that it works as intended
                if s not in data[model]:
                    data[model][s] = {}
                aux = load_dataset(model, s, cache_dir=CACHE_DIR)
                data[model][s]["dates"] = list(aux.keys())
                for extra_key in EXTRA_KEYS:
                    data[model][s][extra_key] = aux["latest"][extra_key]
                try:
                    # data[model][s]['dates'] = list(aux.keys())
                    data[model][s]["correctness"] = [
                        a[metric] for a in aux["latest"]["metrics"]
                    ]
                    # for extra_key in EXTRA_KEYS:
                    #     data[model][s][extra_key] = aux['latest'][extra_key]
                    print("\nOK {:} {:}\n".format(model, s))
                    log.append("\nOK {:} {:}\n".format(model, s))
                except Exception as e:
                    print(f"Error accessing dataset attribute: {e}")
                    try:
                        # aux = load_dataset(model, s, cache_dir=CACHE_DIR)
                        # data[model][s]['dates'] = list(aux.keys())
                        data[model][s]["correctness"] = aux["latest"][metric]
                        # for extra_key in EXTRA_KEYS:
                        #     data[model][s][extra_key] = aux['latest'][extra_key]
                        print("\nOK {:} {:}\n".format(model, s))
                        log.append("\nOK {:} {:}\n".format(model, s))
                    except Exception as e:
                        print(f"Error loading dataset for {model} and {s}: {e}")
                        # data[model][s] = None
                        # print("\nSKIP {:} {:}\n".format(model,s))
                        # skipped_aux+=1
                        # log.append("\nSKIP {:} {:}\n".format(model,s))
                        skipped_aux = skip_model(
                            data, model, s, skipped_aux, log
                        )

            except Exception as e:
                print(f"Error loading dataset for {model} and {s}: {e}")
                # data[model][s] = None
                # skipped_aux+=1
                # log.append("\nSKIP {:} {:}\n".format(model,s))
                skipped_aux = skip_model(data, model, s, skipped_aux, log)
                break
            # except Exception as e:
            # print(f"Error loading dataset for {model} and {s}: {e}")
            # try:
            #     aux = load_dataset(model, s, cache_dir=CACHE_DIR)
            #     data[model][s]['dates'] = list(aux.keys())
            #     data[model][s]['correctness'] = aux['latest'][metric]
            #     for extra_key in EXTRA_KEYS:
            #         data[model][s][extra_key] = aux['latest'][extra_key]
            #     print("\nOK {:} {:}\n".format(model,s))
            #     log.append("\nOK {:} {:}\n".format(model,s))
            # except Exception as e:
            #     print(f"Error loading dataset for {model} and {s}: {e}")
            #     # data[model][s] = None
            #     # print("\nSKIP {:} {:}\n".format(model,s))
            #     # skipped_aux+=1
            #     # log.append("\nSKIP {:} {:}\n".format(model,s))
            #     skipped_aux = skip_model(data, model, s, skipped_aux, log)

        if skipped_aux > 0:
            skipped += 1

        if not args.save_only_once:
            dump_pickle(data, lb_savepath)

        # dict_to_h5(data, lb_savepath)

        print("\nModels skipped so far: {:}\n".format(skipped))

    # dict_to_h5(data, lb_savepath)
    if args.save_only_once:
        dump_pickle(data, lb_savepath)


def skip_model(data, model, scenario_name, skipped_aux, log):
    data[model][scenario_name] = None
    print("\nSKIP {:} {:}\n".format(model, scenario_name))
    skipped_aux += 1
    log.append("\nSKIP {:} {:}\n".format(model, scenario_name))
    return skipped_aux


def download_helm_lite(lb_savepath):
    def get_json_from_url(url):
        try:
            response = requests.get(url)
            response.raise_for_status()  # Raise an exception for 4xx or 5xx status codes
            json_data = response.json()
            return json_data
        except requests.exceptions.RequestException as e:
            print(f"An error occurred: {e}")
            return None

    version_to_run = "v1.0.0"
    overwrite = False
    assert (
        lb_savepath is None or lb_savepath == "None"
    ), "lb_savepath must be None for helm_lite"
    df = pd.read_csv("./generating_data/download_helm/helm_lite.csv")
    tasks_list = list(df.Run)
    template_url = f"https://storage.googleapis.com/crfm-helm-public/lite/benchmark_output/runs/{version_to_run}"
    # save_dir = f"/llmthonskdir/felipe/helm/lite/{version_to_run}"
    save_dir = f"./data/downloaded_helm_lite/{version_to_run}"
    for tasks in [tasks_list]:
        for task in tqdm(tasks):
            cur_save_dir = f"{save_dir}/{task}"
            os.makedirs(cur_save_dir, exist_ok=True)

            for file_type in [
                "run_spec",
                "stats",
                "per_instance_stats",
                "instances",
                "scenario_state",
                "display_predictions",
                "display_requests",
                "scenario",
            ]:
                save_path = f"{cur_save_dir}/{file_type}.json"
                if os.path.exists(save_path) and not overwrite:
                    continue

                # https://storage.googleapis.com/crfm-helm-public/benchmark_output/runs/v0.2.2/babi_qa:task=15,model=AlephAlpha_luminous-base/scenario_state.json

                cur_url = f"{template_url}/{task}/{file_type}.json"
                json.dump(
                    get_json_from_url(cur_url), open(save_path, "w"), indent=2
                )


if __name__ == "__main__":
    main()
