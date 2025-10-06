from tqdm import tqdm
from stnd.utility.utils import parse_list_from_string
import numpy as np

# import torch
# from transformers import AutoTokenizer, AutoModel

MAX_NUM_ANSWERS = 31  # 5 for arc_harness_25, 31 for truth_harness_mc
KEYS_TO_ADD = ["correctness", "predictions"]
SUB_TO_SKIP = ["harness_gsm8k_5"]


# def get_embedder(model_name="BAAI/bge-large-en-v1.5"):
#     # Initialize BGE model and tokenizer
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     model = AutoModel.from_pretrained(model_name)

#     # Helper function to get embeddings
#     def get_embeddings_bge(texts):
#         # Add special tokens for BGE model
#         processed_texts = []
#         for text in texts:
#             if not isinstance(text, str):
#                 raise ValueError(f"Text is not a string: {text}")
#             processed_texts.append(f"Represent this sentence: {text}")
#         texts = processed_texts

#         # Tokenize and encode
#         encoded = tokenizer(
#             texts, padding=True, truncation=True, return_tensors="pt"
#         )

#         # Get embeddings
#         with torch.no_grad():
#             outputs = model(**encoded)
#             embeddings = outputs.last_hidden_state[
#                 :, 0
#             ]  # Use CLS token embedding
#             # Normalize embeddings
#             embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

#         return embeddings.numpy()

#     return get_embeddings_bge


def pad_predictions(predictions, max_num_answers=MAX_NUM_ANSWERS):
    if len(predictions) < max_num_answers:
        predictions.extend(
            [-float("inf")] * (max_num_answers - len(predictions))
        )
    return predictions


def parse_df_with_results(
    df,
    models,
    order,
    sub_to_skip=SUB_TO_SKIP,
    max_num_answers=MAX_NUM_ANSWERS,
    keys_to_add=KEYS_TO_ADD,
    max_models=None,
):
    if order is None:
        models = models
    else:
        models = [models[o] for o in order]

    if max_models is not None:
        models = models[:max_models]

    data = {}
    data["data"] = {}
    data["models"] = models
    max_answers_dict = {}

    for sub in tqdm(df[list(df.keys())[0]].keys()):
        if sub in sub_to_skip:
            continue
        max_answers_dict[sub] = 0
        data["data"][sub] = {}
        # data['data'][sub]['correctness'] = []

        for key in keys_to_add:
            data["data"][sub][key] = []

        for model in models:
            # data['data'][sub]['correctness'].append(df[model][sub]['correctness'])
            for key in keys_to_add:
                if key not in df[model][sub].keys():
                    value_to_add = None
                else:
                    value_to_add = df[model][sub][key]
                if key == "predictions":
                    new_value_to_add = []
                    for model_preds in value_to_add:
                        if isinstance(model_preds, str):
                            parsed_model_preds = parse_list_from_string(
                                model_preds, list_separators=[","]
                            )
                            # new_value_to_add.append(pad_predictions(parsed_model_preds))

                        else:
                            # new_value_to_add.append(pad_predictions(model_preds))
                            parsed_model_preds = model_preds

                        new_value_to_add.append(
                            pad_predictions(parsed_model_preds)
                        )

                        assert (
                            len(new_value_to_add[-1]) == max_num_answers
                        ), f"Num answers not equal to {max_num_answers}: {len(new_value_to_add[-1])} for sub: {sub} and key: {key}"
                        if max_answers_dict[sub] < len(new_value_to_add[-1]):
                            max_answers_dict[sub] = len(new_value_to_add[-1])

                    value_to_add = new_value_to_add
                data["data"][sub][key].append(value_to_add)

        # data['data'][sub]['correctness'] = np.array(data['data'][sub]['correctness']).T.astype(float)

        for key in keys_to_add:
            if key == "predictions":
                data["data"][sub][key] = (
                    np.array(data["data"][sub][key])
                    .transpose(1, 0, 2)
                    .astype(float)
                )
                # Note: Shape after transpose: (num_samples, num_models, num_answers)
            else:
                data["data"][sub][key] = np.array(
                    data["data"][sub][key]
                ).T.astype(float)
                # Note: Shape after transpose: (num_samples, num_models)

    return data, max_answers_dict

    # print(max(max_answers_dict.values()))


# def compare_dicts_with_arrays(d1, d2, prefix=""):
#     if d1.keys() != d2.keys():
#         return False
#     for k in d1.keys():
#         full_path = prefix + f"/{k}"
#         error_str = f"not equal for {full_path}"
#         if isinstance(d1[k], dict):
#             if not compare_dicts_with_arrays(d1[k], d2[k], prefix=full_path):
#                 print(error_str)
#                 return False
#         elif isinstance(d1[k], np.ndarray):
#             if not (
#                 np.array_equal(d1[k], d2[k])
#                 or (np.isnan(d1[k]).all() and np.isnan(d2[k]).all())
#             ):
#                 print(error_str)
#                 return False
#         elif torch.is_tensor(d1[k]):
#             if not (
#                 torch.equal(d1[k], d2[k])
#                 or (d1[k].isnan().all() and d2[k].isnan().all())
#             ):
#                 print(error_str)
#                 return False
#         else:
#             if d1[k] != d2[k]:
#                 print(error_str)
#                 return False
#     return True


def merge_methods(table_avg_base, table_avg_to_merge):
    if table_avg_base is None:
        return table_avg_to_merge
    for bench in table_avg_base.keys():
        for split in table_avg_base[bench].keys():
            for method in table_avg_to_merge[bench][split].keys():
                if method not in table_avg_base[bench][split].keys():
                    table_avg_base[bench][split][method] = table_avg_to_merge[
                        bench
                    ][split][method]
    return table_avg_base
