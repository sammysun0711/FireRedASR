import pandas as pd
import ast
from evaluate import load
import os

def extract_path_from_url(url: str) -> str:
    """
    Extract path starting from 'demo%2Faudio/' from the given URL.
    """
    # Find 'demo%2Faudio' in the URL
    marker = "demo%2Faudio"
    start_index = url.find(marker)
    if start_index == -1:
        return ""  # marker not found
    
    # Cut from marker to the end
    path_with_params = url[start_index:]
    
    # If you want to remove query parameters (?OSSAccessKeyId=...)
    path = path_with_params.split("?")[0]
    
    return path

def filter_short_paths_by_language(df: pd.DataFrame, language_code: str):
    """
    Filters DataFrame for given language_code (e.g., 'zh') and returns short paths.
    """
    # Create 'short_path' column
    df["short_path"] = df["http_url"].apply(extract_path_from_url)
    
    # Filter rows
    filtered_df = df[df["segment_language"] == language_code]
    
    # Return the list of short paths
    return filtered_df["short_path"].tolist()


def load_fireredasr_results(path, key_field="uttid"):
    data_dict = {}
    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()
        try:
            data_list = ast.literal_eval(content)  # Converts Python object string → list/dict
            for entry in data_list:
                data_dict[entry[key_field]] = entry["text"]
                data_dict[entry["wav"]] = entry["wav"]
        except Exception as e:
            print("Error parsing log:", e)
    return data_dict

import json

def wav_text_dict(json_file):
    """
    Reads the JSON file and returns a dict {wav: text}
    """
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)  # list of dicts

    # Build dictionary
    wav_text_map = {item["wav"]: item["text"] for item in data if "wav" in item and "text" in item}
    
    # Print the dictionary
    for wav, text in wav_text_map.items():
        print(f"{wav} -> {text}")

    return wav_text_map


if __name__ == "__main__":
    language_code = "zh"
    # Input and output file names
    input_csv = "/models/ali_audios/1000case_result.csv"   # replace with your CSV file path
    #output_csv = "output.csv" # file to save extracted data

    # Read CSV into DataFrame
    df = pd.read_csv(input_csv)

    # Get short paths for Chinese ('zh')
    zh_short_paths = filter_short_paths_by_language(df, language_code)

    print("Short paths with language == zh:")

    for path in zh_short_paths:
        print(path)
    
    # Example usage
    native_deocder_updated_beam_search = wav_text_dict("ATTENTION_BACKEND_NATIVE_FP32_bs_32_output.json")
    #sdpa_deocder_updated_beam_search = wav_text_dict("ATTENTION_BACKEND_SDPA_FP16_bs_32_output.json")
    sdpa_deocder_updated_beam_search = wav_text_dict("ATTENTION_BACKEND_FLASH_ATTN_FP16_bs_32_output_c052a4f.json")
    
    count = 0
    
    filtered_native_deocder_updated_beam_search = []
    filtered_sdpa_deocder_updated_beam_search = []
    for (native_wav_path, native_res), (sdpa_wav_path, sdpa_res) in zip(native_deocder_updated_beam_search.items(), sdpa_deocder_updated_beam_search.items()):
        if native_wav_path == sdpa_wav_path:
            if os.path.basename(native_wav_path) in zh_short_paths and os.path.basename(sdpa_wav_path) in zh_short_paths:
                filtered_native_deocder_updated_beam_search.append(native_res)
                filtered_sdpa_deocder_updated_beam_search.append(sdpa_res)
                if native_res != sdpa_res:
                    count+=1
                    #print("native_res: ", native_res)
                    #print("sdpa_res:  ", sdpa_res)
    total_len = len(filtered_native_deocder_updated_beam_search)

    from whisper_normalizer.english import EnglishTextNormalizer
    from whisper_normalizer.basic import BasicTextNormalizer
    from zhconv import convert
    # https://github.com/open-speech/cn-text-normalizer
    import cntn
    def ZhTextNormalizer(x):
        x = convert(x, "zh-cn")
        x = cntn.w2s(x)
        return x

    text_normalizers = {
        "zh":ZhTextNormalizer,
        "en":EnglishTextNormalizer(),
        "basic" : BasicTextNormalizer()
    }

    references_before_normalize = filtered_native_deocder_updated_beam_search
    predictions_before_normalize = filtered_sdpa_deocder_updated_beam_search
    predictions = []
    references = []
    for ref, pred in zip(references_before_normalize, predictions_before_normalize):
        language_code = "zh"
        norm = text_normalizers[language_code]
        pred = norm(pred)
        ref = norm(ref)
        predictions.append(pred)
        references.append(ref)

    # print both WER & CER
    # https://huggingface.co/learn/audio-course/zh-CN/chapter5/evaluation
    import evaluate
    wer = evaluate.load("wer")
    cer = evaluate.load("cer")
    print(f"total filtered results: ", total_len)
    print(f"wer_score: {wer.compute(predictions=predictions, references=references) * 100:.2f} %")
    print(f"cer_score: {cer.compute(predictions=predictions, references=references) * 100:.2f} %")
    percentage = count * 1e2 / total_len
    print(f"Mismatch percentage: {percentage:.2f} %")


