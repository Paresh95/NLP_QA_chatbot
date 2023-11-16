import yaml
from typing import Dict
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import pipeline
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from tqdm.auto import tqdm


def read_yaml_config(path: str) -> Dict:
    try:
        with open(path, "r") as f:
            return yaml.safe_load(f.read())
    except yaml.YAMLError as e:
        print(f"Error parsing static.yaml: {e}")
    except Exception as e:
        print(f"Unexpected error reading static.yaml: {e}")
    return {}


def save_text2text_generation_artifacts(yaml_config: dict) -> None:
    model_id = yaml_config["hugging_face_model_path"]
    with tqdm(total=2, desc="Loading Model and Tokenizer") as pbar:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
        pbar.update(1)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        pbar.update(1)
    model.save_pretrained(yaml_config["local_model_path"])
    tokenizer.save_pretrained(yaml_config["local_tokenizer_path"])
    return None


def load_text2text_generation_pipeline(yaml_config: dict) -> HuggingFacePipeline:
    with tqdm(total=2, desc="Loading Model and Tokenizer") as pbar:
        model = AutoModelForSeq2SeqLM.from_pretrained(yaml_config["local_model_path"])
        pbar.update(1)
        tokenizer = AutoTokenizer.from_pretrained(yaml_config["local_tokenizer_path"])
        pbar.update(1)
    pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map=yaml_config["compute"],
        model_kwargs={"max_new_tokens": yaml_config["max_new_tokens"]},
    )
    llm = HuggingFacePipeline(pipeline=pipe)
    return llm
