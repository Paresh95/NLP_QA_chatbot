from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import pipeline
from tqdm.auto import tqdm


def save_text2text_generation_artifacts(
    hugging_face_model_path: str, local_model_path: str, local_tokenizer_path: str
) -> None:
    with tqdm(total=2, desc="Loading Model and Tokenizer") as pbar:
        model = AutoModelForSeq2SeqLM.from_pretrained(hugging_face_model_path)
        pbar.update(1)
        tokenizer = AutoTokenizer.from_pretrained(hugging_face_model_path)
        pbar.update(1)
    model.save_pretrained(local_model_path)
    tokenizer.save_pretrained(local_tokenizer_path)
    return None


def load_text2text_generation_pipeline(
    local_model_path: str, local_tokenizer_path: str, device: str, max_new_tokens: int
) -> pipeline:
    with tqdm(total=2, desc="Loading Model and Tokenizer") as pbar:
        model = AutoModelForSeq2SeqLM.from_pretrained(local_model_path)
        pbar.update(1)
        tokenizer = AutoTokenizer.from_pretrained(local_tokenizer_path)
        pbar.update(1)
    pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map=device,
        model_kwargs={"max_new_tokens": max_new_tokens},
    )
    return pipe
