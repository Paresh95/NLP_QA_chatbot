from utils.general_utils import read_yaml_config
from utils.model_utils import save_text2text_generation_artifacts

if __name__ == "__main__":
    yaml_config = read_yaml_config("parameters.yaml")
    hugging_face_model_path = yaml_config["hugging_face_model_path"]
    local_model_path = yaml_config["local_model_path"]
    local_tokenizer_path = yaml_config["local_tokenizer_path"]
    save_text2text_generation_artifacts(
        hugging_face_model_path, local_model_path, local_tokenizer_path
    )
    print("Files saved successfully")
