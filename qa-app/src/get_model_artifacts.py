from utils import read_yaml_config, save_text2text_generation_artifacts

if __name__ == "__main__":
    yaml_config = read_yaml_config("static.yaml")
    save_text2text_generation_artifacts(yaml_config)
    print("Files saved successfully")
