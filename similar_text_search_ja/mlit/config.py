from similar_text_search_ja import utils


def get_config():
    config_file = utils.get_dir().parent / "config.json"
    return utils.read_json_config(config_file)
