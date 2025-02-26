import yaml
import model.amdm_model as amdm_model
import model.amdm_text_model as amdm_text_model
import model.amdm_style_model as amdm_style_model

def build_model(model_config_file, dataset, device):
    model_config = load_model_file(model_config_file)
    model_name = model_config["model_name"]
    print("Building {} model".format(model_name))
    
    if (model_name == amdm_model.AMDM.NAME):
        model = amdm_model.AMDM(config=model_config, dataset=dataset,device=device)
    elif (model_name == amdm_text_model.AMDM.NAME):
        model = amdm_text_model.AMDM(config=model_config, dataset=dataset,device=device)
    elif (model_name == amdm_style_model.AMDM.NAME):
        model = amdm_style_model.AMDM(config=model_config, dataset=dataset,device=device)
    else:
        assert(False), "Unsupported model: {}".format(model_name)
        
    return model

def load_model_file(file):
    with open(file, "r") as stream:
        config = yaml.safe_load(stream)
    return config
