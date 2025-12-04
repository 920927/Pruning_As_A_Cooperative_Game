from models.short_hf import ShortHFModel

def get_model(args):
    model_name = args.model_path.split('/')[-1]
    short_model = ShortHFModel(model_path=args.model_path, layers_path="model.layers")
    tokenizer = short_model.tokenizer
    return short_model, tokenizer