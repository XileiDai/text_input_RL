from transformers import AutoModel, AutoTokenizer

model_name = "gpt2"
cache_dir = "Pretrained_model"

# Downloads and caches model/tokenizer locally
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir)
