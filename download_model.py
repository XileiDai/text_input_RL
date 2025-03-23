from transformers import AutoModel, AutoTokenizer

model_name = "gpt2"
local_dir = "local_models/gpt2"

# # Download and save
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModel.from_pretrained(model_name)

# tokenizer.save_pretrained(local_dir)
# model.save_pretrained(local_dir)

tokenizer = AutoTokenizer.from_pretrained(local_dir)
model = AutoModel.from_pretrained(local_dir)