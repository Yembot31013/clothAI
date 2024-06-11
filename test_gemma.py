from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")
model = AutoModelForCausalLM.from_pretrained("google/gemma-2b")

input_text = "Write me a poem about Machine Learning."
input_ids = tokenizer(input_text, return_tensors="pt")

outputs = model.generate(**input_ids)
print(tokenizer.decode(outputs[0]))

# save for later in case we want to use the model again offline and don't want to download it again from the Hugging Face model hub and also convert it to tensorflow lite format for low device usage
model.save_pretrained("google/gemma-2b")
tokenizer.save_pretrained("google/gemma-2b")

# convert to tensorflow lite
from transformers import TFAutoModelForCausalLM, AutoConfig

config = AutoConfig.from_pretrained("google/gemma-2b")
model = TFAutoModelForCausalLM.from_pretrained("google/gemma-2b", config=config)
model.save_pretrained("google/gemma-2b_tflite")
tokenizer.save_pretrained("google/gemma-2b_tflite")
