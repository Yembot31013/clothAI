from datasets import load_dataset

dataset = load_dataset("wikimedia/wit_base")

dataset.save_to_disk("wit_base")