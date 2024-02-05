# from pathlib import Path

# cache_directory = Path("~/.cache/huggingface/transformers").expanduser()

# # List all directories (models) in the cache directory
# downloaded_models = [model for model in cache_directory.glob("*") if model.is_dir()]

# # Print the names of the downloaded models
# for model in downloaded_models:
#     print(model.name)


from pathlib import Path

model_directory = Path("~/.cache/huggingface/transformers/TinyLlama/TinyLlama-1.1B-Chat-v1.0").expanduser()

# Delete the model directory
model_directory.rmdir()
