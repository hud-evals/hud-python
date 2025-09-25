from transformers import AutoModelForImageTextToText

model = AutoModelForImageTextToText.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")

print(model.model)

for layer in model.model.visual.blocks:
    print(layer)

# for layer in model.language_model.layers:
#     print(layer)