from vlm_adapters.llava_med_adapter import LLavaMedAdapter

img_path = "data/images/cyanosis_Image_1.jpg"
prompt = "Describe any visible abnormalities in this medical image."

adapter = LLavaMedAdapter()
caption = adapter.generate_caption(img_path, prompt)
print("LLaVA-Med:", caption)
