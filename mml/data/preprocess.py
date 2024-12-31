import os
import pickle
from PIL import Image
from tqdm import tqdm
import torch
import clip
import json
from transformers import CLIPModel, CLIPProcessor


device = "cuda" if torch.cuda.is_available() else "cpu"
# 加载预训练的 CLIP 模型和处理器
# 不能用这个，这个的hidden_size只有512
# model, preprocess = clip.load("ViT-B/32", device=device, download_root="./clip")

model = 'openai/clip-vit-base-patch32'
# model = 'openai/clip-vit-large-patch14'
preprocessor = CLIPProcessor.from_pretrained(model)
model = CLIPModel.from_pretrained(model).vision_model.to(device)
print(f"model embedding size: {model.config.hidden_size}")


# 加载并预处理train_caption
with open("./coco/annotations/train_caption.json", "r") as f:
    train_caption_list = json.load(f)
train_captions = {}
for train_caption in train_caption_list:
    train_captions[int(train_caption["image_id"])] = train_caption["caption"]

# print(list(train_captions.keys())[:100])

# 图片目录
image_dir = "./coco/train2014/train2014"
# image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(".jpg")]
image_files = [f for f in os.listdir(image_dir) if f.endswith(".jpg")]

# 存储图像数据的列表
image_names = []
image_embeddings = []
image_captions = []

# 遍历图片目录，加载每张图片并进行预处理
for image_file in tqdm(image_files):
    # print(f"image_file: {image_file}")
    image_names.append(image_file)

    image_file = os.path.join(image_dir, image_file)
    image = Image.open(image_file)

    # image_input = preprocess(image).unsqueeze(0).to(device)
    # # 使用 CLIP 模型提取图像嵌入
    # with torch.no_grad():
    #     image_features = model.encode_image(image_input)

    image = preprocessor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        image_features = model(**image).pooler_output

    image_embeddings.append(image_features.cpu().numpy())

    # 获取图像的caption
    image_id = int(os.path.basename(image_file)[-10:-4])
    image_caption = train_captions[image_id]

    image_captions.append(image_caption)

    # print(image_features.cpu().numpy())
    # print(image_features.cpu().numpy().shape)  # (1, 768)
    # print(image_id)
    # print(image_caption)
    # break

# 将图像数据保存为 pickle 文件
image_datas = {}
image_datas["image_names"] = image_names
image_datas["image_embeddings"] = image_embeddings
image_datas["image_captions"] = image_captions

with open("image_datas_base32.pkl", "wb") as f:
    pickle.dump(image_datas, f)

