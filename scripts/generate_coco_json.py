# 文件路径建议：scripts/generate_coco_json.py
import json

caption_json_path = "./data/coco/annotations/captions_train2017.json"
output_path = "./data/coco/coco_train_captions.json"

with open(caption_json_path, "r") as f:
    coco = json.load(f)

id2filename = {img["id"]: img["file_name"] for img in coco["images"]}
annotations = coco["annotations"]

pairs = []
for ann in annotations:
    img_id = ann["image_id"]
    caption = ann["caption"]
    file_name = id2filename.get(img_id)
    if file_name:
        pairs.append({
            "image": file_name,
            "caption": caption
        })

with open(output_path, "w") as f:
    json.dump(pairs, f)

print(f"✅ 成功生成图文对：共 {len(pairs)} 条，已保存至 {output_path}")
