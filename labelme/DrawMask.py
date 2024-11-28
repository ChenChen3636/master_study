import matplotlib.pyplot as plt
import matplotlib.patches as patches
import json
from PIL import Image
import numpy as np

#載入JSON檔案
with open('data_sacrum\sacrum_90.json') as f:
    mask_data = json.load(f)
    
# 定義標籤到顏色的映射
label_colors = {
    "S5": "red",
    "S4": "green",
    "S3": "blue",
    "S2": "yellow",
    "S1": "magenta",
    "L5": "cyan",
    "L4": "orange"
}
    
#載入圖片
image_path = 'data_sacrum\sacrum_90.png'
img = Image.open(image_path)

fig, ax = plt.subplots(1)
ax.imshow(img)

# 解析遮罩並繪製
for shape in mask_data["shapes"]:
    label = shape["label"]
    color = label_colors.get(label, "white")  # 使用白色作為未知標籤的默認顏色
    polygon = patches.Polygon(shape["points"], linewidth=2, edgecolor=color, facecolor='none', alpha=0.4)
    ax.add_patch(polygon)
    
plt.show()

