import torch
import torch.nn.functional as F

import json
from PIL import Image, ImageDraw

def read_coco_annotation(file_path):
    with open(file_path, 'r') as f:
        coco_data = json.load(f)
    # 打印数据集的基本信息
    # print("Info:", coco_data.get("info"))
    # print("Licenses:", coco_data.get("licenses"))
    # print("Number of images:", len(coco_data.get("images")))
    data = coco_data.get("annotations")
    print(data[0]['category_id'])
    # print("Number of categories:", len(coco_data.get("categories")))

# # # 示例文件路径
# file_path = 'D:/PyCharm项目/CocoDataSet/annotations/instances_train2017.json'
# read_coco_annotation(file_path)

import xml.etree.ElementTree as ET

# 解析XML文件
tree = ET.parse('D:/PyCharm项目/DataSets/MaskDetect/annotations/maksssksksss0.xml')

# 获取根元素
root = tree.getroot()

# 打印根元素的标签
print(f"根元素标签: {root.tag}")

# 遍历根元素的子元素
for child in root:
    print(f"子元素标签: {child.tag}, 子元素属性: {child.attrib}")
    # 遍历子元素的文本内容
    for sub_child in child:
        print(f"  子子元素标签: {sub_child.tag}, 子子元素文本: {sub_child.text}")


