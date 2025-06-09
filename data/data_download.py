'''
Author: Tageshi fuzongjing@foxmail.com
Date: 2025-06-09 14:36:14
LastEditors: Tageshi fuzongjing@foxmail.com
LastEditTime: 2025-06-09 14:36:15
FilePath: /CA-ViLT/data_download.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import kagglehub

# Download latest version
path = kagglehub.dataset_download("awsaf49/coco-2017-dataset")

print("Path to dataset files:", path)