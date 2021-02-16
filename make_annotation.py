import os
import numpy

"""
This script is for converting the original annotation of INRIAPerson dataset into
that aligned with https://github.com/rafaelpadilla/Object-Detection-Metrics#important-definitions
for later testing AP
"""

ori_anno_folder = "E:/datasets/INRIAPerson/Test/annotations"
new_anno_folder = "E:/datasets/INRIAPerson/Test/annotation_new"

for root, _, files in os.walk(ori_anno_folder):
    for name in files:
        file = os.path.join(root, name)
        with open(file, 'r') as f:
            lines = f.readlines()
            co_num = 0
            for line in lines:
                if line == '':
                    continue
                if line[0] == '#':
                    continue
                if "Bounding" in line:
                    co_num += 1
                    gt = line.split(':')[1]
                    gt = gt.split(' ')
                    x_1 = ""
                    y_1 = ""
                    x_2 = ""
                    y_2 = ""
                    i = 1
                    while gt[1][i] != ',':
                        x_1 += gt[1][i]
                        i += 1
                    i = 0
                    while gt[2][i] != ')':
                        y_1 += gt[2][i]
                        i += 1
                    i = 1
                    while gt[4][i] != ',':
                        x_2 += gt[4][i]
                        i += 1
                    i = 0
                    while gt[5][i] != ')':
                        y_2 += gt[5][i]
                        i += 1
                    with open(new_anno_folder+'/'+name, mode='a') as nf:
                        if co_num == 1:
                            nf.write("person "+x_1+" "+y_1+" "+x_2+" "+y_2)
                        else:
                            nf.write("\nperson "+x_1+" "+y_1+" "+x_2+" "+y_2)

