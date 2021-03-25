import os
import glob
import numpy as np

path_vic = '/home/mvpai1/projects/5Y-M1_2M1_3-multimodal_recognition'
path_img = '/HDD/5th_multimodal/Face'

# print(sorted(glob.glob(os.path.join(path_vic, '*'))))
vic = sorted(glob.glob(os.path.join(path_vic, '*')))
img = sorted(glob.glob(os.path.join(path_img, '*')))
del img[269:309]
print(len(vic), len(img))
# for i in range(len(vic)):
#     a = vic[i].split('/')[-1].split('d')[-1]
#     # if i % 100 == 99:
#     print(a)
all = list(zip(vic, img))
label = []
for i, (vic, img) in enumerate(all):
    # vic, img = all[i]
    vic_a = glob.glob(os.path.join(vic, '*'))
    img_a = glob.glob(os.path.join(img+'/1.6', '*'))
    for j, (vic_b, img_b) in enumerate(zip(vic_a, img_a)):
        vic_b = sorted(glob.glob(os.path.join(vic_b, '*')))[0]
        img_b = sorted(glob.glob(os.path.join(img_b, '*')))[0]
        # f = open('./train_multi.txt', 'a')
        # k = vic_b.replace('.wav', '.npy') + ' ' + img_b + ' ' + str(i) + '\n'
        k = str(i)
        label.append(k)
        # f.write(k)
# f.close()
np.save('./label.npy', np.asarray(label))
