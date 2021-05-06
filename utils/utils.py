import cv2
import os
from align.mtcnn import mtcnn_img
import numpy as np
import glob
import operator

def extract_image(path, name, name_class):
    file_name = name_class.split('@')[0]
    path_target = os.path.join(path, ('data/test_data/' + file_name + '/' + str(name)))
    image = extract_Gfeature(path_target)
    image = prepare_image(image)
    return image.reshape(1,112,112,3)

def save_feature(embedding, path, name_class, re_register):
    output_path = os.path.join(path, ('gallery/img_gallery/{}/'.format(name_class)))
    if re_register == True:
        if not os.path.exists(output_path):
            a = 'Re_register is True, but speaker is not registered before. So we register speaker. Speaker ID : {}'.format(name_class)
        else:
            a = 'Re_register is True, so we register speaker again. Speaker ID : {}'.format(name_class)
    else:
        a = 'We register speaker. Speaker ID : {}'.format(name_class)

    try:
        if not os.path.exists(output_path):
            os.makedirs(output_path)
    except OSError:
        print('Error: Creating directory of data')

    np.savetxt(os.path.join(output_path, 'feature_0.txt'), embedding)

    return a

def prepare_image(img):
    img = img - 127.5
    img = img * 0.0078125
    return img

def extract_Gfeature(path):
    cam = cv2.VideoCapture(path)
    currentframe = 0
    class_gallery = []

    while(True):
        ret, frame = cam.read()
        frame = cv2.flip(cv2.transpose(frame, (1, 2, 0)), 0)
        if ret:
            if currentframe == 29:
                output = mtcnn_img(frame)
                class_gallery.append(output)
                break
            else :
                currentframe += 1
        else:
            break

    return class_gallery[0]

def identify_final(emb_target, gallery_emb, img_list):
    final_sim = dict()
    for i in range(len(gallery_emb)):
        cos_sim_final=[]
        cos_sim = np.dot(emb_target, np.transpose(gallery_emb[i])) / (np.linalg.norm(emb_target) * np.linalg.norm(gallery_emb[i]) + 1e-5)
        cos_sim_final.append(cos_sim)
        final_sim[img_list[i].split(os.sep)[-2]] = np.mean(cos_sim_final)

    max_5_speakers_final = dict(sorted(final_sim.items(), key=operator.itemgetter(1), reverse=True)[:5])
    sorted_speakers_final = sorted(max_5_speakers_final.items(), key=lambda item: item[1], reverse=True)

    top_value = []
    id_top = []
    for i in range(len(sorted_speakers_final)):
        top_value.append(sorted_speakers_final[i][1])
        id_top.append(sorted_speakers_final[i][0])

    return top_value, id_top

def identify_target(ori_path, test_feature):
    feat_img_list = []
    img_list = glob.glob(os.path.join(ori_path,'gallery/img_gallery/*'))
    img_sim = dict()
    for img in img_list:
        feat_img_path = sorted(glob.glob(img + os.sep + 'feature_0.txt'))[-1]
        feat_img_list.append(feat_img_path)
    gallery_emb_img = []
    for img_feat in feat_img_list:
        img_emb = np.loadtxt(img_feat)
        gallery_emb_img.append(np.reshape(img_emb, (1,img_emb.shape[0])))
        cos_sim_img = []
        cos_sim = np.dot(test_feature, img_emb) / (np.linalg.norm(test_feature) * np.linalg.norm(img_emb) + 1e-5)
        cos_sim_img.append(cos_sim)
        img_sim[img_feat.split(os.sep)[-2]] = np.mean(cos_sim_img)

    max_5_speakers = dict(sorted(img_sim.items(), key=operator.itemgetter(1), reverse=True)[:5])
    sorted_speakers = sorted(max_5_speakers.items(), key=lambda item: item[1], reverse=True)

    top_value = []
    id_top = []
    for i in range(len(sorted_speakers)):
        top_value.append(sorted_speakers[i][1])
        id_top.append(sorted_speakers[i][0])
    top_value = np.float64(top_value)

    return top_value, id_top, gallery_emb_img, feat_img_list, test_feature

def expand_dim(x):
    x = np.reshape(x, (1,1,1,x.shape[1]))
    return x

def gallery(path, target):
    gallery_path = glob.glob(os.path.join(path, 'gallery/img_gallery/*'))
    issame = []
    for i, gallery_name in enumerate(gallery_path):
        new_gallery_name = gallery_name.strip().split('/')[-1]
        if str(new_gallery_name) == str(target):
            answer = 1
        else:
            answer = 0

        issame.append(answer)

    if issame.count(1)>0:
        output = True
    else:
        output = False
    return output
