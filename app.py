#-*- coding:utf-8 -*-
from flask import Flask
from flask import json
from flask import request
from utils.utils import extract_image, save_feature, identify_final, identify_target, expand_dim,\
    gallery
from collections import OrderedDict
from utils.voice_speaker_recognition import voice_identification, voice_gallery, voice_registration
from keras_layer_normalization import LayerNormalization
from keras import backend as K
from keras.engine.topology import Layer
import tensorflow as tf
import numpy as np
from keras.models import load_model
import pandas as pd
import os

app = Flask(__name__)
os.environ['CUDA_VISIBLE_DEVICES']='0'
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
config = tf.ConfigProto(device_count = {'GPU': 1})
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.2

def str_to_bool(s):
    if s == 'True':
        return True
    elif s == 'False':
        return False
    else:
        raise ValueError


def new_softmax(a) :
    c = np.max(a)
    exp_a = np.exp(a-c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    y = y.tolist()
    return y


class AMSoftmax(Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(AMSoftmax, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) >= 2

        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[1], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)
        super(AMSoftmax, self).build(input_shape)

    def call(self, inputs):
        inputs = tf.nn.l2_normalize(inputs, dim=1)  # input_l2norm
        self.kernel = tf.nn.l2_normalize(self.kernel, dim=0)  # W_l2norm

        cosine = K.dot(inputs, self.kernel)  # cos = input_l2norm * W_l2norm
        return cosine

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)


def amsoftmax_loss(y_true, y_pred):
    scale = 15.0
    margin = 0.3

    label = tf.reshape(tf.argmax(y_true, axis=-1), shape=(-1, 1))
    label = tf.cast(label, dtype=tf.int32)  # y
    batch_range = tf.reshape(tf.range(tf.shape(y_pred)[0]), shape=(-1, 1))
    indices_of_groundtruth = tf.concat([batch_range, tf.reshape(label, shape=(-1, 1))], axis=1)
    groundtruth_score = tf.gather_nd(y_pred, indices_of_groundtruth)

    m = tf.constant(margin, name='m')
    s = tf.constant(scale, name='s')

    added_margin = tf.cast(tf.greater(groundtruth_score, m), dtype=tf.float32) * m
    added_margin = tf.reshape(added_margin, shape=(-1, 1))
    added_embeddingFeature = tf.subtract(y_pred, y_true * added_margin) * s

    cross_ent = tf.nn.softmax_cross_entropy_with_logits(labels=y_true,
                                                        logits=added_embeddingFeature)
    loss = tf.reduce_mean(cross_ent)
    return loss


def final_result(cos_sim_image, id_top5_image):
    id_prob_image = new_softmax(cos_sim_image)
    max_index_image = id_prob_image.index(max(id_prob_image))
    image_identification_result = id_top5_image[max_index_image]

    return image_identification_result


class ImportGraph():
    def __init__(self, model_pth, meta, ckpt):
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph, config=tf.ConfigProto(gpu_options=gpu_options))
        with self.graph.as_default():
            saver = tf.train.import_meta_graph(os.path.join(model_pth, meta))
            saver.restore(self.sess, os.path.join(model_pth, ckpt))
            self.inputs_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            self.embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")

    def run(self, img_data):
        feed_dict = {self.inputs_placeholder: img_data}
        return self.sess.run(self.embeddings, feed_dict=feed_dict)


class ImportMultiGraph():
    def __init__(self, model_pth, meta, ckpt):
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph, config=tf.ConfigProto(gpu_options=gpu_options))
        with self.graph.as_default():
            saver = tf.train.import_meta_graph(os.path.join(model_pth, meta))
            saver.restore(self.sess, os.path.join(model_pth, ckpt))
            self.inputs_img_placeholder = tf.get_default_graph().get_tensor_by_name("input_img:0")
            self.inputs_vic_placeholder = tf.get_default_graph().get_tensor_by_name("input_vic:0")
            self.embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")

    def run(self, img_data, vic_data):
        feed_dict = {self.inputs_img_placeholder: img_data, self.inputs_vic_placeholder: vic_data}
        return self.sess.run(self.embeddings, feed_dict=feed_dict)


class ImportVoiceGraph():
    def __init__(self, model_path):

        self.graph_vic = tf.Graph()

        with self.graph_vic.as_default():
            self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options), graph=
                self.graph_vic)
            self.sess.run(tf.global_variables_initializer())
            K.set_session(self.sess)
            self.speaker_model = load_model(model_path,
                                            custom_objects={'LayerNormalization':
                                                                LayerNormalization,
                                                            'amsoftmax_loss': amsoftmax_loss})
            self.speaker_model.summary()
            embedding_layer = self.speaker_model.layers[21]
            self.func = K.function([self.speaker_model.layers[0].input], [embedding_layer.output])
            self.inputs_placeholder = tf.get_default_graph().get_tensor_by_name(self.speaker_model
                                                                                .layers[0].input.
                                                                                name)
            self.embeddings = tf.get_default_graph().get_tensor_by_name(self.speaker_model.
                                                                        layers[21].output.name)


    def run(self, data):
        embedding = self.sess.run(self.embeddings, feed_dict={self.inputs_placeholder: data})
        return embedding


@app.route("/Identification_Request", methods=["POST"])
def predict():
    if request.method == "POST":
        test = request.form['test']
        test = str_to_bool(test)
        test_list = request.form['test_list']
        path = request.form['path_dir']
        name = request.form['name']
        a = 'data/test_data/' + name.split('@')[0]
        name_class = name.split('-')[0:-1]
        re_register = request.form['re_register']
        re_register = str_to_bool(re_register)

        name_class = ''.join(name_class)
        result = OrderedDict()
        path_target = os.path.join(path, a)
        path_target = os.path.join(path_target, name)

        model_img = ImportGraph('./best_model/pretrained_img', 'MobileFaceNet_iter_588000\
        .ckpt.meta', 'MobileFaceNet_iter_588000.ckpt')
        model_multi = ImportMultiGraph('./best_model/pretrained_multi', 'MobileFaceNet_iter\
        _300000.ckpt.meta', 'MobileFaceNet_iter_300000.ckpt')
        model_voice = ImportVoiceGraph('./best_model/pretrained_vic/'
                                       'model_2sec_64.h5')

        if test is True:
            f = open(test_list, 'r')
            lines = f.readlines()

            total_target = len(lines)
            hit_image = 0
            hit_speech = 0
            hit_multi = 0

            csv_list = dict()
            csv_list['test_path'] = []
            csv_list['label'] = []
            csv_list['pred'] = []
            csv_list['result'] = []

            f_multi = open('output/False_multi.txt', 'w')
            f_multi.write('test file' + '\t' + 'prediction result' + '\n')
            f_multi.close()
            for line in lines:
                csv_list['test_path'].append(os.path.basename(line).replace('\n',''))
                name = line.split()[0]
                name = name.split('/')[-1]
                name_class = name.split('-')[0:-1]
                name_class = ''.join(name_class)

                # Image Model
                img = extract_image(path, name, name_class)
                emb_array_img = model_img.run(img)
                cos_sim_image, id_top5_image, gallery_emb_img, img_list, target_emb_img =\
                    identify_target(path, emb_array_img)
                image_identification_result = final_result(cos_sim_image, id_top5_image)

                # Voice Model
                path_target = os.path.join(path, 'data/test_data', name_class.split('@')[0], name)
                cos_sim_voice, id_top5_voice, gallery_emb_audio, audio_list, target_emb_audio =\
                    voice_identification(model_voice, path_target)
                voice_identification_result = id_top5_voice[np.argmax(cos_sim_voice)]

                # Multi_Modal_Model
                emb_final_target = model_multi.run(expand_dim(target_emb_img), expand_dim(
                    target_emb_audio))
                assert len(gallery_emb_img) == len(gallery_emb_audio), \
                    "Check gallery of Image/Audio File"
                gallery_emb_final = []
                for i in range(len(gallery_emb_img)):
                    emb_multi = model_multi.run(expand_dim(gallery_emb_img[i]), expand_dim(
                        gallery_emb_audio[i]))
                    gallery_emb_final.append(emb_multi)

                cos_sim_final, id_top5_final = identify_final(emb_final_target, gallery_emb_final
                                                              , img_list)
                final_identification_result = id_top5_final[np.argmax(cos_sim_final)]

                csv_list['label'].append(name_class)
                csv_list['pred'].append(final_identification_result)

                if image_identification_result == name_class:
                    hit_image += 1
                else:
                    f = open('output/False_image.txt', 'a')
                    a = image_identification_result + ' ' + name_class
                    f.write(a)
                    f.close()

                if voice_identification_result == name_class:
                    hit_speech += 1
                else:
                    pass

                if final_identification_result == name_class:
                    hit_multi += 1
                    csv_list['result'].append(1)
                else:
                    csv_list['result'].append(0)
                    f_multi = open('output/False_multi.txt', 'a')
                    f_multi.write(os.path.basename(line).replace('\n', '') + '\t' +
                                  final_identification_result + '\n')
                    f_multi.close()

            acc_image = hit_image / total_target * 100
            acc_speech = hit_speech / total_target * 100
            acc_multi = hit_multi / total_target * 100
            result_image = 'Single-modal Image-based Accuracy of the test sample = {}%'\
                .format(acc_image)
            result_voice = 'Single-modal Audio-based Accuracy of the test sample = {}%'\
                .format(acc_speech)
            result_final = 'Multi-modal Face Recognition Accuracy of the test sample = {}%'\
                .format(acc_multi)
            df = pd.DataFrame(csv_list)
            df = df[['test_path', 'label', 'pred', 'result']]
            df.to_csv('prediction_result.csv', index=False)
            
        else:
            if re_register == True:
                if not os.path.exists(path_target):
                    result_image = "Can't find target data. Clarify target path"
                    result_voice = "Can't find target data. Clarify target path"
                    result_final = "Can't find target data. Clarify target path"

                else:
                    img = extract_image(path, name, name_class, re_register)
                    emb_array = model_img.run(img)
                    result_image = save_feature(emb_array, path, name_class, re_register)
                    result_voice = voice_registration(model_voice, path, name, name_class,
                                                      re_register)
                    result_final = 'Re-registration Completed'

            else:
                image_gallery_exist = gallery(path, name_class)
                voice_gallery_exist = voice_gallery(path, name_class)

                if image_gallery_exist is False and voice_gallery_exist is False:
                    img = extract_image(path, name, name_class)
                    emb_array = model_img.run(img)
                    result_image = save_feature(emb_array, path, name_class, re_register)
                    result_voice = voice_registration(model_voice, path, name, name_class,
                                                      re_register)
                    result_final = 'Registration Completed'

                elif image_gallery_exist is False and voice_gallery_exist is True:
                    img = extract_image(path, name, name_class)
                    emb_array = model_img.run(img)
                    result_image = save_feature(emb_array, path, name_class,
                                                re_register)
                    result_voice = 'Already registered - Speaker ID : {}'.format(name_class)
                    result_final = 'Video model saved'

                elif image_gallery_exist is True and voice_gallery_exist is False:
                    result_voice = voice_registration(model_voice, path, name, name_class,
                                                      re_register)
                    result_image = 'Already registered - Speaker ID : {}'.format(name_class)
                    result_final = 'Voice model saved'

                elif image_gallery_exist is True and voice_gallery_exist is True:
                    # Video
                    img = extract_image(path, name, name_class)
                    emb_array = model_img.run(img)
                    cos_sim_image, id_top5_image, gallery_emb_img, img_list, target_emb_img =\
                        identify_target(path, emb_array)
                    # Audio
                    cos_sim_voice, id_top5_voice, gallery_emb_audio, audio_list, target_emb_audio =\
                        voice_identification(model_voice, path_target)

                    image_identification_result = final_result(cos_sim_image, id_top5_image)
                    voice_identification_result = id_top5_voice[np.argmax(cos_sim_voice)]
                    result_image = 'Image Based Speaker Recognition - Speaker ID : {}'.format(
                        image_identification_result)
                    result_voice = 'Voice Based Speaker Recognition - Speaker ID : {}'.format(
                        voice_identification_result)

                    emb_final_target = model_multi.run(expand_dim(target_emb_img), expand_dim(
                        target_emb_audio))
                    assert len(gallery_emb_img) == len(gallery_emb_audio), "Check gallery of" \
                                                                           " Image/Audio File"
                    gallery_emb_final = []
                    for i in range(len(gallery_emb_img)):
                        emb_multi = model_multi.run(expand_dim(gallery_emb_img[i]), expand_dim(
                            gallery_emb_audio[i]))
                        gallery_emb_final.append(emb_multi)

                    cos_sim_final, id_top5_final = identify_final(emb_final_target,
                                                                  gallery_emb_final, img_list)
                    final_identification_result = id_top5_final[np.argmax(cos_sim_final)]
                    result_final = 'Multi-Modal Speaker Recognition - Speaker ID : {}'.format(
                        final_identification_result)

        # speaker recognition
        result["100001"] = result_image
        result["100002"] = result_voice
        result["100003"] = result_final

        print(result)
        print('Done!!')

        return json.dumps(result)




if __name__ == '__main__':
    print("Loading tensorflow model and Flask starting server...")
    print("Network already loading and app running")
    app.run(host='0.0.0.0', debug=True)
    # app.run(host=os.environ.get('165.132.56.182', '0.0.0.0'), port=8888, debug=True)
