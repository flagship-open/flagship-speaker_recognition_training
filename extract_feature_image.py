import tensorflow as tf
import numpy as np
import argparse
import utils
#from amsoftmax import *
import math
import os

def main():
    parser = argparse.ArgumentParser(description='Multi modal Face recognition')

    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--img_model_path', type=str, default='./models/pretrained_model_img')
    parser.add_argument('--image_size', type=int, default=112)
    parser.add_argument('--dataset_path', type=str, default='./train.txt')
    parser.add_argument('--embedding_size', type=int, default=128)
    parser.add_argument('--save_path', type=str, default='./img_feature.npy')

    args = parser.parse_args()

    graph_vic = tf.Graph()
    graph_img = tf.Graph()

    vic_data, img_data, label = utils.dataset_from_list(args.dataset_path)

    batch_size = args.batch_size
    nrof_images = len(img_data)
    nrof_batches = int(math.ceil(1.0*nrof_images)/batch_size)

    with graph_img.as_default():
        with tf.Session() as session2:
            load_model_img(args.img_model_path)
            inputs_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings_img = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            embedding_size = embeddings_img.get_shape()[1]

            emb_array = np.zeros((nrof_images, embedding_size))

            for i in range(nrof_batches):
                start_index = i*batch_size
                print('Oh Starting!! {}/{}'.format(start_index,nrof_images))
                end_index = min((i+1)*batch_size, nrof_images)
                paths_batch = img_data[start_index:end_index]
                images = utils.load_data(paths_batch, False, args.image_size, args.image_size)
                feed_dict = {inputs_placeholder:images}
                feats = session2.run(embeddings_img, feed_dict=feed_dict)
                feats = utils.l2_normalize(feats)
                emb_array[start_index:end_index,:] = feats

            print(emb_array)
            emb_array = np.array(emb_array)
            np.save(args.save_path, emb_array)


def load_model_img(model):
    # Check if the model is a model directory (containing a metagraph and a checkpoint file)
    #  or if it is a protobuf file with a frozen graph
    model_exp = os.path.expanduser(model)
    if (os.path.isfile(model_exp)):
        print('Model filename: %s' % model_exp)
        with tf.gfile.FastGFile(model_exp, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name='')
    else:
        print('Model directory: %s' % model_exp)
        meta_file, ckpt_file = get_model_filenames(model_exp)

        print('Metagraph file: %s' % meta_file)
        print('Checkpoint file: %s' % ckpt_file)

        saver = tf.train.import_meta_graph(os.path.join(model_exp, meta_file))
        saver.restore(tf.get_default_session(), os.path.join(model_exp, ckpt_file))

def get_model_filenames(model_dir):
    files = os.listdir(model_dir)
    meta_files = [s for s in files if s.endswith('.meta')]
    if len(meta_files) == 0:
        raise ValueError('No meta file found in the model directory (%s)' % model_dir)
    elif len(meta_files) > 1:
        raise ValueError('There should not be more than one meta file in the model directory (%s)' % model_dir)
    meta_file = meta_files[0]
    ckpt = tf.train.get_checkpoint_state(model_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_file = os.path.basename(ckpt.model_checkpoint_path)
        return meta_file, ckpt_file

    meta_files = [s for s in files if '.ckpt' in s]
    max_step = -1
    for f in files:
        step_str = re.match(r'(^model-[\w\- ]+.ckpt-(\d+))', f)
        if step_str is not None and len(step_str.groups()) >= 2:
            step = int(step_str.groups()[1])
            if step > max_step:
                max_step = step
                ckpt_file = step_str.groups()[0]
    return meta_file, ckpt_file

if __name__ == "__main__":
    main()




