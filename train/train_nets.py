# -*- coding: utf-8 -*-

from train.losses.face_losses import insightface_loss, cosineface_loss, combine_loss
from train.nets.Multi_Dense import inference
from train.utils.common import train
from datetime import datetime
import tensorflow as tf
import numpy as np
import argparse
import time
import os

slim = tf.contrib.slim

def get_parser():
    parser = argparse.ArgumentParser(description='parameters to train net')
    parser.add_argument('--max_epoch', default=450, help='epoch to train the network')
    parser.add_argument('--class_number', type=int, default=1211,
                        help='class number depend on your training datasets')
    parser.add_argument('--embed_size_img', type=int,
                        help='Dimensionality of the embedding.', default=128)
    parser.add_argument('--embed_size_vic', type=int,
                        help='Dimensionality of the embedding.', default=64)
    parser.add_argument('--img_path', type=str, default='./img_feature.npy')
    parser.add_argument('--vic_path', type=str, default='./vic_feature.npy')
    parser.add_argument('--label', type=str, default='./label.npy')
    parser.add_argument('--weight_decay', default=5e-5, help='L2 weight regularization.')

    parser.add_argument('--lr_schedule', help='Number of epochs for learning rate piecewise.',
                        default=[120,240,360])
    parser.add_argument('--train_batch_size', default=32, help='batch size to train network')
    parser.add_argument('--test_batch_size', type=int,
                        help='Number of images to process in a batch in the test set.', default=32)
    parser.add_argument('--tfrecords_file_path', default='./datasets/faces_emore/tfrecords',
                        type=str, help='path to the output of tfrecords file path')
    parser.add_argument('--summary_path', default='./output/summary',
                        help='the summary file save path')
    parser.add_argument('--ckpt_path', default='./output/ckpt', help='the ckpt file save path')
    parser.add_argument('--ckpt_best_path', default='./output/ckpt_best',
                        help='the best ckpt file save path')
    parser.add_argument('--log_file_path', default='./output/logs', help='the ckpt file save path')
    parser.add_argument('--saver_maxkeep', default=50, help='tf.train.Saver max keep ckpt files')
    parser.add_argument('--summary_interval', default=400, help='interval to save summary')
    parser.add_argument('--ckpt_interval', default=12000, help='intervals to save ckpt file')
    parser.add_argument('--validate_interval', default=2000, help='intervals to save ckpt file')
    parser.add_argument('--show_info_interval', default=50, help='intervals to save ckpt file')
    parser.add_argument('--pretrained_model', type=str, default='',
                        help='Load a pretrained model before training starts.')
    parser.add_argument('--optimizer', type=str, choices=['ADAGRAD', 'ADADELTA', 'ADAM', 'RMSPROP',
                                                          'MOM'],
                        help='The optimization algorithm to use', default='ADAM')
    parser.add_argument('--log_device_mapping', default=False, help='show device placement log')
    parser.add_argument('--moving_average_decay', type=float,
                        help='Exponential decay for tracking of training parameters.',
                        default=0.999)
    parser.add_argument('--log_histograms',
                        help='Enables logging of weight/bias histograms in tensorboard.',
                        action='store_true')
    parser.add_argument('--prelogits_norm_loss_factor', type=float,
                        help='Loss based on the norm of the activations in the prelogits layer.',
                        default=2e-5)
    parser.add_argument('--prelogits_norm_p', type=float,
                        help='Norm to use for prelogits norm loss.', default=1.0)
    parser.add_argument('--loss_type', default='cosine',
                        help='loss type, choice type are insightface/cosine/combine')
    parser.add_argument('--margin_s', type=float,
                        help='insightface_loss/cosineface_losses/combine_loss loss scale.',
                        default=64.)
    parser.add_argument('--margin_m', type=float,
                        help='insightface_loss/cosineface_losses/combine_loss loss margin.',
                        default=0.35)
    parser.add_argument('--margin_a', type=float,
                        help='combine_loss loss margin a.', default=1.0)
    parser.add_argument('--margin_b', type=float,
                        help='combine_loss loss margin b.', default=0.2)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    with tf.Graph().as_default():
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        args = get_parser()

        # create log dir
        subdir = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
        log_dir = os.path.join(os.path.expanduser(args.log_file_path), subdir)
        if not os.path.isdir(log_dir):
            os.makedirs(log_dir)

        # define global parameters
        global_step = tf.Variable(name='global_step', initial_value=0, trainable=False)
        epoch = tf.Variable(name='epoch', initial_value=-1, trainable=False)
        # define placeholder
        inputs_img = tf.placeholder(name='img_inputs', shape=[None, 1, 1, 128], dtype=tf.float32)
        inputs_vic = tf.placeholder(name='vic_inputs', shape=[None, 1, 1, 64], dtype=tf.float32)
        labels = tf.placeholder(name='labels', shape=[None, ], dtype=tf.int64)
        phase_train_placeholder = tf.placeholder_with_default(tf.constant(False, dtype=tf.bool),
                                                              shape=None, name='phase_train')

        train_img = np.load(args.img_path)
        train_vic = np.load(args.vic_path)
        label = np.load(args.label)

        train_iteration = len(train_img) // args.train_batch_size

        # pretrained model path
        pretrained_model = None
        if args.pretrained_model:
            pretrained_model = os.path.expanduser(args.pretrained_model)
            print('Pre-trained model: %s' % pretrained_model)

        train_img = np.expand_dims(train_img, axis=1)
        train_img = np.expand_dims(train_img, axis=1)
        train_vic = np.expand_dims(train_vic, axis=1)
        train_vic = np.expand_dims(train_vic, axis=1)

        # identity the input, for inference
        inputs_img = tf.identity(inputs_img, 'input_img')
        inputs_vic = tf.identity(inputs_vic, 'input_vic')
        prelogits = inference(inputs_img, inputs_vic, phase_train=phase_train_placeholder,
                              weight_decay=args.weight_decay)

        embeddings = tf.nn.l2_normalize(prelogits, 1, 1e-10, name='embeddings')

        # Norm for the prelogits
        eps = 1e-5
        prelogits_norm = tf.reduce_mean(tf.norm(tf.abs(prelogits) + eps, ord=args.prelogits_norm_p,
                                                axis=1))
        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, prelogits_norm * args.
                             prelogits_norm_loss_factor)

        # inference_loss, logit = cos_loss(prelogits, labels, args.class_number)
        w_init_method = slim.initializers.xavier_initializer()
        if args.loss_type == 'insightface':
            inference_loss, logit = insightface_loss(embeddings, labels, args.class_number,
                                                     w_init_method)
        elif args.loss_type == 'cosine':
            inference_loss, logit = cosineface_loss(embeddings, labels, args.class_number,
                                                    w_init_method)
        elif args.loss_type == 'combine':
            inference_loss, logit = combine_loss(embeddings, labels, args.train_batch_size,
                                                 args.class_number, w_init_method)
        else:
            assert 0, 'loss type error, choice item just one of [insightface, cosine, combine],' \
                      ' please check!'
        tf.add_to_collection('losses', inference_loss)

        # total losses
        regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        total_loss = tf.add_n([inference_loss] + regularization_losses, name='total_loss')

        # define the learning rate schedule
        learning_rate = tf.train.piecewise_constant(epoch, boundaries=args.lr_schedule,
                                                    values=[0.1, 0.01, 0.001, 0.0001],
                                         name='lr_schedule')
        
        # define sess
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=
        args.log_device_mapping, gpu_options=gpu_options)

        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)

        # calculate accuracy
        pred = tf.nn.softmax(logit)
        correct_prediction = tf.cast(tf.equal(tf.argmax(pred, 1), tf.cast(labels, tf.int64)),
                                     tf.float32)
        Accuracy_Op = tf.reduce_mean(correct_prediction)

        # summary writer
        summary = tf.summary.FileWriter(args.summary_path, sess.graph)
        summaries = []
        summaries.append(tf.summary.scalar('inference_loss', inference_loss))
        summaries.append(tf.summary.scalar('total_loss', total_loss))
        summaries.append(tf.summary.scalar('leraning_rate', learning_rate))
        summary_op = tf.summary.merge(summaries)

        train_op = train(total_loss, global_step, args.optimizer, learning_rate,
                         args.moving_average_decay,
                         tf.global_variables(), summaries, args.log_histograms)
        inc_global_step_op = tf.assign_add(global_step, 1, name='increment_global_step')
        inc_epoch_op = tf.assign_add(epoch, 1, name='increment_epoch')

        # saver to load pretrained model or save model
        saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=args.saver_maxkeep)

        # init all variables
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        # load pretrained model
        if pretrained_model:
            print('Restoring pretrained model: %s' % pretrained_model)
            ckpt = tf.train.get_checkpoint_state(pretrained_model)
            print(ckpt)
            saver.restore(sess, ckpt.model_checkpoint_path)

        # output file path
        if not os.path.exists(args.log_file_path):
            os.makedirs(args.log_file_path)
        if not os.path.exists(args.ckpt_best_path):
            os.makedirs(args.ckpt_best_path)

        start_batch_id = 0
        count = 0
        total_accuracy = {}

        for i in range(args.max_epoch):
            for idx in range(start_batch_id, train_iteration):
                voices_train = train_vic[idx * args.train_batch_size:(idx + 1) *
                                                                     args.train_batch_size]
                images_train = train_img[idx * args.train_batch_size:(idx + 1) *
                                                                     args.train_batch_size]
                labels_train = label[idx * args.train_batch_size:(idx + 1) *
                                                                 args.train_batch_size]

                _ = sess.run(inc_epoch_op)


                feed_dict = {inputs_img: images_train, inputs_vic: voices_train, labels:
                    labels_train, phase_train_placeholder: True}
                start = time.time()
                _, total_loss_val, inference_loss_val, reg_loss_val, _, acc_val = \
                sess.run([train_op, total_loss, inference_loss, regularization_losses,
                          inc_global_step_op, Accuracy_Op],
                         feed_dict=feed_dict)
                end = time.time()
                pre_sec = args.train_batch_size/(end - start)

                count += 1
                # print training information
                if count > 0 and count % args.show_info_interval == 0:
                    print('epoch %d, total_step %d, total loss is %.2f , inference loss'
                          ' is %.2f, reg_loss is %.2f, training accuracy is %.6f, time %.3f'
                          ' samples/sec' %
                          (i, count, total_loss_val, inference_loss_val, np.sum(reg_loss_val),
                           acc_val, pre_sec))

                # save summary
                if count > 0 and count % args.summary_interval == 0:
                    feed_dict = {inputs_img: images_train, inputs_vic:voices_train, labels:
                        labels_train, phase_train_placeholder: True}
                    summary_op_val = sess.run(summary_op, feed_dict=feed_dict)
                    summary.add_summary(summary_op_val, count)

                # save ckpt files
                if count > 0 and count % args.ckpt_interval == 0:
                    filename = 'MobileFaceNet_iter_{:d}'.format(count) + '.ckpt'
                    filename = os.path.join(args.ckpt_path, filename)
                    saver.save(sess, filename)
