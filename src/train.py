from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
from datetime import datetime
import os.path
import sys
import time

import numpy as np
from six.moves import xrange
import tensorflow as tf
import threading

from config import *
from dataset import  kitti
from utils.util import sparse_to_dense, bgr_to_rgb, bbox_transform
from nets import *
from float2pow2 import float2pow2_offline

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('dataset', 'KITTI',
                           """Currently only support KITTI dataset.""")
tf.app.flags.DEFINE_string('data_path', '', """Root directory of data""")
tf.app.flags.DEFINE_string('image_set', 'train',
                           """ Can be train, trainval, val, or test""")
tf.app.flags.DEFINE_string('year', '2007',
                            """VOC challenge year. 2007 or 2012"""
                            """Only used for Pascal VOC dataset""")
tf.app.flags.DEFINE_string('train_dir', '/tmp/bichen/logs/squeezeDet/train',
                            """Directory where to write event logs """
                            """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 100000,
                            """Maximum number of batches to run.""")
tf.app.flags.DEFINE_string('net', 'squeezeDet',
                           """Neural net architecture. """)
tf.app.flags.DEFINE_string('pretrained_model_path', '',
                           """Path to the pretrained model.""")
tf.app.flags.DEFINE_integer('summary_step', 10,
                            """Number of steps to save summary.""")
tf.app.flags.DEFINE_integer('checkpoint_step', 1000,
                            """Number of steps to save summary.""")
tf.app.flags.DEFINE_string('gpu', '0', """gpu id.""")
tf.app.flags.DEFINE_integer('reload', True,
                           """whether to reload train from scratch.""")
def draw_featuremap(layer_name, feature_array):
  path_temp = os.path.join(FLAGS.activations_path,layer_name)
  print("the path of feature map is:"+str(path_temp))
  feature_array1 = feature_array.tolist() 
  if tf.gfile.Exists(path_temp):
    tf.gfile.DeleteRecursively(path_temp)
  tf.gfile.MakeDirs(path_temp)
  for i in range(np.shape(feature_array)[-1]):
    print("drawing the" + str(i)+ ' th feature map:' + ' of the ' + str(layer_name))
    new_image = Image.fromarray(feature_array[0][:][:][i])
    new_image =  new_image.convert('RGB')
    saving_path = os.path.join(path_temp,(str(i)+'.png'))
    new_image.save(saving_path)
    print(np.shape(feature_array[0][:][:][i]))
    print(np.shape(feature_array))
    f = open(os.path.join(path_temp,(str(i)+'.txt')),'w')
    f.write(str(feature_array1[0][:][:][i]))
    f.close()



def _draw_box(im, box_list, label_list, color=(0,255,0), cdict=None, form='center'):
  assert form == 'center' or form == 'diagonal', \
      'bounding box format not accepted: {}.'.format(form)

  for bbox, label in zip(box_list, label_list):

    if form == 'center':
      bbox = bbox_transform(bbox)

    xmin, ymin, xmax, ymax = [int(b) for b in bbox]

    l = label.split(':')[0] # text before "CLASS: (PROB)"
    if cdict and l in cdict:
      c = cdict[l]
    else:
      c = color

    # draw box
    cv2.rectangle(im, (xmin, ymin), (xmax, ymax), c, 1)
    # draw label
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(im, label, (xmin, ymax), font, 0.3, c, 1)

def _viz_prediction_result(model, images, bboxes, labels, batch_det_bbox,
                           batch_det_class, batch_det_prob):
  mc = model.mc

  for i in range(len(images)):
    # draw ground truth
    _draw_box(
        images[i], bboxes[i],
        [mc.CLASS_NAMES[idx] for idx in labels[i]],
        (0, 255, 0))

    # draw prediction
    det_bbox, det_prob, det_class = model.filter_prediction(
        batch_det_bbox[i], batch_det_prob[i], batch_det_class[i])

    keep_idx    = [idx for idx in range(len(det_prob)) \
                      if det_prob[idx] > mc.PLOT_PROB_THRESH]
    det_bbox    = [det_bbox[idx] for idx in keep_idx]
    det_prob    = [det_prob[idx] for idx in keep_idx]
    det_class   = [det_class[idx] for idx in keep_idx]

    _draw_box(
        images[i], det_bbox,
        [mc.CLASS_NAMES[idx]+': (%.2f)'% prob \
            for idx, prob in zip(det_class, det_prob)],
        (0, 0, 255))

def _build_model_metrics(model):
  with open(os.path.join(FLAGS.train_dir, 'model_metrics.txt'), 'w') as f:
    f.write('Number of parameter by layer:\n')
    count = 0
    for c in model.model_size_counter:
      f.write('\t{}: {}\n'.format(c[0], c[1]))
      count += c[1]
    f.write('\ttotal: {}\n'.format(count))

    count = 0
    f.write('\nActivation size by layer:\n')
    for c in model.activation_counter:
      f.write('\t{}: {}\n'.format(c[0], c[1]))
      count += c[1]
    f.write('\ttotal: {}\n'.format(count))

    count = 0
    f.write('\nNumber of flops by layer:\n')
    for c in model.flop_counter:
      f.write('\t{}: {}\n'.format(c[0], c[1]))
      count += c[1]
      f.write('\ttotal: {}\n'.format(count))
    f.close()
  print ('Model statistics saved to {}.'.format(os.path.join(FLAGS.train_dir, 'model_metrics.txt')))

  

def train():
  """Train SqueezeDet model"""
  assert FLAGS.dataset == 'KITTI', \
      'Currently only support KITTI dataset'
  os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
  with tf.Graph().as_default():
    assert FLAGS.net == 'shuffleDet' or FLAGS.net == 'squeezeDet' or FLAGS.net == 'squeezeDet+', \
        'Selected neural net architecture not supported: {}'.format(FLAGS.net)
    if FLAGS.net == 'squeezeDet':
      mc = kitti_squeezeDet_config()
      mc.IS_TRAINING = True
      mc.PRETRAINED_MODEL_PATH = FLAGS.pretrained_model_path
      model = SqueezeDet(mc)
    elif FLAGS.net == 'shuffleDet':
      mc = kitti_shuffleDet_config()
      mc.IS_TRAINING = True
      mc.PRETRAINED_MODEL_PATH = FLAGS.pretrained_model_path
      model = shuffleDet(mc)
    imdb = kitti(FLAGS.image_set, FLAGS.data_path, mc)

    # save model size, flops, activations by layers
    _build_model_metrics(model)   

    def _load_data(load_to_placeholder=True):
      # read batch input
      image_per_batch, label_per_batch, box_delta_per_batch, aidx_per_batch, \
          bbox_per_batch = imdb.read_batch()

      label_indices, bbox_indices, box_delta_values, mask_indices, box_values, \
          = [], [], [], [], []
      aidx_set = set()
      num_discarded_labels = 0
      num_labels = 0
      for i in range(len(label_per_batch)): # batch_size
        for j in range(len(label_per_batch[i])): # number of annotations
          num_labels += 1
          if (i, aidx_per_batch[i][j]) not in aidx_set:
            aidx_set.add((i, aidx_per_batch[i][j]))
            label_indices.append(
                [i, aidx_per_batch[i][j], label_per_batch[i][j]])
            mask_indices.append([i, aidx_per_batch[i][j]])
            bbox_indices.extend(
                [[i, aidx_per_batch[i][j], k] for k in range(4)])
            box_delta_values.extend(box_delta_per_batch[i][j])
            box_values.extend(bbox_per_batch[i][j])
          else:
            num_discarded_labels += 1

      if load_to_placeholder:
        image_input = model.ph_image_input
        input_mask = model.ph_input_mask
        box_delta_input = model.ph_box_delta_input
        box_input = model.ph_box_input
        labels = model.ph_labels
      else:
        image_input = model.image_input
        input_mask = model.input_mask
        box_delta_input = model.box_delta_input
        box_input = model.box_input
        labels = model.labels

      feed_dict = {
          image_input: image_per_batch,
          input_mask: np.reshape(
              sparse_to_dense(
                  mask_indices, [mc.BATCH_SIZE, mc.ANCHORS],
                  [1.0]*len(mask_indices)),
              [mc.BATCH_SIZE, mc.ANCHORS, 1]),
          box_delta_input: sparse_to_dense(
              bbox_indices, [mc.BATCH_SIZE, mc.ANCHORS, 4],
              box_delta_values),
          box_input: sparse_to_dense(
              bbox_indices, [mc.BATCH_SIZE, mc.ANCHORS, 4],
              box_values),
          labels: sparse_to_dense(
              label_indices,
              [mc.BATCH_SIZE, mc.ANCHORS, mc.CLASSES],
              [1.0]*len(label_indices)),
      }

      return feed_dict, image_per_batch, label_per_batch, bbox_per_batch

    def _enqueue(sess, coord):
      try:
        while not coord.should_stop():
          feed_dict, _, _, _ = _load_data()
          sess.run(model.enqueue_op, feed_dict=feed_dict)
          if mc.DEBUG_MODE:
            print ("added to the queue")
        if mc.DEBUG_MODE:
          print ("Finished enqueue")
      except Exception, e:
        coord.request_stop(e)

    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    # print(tf.global_variables())


    # variables = tf.trainable_variables() 
    variables = tf.contrib.framework.get_variables_to_restore()

    variables_to_resotre = [v for v in variables if 'Momentum' not in v.name and 'iou' not in v.name and 'global_step' not in v.name and 'preds' not in v.name]
    saver = tf.train.Saver(variables_to_resotre)
    summary_op = tf.summary.merge_all()
    checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
    # saver.restore(sess, checkpoint_path)

    # ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
    # print('ckpt is:'+str(ckpt))
    # if ckpt and ckpt.model_checkpoint_path:
    #     saver.restore(sess, ckpt.model_checkpoint_path)

    summary_writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph)
    init = tf.global_variables_initializer()
    init = tf.initialize_all_variables()
    sess.run(init)

    coord = tf.train.Coordinator()

    if mc.NUM_THREAD > 0:
      enq_threads = []
      for _ in range(mc.NUM_THREAD):
        enq_thread = threading.Thread(target=_enqueue, args=[sess, coord])
        # enq_thread.isDaemon()
        enq_thread.start()
        enq_threads.append(enq_thread)

    threads = tf.train.start_queue_runners(coord=coord, sess=sess)
    run_options = tf.RunOptions(timeout_in_ms=60000)
    if FLAGS.reload:
      checkpoint_path = os.path.join(FLAGS.train_dir, '/')
      print('load the parameters from: '+str(FLAGS.train_dir))
      ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
      if ckpt and ckpt.model_checkpoint_path:
        print('Restores from checkpoint....')
        # Restores from checkpoint
        saver.restore(sess, ckpt.model_checkpoint_path)
        # Assuming model_checkpoint_path looks something like:
        #   /my-favorite-path/cifar10_train/model.ckpt-0,
        # extract global_step from it.
        # step = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
        step = 1
      else:
        step = 1
        print('No checkpoint found!')
    else:
      tf.gfile.DeleteRecursively(FLAGS.train_dir)
      tf.gfile.MakeDirs(FLAGS.train_dir)
      step = 1

    # try:
    for step in xrange(FLAGS.max_steps):
      if coord.should_stop():
        sess.run(model.FIFOQueue.close(cancel_pending_enqueues=True))
        coord.request_stop()
        coord.join(threads)
        break

      start_time = time.time()

      if step % FLAGS.summary_step == 0:
        feed_dict, image_per_batch, label_per_batch, bbox_per_batch = \
            _load_data(load_to_placeholder=False)
        op_list = [
            model.train_op, model.loss, summary_op, model.det_boxes,
            model.det_probs, model.det_class, model.conf_loss,
            model.bbox_loss, model.class_loss
        ]
        _, loss_value, summary_str, det_boxes, det_probs, det_class, \
            conf_loss, bbox_loss, class_loss = sess.run(
                op_list, feed_dict=feed_dict)

        _viz_prediction_result(
            model, image_per_batch, bbox_per_batch, label_per_batch, det_boxes,
            det_class, det_probs)
        image_per_batch = bgr_to_rgb(image_per_batch)
        viz_summary = sess.run(
            model.viz_op, feed_dict={model.image_to_show: image_per_batch})

        summary_writer.add_summary(summary_str, step)
        summary_writer.add_summary(viz_summary, step)
        summary_writer.flush()

        print ('conf_loss: {}, bbox_loss: {}, class_loss: {}'.
            format(conf_loss, bbox_loss, class_loss))
      else:
        if mc.NUM_THREAD > 0:
          _, loss_value, conf_loss, bbox_loss, class_loss = sess.run(
              [model.train_op, model.loss, model.conf_loss, model.bbox_loss,
               model.class_loss], options=run_options)
        else:
          feed_dict, _, _, _ = _load_data(load_to_placeholder=False)
          _, loss_value, conf_loss, bbox_loss, class_loss = sess.run(
              [model.train_op, model.loss, model.conf_loss, model.bbox_loss,
               model.class_loss], feed_dict=feed_dict)

      duration = time.time() - start_time

      assert not np.isnan(loss_value), \
          'Model diverged. Total loss: {}, conf_loss: {}, bbox_loss: {}, ' \
          'class_loss: {}'.format(loss_value, conf_loss, bbox_loss, class_loss)

      if step % 10 == 0:
        num_images_per_step = mc.BATCH_SIZE
        images_per_sec = num_images_per_step / duration
        sec_per_batch = float(duration)
        format_str = ('%s: step %d, loss = %.2f (%.1f images/sec; %.3f '
                      'sec/batch)')
        print (format_str % (datetime.now(), step, loss_value,
                             images_per_sec, sec_per_batch))
        sys.stdout.flush()

      # Save the model checkpoint periodically.
      if step % FLAGS.checkpoint_step == 0 or (step + 1) == FLAGS.max_steps:
        checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step,write_state = True)
      # fix_ops = float2pow2_offline(0, 0, 0, './weight', sess, resave=True, recall=False, convert=False, trt=True)
      # sess.run(fix_ops)
      # exit()
    # except Exception, e:
    #   coord.request_stop(e)
    # finally:
    #   coord.request_stop()
    #   coord.join(threads)

def main(argv=None):  # pylint: disable=unused-argument
  train()


if __name__ == '__main__':
  tf.app.run()
