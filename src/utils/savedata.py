import tensorflow as tf
import numpy as np
import re
import sys
from save_theta import *

# outdir = sys.argv[0]
# print (outdir)

# tf.app.flags.DEFINE_string('gpu', '0', """gpu id.""")
# sess = tf.Session()
# saver = tf.train.import_meta_graph('../../log/train/model.ckpt-310500.meta')

# saver.restore(sess, tf.train.latest_checkpoint(''))

# saver.restore(sess, tf.train.latest_checkpoint('/home/zzx/workspace/squeezeDet/log/train/model.ckpt-310500.data-00000-of-00001'))

# for x in tf.trainable_variables():
#     filename = '../../data/float2fixed' + re.sub(r'/', '_', x.op.name) + ':0'
#     savetheta(filename, x)
def save_data(sess, savedir):
    # sess = tf.Session()
    # saver = tf.train.import_meta_graph('model.ckpt-311000.meta')
    # saver.restore(sess, tf.train.latest_checkpoint(''))

    for x in tf.trainable_variables():
        filename = savedir + '/' + re.sub(r'/', '_', x.op.name) + ':0'
        savetheta(filename, x.eval(session=sess))
        # print (x.op.name)
