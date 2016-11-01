# general
import tensorflow as tf
import numpy as np
import os
import csv
import time
import datetime
from sklearn.metrics import confusion_matrix
import data_helpers

# output file
tf.flags.DEFINE_string("output_file", "test_1.csv", "out put file")

# model names
tf.flags.DEFINE_string("model_name", "train_1", "model name")

# testing data
tf.flags.DEFINE_string("test_file", "test_1", "test data file")
tf.flags.DEFINE_float("classes", 2, "classes")

# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 32, "Batch Size (default: 64)")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")


# Load data
print("Loading data...")
x_test, y_test = data_helpers.load_data(FLAGS.test_file, FLAGS.classes)

print("number of test: %d" % x_test.shape[0])
print("")

# Evaluation
# ==================================================

checkpoint_file = tf.train.latest_checkpoint("./models/%s/checkpoints" % FLAGS.model_name)
graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)
        
        # Get the placeholders from the graph by name
        x = graph.get_operation_by_name("x").outputs[0]
        y = graph.get_operation_by_name("y").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

        # Tensors to evaluate
        probability = graph.get_operation_by_name("output/probability").outputs[0]

        # output file
        count = 0
        with open(os.path.join("tests", FLAGS.output_file), 'w') as f:
            csvwriter = csv.writer(f)
            csvwriter.writerow(['File', 'Class'])
            batches_test = data_helpers.batch_iter(x_test, x_test, FLAGS.batch_size, 1, shuffle=False)
            for x_batch_test, y_batch_test in batches_test:
                count += x_batch_test.shape[0]
                print count
                feed_dict = {
                  x: x_batch_test,
                  dropout_keep_prob: 1.0
                }
                prob = sess.run(probability, feed_dict)
                Classes = prob[:, 0]
                Files =  [f.split("/")[-1] for f in y_batch_test]
                csvwriter.writerows(zip(Files, Classes))
        