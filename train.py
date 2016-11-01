# general
import tensorflow as tf
import numpy as np
import os
import time
import datetime
from sklearn.metrics import roc_auc_score, confusion_matrix

# local
from model import Model
import data_helpers

# Parameters
# ==================================================
# Data parameters
tf.flags.DEFINE_integer("n_channels", 16, "n_channels")
tf.flags.DEFINE_integer("dim1", 129, "dim1")
tf.flags.DEFINE_integer("dim2", 1071, "dim2")
tf.flags.DEFINE_integer("classes", 2, "classes")
tf.flags.DEFINE_string("train_file", "train_1", "training data file")

tf.flags.DEFINE_string("train_option", "dev", "train or dev")

# model names
tf.flags.DEFINE_string("model_name", "train_1", "model name")

# Model Hyperparameters
tf.flags.DEFINE_float("lr", 1e-3, "learning rate (default: 5e-4)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.0, "Dropout keep probability (default: 0.8)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularizaion lambda (default: 0.5)")
tf.flags.DEFINE_float("train_dev_split", 0.20, "train dev split (default: 0.15)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 32, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 10, "Number of training epochs (default: 40)")
tf.flags.DEFINE_integer("evaluate_every", 10, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 10, "Save model after this many steps (default: 100)")

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
x, y = data_helpers.load_data(FLAGS.train_file, FLAGS.classes)

# Randomly shuffle data
np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(y)))
x_shuffled = x[shuffle_indices]
y_shuffled = y[shuffle_indices]

# train/test split
if FLAGS.train_option == "dev":
    N = int(FLAGS.train_dev_split*x.shape[0])
    x_train, x_dev = x_shuffled[:-N], x_shuffled[-N:]
    y_train, y_dev = y_shuffled[:-N], y_shuffled[-N:]

    print("Train/Dev split: {:d}/{:d} \n".format(len(y_train), len(y_dev)))

else:
    x_train, x_dev = x_shuffled, []
    y_train, y_dev = y_shuffled, []

# Training
# ==================================================

with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)

    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # model
        model = Model(FLAGS.n_channels, FLAGS.dim1, FLAGS.dim2, FLAGS.classes, 
                          l2_reg_lambda=FLAGS.l2_reg_lambda)

        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(FLAGS.lr)
        grads_and_vars = optimizer.compute_gradients(model.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # Output directory for models and summaries
        try:
            shutil.rmtree(os.path.join(os.path.curdir, "models", FLAGS.model_name))
        except:
            pass 
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "models", FLAGS.model_name))
        print("Writing to {}\n".format(out_dir))

        # Summaries for loss and accuracy
        loss_summary = tf.scalar_summary("loss", model.loss)
        acc_summary = tf.scalar_summary("accuracy", model.accuracy)

        # Train Summaries
        train_summary_op = tf.merge_summary([loss_summary, acc_summary])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.train.SummaryWriter(train_summary_dir, sess.graph)

        # Dev summaries
        dev_summary_op = tf.merge_summary([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.train.SummaryWriter(dev_summary_dir, sess.graph)

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.all_variables())

        # Initialize all variables
        sess.run(tf.initialize_all_variables())

        # train step
        def train_step(x_batch, y_batch):
            feed_dict = {
              model.x: x_batch,
              model.y: y_batch,
              model.dropout_keep_prob: FLAGS.dropout_keep_prob
            }
            _, step, summaries, loss, accuracy = sess.run(
                [train_op, global_step, train_summary_op, model.loss, model.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            train_summary_writer.add_summary(summaries, step)

        # dev step
        def dev_step(x, y):
            batches = data_helpers.batch_iter(x, y, 64, 1, shuffle=False)

            y_true, y_scores = [], []
            for x_batch, y_batch in batches:
                feed_dict = {
                  model.x: x_batch,
                  model.y: y_batch,
                  model.dropout_keep_prob: 1.0
                }
                scores = sess.run(model.probability, feed_dict)
                y_true.append(y_batch)
                y_scores.append(scores)

            y_true = np.vstack(y_true)
            y_scores = np.vstack(y_scores)

            # auc
            auc = roc_auc_score(y_true, y_scores)
            print("AUC: %.4f" % auc)

            # confusion matrix
            y_pred = np.argmax(y_scores, 1)
            y_true = np.argmax(y_true, 1)
            print("confusion_matrix: ")
            print(confusion_matrix(y_true, y_pred, labels=range(0, FLAGS.classes)))
           
        # Generate batches
        batches = data_helpers.batch_iter_balanced(
            x_train, y_train, FLAGS.batch_size, FLAGS.num_epochs)

        # train 
        for x_batch, y_batch in batches:
            train_step(x_batch, y_batch)
            current_step = tf.train.global_step(sess, global_step)

            if current_step % FLAGS.checkpoint_every == 0:
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                print("\n Saved model checkpoint to {}\n".format(path))

            # testing
            if FLAGS.train_option == "dev" and current_step % FLAGS.evaluate_every == 0:
                print("\n test:")
                dev_step(x_dev, y_dev)
                print("") 
     
