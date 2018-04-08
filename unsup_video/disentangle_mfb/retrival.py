from __future__ import print_function
from __future__ import absolute_import

import sys
import h5py
import numpy as np
import tensorflow as tf


tf.logging.set_verbosity(tf.logging.INFO)


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('input_file', 'features.dat', """Pickle file with the extracted descriptors.""")
tf.app.flags.DEFINE_integer('k', 0, """Check only top-k results for each query. Set to -1 to disable this option.""")


def create_class_dict(image_list):
    class_dict = {}
    id_count = 0
    for f in image_list:
        class_name = f.decode('utf-8').split('_')[0]
        #class_name = f.split('_')[0]
        if class_name not in class_dict.keys():
            class_dict[class_name] = id_count
            id_count += 1
    return class_dict


def class_fraction(image_list, class_dict):
    num_samples = len(image_list)
    count = [0.] * len(class_dict.keys())
    for img in image_list:
        count[class_dict[img.decode('utf-8').split('_')[0]]] += 1
    fraction = [1. * c / num_samples for c in count]
    print('Samples per class: ', count)
    print('Class fraction: ', fraction)
    return count, fraction


def compute_similarities(descriptors):
    """
    Compute similarities between all samples
    :param descriptors: Tensor of shape [num_samples, descriptor_length]
    :return: Tensor of shape [num_samples, num_samples] with pairwise sample similarities
    """
    return tf.matmul(descriptors, descriptors, transpose_b=True)


def compute_precision(target, results):
    num_correct = len([elem for elem in results if elem == target])
    return 1. * num_correct / len(results)


def compute_recall(results):
    pass


def compute_average_precision(target, results, max_k=0):
    average_precision = 0
    relevant_results = 0
    if max_k <= 0:
        max_k = len(results)
    for k in range(max_k):
        relevant_results += int(results[k] == target)
        average_precision += int(results[k] == target) * compute_precision(target, results[:k+1])
    if relevant_results > 0:
        average_precision /= relevant_results
    return average_precision


def main(argv=None):
    # Load descriptors
    dataset_dict = h5py.File(FLAGS.input_file, "r")
    img_list = list(dataset_dict.keys())
    class_dict = create_class_dict(img_list)
    num_samples = len(img_list)
    num_dims = dataset_dict[img_list[0]][0].shape[0]
    descriptors = np.zeros((num_samples, num_dims))
    class_fraction(img_list, class_dict)
    for idx, img in enumerate(img_list):
        descriptors[idx, :] = dataset_dict[img][0] / np.linalg.norm(dataset_dict[img][0])

    # Create and run TF graph
    descriptors_tf = tf.Variable(descriptors, trainable=False)
    similarities_tf = compute_similarities(descriptors_tf)
    sess = tf.InteractiveSession()
    sess.run(tf.initialize_all_variables())
    similarities = sess.run(similarities_tf)

    # Compute precision
    # precision = 0
    # for idx, img in enumerate(img_list):
    #     ranked_list = similarities[idx, :].argsort()[::-1][:FLAGS.k + 1]
    #     ranked_class_list = [class_dict[img_list[s].decode('utf-8').split('_')[0]] for s in ranked_list.tolist()]
    #     precision += compute_precision(ranked_class_list[0], ranked_class_list[1:])
    # precision /= num_samples
    # print('\nPrecision@%d = %.4f' % (FLAGS.k, precision))

    # Compute mAP
    mean_avg_precision = 0
    for idx, img in enumerate(img_list):
        ranked_list = similarities[idx, :].argsort()[::-1]
        ranked_class_list = [class_dict[img_list[s].decode('utf-8').split('_')[0]] for s in ranked_list.tolist()]
        mean_avg_precision += compute_average_precision(ranked_class_list[0], ranked_class_list[1:], FLAGS.k)
        sys.stdout.write('Processed samples: %d/%d \r' % (idx + 1, num_samples))
        sys.stdout.flush()
    mean_avg_precision /= num_samples
    print('\nmAP = %.3f' % mean_avg_precision)


if __name__ == '__main__':
    tf.app.run()
