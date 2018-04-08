"""
Based on: http://www.wildml.com/2016/08/rnns-in-tensorflow-a-practical-guide-and-undocumented-features/
"""

# Note: check ranges. tf.decode_jpeg=[0,1], ffmpeg=[0,255] (JPEG encodes [0,255] uint8 images)

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import ipdb
import random
import threading
from datetime import datetime

import numpy as np
import tensorflow as tf

from py_ops import decode_video
from parse_annotations import read_parsed_annot

# 11841 clips
tf.app.flags.DEFINE_string('videos_directory', '/home/allen/Downloads/UCF-101/', 'Video data directory')
tf.app.flags.DEFINE_string('input_file', '/home/allen/Downloads/UCF-101/traincliplist.txt', 'Text file with (filename, label) pairs')
tf.app.flags.DEFINE_string('annot_directory', '/home/allen/Downloads/UCF-101/annotation/UCF101_24Action_Detection_Annotations/UCF101_24_Annotations', 'Annotation directory')
tf.app.flags.DEFINE_string('output_directory', '/home/allen/Downloads/UCF-101/tf_records/', 'Output data directory')
tf.app.flags.DEFINE_string('name', 'UCF-24-train', 'Name for the subset')

tf.app.flags.DEFINE_integer('num_shards', 100, 'Number of shards. Each job will process num_shards/num_jobs shards.')
tf.app.flags.DEFINE_integer('num_threads', 2, 'Number of threads within this job to preprocess the videos.')
tf.app.flags.DEFINE_integer('num_jobs', 1, 'How many jobs will process this dataset.')
tf.app.flags.DEFINE_integer('job_id', 0, 'Job ID for the multi-job scenario. In range [0, num_jobs-1].')

tf.app.flags.DEFINE_integer('resize_h', 128, 'Height after resize. Set to -1 to keep original video height.')
tf.app.flags.DEFINE_integer('resize_w', 128, 'Width after resize. Set to -1 to keep original video width.')

FLAGS = tf.app.flags.FLAGS

def _int64_feature(value):
    """Wrapper for inserting int64 features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
    """Wrapper for inserting bytes features into Example proto."""
    if isinstance(value, list):
        value = value[0]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _maybe_resize_video(video):
    if FLAGS.resize_side_min == -1:
        return video
    else:
        shape = tf.shape(video)
        height = shape[1]
        width = shape[2]
        new_height, new_width = _smallest_size_at_least(height, width, FLAGS.resize_side_min)
        video = _aspect_preserving_resize(video, FLAGS.resize_side_min)
        video = tf.reshape(video, [-1, new_height, new_width, 3])
        return video


def _get_new_size(video):
    shape = tf.shape(video)
    height = shape[1]
    width = shape[2]
    new_height, new_width = _smallest_size_at_least(height, width, FLAGS.resize_side_min)
    return new_height, new_width


def bbx_to_string(bbx):
    n_obj = bbx.shape[0]
    s = ''
    for i in range(n_obj):
        s += str(bbx[i][0]) + ',' + str(bbx[i][0]) + ',' \
           + str(bbx[i][0]) + ',' + str(bbx[i][0])
        if i != n_obj - 1:
            s += ';'
    return s


def _convert_to_sequential_example(filename, video_buffer, label, text, height, width, sequence_length):
    """Build a SequenceExample proto for an example.
    Args:
        filename: string, path to a video file, e.g., '/path/to/example.avi'
        video_buffer: numpy array with the video frames, with dims [n_frames, height, width, n_channels]
        label: integer or list of integers, identifier for the ground truth for the network
        text: string, unique human-readable, e.g. 'dog'
        height: integer, image height in pixels
        width: integer, image width in pixels
        sequence_length: real length of the data, i.e. number of frames that are not zero-padding
    Returns:
        SequentialExample proto
    """

    # Create SequenceExample instance
    example = tf.train.SequenceExample()

    # Context features (non-sequential features)
    info        = filename.split('.')[0].split('_')
    start_frame = int(info[-2])
    end_frame   = int(info[-1])
    video_name  = '_'.join(info[:-2]).split('/')[-1]

    example.context.feature['height'].int64_list.value.append(height)
    example.context.feature['width'].int64_list.value.append(width)
    example.context.feature['sequence_length'].int64_list.value.append(sequence_length)
    example.context.feature['start_frame'].int64_list.value.append(start_frame)
    example.context.feature['end_frame'].int64_list.value.append(end_frame)
    example.context.feature['filename'].bytes_list.value.append(str.encode(filename))
    example.context.feature['video_name'].bytes_list.value.append(str.encode(video_name))
    example.context.feature['text'].bytes_list.value.append(str.encode(text))
    example.context.feature['label'].int64_list.value.append(label)

    bbx, beg, end = read_parsed_annot(os.path.join(FLAGS.annot_directory, text, video_name + '.txt'))

    # Sequential features
    frames = example.feature_lists.feature_list["frames"]
    bbxes  = example.feature_lists.feature_list["bbx"]

    for i in range(sequence_length):
        frames.feature.add().bytes_list.value.append(video_buffer[i])  # .tostring())
        bbxes.feature.add().bytes_list.value.append(str.encode(bbx_to_string(bbx[i+start_frame-beg])))

    return example


def _convert_to_example(filename, video_buffer, label, text, height, width, sequence_length):
    """Deprecated: use _convert_to_sequential_example instead
    Build an Example proto for an example.
    Args:
        filename: string, path to a video file, e.g., '/path/to/example.avi'
        video_buffer: numpy array with the video frames, with dims [n_frames, height, width, n_channels]
        label: integer or list of integers, identifier for the ground truth for the network
        text: string, unique human-readable, e.g. 'dog'
        height: integer, image height in pixels
        width: integer, image width in pixels
        sequence_length: real length of the data, i.e. number of frames that are not zero-padding
    Returns:
        Example proto
    """
    example = tf.train.Example(features=tf.train.Features(feature={
        'sequence_length': _int64_feature(sequence_length),
        'height': _int64_feature(height),
        'width': _int64_feature(width),
        'class/label': _int64_feature(label),
        'class/text': _bytes_feature(text),
        'filename': _bytes_feature(os.path.basename(filename)),
        'frames': _bytes_feature(video_buffer.tostring())}))

    return example


class VideoCoder(object):
    """Helper class that provides TensorFlow image coding utilities."""

    def __init__(self):
        # Create a single Session to run all image coding calls.
        self._sess = tf.Session()

        # Initializes function that decodes video
        self._video_path = tf.placeholder(dtype=tf.string)
        self._decode_video = decode_video(self._video_path)

        # Initialize function that resizes a frame
        self._resize_video_data = tf.placeholder(dtype=tf.float32, shape=[None, None, None, 3])

        # Initialize function to JPEG-encode a frame
        self._raw_frame = tf.placeholder(dtype=tf.uint8, shape=[None, None, 3])
        self._encode_frame = tf.image.encode_jpeg(self._raw_frame, quality=100)

    def _get_new_dims(self):
        shape = tf.shape(self._resize_video_data)
        seq_length = shape[0]
        return seq_length

    def _resize_video(self, seq_length, new_height, new_width):

        resized_video = tf.image.resize_bilinear(self._resize_video_data, [new_height, new_width],
                                                 align_corners=False)
        resized_video.set_shape([seq_length, new_height, new_width, 3])
        return resized_video

    def decode_video(self, video_data):
        video, _, _, seq_length = self._sess.run(self._decode_video,
                                                 feed_dict={self._video_path: video_data})
        #ipdb.set_trace()
        # video /= 255.
        if FLAGS.resize_w != -1:
            seq_length = self._sess.run(self._get_new_dims(),
                                        feed_dict={self._resize_video_data: video})
            new_height = FLAGS.resize_h
            new_width  = FLAGS.resize_w
            video = self._sess.run(self._resize_video(seq_length, new_height, new_width),
                                   feed_dict={self._resize_video_data: video})
        assert len(video.shape) == 4
        assert video.shape[3] == 3
        return video, seq_length

    def encode_frame(self, raw_frame):
        return self._sess.run(self._encode_frame, feed_dict={self._raw_frame: raw_frame})


def _process_video(filename, coder):
    """
    Process a single video file using FFmpeg
    Args
        filename: path to the video file
        coder: instance of ImageCoder to provide TensorFlow image coding utils.
    Returns:
        video_buffer: numpy array with the video frames
        height: integer, video height in pixels.
        width: integer, width width in pixels.
        seq_length: sequence length (non-zero frames)
    """
    video, seq_length = coder.decode_video(filename)
    video = video.astype(np.uint8)
    encoded_frames_seq = []
    for idx in range(seq_length):
        encoded_frames_seq.append(coder.encode_frame(video[idx, :, :, :]))
    assert len(video.shape) == 4
    height = video.shape[1]
    width = video.shape[2]
    assert video.shape[3] == 3
    return encoded_frames_seq, height, width, np.asscalar(seq_length)


def _process_video_files_batch(coder, thread_index, ranges, name, filenames, texts, labels, num_shards,
                               job_index, num_jobs):
    """
    Process and save list of videos as TFRecord in 1 thread.
    Args:
        coder: instance of VideoCoder to provide TensorFlow video coding utils.
        thread_index: integer, unique batch to run index is within [0, len(ranges)).
        ranges: list of pairs of integers specifying ranges of each batch to
          analyze in parallel.
        name: string, unique identifier specifying the data set
        filenames: list of strings; each string is a path to a video file
        texts: list of strings; each string is human readable, e.g. 'dog'
        labels: list of integer; each integer identifies the ground truth
        num_shards: integer number of shards for this data set.
        job_index: integer, unique job index in range [0, num_jobs-1]
        num_jobs: how many different jobs will process the same data
"""
    assert not num_shards % num_jobs
    num_shards_per_job = num_shards / num_jobs
    # Each thread produces N shards where N = int(num_shards_per_job / num_threads).
    # For instance, if num_shards_per_job = 128, and the num_threads = 2, then the first
    # thread would produce shards [0, 64).
    num_threads = len(ranges)
    assert not num_shards_per_job % num_threads
    num_shards_per_batch = int(num_shards_per_job / num_threads)

    shard_ranges = np.linspace(ranges[thread_index][0], ranges[thread_index][1], num_shards_per_batch + 1).astype(int)
    num_files_in_thread = ranges[thread_index][1] - ranges[thread_index][0]

    counter = 0
    for s in range(num_shards_per_batch):
        # Generate a sharded version of the file name, e.g. 'train-00002-of-00010'
        shard = thread_index * num_shards_per_batch + job_index * num_shards_per_job + s
        output_filename = '%s-%.5d-of-%.5d' % (name, shard, num_shards)
        output_file = os.path.join(FLAGS.output_directory, output_filename)
        writer = tf.python_io.TFRecordWriter(output_file)

        shard_counter = 0
        files_in_shard = np.arange(shard_ranges[s], shard_ranges[s + 1], dtype=int)
        for i in files_in_shard:
            filename = filenames[i]
            label = labels[i]
            text = texts[i]

            video_buffer, height, width, seq_length = _process_video(filename, coder)

            if seq_length == 0:
                print('Skipping video with null length')
                continue

            assert(seq_length == 33)
            video_buffer = video_buffer[:32]
            seq_length = 32

            example = _convert_to_sequential_example(filename, video_buffer, label, text, height, width, seq_length)
            writer.write(example.SerializeToString())
            shard_counter += 1
            counter += 1

            if not counter % 100:
                print('%s [thread %d]: Processed %d of %d videos in thread batch.' %
                      (datetime.now(), thread_index, counter, num_files_in_thread))
                sys.stdout.flush()

        print('%s [thread %d]: Wrote %d video chunks to %s' %
              (datetime.now(), thread_index, shard_counter, output_file))
        sys.stdout.flush()
        shard_counter = 0
    print('%s [thread %d]: Wrote %d video chunks to %d shards.' %
          (datetime.now(), thread_index, counter, num_files_in_thread))
    sys.stdout.flush()


def _process_video_files(name, filenames, texts, labels, num_shards, job_index, num_jobs):
    """
    Process and save list of videos as TFRecord of Example protos.
    Args:
        name: string, unique identifier specifying the data set
        filenames: list of strings; each string is a path to a video file
        texts: list of strings; each string is human readable, e.g. 'dog'
        labels: list of integer; each integer identifies the ground truth
        num_shards: integer number of shards for this data set.
        job_index: integer, unique job index in range [0, num_jobs-1]
        num_jobs: how many different jobs will process the same data
    """
    assert len(filenames) == len(texts)
    assert len(filenames) == len(labels)

    # Break all examples into batches in two levels: first for jobs, then for threads within each job
    num_files = len(filenames)
    num_files_per_job = int(num_files / num_jobs)
    first_file = job_index * num_files_per_job
    last_file = min(num_files, (job_index + 1) * num_files_per_job)
    print('Job #%d will process files in range [%d,%d]' % (job_index, first_file, last_file - 1))
    local_filenames = filenames[first_file:last_file]
    local_texts = texts[first_file:last_file]
    local_labels = labels[first_file:last_file]
    spacing = np.linspace(0, len(local_filenames), FLAGS.num_threads + 1).astype(np.int)
    ranges = []
    threads = []
    for i in range(len(spacing) - 1):
        ranges.append([spacing[i], spacing[i+1]])

    # Launch a thread for each batch.
    print('Launching %d threads for spacings: %s' % (FLAGS.num_threads, ranges))
    sys.stdout.flush()

    # Create a mechanism for monitoring when all threads are finished.
    coord = tf.train.Coordinator()

    # Create a generic TensorFlow-based utility for converting all image codings.
    coder = VideoCoder()

    threads = []
    for thread_index in range(len(ranges)):
        args = (coder, thread_index, ranges, name, local_filenames, local_texts, local_labels, num_shards,
                job_index, num_jobs)
        t = threading.Thread(target=_process_video_files_batch, args=args)
        t.start()
        threads.append(t)

    # Wait for all the threads to terminate.
    coord.join(threads)
    print('%s: Finished writing all %d images in data set.' %
          (datetime.now(), len(local_filenames)))
    sys.stdout.flush()


def _parse_line_ucf101(line):
    filepath, class_id = line.split()[0], line.split()[1]
    classname = filepath.split('/')[0]
    return filepath, classname, class_id


def _find_video_files(input_file, dataset_dir):
    """Build a list of all images files and labels in the data set.
    Args:
        input_file: path to the file list
        dataset_dir: base path to dataset
    Returns:
        filepaths: list of strings; each string is a path to a video file.
        texts: list of string; each string is the class name, e.g. 'playing_football'
        labels: list of integer; each integer identifies the ground truth label id
    """
    lines = [line.strip() for line in open(input_file, 'r')]
    filepaths = list()
    texts = list()
    labels = list()
    for line in lines:
        filepath, classname, class_id = _parse_line_ucf101(line)
        filepaths.append(os.path.join(dataset_dir, filepath))
        labels.append(int(class_id))
        texts.append(classname)

    # Shuffle the ordering of all video files in order to guarantee random ordering of the images with respect to
    # label in the saved TFRecord files. Make the randomization repeatable.
    shuffled_index = list(range(len(filepaths)))
    random.seed(12345)
    random.shuffle(shuffled_index)

    filepaths = [filepaths[i] for i in shuffled_index]
    texts = [texts[i] for i in shuffled_index]
    labels = [labels[i] for i in shuffled_index]

    print('Found %d video files.' % len(filepaths))

    return filepaths, texts, labels


def _process_dataset(name, input_file, dataset_dir, num_shards, job_index, num_jobs):
    """Process a complete data set and save it as a TFRecord.
    Args:
        name: string, unique identifier specifying the data set.
        input_file: path to the file listing (path, anp_label, noun_label, adj_label) tuples
        num_shards: integer number of shards for this data set.
        job_index: integer, unique job index in range [0, num_jobs-1]
        num_jobs: how many different jobs will process the same data
    """
    filenames, texts, labels = _find_video_files(input_file, dataset_dir)
    _process_video_files(name, filenames, texts, labels, num_shards, job_index, num_jobs)


def main(unused_argv):
    assert not int(FLAGS.num_shards / FLAGS.num_jobs) % FLAGS.num_threads, (
        'Please make the FLAGS.num_threads commensurate with FLAGS.num_shards and FLAGS.num_jobs')
    print('Saving results to %s' % FLAGS.output_directory)

    # Run it!
    _process_dataset(FLAGS.name, FLAGS.input_file, FLAGS.videos_directory, FLAGS.num_shards, FLAGS.job_id, FLAGS.num_jobs)


if __name__ == '__main__':
    main(None)
