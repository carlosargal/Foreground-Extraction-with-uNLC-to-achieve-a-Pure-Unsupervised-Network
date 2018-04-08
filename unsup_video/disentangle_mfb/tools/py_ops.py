"""
Custom TensorFlow Python Ops.

    - decode_video(): opens and decodes a video file using FFmpeg.

"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import random
import numpy as np
import tensorflow as tf

from ffmpeg_reader import FFMPEG_VideoReader


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('num_frames', 0,
                            """Number of frames per video (sequence length). Set to 0 for full video.""")
tf.app.flags.DEFINE_boolean('random_chunks', False,
                            """Grab video frames starting from a random position.""")
tf.app.flags.DEFINE_float('fps', -1,
                          """framerate to which the input videos are converted. Use -1 for the original framerate.""")


def _load_video_ffmpeg(filename):
    """
    Load a video as a numpy array using FFmpeg in [0, 255] RGB format.
    :param filename: path to the video file
    :param random_chunk: grab frames starting from a random position
    :return: (video, length) tuple
        video: (n_frames, h, w, 3) numpy array containing video frames, as RGB in range [0, 255]
        height: frame height
        width: frame width
        length: number of non-zero frames loaded from the video (the rest of the sequence is zero-padded)
    """
    if isinstance(filename, bytes):
        filename = filename.decode('utf-8')

    n_frames = FLAGS.num_frames
    random_chunk = FLAGS.random_chunks
    target_fps = -1 #FLAGS.fps

    # Get video params
    video_reader = FFMPEG_VideoReader(filename, target_fps=target_fps)
    w, h = video_reader.size
    fps = video_reader.fps
    if target_fps <= 0:
        target_fps = fps

    video_length = int(video_reader.nframes * target_fps / fps)  # corrected number of frames

    # Determine starting and ending positions
    if n_frames <= 0 or video_length < n_frames:
        n_frames = video_length
    elif random_chunk:  # start from a random position
        start_pos = random.randint(0, video_length - n_frames - 1)
        video_reader.get_frame(1. * start_pos / target_fps, fps=target_fps)

    # Load video chunk as numpy array
    video = np.zeros((n_frames, h, w, 3), dtype=np.float32)
    for idx in range(n_frames):
        video[idx, :, :, :] = video_reader.read_frame()[:, :, :3].astype(np.float32)

    video_reader.close()

    return video, h, w, n_frames


def decode_video(filename):
    """
    Decode frames from a video. Returns frames in [0, 255] RGB format.
    :param filename: string tensor, e.g. dequeue() op from a filenames queue
    :return:
        video: 4-D tensor containing frames of a video: [time, height, width, channel]
        height: frame height
        width: frame width
        length: number of non-zero frames loaded from the video (the rest of the sequence is zero-padded)
    """
    return tf.py_func(_load_video_ffmpeg, [filename], [tf.float32, tf.int64, tf.int64, tf.int64], name='decode_mp4')


if __name__ == '__main__':
    import time
    sess = tf.Session()
    f = tf.placeholder(tf.string)
    video, seq_length = decode_video(f)
    start_time = time.time()
    video_val, seq_length_val = sess.run([video, seq_length],
                                         feed_dict={f: '/gpfs/scratch/bsc31/bsc31953/sports1m/test/v_FzfzSgXuzRg.mp4'})
    total_time = time.time() - start_time
    print('\nSuccessfully loaded video! \n\tDimensions: ', video_val.shape, '\n\tTime: %.3fs' % total_time,
          '\n\tLoaded frames: ', seq_length_val)
    np.save('decoded_frames_%.0f_fps.npy' % FLAGS.fps, video_val)
