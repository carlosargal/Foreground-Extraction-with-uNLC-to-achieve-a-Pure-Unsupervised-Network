"""
This file implements following paper:
Video Segmentation by Non-Local Consensus Voting
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# from __future__ import unicode_literals
import sys
import time
from PIL import Image
import numpy as np
from numpy.ma.core import _arraymethod
from scipy.misc import imresize
import scipy.misc

import _init_paths  # noqa
import os
import utils
import nlc
import vid2shots
import crf
import cv2

def parse_args():
    """
    Parse input arguments
    """
    import argparse
    parser = argparse.ArgumentParser(description='Foreground Segmentation using Non-Local Consensus')
    parser.add_argument(
        '-out', dest='baseOutdir',
        help='Base directory to save output.',
        default='/imatge/carenas/research/uNLC/dataset/uNLC_video_segmented_document', type=str)
    parser.add_argument(
        '-in', dest='imdirFile',
        help='Addresses of file containing list of video image directories.' +
        ' Each imdir will be read alphabetically for video image sequence.',
        type=str)
    parser.add_argument(
        '-numShards', dest='numShards',
        help='Number of shards for parallel running',
        default=1, type=int)
    parser.add_argument(
        '-shardId', dest='shardId',
        help='Shard to work on. Should range between 0 and numShards-1',
        default=0, type=int)
    parser.add_argument(
        '-doload', dest='doload',
        help='load from .npy files already existing run. 0 or 1. Default 0.',
        default=0, type=int)
    parser.add_argument(
        '-dosave', dest='dosave',
        help='save .npy files at each important step Takes lot of space.' +
        ' 0 or 1. Default 0.',
        default=0, type=int)
    parser.add_argument(
        '-crfParams', dest='crfParams',
        help='CRF Params: default=0, deeplab=1, ccnn=2. Default is 0.',
        default=0, type=int)
    parser.add_argument(
        '-seed', dest='seed',
        help='Random seed for numpy and python.', default=2905, type=int)

    args = parser.parse_args()
    return args

def mkdirnotex(filename):
    if not os.path.exists(filename):
        os.makedirs(filename)

def demo_images(imdir, imSequence):

    # For Shot:
    maxShots = 5
    vmax = 0.6
    colBins = 40

    # For NLC:
    redirect = True  # redirecting to output file ? won't print status
    frameGap = 0  # 0 means adjusted automatically per shot (not per video)
    maxSide = 650  # max length of longer side of Im
    minShot = 10  # minimum shot length
    maxShot = 110  # longer shots will be shrinked between [maxShot/2, maxShot]
    binTh = 0.7  # final thresholding to obtain mask
    clearVoteBlobs = True  # remove small blobs in consensus vote; uses binTh
    relEnergy = binTh - 0.1  # relative energy in consensus vote blob removal
    clearFinalBlobs = True  # remove small blobs finally; uses binTh
    maxsp = 10
    iters = 25

    # For CRF:
    gtProb = 0.7
    posTh = binTh
    negTh = 0.4

    # For blob removal post CRF: more like salt-pepper noise removal
    bSize = 25  # 0 means not used, [0,1] relative, >=1 means absolute

    # parse commandline parameters
    args = parse_args()
    np.random.seed(args.seed)
    doload = bool(args.doload) # Default is False
    dosave = bool(args.dosave) # Default is False

    # keep only the current shard
    if args.shardId >= args.numShards:
        print('Give valid shard id which is less than numShards')
        exit(1)

    print('NUM SHARDS: %03d,  SHARD ID: %03d\n\n' % (args.numShards, args.shardId))

    # setup output directory
    suf = imdir.split('/')[-1]
    suffix = suf.split('.')[-2]

    outNlcIm = args.baseOutdir.split('/') + ['nlcim', imdir.split('/')[3], suffix]
    outNlcPy = args.baseOutdir.split('/') + ['nlcpy', imdir.split('/')[3], suffix]
    outCrf = args.baseOutdir.split('/') + ['crfim', imdir.split('/')[3], suffix]
    outIm = args.baseOutdir.split('/') + ['im', imdir.split('/')[3], suffix]

    outNlcIm = '/'.join(outNlcIm)
    outNlcPy = '/'.join(outNlcPy)
    outCrf = '/'.join(outCrf)
    outIm = '/'.join(outIm)
    outVidNlc = args.baseOutdir + '/nlcvid/'
    outVidCRF = args.baseOutdir + '/crfvid/'

    mkdirnotex(outNlcIm)
    mkdirnotex(outNlcPy)
    mkdirnotex(outCrf)
    mkdirnotex(outIm)
    mkdirnotex(outVidNlc)
    mkdirnotex(outVidCRF)

    print('Video OutputDir: ', outNlcIm)

    # resize images if needed
    n, h, w, c = imSequence.shape
    print('Number of frames in the entire video: %d' % (n))
    frac = min(min(1. * maxSide / h, 1. * maxSide / w), 1.0)
    if frac < 1.0:
        n, h, w, c = imresize(imSequence, frac).shape
    imSeq = np.zeros((n, h, w, c), dtype=np.uint8)
    for i in range(n):
        if frac < 1.0:
            imSeq[i] = imresize(imSequence[i], frac)
        else:
            imSeq[i] = imSequence[i]

    final_seq = np.empty((0, h, w, c), dtype=np.uint8)
    final_mask = np.empty((0, h, w), dtype=np.uint8)

    # First run shot detector
    if not doload:
        shotIdx = vid2shots.vid2shots(imSeq, maxShots=maxShots, vmax=vmax, colBins=colBins)
        if dosave:
            np.save(outNlcPy + '/shotIdx_%s.npy' % suffix, shotIdx)
    else:
        shotIdx = np.load(outNlcPy + '/shotIdx_%s.npy' % suffix)
    print('Total Shots: ', shotIdx.shape, shotIdx)
    # Adjust frameGap per shot, and then run NLC per shot
    for s in range(shotIdx.shape[0]):
        suffixShot = suffix + '_shot%d' % (s + 1)

        shotS = shotIdx[s]  # 0-indexed, included
        shotE = imSeq.shape[0] if s == shotIdx.shape[0] - 1 \
            else shotIdx[s + 1]  # 0-indexed, excluded
        shotL = shotE - shotS
        if shotL < minShot:
            continue

        frameGapLocal = frameGap
        if frameGapLocal <= 0 and shotL > maxShot:
            frameGapLocal = int(shotL / maxShot)
        imSeq1 = imSeq[shotS:shotE:frameGapLocal + 1]

        print('\nShot: %d, Shape: ' % (s + 1), imSeq1.shape)
        if not doload:
            maskSeq = nlc.nlc(imSeq1, maxsp=maxsp, iters=iters,
                                outdir=outNlcPy, suffix=suffixShot,
                                clearBlobs=clearVoteBlobs, binTh=binTh,
                                relEnergy=relEnergy,
                                redirect=redirect, doload=doload,
                                dosave=dosave)
            if clearFinalBlobs:
                maskSeq = nlc.remove_low_energy_blobs(maskSeq, binTh)
            if dosave:
                np.save(outNlcPy + '/mask_%s.npy' % suffixShot, maskSeq)
        if doload:
            maskSeq = np.load(outNlcPy + '/mask_%s.npy' % suffixShot)

        print('Number of frames in maskSeq: %d' % (maskSeq.shape[0]))
        # run crf, run blob removal and save as images sequences
        sTime = time.time()
        crfSeq = np.zeros(maskSeq.shape, dtype=np.uint8)
        maskSeq_bin = (maskSeq > binTh).astype(np.uint8)

        for i in range(maskSeq.shape[0]):
            # save soft score as png between 0 to 100.
            # Use binTh*100 to get FG in later usage.
            mask = (maskSeq[i] * 100).astype(np.uint8)
            mask_bin = (maskSeq_bin[i] * 100).astype(np.uint8)

            save_mask = outNlcIm + '/' + suffix + '_%d.png' % (i)
            save_mask_bin = outNlcIm + '/' + suffix + '_bin_%d.png' % (i)
            save_imSeq = outIm + '/' + suffix + '_%d.png' % (i)
            Image.fromarray(mask).save(save_mask)
            Image.fromarray(mask_bin).save(save_mask_bin)
            Image.fromarray(imSeq1[i]).save(save_imSeq)
            crfSeq[i] = crf.refine_crf(imSeq1[i], maskSeq[i], gtProb=gtProb, posTh=posTh, negTh=negTh, crfParams=args.crfParams)
            crfSeq[i] = utils.refine_blobs(crfSeq[i], bSize=bSize)
            save_crfSeq = outCrf + '/' + suffix + '_%d.png' % (i)
            Image.fromarray(crfSeq[i] * 100).save(save_crfSeq)
            if not redirect:
                sys.stdout.write(
                    'CRF, blob removal and saving: [% 5.1f%%]\r' %
                    (100.0 * float((i + 1) / maskSeq.shape[0])))
                sys.stdout.flush()
        eTime = time.time()
        print('CRF, blob removal and saving images finished: %.2f s' %
                (eTime - sTime))

        final_seq = np.concatenate((final_seq, imSeq1), axis=0)
        final_mask = np.concatenate((final_mask, maskSeq_bin), axis=0)

        # save as video
        sTime = time.time()
        vidName = suffix + '_shot%d.avi' % (s + 1)
        utils.im2vid(outVidNlc + vidName, imSeq1, (maskSeq > binTh).astype(np.uint8))
        utils.im2vid(outVidCRF + vidName, imSeq1, crfSeq)
        eTime = time.time()
        print('Saving videos finished: %.2f s' % (eTime - sTime))

    # Tarzip the results of this shard and delete the individual files
    import subprocess
    for i in ['im', 'crfim', 'nlcim']:
        tarDir = args.baseOutdir.split('/') + ['%s' % (i), imdir.split('/')[3], suffix]
        tarDir = '/'.join(tarDir)
        subprocess.call(['tar', '-zcf', tarDir + '.tar.gz', '-C', tarDir, '.'])
        # utils.rmdir_f(tarDir)

    seq_length = final_seq.shape[0]
    return final_seq, final_mask, seq_length

if __name__ == "__main__":
    demo_images()
