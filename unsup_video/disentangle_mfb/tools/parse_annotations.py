import xml.etree.ElementTree
import numpy as np
import ipdb


def parse_annot(annot_path, parsed_path):

    f = open(parsed_path, 'w')

    viper = xml.etree.ElementTree.parse(annot_path).getroot()
    data  = viper[1]
    assert('data' in data.tag)

    n_obj, i = 0, 0
    while n_obj == 0:   # find first non-empty sourcefile
        source   = data[i]
        obj_list = source[1:]
        n_obj    = len(obj_list)
        i       += 1
    beg, end = -1, -1
    for obj in obj_list:
        _beg = int(obj.attrib['framespan'].split(':')[0])   # first frame appeared
        _end = int(obj.attrib['framespan'].split(':')[-1])  # last frame appeared
        if beg == -1: beg, end = _beg, _end
        if _beg < beg: beg = _beg
        if _end > end: end = _end
    f.write(str(beg) + ':' + str(end))
    n_frames = end - beg + 1
    annot = np.zeros([n_obj, n_frames, 4])

    for i, obj in enumerate(obj_list):
        loc = obj[0]
        for bbox in loc:
            span  = bbox.attrib['framespan'].split(':')
            h     = int(bbox.attrib['height'])
            w     = int(bbox.attrib['width'])
            x     = int(bbox.attrib['x'])
            y     = int(bbox.attrib['y'])
            start = int(span[0]) - beg
            end   = int(span[1]) - beg + 1
            for j in range(start, end):
                if j >= n_frames: break
                annot[i, j] = [h, w, x, y]
                
    for i in range(n_frames):
        line = '\n' + str(i+beg) + ':'
        for j in range(n_obj):
            line += str(int(annot[j, i, 0])) + ',' + str(int(annot[j, i, 1])) + ',' \
                  + str(int(annot[j, i, 2])) + ',' + str(int(annot[j, i, 3]))
            if j != n_obj - 1:
                line += ';'
        f.write(line)

    f.close()


def read_parsed_annot(parsed_path):

    f = open(parsed_path, 'r')

    lines    = f.readlines()
    beg, end = int(lines[0].split(':')[0]), int(lines[0].split(':')[1])
    n_frames = end - beg + 1
    n_obj    = len(lines[1].split(':')[1].split(';'))
    bbx      = np.zeros([n_frames, n_obj, 4], dtype=np.float32)

    for i, line in enumerate(lines[1:]):
        infos = line.split(':')[1].split(';')
        for j in range(n_obj):
            info = infos[j].split(',')
            h, w, x, y = int(info[0]), int(info[1]), int(info[2]), int(info[3])
            bbx[i, j]  = [h, w, x, y]

    f.close()

    return bbx, beg, end


def resize_parsed_annot(bbx, h, w, resize_h, resize_w):

    ratio_h = resize_h / float(h)
    ratio_w = resize_w / float(w)
    resize_bbx = np.zeros_like(bbx, dtype=np.int)
    resize_bbx[:, :, 1:3] = bbx[:, :, 1:3] * ratio_w
    resize_bbx[:, :, 0]   = bbx[:, :, 0] * ratio_h
    resize_bbx[:, :, 3]   = bbx[:, :, 3] * ratio_h
    
    return resize_bbx

