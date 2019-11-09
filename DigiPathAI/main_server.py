#!/usr/bin/env python
#
from collections import OrderedDict
from flask import Flask, abort, make_response, render_template, url_for
from flask import flash
from io import BytesIO
import openslide
from openslide import OpenSlide, OpenSlideError
from openslide.deepzoom import DeepZoomGenerator
import os
from optparse import OptionParser
from threading import Lock
import json
import threading
import time
from queue import Queue 

import sys
sys.path.append('..')
from DigiPathAI.Segmentation import getSegmentation

SLIDE_DIR = '.'
SLIDE_CACHE_SIZE = 10
DEEPZOOM_FORMAT = 'jpeg'
DEEPZOOM_TILE_SIZE = 254
DEEPZOOM_OVERLAP = 1
DEEPZOOM_LIMIT_BOUNDS = True
DEEPZOOM_TILE_QUALITY = 75

app = Flask(__name__)
app.config.from_object(__name__)
app.config.from_envvar('DEEPZOOM_MULTISERVER_SETTINGS', silent=True)


class PILBytesIO(BytesIO):
    def fileno(self):
        '''Classic PIL doesn't understand io.UnsupportedOperation.'''
        raise AttributeError('Not supported')


class _SlideCache(object):
    def __init__(self, cache_size, dz_opts):
        self.cache_size = cache_size
        self.dz_opts = dz_opts
        self._lock = Lock()
        self._cache = OrderedDict()

    def get(self, path):
        with self._lock:
            if path in self._cache:
                # Move to end of LRU
                slide = self._cache.pop(path)
                self._cache[path] = slide
                return slide

        osr = OpenSlide(path)
        slide = DeepZoomGenerator(osr, **self.dz_opts)
        try:
            mpp_x = osr.properties[openslide.PROPERTY_NAME_MPP_X]
            mpp_y = osr.properties[openslide.PROPERTY_NAME_MPP_Y]
            slide.mpp = (float(mpp_x) + float(mpp_y)) / 2
        except (KeyError, ValueError):
            slide.mpp = 0

        with self._lock:
            if path not in self._cache:
                if len(self._cache) == self.cache_size:
                    self._cache.popitem(last=False)
                self._cache[path] = slide
        return slide

class _Directory(object):
    def __init__(self, basedir, relpath=''):
        self.name = os.path.basename(relpath)
        self.children = []
        self.children_masks = []
        for name in sorted(os.listdir(os.path.join(basedir, relpath))):
            cur_relpath = os.path.join(relpath, name)
            cur_path = os.path.join(basedir, cur_relpath)
            if os.path.isdir(cur_path):
                cur_dir = _Directory(basedir, cur_relpath)
                if cur_dir.children:
                    self.children.append(cur_dir)
            elif OpenSlide.detect_format(cur_path):
                if 'slide' in os.path.basename(cur_path):
                    #liver-slide-1-slide.tiff -> liver-slide-1-mask.tiff
                    if mask_exists(cur_path):
                        self.children.append(_SlideFile(cur_relpath,True))
                    else:
                        self.children.append(_SlideFile(cur_relpath,False))

class _SlideFile(object):
    def __init__(self, relpath, mask_present):
        self.name = os.path.basename(relpath)
        self.url_path = relpath
        self.mask_present = mask_present


@app.before_first_request
def _setup():
    app.basedir = os.path.abspath(app.config['SLIDE_DIR'])
    config_map = {
        'DEEPZOOM_TILE_SIZE': 'tile_size',
        'DEEPZOOM_OVERLAP': 'overlap',
        'DEEPZOOM_LIMIT_BOUNDS': 'limit_bounds',
    }
    opts = dict((v, app.config[k]) for k, v in config_map.items())
    app.cache = _SlideCache(app.config['SLIDE_CACHE_SIZE'], opts)
    app.segmentation_status = {"status":""}



def mask_exists(path):
    mask_path = '-'.join(path.split('-')[:-1]+["mask"])+'.'+path.split('.')[-1]
    if os.path.isfile(mask_path):
        return True
    else:
        return False


def get_mask_path(path):
    mask_path =  '-'.join(path.split('-')[:-1]+["mask"])+'.'+path.split('.')[-1]
    return mask_path


def _get_slide(path):
    path = os.path.abspath(os.path.join(app.basedir, path))
    if not path.startswith(app.basedir + os.path.sep):
        # Directory traversal
        abort(404)
    if not os.path.exists(path):
        abort(404)
    try:
        slide = app.cache.get(path)
        slide.filename = os.path.basename(path)
        return slide
    except OpenSlideError:
        abort(404)

@app.route('/')
def index():
    return render_template('files.html', root_dir=_Directory(app.basedir))


@app.route('/segment')
def segment():
    x = threading.Thread(target=run_segmentation, args=(app.segmentation_status,))
    x.start()
    return app.segmentation_status


def run_segmentation(status):
    status['status'] = "Running"
    print(status)
    print("Starting segmentation")
    getSegmentation(img_path = status['slide_path'],
                save_path = get_mask_path(status['slide_path']),
                status = status)
    time.sleep(0.1)
    print("Segmentation done")
    status['status'] = "Done"


@app.route('/check_segment_status')
def check_segment_status():
    return app.segmentation_status


@app.route('/<path:path>')
def slide(path):
    slide= _get_slide(path)
    slide_url = url_for('dzi', path=path)
    path = os.path.abspath(os.path.join(app.basedir, path))
    app.segmentation_status['slide_path'] = path
    mask_status = mask_exists(path)
    print(slide_url)
    return render_template('viewer.html', slide_url=slide_url,mask_status=mask_status,
            slide_filename=slide.filename, slide_mpp=slide.mpp, root_dir=_Directory(app.basedir) )


@app.route('/<path:path>.dzi')
def dzi(path):
    slide = _get_slide(path)
    format = app.config['DEEPZOOM_FORMAT']
    resp = make_response(slide.get_dzi(format))
    resp.mimetype = 'application/xml'
    return resp

@app.route('/<path:path>_files/<int:level>/<int:col>_<int:row>.<format>')
def tile(path, level, col, row, format):
    slide = _get_slide(path)
    format = format.lower()
    if format != 'jpeg' and format != 'png':
        # Not supported by Deep Zoom
        abort(404)
    try:
        tile = slide.get_tile(level, (col, row))
    except ValueError:
        # Invalid level or coordinates
        abort(404)
    buf = PILBytesIO()
    tile.save(buf, format, quality=app.config['DEEPZOOM_TILE_QUALITY'])
    resp = make_response(buf.getvalue())
    resp.mimetype = 'image/%s' % format
    return resp


parser = OptionParser(usage='Usage: %prog [options] [slide-directory]')
parser.add_option('-B', '--ignore-bounds', dest='DEEPZOOM_LIMIT_BOUNDS',
            default=True, action='store_false',
            help='display entire scan area')
parser.add_option('-c', '--config', metavar='FILE', dest='config',
            help='config file')
parser.add_option('-d', '--debug', dest='DEBUG', action='store_true',
            help='run in debugging mode (insecure)')
parser.add_option('-e', '--overlap', metavar='PIXELS',
            dest='DEEPZOOM_OVERLAP', type='int',
            help='overlap of adjacent tiles [1]')
parser.add_option('-f', '--format', metavar='{jpeg|png}',
            dest='DEEPZOOM_FORMAT',
            help='image format for tiles [jpeg]')
parser.add_option('-l', '--listen', metavar='ADDRESS', dest='host',
            default='127.0.0.1',
            help='address to listen on [127.0.0.1]')
parser.add_option('-p', '--port', metavar='PORT', dest='port',
            type='int', default=8080,
            help='port to listen on [8080]')
parser.add_option('-Q', '--quality', metavar='QUALITY',
            dest='DEEPZOOM_TILE_QUALITY', type='int',
            help='JPEG compression quality [75]')
parser.add_option('-s', '--size', metavar='PIXELS',
            dest='DEEPZOOM_TILE_SIZE', type='int',
            help='tile size [254]')

def main():
    (opts, args) = parser.parse_args()
    # Load config file if specified
    if opts.config is not None:
        app.config.from_pyfile(opts.config)
    # Overwrite only those settings specified on the command line
    for k in dir(opts):
        if not k.startswith('_') and getattr(opts, k) is None:
            delattr(opts, k)
    app.config.from_object(opts)
    # Set slide directory
    try:
        app.config['SLIDE_DIR'] = args[0]
    except IndexError:
        pass

    app.run(host=opts.host, port=opts.port,debug=True, threaded=True)

if __name__ == "__main__":
    main()
