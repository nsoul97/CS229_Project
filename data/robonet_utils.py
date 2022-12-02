import h5py
import imageio
import pandas as pd
import numpy as np
import glob
import os
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import hashlib
import io
import random
import cv2


class MetaDataContainer:
    def __init__(self, base_path, meta_data):
        self._meta_data = meta_data
        self._base_path = base_path

    def get_file_metadata(self, fname):
        fname = fname.split('/')[-1]
        return self._meta_data.loc[fname]

    def select_objects(self, obj_class_name):
        if isinstance(obj_class_name, str):
            return self._meta_data[[obj_class_name in x for x in self._meta_data['object_classes']]]
        return self._meta_data[[set(obj_class_name) == set(x) for x in self._meta_data['object_classes']]]

    @property
    def frame(self):
        return self._meta_data

    @property
    def files(self):
        return ['{}/{}'.format(self._base_path, f) for f in self.frame.index]

    def get_shuffled_files(self, rng=None):
        files = ['{}/{}'.format(self._base_path, f) for f in self.frame.index]
        if rng:
            rng.shuffle(files)
        else:
            random.shuffle(files)
        return files

    @property
    def base_path(self):
        return self._base_path

    def __getitem__(self, arg):
        return MetaDataContainer(self._base_path, self._meta_data[arg])

    def __contains__(self, item):
        return item in self._meta_data

    def __repr__(self):
        return repr(self._meta_data)

    def __str__(self):
        return str(self._meta_data)

    def __eq__(self, other):
        return self._meta_data == other

    def __ne__(self, other):
        return self._meta_data != other

    def __lt__(self, other):
        return self._meta_data < other

    def __le__(self, other):
        return self._meta_data <= other

    def __gt__(self, other):
        return self._meta_data > other

    def __ge__(self, other):
        return self._meta_data >= other

    def keys(self):
        return self._meta_data.keys()

    def __len__(self):
        return len(self._meta_data)


def load_metadata_dict(fname):
    if not os.path.exists(fname) or not os.path.isfile(fname):
        raise IOError("can't find {}".format(fname))
    buf = open(fname, 'rb').read()

    with h5py.File(io.BytesIO(buf)) as hf:
        meta_data_dict = {'file_version': hf['file_version'][()]}

        meta_data_dict['sha256'] = hashlib.sha256(buf).hexdigest()
        meta_data_dict['sdim'] = hf['env']['state'].shape[1]
        meta_data_dict['state_T'] = hf['env']['state'].shape[0]

        meta_data_dict['adim'] = hf['policy']['actions'].shape[1]
        meta_data_dict['action_T'] = hf['policy']['actions'].shape[0]

        # assumes all cameras have same attributes (if they exist)
        n_cams = hf['env'].attrs.get('n_cams', 0)
        if n_cams:
            meta_data_dict['ncam'] = n_cams

            if hf['env'].attrs['cam_encoding'] == 'mp4':
                meta_data_dict['frame_dim'] = hf['env']['cam0_video']['frames'].attrs['shape'][:2]
                meta_data_dict['img_T'] = hf['env']['cam0_video']['frames'].attrs['T']
                meta_data_dict['img_encoding'] = 'mp4'
                meta_data_dict['image_format'] = hf['env']['cam0_video']['frames'].attrs['image_format']
            else:
                meta_data_dict['frame_dim'] = hf['env']['cam0_video']['frame0'].attrs['shape'][:2]
                meta_data_dict['image_format'] = hf['env']['cam0_video']['frame0'].attrs['image_format']
                meta_data_dict['img_encoding'] = 'jpg'
                meta_data_dict['img_T'] = len(hf['env']['cam0_video'])

        # TODO: remove misc field and shift all to meta-data
        for k in hf['misc'].keys():
            assert k not in meta_data_dict, "key {} already present!".format(k)
            meta_data_dict[k] = hf['misc'][k][()]

        for k in hf['metadata'].attrs.keys():
            assert k not in meta_data_dict, "key {} already present!".format(k)
            meta_data_dict[k] = hf['metadata'].attrs[k]

        if 'low_bound' not in meta_data_dict and 'low_bound' in hf['env']:
            meta_data_dict['low_bound'] = hf['env']['low_bound'][0]

        if 'high_bound' not in meta_data_dict and 'high_bound' in hf['env']:
            meta_data_dict['high_bound'] = hf['env']['high_bound'][0]

        return meta_data_dict


def get_metadata_frame(files):
    if isinstance(files, str):
        base_path = files
        files = sorted(glob.glob('{}/*.hdf5'.format(files)))
        if not files:
            raise ValueError('no hdf5 files found!')

        if os.path.exists('{}/meta_data.pkl'.format(base_path)):
            meta_data = pd.read_pickle('{}/meta_data.pkl'.format(base_path), compression='gzip')

            registered_fnames = set([f for f in meta_data.index])
            loaded_fnames = set([f.split('/')[-1] for f in files])

            if loaded_fnames == registered_fnames:
                return meta_data
            os.remove('{}/meta_data.pkl'.format(base_path))
            print('regenerating meta_data file!')
    elif isinstance(files, (list, tuple)):
        base_path = None
        files = sorted(files)
    else:
        raise ValueError("Must be path to files or list/tuple of filenames")

    with Pool(cpu_count()) as p:
        meta_data = list(tqdm(p.imap(load_metadata_dict, files), total=len(files)))

    data_frame = pd.DataFrame(meta_data, index=[f.split('/')[-1] for f in files])
    if base_path:
        data_frame.to_pickle("{}/meta_data.pkl".format(base_path), compression='gzip')
    return data_frame


def load_metadata(files):
    base_path = files
    if isinstance(files, (tuple, list)):
        base_path = ''
    else:
        files = base_path = os.path.expanduser(base_path)

    return MetaDataContainer(base_path, get_metadata_frame(files))


def load_camera_imgs(cam_index, file_pointer, file_metadata, target_dims, start_time=0, n_load=None):
    cam_group = file_pointer['env']['cam{}_video'.format(cam_index)]
    old_dims = file_metadata['frame_dim']
    length = file_metadata['img_T']
    encoding = file_metadata['img_encoding']
    image_format = file_metadata['image_format']

    if n_load is None:
        n_load = length

    old_height, old_width = old_dims
    target_height, target_width = target_dims
    resize_method = cv2.INTER_CUBIC
    if target_height * target_width < old_height * old_width:
        resize_method = cv2.INTER_AREA

    images = np.zeros((n_load, target_height, target_width, 3), dtype=np.uint8)
    if encoding == 'mp4':
        buf = io.BytesIO(cam_group['frames'][:].tostring())
        img_buffer = [img for t, img in enumerate(imageio.get_reader(buf, format='mp4')) if
                      start_time <= t < n_load + start_time]
    elif encoding == 'jpg':
        img_buffer = [cv2.imdecode(cam_group['frame{}'.format(t)][:], cv2.IMREAD_COLOR)[:, :, ::-1]
                      for t in range(start_time, start_time + n_load)]
    else:
        raise ValueError("encoding not supported")

    for t, img in enumerate(img_buffer):
        if (old_height, old_width) == (target_height, target_width):
            images[t] = img
        else:
            images[t] = cv2.resize(img, (target_width, target_height), interpolation=resize_method)

    if image_format == 'RGB':
        return images
    elif image_format == 'BGR':
        return images[:, :, :, ::-1]
    raise NotImplementedError