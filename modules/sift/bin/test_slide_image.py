reload(slide_image)
import slide_image

# Test 1
path_to_tif = '../../../resources/morphology.ome.tif'
tif_img = slide_image.get_slide_image(path_to_tif)
level = tif_img.levels[0]
level.dtype.itemsize
by_dims = ('X', 'Y', 'Z')
[d in level.axes for d in list(by_dims)]


tif_img.select_level('10MB', by_dims='XY')
level.shape
tif_img.levels

# Test 2
path_to_tif = '../../../resources/HE.tif'
tif_img = slide_image.get_slide_image(path_to_tif)
tif_img._get_slice(
    X=slice(None, 10),
    Y=slice(None, 10),
    S=0
)

tif_img.





path_to_nd2 = '../../../resources/Region_1.nd2'
path_to_tif = '../../../resources/CytAssist.tif'
path_to_tif = '../../../resources/HE.tif'


# nd2_img = slide_image.get_slide_image(path_to_nd2)
# nd2_img.size
# nd2_img.shape
# nd2_img.channels
# nd2_img[1:10, 1:10]

import nd2
f = nd2.ND2File(path_to_nd2)
f.metadata

{'k': 'v'}.pop


def get_args(fn):
    print(fn.__code__.co_varnames)
    def wrapper(*args, **kwargs):
        kwargs.update({'world': 'moon'})
        fn(*args, **kwargs)
    return wrapper

def hello_world(
    hello,
    world
):
    print(hello, world)

get_args(hello_world)(hello='hello', world='world')

from functools import partial
partial(hello_world, sun='sun')(world='world')

def test(hello:int):
    pass

test('hi')

tif_img = slide_image.get_slide_image(path_to_tif)
tif_img[0]

import tifffile
import zarr

tifffile.TiffFile(path_to_tif).pages[0].tags['ImageWidth'].value
tifffile.TiffFile(path_to_tif).pages[0].tags['ImageLength'].value
tifffile.TiffFile(path_to_tif).series[1]
tifffile.TiffFile(path_to_tif).series[0].levels[0].nbytes

tifffile.TiffFile()

store = tifffile.imread(path_to_tif, aszarr=True, level=6)
tiff_zarr = zarr.open(store)
tiff_zarr[(0, slice(None, 100), slice(None,100))]

store = tifffile.imread(
    path_to_tif,
    aszarr=True,
    level=1,

z = zarr.open(store)

import numpy as np
np.dtype('uint16').itemsize