from abc import ABC, abstractmethod
import typing
import os
import nd2
import tifffile
import zarr
import numpy as np
from collections import defaultdict, OrderedDict
import re


_valid_image_formats = dict()


# Decorator to register new image formats below
def _register_image_format(file_type):
    def decorator(fn):
        # Multiple synonymous file extensions, e.g., .tif/.tiff
        if isinstance(file_type, list):
            for ft in file_type:
                _valid_image_formats[ft] = fn
        # One file extension
        else:
            _valid_image_formats[file_type] = fn
        return fn
    return decorator


# Abstract creator class
class SlideImage(ABC):
    """
    An abstract class defining the key behaviors for reading and interacting
    with a slide image, e.g., retrieving channels or image slices. Its purpose
    is to enable factory method-style file loading, so all core utilities
    should make use of other file format-specific libraries.

    The dimensional ordering for slide images is (X, Y, Channel, Z).
    When slicing, the level argument defaults to 0, the bottom level. X/Y/C/Z
    arguments default to all available image data.
    Ex.
    # 0th level, 0th channel, all Z planes, 100 pixels from X and Y dimensions
    my_img_slice = my_img[:100, :100, 0, :]
    Developer notes: Use lazy-loading when possible, and try to maintain the
    object slicing pattern (e.g., my_slide_image[i, j, k]) for accessing
    regions of an image.

    Attributes
    ----------
    file_path : typing.Union[str, os.PathLike]
        Path of the file to load
    """

    def __init__(
        self,
        file_path: typing.Union[str, os.PathLike],
    ):
        # Set file path for lazy reading
        self._file_path = file_path
        
        # '_levels' is a helper param to enable more convenient indexing of 
        # pyramidal files through a 'img.levels[level]' pattern
        self._set_levels()

    @abstractmethod
    def _set_levels(self) -> list:
        """
        Creates a list of 'levels' whose indices are the order of pyramidal
        levels, where 0 is the largest or bottom level.
        
        Returns
        -------
        list
            A list of Level objects
        """
        pass

    @abstractmethod
    def _get_slice(self, level : int, **kwargs) -> np.ndarray:
        """
        Helper function to enable array slicing/indexing aritrary dimension
        orderings.

        Parameters
        ----------
        level : int
            Pyramidal level to access
        kwargs
            Should contain axis names (keys) and slice objects (values)
        Returns
        -------
        np.ndarray
            sliced Numpy array described by kwargs
        """
        return self._get_slice_child(level, **kwargs)

    @abstractmethod
    def _get_level_dtype(self, level: int) -> np.dtype:
        """
        Helper function for data type by pyramidal level.
        Returns e.g., dtype('uint16') for a 16-bit integer array
        """
        pass

    @abstractmethod
    def _get_level_axes(self, level: int) -> str:
        """
        Helper function for getting the ordering of axes by pyramidal level.
        Returns e.g., 'XYZC' for 4d array with X, Y, Z, C axes.
        """
        pass

    @abstractmethod
    def _get_level_shape(self, level: int) -> tuple[int]:
        """
        Helper function for getting dimensional sizes by pyramidal level.
        Returns e.g., (1000, 1000, 10, 3) for 4d array with shape 
        1000 x 1000 x 10 x 3.
        """
        pass

    def __getitem__(self, key):
        """
        Method for slicing images.
        Implement this method in concrete classes based on specifics.
        By default, slices the 0th pyramidal level.
        """
        return self._levels[0][key]

    def select_level(
        self,
        max_mem: typing.Union[int, str],
        by_dims: tuple[str] = None,
    ):
        """
        Return the largest pyramidal level with a size in memory less than or
        equal to 'max_mem'

        The 'max_mem' argument can be an integer (in bytes) or any numerical
        value and one of B, KB, MB, GB, TB units. Optionally, limit request to
        a subset of dimensions.

        Parameters
        ----------
        max_mem : typing.Union[int, str]
            Maximum size in memory of the level to return. Either an integer 
            representing the number of bytes or numerical value and unit (str)
            are accepted
        by_dims : tuple[str], optional
            Subset memory calculation to only these dimensions, by default None

        Returns
        -------
        Level
            The largest pyramidal level with a size in memory meeting the
            supplied arguments

        Raises
        ------
        ValueError
            If no level meets the requirements
        """
        # Internal helper
        def parse_size(size : str):
            units = {"B": 1, "KB": 1e3, "MB": 1e6, "GB": 1e9, "TB": 1e12}
            size = size.upper()
            if not re.match(r' ', size):
                size = re.sub(r'([KMGT]?B)', r' \1', size)
            val, unit = map(str.strip, size.split())
            return int(float(val) * units[unit])

        if isinstance(max_mem, str):
            max_mem = parse_size(max_mem)
        for level in self.levels:
            if by_dims is None:
                by_dims = level.axes
            mask = [d in list(by_dims) for d in level.axes]
            shape = np.array(level.shape)[mask]
            n_bytes = np.prod(shape) * level.dtype.itemsize
            if n_bytes <= max_mem:
                return level
        msg = (
            f"No level found matching memory request of {max_mem} for "
            f"dimensions {''.join(list(by_dims))}"
        )
        raise ValueError(msg)

    @property
    def levels(self):
        """
        List indexing into the pyramidal levels of the file, returning 'Level'
        convenience objects.
        """
        return self._levels

    @property
    def axes(self) -> str:
        """
        Ordering of axes, e.g., 'XYZC' for 4d array with X, Y, Z, C axes.
        By default, references the 0th pyramidal level.
        """
        return self._levels[0].axes

    @property
    def shape(self) -> tuple[int]:
        """
        Dimensional sizes, e.g., (1000, 1000, 10, 3) for 4d array with shape 
        1000 x 1000 x 10 x 3.
        By default, references the 0th pyramidal level.
        """
        return self._levels[0].shape
    
    # Decorator to check level values across all
    @staticmethod
    def _check_level(fn):
        def wrapper(self, *args, **kwargs):
            named_args = dict(zip(fn.__code__.co_varnames[1:], args))
            named_args.update(kwargs)
            level = named_args['level']
            if len(self._levels) <= level:
                msg = f"User-provided level '{level}' does not exist"
                raise ValueError(msg)
            return fn(self, *args, **kwargs)
        return wrapper


@_register_image_format(['.tif', '.tiff'])
class TIFFSlideImage(SlideImage):
    """
    Default creator for Tiff file formats (e.g., .tif or .tiff)

    Parameters
    ----------
    file_path : str
        Path to a valid Tiff file
    """
    _check_level = SlideImage._check_level

    def _set_levels(self):
        """
        Creates a list of 'levels' objects corresponding to pyramidal levels in
        Tiff

        Raises
        ------
        NotImplementedError
            If more than one series is found in the Tiff file (one series is 
            assumed to be pyramidal levels).
        """
        with tifffile.TiffFile(self._file_path, mode='r') as f:
            if len(f.series) != 1:
                msg = (
                    "SlideImage can't yet handle parsing Tiff files with more "
                    "than one series or no series"
                )
                raise NotImplementedError(msg)
            else:
                self._levels = [
                    Level(slide_image=self, level=i)
                    for i in range(len(f.series[0].levels))
                ]

    @_check_level
    def _get_slice(
        self,
        level: int = 0,
        **kwargs,
    ) -> np.ndarray:
        """
        Helper function to enable array slicing/indexing in Tiff files with
        arbitrary dimension orderings.

        Parameters
        ----------
        level : int, optional
            Pyramidal level to access, by default 0
        kwargs : dict
            A dictionary of axis names (keys) and slice objects (values)
        Returns
        -------
        np.ndarray
            sliced Numpy array described by index
        """
        index = defaultdict(lambda: slice(None))
        index.update(kwargs)
        #assert self._validate_slice(index)
        store = tifffile.imread(
            self._file_path,
            aszarr=True,
            level=level,
        )
        z = zarr.open(store)
        slices = tuple(index[ax] for ax in self._get_level_axes(level))
        return z[slices]

    @_check_level
    def _get_level_dtype(self, level: int) -> np.dtype:
        """
        Helper function for getting the ordering of axes by pyramidal level.
        Returns e.g., 'XYZC' for 4d array with X, Y, Z, C axes.
        """
        with tifffile.TiffFile(self._file_path, mode='r') as f:
            return f.series[0].levels[level].dtype

    @_check_level
    def _get_level_axes(self, level: int) -> str:
        """
        Helper function for getting the ordering of axes by pyramidal level.
        Returns e.g., 'XYZC' for 4d array with X, Y, Z, C axes.
        """
        with tifffile.TiffFile(self._file_path, mode='r') as f:
            return f.series[0].levels[level].axes

    @_check_level
    def _get_level_shape(self, level: int) -> str:
        """
        Helper function for getting the ordering of axes by pyramidal level.
        Returns e.g., 'XYZC' for 4d array with X, Y, Z, C axes.
        """
        print(level)
        with tifffile.TiffFile(self._file_path, mode='r') as f:
            return f.series[0].levels[level].shape

'''
# Concrete creator for .nd2 files
# TODO: No good solution for working with .nd2 files which is memory-efficient
# and fast, so loading full image to memory for now
@_register_image_format('.nd2')
class ND2SlideImage(SlideImage):
    """
    Default creator for nd2 (Nikon NIS Elements) file format

    Parameters
    ----------
    file_path : str
        Path to a valid nd2 file
    """

    _check_level = SlideImage._check_level

    def __init__(self, file_path):
        super().__init__(file_path)
        self._img = nd2.imread(self._file_path)

    def _set_levels(self):
        """
        ND2 files can only contain one level
        """
        self._levels = [Level(slide_image=self, level=0)]

    @_check_level
    def _get_slice(
        self,
        level: int = 0,
        **kwargs,
    ) -> np.ndarray:
        """
        Helper function to enable array slicing/indexing in Tiff files with
        arbitrary dimension orderings.

        Parameters
        ----------
        level : int, optional
            Pyramidal level to access, by default 0
        kwargs : dict
            A dictionary of axis names (keys) and slice objects (values)
        Returns
        -------
        np.ndarray
            sliced Numpy array described by index
        """
        index = defaultdict(lambda: slice(None))
        index.update(kwargs)
        #assert self._validate_slice(index)
        store = tifffile.imread(
            self._file_path,
            aszarr=True,
            level=level,
        )
        z = zarr.open(store)
        slices = tuple(index[ax] for ax in self._get_level_order(level))
        return z[slices]

    @_check_level
    def _get_level_dtype(self, level: int) -> np.dtype:
        """
        Helper function for getting the ordering of axes by pyramidal level.
        Returns e.g., 'XYZC' for 4d array with X, Y, Z, C axes.
        """
        with tifffile.TiffFile(self._file_path, mode='r') as f:
            return f.series[0].levels[level].dtype

    @_check_level
    def _get_level_axes(self, level: int) -> str:
        """
        Helper function for getting the ordering of axes by pyramidal level.
        Returns e.g., 'XYZC' for 4d array with X, Y, Z, C axes.
        """
        with tifffile.TiffFile(self._file_path, mode='r') as f:
            return f.series[0].levels[level].dims

    @_check_level
    def _get_level_shape(self, level: int) -> str:
        """
        Helper function for getting the ordering of axes by pyramidal level.
        Returns e.g., 'XYZC' for 4d array with X, Y, Z, C axes.
        """
        with tifffile.TiffFile(self._file_path, mode='r') as f:
            return f.series[0].levels[level].shape

    def __init__(self, file_path):
        super().__init__(file_path)
        # Initialize nd2 object as ndarray
        
        # Set length, layer size(s), etc.
        self._set_metadata()

    def _set_metadata(self):
        # Implementation for extracting metadata from .nd2 files
        
        with nd2.ND2File(self._file_path) as f:
            self.size = f.size
            self.channels = f._channel_names
            self.shape = f.shape
            self.mpp = f.voxel_size()

    def __getitem__(self, key):
        # Implementation for slicing .nd2 images
        return self._img[key]
'''
'''
# TODO: Concrete creator for .mrxs files
class MRXSSlideImage(SlideImage):
    def __init__(self, file_path):
        super().__init__(file_path)
        # Initialize mirax file object

    def get_metadata(self):
        # Implementation for extracting metadata from .mrxs files
        pass

    def get_slice(self):
        # Implementation for slicing .mrxs images
        pass
'''


class Level:
    """
    Lightweight class to make accessing pyramidal levels of an image more
    convenient through the 'levels' attribute of SlideImage classes.

    Parameters
    ----------
    slide_image: SlideImage
        The associated SlideImage object which will be doing all the real work
    level: int
        The level to access
    """
    def __init__(self, slide_image : SlideImage, level : int) -> None:
        self.slide_image = slide_image
        self.level = level

    @property
    def dtype(self) -> np.dtype:
        """
        Getter for parent property
        """
        return self.slide_image._get_level_dtype(self.level)
    
    @property
    def axes(self) -> str:
        """
        Getter for parent property
        """
        return self.slide_image._get_level_axes(self.level)
    
    @property
    def shape(self) -> tuple[int]:
        """
        Getter for parent property
        """
        return self.slide_image._get_level_shape(self.level)
    
    def get_slice(self, **kwargs) -> np.array:
        """
        Wrapper for parent method
        """
        return self.slide_image._get_slice(self.level, **kwargs)

    def __getitem__(self, item) -> np.ndarray:
        """
        Custom indexing for arbitrary numbers of pyramidal levels and arbitrary
        orderings of dimensions in file formats.
        """
        try:
            iter(item)
        except TypeError:
            item = [item]
        index = dict(zip(list(self.axes), item))
        return self.slide_image._get_slice(self.level, **index)


# Factory method
def get_slide_image(
    file_path: typing.Union[str, os.PathLike],
) -> typing.Type[SlideImage]:
    """
    Factory pattern implementation of image file loading for various supported
    slide formats. Its return type, SlideImage, supports key behaviors for
    reading and interacting with a slide image, e.g., retrieving channels or
    image slices.

    Currently supported formats are .nd2.

    Parameters
    ----------
    file_path : typing.Union[str, os.PathLike]
        The file path of the image to load

    Returns
    -------
    typing.Type[SlideImage]
        A subclass of SlideImage according to the extension of the file path

    Raises
    ------
    ValueError
        If the file type isn't supported or the path provided is not a file
        (e.g., a directory)
    """
    _, extension = os.path.splitext(file_path)
    if extension not in _valid_image_formats:
        raise ValueError("Unsupported file format")
    return _valid_image_formats[extension](file_path)
