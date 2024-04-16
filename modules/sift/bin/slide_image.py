from abc import ABC, abstractmethod
import nd2
import tifffile
import zarr
import cv2
import numpy as np
from collections import defaultdict, OrderedDict
import re
from xml.etree import ElementTree
import typing
import os


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
        # Levels wraps list class with a custom index error
        self._levels = Levels(self._get_levels())
        assert len(self._levels) > 0

    @abstractmethod
    def _get_levels(self) -> list:
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
            f"No level found matching memory request of {max_mem}B for "
            f"dimensions {''.join(list(by_dims))}. Smallest level size is "
            f"{n_bytes}B"
        )
        raise ValueError(msg)

    # TODO: Implement more comprehensive metadata collection before saving
    def write_ome_tiff(
        self,
        file_path: typing.Union[str, os.PathLike] = None,
        subresolutions: int = 0,
    ) -> None:
        """
        Write a multi-level, pyramidal OME-TIFF file with minimal metadata to 
        the provided file path

        Each pyramidal level is a downsampling of the full-resolution plane in
        the X and Y dimensions and the resolution is unchanged in the other
        dimensions. Levels are created with a fixed downsampling factor of 2
        for each level in 'subresolutions'

        More info on OME-TIFF file format can be found here:
        https://docs.openmicroscopy.org/ome-model/6.3.1/ome-tiff/specification.html

        Parameters
        ----------
        file_path : typing.Union[str, os.PathLike], optional
            Path to write new OME-TIFF. If None, writes a .ome.tif file to the
            same directory as 'self._file_path', by default None
        subresolutions : int, optional
            Number of downsampled levels to write, by default 0

        Raises
        ------
        ValueError
            If provided the wrong file extension in 'file_path'
        ValueError
            If axes cannot be coerced into a pyramidal format
        """
        # Destination filepath handling
        if file_path is None:
            if self._file_path.endswith('.ome.tif'):
                file_path = self._file_path
            else:
                file_path, extension = os.path.splitext(self._file_path)
                file_path += '.ome.tif'
        elif not file_path.endswith('.ome.tif'):
            msg = (
                f"File path '{file_path}' does end with the extension "
                "'.ome.tif'"
            )
            raise ValueError(msg)

        # Rearrange array; generally, YX should be last
        default_order = 'SCZYX'
        order = list(filter(lambda c: c in self.axes, default_order))
        if len(self.axes) != len(order):
            msg = (
                "Can't save .ome.tif due to unrecognized labeled axes: "
                f"{', '.join(set(self.axes).difference(order))}"
            )
            raise ValueError(msg)
        if not order[-2:] != ['Y','X']:
            msg = f"Last two axes of image are {''.join(order[-2:])}, not YX"
            raise ValueError(msg)
        img = np.moveaxis(
            self[:],
            [self.axes.index(i) for i in order],
            range(len(order)),
        )

        # Write multi-level Tiff
        # TODO: function for getting mpp
        metadata = {
            'Axes': order,
            'PhysicalSizeX': self.mpp['X'],
            'PhysicalSizeXUnit': "µm",
            'PhysicalSizeY': self.mpp['Y'],
            'PhysicalSizeYUnit': "µm",
            'Channel': {'Name': self.channels},
        }
        with tifffile.TiffWriter(file_path,  bigtiff=True) as writer:
            # Pyramid level 0
            resolution = np.array([1e4 / self.mpp['X']] * 2)
            kwargs = dict(
                photometric='rgb' if len(self.channels)==3 else 'minisblack',
                tile=(1024, 1024),
                dtype=self.dtype,
                compression='jpeg2000',
                resolutionunit='CENTIMETER'
            )
            writer.write(
                img,
                subifds=subresolutions,
                resolution=resolution,
                metadata=metadata,
                **kwargs,
            )
            # All other pyramidal sub-resolutions
            for i in range(1, subresolutions+1):
                sub_img = cv2.resize(
                    img,
                    fx=2**-i,
                    fy=2**-i,
                    interpolation=cv2.INTER_AREA,
                )
                writer.write(
                    sub_img,
                    subfiletype=1,
                    resolution=resolution * 2**-i,
                    **kwargs,
                )

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

    @property
    def dtype(self) -> np.dtype:
        """
        Helper function for getting the numpy data type of the pyramidal level.
        Returns e.g., np.dtype('uint16') for 16-bit int array
        By default, references the 0th pyramidal level.
        """
        return self._levels[0].dtype
    
    @property
    def channels(self) -> list[str]:
        """
        Helper function for getting the numpy data type of the pyramidal level.
        Returns e.g., np.dtype('uint16') for 16-bit int array
        By default, references the 0th pyramidal level.
        """
        return self._levels[0].channels

    @property
    def mpp(self) -> dict[str, float]:
        """
        Helper function for getting the microns per pixel for X/Y image axes.
        Returns e.g., {'X': 0.2137, 'Y': 0.2137} for an square aspect image
        with 0.2317 microns per pixel.
        """
        return self._levels[0].mpp


@_register_image_format(['.tif', '.tiff'])
class TIFFSlideImage(SlideImage):
    """
    Default creator for Tiff file formats (e.g., .tif or .tiff)

    Parameters
    ----------
    file_path : str
        Path to a valid Tiff file
    """

    def _get_levels(self):
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
                return [
                    Level(slide_image=self, level=i)
                    for i in range(len(f.series[0].levels))
                ]


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

    def _get_level_dtype(self, level: int) -> np.dtype:
        """
        Helper function for getting the numpy data type of the pyramidal level.
        Returns e.g., np.dtype('uint16') for 16-bit int array
        """
        with tifffile.TiffFile(self._file_path, mode='r') as f:
            return f.series[0].levels[level].dtype

    def _get_level_axes(self, level: int) -> str:
        """
        Helper function for getting the ordering of axes by pyramidal level.
        Returns e.g., 'XYZC' for 4d array with X, Y, Z, C axes
        """
        with tifffile.TiffFile(self._file_path, mode='r') as f:
            return f.series[0].levels[level].axes

    def _get_level_shape(self, level: int) -> str:
        """
        Helper function for getting the ordering of axes by pyramidal level.
        Returns e.g., 'XYZC' for 4d array with X, Y, Z, C axes
        """
        with tifffile.TiffFile(self._file_path, mode='r') as f:
            return f.series[0].levels[level].shape

    def _get_level_channels(self, level: int) -> list[str]:
        """
        Helper function for getting the channel names by pyramidal level.
        Where no channel names exist, returns an empty list.
        """
        with tifffile.TiffFile(self._file_path) as f:
            if f.is_ome:
                tree = ElementTree.fromstring(f.ome_metadata)
                prefix = tree.tag.rstrip('OME')
                channels = tree.findall(f".//{prefix}Channel")
                return [c.attrib['Name'] for c in channels]
            else:
                return []


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

    def __init__(self, file_path):
        super().__init__(file_path)
        self._img = nd2.imread(self._file_path)

    def _get_levels(self):
        """
        ND2 files can only contain one level
        """
        return [Level(slide_image=self, level=0)]

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
        slices = tuple(index[ax] for ax in self._get_level_axes(level))
        return self._img[slices]

    def _get_level_dtype(self, level: int) -> np.dtype:
        """
        Helper function for getting the numpy data type of the pyramidal level.
        Returns e.g., np.dtype('uint16') for 16-bit int array
        """
        with nd2.ND2File(self._file_path) as f:
            return f.dtype

    def _get_level_axes(self, level: int) -> str:
        """
        Helper function for getting the ordering of axes by pyramidal level.
        Returns e.g., 'XYZC' for 4d array with X, Y, Z, C axes.
        """
        with nd2.ND2File(self._file_path) as f:
            return ''.join(f.sizes.keys())

    def _get_level_shape(self, level: int) -> str:
        """
        Dimensional sizes, e.g., (1000, 1000, 10, 3) for 4d array with shape 
        1000 x 1000 x 10 x 3.
        """
        with nd2.ND2File(self._file_path) as f:
            return f.shape

    def _get_level_channels(self, level: int) -> list[str]:
        """
        Helper function for getting the channel names by pyramidal level.
        Where no channel names exist, returns an empty list.
        """
        with nd2.ND2File(self._file_path) as f:
            try:
                channels = f.metadata.channels
            except AttributeError:
                return []
            channels.sort(key = lambda c: c.channel.index)
            return [c.channel.name for c in channels]


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

    def __getattr__(self, name):
        """
        Getter for parent property
        """
        fn = f'_get_level_{name}'
        if hasattr(self.slide_image, fn):
                self.slide_image.__getattribute__(fn)(self.level)
        else:
            raise AttributeError

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


class Levels(list):
    """
    Extends list class with custom IndexError
    """
    def __getitem__(self, index):
        try:
            return super.__getitem__(index)
        except IndexError:
            msg = f"Level {index} does not exist in SlideImage"
            raise IndexError(msg)


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
