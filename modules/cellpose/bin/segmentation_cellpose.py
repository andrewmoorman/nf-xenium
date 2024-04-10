import numpy as np
from cellpose import models
from slide_image import get_slide_image
import os
import typing


def segment_cellpose(
    path_in: typing.Union[str, os.PathLike],
    path_out: typing.Union[str, os.PathLike],
    **kwargs,
) -> None:
    """
    Wrapper to create Cellpose model and get segmentation masks. 

    Keyword arguments are first passed through to cellpose.models.Cellpose, if
    applicable, and then to cellpose.models.Cellpose.eval

    Parameters
    ----------
    path_in : typing.Union[str, os.PathLike]
        Path to image to segment. Images can be of types supported by the
        SlideImage module
    path_out : typing.Union[str, os.PathLike]
        Path to save the image segmentation to. Only .npy files are currently
        supported
    kwargs : key, value mappings
        Other parameters are passed to Cellpose model construction and eval
    """
    img = get_slide_image(path_in)
    model = filter_kwargs(models.CellPose, kwargs)()
    masks, _, _, _ = filter_kwargs(model.eval, kwargs)(img)
    np.save(path_out, masks)


def filter_kwargs(func : function, kwargs : typing.Dict) -> function:
    """
    Helper function to extract relevant arguments for some function 'func' and 
    supply them to the function

    Developer notes: For convenience, I override any kwargs supplied 

    Parameters
    ----------
    func : function
        Function to get arguments for
    kwargs : typing.Dict
        Full collection of keyword arguments. This should include some args
        not used by the function (otherwise just call it directly)

    Returns
    -------
    function
        Original function with filtered kwargs supplied
    """
    filtered_kwargs = dict()
    for arg in func.__code__.co_varnames:
        if arg in kwargs:
            filtered_kwargs[arg] = kwargs[arg]
    def wrapper(*args, **kwargs):
        kwargs.update(filtered_kwargs)
        return func(*args, **kwargs)
    return wrapper
