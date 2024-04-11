import os
import torch
import argparse
import itertools
import numpy as np
import SimpleITK as sitk

from time import time
from torch import nn
from copy import deepcopy
from torch.cuda.amp import autocast
from skimage.transform import resize
from typing import OrderedDict, Tuple, List
from scipy.ndimage import gaussian_filter, map_coordinates

join = os.path.join

# 1. Preprocess
def create_nonzero_mask(data):
    from scipy.ndimage import binary_fill_holes
    assert len(data.shape) == 4 or len(data.shape) == 3, "data must have shape (C, X, Y, Z) or shape (C, X, Y)"
    nonzero_mask = np.zeros(data.shape[1:], dtype=bool)
    for c in range(data.shape[0]):
        this_mask = data[c] != 0
        nonzero_mask = nonzero_mask | this_mask
    nonzero_mask = binary_fill_holes(nonzero_mask)
    return nonzero_mask

def get_bbox_from_mask(mask, outside_value=0):
    mask_voxel_coords = np.where(mask != outside_value)
    minzidx = int(np.min(mask_voxel_coords[0]))
    maxzidx = int(np.max(mask_voxel_coords[0])) + 1
    minxidx = int(np.min(mask_voxel_coords[1]))
    maxxidx = int(np.max(mask_voxel_coords[1])) + 1
    minyidx = int(np.min(mask_voxel_coords[2]))
    maxyidx = int(np.max(mask_voxel_coords[2])) + 1
    return [[minzidx, maxzidx], [minxidx, maxxidx], [minyidx, maxyidx]]

def crop_to_bbox(image, bbox):
    assert len(image.shape) == 3, "only supports 3d images"
    resizer = (slice(bbox[0][0], bbox[0][1]), slice(bbox[1][0], bbox[1][1]), slice(bbox[2][0], bbox[2][1]))
    return image[resizer]

def crop_to_nonzero(data, seg=None, nonzero_label=-1):
    nonzero_mask = create_nonzero_mask(data)
    bbox = get_bbox_from_mask(nonzero_mask, 0)

    cropped_data = []
    for c in range(data.shape[0]):
        cropped = crop_to_bbox(data[c], bbox)
        cropped_data.append(cropped[None])
    data = np.vstack(cropped_data)

    if seg is not None:
        cropped_seg = []
        for c in range(seg.shape[0]):
            cropped = crop_to_bbox(seg[c], bbox)
            cropped_seg.append(cropped[None])
        seg = np.vstack(cropped_seg)

    nonzero_mask = crop_to_bbox(nonzero_mask, bbox)[None]
    if seg is not None:
        seg[(seg == 0) & (nonzero_mask == 0)] = nonzero_label
    else:
        nonzero_mask = nonzero_mask.astype(int)
        nonzero_mask[nonzero_mask == 0] = nonzero_label
        nonzero_mask[nonzero_mask > 0] = 0
        seg = nonzero_mask
    return data, seg, bbox

def crop(data, properties, seg=None):
    data, seg, bbox = crop_to_nonzero(data, seg, nonzero_label=-1)

    properties["crop_bbox"] = bbox
    properties['classes'] = np.unique(seg)
    seg[seg < -1] = 0
    properties["size_after_cropping"] = data[0].shape
    return data, seg, properties

def resize_segmentation(segmentation, new_shape, order=3):
    '''
    Resizes a segmentation map. Supports all orders (see skimage documentation). Will transform segmentation map to one
    hot encoding which is resized and transformed back to a segmentation map.
    This prevents interpolation artifacts ([0, 0, 2] -> [0, 1, 2])
    :param segmentation:
    :param new_shape:
    :param order:
    :return:
    '''
    tpe = segmentation.dtype
    unique_labels = np.unique(segmentation)
    assert len(segmentation.shape) == len(new_shape), "new shape must have same dimensionality as segmentation"
    if order == 0:
        return resize(segmentation.astype(float), new_shape, order, mode="edge", clip=True, anti_aliasing=False).astype(tpe)
    else:
        reshaped = np.zeros(new_shape, dtype=segmentation.dtype)

        for i, c in enumerate(unique_labels):
            mask = segmentation == c
            reshaped_multihot = resize(mask.astype(float), new_shape, order, mode="edge", clip=True, anti_aliasing=False)
            reshaped[reshaped_multihot >= 0.5] = c
        return reshaped
    
def resample_data_or_seg(data, new_shape, is_seg, axis=None, order=3, do_separate_z=False, order_z=0):
    """
    separate_z=True will resample with order 0 along z
    :param data:
    :param new_shape:
    :param is_seg:
    :param axis:
    :param order:
    :param do_separate_z:
    :param order_z: only applies if do_separate_z is True
    :return:
    """
    assert len(data.shape) == 4, "data must be (c, x, y, z)"
    assert len(new_shape) == len(data.shape) - 1
    if is_seg:
        resize_fn = resize_segmentation
        kwargs = OrderedDict()
    else:
        resize_fn = resize
        kwargs = {'mode': 'edge', 'anti_aliasing': False}
    dtype_data = data.dtype
    shape = np.array(data[0].shape)
    new_shape = np.array(new_shape)
    if np.any(shape != new_shape):
        data = data.astype(float)
        if do_separate_z:
            # print("separate z, order in z is", order_z, "order inplane is", order)
            assert len(axis) == 1, "only one anisotropic axis supported"
            axis = axis[0]
            if axis == 0:
                new_shape_2d = new_shape[1:]
            elif axis == 1:
                new_shape_2d = new_shape[[0, 2]]
            else:
                new_shape_2d = new_shape[:-1]

            reshaped_final_data = []
            for c in range(data.shape[0]):
                reshaped_data = []
                for slice_id in range(shape[axis]):
                    if axis == 0:
                        reshaped_data.append(resize_fn(data[c, slice_id], new_shape_2d, order, **kwargs).astype(dtype_data))
                    elif axis == 1:
                        reshaped_data.append(resize_fn(data[c, :, slice_id], new_shape_2d, order, **kwargs).astype(dtype_data))
                    else:
                        reshaped_data.append(resize_fn(data[c, :, :, slice_id], new_shape_2d, order, **kwargs).astype(dtype_data))
                reshaped_data = np.stack(reshaped_data, axis)
                if shape[axis] != new_shape[axis]:

                    # The following few lines are blatantly copied and modified from sklearn's resize()
                    rows, cols, dim = new_shape[0], new_shape[1], new_shape[2]
                    orig_rows, orig_cols, orig_dim = reshaped_data.shape

                    row_scale = float(orig_rows) / rows
                    col_scale = float(orig_cols) / cols
                    dim_scale = float(orig_dim) / dim

                    map_rows, map_cols, map_dims = np.mgrid[:rows, :cols, :dim]
                    map_rows = row_scale * (map_rows + 0.5) - 0.5
                    map_cols = col_scale * (map_cols + 0.5) - 0.5
                    map_dims = dim_scale * (map_dims + 0.5) - 0.5

                    coord_map = np.array([map_rows, map_cols, map_dims])
                    if not is_seg or order_z == 0:
                        reshaped_final_data.append(map_coordinates(reshaped_data, coord_map, order=order_z,
                                                                   mode='nearest')[None].astype(dtype_data))
                    else:
                        unique_labels = np.unique(reshaped_data)
                        reshaped = np.zeros(new_shape, dtype=dtype_data)

                        for i, cl in enumerate(unique_labels):
                            reshaped_multihot = np.round(
                                map_coordinates((reshaped_data == cl).astype(float), coord_map, order=order_z,
                                                mode='nearest'))
                            reshaped[reshaped_multihot > 0.5] = cl
                        reshaped_final_data.append(reshaped[None].astype(dtype_data))
                else:
                    reshaped_final_data.append(reshaped_data[None].astype(dtype_data))
            reshaped_final_data = np.vstack(reshaped_final_data)
        else:
            # print("no separate z, order", order)
            reshaped = []
            for c in range(data.shape[0]):
                reshaped.append(resize_fn(data[c], new_shape, order, **kwargs)[None].astype(dtype_data))
            reshaped_final_data = np.vstack(reshaped)
        return reshaped_final_data.astype(dtype_data)
    else:
        # print("no resampling necessary")
        return data

def get_lowres_axis(new_spacing):
    axis = np.where(max(new_spacing) / np.array(new_spacing) == 1)[0]  # find which axis is anisotropic
    return axis

def get_do_separate_z(spacing, anisotropy_threshold=3):
    do_separate_z = (np.max(spacing) / np.min(spacing)) > anisotropy_threshold
    return do_separate_z

def resample_patient(data, seg, original_spacing, target_spacing, order_data=3, order_seg=0, force_separate_z=False,
                     order_z_data=0, order_z_seg=0,
                     separate_z_anisotropy_threshold=3):
    """
    :param data:
    :param seg:
    :param original_spacing:
    :param target_spacing:
    :param order_data:
    :param order_seg:
    :param force_separate_z: if None then we dynamically decide how to resample along z, if True/False then always
    /never resample along z separately
    :param order_z_seg: only applies if do_separate_z is True
    :param order_z_data: only applies if do_separate_z is True
    :param separate_z_anisotropy_threshold: if max_spacing > separate_z_anisotropy_threshold * min_spacing (per axis)
    then resample along lowres axis with order_z_data/order_z_seg instead of order_data/order_seg

    :return:
    """
    assert not ((data is None) and (seg is None))
    if data is not None:
        assert len(data.shape) == 4, "data must be c x y z"
    if seg is not None:
        assert len(seg.shape) == 4, "seg must be c x y z"

    if data is not None:
        shape = np.array(data[0].shape)
    else:
        shape = np.array(seg[0].shape)
    new_shape = np.round(((np.array(original_spacing) / np.array(target_spacing)).astype(float) * shape)).astype(int)

    if force_separate_z is not None:
        do_separate_z = force_separate_z
        if force_separate_z:
            axis = get_lowres_axis(original_spacing)
        else:
            axis = None
    else:
        if get_do_separate_z(original_spacing, separate_z_anisotropy_threshold):
            do_separate_z = True
            axis = get_lowres_axis(original_spacing)
        elif get_do_separate_z(target_spacing, separate_z_anisotropy_threshold):
            do_separate_z = True
            axis = get_lowres_axis(target_spacing)
        else:
            do_separate_z = False
            axis = None

    if axis is not None:
        if len(axis) == 3:
            # every axis has the spacing, this should never happen, why is this code here?
            do_separate_z = False
        elif len(axis) == 2:
            # this happens for spacings like (0.24, 1.25, 1.25) for example. In that case we do not want to resample
            # separately in the out of plane axis
            do_separate_z = False
        else:
            pass

    if data is not None:
        data_reshaped = resample_data_or_seg(data, new_shape, False, axis, order_data, do_separate_z,
                                             order_z=order_z_data)
    else:
        data_reshaped = None
    if seg is not None:
        seg_reshaped = resample_data_or_seg(seg, new_shape, True, axis, order_seg, do_separate_z, order_z=order_z_seg)
    else:
        seg_reshaped = None
    return data_reshaped, seg_reshaped

def resample_and_normalize(data, target_spacing, properties, seg=None, force_separate_z=None):
    data[np.isnan(data)] = 0

    data, seg = resample_patient(data, seg, np.array(properties["original_spacing"]), target_spacing,
                                 3, 1, force_separate_z=force_separate_z, order_z_data=0, order_z_seg=0,
                                 separate_z_anisotropy_threshold=3)
    
    if seg is not None:
        seg[seg < -1] = 0

    properties["size_after_resampling"] = data[0].shape
    properties["spacing_after_resampling"] = target_spacing
    for c in range(len(data)):
        mask = seg[-1] >= 0
        data[c][mask] = (data[c][mask] - data[c][mask].mean()) / (data[c][mask].std() + 1e-8)
        data[c][mask == 0] = 0
            
    return data, seg, properties

def load_case_from_list_of_files(data_files, seg_file=None):
    assert isinstance(data_files, list) or isinstance(data_files, tuple), "case must be either a list or a tuple"
    properties = OrderedDict()
    data_itk = [sitk.ReadImage(f) for f in data_files]

    properties["original_size_of_raw_data"] = np.array(data_itk[0].GetSize())[[2, 1, 0]]
    properties["original_spacing"] = np.array(data_itk[0].GetSpacing())[[2, 1, 0]]
    properties["list_of_data_files"] = data_files
    properties["seg_file"] = seg_file

    properties["itk_origin"] = data_itk[0].GetOrigin()
    properties["itk_spacing"] = data_itk[0].GetSpacing()
    properties["itk_direction"] = data_itk[0].GetDirection()

    data_npy = np.vstack([sitk.GetArrayFromImage(d)[None] for d in data_itk])
    if seg_file is not None:
        seg_itk = sitk.ReadImage(seg_file)
        seg_npy = sitk.GetArrayFromImage(seg_itk)[None].astype(np.float32)
    else:
        seg_npy = None
    return data_npy.astype(np.float32), seg_npy, properties

def preprocess_test_case(data_files, target_spacing=[1,1,1], seg_file=None):
    data, seg, properties = load_case_from_list_of_files(data_files, seg_file)
    data, seg, properties = crop(data, properties, seg)
    data, seg, properties = resample_and_normalize(data, target_spacing, properties, seg,
                                                        force_separate_z=None)
    return data.astype(np.float32), seg, properties

#################################################################################
# 2. Network
class ConvDropoutNormNonlin(nn.Module):
    def __init__(self, input_channels, output_channels,
                 conv_op=nn.Conv3d, conv_kwargs=None,
                 norm_op=nn.InstanceNorm3d, norm_op_kwargs={'eps': 1e-5, 'affine': True},
                 nonlin=nn.LeakyReLU, nonlin_kwargs={'negative_slope': 1e-2, 'inplace': True}):
        super(ConvDropoutNormNonlin, self).__init__()
        if conv_kwargs is None:
            conv_kwargs = {'kernel_size': 3, 'stride': 1, 'padding': 1, 'dilation': 1, 'bias': True}

        self.nonlin_kwargs = nonlin_kwargs
        self.nonlin = nonlin
        self.norm_op_kwargs = norm_op_kwargs
        self.conv_kwargs = conv_kwargs
        self.conv_op = conv_op
        self.norm_op = norm_op

        self.conv = self.conv_op(input_channels, output_channels, **self.conv_kwargs)
        self.instnorm = self.norm_op(output_channels, **self.norm_op_kwargs)
        self.lrelu = self.nonlin(**self.nonlin_kwargs)

    def forward(self, x):
        x = self.conv(x)
        return self.lrelu(self.instnorm(x))
    
class Upsample(nn.Module):
    def __init__(self, size=None, scale_factor=None, mode='nearest', align_corners=False):
        super(Upsample, self).__init__()
        self.align_corners = align_corners
        self.mode = mode
        self.scale_factor = scale_factor
        self.size = size

    def forward(self, x):
        return nn.functional.interpolate(x, size=self.size, scale_factor=self.scale_factor, mode=self.mode,
                                         align_corners=self.align_corners)

class CSA(nn.Module):
    def __init__(self, inc=512, kernel_size=3, ratio=0.25, sort_small_first=False):
        super(CSA, self).__init__()
        self.inconv = nn.Conv3d(inc, inc, kernel_size, 1, 1)
        self.innorm = nn.InstanceNorm3d(inc)
        self.lrelu = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        self.avg = nn.AdaptiveAvgPool3d(1)
        self.ch_order = nn.Sequential(
            nn.Linear(inc, int(inc*ratio)),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Linear(int(inc*ratio), inc),
            nn.Sigmoid()
        )
        self.sort_small_first = sort_small_first

    def forward(self, x:torch.Tensor):
        b,c,d,h,w = x.size()

        x = self.inconv(x)
        x = self.lrelu(self.innorm(x))
        x_res = x
        ch_order = torch.argsort(self.ch_order(self.avg(x).view(b,c)), descending=not self.sort_small_first)
        
        return x_res + self.exchange(x, ch_order)

    @staticmethod
    def exchange(x: torch.Tensor, channel_order: torch.Tensor):
        b,c,d,h,w = x.size()
        new_x = []
        for batch in range(b):
            batch_order = channel_order[batch]
            new_x.append(x[batch][batch_order].unsqueeze(0))
        return torch.vstack(new_x)
    
class CSADropoutNormNonlin(nn.Module):
    def __init__(self, channels,
                 norm_op=nn.InstanceNorm3d, norm_op_kwargs=None,
                 nonlin=nn.LeakyReLU, nonlin_kwargs=None,
                 sort_small_first=False):
        super(CSADropoutNormNonlin, self).__init__()
        if nonlin_kwargs is None:
            nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        if norm_op_kwargs is None:
            norm_op_kwargs = {'eps': 1e-5, 'affine': True}

        self.nonlin_kwargs = nonlin_kwargs
        self.nonlin = nonlin
        self.norm_op_kwargs = norm_op_kwargs
        self.norm_op = norm_op

        self.conv = CSA(channels, kernel_size=3, sort_small_first=sort_small_first)
        self.instnorm = self.norm_op(channels, **self.norm_op_kwargs)
        self.lrelu = self.nonlin(**self.nonlin_kwargs)

    def forward(self, x):
        x = self.conv(x)
        return self.lrelu(self.instnorm(x))
    
class SplitConvCSA(nn.Module):
    def __init__(self, num_features) -> None:
        super().__init__()

        self.num_features = num_features
        self.t1_conv = ConvDropoutNormNonlin(1, num_features)
        self.t1_csa = CSADropoutNormNonlin(num_features, sort_small_first=True)
        self.t1ce_conv = ConvDropoutNormNonlin(1, num_features)
        self.t1ce_csa = CSADropoutNormNonlin(num_features)
        self.t2_conv = ConvDropoutNormNonlin(1, num_features)
        self.t2_csa = CSADropoutNormNonlin(num_features)
        self.t2flare_conv = ConvDropoutNormNonlin(1, num_features)
        self.t2flare_csa = CSADropoutNormNonlin(num_features)

    def feature_chosen(self, t1___, t1ce_, t2___, flair, missing_index):
        '''
        t1 <-> t2, t1ce <-> flair
        '''
        if missing_index == 0:   # t1
            batch1 = torch.cat([t1___[:,self.num_features*0//4:self.num_features*1//4,...],
                                t1___[:,self.num_features*1//4:self.num_features*2//4,...],
                                t1___[:,self.num_features*2//4:self.num_features*3//4,...],
                                t1___[:,self.num_features*3//4:self.num_features*4//4,...]], dim=1)
        elif missing_index == 1: # t1ce
            batch1 = torch.cat([t1ce_[:,self.num_features*1//4:self.num_features*2//4,...],
                                t1ce_[:,self.num_features*0//4:self.num_features*1//4,...],
                                t1ce_[:,self.num_features*3//4:self.num_features*4//4,...],
                                t1ce_[:,self.num_features*2//4:self.num_features*3//4,...]], dim=1)
        elif missing_index == 2: # t2
            batch1 = torch.cat([t2___[:,self.num_features*2//4:self.num_features*3//4,...],
                                t2___[:,self.num_features*3//4:self.num_features*4//4,...],
                                t2___[:,self.num_features*0//4:self.num_features*1//4,...],
                                t2___[:,self.num_features*1//4:self.num_features*2//4,...]], dim=1)
        elif missing_index == 3: # flair
            batch1 = torch.cat([flair[:,self.num_features*3//4:self.num_features*4//4,...],
                                flair[:,self.num_features*2//4:self.num_features*3//4,...],
                                flair[:,self.num_features*1//4:self.num_features*2//4,...],
                                flair[:,self.num_features*0//4:self.num_features*1//4,...]], dim=1)
        elif missing_index == 4: # t1 t1ce
            batch1 = torch.cat([t1___[:,self.num_features*0//4:self.num_features*1//4,...],
                                t1ce_[:,self.num_features*0//4:self.num_features*1//4,...],
                                t1___[:,self.num_features*2//4:self.num_features*3//4,...],
                                t1ce_[:,self.num_features*2//4:self.num_features*3//4,...]], dim=1)
        elif missing_index == 5: # t1 t2
            batch1 = torch.cat([t1___[:,self.num_features*0//4:self.num_features*1//4,...],
                                t1___[:,self.num_features*1//4:self.num_features*2//4,...],
                                t2___[:,self.num_features*0//4:self.num_features*1//4,...],
                                t2___[:,self.num_features*1//4:self.num_features*2//4,...]], dim=1)
        elif missing_index == 6: # t1 flair
            batch1 = torch.cat([t1___[:,self.num_features*0//4:self.num_features*1//4,...],
                                t1___[:,self.num_features*1//4:self.num_features*2//4,...],
                                flair[:,self.num_features*1//4:self.num_features*2//4,...],
                                flair[:,self.num_features*0//4:self.num_features*1//4,...]], dim=1)
        elif missing_index == 7: # t1ce t2
            batch1 = torch.cat([t1ce_[:,self.num_features*1//4:self.num_features*2//4,...],
                                t1ce_[:,self.num_features*0//4:self.num_features*1//4,...],
                                t2___[:,self.num_features*0//4:self.num_features*1//4,...],
                                t2___[:,self.num_features*1//4:self.num_features*2//4,...]], dim=1)
        elif missing_index == 8: # t1ce flair
            batch1 = torch.cat([t1ce_[:,self.num_features*1//4:self.num_features*2//4,...],
                                t1ce_[:,self.num_features*0//4:self.num_features*1//4,...],
                                flair[:,self.num_features*1//4:self.num_features*2//4,...],
                                flair[:,self.num_features*0//4:self.num_features*1//4,...]], dim=1)
        elif missing_index == 9: # t2 flair
            batch1 = torch.cat([t2___[:,self.num_features*2//4:self.num_features*3//4,...],
                                flair[:,self.num_features*2//4:self.num_features*3//4,...],
                                t2___[:,self.num_features*0//4:self.num_features*1//4,...],
                                flair[:,self.num_features*0//4:self.num_features*1//4,...]], dim=1)
        elif missing_index == 10: # t1 t1ce t2
            batch1 = torch.cat([t1___[:,self.num_features*0//4:self.num_features*1//4,...],
                                t1ce_[:,self.num_features*0//4:self.num_features*1//4,...],
                                t2___[:,self.num_features*0//4:self.num_features*1//4,...],
                                t2___[:,self.num_features*1//4:self.num_features*2//4,...]], dim=1)
        elif missing_index == 11: # t1 t1ce flair
            batch1 = torch.cat([t1___[:,self.num_features*0//4:self.num_features*1//4,...],
                                t1ce_[:,self.num_features*0//4:self.num_features*1//4,...],
                                flair[:,self.num_features*1//4:self.num_features*2//4,...],
                                flair[:,self.num_features*0//4:self.num_features*1//4,...]], dim=1)
        elif missing_index == 12: # t1 t2 flair
            batch1 = torch.cat([t1___[:,self.num_features*0//4:self.num_features*1//4,...],
                                t1___[:,self.num_features*1//4:self.num_features*2//4,...],
                                t2___[:,self.num_features*0//4:self.num_features*1//4,...],
                                flair[:,self.num_features*0//4:self.num_features*1//4,...]], dim=1)
        elif missing_index == 13: # t1ce t2 flair
            batch1 = torch.cat([t1ce_[:,self.num_features*1//4:self.num_features*2//4,...],
                                t1ce_[:,self.num_features*0//4:self.num_features*1//4,...],
                                t2___[:,self.num_features*0//4:self.num_features*1//4,...],
                                flair[:,self.num_features*0//4:self.num_features*1//4,...]], dim=1)
        elif missing_index == 14: # t1 t1ce t2 flair
            batch1 = torch.cat([t1___[:,self.num_features*0//4:self.num_features*1//4,...],
                                t1ce_[:,self.num_features*0//4:self.num_features*1//4,...],
                                t2___[:,self.num_features*0//4:self.num_features*1//4,...],
                                flair[:,self.num_features*0//4:self.num_features*1//4,...]], dim=1)
        else:
            raise IndexError
        
        return batch1
        
    def forward(self, x, missing_index):
        t1, t1ce, t2, t2flare = torch.split(x, 1, dim=1)
        t1: torch.Tensor = self.t1_csa(self.t1_conv(t1))
        t1ce: torch.Tensor = self.t1ce_csa(self.t1ce_conv(t1ce))
        t2: torch.Tensor = self.t2_csa(self.t2_conv(t2))
        t2flare: torch.Tensor = self.t2flare_csa(self.t2flare_conv(t2flare))

        newx = self.feature_chosen(
                t1[0,...].unsqueeze(0),
                t1ce[0,...].unsqueeze(0), 
                t2[0,...].unsqueeze(0), 
                t2flare[0,...].unsqueeze(0),
                missing_index[0]
            )
        for b, missnum in enumerate(missing_index[1:], start=1):
            newx = torch.cat([newx, 
                              self.feature_chosen(
                                t1[b,...].unsqueeze(0),
                                t1ce[b,...].unsqueeze(0), 
                                t2[b,...].unsqueeze(0), 
                                t2flare[b,...].unsqueeze(0),
                                missnum)
                              ], dim=0)
        return newx
    
class DeMoSeg(nn.Module):
    def __init__(self,
                 input_channels=4,
                 base_num_features=32,
                 num_classes=4,
                 num_pool=5,
                 do_ds=True,
                 upsample=False,
                 modality=14
                ):
        super().__init__()
        self.num_classes = num_classes
        self.input_channels = input_channels
        self.modality = modality
        self.num_pool = num_pool
        self.do_ds = do_ds # Deep Supervision
        self.upsample = upsample # Upsample small scale output logits

        self.conv_blocks_localization = [] # Up
        self.conv_blocks_context = [] # Down
        self.tu = [] # ConvTranspose
        self.seg_outputs = [] # Segment Head
        self.upsample_ops = [] # Upsample small scale output logits

        self.inference_apply_nonlin = nn.Softmax(dim=1)
        self.training_apply_nonlin = nn.Identity()

        self.features_per_stage=[min(base_num_features * 2 ** i, 320) for i in range(num_pool+1)]
        self.conv_kwargs = {'kernel_size': 3, 'stride': 1, 'padding': 1, 'dilation': 1, 'bias': True}
        self.conv_kwargs_down = {'kernel_size': 3, 'stride': 2, 'padding': 1, 'dilation': 1, 'bias': True}
        self.norm_op_kwargs = {'eps': 1e-5, 'affine': True}
        self.nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        self.weightInitializer = InitWeights_He(1e-2)

        # 1. building Encoder
        self.conv_blocks_context.append(SplitConvCSA(self.features_per_stage[0])) # FD + CSSA + RCR
        for in_fea, out_fea in zip(self.features_per_stage[:-1], self.features_per_stage[1:]):
            self.conv_blocks_context.append(nn.Sequential(
                ConvDropoutNormNonlin(in_fea, out_fea, conv_kwargs=self.conv_kwargs_down),
                CSADropoutNormNonlin(out_fea, sort_small_first=True)
            ))

        # 2. building TransposeConv and Decoder
        for in_fea, up_out_fea in zip(self.features_per_stage[::-1][:-1], self.features_per_stage[::-1][1:]):
            self.tu.append(
                nn.ConvTranspose3d(in_fea, up_out_fea, kernel_size=2, stride=2, bias=False)
            )
            self.conv_blocks_localization.append(nn.Sequential(
                ConvDropoutNormNonlin(up_out_fea+up_out_fea, up_out_fea, conv_kwargs=self.conv_kwargs),
                CSADropoutNormNonlin(up_out_fea, sort_small_first=True)
            ))

        # 3. building Segment Head
        for fea in self.features_per_stage[:-1][::-1]:
            self.seg_outputs.append(nn.Conv3d(fea, self.num_classes, 1, 1, bias=False))

        # 4. building Upsample ops
        pool_op_kernel_sizes = [[2,2,2] for _ in range(num_pool)]
        cum_upsample = np.cumprod(np.vstack(pool_op_kernel_sizes), axis=0)[:-1][::-1]
        for layer in range(len(cum_upsample)):
            if self.upsample:
                self.upsample_ops.append(Upsample(scale_factor=tuple([int(i) for i in cum_upsample[layer]]),
                                                  mode='trilinear'))
            else:
                self.upsample_ops.append(nn.Identity())
        self.upsample_ops.append(nn.Identity()) # for largest scale

        self.conv_blocks_localization = nn.ModuleList(self.conv_blocks_localization)
        self.conv_blocks_context = nn.ModuleList(self.conv_blocks_context)
        self.tu = nn.ModuleList(self.tu)
        self.seg_outputs = nn.ModuleList(self.seg_outputs)
        self.upsample_ops = nn.ModuleList(self.upsample_ops)

        self.apply(self.weightInitializer)

    def forward(self, x):
        if self.training:
            missing_case = torch.randint(0,15,(x.shape[0],)).cpu().numpy()
        else:
            missing_case = torch.zeros(x.shape[0], dtype=torch.int64).fill_(self.modality).cpu().numpy()
        x = self.MissingSituationGeneratemorebatch(x, missing_case)
        skips = []
        seg_outputs = []

        # 1. Encoder
        for i, layer_enc in enumerate(self.conv_blocks_context):
            x = layer_enc(x, missing_case) if i == 0 else layer_enc(x)
            skips.append(x)

        skips.pop()
        # 2. Decoder
        for transpose, layer_dec, seg_head, up_op in zip(self.tu, self.conv_blocks_localization, self.seg_outputs, self.upsample_ops):
            x = layer_dec(torch.cat([transpose(x), skips.pop()], dim=1))
            seg_outputs.append(up_op(self.training_apply_nonlin(seg_head(x))))

        if self.do_ds:
            return seg_outputs[::-1]
        else:
            return seg_outputs[-1]

    @staticmethod
    def MissingSituationGeneratemorebatch(x: torch.Tensor, missing_cases_index: np.ndarray):
        """
        Our modality order is t1, t1ce, t2, flare
        0: t1
        1: t1ce
        2: t2
        3: flare
        4: t1, t1ce
        5: t1, t2
        6: t1, flare
        7: t1ce, t2
        8: t1ce, flare
        9: t2, flare
        10: t1, t1ce, t2
        11: t1, t1ce, flare
        12: t1, t2, flare
        13: t1ce, t2, flare
        14: t1, t1ce, t2, flare
        """
        missing_situation_dict = {
            0: [1,0,0,0],
            1: [0,1,0,0],
            2: [0,0,1,0],
            3: [0,0,0,1],
            4: [1,1,0,0],
            5: [1,0,1,0],
            6: [1,0,0,1],
            7: [0,1,1,0],
            8: [0,1,0,1],
            9: [0,0,1,1],
            10: [1,1,1,0],
            11: [1,1,0,1],
            12: [1,0,1,1],
            13: [0,1,1,1],
            14: [1,1,1,1]
        }
        random_miss = [missing_situation_dict[i] for i in missing_cases_index]
        random_miss = torch.from_numpy(np.array(random_miss)).to(x.device).view(x.shape[0], 4, 1, 1, 1)
        x = x * random_miss
        return x
    
class InitWeights_He(object):
    def __init__(self, neg_slope=1e-2):
        self.neg_slope = neg_slope

    def __call__(self, module):
        if isinstance(module, nn.Conv3d) or isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d) or isinstance(module, nn.ConvTranspose3d):
            module.weight = nn.init.kaiming_normal_(module.weight, a=self.neg_slope)
            if module.bias is not None:
                module.bias = nn.init.constant_(module.bias, 0)

########################################
# 2.5 TTA
def predict_3D(network, x: np.ndarray, do_mirroring: bool = True, 
                step_size: float = 0.5, patch_size: Tuple[int, ...] = None, 
                pad_kwargs: dict = None, device='cuda') -> Tuple[np.ndarray, np.ndarray]:
    
    if device == 'cuda':
        torch.cuda.empty_cache()

    if pad_kwargs is None:
        pad_kwargs = {'constant_values': 0}

    with autocast(enabled=device == 'cuda'):
        with torch.no_grad():
            res = predict_3D_tiled(network, x, step_size, do_mirroring, 
                                   patch_size, pad_kwargs=pad_kwargs,
                                   device=device)
            
    return res

def compute_steps_for_sliding_window(patch_size: Tuple[int, ...], image_size: Tuple[int, ...], step_size: float) -> List[List[int]]:
        target_step_sizes_in_voxels = [i * step_size for i in patch_size]

        num_steps = [int(np.ceil((i - k) / j)) + 1 for i, j, k in zip(image_size, target_step_sizes_in_voxels, patch_size)]

        steps = []
        for dim in range(len(patch_size)):
            max_step_value = image_size[dim] - patch_size[dim]
            if num_steps[dim] > 1:
                actual_step_size = max_step_value / (num_steps[dim] - 1)
            else:
                actual_step_size = 99999999999  # does not matter because there is only one step at 0

            steps_here = [int(np.round(actual_step_size * i)) for i in range(num_steps[dim])]

            steps.append(steps_here)

        return steps

def predict_3D_tiled(network: torch.nn.Module, x: np.ndarray, 
                     step_size: float, do_mirroring: bool, 
                     patch_size: tuple, pad_kwargs: dict, device='cuda') -> Tuple[np.ndarray, np.ndarray]:
    if device == 'cuda':
        device = next(network.parameters()).device.index
        all_in_gpu = True
    else:
        all_in_gpu = False

    data, slicer = pad_nd_image(x, patch_size, "constant", pad_kwargs, True, None)
    data_shape = data.shape
    num_classes = network.num_classes

    steps = compute_steps_for_sliding_window(patch_size, data_shape[1:], step_size)
    num_tiles = len(steps[0]) * len(steps[1]) * len(steps[2])

    if num_tiles > 1:
        gaussian_importance_map = get_gaussian(patch_size, sigma_scale=1. / 8)
        gaussian_importance_map = torch.from_numpy(gaussian_importance_map)

        if all_in_gpu and torch.cuda.is_available():
            gaussian_importance_map = gaussian_importance_map.cuda(device, non_blocking=True)

    else:
        gaussian_importance_map = None

    if all_in_gpu:
        if num_tiles > 1:
            gaussian_importance_map = gaussian_importance_map.half()
            gaussian_importance_map[gaussian_importance_map == 0] = gaussian_importance_map[
                gaussian_importance_map != 0].min()
            add_for_nb_of_preds = gaussian_importance_map
        else:
            add_for_nb_of_preds = torch.ones(patch_size, device=device)

        aggregated_results = torch.zeros([num_classes] + list(data.shape[1:]), dtype=torch.half, device=device)
        data = torch.from_numpy(data).cuda(non_blocking=True)
        aggregated_nb_of_predictions = torch.zeros([num_classes] + list(data.shape[1:]), dtype=torch.half, device=device)

    else:
        if num_tiles > 1:
            gaussian_importance_map[gaussian_importance_map == 0] = gaussian_importance_map[
                gaussian_importance_map != 0].min()
            add_for_nb_of_preds = gaussian_importance_map.cpu().numpy()
        else:
            add_for_nb_of_preds = np.ones(patch_size, dtype=np.float32)
        data = torch.from_numpy(data)
        aggregated_results = np.zeros([num_classes] + list(data.shape[1:]), dtype=np.float32)
        aggregated_nb_of_predictions = np.zeros([num_classes] + list(data.shape[1:]), dtype=np.float32)

    for x in steps[0]:
        lb_x = x
        ub_x = x + patch_size[0]
        for y in steps[1]:
            lb_y = y
            ub_y = y + patch_size[1]
            for z in steps[2]:
                lb_z = z
                ub_z = z + patch_size[2]

                predicted_patch = mirror_and_pred_3D(network,
                    data[None, :, lb_x:ub_x, lb_y:ub_y, lb_z:ub_z], do_mirroring,
                    gaussian_importance_map, device=device)[0]
                
                if all_in_gpu:
                    predicted_patch = predicted_patch.half()
                else:
                    predicted_patch = predicted_patch.cpu().numpy()
                
                aggregated_results[:, lb_x:ub_x, lb_y:ub_y, lb_z:ub_z] += predicted_patch
                aggregated_nb_of_predictions[:, lb_x:ub_x, lb_y:ub_y, lb_z:ub_z] += add_for_nb_of_preds
    

    slicer = tuple([slice(0, aggregated_results.shape[i]) for i in range(len(aggregated_results.shape) - (len(slicer) - 1))] + slicer[1:])
    aggregated_results = aggregated_results[slicer]
    aggregated_nb_of_predictions = aggregated_nb_of_predictions[slicer]

    aggregated_results /= aggregated_nb_of_predictions
    del aggregated_nb_of_predictions

    predicted_segmentation = aggregated_results.argmax(0)

    if all_in_gpu:
        aggregated_results = aggregated_results.detach().cpu().numpy()
    return predicted_segmentation, aggregated_results

def mirror_and_pred_3D(network, x, 
                       do_mirroring: bool = True,
                       mult = None, device=None) -> torch.tensor:
    pred = network.inference_apply_nonlin(network(x))

    if do_mirroring:
        mirror_axes = (0, 1, 2)
        mirror_axes_iter = [c for i in range(len(mirror_axes)) for c in itertools.combinations([m + 2 for m in mirror_axes], i + 1)]
        for axes in mirror_axes_iter:
            pred += torch.flip(network.inference_apply_nonlin(network(torch.flip(x, (*axes,)))), (*axes,))
        pred /= (len(mirror_axes_iter) + 1)

    if mult is not None:
        pred[:, :] *= mult

    return pred

def pad_nd_image(image, new_shape=None, mode="constant", kwargs=None, return_slicer=False, shape_must_be_divisible_by=None):
    
    if kwargs is None:
        kwargs = {'constant_values': 0}

    if new_shape is not None:
        old_shape = np.array(image.shape[-len(new_shape):])
    else:
        assert shape_must_be_divisible_by is not None
        assert isinstance(shape_must_be_divisible_by, (list, tuple, np.ndarray))
        new_shape = image.shape[-len(shape_must_be_divisible_by):]
        old_shape = new_shape

    num_axes_nopad = len(image.shape) - len(new_shape)

    new_shape = [max(new_shape[i], old_shape[i]) for i in range(len(new_shape))]

    if not isinstance(new_shape, np.ndarray):
        new_shape = np.array(new_shape)

    if shape_must_be_divisible_by is not None:
        if not isinstance(shape_must_be_divisible_by, (list, tuple, np.ndarray)):
            shape_must_be_divisible_by = [shape_must_be_divisible_by] * len(new_shape)
        else:
            assert len(shape_must_be_divisible_by) == len(new_shape)

        for i in range(len(new_shape)):
            if new_shape[i] % shape_must_be_divisible_by[i] == 0:
                new_shape[i] -= shape_must_be_divisible_by[i]

        new_shape = np.array([new_shape[i] + shape_must_be_divisible_by[i] - new_shape[i] % shape_must_be_divisible_by[i] for i in range(len(new_shape))])

    difference = new_shape - old_shape
    pad_below = difference // 2
    pad_above = difference // 2 + difference % 2
    pad_list = [[0, 0]]*num_axes_nopad + list([list(i) for i in zip(pad_below, pad_above)])

    if not ((all([i == 0 for i in pad_below])) and (all([i == 0 for i in pad_above]))):
        res = np.pad(image, pad_list, mode, **kwargs)
    else:
        res = image

    if not return_slicer:
        return res
    else:
        pad_list = np.array(pad_list)
        pad_list[:, 1] = np.array(res.shape) - pad_list[:, 1]
        slicer = list(slice(*i) for i in pad_list)
        return res, slicer
    
def get_gaussian(patch_size, sigma_scale=1. / 8) -> np.ndarray:
    tmp = np.zeros(patch_size)
    center_coords = [i // 2 for i in patch_size]
    sigmas = [i * sigma_scale for i in patch_size]
    tmp[tuple(center_coords)] = 1
    gaussian_importance_map = gaussian_filter(tmp, sigmas, 0, mode='constant', cval=0)
    gaussian_importance_map = gaussian_importance_map / np.max(gaussian_importance_map) * 1
    gaussian_importance_map = gaussian_importance_map.astype(np.float32)

    # gaussian_importance_map cannot be 0, otherwise we may end up with nans!
    gaussian_importance_map[gaussian_importance_map == 0] = np.min(
        gaussian_importance_map[gaussian_importance_map != 0])

    return gaussian_importance_map

def predict_preprocessed_data(network, data, do_mirroring = True, step_size = 0.5, device='cuda'):
        
    network.do_ds = False
    current_mode = network.training
    network.eval()
    ret = predict_3D(network, data, do_mirroring=do_mirroring, 
                     step_size=step_size,
                     patch_size=[128, 128, 128], 
                     pad_kwargs={'constant_values': 0},
                     device=device)
    network.train(current_mode)
    
    return ret

#################################################################################
# 3. Output
def save_segmentation_nifti_from_softmax(segmentation_softmax, out_fname: str,
                                         properties_dict: dict, order: int = 1,
                                         region_class_order: Tuple[Tuple[int]] = None,
                                         seg_postprogess_fn: callable = None, seg_postprocess_args: tuple = None,
                                         resampled_npz_fname: str = None,
                                         non_postprocessed_fname: str = None, force_separate_z: bool = None,
                                         interpolation_order_z: int = 0, verbose: bool = False):
    

    if isinstance(segmentation_softmax, str):
        del_file = deepcopy(segmentation_softmax)
        if segmentation_softmax.endswith('.npy'):
            segmentation_softmax = np.load(segmentation_softmax)
        elif segmentation_softmax.endswith('.npz'):
            segmentation_softmax = np.load(segmentation_softmax)['softmax']
        os.remove(del_file)

    # first resample, then put result into bbox of cropping, then save
    current_shape = segmentation_softmax.shape
    shape_original_after_cropping = properties_dict.get('size_after_cropping')
    shape_original_before_cropping = properties_dict.get('original_size_of_raw_data')

    if np.any([i != j for i, j in zip(np.array(current_shape[1:]), np.array(shape_original_after_cropping))]):
        if force_separate_z is None:
            if get_do_separate_z(properties_dict.get('original_spacing')):
                do_separate_z = True
                lowres_axis = get_lowres_axis(properties_dict.get('original_spacing'))
            elif get_do_separate_z(properties_dict.get('spacing_after_resampling')):
                do_separate_z = True
                lowres_axis = get_lowres_axis(properties_dict.get('spacing_after_resampling'))
            else:
                do_separate_z = False
                lowres_axis = None
        else:
            do_separate_z = force_separate_z
            if do_separate_z:
                lowres_axis = get_lowres_axis(properties_dict.get('original_spacing'))
            else:
                lowres_axis = None

        if lowres_axis is not None and len(lowres_axis) != 1:
            # this happens for spacings like (0.24, 1.25, 1.25) for example. In that case we do not want to resample
            # separately in the out of plane axis
            do_separate_z = False

        if verbose: print("separate z:", do_separate_z, "lowres axis", lowres_axis)
        seg_old_spacing = resample_data_or_seg(segmentation_softmax, shape_original_after_cropping, is_seg=False,
                                               axis=lowres_axis, order=order, do_separate_z=do_separate_z,
                                               order_z=interpolation_order_z)
        # seg_old_spacing = resize_softmax_output(segmentation_softmax, shape_original_after_cropping, order=order)
    else:
        if verbose: print("no resampling necessary")
        seg_old_spacing = segmentation_softmax

    if resampled_npz_fname is not None:
        np.savez_compressed(resampled_npz_fname, softmax=seg_old_spacing.astype(np.float16))
        # this is needed for ensembling if the nonlinearity is sigmoid
        if region_class_order is not None:
            properties_dict['regions_class_order'] = region_class_order

    if region_class_order is None:
        seg_old_spacing = seg_old_spacing.argmax(0)
    else:
        seg_old_spacing_final = np.zeros(seg_old_spacing.shape[1:])
        for i, c in enumerate(region_class_order):
            seg_old_spacing_final[seg_old_spacing[i] > 0.5] = c
        seg_old_spacing = seg_old_spacing_final

    bbox = properties_dict.get('crop_bbox')

    if bbox is not None:
        seg_old_size = np.zeros(shape_original_before_cropping, dtype=np.uint8)
        for c in range(3):
            bbox[c][1] = np.min((bbox[c][0] + seg_old_spacing.shape[c], shape_original_before_cropping[c]))
        seg_old_size[bbox[0][0]:bbox[0][1],
        bbox[1][0]:bbox[1][1],
        bbox[2][0]:bbox[2][1]] = seg_old_spacing
    else:
        seg_old_size = seg_old_spacing

    if seg_postprogess_fn is not None:
        seg_old_size_postprocessed = seg_postprogess_fn(np.copy(seg_old_size), *seg_postprocess_args)
    else:
        seg_old_size_postprocessed = seg_old_size

    seg_resized_itk = sitk.GetImageFromArray(seg_old_size_postprocessed.astype(np.uint8))
    seg_resized_itk.SetSpacing(properties_dict['itk_spacing'])
    seg_resized_itk.SetOrigin(properties_dict['itk_origin'])
    seg_resized_itk.SetDirection(properties_dict['itk_direction'])
    sitk.WriteImage(seg_resized_itk, out_fname)

    if (non_postprocessed_fname is not None) and (seg_postprogess_fn is not None):
        seg_resized_itk = sitk.GetImageFromArray(seg_old_size.astype(np.uint8))
        seg_resized_itk.SetSpacing(properties_dict['itk_spacing'])
        seg_resized_itk.SetOrigin(properties_dict['itk_origin'])
        seg_resized_itk.SetDirection(properties_dict['itk_direction'])
        sitk.WriteImage(seg_resized_itk, non_postprocessed_fname)

#################################################################################
def DeMoSeg_Infer(input_file_path, output_folder, model_path, tta, modality, device, print):
    output_filename = output_folder

    # 0. make something usable
    if device == "gpu": 
        device = "cuda"
    if device == "cuda" and not torch.cuda.is_available():
        print("No GPU detected. Running on CPU.")
        device = "cpu"

    # 1. preprocess
    print("Start preprocessing")
    preprocess_start_time = time()
    Four_Modality_file_path = [input_file_path.split('_0000.nii.gz')[0] + f"_000{i}.nii.gz" for i in range(4)]
    d, _, properties = preprocess_test_case(Four_Modality_file_path, target_spacing=[1,1,1])
    print(f"    preprocessing time: {time()-preprocess_start_time: .2f}s")

    # 2. infer
    print("Start inference")
    infer_start_time = time()
    network = DeMoSeg(
        input_channels=4,
        base_num_features=32,
        num_classes=4,
        num_pool=5,
        modality=modality
    )
    if device=='cuda':
        network.cuda()
    network.inference_apply_nonlin = torch.nn.Softmax(dim=1)
    params = [torch.load(model_path, map_location=torch.device('cpu'))]
    network.load_state_dict(params[0]['state_dict'])
    softmax = predict_preprocessed_data(network, d, do_mirroring=tta, step_size=0.5, device=device)[1]
    if device=='cuda':
        torch.cuda.empty_cache()
    print(f"    infer time: {time()-infer_start_time: .2f}s")

    # 3. output
    print("Start saving")
    save_start_time = time()
    bytes_per_voxel = 4
    if device == 'cuda':
        bytes_per_voxel = 2  # if all_in_gpu then the return value is half (float16)
    if np.prod(softmax.shape) > (2e9 / bytes_per_voxel * 0.85):  # * 0.85 just to be save
        np.save(output_filename[:-4] + ".npy", softmax)
        softmax = output_filename[:-4] + ".npy"

    save_segmentation_nifti_from_softmax(
        softmax, output_filename, properties, 1, None,
        None, None,
        None, None, None, 0
    )
    print(f"    saving time: {time()-save_start_time: .2f}s")
    print(f"\nTotal DeMoSeg time: {time()-preprocess_start_time: .2f}s")
    


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Segment 104 anatomical structures in CT images.",
                                     epilog="Written by Jakob Wasserthal. If you use this tool please cite https://pubs.rsna.org/doi/10.1148/ryai.230024")

    parser.add_argument("-i", metavar="filepath", dest="input",
                        help="MRI nifti image or folder of dicom slices",
                        type=str, required=True)

    parser.add_argument("-o", metavar="directory", dest="output",
                        help="Output directory for segmentation masks",
                        type=str, required=True)
    
    parser.add_argument("-p", "--pth",
                        help="model pth path",
                        type=str, required=True)
    
    parser.add_argument("-tta", "--TTA", action="store_true",
                        help="Testing time augmentation",
                        default=False)
    
    parser.add_argument("-m", "--modality", type=int,
                        help="Missing situation index, must within [0,14]",
                        default=14)
    
    parser.add_argument("-d", "--device", choices=["gpu", "cpu", "mps"],
                        help="Device to run on (default: gpu).",
                        default="gpu")

    args = parser.parse_args()

    DeMoSeg_Infer(
        input_file_path=args.input,
        output_folder=args.output,
        model_path=args.pth,
        tta=args.TTA,
        modality=args.modality,
        device=args.device
    )