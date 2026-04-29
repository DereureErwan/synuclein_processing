import numpy as np
import torch
import cv2

from skimage import measure
from skimage.color import rgb2hed
from skimage.measure import label, regionprops
from scipy.spatial import ConvexHull, distance_matrix
from scipy import ndimage
from shapely.geometry import MultiPoint

from torchvision.models.feature_extraction import get_graph_node_names, create_feature_extractor

from torchvision.transforms import Normalize
from copy import deepcopy

import denseCRF

import skimage.morphology as sk_morphology


class DinoVisionTransformerClassifier(torch.nn.Module):
    def __init__(self, dino):
        super(DinoVisionTransformerClassifier, self).__init__()
        self.transformer = deepcopy(dino)
        self.classifier = torch.nn.Linear(768, 1)

    def forward(self, x):
        x = self.transformer(x)
        x = self.classifier(x)
        return x


def get_attn_extractor(model):

    feature_model = model.transformer
    return_nodes = {
        f"blocks.{i}.attn.softmax": f"layer_{i}_attn_matrix"
        for i in range(12)
        }

    attn_extractor = create_feature_extractor(feature_model, return_nodes=return_nodes)

    return attn_extractor
    
def crop_around_cell(img,x,y,crop_larger):
    """
    Crop a cell around its center of mass 
    """
    if len(img.shape) == 2 : #mask
        pad = int(crop_larger/2)
        img_padded = np.pad(img, ((pad,pad),(pad,pad)),"constant" ) #0 padding
        x+=pad
        y+=pad
        crop_cell = img_padded[x-int(crop_larger/2):x+int(crop_larger/2),y-int(crop_larger/2):y+int(crop_larger/2)]
    elif len(img.shape) == 3 : #rgb
        pad = int(crop_larger/2)
        img_padded = np.pad(img, ((pad,pad),(pad,pad),(0,0)),"constant" ) #0 padding
        x+=pad
        y+=pad
        crop_cell = img_padded[x-int(crop_larger/2):x+int(crop_larger/2),y-int(crop_larger/2):y+int(crop_larger/2),:]
    return crop_cell



def densecrf(I, P, param):
    return denseCRF.densecrf(I, P, param)


def normalize_to_unit_range(image: np.ndarray) -> np.ndarray:
    min_val = image.min()
    max_val = image.max()
    if max_val == min_val:
        return np.zeros_like(image)
    return (image - min_val) / (max_val - min_val)


def crf_refine(image, prob):
    prob = prob[..., None]
    prob_background = 1 - prob
    prob = np.concatenate((prob, prob_background), axis=-1)

    I = np.uint8(image * 255)

    param = (10.0, 80, 13, 3.0, 3, 5.0)
    lab = densecrf(I, prob, param)
    return 1 - lab


def segment_attn(image, attn_extractor):
    img = image.unsqueeze(0)

    with torch.no_grad():
        attn_matrices = attn_extractor(img)

    attn_matrix = attn_matrices['layer_11_attn_matrix'][0]

    cls_attn = attn_matrix[:, 0, 5:]
    cls_attn = cls_attn / cls_attn.sum(dim=-1, keepdim=True)
    cls_attn = cls_attn.mean(dim=0)

    attn_seg = cls_attn.detach().cpu().numpy().reshape(37, 37)
    attn_seg = attn_seg / attn_seg.max()
    attn_seg = (attn_seg * 255).astype("uint8")
    attn_seg = cv2.resize(attn_seg, (592, 592), interpolation=cv2.INTER_AREA)

    prob = np.asarray(attn_seg, np.float32) / 255
    prob[prob > 0.01] = 1

    im_to_segment = img[0].permute(1, 2, 0).cpu().numpy() * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))

    return crf_refine(im_to_segment, prob)


def max_feret_diameter(mask):
    contours = measure.find_contours(mask, 0.5)
    if not contours:
        return 0

    contour = max(contours, key=len)
    hull = ConvexHull(contour)
    hull_points = contour[hull.vertices]

    dist_matrix = distance_matrix(hull_points, hull_points)
    return np.max(dist_matrix)


def segment_red_stain(image):
    ihc_hed = rgb2hed(image)

    red_stain = normalize_to_unit_range(ihc_hed[:, :, 1])
    red_stain[red_stain > 0.5] = 1
    red_stain[red_stain < 0.5] = 0

    return red_stain


def combine_masks(A, B):
    intersection = np.logical_and(A, B)

    labeled_A = label(A)
    regions_A = regionprops(labeled_A)

    result = np.zeros_like(A)
    for region in regions_A:
        component_mask = (labeled_A == region.label)
        if np.any(np.logical_and(component_mask, intersection)):
            result = np.logical_or(result, component_mask)

    return result.astype(int)


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px != py:
            self.parent[py] = px


def merge_labels_based_on_distance(segmentation_mask, distance_threshold=0.2):
    props = regionprops(segmentation_mask)

    polygons = []
    label_to_idx = {}

    for i, prop in enumerate(props):
        coords = prop.coords
        polygon = MultiPoint(coords).convex_hull
        polygons.append(polygon)
        label_to_idx[prop.label] = i

    n = len(polygons)
    uf = UnionFind(n)

    for i in range(n):
        for j in range(i + 1, n):
            if polygons[i].distance(polygons[j]) < distance_threshold:
                uf.union(i, j)

    groups = {}
    new_label_id = 1
    old_label_to_new_label = {}

    for old_label, idx in label_to_idx.items():
        root = uf.find(idx)
        if root not in groups:
            groups[root] = new_label_id
            new_label_id += 1
        old_label_to_new_label[old_label] = groups[root]

    merged_mask = np.zeros_like(segmentation_mask)
    for old_label, new_label in old_label_to_new_label.items():
        merged_mask[segmentation_mask == old_label] = new_label

    return merged_mask


def is_degenerate_labeled_mask(mask, threshold=0.95):
    return np.mean(mask != 0) > threshold

def mask_percent(np_img):
  """
  Determine the percentage of a NumPy array that is masked (how many of the values are 0 values).

  Args:
    np_img: Image as a NumPy array.

  Returns:
    The percentage of the NumPy array that is masked.
  """
  if (len(np_img.shape) == 3) and (np_img.shape[2] == 3):
    np_sum = np_img[:, :, 0] + np_img[:, :, 1] + np_img[:, :, 2]
    mask_percentage = 100 - np.count_nonzero(np_sum) / np_sum.size * 100
  else:
    mask_percentage = 100 - np.count_nonzero(np_img) / np_img.size * 100
  return mask_percentage


def filter_remove_small_objects(np_img, min_size=3000, avoid_overmask=True, overmask_thresh=95, output_type="uint8"):
  """
  Filter image to remove small objects (connected components) less than a particular minimum size. If avoid_overmask
  is True, this function can recursively call itself with progressively smaller minimum size objects to remove to
  reduce the amount of masking that this filter performs.

  Args:
    np_img: Image as a NumPy array of type bool.
    min_size: Minimum size of small object to remove.
    avoid_overmask: If True, avoid masking above the overmask_thresh percentage.
    overmask_thresh: If avoid_overmask is True, avoid masking above this threshold percentage value.
    output_type: Type of array to return (bool, float, or uint8).

  Returns:
    NumPy array (bool, float, or uint8).
  """

  rem_sm = np_img/255

  rem_sm = np_img.astype(bool)  # make sure mask is boolean

  
  rem_sm = sk_morphology.remove_small_objects(rem_sm, min_size=min_size)
  mask_percentage = mask_percent(rem_sm)
  if (mask_percentage >= overmask_thresh) and (min_size >= 1) and (avoid_overmask is True):
    new_min_size = min_size / 2
    print(new_min_size)
    #print("Mask percentage %3.2f%% >= overmask threshold %3.2f%% for Remove Small Objs size %d, so try %d" % (mask_percentage, overmask_thresh, min_size, new_min_size))
    rem_sm = filter_remove_small_objects(np_img, new_min_size, avoid_overmask, overmask_thresh, output_type)
  np_img = rem_sm

  if output_type == "bool":
    pass
  elif output_type == "float":
    np_img = np_img.astype(float)
  else:
    np_img = np_img.astype("uint8") * 255

  #util.np_info(np_img, "Remove Small Objs", t.elapsed())
  return np_img

def segment_protein_aggregates_tile(
    tile,
    model,
    attn_extractor,
    device,
    threshold_max_feret=33,
    threshold_distance=20
):

    normalize = Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

    height = 592
    image = cv2.resize(tile, (height, height), interpolation=cv2.INTER_AREA)

    image_model = np.transpose(image, (-1, 0, 1))
    image_model = torch.from_numpy(image_model)[None, ...]
    image_model = normalize(image_model.float().to(device))

    with torch.no_grad():
        pred_ys = model(image_model)

    pred_ys = torch.flatten(pred_ys)
    pred_ys = torch.sigmoid(pred_ys)
    pred_ys = (pred_ys > 0.5).float()

    image = (image * 255).astype("uint8")

    n_aggregates = 0
    crops = {"rgb": [], "mask": []}
    dict_aggregates = []

    height_original = tile.shape[0]

    seg_feret = np.zeros((height_original, height_original))
    seg_all = np.zeros((height_original, height_original))

    if pred_ys[0] == 0:
        attn_seg = segment_attn(image_model[0], attn_extractor)
        red_stain = segment_red_stain(image)

        combined_segmentation = combine_masks(attn_seg, red_stain)

        combined_segmentation = cv2.resize(
            (combined_segmentation.astype("uint8") * 255),
            (height_original, height_original),
            interpolation=cv2.INTER_AREA
        )

        combined_segmentation = combined_segmentation / 255
        combined_segmentation[combined_segmentation > 0] = 1

        combined_segmentation = filter_remove_small_objects(
            (combined_segmentation.astype("uint8") * 255), 150, avoid_overmask=False
        )

        combined_segmentation = combined_segmentation / 255

        labeled, _ = ndimage.label(combined_segmentation)

        instance_segmentation = merge_labels_based_on_distance(
            labeled, distance_threshold=threshold_distance
        )

        if not is_degenerate_labeled_mask(instance_segmentation):
            seg_all = instance_segmentation

            for aggregate in np.unique(instance_segmentation)[1:]:
                tile_mask_aggregate = (instance_segmentation == aggregate)

                max_diameter = max_feret_diameter(tile_mask_aggregate)

                if max_diameter > threshold_max_feret:

                    xx,yy = np.where(tile_mask_aggregate)
                    
                    x_tile_border = int(np.mean(xx))
                    y_tile_border = int(np.mean(yy))
                    
                    crop_aggregate_mask = crop_around_cell(
                        tile_mask_aggregate,
                        x_tile_border,
                        y_tile_border,
                        crop_larger=256
                    ).astype(bool)

                    crop_aggregate_rgb = crop_around_cell(
                        tile,
                        x_tile_border,
                        y_tile_border,
                        crop_larger=256
                    )
                    dict_aggregate = {
                        "x": int(x_tile_border),
                        "y": int(y_tile_border),
                    }

                    seg_feret[tile_mask_aggregate] = aggregate
                    n_aggregates += 1

                    crops["rgb"].append(crop_aggregate_rgb)
                    crops["mask"].append(crop_aggregate_mask)
                    dict_aggregates.append(dict_aggregate)

    return seg_feret, seg_all, crops, dict_aggregates
