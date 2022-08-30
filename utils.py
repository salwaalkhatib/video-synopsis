from xml.dom.expatbuilder import FILTER_REJECT
from keras.preprocessing import image
from keras.applications import mobilenet
import matplotlib.pyplot as plt
import itertools
import numpy as np
import cv2
import os
import json
import shutil

COLOR_CLASS_INDEX_PATH = "color_model_classes.json"
TYPE_CLASS_INDEX_PATH = "type_model_classes.json"

def mrcnn_inference(model, image):
    """Run inference using the Mask RCNN model and return result

    Args:
        model (class 'mrcnn.model.MaskRCNN'): the mrcnn to use for inference
        image (numpy.ndarray): image to use for inference

    Returns:
        list: list of results
    """
    return model.detect([image], verbose=0)

def return_frame(cap, frameID):
    """returns a frame from a video using frame number

    Args:
        cap (cv2.VideoCapture): object of video capturing from video file
        frameID (int): frame number to retrieve

    Returns:
        numpy array: frame as numpy array
    """    
    cap.set(cv2.CAP_PROP_POS_FRAMES, frameID)
    ret, frame = cap.read()
    return frame

def generate_background(videoPath, frameNbrs):
    
    """
    Generating background image(s)
    
    Keyword arguments:
    videoPath -- string of the name of the video file to process including the path
    frameNbrs -- a list of frame numbers to calculate the median over
    """
    assert isinstance(videoPath, str), "[ERROR4dx] Filename should be a string"
    cap = cv2.VideoCapture(videoPath)

    # Check if camera opened successfully
    if (cap.isOpened() == False):
        print("Error opening video file")

    # get duration of video to check if it is longer than the timestep
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    
    #Store selected frames in an array
    frames = []

    for fid in frameNbrs:
        frame = return_frame(cap, fid)
        frames.append(frame)

    # Calculate the median along the time axis
    medianFrame = np.median(frames, axis=0).astype(dtype=np.uint8)

    cap.release()

    return medianFrame

def filter_results(results, indices):
    """filters results dictionary from certain predictions

    Args:
        results (dict): returned by the mask rcnn prediction
        indices (list): indices of predictions to remove from results

    Returns:
        dict: filtered results including only desirable predictions
    """    
    results['class_ids'] = np.delete(results['class_ids'], indices)
    results['masks'] = np.delete(results['masks'], indices, axis = 2)
    results['rois'] = np.delete(results['rois'], indices, axis = 0)
    results['scores'] = np.delete(results['scores'], indices)
    return results

def filter_classes(results, class_ids):
    """filter the results of Mask R-CNN that is pretrained on COCO dataset

    Args:
        results (list of dicts): returned by the MaskRCNN.model.detect()
        class_ids (list of ints): a list of desired class_ids to be kept
    """
    ids_to_remove = []
    for idx, id in enumerate(results['class_ids']):
        if(id not in class_ids):
            ids_to_remove.append(idx)
    return filter_results(results, ids_to_remove)

def mask_to_polygon(mask):
    """turns a binary mask into a polygon

    Args:
        mask (numpy.ndarray): the binary mask predicted by mask rcnn

    Returns:
        numpy.ndarray: polygon which is a set of x,y coordinates that make up the mask
    """    
    return np.column_stack(np.where(mask > 0))

def polygon_area(polygon):
    """calculates the area of a polygon in 2D space

    Args:
        polygon (numpy.ndarray): array of x,y coordinates that make up the polygon

    Returns:
        float: area of the polygon
    """    
    return cv2.contourArea(polygon)

def postprocess_masks(results, frameArea, area_threshold, confidence_threshold):
    """Postprocesses output to remove undesirable masks

    Args:
        results (dict): results from mask rcnn prediction
        frameArea (int): area of frame (width*height)
        area_threshold (float): percentage of the area of frame that represents the maximum allowed area for a single mask
        confidence_threshold (float): minimum cofidence percentage per detection

    Returns:
        dict: cleaned results including only desirable masks
    """    
    indices = []
    for i in range(results['masks'].shape[2]):
        if((polygon_area(mask_to_polygon(results['masks'][:,:,i])) >= int(frameArea * area_threshold)) or (results['scores'][i] < confidence_threshold)):
            indices.append(i)
    return filter_results(results, indices)

def rle_encode(mask):
    """Encodes a mask in Run Length Encoding (RLE).
    Returns a string of space-separated values.
    """
    assert mask.ndim == 2, "Mask must be of shape [Height, Width]"
    # Flatten it column wise
    m = mask.T.flatten()
    # Compute gradient. Equals 1 or -1 at transition points
    g = np.diff(np.concatenate([[0], m, [0]]), n=1)
    # 1-based indicies of transition points (where gradient != 0)
    rle = np.where(g != 0)[0].reshape([-1, 2]) + 1
    # Convert second index in each pair to lenth
    rle[:, 1] = rle[:, 1] - rle[:, 0]
    return " ".join(map(str, rle.flatten()))

def mask_to_rle(image_id, mask, scores):
    "Encodes instance masks to submission format."
    assert mask.ndim == 3, "Mask must be [H, W, count]"
    # If mask is empty, return line with image ID only
    if mask.shape[-1] == 0:
        return "{},".format(image_id)
    # Remove mask overlaps
    # Multiply each instance mask by its score order
    # then take the maximum across the last dimension
    order = np.argsort(scores)[::-1] + 1  # 1-based descending
    mask = np.max(mask * np.reshape(order, [1, 1, -1]), -1)
    # Loop over instance masks
    lines = []
    for o in order:
        m = np.where(mask == o, 1, 0)
        # Skip if empty
        if m.sum() == 0.0:
            continue
        rle = rle_encode(m)
        lines.append("{}, {}".format(image_id, rle))
    return "\n".join(lines)

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    """resizes an image while maintaining aspect ratio

    Args:
        image (image): the image to be resized
        width (int, optional): width to resize to. Defaults to None.
        height (int, optional): height to resize to. Defaults to None.
        inter (cv2 constant, optional): interpolation method. Defaults to cv2.INTER_AREA.

    Returns:
        _type_: resized image
    """    
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized

def preprocess_image_mobilenet(img_path, dim):
    """process image to be mobilenet friendly

    Args:
        img_path (string): path to image
        dim (tuple): target size 

    Returns:
        numpy array: pre-processed image
    """    
    img = image.load_img(img_path, target_size=dim)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    pImg = mobilenet.preprocess_input(img_array)
    return pImg

def index_to_label(index, model):
    """Retrieves class name from index

    Args:
        index (int): index of class
        model (string): model name

    Raises:
        NameError: if model name is not known

    Returns:
        string: class name
        int: number of classes for model
    """    
    if(model == "color"):
        with open(COLOR_CLASS_INDEX_PATH) as f:
            classes = json.loads(f.read())
    elif(model == "type"):
        with open(TYPE_CLASS_INDEX_PATH) as f:
            classes = json.loads(f.read())
    else:
        raise NameError(f"[ERROR] {model} model not specified.")

    return len(classes), classes[str(index)]  

def decode_mobilenet_predictions(predictions, model, top=1):
    """decodes predictions made by mobilenet

    Args:
        predictions (numpy array): output of mobilenet
        top (int, optional): number of top predictions to return. Defaults to 1.

    Returns:
        list: class names predicted
    """
    len,  = index_to_label(0, model)
    if(predictions.shape[1] != len):
        raise ValueError(f"[ERROR] decode_mobilenet_predictions expects a batch of predictions\
            with shape (samples, {len}). Found array with shape {predictions.shape}")
    
    results = []
    for pred in predictions:
        top_indices = pred.argsort()[-top:][::-1]
        top_confs = sorted(pred, reverse=True)[:top]
        result = [(index_to_label(index, model)[1], top_confs[count]) for count, index in enumerate(top_indices)]
        results.append(result)
    return results

def postprocess_tubes(path, minDetections):
    """Postprocesses tubes after detection is over

    Args:
        path (string): path to tubes folder
        minDetections (int): minimum number of detections per tube to counted as a tube
        
    Returns:
        list: tubes that were removed
    """
    assert os.path.isdir(path), f"[ERROR] {path} does not exist"
    tubes = os.listdir(path)
    tubes_to_remove = []
    # Check which tubes have less than the minimum number of detections
    for tube in tubes:
        if(len(os.listdir(os.path.join(path, tube))) < minDetections):
            tubes_to_remove.append(tube)
    # Remove tubes
    for tube in tubes_to_remove:
        shutil.rmtree(os.path.join(path, tube), ignore_errors=False, onerror=None)
    return tubes_to_remove

def alphabet_permutations(n):
    """Creates a list of alphabet permutations

    Args:
        n (int): length of list to create

    Returns:
        list: list of n permutations
    """    
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    res = [alphabet[i: j] for i in range(len(alphabet)) 
            for j in range(i + 1, len(alphabet) + 1)]
    return sorted(res, key =len)[:n]

def mask_to_boundary(mask, dilation_ratio=0.02):
    """
    Convert binary mask to boundary mask.
    :param mask (numpy array, uint8): binary mask
    :param dilation_ratio (float): ratio to calculate dilation = dilation_ratio * image_diagonal
    :return: boundary mask (numpy array)
    """
    h, w = mask.shape
    img_diag = np.sqrt(h ** 2 + w ** 2)
    dilation = int(round(dilation_ratio * img_diag))
    if dilation < 1:
        dilation = 1
    # Pad image so mask truncated by the image border is also considered as boundary.
    new_mask = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    kernel = np.ones((3, 3), dtype=np.uint8)
    new_mask_erode = cv2.erode(new_mask, kernel, iterations=dilation)
    mask_erode = new_mask_erode[1 : h + 1, 1 : w + 1]
    # G_d intersects G in the paper.
    return mask - mask_erode

def boundary_iou(gt, dt, dilation_ratio=0.02):
    """
    Compute boundary iou between two binary masks.
    :param gt (numpy array, uint8): binary mask
    :param dt (numpy array, uint8): binary mask
    :param dilation_ratio (float): ratio to calculate dilation = dilation_ratio * image_diagonal
    :return: boundary iou (float)
    """
    gt_boundary = mask_to_boundary(gt, dilation_ratio)
    dt_boundary = mask_to_boundary(dt, dilation_ratio)
    intersection = ((gt_boundary * dt_boundary) > 0).sum()
    union = ((gt_boundary + dt_boundary) > 0).sum()
    boundary_iou = intersection / union
    return boundary_iou

def overlap_masks(tube1, tube2, frame1, frame2):
    """Computes the overlap between two tubes at a certain frame

    Args:
        tube1 (string): first tube path
        tube2 (string): second tube path
        frame1 (int): frame number
        frame2 (int): frame number

    Returns:
        float: IoU
    """    
    frame_tube1 = os.listdir(f"tubes/{tube1}")[frame1]
    frame_tube2 = os.listdir(f"tubes/{tube2}")[frame2]
    # im1 = mask_to_boundary(cv2.imread(glob.glob(f'{tube1}/*_{str(frame1)}_*.png')[0])[:, :, 2])
    # im2 = mask_to_boundary(cv2.imread(glob.glob(f'{tube2}/*_{str(frame2)}_*.png')[0])[:, :, 2])
    im1 = mask_to_boundary(cv2.imread(f"tubes/{tube1}/{frame_tube1}")[:, :, 2])
    im2 = mask_to_boundary(cv2.imread(f"tubes/{tube2}/{frame_tube2}")[:, :, 2])
    return boundary_iou(im1, im2)

def plot_classification_report(classificationReport, title='Classification report', cmap='RdBu'):
    """Plots the result of sklearn's classification report

    Args:
        classificationReport (string): the classification report of the model
        title (str, optional): title of the plot. Defaults to 'Classification report'.
        cmap (str, optional): colormap instance. Defaults to 'RdBu'.
    """    

    classificationReport = classificationReport.replace('\n\n', '\n')
    classificationReport = classificationReport.replace(' / ', '/')
    lines = classificationReport.split('\n')

    classes, plotMat, support, class_names = [], [], [], []
    for line in lines[1:]:  # if you don't want avg/total result, then change [1:] into [1:-1]
        t = line.strip().split()
        if len(t) < 2:
            continue
        classes.append(t[0])
        v = [float(x) for x in t[1: len(t) - 1]]
        support.append(int(t[-1]))
        class_names.append(t[0])
        plotMat.append(v)

    plotMat = np.array(plotMat)
    xticklabels = ['Precision', 'Recall', 'F1-score']
    yticklabels = ['{0} ({1})'.format(class_names[idx], sup)
                   for idx, sup in enumerate(support)]

    plt.imshow(plotMat, interpolation='nearest', cmap=cmap, aspect='auto')
    plt.title(title)
    plt.colorbar()
    plt.xticks(np.arange(3), xticklabels, rotation=45)
    plt.yticks(np.arange(len(classes)), yticklabels)

    upper_thresh = plotMat.min() + (plotMat.max() - plotMat.min()) / 10 * 8
    lower_thresh = plotMat.min() + (plotMat.max() - plotMat.min()) / 10 * 2
    for i, j in itertools.product(range(plotMat.shape[0]), range(plotMat.shape[1])):
        plt.text(j, i, format(plotMat[i, j], '.2f'),
                 horizontalalignment="center",
                 color="white" if (plotMat[i, j] > upper_thresh or plotMat[i, j] < lower_thresh) else "black")

    plt.ylabel('Metrics')
    plt.xlabel('Classes')
    plt.tight_layout()
