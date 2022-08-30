import cv2
import sys
import os
import matplotlib.pyplot as plt
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
import numpy as np
import utils
import time
import random
import json
from centroidtracker import CentroidTracker
import poisson
import mobilenetclassifier
import basinhopping
import shutil
#from filevideostream import FileVideoStream
# remove later, remove utils from virtualenv
from imutils.video import FPS
from filevideostream import FileVideoStream

# Root directory of the project
ROOT_DIR = os.path.abspath("Mask_RCNN/")
# Import Mask RCNN
sys.path.append(ROOT_DIR)
# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
import coco

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']

if __name__ == '__main__':

    # Load framework configurations
    try:
        with open("config-file.json" ) as f:
            # Parsing configuration parameters
            configs = json.loads(f.read())
            sampling_rate = configs['sampling_rate']
            time_interval = configs['time_interval']
            max_disappeared = configs['max_disappeared']
            width = configs['width']
            video_path = configs['video_path']
            queue_buffer_size = configs['queue_buffer_size']
            area_threshold = configs['area_threshold']
            confidence_threshold = configs['confidence_threshold']
            minimum_detections = configs['minimum_detections']
            ipt_color = configs['ipt_color']
            ipt_type = configs['ipt_type']
            color_input_size = eval(configs['color_input_size'])
            color_output_size = configs['color_output_size']
            color_weights = configs['color_weights']
            type_input_size = eval(configs['type_input_size'])
            type_output_size = configs['type_output_size']
            type_weights = configs['type_weights']
            filter_color = configs['filter_color']
            filter_type = configs['filter_type']
            optimization_iterations = configs['optimization_iterations']
            optimization_stepsize = configs['optimization_stepsize']
            optimized_mapping = configs['optimized_mapping']
    except EnvironmentError:
        raise NameError('[ERROR] Problem reading and/or parsing config file.')

    assert os.path.isfile(video_path), f"[ERROR] {video_path} file does not exist"

    config = InferenceConfig()

    # Create model object in inference mode.
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

    # Load weights trained on MS-COCO
    model.load_weights(COCO_MODEL_PATH, by_name=True)

    # start the file video stream thread and allow the buffer to
    # start to fill
    print("[INFO] Starting video file thread...")
    fvs = FileVideoStream(video_path, queue_size = queue_buffer_size).start()
    # Sleep for one second while the queue fills up
    time.sleep(1.0)
    # Get fps of video
    fps = fvs.fps
    print(f"[INFO] FPS of video is: {fps}")

    # start the FPS timer
    fps_timer = FPS().start()

    # Initialize Centroid Tracker
    tracker = CentroidTracker(maxDisappeared = max_disappeared)

    # Initialize arguments
    detections_per_frame = {}
    frameNbr = 0

    # Loop over frames from the video file stream
    while fvs.more():

        # Capture frame-by-frame
        frame = fvs.read()
        frame_original = frame
        
        # Check if stream reached its end
        if frame is None:
            print("[INFO] Can't receive frame (stream end?). Exiting ...")
            break

        # Resize the frame
        frame = utils.image_resize(frame, width = width)
        # Getting the new height
        height, *rest = frame.shape

        # Run inference on frame using mrcnn model
        result = utils.mrcnn_inference(model, frame)

        # Filter classes
        r = utils.filter_classes(result[0], [class_names.index('car'), class_names.index('truck'), class_names.index('bus')])

        # Post process the output
        new_results = utils.postprocess_masks(r, width*height, area_threshold, confidence_threshold)
        
        # Swap x,y coordinates in preparation for tracker
        for box in new_results['rois']:
            box[0], box[1], box[2], box[3] = box[1], box[0], box[3], box[2]
        
        # Update tracker with bounding boxes
        objects = tracker.update(new_results['rois'])

        # Visualize tracking results
        for (objectID, centroid) in objects.items():
            text = "id: {}".format(objectID)
            cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

        # Extract masks
        for index, bbox in enumerate(new_results['rois']):
            center =  np.array([int((bbox[0]+ bbox[2])/2), int((bbox[1]+ bbox[3])/2)])
            found = False
            for (objectID, centroid) in objects.items():
                if((center==centroid).all()):
                    found = True
                    path_masks = f"tubes/{objectID}/"
                    path_bboxes = f"bboxes/{objectID}/"
                    # Get number of object in tube
                    if(not os.path.exists(path_bboxes)):
                        os.makedirs(path_bboxes)
                    if(not os.path.exists(path_masks)):
                        os.makedirs(path_masks)
                        objectNbr = "0"
                    else:
                        objectNbr = str(int(sorted([int(os.path.splitext(element)[0].split("_")[0]) for element in os.listdir(path_masks)])[-1]) + 1)
                    # Save bbox for visualization and content analysis
                    cv2.imwrite(f"{path_bboxes}/{objectNbr}_{frameNbr}_{objectID}.png", frame_original[bbox[1]:bbox[3], bbox[0]:bbox[2]])
                    # Save mask
                    mask = new_results['masks'][:,:,index].astype(np.uint8)
                    mask*=255
                    # mask name: (instance of ID)_(number of frame it appears in)_(ID).png
                    cv2.imwrite(f"{path_masks}/{objectNbr}_{frameNbr}_{objectID}.png", mask)

        # Update number of detections per frame
        detections_per_frame[frameNbr] = len(new_results['class_ids'])
          
        # Generate a background for each time period
        background_path = "background images/"
        backgrounds_generated = frameNbr / int(fps * time_interval)
        if(frameNbr % int(fps * time_interval) == 0 and frameNbr!=0):
            detections_per_frame = dict(sorted(detections_per_frame.items(), key=lambda item: item[1]))
            frames_to_sample = list(detections_per_frame.keys())[:int(fps * time_interval * sampling_rate)]
            if(not os.path.exists(background_path)):
                os.makedirs(background_path)
            cv2.imwrite(f"{background_path}{str(backgrounds_generated)}.png", utils.generate_background(video_path, frames_to_sample))
            print(f"[INFO] Creating background image#{backgrounds_generated}")
            detections_per_frame.clear()
        else:
            print(str(frameNbr % int(fps * time_interval)))
            print(str(frameNbr % (fps * time_interval)))
        
        #Visualize the post-processed results
        # Swap x,y coordinates in preparation for tracker
        # for box in new_results['rois']:
        #     box[0], box[1], box[2], box[3] = box[1], box[0], box[3], box[2]
        # visualize.display_instances(frame, new_results['rois'], new_results['masks'], new_results['class_ids'], 
        # class_names, new_results['scores'])
        # #display the size of the queue on the frame
        # cv2.putText(frame, "Queue Size: {}".format(fvs.Q.qsize()),
        #     (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imwrite(f"frames-with-tracking/{frameNbr}.png", frame)

        print("[INFO] Frame {} done.\n".format(str(frameNbr)))
        frameNbr+=1

        if cv2.waitKey(1) == ord('q'):
            break
    
    # Create masks
    # tubes = [os.listdir(f'tubes/{tube}') for tube in os.listdir("tubes/")]
    # print(tubes)
    # groupings = [list(i) for i in zip(*tubes)]
    # max_len = max(len(i) for i in tubes)
    # groupings = [[i[o] for i in tubes if len(i) > o] for o in range(max_len)]
    # result = 0
    # for masks in groupings:
    #     for mask in masks:
    #         img = cv2.imread(os.path.join('tubes', mask.split("_")[2].split(".")[0], mask)).astype("float32")
    #         result += 255*img
    #     result = result.clip(0, 255).astype("uint8")
    #     cv2.imshow('result', result)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()
    #     result = 0

    # Post-process tubes
    tubes_removed = utils.postprocess_tubes("tubes/", minimum_detections)
    tubes_removed = utils.postprocess_tubes("bboxes/", minimum_detections)
    if(tubes_removed):
        print(f"[INFO] {tubes_removed} tubes removed in postprocessing")
    else:
        print("[INFO] No tubes were removed in postprocessing")

    # Content Analysis
    color_classifier = mobilenetclassifier.mobilenet_color_model(color_input_size, color_output_size, color_weights)
    type_classifier = mobilenetclassifier.mobilenet_type_model(type_input_size, type_output_size, type_weights)
    tubes_to_remove = []
    # Run prediction over each tube
    for tube in os.listdir('bboxes'):
        nbrDetections = len(os.listdir(f'bboxes/{tube}'))
        # Check if the number of detections is less than ipt
        if(nbrDetections < ipt_color):
            ipt_color = nbrDetections
        if(nbrDetections < ipt_type):
            ipt_type = nbrDetections
        # Prepare frames for prediction
        frames = [utils.preprocess_image_mobilenet(os.path.join('bboxes', tube, file), color_input_size[:2]) for file in os.listdir(f'bboxes/{tube}') if file.endswith(".png")]
        random.shuffle(frames)
        frames_color = frames[:ipt_color]
        frames_type = frames[:ipt_type]
        predictions_color = color_classifier.predict(np.vstack(frames_color))
        predictions_type = type_classifier.predict(np.vstack(frames_type))
        # Determine label for tube
        # Sum confidence scores across axis 0
        # Determine label of the argmax of the result
        summed_color = np.sum(predictions_color, axis=0)
        summed_type = np.sum(predictions_type, axis=0)
        result_color = utils.index_to_label(np.argmax(summed_color, axis=0), "color")[1]
        result_type = utils.index_to_label(np.argmax(summed_type, axis=0), "type")[1]
        # Save label as txt file
        with open(f"bboxes/{tube}/labels.txt", "w+") as f:
            f.write(" ".join([result_color, result_type]))
        # Get tubes to filter out
        if(result_color not in filter_color or result_type not in filter_type):
            tubes_to_remove.append(tube)
    
    # Filter out tubes
    if(tubes_to_remove):    
        for tube in tubes_to_remove:
            try:
                shutil.rmtree(f"tubes/{tube}")
                print(f"[ERROR] Removed tube {tube}")
            except:
                print(f"[ERROR] Could not find {tube}")
        print(f"[INFO] Tubes {tubes_to_remove} filtered out according to filtration conditions")    
    else:
        print("[INFO] No tubes filtered out")
    
    # Tube shifting using Basin Hopping optimization and Object Stitching
    if(optimized_mapping):
        optimize = basinhopping.TubeShifting("tubes/", optimization_iterations, optimization_stepsize)
        solution = optimize.start()
        optimized_mapping = {}
        tubes = os.listdir('tubes/')

        # Create optimized mapping dictionary
        # Find End times of tubes in new mapping
        for idx, startTime in enumerate(solution):
            optimized_mapping[tubes[idx]] = (startTime, startTime + len(os.listdir(os.path.join('tubes/', tubes[idx]))) -1)
        print(f"[INFO] Optimal mapping (Tube: (startTime,)) is {optimized_mapping}")

        # Object Stitching using optimal mapping
        #Find overall length of video
        min = list(optimized_mapping.values())[0][0]
        max = list(optimized_mapping.values())[0][1]
        for elem in  list(optimized_mapping.values()):
            if(elem[0] < min):
                min = elem[0]
            if(elem[1] > max):
                max = elem[1]
        video_length = max - min + 1
        print(f"[INFO] Number of frames in output video: {video_length}")

        # Read video
        cap = cv2.VideoCapture(video_path)
        # Check if video opened successfully
        if (cap.isOpened() == False):
            raise NameError("[ERROR] Error opening video file")

        for i in range(video_length):
            print(f"[INFO] Stitching frame {i}")
            # Assume we have one background image for now
            bg = cv2.imread('background images/1.0.png')
            for tube in optimized_mapping.keys():
                if(i in range(optimized_mapping[tube][0], optimized_mapping[tube][1])):
                    files = os.listdir(f"tubes/{tube}/")
                    files.sort(key=lambda x:int(x.split("_")[0]))
                    nbrOfInstance = i - optimized_mapping[tube][0] + 1
                    mask = cv2.imread(f"tubes/{tube}/{files[nbrOfInstance]}", 0)
                    source = utils.return_frame(cap, int(files[nbrOfInstance].split("_")[1]))
                    bg = poisson.process(mask, source, bg)
                else:
                    # This frame does not contain an instance of this tube
                    continue
            cv2.imwrite(f"output/{str(i)}.png", bg)
            print(f"[INFO] Finished stitching frame {i}")
    else:
        # Object Stitching using best-case scenario mapping
        # map all tubes to time 0
        tubes = os.listdir('tubes/')
        backgrounds_generated = 3
        # Find max length of tube --> number of frames in output video
        max_tube_len = max([len(os.listdir(f'tubes/{tube}')) for tube in tubes])
        print(f"[INFO] Number of frames in output video: {max_tube_len}")
        # Find how many frames to create using each created background frame
        frames_per_background = int(max_tube_len/backgrounds_generated)
        print(f"[INFO] {frames_per_background} frames for each generated background")

        cap = cv2.VideoCapture(video_path)
        # Check if video opened successfully
        if (cap.isOpened() == False):
            raise NameError("[ERROR] Error opening video file")

        print('[INFO] Stitching started...')
        for i in range(max_tube_len):
            print(f"[INFO] Stitching frame {i}")
            # Assume we have one background image for now
            bg = cv2.imread('background images/1.0.png')
            for tube in tubes:
                files = os.listdir(f'tubes/{tube}')
                if(len(files) < i+1):
                    continue
                mask = cv2.imread(f"tubes/{tube}/{files[i]}", 0)
                source = utils.return_frame(cap, int(files[i].split("_")[1]))
                bg = poisson.process(mask, source, bg)
            cv2.imwrite(f"output/{str(i)}.png", bg)
            print(f"[INFO] Finished stitching frame#{i}")

    # Create synopsis video from frames
    img_array = []
    for file in os.listdir("output/"):
        img = cv2.imread("output/" + file)
        height, width, layers = img.shape
        size = (width,height)
        img_array.append(img)
    # Create video writer instance
    out = cv2.VideoWriter('synopsis.mp4', cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
    
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()

    # stop the timer and display FPS information
    fps_timer.stop()
    print("[INFO] elasped time: {:.2f}".format(fps_timer.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format((frameNbr+1)/fps_timer.elapsed()))

    # Closes all the windows currently opened
    cv2.destroyAllWindows()

    # Video capturing from video files
    # cap = cv2.VideoCapture(path)
    # # Get fps of video
    # fps = cap.get(cv2.CAP_PROP_FPS)

    # Loop until the end of the video
    # while (cap.isOpened()):
    
    #     # Capture frame-by-frame
    #     ret, frame = cap.read()

    #     # if frame is read correctly ret is True
    #     if not ret:
    #         print("[ERROR] Can't receive frame (stream end?). Exiting ...")
    #         break

    #     # Resize the frame
    #     frame = cv2.resize(frame, (width,height))

    #     # Run inference on frame
    #     result = utils.mrcnn_inference(model, frame)

    #     # Filter classes
    #     r = utils.filter_classes(result[0], [class_names.index('car'), class_names.index('truck'), class_names.index('bus')])

    #     # Post process the output
    #     new_results = utils.postprocess_masks(r, width*height)

    #     # Update tracker with bounding boxes
    #     objects = tracker.update(new_results['rois'])
    #     for (objectID, centroid) in objects.items():
    #         print("object-id: {}".format(objectID))

    #     # Update number of detections per frame
    #     detections_per_frame[frameNbr] = len(new_results['class_ids'])
        
    #     # Generate a background for each time period
    #     if(frameNbr % (fps * 10) == 0 and frameNbr!=0):
    #         detections_per_frame = dict(sorted(detections_per_frame.items(), key=lambda item: item[1]))
    #         frames_to_sample = list(detections_per_frame.keys())[:int(fps * 10 * sampling_rate)]
    #         cv2.imwrite("D:/Video Synopsis Tool/background images/" + str(int(frameNbr/fps))+"_bg.png", utils.generate_background(path, frames_to_sample))
    #         detections_per_frame.clear()
        
    #     # Visualize the post-processed results
    #     # visualize.display_instances(frame, new_results['rois'], new_results['masks'], new_results['class_ids'], 
    #     # class_names, new_results['scores'])

    #     print("frame {} done.\n".format(str(frameNbr)))
    #     frameNbr+=1
    #     if cv2.waitKey(1) == ord('q'):
    #         break
    
    # release the video capture object
    # cap.release()
