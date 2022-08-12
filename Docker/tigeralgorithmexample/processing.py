import numpy as np
from tqdm import tqdm
import segmentation_models as sm
import tensorflow as tf
tf.config.list_physical_devices('GPU')
sm.set_framework('tf.keras')
sm.framework()

cell_det_model = sm.Unet(backbone_name = 'inceptionv3', input_shape= (256,256,3), encoder_weights = None)
cell_det_model.load_weights('/home/user/det.h5')

seg_model = sm.Unet(backbone_name = 'inceptionv3', input_shape= (256,256,3), encoder_weights=None, classes = 3, activation = 'Softmax')
seg_model.load_weights('/home/user/seg.hdf5')

preprocess_input = sm.get_preprocessing('inceptionv3')

from .gcio import (
    TMP_DETECTION_OUTPUT_PATH,
    TMP_SEGMENTATION_OUTPUT_PATH,
    TMP_TILS_SCORE_PATH,
    copy_data_to_output_folders,
    get_image_path_from_input_folder,
    get_tissue_mask_path_from_input_folder,
    initialize_output_folders,
)
from .rw import (
    READING_LEVEL,
    WRITING_TILE_SIZE,
    DetectionWriter,
    SegmentationWriter,
    TilsScoreWriter,
    open_multiresolutionimage_image,
)

from scipy.spatial.distance import cdist
def extract_predictions(probabilities, confidence_threshold):    
    indices = np.meshgrid(np.arange(0,probabilities.shape[1]),np.arange(0,probabilities.shape[0]))
    indices_x = indices[0]
    indices_y = indices[1]
    indices_x = indices_x.reshape((probabilities.shape[0]*probabilities.shape[1],1))
    indices_y = indices_y.reshape((probabilities.shape[0]*probabilities.shape[1],1))    
    probabilities = probabilities.reshape((probabilities.shape[0]*probabilities.shape[1],1))    
    boxes_pred = np.concatenate((indices_x,indices_y,probabilities),axis = 1)
    boxes_pred = boxes_pred[np.argsort(boxes_pred[:, 2])[::-1]]
    boxes_pred = boxes_pred[boxes_pred[:,2]>=confidence_threshold,:]   
    return boxes_pred

def non_max_supression_distance(points, distance_threshold):
    log_val = np.ones(points.shape[0])
    wanted = []
    for i in range(points.shape[0]):
        if log_val[i]:
            hit = cdist(np.expand_dims(points[i,:2],0),points[:,:2])
            hit = np.argwhere(hit<=distance_threshold)
            log_val[hit] = 0
            wanted.append(points[i,:])
    wanted = np.array(wanted)  
    return wanted

def run_segmentation_detection(imgs, msks):
    imgs_seg = [preprocess_input(i) for i in imgs]
    imgs_det = [i/255.0 for i in imgs]
    if len(imgs_seg)>1:
        imgs_seg = np.array(imgs_seg)
        imgs_det = np.array(imgs_det)
        msks = np.array(msks)
    else:
        imgs_seg = np.expand_dims(imgs_seg, axis = 0)
        imgs_det = np.expand_dims(imgs_det, axis = 0) 
        msks = np.expand_dims(msks, axis = 0)                  
    
    predicted_masks_seg = seg_model.predict(imgs_seg, batch_size = 32, verbose = 0)    
    predicted_masks_seg = np.argmax(predicted_masks_seg, axis = 3)            
    predicted_masks_seg[predicted_masks_seg==0] = 3            
    predicted_masks_seg = predicted_masks_seg*msks          
    
    predicted_masks_det = cell_det_model.predict(imgs_det,  batch_size = 32, verbose = 0)
    predicted_masks_det = predicted_masks_det[:,:,:,0]                           
    detections = []    
    for n in range(len(predicted_masks_det)):        
        tils = extract_predictions(predicted_masks_det[n], confidence_threshold = 0.1)
        tils = non_max_supression_distance(tils,distance_threshold = 12)                   
        if tils.shape[0]>0:
            tils = [tils[i,:] for i in range(tils.shape[0]) if msks[n][int(tils[i,1]),int(tils[i,0])]==1]
        detections.append(tils)
    return predicted_masks_seg, detections

def process():  
    
    level = READING_LEVEL
    tile_size = WRITING_TILE_SIZE # should be a power of 2

    initialize_output_folders()

    # get input paths
    image_path = get_image_path_from_input_folder()    
    tissue_mask_path = get_tissue_mask_path_from_input_folder() 

    print(f'Processing image: {image_path}')
    print(f'Processing with mask: {tissue_mask_path}')

    # open images
    image = open_multiresolutionimage_image(path=image_path)
    tissue_mask = open_multiresolutionimage_image(path=tissue_mask_path)

    # get image info
    dimensions = image.getDimensions()
    spacing = image.getSpacing()

    # create writers 
    segmentation_writer = SegmentationWriter(
        TMP_SEGMENTATION_OUTPUT_PATH,
        tile_size=tile_size,
        dimensions=dimensions,
        spacing=spacing,
    )  
    detection_writer = DetectionWriter(TMP_DETECTION_OUTPUT_PATH)
    tils_score_writer = TilsScoreWriter(TMP_TILS_SCORE_PATH)        

    print("Processing image...")
    # loop over image and get tiles
    no_of_stroma = 0
    no_of_TILs = 0    
    x_vals = []
    y_vals = []
    img_tiles = []
    tissue_masks = []
    counts = 0    
    y_range = range(0, dimensions[1], tile_size)
    x_range = range(0, dimensions[0], tile_size)
    
    for y in tqdm(y_range):
        for x in x_range:
            tissue_mask_tile = tissue_mask.getUCharPatch(
                startX=x, startY=y, width=tile_size, height=tile_size, level=level
            ).squeeze()
            
            if not np.any(tissue_mask_tile):
                continue
                
            counts += 1

            image_tile = image.getUCharPatch(startX=x, startY=y, width=tile_size, height=tile_size, level=level)
            
            x_vals.append(x)
            y_vals.append(y)
            img_tiles.append(image_tile)
            tissue_masks.append(tissue_mask_tile)
            
            if counts == 512:                
                predicted_masks, predicted_tils = run_segmentation_detection(img_tiles, tissue_masks)
                for x,y,predicted_mask,detections in zip(x_vals,y_vals,predicted_masks,predicted_tils):
                    segmentation_writer.write_segmentation(tile=predicted_mask, x=x, y=y)  
                    detection_writer.write_detections(detections=detections, spacing=spacing, x_offset=x, y_offset=y) 
                    # TILs Score
                    if len(detections)>0:
                        for i in detections:
                            if predicted_mask[int(i[1]),int(i[0])] == 2:
                                no_of_TILs += 1
                    no_of_stroma += np.count_nonzero(predicted_mask==2) 
                counts = 0
                x_vals = []
                y_vals = []
                img_tiles = []
                tissue_masks = [] 
    
    if len(img_tiles)>0:
        predicted_masks, predicted_tils = run_segmentation_detection(img_tiles, tissue_masks)
        for x,y,predicted_mask,detections in zip(x_vals,y_vals,predicted_masks,predicted_tils):
            segmentation_writer.write_segmentation(tile=predicted_mask, x=x, y=y)  
            detection_writer.write_detections(detections=detections, spacing=spacing, x_offset=x, y_offset=y) 
            # TILs Score
            if len(detections)>0:
                for i in detections:
                    if predicted_mask[int(i[1]),int(i[0])] == 2:
                        no_of_TILs += 1
            no_of_stroma += np.count_nonzero(predicted_mask==2)  
        
    print("Saving...")
    # save segmentation and detection
    segmentation_writer.save()
    detection_writer.save()

    print('Number of detections', len(detection_writer.detections))
    
    print("Compute tils score...")
    # compute tils score
    tils_score = int(np.round(no_of_TILs*16*16/no_of_stroma*100))
    tils_score_writer.set_tils_score(tils_score=tils_score)
    print('TILs Score = '+str(tils_score))

    print("Saving...")
    # save tils score
    tils_score_writer.save()

    print("Copy data...")
    # save data to output folder
    copy_data_to_output_folders()

    print("Completed!")
