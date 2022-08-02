import numpy as np
from tqdm import tqdm
import segmentation_models as sm

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
    for y in tqdm(range(-64, dimensions[1], tile_size)):
        for x in range(-64, dimensions[0], tile_size):
            tissue_mask_tile = tissue_mask.getUCharPatch(
                startX=x, startY=y, width=tile_size*2, height=tile_size*2, level=level
            ).squeeze()
            
            if not np.any(tissue_mask_tile):
                continue

            image_tile = image.getUCharPatch(startX=x, startY=y, width=tile_size*2, height=tile_size*2, level=level)
            
            image_tile_preprocess = preprocess_input(image_tile)
            image_tile_preprocess = np.expand_dims(image_tile_preprocess, axis = 0)
            
            image_tile_det = (image_tile/255.0).astype('float32')            
            image_tile_det = np.expand_dims(image_tile_det, axis = 0)            
            
            predicted_mask = seg_model.predict(image_tile_preprocess, batch_size = 32, verbose = 0)
            predicted_mask = predicted_mask[0,:,:,:]
            predicted_mask = np.argmax(predicted_mask, axis = 2)
            predicted_mask = predicted_mask[64:192,64:192]
            predicted_mask[predicted_mask==0] = 3            
            predicted_mask = predicted_mask*tissue_mask_tile[64:192,64:192]             
            segmentation_writer.write_segmentation(tile=predicted_mask, x=x+64, y=y+64)  
            
            detections = cell_det_model.predict(image_tile_det,  batch_size = 32, verbose = 0)
            detections = detections[0,:,:,0]                        
            detections = extract_predictions(detections, confidence_threshold = 0.1)
            detections = non_max_supression_distance(detections,distance_threshold = 12)            
            wanted = []
            for n in detections:
                if n[0]>=65 and n[0]<=191 and n[1]>=65 and n[1]<=191:
                    wanted.append(n)            
            det_wanted = []        
            if len(wanted)>0:
                det_wanted = [i for i in wanted if tissue_mask_tile[int(i[1]),int(i[0])]==1]
            detection_writer.write_detections(detections=det_wanted, spacing=spacing, x_offset=x, y_offset=y) 
            
            # TILs Score
            if len(det_wanted)>0:
                for i in det_wanted:
                    if predicted_mask[int(i[1]-64),int(i[0]-64)] == 2:
                        no_of_TILs += 1           
            no_of_stroma += np.count_nonzero(predicted_mask==2)
        
    print("Saving...")
    # save segmentation and detection
    segmentation_writer.save()
    detection_writer.save()

    print('Number of detections', len(detection_writer.detections))
    
    print("Compute tils score...")
    # compute tils score
    tils_score = int((no_of_TILs*16*16/no_of_stroma)*100)
    tils_score_writer.set_tils_score(tils_score=tils_score)
    print('TILs Score = '+str(tils_score))

    print("Saving...")
    # save tils score
    tils_score_writer.save()

    print("Copy data...")
    # save data to output folder
    copy_data_to_output_folders()

    print("Completed!")
