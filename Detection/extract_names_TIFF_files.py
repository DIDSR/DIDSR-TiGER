import os
def extract_names_TIFF_files(directory):
    #This function extract names of the TIFF files stored in a directory and puts them in 3 groups: 
    # files starting with TCGA, TC and everything else
    images_names = os.listdir(directory)
    TIFF_images_names = {}
    TIFF_images_names['TCGA'] = []
    TIFF_images_names['TC'] = []
    TIFF_images_names['JB'] = []
    for i in images_names:
        if i.endswith('tif'):
            if i.startswith('TCGA'):
                TIFF_images_names['TCGA'].append(i)
            elif i.startswith('TC_S01'):
                TIFF_images_names['TC'].append(i)
            else:
                TIFF_images_names['JB'].append(i)
    TIFF_images_names['TCGA'].sort()
    TIFF_images_names['TC'].sort()
    TIFF_images_names['JB'].sort()
    return TIFF_images_names