import numpy as np
import os
import subprocess
from scipy.misc import imread,imresize
import pandas as pd

def fetch_lfw_dataset(attrs_name = "lfw_attributes.txt",
                      images_name = "lfw-deepfunneled",
                      raw_images_name = "lfw",
                      use_raw=False,
                      dx=80,dy=80,
                      dimx=45,dimy=45):

    # read attrs
    # the header row begins with a #, which we want to ignore
    with open(attrs_name) as attributes_file:
        attributes_file.readline()
        ugly_header = attributes_file.read(2)
        assert ugly_header == "#\t"
        df_attrs = pd.read_csv(attributes_file, sep='\t', skipinitialspace=True) 
    
    #read photos
    dirname = raw_images_name if use_raw else images_name
    photo_ids = []
    initial_depth = dirname.count(os.sep)
    for dirpath, dirnames, filenames in os.walk(dirname):
        if dirpath.count(os.sep) - initial_depth > 1:
            continue
        for fname in filenames:
            if fname.endswith(".jpg"):
                photo_id = fname[:-4].replace('_',' ').split()
                person_id = ' '.join(photo_id[:-1])
                photo_number = int(photo_id[-1])
                fpath = os.path.join(dirpath, fname)
                photo_ids.append({'person':person_id,'imagenum':photo_number,'photo_path':fpath})

    photo_ids = pd.DataFrame(photo_ids)

    # mass-merge
    # (photos now have same order as attributes)
    df = pd.merge(df_attrs,photo_ids,on=('person','imagenum'))

    assert len(df)==len(df_attrs),"lost some data when merging dataframes"

    #image preprocessing
    all_photos =df['photo_path'].apply(imread)\
                                .apply(lambda img:img[dy:-dy,dx:-dx])\
                                .apply(lambda img: imresize(img,[dimx,dimy]))

    all_photos = np.stack(all_photos.values).astype('uint8')
    all_attrs = df.drop(["photo_path","person","imagenum"], axis=1)
    
    return all_photos, all_attrs
    
