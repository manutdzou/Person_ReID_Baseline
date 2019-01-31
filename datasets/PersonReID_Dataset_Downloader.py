from __future__ import print_function
import warnings
warnings.filterwarnings('ignore','.*conversion.*')

import os
import h5py
import zipfile
import shutil
import requests
import numpy as np
from PIL import Image
import argparse

################################
# Dataset with Google Drive IDs#
################################

dataset = {
    'CUHK03': '1BO4G9gbOTJgtYIB0VNyHQpZb8Lcn-05m',
    'Market1501': '0B2FnquNgAXonU3RTcE1jQlZ3X0E',
    'Market1501Attribute' : '1YMgni5oz-RPkyKHzOKnYRR2H3IRKdsHO',
    'DukeMTMC': '1qtFGJQ6eFu66Tt7WG85KBxtACSE8RBZ0',
    'DukeMTMCAttribute' : '1eilPJFnk_EHECKj2glU_ZLLO7eR3JIiO'
}

##########################
# Google Drive Downloader#
##########################

def gdrive_downloader(destination, id):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)    

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)

#############################
# Convert CUHK03 into Images#
#############################
def cuhk03_to_image(CUHK03_dir):
    
    f = h5py.File(os.path.join(CUHK03_dir,'cuhk-03.mat'))

    detected_labeled = ['detected','labeled']
    print('converting')
    for data_type in detected_labeled:

        datatype_dir = os.path.join(CUHK03_dir, data_type)
        if not os.path.exists(datatype_dir):
                os.makedirs(datatype_dir)

        for campair in range(len(f[data_type][0])):
            campair_dir = os.path.join(datatype_dir,'P%d'%(campair+1))
            cam1_dir = os.path.join(campair_dir,'cam1')
            cam2_dir = os.path.join(campair_dir,'cam2')

            if not os.path.exists(campair_dir):
                os.makedirs(campair_dir)
            if not os.path.exists(cam1_dir):
                os.makedirs(cam1_dir)
            if not os.path.exists(cam2_dir):
                os.makedirs(cam2_dir)

            for img_no in range(f[f[data_type][0][campair]].shape[0]):
                if img_no < 5:
                    cam_dir = 'cam1'
                else:
                    cam_dir = 'cam2'
                for person_id in range(f[f[data_type][0][campair]].shape[1]):
                    img = np.array(f[f[f[data_type][0][campair]][img_no][person_id]])
                    if img.shape[0] !=2:
                        img = np.transpose(img, (2,1,0))
                        im = Image.fromarray(img)
                        im.save(os.path.join(campair_dir, cam_dir, "%d-%d.jpg"%(person_id+1,img_no+1)))

###########################
# ReID Dataset Downloader#
###########################

def PersonReID_Dataset_Downloader(save_dir, dataset_name):
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    save_dir_exist = os.path.join(save_dir , dataset_name)

    if not os.path.exists(save_dir_exist):
        temp_dir = os.path.join(save_dir , 'temp')

        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)

        destination = os.path.join(temp_dir , dataset_name)

        id = dataset[dataset_name]

        print("Downloading %s" % dataset_name)
        gdrive_downloader(destination, id)

        zip_ref = zipfile.ZipFile(destination)
        print("Extracting %s" % dataset_name)
        zip_ref.extractall(save_dir)
        zip_ref.close()
        shutil.rmtree(temp_dir)
        print("Done")
        if dataset_name == 'CUHK03':
            print('Converting cuhk03.mat into images')
            cuhk03_to_image(os.path.join(save_dir,'CUHK03'))
            print('Done')
    else:
        print("Dataset Check Success: %s exists!" %dataset_name)

#For United Testing and External Use
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Dataset Name and Dataset Directory')
    parser.add_argument(dest="save_dir", action="store", default="~/Datasets/",help="")
    parser.add_argument(dest="dataset_name", action="store", type=str,help="")
    args = parser.parse_args() 
    PersonReID_Dataset_Downloader(args.save_dir,args.dataset_name)
