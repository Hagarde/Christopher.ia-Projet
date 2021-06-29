 from params import * 
 from keras.preprocessing.image import smart_resize
 import numpy as np
 
 def load_video(camera,image_size) : 
    ret = True 
    i=0
    out = np.zeros((NMBR_FRAME_PER_VIDEO,image_size[0],image_size[1],3))

    while ret and i<NMBR_FRAME_PER_VIDEO: 
        ret, frame = camera.read()
        frame = smart_resize(np.asarray(frame),image_size)   
        out[i,:,:,:]  = frame        
        i = i + 1
       
    return out