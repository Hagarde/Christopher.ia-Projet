# General data parameters
DATASET_PATH = 'dataset-resized' # à changer si collab 
SHUFFLE_DATA = True

# Data generator parameters
TRAINING_BATCH_SIZE = 128
TRAINING_IMAGE_SIZE = (128,128)
VALIDATION_BATCH_SIZE = 128
VALIDATION_IMAGE_SIZE = (128,128)
TESTING_BATCH_SIZE = 128
TESTING_IMAGE_SIZE = (128,128)
NUMBER_OF_CHANNELS = 3
learning_rate = 10**-3
NMBR_OF_FRAME_PER_VIDEO = 500

# data du réseau

num_filters_1 = 64
num_filters_2 = 128
num_filters_3 = 128
num_filters_4 = 128
num_filters_5 = 128
num_filters_6 = 128
filter_size = 3
pool_size = (2,2)
epochs = 30
label_number = 6
dropout_rate = 0.25