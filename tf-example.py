import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from matplotlib.pyplot import imread
import datetime

# Parameters for the notebook 

# Amount of images to use from the data (t speed up things)
MAX_SAMPLES = 1000

# Image resolution
IMG_SIZE = 244

# Batch size
BATCH_SIZE = 16

physical_devices = tf.config.list_physical_devices("GPU")
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

labels = pd.read_csv('resources/dogs/labels.csv')
labels.head()

base_path = "resources/dogs/train/"
fmt = ".jpg"
file_names = [f"{base_path}{filename}{fmt}" for filename in labels.id]

breeds = labels.breed.unique()

# Same as above for all labels, then we convert it into integers for our Model
final_labels = np.array([label == breeds for label in labels.breed])
final_labels = final_labels.astype(int)

# Creating our own validation set

# First we give our data proper names
x = file_names
y = final_labels

# Splitting training and validation
x_train, x_val, y_train, y_val = train_test_split(x[:MAX_SAMPLES], y[:MAX_SAMPLES], test_size=0.2)

"""
Read image from file and prepare it for the model
"""
def process_img(path, size = IMG_SIZE):
    # Read raw image
    image = tf.io.read_file(path)
    # Decode jpg with 3 channels (RGB)
    image = tf.image.decode_jpeg(image, channels = 3)
    # Normalize values so they are between 0-1
    image = tf.image.convert_image_dtype(image, tf.float32)
    # Resize image to our specified size
    image = tf.image.resize(image, size=[size, size])
    
    return image
    
# Turning the data into batches

"""
Returns a tuple with image and label as tensors
"""
def get_image_label(path, label):
    return process_img(path), label
    
    
    
def get_data_batches(x, y=None, batch_size=BATCH_SIZE, test_data=False, valid_data=False):
    if test_data:
        print("Creating test batches...")
        # Make a dataset from the filenames
        data = tf.data.Dataset.from_tensor_slices(tf.constant(x))
        # For each image, preprocess it using our function
        data_batch = data.map(process_img).batch(batch_size)
    else:
        # Make a dataset from the filenames
        data = tf.data.Dataset.from_tensor_slices((tf.constant(x), # filepaths
                                                   tf.constant(y))) # labels
        
        if not valid_data:
            print("Creating shuffled Training bathces...")
            # shuffling data, it is faster to shuffle filenames
            data = data.shuffle(buffer_size=len(x))
        else:
            print("Creating validation batches...")
            
        # For each image, preprocess it using our function
        data_batch = data.map(get_image_label).batch(batch_size)
    return data_batch
    
train_data = get_data_batches(x_train, y_train)
val_data = get_data_batches(x_val, y_val, valid_data=True)



# Model definitions
# Some number of 244x244 3-channel (RGB) images
INPUT = [None, IMG_SIZE, IMG_SIZE, 3]

# 120 elements vectors, one for each breed
OUTPUT = 120

# Model URL from TF HUB
MODEL_URL = "https://tfhub.dev/google/imagenet/inception_v1/classification/4"

def create_model(input_shape=INPUT, outputs=OUTPUT, model_url=MODEL_URL):
    model = tf.keras.Sequential([
        hub.KerasLayer(model_url),
        tf.keras.layers.Dense(units=outputs,
                              activation="softmax")
    ])

    model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
                  optimizer=tf.keras.optimizers.Adam(),
                  metrics = ['accuracy'])

    model.build(input_shape)
    
    return model

model = create_model()
model.summary()

# Callback to generate tensorboard logs
def create_tensorboard_callback(log_dir="resources/dogs/logs"):
    log_path = f"{log_dir}/{datetime.datetime.now().strftime('%d%m%Y-%H%M%S')}"
    
    return tf.keras.callbacks.TensorBoard(log_path)
    
# Callback for early stopping
def create_early_stopping_callback(**kwargs):
    return tf.keras.callbacks.EarlyStopping(**kwargs)
    
NUM_EPOCHS = 100

def train_model(model, train_data, val_data, callbacks = [], epochs=NUM_EPOCHS):
    model.fit(train_data,
              epochs=epochs,
              validation_data=val_data,
              validation_freq=1,
              callbacks=callbacks)
    return model
    
trained_model = train_model(create_model(), 
                            train_data, 
                            val_data, 
                            callbacks = [create_tensorboard_callback(), 
                                         create_early_stopping_callback(patience=3)])
