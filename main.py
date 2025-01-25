import tensorflow as tf
import numpy as np
import os
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from matplotlib.pyplot import imread as mpl_imread
from skimage.transform import resize
from tensorflow.keras.losses import MeanSquaredError

np.random.seed(678)
tf.random.set_seed(5678)

class ConLayerLeft(tf.keras.layers.Layer):
    def __init__(self, kernel_size, in_channels, out_channels):
        super(ConLayerLeft, self).__init__()
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        # Change the shape order to match TensorFlow's conv2d expectations
        self.w = self.add_weight(shape=(kernel_size, kernel_size, in_channels, out_channels),
                                initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05))


    def call(self, inputs, stride=1):
        current_shape_size = inputs.shape
        output_shape = [tf.shape(inputs)[0],
                       int(current_shape_size[1]),
                       int(current_shape_size[2]),
                       self.out_channels]

        layer = tf.nn.conv2d(inputs, self.w, strides=[1, stride, stride, 1], padding='SAME')
        layerA = tf.nn.relu(layer)
        return layerA


class ConLayerRight(tf.keras.layers.Layer):
    def __init__(self, kernel_size, in_channels, out_channels):
        super(ConLayerRight, self).__init__()
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.w = self.add_weight(shape=(kernel_size, kernel_size, out_channels, in_channels),
                                initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05))

    def call(self, inputs, stride=1, output=1):
        current_shape_size = inputs.shape
        output_shape = [tf.shape(inputs)[0],
                       int(current_shape_size[1]),  # Remove the *2
                       int(current_shape_size[2]),  # Remove the *2
                       self.out_channels]
        
        layer = tf.nn.conv2d_transpose(inputs, self.w, output_shape=output_shape,
                                     strides=[1, 1, 1, 1], padding='SAME')  # Change strides to [1,1,1,1]
        layerA = tf.nn.relu(layer)
        return layerA


data_location = "./DRIVE/training/images/"
train_data = []
for dirName, subdirList, fileList in sorted(os.walk(data_location)):
    for filename in fileList:
        if ".tif" in filename.lower():
            train_data.append(os.path.join(dirName, filename))

data_location = "./DRIVE/training/1st_manual/"
train_data_gt = []
for dirName, subdirList, fileList in sorted(os.walk(data_location)):
    for filename in fileList:
        if ".tif" in filename.lower():
            train_data_gt.append(os.path.join(dirName, filename))

train_images = np.zeros(shape=(128, 256, 256, 1))
train_labels = np.zeros(shape=(128, 256, 256, 1))

for file_index in range(len(train_data)):
    train_images[file_index, :, :] = np.expand_dims(
        resize(mpl_imread(train_data[file_index], as_gray=True), (256, 256)), axis=2
    )
    train_labels[file_index, :, :] = np.expand_dims(
        resize(mpl_imread(train_data_gt[file_index], as_gray=True), (256, 256)), axis=2
    )

train_images = (train_images - train_images.min()) / (train_images.max() - train_images.min() + 1e-8)
train_labels = (train_labels - train_labels.min()) / (train_labels.max() - train_labels.min() + 1e-8)

num_epochs = 100
init_lr = 0.0001
batch_size = 2

# Replace with a proper Keras Model definition
class RetinalCNN(tf.keras.Model):
    def __init__(self, layers):
        super(RetinalCNN, self).__init__()
        self.layer_list = layers
        
    def call(self, inputs):
        x = inputs
        for layer in self.layer_list:
            x = layer(x)
        return x

# Initialize the number of channels
current_channels = 1  # Initial input channel

# Create a list to store the layers
layers = []

# Loop to create Convolutional layers with the specified configurations
layer_configs = [
    (24, 3), (24, 3), (24, 3),
    (48, 3), (48, 3), (24, 3),
    (48, 3), (48, 3), (24, 3),
    (48, 3), (48, 3), (24, 3),
    (48, 3), (48, 3), (96, 3),
    (72, 3), (36, 3), (18, 3),
    (24, 3), (12, 3), (6, 3),
    (12, 3), (6, 3), (3, 3),
    (6, 3), (3, 3), (3, 3),
    (1, 3)
]

for i, (out_channels, kernel_size) in enumerate(layer_configs):
    if i == 25:  # Special case for the transition layer l6_1
        layers.append(ConLayerRight(kernel_size, current_channels, out_channels))
    else:
        layers.append(ConLayerLeft(kernel_size, current_channels, out_channels))
    
    current_channels = out_channels  # Update the current number of channels


# Create model instance
model = RetinalCNN(layers)

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=init_lr),
             loss='mse',
             metrics=['mae'])

# Training loop
for iter in range(num_epochs):
    for current_batch_index in range(0, len(train_images), batch_size):
        current_batch = train_images[current_batch_index:current_batch_index + batch_size]
        current_label = train_labels[current_batch_index:current_batch_index + batch_size]
        
        # Train on batch
        loss = model.train_on_batch(current_batch, current_label)
        print(' Iter: ', iter, " Cost:  %.32f" % loss[0], end='\r')


# Construct the model
model = tf.keras.Sequential(layers)

# Instantiate the MeanSquaredError loss function
mse_loss = MeanSquaredError()


# Define the optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=init_lr)

# Use GradientTape to compute gradients and optimize
for iter in range(num_epochs):
    for current_batch_index in range(0, len(train_images), batch_size):
        current_batch = train_images[current_batch_index:current_batch_index + batch_size, :, :, :]
        current_label = train_labels[current_batch_index:current_batch_index + batch_size, :, :, :]
        
        with tf.GradientTape() as tape:
            predictions = model(current_batch)  # Forward pass through your model
            loss_value = tf.reduce_mean(tf.square(predictions - current_label))
        
        gradients = tape.gradient(loss_value, model.trainable_variables)  # Compute gradients
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))  # Apply gradients to update variables
        
        print(' Iter: ', iter, " Cost:  %.32f" % loss_value.numpy(), end='\r')
    
    print('\n-----------------------')
    train_images, train_labels = shuffle(train_images, train_labels)
    
    if iter % 2 == 0:
        test_example = train_images[:2, :, :, :]
        test_example_gt = train_labels[:2, :, :, :]
        sess_results = model(test_example)

        for idx in range(2):
            plt.figure()
            plt.imshow(np.squeeze(test_example[idx, :, :]), cmap='gray')
            plt.axis('off')
            plt.title(f'Original Image {idx}')
            plt.savefig(f'train_change/{iter}a_Original_Image_{idx}.png')

            plt.figure()
            plt.imshow(np.squeeze(test_example_gt[idx, :, :]), cmap='gray')
            plt.axis('off')
            plt.title(f'Ground Truth Mask {idx}')
            plt.savefig(f'train_change/{iter}b_Original_Mask_{idx}.png')

            plt.figure()
            plt.imshow(np.squeeze(sess_results[idx, :, :]), cmap='gray')
            plt.axis('off')
            plt.title(f'Generated Mask {idx}')
            plt.savefig(f'train_change/{iter}c_Generated_Mask_{idx}.png')

            plt.figure()
            plt.imshow(np.multiply(np.squeeze(test_example[idx, :, :]), np.squeeze(test_example_gt[idx, :, :])), cmap='gray')
            plt.axis('off')
            plt.title(f'Ground Truth Overlay {idx}')
            plt.savefig(f'train_change/{iter}d_Original_Image_Overlay_{idx}.png')

            plt.figure()
            plt.imshow(np.multiply(np.squeeze(test_example[idx, :, :]), np.squeeze(sess_results[idx, :, :])), cmap='gray')
            plt.axis('off')
            plt.title(f'Generated Overlay {idx}')
            plt.savefig(f'train_change/{iter}e_Generated_Image_Overlay_{idx}.png')

            plt.close('all')


    for data_index in range(0,len(train_images),batch_size):
        current_batch = train_images[current_batch_index:current_batch_index+batch_size,:,:,:]
        current_label = train_labels[current_batch_index:current_batch_index+batch_size,:,:,:]
        sess_results = sess.run(layer10,feed_dict={x:current_batch})

        plt.figure()
        plt.imshow(np.squeeze(current_batch[0,:,:,:]),cmap='gray')
        plt.axis('off')
        plt.title(str(data_index)+"a_Original Image")
        plt.savefig('gif/'+str(data_index)+"a_Original_Image.png")

        plt.figure()
        plt.imshow(np.squeeze(current_label[0,:,:,:]),cmap='gray')
        plt.axis('off')
        plt.title(str(data_index)+"b_Original Mask")
        plt.savefig('gif/'+str(data_index)+"b_Original_Mask.png")
        
        plt.figure()
        plt.imshow(np.squeeze(sess_results[0,:,:,:]),cmap='gray')
        plt.axis('off')
        plt.title(str(data_index)+"c_Generated Mask")
        plt.savefig('gif/'+str(data_index)+"c_Generated_Mask.png")

        plt.figure()
        plt.imshow(np.multiply(np.squeeze(current_batch[0,:,:,:]),np.squeeze(current_label[0,:,:,:])),cmap='gray')
        plt.axis('off')
        plt.title(str(data_index)+"d_Original Image Overlay")
        plt.savefig('gif/'+str(data_index)+"d_Original_Image_Overlay.png")
       
        plt.figure()
        plt.imshow(np.multiply(np.squeeze(current_batch[0,:,:,:]),np.squeeze(sess_results[0,:,:,:])),cmap='gray')
        plt.axis('off')
        plt.title(str(data_index)+"e_Generated Image Overlay")
        plt.savefig('gif/'+str(data_index)+"e_Generated_Image_Overlay.png")

        plt.close('all')


# -- end code --
