import tensorflow as tf
import numpy as np
import os
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from matplotlib.pyplot import imread as mpl_imread
from skimage.transform import resize
from tensorflow.keras.losses import MeanSquaredError
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox

np.random.seed(678)
tf.random.set_seed(5678)

class ConLayerLeft(tf.keras.layers.Layer):
    def __init__(self, kernel_size, in_channels, out_channels):
        super(ConLayerLeft, self).__init__()
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
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
                       int(current_shape_size[1]),
                       int(current_shape_size[2]),
                       self.out_channels]
        
        layer = tf.nn.conv2d_transpose(inputs, self.w, output_shape=output_shape,
                                     strides=[1, 1, 1, 1], padding='SAME')
        layerA = tf.nn.relu(layer)
        return layerA

class RetinalCNN(tf.keras.Model):
    def __init__(self, layers):
        super(RetinalCNN, self).__init__()
        self.layer_list = layers
        
    def call(self, inputs):
        x = inputs
        for layer in self.layer_list:
            x = layer(x)
        return x

class RetinalCNNGui:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("RetinalCNN Settings")
        self.root.geometry("600x700")
        
        # Training parameters
        self.num_epochs = tk.IntVar(value=100)
        self.learning_rate = tk.DoubleVar(value=0.0001)
        self.batch_size = tk.IntVar(value=2)
        self.is_training = False
        
        # Data loading and preprocessing
        self.load_data()
        self.create_model()
        self.create_widgets()
    
    def load_data(self):
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

        self.train_images = np.zeros(shape=(128, 256, 256, 1))
        self.train_labels = np.zeros(shape=(128, 256, 256, 1))

        for file_index in range(len(train_data)):
            self.train_images[file_index, :, :] = np.expand_dims(
                resize(mpl_imread(train_data[file_index], as_gray=True), (256, 256)), axis=2
            )
            self.train_labels[file_index, :, :] = np.expand_dims(
                resize(mpl_imread(train_data_gt[file_index], as_gray=True), (256, 256)), axis=2
            )

        self.train_images = (self.train_images - self.train_images.min()) / (self.train_images.max() - self.train_images.min() + 1e-8)
        self.train_labels = (self.train_labels - self.train_labels.min()) / (self.train_labels.max() - self.train_labels.min() + 1e-8)

    def create_model(self):
        current_channels = 1
        self.layers = []
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
            if i == 25:
                self.layers.append(ConLayerRight(kernel_size, current_channels, out_channels))
            else:
                self.layers.append(ConLayerLeft(kernel_size, current_channels, out_channels))
            current_channels = out_channels

    def create_widgets(self):
        # Settings Frame
        settings_frame = ttk.LabelFrame(self.root, text="Training Settings", padding=10)
        settings_frame.pack(fill="x", padx=10, pady=5)

        # Training parameters
        ttk.Label(settings_frame, text="Number of Epochs:").grid(row=0, column=0, padx=5, pady=5)
        ttk.Entry(settings_frame, textvariable=self.num_epochs).grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Label(settings_frame, text="Learning Rate:").grid(row=1, column=0, padx=5, pady=5)
        ttk.Entry(settings_frame, textvariable=self.learning_rate).grid(row=1, column=1, padx=5, pady=5)
        
        ttk.Label(settings_frame, text="Batch Size:").grid(row=2, column=0, padx=5, pady=5)
        ttk.Entry(settings_frame, textvariable=self.batch_size).grid(row=2, column=1, padx=5, pady=5)

        # Progress Frame
        progress_frame = ttk.LabelFrame(self.root, text="Training Progress", padding=10)
        progress_frame.pack(fill="x", padx=10, pady=5)

        self.progress = ttk.Progressbar(progress_frame, length=400, mode='determinate')
        self.progress.pack(pady=10)
        
        self.status_label = ttk.Label(progress_frame, text="Ready to start training")
        self.status_label.pack(pady=5)

        # Control Buttons
        button_frame = ttk.Frame(self.root)
        button_frame.pack(pady=10)
        
        ttk.Button(button_frame, text="Start Training", command=self.start_training).pack(side="left", padx=5)
        ttk.Button(button_frame, text="Stop", command=self.stop_training).pack(side="left", padx=5)
        
        # Results Frame
        self.results_frame = ttk.LabelFrame(self.root, text="Training Results", padding=10)
        self.results_frame.pack(fill="both", expand=True, padx=10, pady=5)

    def start_training(self):
        if not self.is_training:
            self.is_training = True
            self.progress['value'] = 0
            
            try:
                model = RetinalCNN(self.layers)
                model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate.get()),
                            loss='mse',
                            metrics=['mae'])

                epochs = self.num_epochs.get()
                batch_size = self.batch_size.get()
                progress_step = 100.0 / epochs

                for iter in range(epochs):
                    if not self.is_training:
                        break

                    for current_batch_index in range(0, len(self.train_images), batch_size):
                        current_batch = self.train_images[current_batch_index:current_batch_index + batch_size]
                        current_label = self.train_labels[current_batch_index:current_batch_index + batch_size]
                        
                        loss = model.train_on_batch(current_batch, current_label)
                        self.status_label.config(text=f'Epoch: {iter+1}/{epochs}, Loss: {loss[0]:.6f}')
                        self.root.update()

                    self.progress['value'] += progress_step
                    self.root.update_idletasks()

                    if iter % 2 == 0:
                        self.display_results(model, iter)

                if self.is_training:
                    messagebox.showinfo("Success", "Training completed successfully!")
                
            except Exception as e:
                messagebox.showerror("Error", f"Training failed: {str(e)}")
            
            finally:
                self.is_training = False

    def stop_training(self):
        self.is_training = False
        self.status_label.config(text="Training stopped by user")

    def display_results(self, model, epoch):
        test_example = self.train_images[:2]
        predictions = model(test_example)
        
        # Display results logic here
        # You can add visualization code using matplotlib
        pass

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    gui = RetinalCNNGui()
    gui.run()
