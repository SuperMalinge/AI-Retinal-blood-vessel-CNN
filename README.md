# AI-Retinal-blood-vessel-CNN
This code implements a Convolutional Neural Network (CNN) for retinal blood vessel segmentation in medical images. Here are the key aspects:

Purpose: It processes retinal fundus images to automatically detect and segment blood vessels, which is valuable for diagnosing various eye conditions.

Main components:

Custom CNN architecture with left (ConLayerLeft) and right (ConLayerRight) convolutional layers
Training pipeline using the DRIVE dataset (Digital Retinal Images for Vessel Extraction)
Image preprocessing and normalization
Visualization of results including original images, ground truth masks, and generated segmentation masks

Training process:
Uses 128 training images of size 256x256 pixels
Runs for 100 epochs with a batch size of 2
Implements gradient-based optimization using Adam optimizer
Saves progress visualizations in the 'train_change' directory

Output:
Generates vessel segmentation masks
Creates comparison visualizations showing:
Original retinal images
Ground truth vessel masks
Generated vessel masks
Overlay visualizations of vessels on original images

This code is particularly useful in medical image analysis for automated detection of retinal vasculature, which can help in early diagnosis of conditions like diabetic retinopathy.
