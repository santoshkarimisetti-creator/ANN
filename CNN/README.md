# CNN Projects

This folder contains image-based deep learning projects built with Convolutional Neural Networks (CNNs).

## Subfolders

### 1. Classification/
This subfolder contains the leaf disease classification project.

- Trains a CNN on plant leaf images
- Uses image folders as classes
- Validates the model on a held-out set
- Saves the trained model and tests it on new images
- Includes the notebook: PlantDisease_Classification.ipynb

### 2. object detection/
This subfolder contains the object detection project.

- Uses YOLOv8 for detecting objects in disaster response images
- Includes scripts and notebook files for detection workflows
- Uses pretrained weights and generated model outputs during training

## Typical Workflow

1. Place image datasets inside the relevant subfolder.
2. Open the notebook for the project you want to run.
3. Train the model.
4. Save the final model or weights.
5. Test on validation images or an external image.

## Tools Used

- Python
- TensorFlow / Keras
- OpenCV
- NumPy
- Matplotlib
- Scikit-learn
- Ultralytics YOLOv8

## Important Notes

- Large image datasets should not be committed to Git.
- Model files, trained weights, logs, and generated outputs are usually stored locally.
- The root project-level .gitignore is configured to ignore common CNN training artifacts.

## Projects Summary

- Plant disease image classification using CNN
- Disaster response object detection using YOLOv8
