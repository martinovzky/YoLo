# Office Items YOLOv8m Object Detection

This is my object detection project for identifying office items using YOLOv8m. I created this model for my computer vision class project during Easter break when I was back in my bedroom at home. The only objects readily available to detect around me were office items from my desk, so I decided to finetune YOLOv8 on this dataset as a way to learn how these models work and how to use them in practice.

## Dataset

https://universe.roboflow.com/workspace1-1gbmx/office-items-plugt/dataset/26

## Model Training

I trained this model on Google Colab using their L4 GPU. The training process used the following parameters:

- **Model**: YOLOv8m (medium size)
- **Epochs**: 50
- **Batch Size**: 32
- **Image Size**: 640x640
- **Training Time**: ~2 hours (7,045 seconds)

### Key Results

By the end of training (epoch 50), the model achieved:
- mAP50-95: 0.537 (mean Average Precision)
- Precision: 0.725
- Recall: 0.682

These metrics indicate the model performs well at detecting office items.

### Training Visualization

The training process generated visualizations to monitor performance:
- Confusion matrix showing how well the model distinguishes between different office items
- Precision-recall curves displaying the model's accuracy
- Training batch images showing what the model "sees" during training

All these visualization files are available in the `runs/train/OfficeItems_yolov8m/` folder.

## Using the Model

### Webcam Detection

To use the webcam detection:
1. Ensure Python is installed on your system
2. Run the webcam script: `python webcam_detection.py`
3. Position office items in view of your webcam for real-time detection

## Notes

Make sure to put your phone on airplane mode when running inference.

### Tips for Better Performance

- Adequate lighting improves detection accuracy
- Minimize background clutter for better results
- Position items so they are clearly visible and not obscured
- Adjust camera angle as needed for optimal detection

## What I Learned

This project gave me hands-on experience with:
- Data preparation and augmentation techniques
- Hyperparameter tuning
- The importance of quality training data
- Practical applications of computer vision models

It was a cool task that helped me understand how object detection models work in real-world scenarios, even with a simple dataset of everyday office items from my desk.

