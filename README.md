# Office Items YOLOv8m Object Detection

Object detection project for identifying office items using YOLOv8m. I made this project during Easter break when I was back in my bedroom at home. The only objects readily available to detect around me were office items from my desk, so I decided to finetune YOLOv8 on this dataset as a way to learn how these models work and how to use them in practice.

## Dataset

https://universe.roboflow.com/workspace1-1gbmx/office-items-plugt/dataset/26

## Model Training

THe model was trained on Google Colab using their L4 GPU. Hyperparameters:

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

These metrics indicate the model performs reasonably well at detecting office items, though there's definitely potential for further improvement with more data or tuning.

**Training Results (Metrics Curves):**
<img src="./runs/train/OfficeItems_yolov8m/results.png" alt="Training Results" width="500"/>

**Confusion Matrix (Normalized):**
<img src="./runs/train/OfficeItems_yolov8m/confusion_matrix_normalized.png" alt="Confusion Matrix" width="500"/>

The other visualization files are available in the `runs/train/OfficeItems_yolov8m/` folder


## Using the Model

### Webcam Detection

To use the webcam detection:
1. Ensure Python is installed on your system
2. Run the webcam script: `python webcam_detection.py`
3. Position office items in view of your webcam for real-time detection

## Notes

Make sure to put your phone on airplane mode when running inference.

### Tips for Better Performance

- A computer with a powerfull GPU for better inference 
- Adequate lighting improves detection accuracy
- Minimize background clutter for better results
- Position items so they are clearly visible and not obscured
- Adjust camera angle as needed for optimal detection




It was a cool task that helped me understand how object detection models work in real-world scenarios, even with a simple dataset of everyday office items from my desk. 

