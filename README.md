# Dhvani_Hackathon



## Question_2 :- Simulate and visualize a 3D dynamical system (Lorenz attractor) that models a “bee’s” path in space.

## Question_3 :- Vehicle Detection & Classification (Hackathon)
Goal. Detect vehicles in road scenes and classify them into car, rickshaw, bus, motorbike, returning bounding boxes + class labels.

Dataset.
 - Train: 3003 images + 3003 VOC XMLs (one XML per image).
 - Test: 500 images (no labels provided).
 - Labels cleaned by canonicalizing synonyms (e.g., suv/minivan/policecar → car; auto rickshaw/three wheelers → rickshaw; scooter → motorbike).
 - Skipped bad XMLs and clipped boxes to image bounds.

Data prep.
 - Parsed VOC → converted to YOLO format (normalized x_c, y_c, w, h).
 - Multilabel-stratified 85/15 split for validation to preserve class co-occurrence.
 - Oversampled motorbike images to reduce class imbalance.

Model.
 - YOLOv8m (Ultralytics) pretrained on COCO → transfer learning.
 - Two-phase training (high-res 1024, auto-fallback 896):
     Phase-1: warmup with backbone frozen (10 epochs).
     Phase-2: unfrozen full fine-tune (target 100 epochs).
 - Checkpoints saved each epoch; epoch-59 selected as best for deployment.

Validation (on 15% split).
 - Representative epoch-59 metrics: Precision ≈ 0.773, Recall ≈ 0.736, mAP@50 ≈ 0.770, mAP@50–95 ≈ 0.546.
 - Curves produced: training losses, PR, Precision-Confidence, Recall-Confidence, F1-Confidence, confusion matrices.

Inference on test (500 images).
 - Used epoch-59 weights; TTA on, conf=0.30, NMS IoU=0.55 to favor precision and reduce duplicate boxes.
 - Saved annotated images, YOLO txt predictions, and a CSV (image, class_name, conf, xmin, ymin, xmax, ymax).

Exploratory data analysis (EDA).
 - Plots: objects per class, bbox width/height/area histograms, center heatmap (x,y), width×height hexbin, objects per image.
 - Insights: dataset is small-object heavy and objects concentrate near the road band (y≈0.55–0.65), supporting high-res training and careful NMS/conf tuning.

Deliverables.
 - Trained weights (epoch-59 .pt), class file (data.yaml), test predictions (images/txt/CSV), EDA figures, evaluation plots, and a pipeline flowchart.

Notes & next steps.
 - If time allowed: full retrain on train+val, class-specific thresholds (e.g., slightly higher for motorbike), and lightweight test-time ensembling.
 - Report clarifies: evaluation plots were generated during training (up to epoch 79), while final inference used epoch-59 (best mAP50-95).









Ask ChatGPT

