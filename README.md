# CS4372_Hwk3_Shyam_and_Rudy
Overview

Classify 10 monkey species using MobileNetV2 (ImageNet) with a custom head and light fine-tuning. Trains on a Drive-hosted dataset; produces history plots, a 25-image prediction grid, a tuning table, and test metrics (incl. Macro-F1).

Dataset([https://www.kaggle.com/code/paultimothymooney/identify-monkey-species-from-image/notebook]([URL](URL)))
    •    Source: 10-monkey-species (images only).
    
    •    Prep: Merge provided training+validation, then split 70/15/15 into train/val/test.
    
    •    Classes: n0…n9 (optionally map to names if a labels file exists).

  
Environment

    •    Python 3.12, TensorFlow 2.20, Keras 3 (Colab default).
    
    •    Key libs: tensorflow, scikit-learn, matplotlib, pandas, numpy.

    
How to Run (Colab)

    1    Upload dataset to Drive and set GDRIVE_SOURCE_FOLDER.
    
    2    Run cells: mount Drive → fetch/prepare (Sections 3–5) → pipelines (6) → model (7).
    
    3    Train: Feature extraction (9) → fine-tune top ~20% (10).
    
    4    Evaluate & export: plots (11), test metrics + Macro-F1 + 25-grid (12), tuning table (13), artifacts (14).
    
Model & Training

    •    Backbone: MobileNetV2(include_top=False, weights="imagenet", name="backbone").
    
    •    Head: GAP → Dropout(0.2) → Dense(num_classes, softmax).
    
    •    Preprocessing: mobilenet_v2.preprocess_input inside the model; images resized to 224×224×3.
    
    •    Optimizers: Adam (FE 1e-3, FT 1e-4).
    
    •    Callbacks: Checkpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger.

    
Evaluation & Deliverables

    •    Held-out test: loss, accuracy, Macro-F1 (scikit-learn) + optional confusion matrix.
    
    •    25-image prediction grid (with true/pred labels) and CSV.
    
    •    Training history plots (FE & FT).
    
    •    Tuning table with train/val/test accuracy and test Macro-F1.
    
