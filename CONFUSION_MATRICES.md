# Confusion matrices in the original desktop GUI

Run `python main.py gui` with the original project's dependencies installed (`python -m pip install -r requirements.txt`). This update preserves the existing desktop widgets and adds a **Confusion Matrices** menu at the top. It does not change model weights, preprocessing, configuration, or training.

- **Individual image:** classify an image normally, then choose Individual image from the menu. Assign its known true class and click Show matrix. A single image contributes exactly one count; this is not a meaningful estimate of general model accuracy.
- **Batch:** run Batch Process normally, then choose Latest batch. Select rows and assign their known class, or import a CSV containing `path,true_label` columns. Use the full image paths shown in the table. Multiple rows may be selected together. CSV paths must be unique; class names must match config.json exactly.
- **Comparison:** use Compare Images normally, then choose Latest comparison. Supply known labels for both successful predictions. Failed predictions are excluded and counted.
- **Overall:** choose Overall dataset folder and select the root containing folders such as Forest, River, AnnualCrop, etc. The evaluator recursively reads these class folders, runs the original predictor, and calculates a fresh matrix. Unlabeled folders are not evaluated. Cancel discards the incomplete evaluation. No training occurs.

Rows represent true labels; columns represent predicted labels. Labels inferred from parent folders can be corrected for individual, batch, and comparison results. The matrix view includes evaluated/excluded counts, accuracy for the selected labeled images, CSV export, and a Matplotlib toolbar for PNG export.

For a meaningful overall evaluation, select a held-out test set. Evaluating training images is not a held-out accuracy measurement. The uploaded RAR has no data folder; select the dataset already on your computer. Existing saved training plots are not overwritten.
