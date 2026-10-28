# TerraScope spatial workspace

Use Python 3.11 or 3.12. Create a dedicated environment to avoid conflicts with existing packages. On Windows Command Prompt, from this project folder:

```bat
py -3.11 -m venv .venv
.venv\Scripts\python -m pip install -r requirements-spatial.txt
.venv\Scripts\python -m streamlit run spatial_app.py
```

For macOS/Linux:

```sh
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements-spatial.txt
python -m streamlit run spatial_app.py
```

Open the address printed by Streamlit. On Windows, `launch_spatial.bat` launches the UI once dependencies are installed. The original desktop UI remains available with `python main.py gui`.

## Presentation walkthrough
1. Select an included sample in Scene analysis.
2. Inspect it using viewport zoom and center controls.
3. Click Analyze active scene. Explain the predicted class and probability distribution.
4. Adjust the review threshold to demonstrate uncertainty handling.
5. Upload several images and analyze the entire batch.
6. Export JSON or CSV from Mission history.
7. Open Model evidence for the original confusion matrix and training history.

The pretrained model, architecture, configuration, training code, and inference preprocessing are unchanged. No retraining is required or performed. The model classifies an entire scene into one of ten EuroSAT classes. It does not detect object boundaries, produce segmentation, or infer GPS coordinates. Zoom affects display only. Uploaded images are processed in temporary files and deleted afterward; session history is in memory. Supplied evaluation evidence records 63.93% overall accuracy. Use representative satellite images and explain limitations honestly.

The browser UI runs through Python and cannot run on GitHub Pages. GitHub hosts the downloadable source and model; run the application locally for the presentation.
