Human Head Counter — Streamlit Community Cloud
A Streamlit app that accepts one JPG, JPEG, PNG, or WEBP photo, estimates the number of visible human heads using a YOLOv8 head detector, draws detections on the image, and allows the marked-up photo to be downloaded.
Files
- app.py — complete Streamlit application
- requirements.txt — Python dependencies
Run locally
Use Python 3.12 for a close match to Streamlit Community Cloud.
python -m venv .venv
Windows PowerShell:
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
The first run downloads the model weights from Hugging Face. An internet connection is required for that first load.
Deploy to Streamlit Community Cloud
1. Create a GitHub repository, for example human-head-counter.
2. Upload app.py and requirements.txt to the repository root and commit the files to the main branch.
3. Open https://share.streamlit.io/ and sign in with GitHub.
4. Choose Create app and select the human-head-counter repository, branch main, and main file path app.py.
5. In Advanced settings, choose Python 3.12 if a Python-version option is shown.
6. Click Deploy. The initial build downloads Python dependencies; the first model load downloads the weights.
7. After deployment, open the generated .streamlit.app URL and test the app with a few representative photos.
Notes and limitations
- The model was trained on railway-platform and event-entrance crowd images, not specifically on classroom and meeting photographs. Validate the results against manually counted photos before relying on it.
- Heads that are small, blurred, occluded, or outside the model's training domain may be missed. False detections are also possible.
- The slider changes the model's confidence threshold. Lower thresholds usually return more detections but can add false positives.
- The model card lists CC BY-SA 4.0 and states that the dataset and derived model have share-alike requirements. In addition, Ultralytics offers its YOLO software/models under AGPL-3.0 or an Enterprise License. Review both sets of terms and obtain qualified advice as needed before commercial or proprietary use.
- Uploaded photos are processed on the app server. This code does not intentionally save uploads to a database or permanent file, but it is not a substitute for your organization's privacy and consent requirements.
Model
AmineSam/irail-crowd-counting-yolov8n on Hugging Face: https://huggingface.co/AmineSam/irail-crowd-counting-yolov8n
