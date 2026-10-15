Human Head Counter
Files
- app.py: Streamlit application
- requirements.txt: pinned Python dependencies
Deploy to Streamlit Community Cloud
1. Upload both files to the root of a GitHub repository.
2. Create a Streamlit Community Cloud app using that repository.
3. Choose branch main and main file path app.py.
4. Choose Python 3.12 if available for the deployment.
5. Deploy the app.
Important
- This project uses a pretrained head-detection model hosted on Hugging Face.
- The first run downloads model weights and may take longer.
- CPU inference can be slow on large images.
- Model performance on classroom and meeting images must be checked against manually counted examples.
- Review the model's and Ultralytics' licenses before commercial use.
