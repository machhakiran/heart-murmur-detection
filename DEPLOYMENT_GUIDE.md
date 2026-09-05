# 🚀 Clean Step-by-Step Deployment Guide

Deploying the **AI-Powered Heart Murmur Detection System** under your personal accounts:
1. **Kaggle** (Dataset access)
2. **Google Colab** (GPU Model Training)
3. **Hugging Face Hub** (Model hosting & weights repository)
4. **GitHub** (Application code repository)
5. **Streamlit Community Cloud** (Live interactive web application)

---

## 📋 Accounts Checklist

Make sure you are signed into:
- [x] [Kaggle](https://www.kaggle.com)
- [x] [Google Colab](https://colab.research.google.com)
- [x] [Hugging Face](https://huggingface.co)
- [x] [GitHub](https://github.com)
- [x] [Streamlit Community Cloud](https://share.streamlit.io)

---

## Step 1: Get Your Kaggle API Key

1. Log into your account at [kaggle.com](https://www.kaggle.com).
2. Click on your profile picture in the top-right corner and select **Settings**.
3. Scroll down to the **API** section.
4. Click **Create New Token**.
5. A file named `kaggle.json` will automatically download to your computer. Keep this file ready.

---

## Step 2: Get Your Hugging Face Access Token

1. Log into your account at [huggingface.co](https://huggingface.co).
2. Click your profile avatar (top-right) ➔ **Settings** ➔ **Access Tokens** (or visit [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)).
3. Click **+ Create new token**.
4. Set:
   - **Token type**: `Write`
   - **Token name**: `heart-murmur-uploader`
5. Click **Create token** and copy the token (starts with `hf_...`). Keep it safe.

---

## Step 3: Train & Upload Model using Google Colab

1. Open the notebook in Google Colab with **one click**:
   👉 [**Open `train_lstm_heart_murmur.ipynb` in Google Colab**](https://colab.research.google.com/github/machhakiran/heart-murmur-detection/blob/main/colab/train_lstm_heart_murmur.ipynb)
   *(Make sure you are logged in as `machhakiran@gmail.com` in your browser).*
2. In Colab, switch to GPU runtime:
   - Click **Runtime ➔ Change runtime type**.
   - Select **T4 GPU** under Hardware accelerator and click **Save**.
3. Run **Step 1** (Installs packages: `librosa`, `torch`, `huggingface_hub`, etc.).
4. Run **Step 2**:
   - A file upload button will appear. Click **Choose Files** and select your `kaggle.json` file.
   - Colab will automatically download and extract the Kaggle dataset (`abdallahaboelkhair/heartbeat-sound`).
5. Run **Steps 3 through 8**:
   - Automatically pre-processes audio, extracts 60 acoustic features (MFCC, Spectral, Chroma, RMS).
   - Trains the BiLSTM model with attention pooling.
   - Generates the confusion matrix and clinical performance report.
6. Run **Step 9 (Export to Hugging Face)**:
   - Paste your **Hugging Face Write Token** when prompted.
   - Enter your model repo name when asked, for example:
     ```
     machhakiran/heart-murmur-bilstm
     ```
   - The script will automatically push `heart_murmur_lstm.pt`, `config.json`, and `README.md` to your Hugging Face Hub!
   - Note down your model repo name: `machhakiran/heart-murmur-bilstm`.

---

## Step 4: Your Project on GitHub

Your repository is already live on your GitHub account (`machhakiran`):
👉 **[https://github.com/machhakiran/heart-murmur-detection](https://github.com/machhakiran/heart-murmur-detection)**

If you make any new local edits later, simply run:
```bash
git add .
git commit -m "update: improvements"
git push origin main
```

---

## Step 5: Deploy on Streamlit Community Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io) in Chrome (sign in using your GitHub account `machhakiran` / `machhakiran@gmail.com`).
2. Click **New app** (or **Create app**).
3. Fill in the deployment form:
   - **Repository**: `machhakiran/heart-murmur-detection`
   - **Branch**: `main`
   - **Main file path**: `app.py`
   - **App URL**: choose your preferred subdomain (e.g., `cardio-murmur-ai.streamlit.app`)
4. Click **Advanced settings...**:
   - In the **Secrets** box, paste:
     ```toml
     HF_MODEL_REPO = "machhakiran/heart-murmur-bilstm"
     ```
5. Click **Deploy!**

---

## Step 6: Test & Verify Your Live App

1. Once deployed, your web application will open automatically at your custom Streamlit URL!
2. Test immediate features:
   - **Benchmark Samples**: Select "Normal Heartbeat", "Systolic Heart Murmur", or "Extrasystole" and click **Run Diagnostic Analysis**.
   - **Upload Custom Audio**: Upload any digital stethoscope `.wav` or `.mp3` recording.
   - **Inspect Visualizations**: Review the Phonocardiogram waveform, frequency Mel spectrogram, and confidence distribution.
   - **Model Verification**: Check the sidebar to verify your model is successfully loaded from your Hugging Face Hub repository (`HF Hub (your-username/heart-murmur-bilstm)`).
