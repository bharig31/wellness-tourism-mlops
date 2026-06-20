"""
Hosting Script for Hugging Face Space Deployment
This script pushes all deployment files to a Hugging Face Space.
"""

import os
import sys
from huggingface_hub import HfApi

# Get your Hugging Face username from environment variable or command line argument
HF_USERNAME = os.getenv('HF_USERNAME') or (sys.argv[1] if len(sys.argv) > 1 else 'your-username-here')

# Configuration
SPACE_NAME = f"{HF_USERNAME}/wellness-tourism-predictor"
SPACE_TYPE = "docker"  # For Streamlit, use docker with Streamlit in Dockerfile

def deploy_to_space():
    """
    Deploy the Streamlit application to Hugging Face Space.
    """
    if HF_USERNAME == 'your-username-here':
        print("ERROR: Please set HF_USERNAME environment variable or pass username as argument")
        print("Usage: python deploy_to_huggingface.py your_huggingface_username")
        return
    
    # Login to Hugging Face
    from huggingface_hub import login
    token = os.getenv('HF_TOKEN')
    if not token:
        print("ERROR: HF_TOKEN environment variable not set")
        print("Set it with: set HF_TOKEN=your_huggingface_token")
        return
    login(token=token)
    
    # Initialize Hugging Face API
    api = HfApi()
    
    print(f"Deploying to Hugging Face Space: {SPACE_NAME}")
    
    # Create or update the space
    try:
        api.create_repo(
            repo_id=SPACE_NAME,
            repo_type="space",
            space_sdk=SPACE_TYPE,
            private=False
        )
        print(f"Created new space: {SPACE_NAME}")
    except Exception as e:
        print(f"Space already exists or error creating: {e}")
        print(f"Proceeding with upload to existing space: {SPACE_NAME}")
    
    # Files to upload
    files_to_upload = [
        "app.py",
        "utils.py",
        "requirements-deployment.txt",
        "Dockerfile"
    ]
    
    # Upload each file
    for file in files_to_upload:
        if os.path.exists(file):
            print(f"Uploading {file}...")
            api.upload_file(
                path_or_fileobj=file,
                path_in_repo=file,
                repo_id=SPACE_NAME,
                repo_type="space"
            )
            print(f"✓ {file} uploaded successfully")
        else:
            print(f"✗ {file} not found, skipping")
    
    # Create README for the space
    readme_content = """---
title: Wellness Tourism Package Predictor
emoji: ✈️
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: 1.28.0
app_file: app.py
pinned: false
---

# Wellness Tourism Package Predictor

This Streamlit application predicts whether a customer is likely to purchase the Wellness Tourism Package based on their demographic information, travel history, and interests.

## Features

- Interactive customer profile input
- Real-time prediction using trained ML model
- Probability breakdown visualization
- Model information and feature details

## Model

The model was trained using ensemble methods (Random Forest, XGBoost, Gradient Boosting) on customer data from "Visit with Us" travel company.

## How to Use

1. Fill in the customer information in the sidebar
2. Click "Predict Purchase Probability"
3. View the prediction results and confidence level
"""
    
    print("Uploading README.md...")
    api.upload_file(
        path_or_fileobj=readme_content.encode('utf-8'),
        path_in_repo="README.md",
        repo_id=SPACE_NAME,
        repo_type="space"
    )
    print("✓ README.md uploaded successfully")
    
    print("\n" + "="*50)
    print("Deployment completed successfully!")
    print(f"Space URL: https://huggingface.co/spaces/{SPACE_NAME}")
    print("="*50)

if __name__ == "__main__":
    deploy_to_space()
