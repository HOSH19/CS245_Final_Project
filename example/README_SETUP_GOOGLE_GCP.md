# Using Google GCP Credits for AgentSociety Challenge

This guide shows you how to use your $50 Google GCP credits with the enhanced agents.

## 🔑 Getting Your Google API Key

### Step 1: Access Google AI Studio
1. Go to [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Sign in with your Google account (the one with GCP credits)

### Step 2: Create API Key
1. Click **"Create API Key"**
2. Select your **GCP project** (where you have the $50 credits)
3. Copy the generated API key
4. Save it securely!

### Step 3: Enable Billing
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Navigate to **Billing** → **Link a billing account**
3. Make sure your project with the $50 credits is linked
4. The Gemini API will use these credits automatically

## 🚀 Quick Start

### Option A: Using .env File (Recommended - Most Secure)

```bash
# 1. Navigate to example directory
cd example

# 2. Create .env file from template
cp env_template.txt .env

# 3. Edit .env file with your actual API keys
Open .env in your editor and replace the placeholder values

# 4. Install python-dotenv (if not already installed)
pip install python-dotenv