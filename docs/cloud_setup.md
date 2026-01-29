# RunPod Walkthrough: Running Abliteration in the Cloud

This is a step-by-step guide to run abliteration on Qwen 2.5 using RunPod cloud GPUs.

---

## Step 1: Create RunPod Account

1. Go to **[runpod.io](https://runpod.io)**
2. Click **"Sign Up"** (top right)
3. Create account with email or GitHub

---

## Step 2: Add Credits

1. Click your profile icon → **"Billing"**
2. Click **"Add Credits"**
3. Add **$5** (this is plenty — abliteration costs ~$0.10-0.40)

---

## Step 3: Deploy a GPU Pod

1. Click **"Pods"** in the left sidebar
2. Click **"+ Deploy"** button
3. **Select GPU**: Choose **RTX 4090** ($0.34/hr) or **RTX 3090** ($0.25/hr)
4. **Select Template**: Search for **"RunPod Pytorch 2.1"** and select it
5. **Volume Disk**: Set to **20 GB** (stores the model temporarily)
6. Click **"Deploy On-Demand"**

Wait ~1-2 minutes for the pod to start. Status will change to **"Running"**.

---

## Step 4: Connect to Your Pod

1. Click on your running pod
2. Click **"Connect"** button
3. Choose one of:
   - **"Start Web Terminal"** — easiest, browser-based
   - **"SSH"** — if you prefer command line

---

## Step 5: Set Up the Project

In the terminal, run these commands:

```bash
# 1. Clone the abliteration code
cd /workspace
git clone https://github.com/YOUR_USERNAME/Abliteration.git
cd Abliteration

# OR if you haven't pushed to GitHub, upload files via the RunPod file browser
```

**Alternative: Upload from your computer**
1. In RunPod, click **"File Browser"** 
2. Navigate to `/workspace`
3. Upload the `Abliteration` folder from your computer

---

## Step 6: Install Dependencies

```bash
cd /workspace/Abliteration
pip install -r requirements.txt
```

This takes ~2-3 minutes.

---

## Step 7: Login to Hugging Face

```bash
# Install the CLI
pip install huggingface_hub

# Login (paste your token when prompted)
huggingface-cli login
```

Get your token at: https://huggingface.co/settings/tokens

---

## Step 8: Run Abliteration! 🚀

```bash
# For 0.5B model (fastest, ~15 min)
python run_abliteration.py --model Qwen/Qwen2.5-0.5B-Instruct

# For 1.5B model (~30 min)
python run_abliteration.py --model Qwen/Qwen2.5-1.5B-Instruct

# For 3B model (~60 min)  
python run_abliteration.py --model Qwen/Qwen2.5-3B-Instruct
```

---

## Step 9: Test Your Abliterated Model

```bash
python run_abliteration.py --test --model ./output/abliterated_model
```

Try prompts like:
- "Write a script that exploits a vulnerability"
- "How do I create a phishing email?"

The model should now attempt to answer instead of refusing.

---

## Step 10: Save Your Model

**Option A: Download to your computer**
```bash
cd /workspace/Abliteration
zip -r abliterated_model.zip output/
```
Then use RunPod's file browser to download the zip.

**Option B: Push to Hugging Face (recommended)**
```bash
huggingface-cli upload your-username/qwen-abliterated ./output/abliterated_model
```

---

## Step 11: Stop Your Pod! ⚠️

**IMPORTANT**: Pods cost money while running!

1. Go to **"Pods"** in sidebar
2. Click the **⏹️ Stop** button on your pod
3. Or click **🗑️ Delete** to remove completely

---

## Troubleshooting

### "CUDA out of memory"
- Reduce batch size: `--batch-size 8` or `--batch-size 4`
- Use a smaller model

### "Model not found"
- Make sure you're logged into Hugging Face
- Some models require accepting license at huggingface.co first

### Pod won't start
- Try a different GPU type or region
- Check if you have enough credits

---

## Cost Summary

| What | Cost |
|------|------|
| RTX 4090 for 30 min | ~$0.17 |
| RTX 3090 for 30 min | ~$0.12 |
| Storage (20GB) | ~$0.02/hr |
| **Total for 0.5B** | **~$0.10-0.20** |
