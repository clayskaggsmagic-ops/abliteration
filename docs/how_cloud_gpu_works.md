# How Cloud GPU Computing Works

## The Mental Model

Think of RunPod as **renting a powerful computer in a data center** that you control remotely.

```
┌─────────────────┐         ┌─────────────────────────────────┐
│  YOUR COMPUTER  │  ───►   │     RUNPOD CLOUD SERVER         │
│  (MacBook)      │ browser │  ┌─────────────────────────────┐│
│                 │         │  │ • 24GB GPU (RTX 4090)       ││
│  • Write code   │         │  │ • 32GB RAM                  ││
│  • Browse web   │         │  │ • Linux OS                  ││
│  • Light tasks  │         │  │ • Your code runs HERE       ││
│                 │         │  └─────────────────────────────┘│
└─────────────────┘         └─────────────────────────────────┘
```

## Your Two Options for Workflow

### Option A: Code Locally → Upload → Run on Cloud
```
1. Write/edit code on your Mac (what you're doing now)
2. Upload files to RunPod (via file browser, git, or SCP)
3. Run the heavy computation on RunPod
4. Download results when done
```
**Pros:** Use your familiar editor, version control, etc.
**Cons:** Need to keep syncing files

### Option B: Code Directly on RunPod
```
1. Open Jupyter Lab on RunPod
2. Write code in their browser-based editor
3. Run everything there
4. Download results when done
```
**Pros:** No syncing needed
**Cons:** Less comfortable than your local editor

---

## What We're Doing Now

**You coded locally** → I created all the files in `/Users/clayskaggs/Developer/Abliteration/`

**Now we're uploading to cloud** → Recreating those files on RunPod via terminal commands

**Then it runs on the GPU** → The model downloads and processes on their fast hardware

**When done** → You download just the result (the abliterated model), not everything

---

## Why Use Cloud GPU?

| Task | Your Mac | Cloud GPU |
|------|----------|-----------|
| Download 400MB model | ❌ You said no | ✅ Stays on their server |
| Process with GPU | ❌ Slow/impossible | ✅ Fast (RTX 4090) |
| Storage after done | ❌ Uses your disk | ✅ Deleted when pod stops |

---

## The Files We're Creating

```
/workspace/Abliteration/          ← On RunPod's server
├── requirements.txt              ← Package list (done!)
├── src/
│   ├── __init__.py
│   ├── chat_templates.py        ← Qwen chat format
│   ├── datasets.py              ← Loads harmful/harmless data
│   ├── utils.py                 ← Helper functions
│   └── abliterate.py            ← Main logic
└── run_abliteration.py          ← Entry point
```

These are copies of what's on your Mac, but they run on the cloud GPU.

---

## Cost Reminder

- **Running:** ~$0.34/hour
- **Stopped:** $0/hour (but storage ~$0.01/hr if you keep the pod)
- **Deleted:** $0

⚠️ **Always stop/delete your pod when done!**
