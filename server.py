import os
import re
import math
import tempfile
import cv2
import numpy as np
import torch
import time
from collections import Counter
from typing import List, Dict, Optional

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import PlainTextResponse

from ultralytics import YOLO
from decord import VideoReader, cpu
from transformers import VideoMAEFeatureExtractor, VideoMAEForVideoClassification
from groq import Groq

# ---------------- CONFIG ----------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("Missing GROQ_API_KEY environment variable. Set: export GROQ_API_KEY=\"your_key\"")

# ---------------- MODELS ----------------
yolo_model = YOLO("yolov8x.pt")
VIDEOMAE_MODEL = "MCG-NJU/videomae-base-finetuned-kinetics"
videomae_extractor = VideoMAEFeatureExtractor.from_pretrained(VIDEOMAE_MODEL)
videomae = VideoMAEForVideoClassification.from_pretrained(VIDEOMAE_MODEL).to(DEVICE).eval()
client = Groq(api_key=GROQ_API_KEY)

# ---------------- HELPERS ----------------
def sample_even_frames_decord(video_path: str, num_frames: int = 16) -> List[np.ndarray]:
    vr = VideoReader(video_path, ctx=cpu(0))
    total = len(vr)
    if total == 0:
        return []
    idxs = [min(total - 1, math.floor(i * (total / num_frames))) for i in range(num_frames)]
    batch = vr.get_batch(idxs).asnumpy()
    return [f for f in batch]

def extract_sampled_frames_cv(video_path: str, num_frames: int = 32) -> List[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if total == 0:
        cap.release()
        return []
    idxs = np.linspace(0, total - 1, num_frames, dtype=int)
    frames = []
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            continue
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames

def detect_objects_across_frames(frames: List[np.ndarray], imgsz: int = 640, conf: float = 0.25) -> Counter:
    hist = Counter()
    if not frames:
        return hist
    step = max(1, len(frames) // 24)
    for i in range(0, len(frames), step):
        try:
            results = yolo_model.predict(frames[i], imgsz=imgsz, conf=conf, device=DEVICE, verbose=False)
            if not results:
                continue
            r0 = results[0]
            try:
                cls_list = r0.boxes.cls.tolist()
            except Exception:
                cls_list = []
            for cls_id in cls_list:
                label = yolo_model.names[int(cls_id)]
                hist.update([label])
        except Exception:
            continue
    return hist

def recognize_action_videomae(frames: List[np.ndarray]) -> str:
    if not frames:
        return ""
    if len(frames) != 16:
        idxs = np.linspace(0, len(frames) - 1, 16, dtype=int)
        frames = [frames[i] for i in idxs]
    inputs = videomae_extractor(images=frames, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        if DEVICE == "cuda":
            amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                logits = videomae(**inputs).logits
        else:
            logits = videomae(**inputs).logits
    pred = int(logits.argmax(-1).item())
    label = videomae.config.id2label.get(pred, "")
    return label.replace("_", " ")

def build_scene_summary(object_hist: Counter, action: str, extra_facts: Optional[str] = None) -> str:
    top_objs = [o for o, _ in object_hist.most_common(6)]
    obj_part = ", ".join(top_objs) if top_objs else ""
    action_part = action.strip()
    pieces = []
    if obj_part:
        pieces.append(f"Objects: {obj_part}.")
    if action_part:
        pieces.append(f"Primary action: {action_part}.")
    if extra_facts:
        pieces.append(extra_facts)
    if not pieces:
        return "No clear objects or actions were detected in the video."
    return " ".join(pieces)

def is_mcq(prompt: str) -> bool:
    return bool(re.search(r"\bA\.", prompt, re.IGNORECASE))

def parse_mcq_options(prompt: str) -> Dict[str, str]:
    parts = re.split(r"(\b[A-D]\.)", prompt)
    options: Dict[str, str] = {}
    cur = None; buf = []
    for p in parts:
        if re.fullmatch(r"[A-D]\.", p):
            if cur and buf:
                options[cur] = " ".join(" ".join(buf).split()).strip()
            cur = p[0]; buf = []
        else:
            if cur:
                buf.append(p)
    if cur and buf:
        options[cur] = " ".join(" ".join(buf).split()).strip()
    for k in list(options.keys()):
        options[k] = re.sub(r"\s*[A-D]\.\s*$", "", options[k]).strip()
    return options

def groq_answer_mcq(summary: str, prompt: str, options: Dict[str,str]) -> str:
    messages = [
        {"role":"system","content":"You are a video QA assistant. Use ONLY the provided summary to answer MCQs."},
        {"role":"user","content": f"Video summary: {summary}\nQuestion: {prompt}\nOptions: {'; '.join([f'{k}. {v}' for k,v in options.items()])}\n\nRespond with EXACTLY one option in the format 'LETTER. option text'."}
    ]
    resp = client.chat.completions.create(model=GROQ_MODEL, messages=messages, temperature=0.0, max_tokens=30)
    text = resp.choices[0].message.content.strip()
    m = re.match(r"^\s*([A-D])\s*\.?\s*(.*)$", text, flags=re.IGNORECASE)
    if m:
        letter, tail = m.group(1).upper(), m.group(2).strip()
        if not tail and letter in options:
            return f"{letter}. {options[letter]}"
        return f"{letter}. {tail or options.get(letter,'')}".strip()
    if options:
        k = sorted(options.keys())[0]
        return f"{k}. {options[k]}"
    return text

def groq_answer_descriptive(summary: str, prompt: str) -> str:
    messages = [
        {"role":"system","content":"You are a concise assistant that answers questions about videos using the provided summary."},
        {"role":"user","content": f"Video summary: {summary}\nQuestion: {prompt}\nAnswer in 1-2 short sentences, plain text only."}
    ]
    resp = client.chat.completions.create(model=GROQ_MODEL, messages=messages, temperature=0.2, max_tokens=120)
    return resp.choices[0].message.content.strip()

# ---------------- APP ----------------
app = FastAPI()

@app.post("/infer", response_class=PlainTextResponse)
async def infer(video: UploadFile = File(...), prompt: str = Form(...)):
    start_time = time.time()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(await video.read())
        video_path = tmp.name

    frames_for_action = sample_even_frames_decord(video_path, num_frames=16)
    frames_for_objects = extract_sampled_frames_cv(video_path, num_frames=32)

    object_hist = detect_objects_across_frames(frames_for_objects)
    action_label = recognize_action_videomae(frames_for_action)
    summary = build_scene_summary(object_hist, action_label)

    if is_mcq(prompt):
        options = parse_mcq_options(prompt)
        if not options:
            answer = groq_answer_descriptive(summary, prompt)
        else:
            answer = groq_answer_mcq(summary, prompt, options)
    else:
        answer = groq_answer_descriptive(summary, prompt)

    total_time = time.time() - start_time
    print(f"[PERFORMANCE] Total inference time: {total_time:.2f} seconds")

    return f"{prompt}\nAnswer: {answer}"
