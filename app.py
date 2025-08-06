import cv2
import json
import time
import streamlit as st
from pathlib import Path
from ultralytics import YOLO
from langchain_groq import ChatGroq
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# -----------------------------
# CONFIGURATION
# -----------------------------
st.set_page_config(page_title="🎥 Video Event Summarizer AI Agent", layout="wide")
st.title("🎥 Video Event Summarizer with AI Agent")

# Load YOLO model
model = YOLO("yolov8x.pt")  # yolov8s.pt for better accuracy

# Groq API key
GROQ_API_KEY = ""

# Initialize ChatGroq LLM
llm = ChatGroq(model="llama-3.1-8b-instant", groq_api_key=GROQ_API_KEY)

# Prompts
event_prompt = PromptTemplate(
    input_variables=["objects"],
    template="Detected objects: {objects}. "
             "Generate a single-sentence realistic event for a road or parking lot scene."
)
event_chain = LLMChain(llm=llm, prompt=event_prompt)

summary_prompt = PromptTemplate(
    input_variables=["events"],
    template="Here is the list of detected events from a video:\n\n{events}\n\n"
             "Generate a detailed summary of the video in 2-3 sentences."
)
summary_chain = LLMChain(llm=llm, prompt=summary_prompt)

agent_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="You are an AI agent analyzing a video. "
             "Context: {context}\nQuestion: {question}\nAnswer conversationally:"
)
agent_chain = LLMChain(llm=llm, prompt=agent_prompt)

# -----------------------------
# VIDEO UPLOAD
# -----------------------------
video_file = st.file_uploader("📤 Upload a video", type=["mp4", "avi", "mov"])

if video_file:
    video_path = Path("temp_video.mp4")
    with open(video_path, "wb") as f:
        f.write(video_file.read())

    st.info("⏳ Processing video")

    cap = cv2.VideoCapture(str(video_path))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_interval = max(1, int(fps * 1))  # Every 1 second
    frame_index = 0
    events_log = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_index % frame_interval == 0:
            timestamp = time.strftime("%H:%M:%S", time.gmtime(frame_index / fps))

            # Object detection
            results = model(frame, verbose=False)
            detected_objects = []
            for r in results:
                for c in r.boxes.cls:
                    detected_objects.append(model.names[int(c)])

            if detected_objects:
                # Generate event description
                event_text = event_chain.run(objects=", ".join(set(detected_objects)))
                events_log.append({"timestamp": timestamp, "event": event_text})

        frame_index += 1

    cap.release()

    # -----------------------------
    # SAVE JSON LOG (HIDDEN FROM UI)
    # -----------------------------
    json_file = "video_events.json"
    with open(json_file, "w") as f:
        json.dump(events_log, f, indent=4)

    st.success("✅ Video processing completed!")

    # -----------------------------
    # FINAL VIDEO SUMMARY
    # -----------------------------
    final_summary = summary_chain.run(events=json.dumps(events_log, indent=4))
    st.subheader("📌 Final Video Summary")
    st.write(final_summary)

    # Download JSON only (no JSON display on UI)
    st.download_button("📥 Download Event JSON",
                       data=json.dumps(events_log, indent=4),
                       file_name=json_file,
                       mime="application/json")

    # -----------------------------
    # CHATGPT-STYLE MULTI-TURN BOT
    # -----------------------------
    st.subheader("🤖 Chat with the AI Agent About the Video")

    # Initialize session state for chat
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display chat history in ChatGPT style
    for chat in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(chat["user"])
        with st.chat_message("assistant"):
            st.write(chat["ai"])

    # Chat input box
    user_query = st.chat_input("Ask anything about the video...")

    if user_query:
        # Generate AI response
        ai_response = agent_chain.run({
            "context": json.dumps(events_log, indent=4),
            "question": user_query
        })

        # Append to history
        st.session_state.chat_history.append({
            "user": user_query,
            "ai": ai_response
        })

        # Display the new messages immediately
        with st.chat_message("user"):
            st.write(user_query)
        with st.chat_message("assistant"):
            st.write(ai_response)
