Visual Understanding Agentic Chat Assistant

🚀 Project Overview:

The Visual Understanding Agentic Chat Assistant is a cutting-edge AI system designed to process short video clips (≤2 minutes), recognize key events, summarize content—including guideline/violation detection—and engage in natural, multi-turn dialogue with users. Our agentic pipeline bridges state-of-the-art video understanding, event recognition, and conversational AI, allowing anyone to query complex video scenes as naturally as having a conversation with a human expert.


🏗️ System Architecture:

Below is the high-level architecture of our assistant, as visualized using Napkin AI:

A)![System ArchitectureVideo Preprocessing**: Extracts and resizes video frames for efficient analysis.

B)Event Recognition: Utilizes state-of-the-art AI models to identify key events and actions in each frame.

C)Summarization & Detection: Summarizes events and detects guideline violations.

D)Conversation Management: Maintains dialogue context across multiple conversational turns.

E)Response Formatting: Converts AI analysis into clear, human-readable language.


⚙️ Tech Stack Justification:

1. Frontend
Streamlit – For building the interactive web interface

HTML/CSS (via Streamlit Components) – For minimal styling and layout

2. Backend
Python 3.9+ – Core programming language

OpenCV – For video processing and frame extraction

Ultralytics YOLOv8 – Object detection on video frames

LangChain – Prompt management and LLM pipeline

Groq LLM / OpenAI API – For event generation and video summarization

3. Data Handling & Storage
JSON – Event logging and export

Pandas – Optional, for tabular data handling if needed

4. Additional Tools
FFmpeg (optional) – For any advanced video conversion or frame extraction

Pathlib – For file handling

time / datetime – For generating timestamps

5. Deployment
Streamlit Cloud / Local Server – Easy deployment and sharing

GitHub – Version control and project hosting


🧑💻 Setup & Installation:

Follow these steps to get your assistant running locally:
cd VU-Chat-Assistant
pip install -r requirements.txt
# (Optional) Configure .env for API keys or DB settings.
uvicorn app.main:app --reload

🕹️ Usage Instructions:

How to Interact:

1)Open the Interface (Web or CLI).

2)Upload your video clip (≤2 minutes).

3)Chat naturally—ask questions about events, violations, timelines, or request summaries.

Example Dialog:

User: [Uploads video]
Assistant: "Events detected — Car 2 ran a red light at 00:43; Pedestrian crossed at 01:15."
User: "Who broke the rules first?"
Assistant: "Car 2 was the first to violate at 00:43."
User: "Were there any illegal pedestrian crossings?"
Assistant: "Yes, one—Pedestrian X crossed against the signal at 01:15."


📝 Features Implemented:

A)Video Event Recognition: Scene-level detection and annotation of specified activities (vehicles, pedestrians, traffic lights).

B)Summarization & Violation Detection: Time-stamped, interpretable summaries about video and guideline adherence.

C)Multi-Turn Dialogue: Memory-aware, context-retentive communication for in-depth Q&A.

D)Agentic Modularity: Clean separation of modules, easily extended to add new models/events.


✨ Innovation & Stand-Out Elements:

1)Hybrid VLM + LLM Architecture: Combines the strengths of vision-language and conversational models for robust understanding.

2)Modern, Modular Design: Intuitive, maintainable, and ready for future scale-up or challenge rounds.

3)User-First UX: Natural, conversational access to complex video analysis for both technical and non-technical users.

📈 Next Steps:

1)Extend to real-time stream processing and longer video durations.

2)Add multi-language/chatbot accessibility.

3)Integrate performance benchmarking for Round 2.

🎥 Demo Video: https://drive.google.com/file/d/1DzAVxTinLZO3gKW8tZLgrpyyriaICuya/view?usp=sharing

🖼️ Architecture Diagram: https://drive.google.com/file/d/1rmGZmFwCq3SMG1ZZC1M-DhNfkjcqdtin/view?usp=sharing




