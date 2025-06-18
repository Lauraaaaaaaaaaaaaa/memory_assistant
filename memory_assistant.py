import ollama
import streamlit as st
import os
import json
from langchain_google_genai import ChatGoogleGenerativeAI
from datetime import datetime
import uuid
from enum import Enum
import cv2
from PIL import Image
import numpy as np
from PIL import ImageOps

st.set_page_config(layout="wide")

# MODEL CONFIGURATION (Switch between local or cloud)
class LLMModel(Enum):
    GEMINI = "gemini"
    PHI3 = "phi3"

USE_MODEL: LLMModel = LLMModel.PHI3

def config_llm(model: LLMModel = USE_MODEL, temperature: float = 0.2) -> object:
    if model == LLMModel.GEMINI:
        os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
        return ChatGoogleGenerativeAI(
            model="gemini-1.5-flash", temperature=temperature, max_output_tokens=512
        )
    elif model == LLMModel.PHI3:
        return ollama(model="phi3", temperature=temperature)
    else:
        raise ValueError("Unsupported model type.")

def get_llm() -> object:
    return config_llm(model=USE_MODEL, temperature=0.6)

# ----------------------------
# MEMORY CUES
MEMORY_CUE_PATH = "user_memory_cues.json"  # Where you store known facts

def load_memory_cues():
    if os.path.exists(MEMORY_CUE_PATH):
        with open(MEMORY_CUE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def build_system_prompt(memory_cues: dict):
    base_prompt = (
        "You are a kind, patient, and supportive assistant helping dementia patients with memory. "
        "Speak clearly, offer encouragement, and help them recall positive memories. "
        "Avoid correcting them directly, even if their memory seems confused. Just guide gently and ask simple, friendly follow-ups."
    )

    if memory_cues:
        personal_info = "\n".join(f"- {key.replace('_', ' ').capitalize()}: {val}" for key, val in memory_cues.items())
        base_prompt += f"\n\nPersonal information about the user:\n{personal_info}"

    base_prompt += """
    
Here are examples of how to respond:

User: I remember my daughter Ana used to visit every weekend.
Assistant: That sounds lovely. What would you two usually do together on those weekends?

User: I think I used to be a teacher… right?
Assistant: Yes, you taught for many years. Do you remember any favorite students or subjects?

User: I feel like I'm forgetting things more.
Assistant: That's okay. You're doing great, and I'm here to help you remember the moments that matter most.
"""
    return base_prompt

# ----------------------------
# CHAT HISTORY

def initialize_chat():
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

def update_history(role, content):
    st.session_state.chat_history.append({"role": role, "content": content})

def format_messages(system_prompt):
    return [{"role": "system", "content": system_prompt}] + st.session_state.chat_history

# ----------------------------
# CALENDAR & REMINDERS SECTION

REMINDERS_FILE = "reminders.json"

def load_reminders():
    if os.path.exists(REMINDERS_FILE):
        with open(REMINDERS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_reminders(reminders):
    with open(REMINDERS_FILE, "w", encoding="utf-8") as f:
        json.dump(reminders, f)

def add_reminder(title, date, time, notes):
    reminders = load_reminders()
    reminder_id = str(uuid.uuid4())
    reminders.append({
        "id": reminder_id,
        "title": title,
        "date": date,
        "time": time,
        "notes": notes
    })
    save_reminders(reminders)
    return reminder_id

def delete_reminder(reminder_id):
    reminders = load_reminders()
    reminders = [reminder for reminder in reminders if reminder["id"] != reminder_id]
    save_reminders(reminders)

def get_upcoming_reminders():
    reminders = load_reminders()
    upcoming_reminders = []
    current_date = datetime.now()
    for reminder in reminders:
        reminder_date = datetime.strptime(reminder["date"], "%Y-%m-%d")
        if 0 <= (reminder_date - current_date).days < 7:
            upcoming_reminders.append(reminder)
    return upcoming_reminders

# ----------------------------
# MEMORIES GALLERY
# Define the moods and their corresponding colors and emojis
moods = [
    {"name": "grateful", "color": "bg-amber-100 text-amber-800", "emoji": "🙏"},
    {"name": "peaceful", "color": "bg-blue-100 text-blue-800", "emoji": "🕊️"},
    {"name": "joyful", "color": "bg-pink-100 text-pink-800", "emoji": "😊"},
    {"name": "warm", "color": "bg-orange-100 text-orange-800", "emoji": "🤗"},
    {"name": "inspired", "color": "bg-purple-100 text-purple-800", "emoji": "✨"}
]

# Initialize memories in session state
def initialize_memories():
    if "memories" not in st.session_state:
        st.session_state.memories = []

def save_memory_cue(key, value):
    memory_cues = load_memory_cues()
    memory_cues[key] = value
    try:
        with open(MEMORY_CUE_PATH, "w") as f:
            json.dump(memory_cues, f)
    except:
        pass

# ----------------------------
# CAMERA SECTION
# facial recognition of loved ones by uploading or taking a photo


# ----------------------------
# STREAMLIT UI
# ----------------------------

def landing_page():
    st.markdown("""
    <style>
    .circle-btn {
        width: 200px;
        height: 200px;
        border-radius: 50%;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        margin: 24px auto;
        font-size: 1.1em;
        font-weight: 600;
        box-shadow: 0 2px 12px #0001;
        border: none;
        cursor: pointer;
        transition: box-shadow 0.2s, transform 0.2s;
        background: #f6f8fb;
        position: relative;
        text-align: center;
    }
    .circle-btn:hover {
        box-shadow: 0 6px 24px #0002;
        transform: scale(1.05);
    }
    .circle-content {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        height: 100%;
        width: 100%;
    }
    .circle-icon {
        font-size: 2.5em;
        margin-bottom: 12px;
    }
    .circle-title {
        font-size: 1.15em;
        font-weight: 700;
        margin-bottom: 6px;
        color: #222;
        line-height: 1.2;
        word-break: break-word;
    }
    .circle-desc {
        font-size: 0.95em;
        color: #666;
        font-weight: 400;
        text-align: center;
        line-height: 1.2;
        word-break: break-word;
    }
    </style>
    """, unsafe_allow_html=True)

    features = [
        {"icon": "💬", "title": "Digital Diary", "desc": "Record memories and ask personal questions", "key": "diary", "color": "#e3eafe"},
        {"icon": "📷", "title": "Face Recognition", "desc": "Identify familiar faces", "key": "face", "color": "#ffe7c2"},
        {"icon": "👨‍👩‍👧‍👦", "title": "Loved Ones", "desc": "Remember important people", "key": "loved", "color": "#d6f5e6"},
        {"icon": "🖼️", "title": "Memory Gallery", "desc": "Browse photos and memories", "key": "gallery", "color": "#e9e6fa"},
        {"icon": "📅", "title": "Calendar & Reminders", "desc": "Medication schedule and appointments", "key": "calendar", "color": "#ffeaea"},
        {"icon": "🎮", "title": "Memory Games", "desc": "Science-proven cognitive exercises", "key": "games", "color": "#fff7d1"},
    ]
    rows = [features[:3], features[3:]]
    for row in rows:
        cols = st.columns(3)
        for i, feature in enumerate(row):
            with cols[i]:
                with st.form(key=f"circle_form_{feature['key']}"):
                    submitted = st.form_submit_button(
                        label=" ",
                        use_container_width=True
                    )
                    st.markdown(
                        f"""
                        <div class="circle-btn" style="background:{feature['color']};">
                            <div class="circle-content">
                                <div class="circle-icon">{feature['icon']}</div>
                                <div class="circle-title">{feature['title']}</div>
                                <div class="circle-desc">{feature['desc']}</div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    if submitted:
                        st.session_state.selected_page = feature['key']

def main():
    st.title("🧠 Memory Companion")
    initialize_chat()
    initialize_memories()
    if "selected_page" not in st.session_state:
        st.session_state.selected_page = None
    if "known_faces" not in st.session_state:
        st.session_state.known_faces = []
    if "gallery" not in st.session_state:
        st.session_state.gallery = []

    if st.session_state.selected_page is None:
        landing_page()
    else:
        if st.button("⬅️ Back", key="back_btn"):
            st.session_state.selected_page = None

        if st.session_state.selected_page == "diary":
            st.header("💬 Digital Diary")
            tab1, tab2 = st.tabs(["Chat", "Add Memory"])
            with tab1:
                # Chat tab content
                # Ensure system_prompt is defined
                memory_cues = load_memory_cues()
                system_prompt = build_system_prompt(memory_cues)
                # Show full chat history
                if st.session_state.chat_history:
                    st.markdown("### Our Conversation:")
                    for msg in st.session_state.chat_history:
                        if msg["role"] == "user":
                            st.write(f"**You:** {msg['content']}")
                        else:
                            st.write(f"**Companion:** {msg['content']}")
                else:
                    st.markdown("""
                    ### ✨ Hey friend! 
                    You can:
                    - Share memories with me
                    - Tell me how you're feeling  
                    - Ask me personal questions
                    - Just have a friendly chat
                    """)

                # User input
                user_input = st.text_input("What would you like to tell me or remember today?", key="user_input")

                if user_input:
                    update_history("user", user_input)

                    with st.spinner("Thinking..."):
                        try:
                            llm = get_llm()
                            messages = format_messages(system_prompt)
                            response = llm.invoke(messages)
                            update_history("assistant", response.content)
                        except Exception as e:
                            st.error(f"Error generating response: {str(e)}")
                            update_history("assistant", "I'm sorry, I'm having trouble connecting right now. Please try again.")

                st.rerun()  # Use st.rerun() instead of st.experimental_rerun()

            with tab2:
                # 💭 Memory Form (relocated here from tab3)
                with st.form("add_memory"):
                    st.subheader("💭 Add a New Memory")
                    new_memory_text = st.text_area("What made you smile today?")
                    selected_mood = st.selectbox("How did it make you feel?", [mood["name"] for mood in moods])
                    
                    if st.form_submit_button("Save Memory"):
                        if new_memory_text:
                            new_memory = {
                                "id": len(st.session_state.memories) + 1, 
                                "text": new_memory_text, 
                                "date": datetime.date.today().strftime("%Y-%m-%d"), 
                                "mood": selected_mood
                            }
                            st.session_state.memories.append(new_memory)
                            st.success("Memory saved!")
                            st.rerun()

        elif st.session_state.selected_page == "face":
            st.header("🧑‍🤝‍🧑 Face Recognition")
            tab1, tab2 = st.tabs(["Upload Photo", "Take Photo"])
            with tab1:
                # Load OpenCV's pre-trained Haar Cascade face detector
                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

                # Simulated in-memory photo storage
                if "gallery" not in st.session_state:
                    st.session_state.gallery = []

                # Each known face: {"face_img": np.array(gray), "name": str, "description": str}
                if "known_faces" not in st.session_state:
                    st.session_state.known_faces = []

                def mse(imageA, imageB):
                    # Compute Mean Squared Error between two images
                    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
                    err /= float(imageA.shape[0] * imageA.shape[1])
                    return err

                def recognize_face(face_img_gray, threshold=2000):
                    # Compare the detected face with known faces by MSE; lower MSE means more similar
                    for known in st.session_state.known_faces:
                        known_face = known["face_img"]
                        if known_face.shape != face_img_gray.shape:
                            # Resize known face to match detected face shape
                            known_face_resized = cv2.resize(known_face, (face_img_gray.shape[1], face_img_gray.shape[0]))
                        else:
                            known_face_resized = known_face
                        
                        error = mse(face_img_gray, known_face_resized)
                        if error < threshold:
                            return known["name"], known["description"]
                    return None, None

                uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
                if uploaded_file:
                    image = Image.open(uploaded_file)
                    image = ImageOps.exif_transpose(image)  # Correct orientation
                    image = ImageOps.fit(image, (640, 480), method=Image.Resampling.LANCZOS)  # Now resize it keeping proportions

                    # Convert PIL to OpenCV
                    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

                    if faces is not None and len(faces) > 0:
                        if faces is not None and len(faces) > 0:
                            # Find indices of unknown faces
                            unknown_indices = []
                            for i, (x, y, w, h) in enumerate(faces):
                                face_img_gray = gray[y:y+h, x:x+w]
                                name, desc = recognize_face(face_img_gray)
                                if name is None:
                                    unknown_indices.append(i)

                            for i, (x, y, w, h) in enumerate(faces):
                                face_img_gray = gray[y:y+h, x:x+w]
                                name, desc = recognize_face(face_img_gray)
                                color = (0, 255, 0) if name else (0, 0, 255)

                                if name:
                                    label = name
                                else:
                                    if len(unknown_indices) == 1:
                                        label = "Unknown"
                                    else:
                                        unknown_number = unknown_indices.index(i) + 1
                                        label = f"Unknown{unknown_number}"

                                cv2.rectangle(img_cv, (x, y), (x + w, y + h), color, 2)
                                cv2.putText(img_cv, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                            # Let user optionally label unknown faces
                            for i, (x, y, w, h) in enumerate(faces):
                                face_img_gray = gray[y:y+h, x:x+w]
                                name, desc = recognize_face(face_img_gray)
                                if name is None:
                                    st.markdown(f"### Label unknown face #{i+1}")
                                    face_name = st.text_input(f"Name for face #{i+1}", key=f"name_{i}")
                                    face_desc = st.text_input(f"Description for face #{i+1}", key=f"desc_{i}")
                                    
                                    loved = st.checkbox(f"Add {face_name} to Loved Ones?", key=f"loved_{i}")
                                    if st.button(f"Save face #{i+1} info", key=f"saveface_{i}"):
                                        st.session_state.known_faces.append({
                                            "face_img": face_img_gray,
                                            "name": face_name,
                                            "description": face_desc,
                                            "loved_one": loved
                                        })
                                        st.success(f"Saved face #{i+1} as {face_name}")

                        st.image(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB), use_container_width=True)

                        description = st.text_input("Enter photo description", key="upload_desc")

                        if st.button("Save Photo", key="upload_save"):
                            st.session_state.gallery.append({
                                "image": Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)),
                                "description": description,
                                "date": datetime.now().strftime("%m/%d/%Y")
                            })

            with tab2:
                if "camera_frame" not in st.session_state:
                    st.session_state.camera_frame = None

                if st.button("📷 Take Photo"):
                    cap = cv2.VideoCapture(0)
                    ret, frame = cap.read()
                    cap.release()
                    if ret:
                        st.session_state.camera_frame = frame
                    else:
                        st.error("Failed to access the camera.")

                if st.session_state.camera_frame is not None:
                    frame = st.session_state.camera_frame.copy()
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

                    if faces is not None and len(faces) > 0:
                        # Find indices of unknown faces
                        unknown_indices = []
                        for i, (x, y, w, h) in enumerate(faces):
                            face_img_gray = gray[y:y+h, x:x+w]
                            name, desc = recognize_face(face_img_gray)
                            if name is None:
                                unknown_indices.append(i)

                        for i, (x, y, w, h) in enumerate(faces):
                            face_img_gray = gray[y:y+h, x:x+w]
                            name, desc = recognize_face(face_img_gray)
                            color = (0, 255, 0) if name else (0, 0, 255)

                            if name:
                                label = name
                            else:
                                if len(unknown_indices) == 1:
                                    label = "Unknown"
                                else:
                                    unknown_number = unknown_indices.index(i) + 1
                                    label = f"Unknown{unknown_number}"

                            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                        # Label unknown faces on camera photo
                        for i, (x, y, w, h) in enumerate(faces):
                            face_img_gray = gray[y:y+h, x:x+w]
                            name, desc = recognize_face(face_img_gray)
                            if name is None:
                                st.markdown(f"### Label unknown face #{i+1}")
                                face_name = st.text_input(f"Name for face #{i+1}", key=f"cam_name_{i}")
                                face_desc = st.text_input(f"Description for face #{i+1}", key=f"cam_desc_{i}")

                                loved = st.checkbox(f"Add {face_name} to Loved Ones?", key=f"loved_{i}")
                                if st.button(f"Save face #{i+1} info", key=f"saveface_{i}"):
                                    st.session_state.known_faces.append({
                                        "face_img": face_img_gray,
                                        "name": face_name,
                                        "description": face_desc,
                                        "loved_one": loved
                                    })
                                    st.success(f"Saved face #{i+1} as {face_name}")

                    st.image(frame, channels="BGR", use_container_width=True)

                    description = st.text_input("Enter photo description", key="camera_desc")

                    if st.button("💾 Save Photo", key="camera_save"):
                        image = Image.fromarray(cv2.cvtColor(st.session_state.camera_frame, cv2.COLOR_BGR2RGB))
                        st.session_state.gallery.append({
                            "image": image,
                            "description": description,
                            "date": datetime.now().strftime("%m/%d/%Y")
                        })
                        st.session_state.camera_frame = None

                    if st.button("🔄 Take Another Photo"):
                        st.session_state.camera_frame = None

            # --- Photo Gallery ---
            if st.session_state.gallery:
                st.markdown("---")
                st.subheader("📷 Photo Gallery")

                col1, col2 = st.columns(2)
                for i, photo in enumerate(st.session_state.gallery):
                    col = col1 if i % 2 == 0 else col2
                    with col:
                        st.image(photo["image"], use_container_width=True)
                        st.markdown(f"**{photo['description']}**", unsafe_allow_html=True)
                        st.markdown(f"<div style='color:gray; font-size: 0.9em;'>Date: {photo['date']}</div>", unsafe_allow_html=True)

        elif st.session_state.selected_page == "loved":
            st.header("👨‍👩‍👧‍👦 Loved Ones")
            tab1, tab2 = st.tabs(["View Loved Ones", "Add Loved One"])
            with tab1:
                loved_faces = [f for f in st.session_state.known_faces if f.get("loved_one")]

                if loved_faces:
                    # Create columns
                    col1, col2, col3 = st.columns(3)
                    cols = [col1, col2, col3]
                    col_heights = [0, 0, 0]

                    # Estimate "height" for each card
                    def estimate_height(face):
                        desc_lines = len(face.get("description", "").split("\n"))
                        desc_lines += len(face.get("description", "")) // 50  # wrap estimate
                        return 3 + desc_lines  # 3 lines: name, icon+desc, image

                    # Sort faces into the column with the lowest total height
                    for face in loved_faces:
                        est_height = estimate_height(face)
                        col_idx = col_heights.index(min(col_heights))  # column with least height
                        col_heights[col_idx] += est_height

                        with cols[col_idx]:
                            st.markdown(f"### {face['name']}")
                            if face.get("description"):
                                st.write(f"📝 {face['description']}")
                            face_img = cv2.resize(face["face_img"], (150, 150))
                            face_img_color = cv2.cvtColor(face_img, cv2.COLOR_GRAY2RGB)
                            st.image(face_img_color, channels="RGB")
                else:
                    st.write("You haven't added any loved ones yet.")

            with tab2:
                # ... your add loved one code ...
                pass

        elif st.session_state.selected_page == "gallery":
            st.header("🖼️ Memory Gallery")
            tab1, tab2 = st.tabs(["Gallery", "Add Memory"])
            with tab1:
                # Display the memories
                if st.session_state.memories:
                    for memory in reversed(st.session_state.memories):  # Show newest first
                        mood_data = next((mood for mood in moods if mood["name"] == memory["mood"]), None)
                        if mood_data:
                            st.markdown(f"### {mood_data['emoji']} {memory['mood'].title()}")
                            st.write(memory["text"])
                            st.caption(f"📅 {memory['date']}")
                            st.divider()
                else:
                    st.write("No memories yet. Start capturing the beautiful moments in your day!")

            with tab2:
                # ... your add memory code ...
                pass

        elif st.session_state.selected_page == "calendar":
            st.header("📅 Calendar & Reminders")
            tab1, tab2 = st.tabs(["Reminders", "Add Reminder"])
            with tab1:
                # Show upcoming reminders
                st.subheader("📋 Upcoming Reminders")
                upcoming_reminders = get_upcoming_reminders()
                
                if upcoming_reminders:
                    for reminder in upcoming_reminders:
                        col1, col2 = st.columns([4, 1])
                        with col1:
                            st.write(f"📅 **{reminder['title']}**")
                            st.write(f"📆 {reminder['date']} | ⌚ {reminder['time']}")
                            if reminder['notes']:
                                st.write(f"📝 {reminder['notes']}")
                        with col2:
                            if st.button("Delete", key=f"del_{reminder['id']}"):
                                delete_reminder(reminder['id'])
                                st.rerun()
                        st.divider()
                else:
                    st.write("No upcoming reminders.")

            with tab2:
                # ... your add reminder code ...
                pass

        elif st.session_state.selected_page == "games":
            st.header("🎮 Memory Games")
            tab1, tab2 = st.tabs(["Games", "Progress"])
            with tab1:
                st.write("Games coming soon!")
            with tab2:
                st.write("Progress tracking coming soon!")

if __name__ == "__main__":
    main()
