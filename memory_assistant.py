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
# LOVED ONES DATA PERSISTENCE
LOVED_ONES_FILE = "loved_ones.json"

def save_loved_ones():
    """Save loved ones data to file"""
    try:
        loved_ones_data = []
        for face in st.session_state.known_faces:
            if face.get("loved_one"):
                # Convert numpy array to list for JSON serialization
                face_data = {
                    "name": face["name"],
                    "description": face["description"],
                    "loved_one": face["loved_one"],
                    "face_img_gray": face["face_img_gray"].tolist() if hasattr(face["face_img_gray"], 'tolist') else face["face_img_gray"],
                    "display_img": face["display_img"].tolist() if hasattr(face["display_img"], 'tolist') else face["display_img"]
                }
                loved_ones_data.append(face_data)
        
        with open(LOVED_ONES_FILE, "w", encoding="utf-8") as f:
            json.dump(loved_ones_data, f)
    except Exception as e:
        st.error(f"Error saving loved ones: {e}")

def load_loved_ones():
    """Load loved ones data from file"""
    try:
        if os.path.exists(LOVED_ONES_FILE):
            with open(LOVED_ONES_FILE, "r", encoding="utf-8") as f:
                loved_ones_data = json.load(f)
            
            # Convert list back to numpy array and add unique IDs if missing
            for face_data in loved_ones_data:
                # Compatibility for old data using "face_img"
                if "face_img" in face_data and "face_img_gray" not in face_data:
                    face_data["face_img_gray"] = face_data.pop("face_img")

                if "face_img_gray" in face_data and isinstance(face_data["face_img_gray"], list):
                    face_data["face_img_gray"] = np.array(face_data["face_img_gray"], dtype=np.uint8)
                
                if "display_img" in face_data and isinstance(face_data["display_img"], list):
                    face_data["display_img"] = np.array(face_data["display_img"], dtype=np.uint8)

                # --- Data Migration: Add unique ID if it doesn't exist ---
                if "id" not in face_data:
                    face_data["id"] = str(uuid.uuid4())
            
            return loved_ones_data
        return []
    except Exception as e:
        st.error(f"Error loading loved ones: {e}")
        return []

# ----------------------------
# CENTRALIZED FACE MANAGEMENT
# ----------------------------

def add_or_update_known_face(name, description, face_img_gray, is_loved_one, face_id=None, display_img=None):
    """
    Adds a new face or updates an existing one in st.session_state.known_faces.
    A new face is assigned a unique ID.
    """
    if face_id:
        # This is an update to an existing face
        for face in st.session_state.known_faces:
            if face.get("id") == face_id:
                face["name"] = name
                face["description"] = description
                face["loved_one"] = is_loved_one
                # Optionally update face_img if a new one is provided
                if face_img_gray is not None:
                    face["face_img_gray"] = face_img_gray
                if display_img is not None:
                    face["display_img"] = display_img
                break
    else:
        # This is a new face
        new_face = {
            "id": str(uuid.uuid4()),
            "name": name,
            "description": description,
            "face_img_gray": face_img_gray,
            "display_img": display_img,
            "loved_one": is_loved_one
        }
        st.session_state.known_faces.append(new_face)
    
    # Persist changes to the file
    save_loved_ones()

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
        # Load existing loved ones from file
        saved_loved_ones = load_loved_ones()
        st.session_state.known_faces = saved_loved_ones
    if "gallery" not in st.session_state:
        st.session_state.gallery = []
    if 'just_added_loved_one' not in st.session_state:
        st.session_state.just_added_loved_one = None

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
                        known_face = known["face_img_gray"]
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
                                        # Also grab the color crop for display
                                        color_crop = img_cv[y:y+h, x:x+w]
                                        color_crop_rgb = cv2.cvtColor(color_crop, cv2.COLOR_BGR2RGB)
                                        add_or_update_known_face(face_name, face_desc, face_img_gray, loved, display_img=color_crop_rgb)
                                        st.rerun()

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
                                    color_crop = frame[y:y+h, x:x+w]
                                    color_crop_rgb = cv2.cvtColor(color_crop, cv2.COLOR_BGR2RGB)
                                    add_or_update_known_face(face_name, face_desc, face_img_gray, loved, display_img=color_crop_rgb)
                                    st.rerun()

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
                    # Create columns for the grid layout
                    cols = st.columns(3)
                    
                    # Distribute loved ones into columns in a round-robin fashion
                    for i, face in enumerate(loved_faces):
                        with cols[i % 3]:
                            with st.container(border=True):
                                # Use the unique ID for the key to ensure components are unique
                                face_key = face.get("id", f"face_{face['name'].replace(' ', '_')}_{i}")
                                
                                st.markdown(f"### {face['name']}")
                                
                                # Set a minimum height for the description area for consistent card height
                                description_text = face.get("description", "")
                                st.markdown(
                                    f'<div style="word-wrap: break-word; overflow-wrap: break-word; min-height: 80px;">'
                                    f'📝 {description_text}'
                                    f'</div>',
                                    unsafe_allow_html=True
                                )

                                # Display face image with a placeholder
                                display_image_data = face.get("display_img")
                                if display_image_data is not None and isinstance(display_image_data, np.ndarray):
                                    display_image_resized = cv2.resize(display_image_data, (150, 150))
                                    st.image(display_image_resized, channels="RGB")
                                elif isinstance(face.get("face_img_gray"), np.ndarray):
                                    face_img = cv2.resize(face["face_img_gray"], (150, 150))
                                    face_img_color = cv2.cvtColor(face_img, cv2.COLOR_GRAY2RGB)
                                    st.image(face_img_color, channels="RGB")
                                else:
                                    # Image placeholder to maintain layout consistency
                                    st.markdown(
                                        '<div style="height: 150px; display: flex; align-items: center; '
                                        'justify-content: center; background-color: #f0f2f6; border-radius: 8px;">'
                                        'No photo'
                                        '</div>', 
                                        unsafe_allow_html=True
                                    )
                                
                                # Edit and Delete buttons
                                col_edit, col_delete = st.columns(2)
                                with col_edit:
                                    if st.button("✏️ Edit", key=f"edit_{face_key}"):
                                        st.session_state.editing_face = face
                                        st.session_state.edit_face_index = st.session_state.known_faces.index(face)
                                        st.rerun() # Rerun to bring the edit form into view
                                with col_delete:
                                    if st.button("🗑️ Delete", key=f"delete_{face_key}"):
                                        face_to_remove = next((f for f in st.session_state.known_faces if f.get("id") == face.get("id")), None)
                                        if face_to_remove:
                                            st.session_state.known_faces.remove(face_to_remove)
                                            save_loved_ones()
                                            st.success(f"Removed {face['name']} from loved ones")
                                            st.rerun()
                            
                            # Add vertical space between cards in the same column
                            st.markdown("<br>", unsafe_allow_html=True)
                    
                    # Edit form (appears when editing)
                    if "editing_face" in st.session_state:
                        st.markdown("---")
                        st.subheader(f"✏️ Edit {st.session_state.editing_face['name']}")
                        
                        with st.form("edit_loved_one"):
                            new_name = st.text_input("Name", value=st.session_state.editing_face['name'])
                            new_desc = st.text_area("Description", value=st.session_state.editing_face['description'])
                            
                            col_save, col_cancel = st.columns(2)
                            with col_save:
                                if st.form_submit_button("💾 Save Changes"):
                                    face_id = st.session_state.editing_face.get("id")
                                    is_loved = st.session_state.editing_face.get("loved_one", True)
                                    add_or_update_known_face(new_name, new_desc, None, is_loved, face_id=face_id)
                                    
                                    del st.session_state.editing_face
                                    del st.session_state.edit_face_index
                                    st.success("Changes saved!")
                                    st.rerun()
                            with col_cancel:
                                if st.form_submit_button("❌ Cancel"):
                                    del st.session_state.editing_face
                                    del st.session_state.edit_face_index
                                    st.rerun()
                else:
                    st.write("You haven't added any loved ones yet.")
                    st.info("💡 Tip: You can add loved ones from the Face Recognition section or use the 'Add Loved One' tab above.")

            with tab2:
                if st.session_state.get('just_added_loved_one'):
                    st.success(f"✅ {st.session_state.just_added_loved_one} has been added to your loved ones!")
                    st.info("You can view them in the 'View Loved Ones' tab.")
                    if st.button("➕ Add Another Loved One"):
                        st.session_state.just_added_loved_one = None
                        st.rerun()
                else:
                    st.subheader("➕ Add a New Loved One")
                    
                    with st.form("add_loved_one"):
                        loved_name = st.text_input("Name", placeholder="Enter their name")
                        loved_desc = st.text_area("Description", placeholder="Tell me about this person...")
                        
                        photo_option = st.radio("Add a photo?", ["No photo", "Upload photo"])
                        
                        # The file uploader widget is now created unconditionally
                        uploaded_photo = st.file_uploader("Choose a photo", type=["jpg", "jpeg", "png"], key="loved_photo")

                        # Show a preview only if the option is selected AND a file is uploaded
                        if photo_option == "Upload photo" and uploaded_photo is not None:
                            image = Image.open(uploaded_photo)
                            st.image(image, caption="Preview", width=200)
                        
                        submitted = st.form_submit_button("Add Loved One")

                        if submitted:
                            if not (loved_name and loved_desc):
                                st.error("Please provide both name and description.")
                            else:
                                with st.spinner("Adding..."):
                                    face_img_gray, display_img = None, None
                                    photo_processed_successfully = False

                                    if photo_option == "No photo":
                                        # Create a simple placeholder image for both
                                        face_img_gray = np.zeros((100, 100), dtype=np.uint8)
                                        face_img_gray.fill(128)  # Gray background
                                        display_img = cv2.cvtColor(face_img_gray, cv2.COLOR_GRAY2RGB)
                                        photo_processed_successfully = True
                                    
                                    elif photo_option == "Upload photo":
                                        if uploaded_photo is not None:
                                            # Process the image to get both color and grayscale versions
                                            image = Image.open(uploaded_photo)
                                            image = ImageOps.exif_transpose(image)
                                            
                                            display_image = ImageOps.fit(image, (200, 200), method=Image.Resampling.LANCZOS)
                                            display_img = np.array(display_image)
                                            
                                            gray = cv2.cvtColor(display_img, cv2.COLOR_RGB2GRAY)
                                            face_img_gray = cv2.resize(gray, (100, 100))
                                            photo_processed_successfully = True
                                        else:
                                            st.error("You selected 'Upload photo'. Please upload a file or select 'No photo'.")

                                    if photo_processed_successfully:
                                        add_or_update_known_face(loved_name, loved_desc, face_img_gray, True, display_img=display_img)
                                        st.session_state.just_added_loved_one = loved_name
                                        st.rerun()

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
