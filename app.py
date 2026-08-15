# eventlet.monkey_patch() MUST run before any other module (Flask, Werkzeug,
# pydantic via groq, etc.) is imported, or eventlet can't make their I/O
# cooperative — this was previously happening too late, which was throwing
# a monkey-patch exception on every worker boot and contributing to the
# event loop stalls that triggered gunicorn's WORKER TIMEOUT/SIGKILL cycle.
import eventlet
eventlet.monkey_patch()

from flask import Flask, render_template, Response, jsonify, request, session, redirect, url_for
import cv2
import mediapipe as mp
import os
import logging
import sqlite3
import time
from datetime import datetime, date,timedelta
from flask_socketio import SocketIO, emit
from groq import Groq
import numpy as np  
import base64
import math
from werkzeug.security import generate_password_hash, check_password_hash
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.getLogger('absl').setLevel(logging.ERROR)

app = Flask(__name__)

# SECRET_KEY must come from the environment in production. Fail loudly
# instead of silently deploying with a publicly-known key.
SECRET_KEY = os.environ.get("SECRET_KEY")
if not SECRET_KEY:
    raise RuntimeError(
        "SECRET_KEY environment variable is not set. "
        "Generate one with: python -c \"import secrets; print(secrets.token_hex(32))\" "
        "and set it in your .env file (locally) or host's environment variables (in production)."
    )
app.secret_key = SECRET_KEY
app.static_folder = 'static'

# Restrict this to your real frontend origin(s) in production instead of "*"
CORS_ALLOWED_ORIGINS = os.environ.get("CORS_ALLOWED_ORIGINS", "*")
# logger/engineio_logger=True prints the real reason for every connect/
# disconnect/error to stdout, which shows up in Render's Logs tab — turn
# this off again once the connection issue is diagnosed and fixed, since
# it's noisy for normal operation.
socketio = SocketIO(
    app,
    cors_allowed_origins=CORS_ALLOWED_ORIGINS,
    logger=True,
    engineio_logger=True,
    ping_timeout=60,
    ping_interval=25,
)

DB_PATH = os.environ.get("DB_PATH", "attendance.db")
SPEECH_DB_PATH = os.environ.get("SPEECH_DB_PATH", "speech.db")

DEBUG = os.environ.get("FLASK_DEBUG", "false").lower() == "true"

LOCKED_ABSENT = "LOCKED ABSENT"

# In-memory per-user live state (face/expression/gesture/filter/speech/etc).
# Keyed by username so concurrent users never see or overwrite each other's
# live camera/mic status — DB writes (attendance, speech records) are also
# always tagged with the acting user's own username, never a shared global.
# NOTE: this lives in process memory, so it resets on restart/redeploy and
# won't be shared across multiple server instances/workers. Fine for a
# single-worker deployment (this project's Procfile uses `-w 1`); a
# multi-worker or multi-instance deployment would need this moved to a
# shared store (e.g. Redis) instead.
user_states = {}

def get_user_state(username):
    """Return (creating if needed) the live state dict for this username."""
    today_str = date.today().isoformat()
    state = user_states.get(username)
    if state is None or state.get('session_date') != today_str:
        state = {
            'user_id': username,
            'face_detected': False,
            'expression': 'neutral',
            'gesture': 'none',
            'current_filter': 'normal',
            'attendance_status': 'Absent',
            'present_count': 0,
            'absent_count': 0,
            'is_listening': False,
            'current_speech_text': '',
            'session_date': today_str,
        }
        user_states[username] = state
    return state

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS attendance
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  userid TEXT NOT NULL,
                  date TEXT NOT NULL,
                  status TEXT NOT NULL,
                  last_updated TEXT NOT NULL,
                  is_locked INTEGER DEFAULT 0)''')
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  username TEXT NOT NULL UNIQUE,
                  email TEXT NOT NULL UNIQUE,
                  password_hash TEXT NOT NULL,
                  created_at TEXT NOT NULL)''')
    conn.commit()
    conn.close()

def init_speech_db():
    conn = sqlite3.connect(SPEECH_DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS speech_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            date TEXT NOT NULL,
            time TEXT NOT NULL,
            text_content TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

def set_attendance(userid, status):
    today_str = date.today().isoformat()
    now_str = datetime.now().isoformat(timespec='seconds')
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT COUNT(*) FROM attendance WHERE userid=? AND status="Absent"', (userid,))
    total_absent = c.fetchone()[0]
    c.execute('SELECT id, status, is_locked FROM attendance WHERE userid=? AND date=?', (userid, today_str))
    row = c.fetchone()
    # if total_absent >= 3:
    #     if row:
    #         c.execute('UPDATE attendance SET status="Absent", is_locked=1, last_updated=? WHERE id=?', (now_str, row[0]))
    #     else:
    #         c.execute('INSERT INTO attendance (userid, date, status, last_updated, is_locked) VALUES (?, ?, "Absent", ?, 1)',
    #                   (userid, today_str, now_str))
    # print(f"🔒 LOCKED {userid} - {total_absent} total absences!")
    # conn.commit()
    # conn.close()
    # return False
    if row:
        record_id, _, is_locked = row
        if is_locked == 1:
            conn.close()
            return False
        c.execute('UPDATE attendance SET status=?, last_updated=? WHERE id=?', (status, now_str, record_id))
    else:
        c.execute('INSERT INTO attendance (userid, date, status, last_updated) VALUES (?, ?, ?, ?)',
                  (userid, today_str, status, now_str))
    conn.commit()
    conn.close()
    return True

def save_speech_record(user_id, text_content):
    now = datetime.now()
    conn = sqlite3.connect(SPEECH_DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO speech_records (user_id, date, time, text_content) VALUES (?, ?, ?, ?)",
              (user_id, now.date().isoformat(), now.time().isoformat(timespec='seconds'), text_content))
    conn.commit()
    conn.close()

def get_attendance_counts(userid):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT COUNT(*) FROM attendance WHERE userid=? AND status="Present"', (userid,))
    present = c.fetchone()[0]
    c.execute('SELECT COUNT(*) FROM attendance WHERE userid=? AND status="Absent"', (userid,))
    absent = c.fetchone()[0]
    c.execute('SELECT COUNT(*) FROM attendance WHERE userid=? AND is_locked=1', (userid,))
    locked = c.fetchone()[0]
    conn.close()
    return present, absent, locked

def get_user_by_username(username):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT id, username, email, password_hash FROM users WHERE username=?', (username,))
    row = c.fetchone()
    conn.close()
    return row

def create_user(username, email, password):
    password_hash = generate_password_hash(password)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        'INSERT INTO users (username, email, password_hash, created_at) VALUES (?, ?, ?, ?)',
        (username, email, password_hash, datetime.now().isoformat(timespec='seconds'))
    )
    conn.commit()
    conn.close()

# Speech-to-text now happens in the VISITOR'S BROWSER using the Web Speech
# API (see the 'speech_text' Socket.IO handler below and index.html), so
# there's no server-side microphone loop anymore.

# Initialize databases
init_db()
init_speech_db()

# ML setup
# The webcam itself is now captured in the VISITOR'S BROWSER via getUserMedia
# and frames are streamed to this server over Socket.IO — no server-side
# cv2.VideoCapture() needed anymore, so this works on any cloud host.
DETECT_WIDTH, DETECT_HEIGHT = 320, 240
DISPLAY_WIDTH, DISPLAY_HEIGHT = 640, 480

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, model_complexity=0, max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Static config (not per-user)
filters = ["normal", "bw", "red", "blur", "cartoon"]

def login_required(view_func):
    """Redirect to /login if there's no authenticated user in the session."""
    from functools import wraps
    @wraps(view_func)
    def wrapped(*args, **kwargs):
        if not session.get('user_id'):
            return redirect('/login')
        return view_func(*args, **kwargs)
    return wrapped

# Your existing functions (fingers_up, detect_gesture, filters - unchanged)
def fingers_up(hand, hand_label):
    tips = [4, 8, 12, 16, 20]
    fingers = []
    if hand_label == "Right":
        fingers.append(hand.landmark[tips[0]].x < hand.landmark[tips[0]-1].x)
    else:
        fingers.append(hand.landmark[tips[0]].x > hand.landmark[tips[0]-1].x)
    for i in range(1, 5):
        fingers.append(hand.landmark[tips[i]].y < hand.landmark[tips[i]-2].y)
    return fingers

def detect_gesture(hand, hand_label):
    f = fingers_up(hand, hand_label)
    if f == [0,0,0,0,0]: return "✊"
    if f == [1,1,1,1,1]: return "🤚"
    if f == [1,0,0,0,0]: return "👍"
    if f == [0,1,1,0,0]: return "✌️"
    if f == [0,1,0,0,0]: return "☝️"
    if f == [0,1,1,1,0]: return "🤟"
    if f[0] == 1 and hand.landmark[4].y > hand.landmark[3].y:
        return "👎"
    thumb = hand.landmark[4]
    index = hand.landmark[8]
    dist = math.hypot(thumb.x-index.x, thumb.y-index.y)
    if dist < 0.04:
        return "👌"
    return "none"

def filter_bw(frame):
    return cv2.cvtColor(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)

def filter_red(frame):
    red = frame.copy()
    red[:,:,2] = cv2.add(red[:,:,2], 60)
    return red

def filter_blur(frame):
    return cv2.GaussianBlur(frame, (21,21), 0)

def filter_cartoon(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
    color = cv2.bilateralFilter(frame, 9, 300, 300)
    return cv2.bitwise_and(color, color, mask=edges)

def process_frame(frame, state):
    """
    Run face/expression/gesture detection + the active filter on a single
    frame that was captured in the browser (getUserMedia) and sent to us.
    Mutates the given per-user `state` dict (from get_user_state()) instead
    of module-level globals, so each logged-in user's live detection status
    is fully isolated from every other concurrently connected user.
    Returns the annotated frame.
    """
    small_frame = cv2.resize(frame, (DETECT_WIDTH, DETECT_HEIGHT))
    gray_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
    rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    faces = face_cascade.detectMultiScale(gray_small, 1.3, 5)
    state['face_detected'] = len(faces) > 0
    state['expression'] = "neutral"

    frame_h, frame_w = frame.shape[:2]
    for (x, y, w, h) in faces:
        x, y, w, h = int(x*frame_w/DETECT_WIDTH), int(y*frame_h/DETECT_HEIGHT), \
                    int(w*frame_w/DETECT_WIDTH), int(h*frame_h/DETECT_HEIGHT)
        roi_gray = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
        smiles = smile_cascade.detectMultiScale(roi_gray, 1.8, 20)
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.3, 10)

        if len(smiles)>0: state['expression'] = "Smile 😊"
        elif len(eyes)==0: state['expression'] = "Angry 😠"
        elif len(eyes)==1: state['expression'] = "Sad 😞"
        elif len(eyes)>=2 and h>180: state['expression'] = "Stunned 😲"
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)

    new_status = "Present" if state['face_detected'] else "Absent"
    if new_status == "Present":
        state['present_count'] += 1
    else:
        state['absent_count'] += 1

    user_id = state['user_id']
    if state['absent_count'] >= 5000 and state['attendance_status'] != LOCKED_ABSENT:
        set_attendance(user_id, "Absent")
        state['attendance_status'] = LOCKED_ABSENT
    elif new_status != state['attendance_status'] and state['attendance_status'] != LOCKED_ABSENT:
        updated = set_attendance(user_id, new_status)
        if updated:
            state['attendance_status'] = new_status

    # Yield to eventlet's event loop between CPU-heavy stages so the
    # Socket.IO ping/pong heartbeat doesn't get starved by this frame's
    # processing (which previously caused false-positive disconnects).
    eventlet.sleep(0)

    # Hand detection
    result = hands.process(rgb_small)
    state['gesture'] = "none"
    if result.multi_hand_landmarks and result.multi_handedness:
        for hand_landmarks, handedness in zip(result.multi_hand_landmarks, result.multi_handedness):
            hand_label = handedness.classification[0].label
            g = detect_gesture(hand_landmarks, hand_label)
            state['gesture'] = f"{hand_label}:{g}"
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    eventlet.sleep(0)

    # Apply filters
    current_filter = state['current_filter']
    if current_filter == "bw":
        frame = filter_bw(frame)
    elif current_filter == "red":
        frame = filter_red(frame)
    elif current_filter == "blur":
        frame = filter_blur(frame)
    elif current_filter == "cartoon":
        frame = filter_cartoon(frame)

    cv2.putText(frame, f"Mic: {'ON' if state['is_listening'] else 'OFF'}", (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 1), 2)

    return frame

# Routes
@app.route('/')
def index():
    return render_template('front_page.html')

@app.route('/signup', methods=['GET'])
def signup_page():
    return render_template('signup_page.html')

@app.route('/login', methods=['GET'])
def login_page():
    return render_template('middle_page.html')

@app.route('/attendance-all')
def attendance_all():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT id, userid, date, status, last_updated, is_locked FROM attendance ORDER BY date DESC, id DESC LIMIT 100')
    rows = c.fetchall()
    conn.close()
    html = '<!DOCTYPE html><html><head><title>Attendance Records</title><style>body{font-family:Arial;margin:40px;background:#f5f5f5;}table{width:100%;border-collapse:collapse;background:white;box-shadow:0 4px 12px rgba(0,0,0,0.1);}th,td{padding:12px;text-align:left;border-bottom:1px solid #eee;}th{background:linear-gradient(135deg,#28a745,#20c997);color:white;}.present{background:#d4edda;}.absent{background:#f8d7da;}.locked{background:#fff3cd;}</style></head><body>'
    html += '<h2>📋 All Attendance Records</h2>'
    html += '<table><tr><th>ID</th><th>User</th><th>Date</th><th>Status</th><th>Time</th><th>Lock</th></tr>'
    for r in rows:
        status_class = 'present' if r[3] == 'Present' else 'absent'
        lock = '🔒 LOCKED' if r[5] else ''
        html += f'<tr class="{status_class}"><td>{r[0]}</td><td>{r[1]}</td><td>{r[2]}</td><td>{r[3]}</td><td>{r[4][11:]}</td><td>{lock}</td></tr>'
    return html


@app.route('/signup', methods=['POST'])
def signup():
    if not request.form:
        return jsonify({'success': False, 'message': '❌ No form data received!'}), 400

    username = (request.form.get('username') or '').strip()
    email = (request.form.get('email') or '').strip()
    password = request.form.get('password') or ''

    if not username or not email or not password:
        return jsonify({'success': False, 'message': '❌ Username, email, and password are all required!'}), 400

    if len(username) < 3:
        return jsonify({'success': False, 'message': '❌ Username must be at least 3 characters!'}), 400

    if '@' not in email or '.' not in email.split('@')[-1]:
        return jsonify({'success': False, 'message': '❌ Please enter a valid email!'}), 400

    if len(password) < 6:
        return jsonify({'success': False, 'message': '❌ Password must be at least 6 characters!'}), 400

    try:
        create_user(username, email, password)
    except sqlite3.IntegrityError:
        return jsonify({'success': False, 'message': '❌ That username or email is already registered!'}), 400

    row = get_user_by_username(username)
    session['user_id'] = row[0]
    session['username'] = row[1]

    print(f"✅ New user registered: {username}")
    return jsonify({
        'success': True,
        'message': f'🎉 Account created! Welcome, {username}.',
        'redirect': '/dashboard'
    })


@app.route('/login', methods=['POST'])
def do_login():
    if not request.form:
        return jsonify({'success': False, 'message': '❌ No form data received!'}), 400

    username = (request.form.get('username') or '').strip()
    password = request.form.get('password') or ''

    if not username or not password:
        return jsonify({'success': False, 'message': '❌ Username and password are required!'}), 400

    row = get_user_by_username(username)
    if not row or not check_password_hash(row[3], password):
        return jsonify({'success': False, 'message': '❌ Invalid username or password!'}), 401

    session['user_id'] = row[0]
    session['username'] = row[1]

    print(f"✅ Login: {username}")
    return jsonify({
        'success': True,
        'message': f'🎉 Welcome back, {username}!',
        'redirect': '/dashboard'
    })


@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('index.html')

@app.route('/status')
@login_required
def status():
    username = session['username']
    state = get_user_state(username)
    present, absent, locked = get_attendance_counts(username)
    return jsonify({
        'face': state['face_detected'],
        'expression': state['expression'],
        'gesture': state['gesture'],
        'filter': state['current_filter'],
        'attendance': state['attendance_status'],
        'speech': state['current_speech_text'],
        'listening': state['is_listening'],
        'user': username,
        'present_count': state['present_count'],
        'absent_count': state['absent_count'],
        'total_present': present,
        'total_absent': absent,
        'locked': locked > 0,
        'verified': True
    })

@app.route('/logout')
def logout():
    session.clear()
    return redirect('/login')

@app.route('/toggle-speech', methods=['POST'])
@login_required
def toggle_speech():
    state = get_user_state(session['username'])
    state['is_listening'] = not state['is_listening']
    print(f"Speech listening ({session['username']}): {'ON' if state['is_listening'] else 'OFF'}")
    return jsonify({'listening': state['is_listening']})

@app.route('/filter/<name>')
@login_required
def set_filter(name):
    state = get_user_state(session['username'])
    if name in filters:
        state['current_filter'] = name
    return jsonify({'filter': state['current_filter']})

@app.route('/speech-records')
def speech_records():
    conn = sqlite3.connect(SPEECH_DB_PATH)
    c = conn.cursor()
    c.execute('SELECT id, date, time, text_content FROM speech_records ORDER BY id DESC LIMIT 50')
    rows = c.fetchall()
    conn.close()
    # HTML table generation (unchanged)
    html = '<!DOCTYPE html><html><head><title>Speech Records</title><style>body{font-family:Arial;margin:40px;background:#f5f5f5;}table{width:100%;border-collapse:collapse;background:white;box-shadow:0 4px 12px rgba(0,0,0,0.1);}th,td{padding:12px;text-align:left;border-bottom:1px solid #eee;}th{background:linear-gradient(135deg,#007bff,#0056b3);color:white;}</style></head><body>'
    html += '<h2>🎤 Speech Records (Last 50)</h2>'
    html += '<table><tr><th>ID</th><th>Date</th><th>Time</th><th>Text</th></tr>'
    for r in rows:
        html += f'<tr><td>{r[0]}</td><td>{r[1]}</td><td>{r[2]}</td><td>{r[3][:80]}...</td></tr>'
    html += '</table></body></html>'
    return html

@socketio.on('connect')
def handle_connect():
    """Reject the socket connection outright if there's no logged-in session,
    and log every connect attempt so it's visible in the server logs."""
    username = session.get('username')
    if not username:
        print("⚠️  Socket connect rejected: no authenticated session.")
        return False  # refuses the connection; client gets a 'connect_error'
    print(f"🔌 Socket connected: {username}")

@socketio.on('frame')
def handle_frame(data):
    """
    Receives a single JPEG frame captured client-side via getUserMedia
    (sent as a base64 data URL string), runs the same detection/filter
    pipeline the old server-side camera loop used, and emits the
    annotated frame back to that same client only. Uses the logged-in
    user's own isolated state (see get_user_state) so concurrent users'
    detection results never mix.
    """
    username = session.get('username')
    if not username:
        emit('processing_error', {'stage': 'auth', 'message': 'Not logged in — please refresh and sign in again.'})
        return
    try:
        data_url = data.get('image', '') if isinstance(data, dict) else data
        # Strip the "data:image/jpeg;base64," prefix if present
        if ',' in data_url:
            data_url = data_url.split(',', 1)[1]
        img_bytes = base64.b64decode(data_url)
        np_arr = np.frombuffer(img_bytes, dtype=np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            emit('processing_error', {'stage': 'decode', 'message': 'Server could not decode the video frame.'})
            return

        state = get_user_state(username)
        processed = process_frame(frame, state)

        ok, buffer = cv2.imencode('.jpg', processed, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if not ok:
            emit('processing_error', {'stage': 'encode', 'message': 'Server could not encode the processed frame.'})
            return
        out_b64 = base64.b64encode(buffer).decode('utf-8')
        emit('processed_frame', {'image': f'data:image/jpeg;base64,{out_b64}'})
    except Exception as e:
        import traceback
        traceback.print_exc()  # full traceback in server logs for real debugging
        print(f"Frame processing error ({username}): {e}")
        emit('processing_error', {'stage': 'process', 'message': str(e)})


@socketio.on('speech_text')
def handle_speech_text(data):
    """
    Receives a transcribed phrase from the browser's Web Speech API
    (see index.html) instead of the old server-side sr.Microphone() loop.
    Tagged and stored under the logged-in user's own username.
    """
    username = session.get('username')
    if not username:
        return
    text = (data.get('text', '') if isinstance(data, dict) else str(data)).strip()
    if not text:
        return
    state = get_user_state(username)
    state['current_speech_text'] = text
    save_speech_record(username, text)
    print(f"✅ HEARD ({username}):", text)


@socketio.on('message')
def handle_message(data):
    username = session.get('username')
    if not username:
        emit('response', {'message': "⚠️ Please log in first."})
        return

    state = get_user_state(username)
    user_message = data['message']

    context = f"""
    You are an AI Attendance Assistant for user '{username}'.
    Attendance: {state['attendance_status']}
    Present count: {state['present_count']}
    Absent count: {state['absent_count']}
    Gesture: {state['gesture']}
    Expression: {state['expression']}
    Speech: {state['current_speech_text']}
    """
    
    if client is None:
        emit('response', {'message': "⚠️ Groq API key not configured on the server."})
        return

    try:
        # Groq API Call for ultra-fast response
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",  # Fastest model for <1s responses
            messages=[
                {"role": "system", "content": context},
                {"role": "user", "content": user_message}
            ],
            stream=False # Set to True if you want a typing effect
        )
        
        reply = completion.choices[0].message.content
        emit('response', {'message': reply})
        
    except Exception as e:
        emit('response', {'message': f"Groq AI error: {e}"})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    print("🚀 Smart Attendance System Started!")
    print(f"🌐 Login: http://localhost:{port}/login")
    # debug=True (via FLASK_DEBUG=true) should only ever be used locally.
    socketio.run(app, debug=DEBUG, host='0.0.0.0', port=port)