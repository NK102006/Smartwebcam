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
import random
import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
otp_storage = {}
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
socketio = SocketIO(app, cors_allowed_origins=CORS_ALLOWED_ORIGINS)

DB_PATH = os.environ.get("DB_PATH", "attendance.db")
SPEECH_DB_PATH = os.environ.get("SPEECH_DB_PATH", "speech.db")

DEBUG = os.environ.get("FLASK_DEBUG", "false").lower() == "true"

# OTP Configuration - set these in your .env file / host's env vars
OTP_SENDER_EMAIL = os.environ.get("OTP_SENDER_EMAIL")
OTP_SENDER_PASSWORD = os.environ.get("OTP_SENDER_PASSWORD")
if not OTP_SENDER_EMAIL or not OTP_SENDER_PASSWORD:
    print("⚠️  OTP_SENDER_EMAIL / OTP_SENDER_PASSWORD not set — OTP emails will fail to send.")

is_listening = False
current_speech_text = ""
present_count = 0
absent_count = 0
LOCKED_ABSENT = "LOCKED ABSENT"

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

def send_otp_email(email, otp):
    try:
        msg = MIMEMultipart()
        msg['From'] = OTP_SENDER_EMAIL
        msg['To'] = email
        msg['Subject'] = '🔐 Smart Attendance OTP'
        msg.attach(MIMEText(f'Your OTP is: **{otp}**\n\nExpires in 5 minutes.', 'plain'))
        
        context = ssl.create_default_context()
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls(context=context)
            server.login(OTP_SENDER_EMAIL, OTP_SENDER_PASSWORD)  # App password here
            server.sendmail(OTP_SENDER_EMAIL, email, msg.as_string())
        print(f"✅ Email sent!")
        return True
    except Exception as e:
        print(f"❌ Email error: {e}")
        return False

def generate_otp():
    """Generate 6-digit random OTP"""
    return str(random.randint(100000, 999999))

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
hands = mp_hands.Hands(min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Global state
face_detected = False
expression = "neutral"
gesture = "none"
current_filter = "normal"
filters = ["normal", "bw", "red", "blur", "cartoon"]
CURRENT_USERID='' 
attendance_status = "Absent"

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

_session_start = None

def process_frame(frame):
    """
    Run face/expression/gesture detection + the active filter on a single
    frame that was captured in the browser (getUserMedia) and sent to us.
    Mutates the global status variables and returns the annotated frame,
    same logic as the original generate_frames(), just driven by an
    incoming frame instead of a local cv2.VideoCapture() loop.
    """
    global face_detected, expression, gesture, current_filter, attendance_status
    global present_count, absent_count, _session_start

    today_str = date.today().isoformat()
    if _session_start != today_str:
        present_count = 0
        absent_count = 0
        _session_start = today_str

    small_frame = cv2.resize(frame, (DETECT_WIDTH, DETECT_HEIGHT))
    gray_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
    rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    faces = face_cascade.detectMultiScale(gray_small, 1.3, 5)
    face_detected = len(faces) > 0
    expression = "neutral"

    frame_h, frame_w = frame.shape[:2]
    for (x, y, w, h) in faces:
        x, y, w, h = int(x*frame_w/DETECT_WIDTH), int(y*frame_h/DETECT_HEIGHT), \
                    int(w*frame_w/DETECT_WIDTH), int(h*frame_h/DETECT_HEIGHT)
        roi_gray = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
        smiles = smile_cascade.detectMultiScale(roi_gray, 1.8, 20)
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.3, 10)

        if len(smiles)>0: expression = "Smile 😊"
        elif len(eyes)==0: expression = "Angry 😠"
        elif len(eyes)==1: expression = "Sad 😞"
        elif len(eyes)>=2 and h>180: expression = "Stunned 😲"
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)

    new_status = "Present" if face_detected else "Absent"
    if new_status == "Present":
        present_count += 1
    else:
        absent_count += 1

    if absent_count >= 5000 and attendance_status != LOCKED_ABSENT:
        set_attendance(CURRENT_USERID, "Absent")
        attendance_status = LOCKED_ABSENT
    elif new_status != attendance_status and attendance_status != LOCKED_ABSENT:
        updated = set_attendance(CURRENT_USERID, new_status)
        if updated:
            attendance_status = new_status

    # Hand detection
    result = hands.process(rgb_small)
    gesture = "none"
    if result.multi_hand_landmarks and result.multi_handedness:
        for hand_landmarks, handedness in zip(result.multi_hand_landmarks, result.multi_handedness):
            hand_label = handedness.classification[0].label
            gesture = detect_gesture(hand_landmarks, hand_label)
            gesture = f"{hand_label}:{gesture}"
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Apply filters
    if current_filter == "bw":
        frame = filter_bw(frame)
    elif current_filter == "red":
        frame = filter_red(frame)
    elif current_filter == "blur":
        frame = filter_blur(frame)
    elif current_filter == "cartoon":
        frame = filter_cartoon(frame)

    cv2.putText(frame, f"Mic: {'ON' if is_listening else 'OFF'}", (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 1), 2)

    return frame

# Routes
@app.route('/')
def index():
    return render_template('front_page.html')

@app.route('/login')
def login_page():
    return render_template('middle_page.html')

@app.route('/otp')
def otp_page():
    if 'otp_verified' not in session:
        return render_template('otp_page.html', error="Please enter your email first!")
    return render_template('otp_page.html')

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


@app.route('/send-otp', methods=['POST'])
def send_otp():
    # ✅ SAFE WAY - Check if form data exists first
    if not request.form:
        return jsonify({
            'success': False, 
            'message': '❌ No form data received!'
        }), 400
    
    email = request.form.get("email")
    
    # ✅ NULL CHECK BEFORE STRIP
    if not email:  # None, empty string, or whitespace
        return jsonify({
            'success': False, 
            'message': '❌ Please enter your email address!'
        }), 400
    
    email = email.strip()
    
    if '@' not in email or not email.endswith(('.com', '.in', '.org', '.edu')):
        return jsonify({
            'success': False, 
            'message': '❌ Please enter a valid email!'
        }), 400
    
    # Generate OTP
    otp = generate_otp()
    
    # Store safely
    session['otp'] = otp
    session['email'] = email
    session['otp_time'] = time.time()
    session['otp_attempts'] = 0

    username = email.split("@")[0]
    # session['username']=username
    global CURRENT_USERID
    CURRENT_USERID=username
    
    print(f"🔢 Generated OTP: {otp} for {email}")
    
    # Send email
    if send_otp_email(email, otp):
        return jsonify({
            'success': True, 
            'message': f'✅ OTP sent to {email}! Check inbox/spam.',
            'redirect': '/otp'
        })
    else:
        session.clear()  # Clean up on failure
        return jsonify({
            'success': False, 
            'message': '❌ Email failed. Check Gmail App Password!'
        }), 500

@app.route('/verify-otp', methods=['POST'])
def verify_otp():
    # Check if form data exists
    if not request.form:
        return jsonify({
            'success': False, 
            'message': '❌ No form data received!'
        }), 400
    
    # Get and validate OTP
    user_otp = request.form.get("otp")
    if not user_otp:
        return jsonify({
            'success': False, 
            'message': '❌ Please enter OTP!'
        }), 400
    
    user_otp = user_otp.strip()
    if len(user_otp) != 6 or not user_otp.isdigit():
        return jsonify({
            'success': False, 
            'message': '❌ OTP must be 6 digits!'
        }), 400
    
    # Get stored data
    email = session.get('email')
    stored_otp = session.get('otp')
    
    if not email or not stored_otp:
        return jsonify({
            'success': False, 
            'message': '❌ Session expired. Please resend OTP!'
        }), 400
    
    # Check expiration (5 minutes = 300 seconds)
    otp_age = time.time() - session.get('otp_time', 0)
    attempts = session.get('otp_attempts', 0)
    
    if otp_age > 300:
        session.clear()
        return jsonify({
            'success': False, 
            'message': '⏰ OTP expired! Click RESEND.',
            'expired': True
        }), 400
    
    if attempts >= 3:
        session.clear()
        return jsonify({
            'success': False, 
            'message': '❌ Too many failed attempts!'
        }), 400
    
    # ✅ SUCCESS CHECK
    if user_otp == stored_otp:
        session['otp_verified'] = True
        session['verified_email'] = email
        session['login_time'] = time.time()
        
        # Clean up
        session.pop('otp', None)
        session.pop('otp_time', None)
        session.pop('otp_attempts', None)
        
        print(f"✅ OTP verified for {email}")
        return jsonify({
            'success': True, 
            'message': '🎉 Verification successful!',
            'redirect': '/dashboard'
        })
    
    # ❌ FAILED ATTEMPT
    attempts += 1
    session['otp_attempts'] = attempts
    remaining = 3 - attempts
    
    return jsonify({
        'success': False, 
        'message': f'❌ Wrong OTP! {remaining} attempts left.',
        'attempts_left': remaining
    })

# ADD THIS MISSING resend_otp ROUTE
@app.route('/resend-otp', methods=['POST'])
def resend_otp():
    if not request.form:
        return jsonify({'success': False, 'message': '❌ No form data!'}), 400
    
    email = request.form.get('email') or session.get('email')
    if not email:
        return jsonify({'success': False, 'message': '❌ No email found!'}), 400
    
    # Clear old data
    session.pop('otp', None)
    session.pop('otp_time', None)
    session.pop('otp_attempts', None)
    
    # Generate new OTP
    otp = generate_otp()
    session['otp'] = otp
    session['email'] = email
    session['otp_time'] = time.time()
    session['otp_attempts'] = 0
    
    
    print(f"🔄 RESENT OTP: {otp} for {email}")
    
    if send_otp_email(email, otp):
        return jsonify({
            'success': True, 
            'message': f'✅ New OTP sent to {email}!',
            'redirect': '/otp'
        })
    else:
        return jsonify({
            'success': False, 
            'message': '❌ Failed to send OTP!'
        }), 500


@app.route('/dashboard')
def dashboard():
    if not session.get('otp_verified'):
        return redirect('/login')
    return render_template('index.html')

@app.route('/status')
def status():
    present, absent, locked = get_attendance_counts(CURRENT_USERID)
    return jsonify({
        'face': face_detected,
        'expression': expression,
        'gesture': gesture,
        'filter': current_filter,
        'attendance': attendance_status,
        'speech': current_speech_text,
        'listening': is_listening,
        'user': CURRENT_USERID,
        'present_count': present_count,
        'absent_count': absent_count,
        'total_present': present,
        'total_absent': absent,
        'locked': locked > 0,
        'verified': session.get('otp_verified', False)
    })

@app.route('/logout')
def logout():
    session.clear()
    return redirect('/login')

# Other routes unchanged...
@app.route('/toggle-speech', methods=['POST'])
def toggle_speech():
    global is_listening
    is_listening = not is_listening
    print(f"Speech listening: {'ON' if is_listening else 'OFF'}")
    return jsonify({'listening': is_listening})

@app.route('/filter/<name>')
def set_filter(name):
    global current_filter
    if name in filters:
        current_filter = name
    return jsonify({'filter': current_filter})

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

@socketio.on('frame')
def handle_frame(data):
    """
    Receives a single JPEG frame captured client-side via getUserMedia
    (sent as a base64 data URL string), runs the same detection/filter
    pipeline the old server-side camera loop used, and emits the
    annotated frame back to that same client only.
    """
    try:
        data_url = data.get('image', '') if isinstance(data, dict) else data
        # Strip the "data:image/jpeg;base64," prefix if present
        if ',' in data_url:
            data_url = data_url.split(',', 1)[1]
        img_bytes = base64.b64decode(data_url)
        np_arr = np.frombuffer(img_bytes, dtype=np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            return

        processed = process_frame(frame)

        ok, buffer = cv2.imencode('.jpg', processed, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if not ok:
            return
        out_b64 = base64.b64encode(buffer).decode('utf-8')
        emit('processed_frame', {'image': f'data:image/jpeg;base64,{out_b64}'})
    except Exception as e:
        print(f"Frame processing error: {e}")


@socketio.on('speech_text')
def handle_speech_text(data):
    """
    Receives a transcribed phrase from the browser's Web Speech API
    (see index.html) instead of the old server-side sr.Microphone() loop.
    """
    global current_speech_text
    text = (data.get('text', '') if isinstance(data, dict) else str(data)).strip()
    if not text:
        return
    current_speech_text = text
    user_id = CURRENT_USERID or "unknown_user"
    save_speech_record(user_id, text)
    print(f"✅ HEARD ({user_id}):", text)


@socketio.on('message')
def handle_message(data):
    user_message = data['message']
    
    # Context string you already have in your app.py
    context = f"""
    You are an AI Attendance Assistant.
    Attendance: {attendance_status}
    Present count: {present_count}
    Absent count: {absent_count}
    Gesture: {gesture}
    Expression: {expression}
    Speech: {current_speech_text}
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
    print("🚀 Smart Attendance System with OTP Started!")
    print(f"🌐 Login: http://localhost:{port}/login")
    # debug=True (via FLASK_DEBUG=true) should only ever be used locally.
    socketio.run(app, debug=DEBUG, host='0.0.0.0', port=port)