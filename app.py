import os
from flask import Flask, render_template, request, jsonify
from openai import OpenAI
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = os.urandom(24)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mp3', 'mp4', 'mpeg', 'mpga', 'm4a', 'wav', 'webm'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def transcribe_audio(file_path, api_key):
    client = OpenAI(api_key=api_key)
    try:
        with open(file_path, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="text"
            )
        return transcription
    except Exception as e:
        return f"Wystąpił błąd: {e}"

# Decorator for error handling
def error_handling(f):
    def wrapper(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception as e:
            return jsonify({'error': f"Wystąpił błąd: {str(e)}"}), 500
    # Maintain original function name and docstring
    wrapper.__name__ = f.__name__
    wrapper.__doc__ = f.__doc__
    return wrapper

@app.route('/', methods=['GET', 'POST'])
@error_handling
def index():
    if request.method == 'POST':
        if 'audio' not in request.files:
            return jsonify({'error': 'Brak pliku audio.'}), 400
        file = request.files['audio']
        if file.filename == '':
            return jsonify({'error': 'Nie wybrano pliku.'}), 400
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            api_key = request.form.get('api_key')
            if not api_key:
                return jsonify({'error': 'Brak klucza API.'}), 401

            transcription = transcribe_audio(file_path, api_key)
            return jsonify({'transcription': transcription})
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
@error_handling
def summarize():
    data = request.json
    transcription = data.get('transcription')
    api_key = data.get('api_key')

    if not transcription or not api_key:
        return jsonify({'error': 'Brak transkrypcji lub klucza API.'}), 400

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "Ty jesteś asystentem, który specjalizuje się w tworzeniu czytelnych i zwięzłych notatek akademickich..."},
            {"role": "user", "content": f"Jestem w trakcie nauki do egzaminu i potrzebuję czytelnych notatek... {transcription}"}
        ]
    )
    summary = response.choices[0].message.content
    return jsonify({'summary': summary})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8001)