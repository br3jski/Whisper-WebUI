import os
from flask import Flask, render_template, request, jsonify
from openai import OpenAI
from werkzeug.utils import secure_filename
from werkzeug.serving import run_simple

app = Flask(__name__)
app.secret_key = os.urandom(24)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mp3', 'mp4', 'mpeg', 'mpga', 'm4a', 'wav', 'webm'}
app.config['TIMEOUT'] = 300  # 5 mins
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.after_request
def add_header(response):
    response.headers['Cross-Origin-Opener-Policy'] = 'same-origin'
    response.headers['Cross-Origin-Embedder-Policy'] = 'credentialless'
    return response

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def transcribe_audio(file_path, api_key):
    client = OpenAI(api_key=api_key, timeout=300)  # 5 minut timeout
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

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
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
            else:
                return jsonify({'error': 'Niedozwolony typ pliku.'}), 400
        except Exception as e:
            app.logger.error(f"Wystąpił błąd: {str(e)}")
            return jsonify({'error': f"Wystąpił nieoczekiwany błąd: {str(e)}"}), 500
    return render_template('index.html')

@app.route('/stream', methods=['GET', 'POST'])
def stream():
    if request.method == 'POST':
        try:
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
                
                # Usuń plik po transkrypcji
                os.remove(file_path)
                
                return jsonify({'transcription': transcription})
            else:
                return jsonify({'error': 'Niedozwolony typ pliku.'}), 400
        except Exception as e:
            app.logger.error(f"Wystąpił błąd w /stream: {str(e)}")
            return jsonify({'error': f"Wystąpił nieoczekiwany błąd: {str(e)}"}), 500
    return render_template('stream.html')

@app.route('/summarize', methods=['POST'])
def summarize():
    data = request.json
    transcription = data.get('transcription')
    api_key = data.get('api_key')

    if not transcription or not api_key:
        return jsonify({'error': 'Brak transkrypcji lub klucza API.'}), 400

    client = OpenAI(api_key=api_key)
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "Ty jesteś asystentem, który specjalizuje się w tworzeniu czytelnych i zwięzłych notatek akademickich. Twoim celem jest przekształcenie transkrypcji wykładów lub innych dokumentów akademickich w formę, która ułatwi naukę i przygotowanie do egzaminów. Wspierasz tworzenie map myśli, list punktowanych oraz innych czytelnych formatów. Zachowaj kluczowe informacje i porządkuj dane w logiczne struktury."},
                {"role": "user", "content": f"Jestem w trakcie nauki do egzaminu i potrzebuję czytelnych notatek bazujących na poniższej transkrypcji. Proszę przekształć treść w zwięzłą formę, która będzie przydatna do powtórek, taką jak mapa myśli lub lista punktów. Oto transkrypcja, z której należy stworzyć notatki:\n\n{transcription}"}
            ]
        )
        summary = response.choices[0].message.content
        return jsonify({'summary': summary})
    except Exception as e:
        return jsonify({'error': f"Wystąpił błąd podczas generowania podsumowania: {str(e)}"}), 500
    
@app.errorhandler(Exception)
def handle_exception(e):
    app.logger.error(f"Nieobsłużony wyjątek: {str(e)}")
    return jsonify({'error': 'Wystąpił nieoczekiwany błąd serwera.'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8001)