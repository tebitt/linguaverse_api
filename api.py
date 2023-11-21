from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_pymongo import MongoClient, ObjectId
import openai
import os
import json
import requests
import base64
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google_auth_oauthlib.flow import InstalledAppFlow
import re
from dotenv import load_dotenv
from google.cloud import texttospeech
import whisper
from pydub import AudioSegment
from pydub.playback import play
import io
from langdetect import detect
import shutil
from werkzeug.utils import secure_filename
import sounddevice as sd
import soundfile as sf
import threading


app = Flask(__name__)
CORS(app)
recording_thread = None
recording_flag = threading.Event()

def record_audio(filename='temp/audio.wav', fs=44100, channels=1):
    global recording_flag
    with sf.SoundFile(filename, mode='w', samplerate=fs, channels=channels) as file:
        with sd.InputStream(samplerate=fs, channels=channels, callback=lambda indata, frames, time, status: file.write(indata)):
            recording_flag.wait()  # Wait for stop signal

@app.route('/ai', methods=['GET'])
def ai():
    openai.api_key = os.getenv('OPENAI_API_KEY')
    return openai.Model.list()


@app.route('/start', methods=['GET'])
def start_recording():
    global recording_thread, recording_flag
    if recording_thread is None or not recording_thread.is_alive():
        recording_flag.clear()
        recording_thread = threading.Thread(target=record_audio)
        recording_thread.start()
        return jsonify({"message": "Recording started"}), 200
    else:
        return jsonify({"error": "Recording is already in progress"}), 400

@app.route('/stop', methods=['GET'])
def stop_recording():
    global recording_thread, recording_flag
    if recording_thread is not None and recording_thread.is_alive():
        recording_flag.set()
        recording_thread.join()
        return jsonify({"message": "Recording stopped"}), 200
    else:
        return jsonify({"error": "No recording in progress"}), 400
# Main endpoint for processing chat. It performs several operations
@app.route('/chat', methods=['GET','POST'])
def chat():
    load_dotenv()
    # Connect to MongoDB and fetches the latest image and audio 
    mongo = MongoClient(os.getenv('MONGO_URI'))
    openai.api_key = os.getenv('OPENAI_API_KEY')
    db = mongo['lingua']
    collection = db['preset_log']
    img_collection = db['photo']
    # audio_collection = db['audio']
    url = img_collection.find().sort('_id', -1).limit(1).next() 
    # audio_url = audio_collection.find().sort('_id', -1).limit(1).next() 
    # Extracting file ID from the Google Drive URL
    object_id_str = os.getenv('PRIMING_OBJECT_ID')
    object_id = ObjectId(object_id_str)    
    item = collection.find_one({"_id": object_id})
    item_dict = dict(item) 
    item_dict['_id'] = str(item_dict['_id'])
    messages = json.loads(item_dict['messages'])
    init_text(messages)
    file_id = extract_file_id_from_google_drive_url(url['file_url'])
    # audio_id = extract_file_id_from_google_drive_url(audio_url['file_url'])
    # Creating a direct download URL for the Google Drive file
    file_url = f'https://drive.google.com/uc?id={file_id}&export=download'
    # audio_url = f'https://drive.google.com/uc?id={audio_id}&export=download'
    destination = os.path.join(r'./temp', 'ad.jpg')
    # Downloads the image and audio from Google Drive
    download_file_from_google_drive(file_url, destination)
    # destination = os.path.join(r'./temp', 'audio.wav')
    # download_file_from_google_drive(audio_url, destination)
    # Transcribes the audio file
    # user_input = request.json.get('input') #get input from request
    # shutil.move('../Downloads/audio.wav', 'temp/audio.wav')
    user_input = transcribe('temp/audio.wav')
    print(user_input)
    # Prepare the prompt for the GPT model adding the user's input collected from the transcription
    prepare_prompt(user_input)
    with open('temp/logs.json', 'r') as f:
        logs = json.load(f) 
     # Send the transcription to OpenAI's GPT model for response
    response = send_to_openai(logs)
    logs.append(response['choices'][0]['message'])
    with open('temp/logs.json', 'w') as f:
        json.dump(logs, f, indent=2) 
    answer = response['choices'][0]['message']['content']
    print(answer)
    language = detect(answer)
    # Converts the GPT response to speech using Google Cloud TTS outputing a mp3 file called output.mp3
    text_to_speech(answer, language)
    play_audio('temp/output.mp3')
    return answer + '\n'

# Initialize the chat logs. If the logs file doesn't exist, it creates one
def init_text(message):
    json_message = json.dumps(message)
    if not os.path.exists('temp/logs.json'):
        with open('temp/logs.json', 'w') as f:
            f.write(json_message)
    else: return

# Prepare the prompt for the GPT model. It appends the user's input and image to the chat logs
def prepare_prompt(text):
    with open('temp/logs.json', 'r') as f:
        logs = json.load(f)
    base64_image = encode_image('temp/mechatronics.jpg')
    data = [{"type": "text", "text": text}, {"type": "img_url", "image_url": { "url": f"data:image/png;base64,{base64_image}" }}]
    logs.append({"role": "user", "content": data})
    with open('temp/logs.json', 'w') as f:
        json.dump(logs, f, indent=2) 

# Encode the image at the given path to Base64, a format suitable for embedding in JSON
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
    
# Send the chat log to OpenAI and gets a response from GPT
def send_to_openai(messages):
    """Send messages to OpenAI API and get a response."""
    headers = {
        'Authorization': f'Bearer {os.getenv("OPENAI_API_KEY")}',
        'Content-Type': 'application/json'
    }
    
    payload = {
        "model": "gpt-4-vision-preview",
        "messages": messages,
        "max_tokens": 3000
    }

    response = requests.post(
        'https://api.openai.com/v1/chat/completions',
        headers=headers,
        json=payload
    )
    return response.json()

# Download a file from Google Drive given its URL and saves it to the specified destination
def download_file_from_google_drive(url, destination):
    session = requests.Session()
    response = session.get(url, stream = True)
    token = get_confirm_token(response)
    if token:
        params = { 'confirm' : token }
        response = session.get(url, params = params, stream = True)
    save_response_content(response, destination)   

# Extract a confirmation token from Google Drive's response
# This is needed for downloading files
def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'): return value
    return None

# Extract the file ID from a Google Drive URL
# This ID is used to construct a direct download URL
def extract_file_id_from_google_drive_url(url):
    pattern = r'/d/([0-9A-Za-z_-]{33}|[0-9A-Za-z_-]{19})'
    match = re.search(pattern, url)
    if match: return match.group(1)
    else: return None
    
# Save the content from a Google Drive download response to a file at the specified destination
def save_response_content(response, destination):
    CHUNK_SIZE = 32768
    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: f.write(chunk)

# Transcribe the audio data using Whisper model
# Return the transcribed text
def transcribe(audio_data):
    model = whisper.load_model("base")
    result = model.transcribe(audio_data)
    return result["text"]

# Convert the given text to speech using Google Cloud TTS in the specified language
def text_to_speech(text, language_code):
    # The language code is mapped from a short language code to Google's format
    language_mapping = {
        'af': 'af-ZA', 'ar': 'ar-XA', 'bg': 'bg-BG', 'bn': 'bn-IN', 'ca': 'ca-ES', 'cs': 'cs-CZ',
        'cy': 'en-US', 'da': 'da-DK', 'de': 'de-DE', 'el': 'el-GR', 'en': 'en-US', 'es': 'es-ES',
        'et': 'et-EE', 'fa': 'fa-IR', 'fi': 'fi-FI', 'fr': 'fr-FR', 'gu': 'gu-IN', 'he': 'he-IL',
        'hi': 'hi-IN', 'hr': 'hr-HR', 'hu': 'hu-HU', 'id': 'id-ID', 'it': 'it-IT', 'ja': 'ja-JP',
        'kn': 'kn-IN', 'ko': 'ko-KR', 'lt': 'lt-LT', 'lv': 'lv-LV', 'mk': 'mk-MK', 'ml': 'ml-IN',
        'mr': 'mr-IN', 'ne': 'ne-NP', 'nl': 'nl-NL', 'no': 'nb-NO', 'pa': 'pa-IN', 'pl': 'pl-PL',
        'pt': 'pt-PT', 'ro': 'ro-RO', 'ru': 'ru-RU', 'sk': 'sk-SK', 'sl': 'sl-SI', 'so': 'so-SO',
        'sq': 'sq-AL', 'sv': 'sv-SE', 'sw': 'sw-TZ', 'ta': 'ta-IN', 'te': 'te-IN', 'th': 'th-TH',
        'tl': 'en-US', 'tr': 'tr-TR', 'uk': 'uk-UA', 'ur': 'ur-PK', 'vi': 'vi-VN', 'zh-cn': 'cmn-CN',
        'zh-tw': 'cmn-TW'
    }
    language_code = language_mapping[language_code]
    client = texttospeech.TextToSpeechClient()
    synthesis_input = texttospeech.SynthesisInput(text=text)
    voice = texttospeech.VoiceSelectionParams(language_code=language_code, ssml_gender=texttospeech.SsmlVoiceGender.FEMALE)
    audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)
    response = client.synthesize_speech(input=synthesis_input, voice=voice, audio_config=audio_config)
    with open("temp/output.mp3", "wb") as out:
        out.write(response.audio_content)
        print("Audio content written to file 'output.mp3'")
    
# Play the audio file located at the given file path using Pydub
def play_audio(file_path):
    audio = AudioSegment.from_mp3(file_path)
    play(audio)
    # After playing, it cleans up by deleting the original audio and the output MP3 file
    if os.path.exists('temp/audio.wav'):
        os.remove('temp/audio.wav')
    if os.path.exists('temp/output.mp3'):
        os.remove('temp/output.mp3')

# Clean up and delete temporary files used during the chat session
@app.route('/close', methods=['GET'])
def close():
    if os.path.exists('temp/logs.json'):
        os.remove('temp/logs.json')
    # if os.path.exists('temp/ad.jpg'):
    #     os.remove('temp/ad.jpg')
    return 'Files deleted\n'

# Main entry point for the Flask application
# Create a 'temp' directory if it doesn't exist and starts the Flask app
if __name__ == '__main__':
    if not os.path.exists('temp'):
        os.makedirs('temp')
    if os.path.exists('temp/logs.json'):
        os.remove('temp/logs.json')
    # if os.path.exists('temp/ad.jpg'):
    #     os.remove('temp/ad.jpg')
    app.run(port='6969', debug=True)