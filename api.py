from flask import Flask, request
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

app = Flask(__name__)
CORS(app)

@app.route('/ai', methods=['GET'])
def ai():
    openai.api_key = os.getenv('OPENAI_API_KEY')
    return openai.Model.list()

@app.route('/chat', methods=['POST'])
def chat():
    load_dotenv()
    mongo = MongoClient(os.getenv('MONGO_URI'))
    openai.api_key = os.getenv('OPENAI_API_KEY')
    user_input = request.json.get('input') #get input from request
    db = mongo['lingua']
    collection = db['preset_log']
    img_collection = db['photo']
    url = img_collection.find().sort('_id', -1).limit(1).next() 
    # Extracting file ID from the Google Drive URL
    object_id_str = os.getenv('PRIMING_OBJECT_ID')
    object_id = ObjectId(object_id_str)    
    item = collection.find_one({"_id": object_id})
    item_dict = dict(item) # convert to dictionary
    item_dict['_id'] = str(item_dict['_id']) # convert ObjectId to string
    messages = json.loads(item_dict['messages'])
    init_text(messages)
    file_id = extract_file_id_from_google_drive_url(url['file_url'])
    # Creating a direct download URL for the Google Drive file
    file_url = f'https://drive.google.com/uc?id={file_id}&export=download'
    destination = os.path.join(r'./temp', 'ad.jpg')
    download_file_from_google_drive(file_url, destination)

    # The path where you want to save the downloaded file, inside the "photo" folder
    prepare_prompt(user_input)
    with open('temp/logs.json', 'r') as f:
        logs = json.load(f) 
    response = send_to_openai(logs)
    logs.append(response['choices'][0]['message'])
    with open('temp/logs.json', 'w') as f:
        json.dump(logs, f, indent=2) 
    return response['choices'][0]['message']['content']+'\n'

def init_text(message):
    # Assuming message is already a dictionary or list that json.dumps can serialize.
    json_message = json.dumps(message)  # This will convert single quotes to double quotes.
    
    if not os.path.exists('temp/logs.json'):
        with open('temp/logs.json', 'w') as f:
            f.write(json_message)
    else:
        return

def prepare_prompt(text):
    with open('temp/logs.json', 'r') as f:
        logs = json.load(f)  # Load the list of logs
    base64_image = encode_image('temp/ad.jpg')
    data = [{"type": "text", "text": text}, {"type": "img_url", "image_url": { "url": f"data:image/png;base64,{base64_image}" }}]
    logs.append({"role": "user", "content": data})
    # Write the updated list back to the file
    with open('temp/logs.json', 'w') as f:
        json.dump(logs, f, indent=2) 

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
    
def send_to_openai(messages):
    """Send messages to OpenAI API and get a response."""
    headers = {
        'Authorization': f'Bearer {os.getenv("OPENAI_API_KEY")}',
        'Content-Type': 'application/json'
    }
    
    payload = {
        "model": "gpt-4-vision-preview",  # Adjust the model according to your needs
        "messages": messages,
        "max_tokens": 3000
    }
    
    response = requests.post(
        'https://api.openai.com/v1/chat/completions',
        headers=headers,
        json=payload
    )
    
    return response.json()

def download_file_from_google_drive(url, destination):
    session = requests.Session()

    response = session.get(url, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'confirm' : token }
        response = session.get(url, params = params, stream = True)

    save_response_content(response, destination)   

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def extract_file_id_from_google_drive_url(url):
    pattern = r'/d/([0-9A-Za-z_-]{33}|[0-9A-Za-z_-]{19})'
    match = re.search(pattern, url)

    if match:
        return match.group(1)
    else:
        return None
    
def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)




@app.route('/close', methods=['GET'])
def close():
    if os.path.exists('temp/logs.json'):
        os.remove('temp/logs.json')
    if os.path.exists('temp/ad.jpg'):
        os.remove('temp/ad.jpg')
    return 'Files deleted\n'

if __name__ == '__main__':
    if not os.path.exists('temp'):
        os.makedirs('temp')
    app.run(port='6969', debug=True)