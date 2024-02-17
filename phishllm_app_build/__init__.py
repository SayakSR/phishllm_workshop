from flask import Flask, request, jsonify, render_template
import requests
import random
import time
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from bs4 import BeautifulSoup
import csv
from urllib.parse import urlparse
import pytz
from datetime import datetime
import socket

app = Flask(__name__)

whitelist_domains = set()

def read_whitelist(file_name):
    
    try:
        with open(file_name, mode='r', newline='', encoding='utf-8') as file:
            reader = csv.reader(file)
            for row in reader:
                whitelist_domains.update(row)
    except FileNotFoundError:
        print(f"File {file_name} not found.")

def read_recent_urls(file_name):
   
    recent_urls = []
    try:
        with open(file_name, 'r') as file:
            for line in file:
                parts = line.strip().split(',')
                if len(parts) == 2:
                    url, age = parts[0].strip('[]'), parts[1]
                    recent_urls.append({'url': url, 'age': age})
    except FileNotFoundError:
        print(f"File {file_name} not found.")
    return list(reversed(recent_urls))

def get_domain_from_url(url):
    netloc = urlparse(url).netloc
    domain_parts = netloc.split('.')
    if domain_parts[0] == 'www':
        domain_parts.pop(0)
    return '.'.join(domain_parts[-2:])

def is_domain_whitelisted(domain):
    return domain in whitelist_domains

def load_model(model_path):
    model = DistilBertForSequenceClassification.from_pretrained(model_path)
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    return model, tokenizer

def predict(text, model, tokenizer):
    inputs = tokenizer(text=text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        logits = model(**inputs).logits
    scores = torch.softmax(logits, dim=1)
    prediction = scores.argmax().item()
    prediction_score = scores[0][prediction].item()
    return "phishing" if prediction == 0 else "benign", prediction_score

def html_to_structured_text(html_content): # Ummm... This function is not JSON parsing!
    soup = BeautifulSoup(html_content, 'html.parser')
    structured_data = []
    for element in soup.find_all(['h1', 'h2', 'h3', 'p', 'a', 'ul', 'ol', 'li']):
        structured_data.append(element.get_text(strip=True))
    return " ".join(structured_data)

def fetch_html(url):
    headers = {
        'User-Agent': random.choice([
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0.3 Safari/605.1.15',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:88.0) Gecko/20100101 Firefox/88.0'
        ])
    }
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    return response.text

def get_ip_address(url):
    try:
        ip_address = socket.gethostbyname(urlparse(url).netloc)
        return ip_address
    except Exception as e:
        print(e)
        return e

def log_prediction(url, prediction, prediction_score, unique_id, timestamp):
    ip_address = get_ip_address(url)
    with open('predictions.txt', 'a') as f:
        f.write(f"{unique_id}, {url}, {ip_address}, {prediction}, {prediction_score}, {timestamp}\n")

def log_detection(url, prediction_score, timestamp):
    try:
        existing_urls = set()
        with open('/root/freephish_stable/detections.txt', 'r') as file:
            for line in file:
                existing_url = line.split(',')[0].strip('[]')
                existing_urls.add(existing_url)
        utc_dt = datetime.utcfromtimestamp(timestamp).replace(tzinfo=pytz.utc)
        cst_dt = utc_dt.astimezone(pytz.timezone('America/Chicago'))
        formatted_time = cst_dt.strftime('%Y-%m-%d %H:%M:%S')
        ip_address = get_ip_address(url)
        if url not in existing_urls:
            with open('/root/freephish_stable/detections.txt', 'a') as f:
                f.write(f"[{url}],{ip_address},{formatted_time},{prediction_score}\n")

    except FileNotFoundError:
        pass  #


def generate_unique_id():
    return str(random.randint(10**4, 10**9 - 1))

@app.route('/check_url', methods=['POST'])
def check_url():
    data = request.json
    url = data.get('url', '')
    domain = get_domain_from_url(url)

    if is_domain_whitelisted(domain):
        return jsonify({'result': 'benign'})

    current_time = time.time()
    unique_id = generate_unique_id()

    model_path = '/root/freephish_stable/html_structure_model'  
    model, tokenizer = load_model(model_path)

    try:
        html_content = fetch_html(url)
        structured_text = html_to_structured_text(html_content)
        prediction, prediction_score = predict(structured_text, model, tokenizer)
        log_prediction(url, prediction, prediction_score, unique_id, current_time)
        

        if prediction == "phishing":
            log_detection(url, prediction_score, current_time)

       
    except Exception as e:
        log_prediction(url, 'error', unique_id, current_time)
        

@app.route('/web_request', methods=['GET'])
def web_request():
    url = request.args.get('url', '')
    domain = get_domain_from_url(url)

    if is_domain_whitelisted(domain):
        return jsonify({'result': 'benign', 'score': None})

    current_time = time.time()
    unique_id = generate_unique_id()

    model_path = '/root/freephish_stable/html_structure_model' 
    model, tokenizer = load_model(model_path)

    try:
        html_content = fetch_html(url)
        structured_text = html_to_structured_text(html_content)
        prediction, prediction_score = predict(structured_text, model, tokenizer)
        log_prediction(url, prediction, prediction_score, unique_id, current_time)

        if prediction == "phishing":
            log_detection(url, prediction_score, current_time)

        return jsonify({'result': prediction, 'score': prediction_score})

    except Exception as e:
        log_prediction(url, 'error', 0, unique_id, current_time)  
        return jsonify({'result': 'error', 'score': None})  

@app.route('/')
def home():
    recent_urls = []
    try:
        with open('/root/freephish_stable/detections.txt', 'r') as file:
            for line in file:
                parts = line.strip().split(',')
                if len(parts) >= 4:  
                    score = float(parts[3].strip())  
                    
                    if score > 0.8:
                        confidence = "High"
                    elif score >= 0.5:
                        confidence = "Medium"
                    else:
                        confidence = "Low"

                    url_data = {
                        'url': parts[0].strip('[]'),
                        'ip': parts[1].strip(),
                        'age': parts[2].strip(),
                        'confidence': confidence  
                    }
                    recent_urls.append(url_data)
        recent_urls.reverse()  
    except FileNotFoundError:
        print("detections.txt not found.")

    return render_template('index.html', recent_urls=recent_urls)

if __name__ == '__main__':
    read_whitelist("/root/freephish_stable/whitelist.csv")
    app.run(host='0.0.0.0', port=5012)
