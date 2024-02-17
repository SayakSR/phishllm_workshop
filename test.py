import csv
import json
from bs4 import BeautifulSoup
import requests
from urllib.parse import urlparse
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch

MODEL_PATH = 'html_structure_model'  
model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

whitelist_domains = set()

def read_whitelist(file_name):
    try:
        with open(file_name, mode='r', newline='', encoding='utf-8') as file:
            reader = csv.reader(file)
            for row in reader:
                if len(row) > 1:  
                    whitelist_domains.add(row[1].strip()) 
    except FileNotFoundError:
        print(f"File {file_name} not found.")

def get_domain_from_url(url):
    netloc = urlparse(url).netloc
    domain_parts = netloc.split('.')
    if domain_parts[0] == 'www':
        domain_parts.pop(0)
    return '.'.join(domain_parts[-2:])

def is_domain_whitelisted(url):
    domain = get_domain_from_url(url)
    return domain in whitelist_domains

def fetch_html(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.text
    except requests.RequestException as e:
        print(f"Failed to fetch URL {url}: {e}")
        return None

def html_to_structured_text(file_path, output_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        html_content = file.read()

    soup = BeautifulSoup(html_content, 'lxml')
    structured_data = []

    for element in soup.find_all(['h1', 'h2', 'h3', 'p', 'a', 'ul', 'ol', 'li']):
        element_data = {
            'type': element.name,
            'attributes': dict(element.attrs),
            'text': element.get_text(strip=True)
        }
        if element.name in ['ul', 'ol']:
            element_data['items'] = [li.get_text(strip=True) for li in element.find_all('li')]
        
        structured_data.append(element_data)

    with open(output_path, 'w', encoding='utf-8') as output_file:
        json.dump(structured_data, output_file, indent=4)

def predict_from_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        structured_data = json.load(file)
    text = " ".join([item['text'] for item in structured_data if 'text' in item])
    return predict(text)

def predict(text):
    inputs = tokenizer(text=text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        logits = model(**inputs).logits
    scores = torch.softmax(logits, dim=1)
    prediction = scores.argmax().item()
    prediction_score = scores[0][prediction].item()
    return "Phishing" if prediction == 0 else "Benign", prediction_score

def main():
    read_whitelist("whitelist.csv")  
    
    url = input("Please enter the URL to check: ")
    
    if is_domain_whitelisted(url):
        print("The domain of the URL is whitelisted. It is considered benign.")
        return
    
    html_content = fetch_html(url)
    if html_content is None:
        print("Failed to fetch the URL's content.")
        return
    
    temp_html_path = "temp.html"
    with open(temp_html_path, 'w', encoding='utf-8') as file:
        file.write(html_content)
    
    json_output_path = "structured_data.json"
    html_to_structured_text(temp_html_path, json_output_path)
    
    prediction, score = predict_from_json(json_output_path)
    print(f"Prediction: {prediction} (Score: {score:.4f})")

if __name__ == "__main__":
    main()
