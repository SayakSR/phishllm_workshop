# phishllm_workshop

import os
from bs4 import BeautifulSoup
import csv

def directory_html_to_csv(directory_path, output_csv_path):
    with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        csvwriter = csv.writer(csvfile)
        # Write the column headers
        csvwriter.writerow(['text', 'label'])
        
        # Iterate over each file in the directory
        for filename in os.listdir(directory_path):
            if filename.endswith('.txt'):  # Ensure it's an HTML file saved as .txt
                file_path = os.path.join(directory_path, filename)
                
                with open(file_path, 'r', encoding='utf-8') as file:
                    html_content = file.read()
                
                soup = BeautifulSoup(html_content, 'lxml')
                
                # Extract text from specified tags
                for element in soup.find_all(['h1', 'h2', 'h3', 'p', 'a', 'ul', 'ol', 'li']):
                    if element.name in ['ul', 'ol']:
                        items = [li.get_text(strip=True) for li in element.find_all('li')]
                        for item in items:
                            csvwriter.writerow([item, 0])
                    else:
                        text = element.get_text(strip=True)
                        csvwriter.writerow([text, 0])
    
    print(f"Phishing data saved to {output_csv_path}")

# Specify the path to your directory and output CSV file
directory_path = 'phish_html'  # Update this path
output_csv_path = 'phishing.csv'
directory_html_to_csv(directory_path, output_csv_path)
