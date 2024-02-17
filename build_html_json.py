from bs4 import BeautifulSoup
import json

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

    print(f"Structured data saved to {output_path}")

html_file_path = input("Enter the path to your HTML file: ")
output_file_path = input("Enter the path for the output structured text file: ")

html_to_structured_text(html_file_path, output_file_path)
