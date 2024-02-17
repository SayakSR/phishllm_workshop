import os

source_dir = 'phish_data'
dest_dir = 'phish_html'

if not os.path.exists(dest_dir):
    os.makedirs(dest_dir)

counter=0

for root, dirs, files in os.walk(source_dir):
    for file in files:
        counter=counter+1
        print(f"Progress:{counter}")
        if file.endswith('html.txt'):
            file_path = os.path.join(root, file)
            folder_name = os.path.basename(root)
            dest_file_path = os.path.join(dest_dir, f"{folder_name}_{file}")
            copy_command = f"cp '{file_path}' '{dest_file_path}'"
            os.system(copy_command)

print("Files copied successfully.")
