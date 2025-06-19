import json
import sys

def filter_json(input_file, output_file):
    with open(input_file, 'r') as file:
        data = json.load(file)
    
    filtered_data = []
    for entry in data:
        if entry['data_format'] == 'SVS':
            filtered_entry = {
                'file_name': entry['file_name'],
                'file_id': entry['file_id'],
                'case_id': entry['associated_entities'][0]['case_id']
            }
            filtered_data.append(filtered_entry)

    with open(output_file, 'w') as file:
        json.dump(filtered_data, file, indent=2)

    print(f"Filtered data has been written to '{output_file}'")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <input_file> <output_file>")
    else:
        input_file = sys.argv[1]
        output_file = sys.argv[2]
        filter_json(input_file, output_file)