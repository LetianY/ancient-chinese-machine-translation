import json


def decode_unicode_escape(json_str):
    """Decode unicode escape sequences in a JSON string."""
    return json.loads(json_str)

def convert_json_unicode_to_utf8(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
        for line in infile:
            decoded_data = decode_unicode_escape(line.strip())
            json.dump(decoded_data, outfile, ensure_ascii=False, indent=4)  # Use indent for pretty-printing
            outfile.write('\n')


# Example usage:
input_path = 'seed123_step0.json'
output_path = 'output_utf8.json'

convert_json_unicode_to_utf8(input_path, output_path)
