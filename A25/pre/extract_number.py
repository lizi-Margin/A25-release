import re
def extract_number(filename):
    match = re.search(r"(\d+)(?=\.\w+$)", filename)
    if not match: print("[extract_number] Warning: regex failed to match")
    return int(match.group(1)) if match else 0