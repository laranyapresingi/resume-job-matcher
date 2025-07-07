import re
import os

def clean_job_description(raw_text):
    text = re.sub(r'#.*', '', raw_text)
    text = re.sub(r'(?i)(Responsibilities|Qualifications|Requirements|About us):?', '', text)
    text = re.sub(r'[-•●▪️]', '', text)
    text = re.sub(r'\n{2,}', '\n', text)
    text = text.strip()
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    return ' '.join(lines)

def clean_all_jds(input_dir='raw_job_desc', output_dir='jds_clean'):
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".txt"):
                # Full input path
                in_path = os.path.join(root, file)
                
                # Subdirectory relative to raw_jds
                rel_path = os.path.relpath(in_path, input_dir)
                
                # Corresponding output path
                out_path = os.path.join(output_dir, rel_path)
                os.makedirs(os.path.dirname(out_path), exist_ok=True)

                with open(in_path, 'r', encoding='utf-8') as f:
                    raw = f.read()

                cleaned = clean_job_description(raw)

                with open(out_path, 'w', encoding='utf-8') as f:
                    f.write(cleaned)

                print(f"[✓] Cleaned: {rel_path}")

if __name__ == "__main__":
    clean_all_jds()
