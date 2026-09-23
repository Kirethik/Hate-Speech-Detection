import re

with open('N:/Civitas/converters/ruhsold.py', 'r', encoding='utf-8') as f:
    text = f.read()

text = text.replace('from label_maps import TARGET_CLASSES, SEVERITY_CLASSES', 'from label_maps import TARGET_CLASSES, SEVERITY_CLASSES, RUHSOLD_SEVERITY_MAP')
text = re.sub(r'LABEL_TO_SEVERITY = \{.*?\n\}\n', '', text, flags=re.DOTALL)
text = text.replace('LABEL_TO_SEVERITY', 'RUHSOLD_SEVERITY_MAP')

with open('N:/Civitas/converters/ruhsold.py', 'w', encoding='utf-8') as f:
    f.write(text)
