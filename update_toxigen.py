import re

with open('N:/Civitas/converters/toxigen.py', 'r', encoding='utf-8') as f:
    text = f.read()

text = text.replace('from label_maps import TARGET_CLASSES, SEVERITY_CLASSES', 'from label_maps import TARGET_CLASSES, SEVERITY_CLASSES, TOXIGEN_TARGET_MAP')
text = re.sub(r'TARGET_MAP = \{.*?\n\}\n', '', text, flags=re.DOTALL)
text = text.replace('TARGET_MAP', 'TOXIGEN_TARGET_MAP')

with open('N:/Civitas/converters/toxigen.py', 'w', encoding='utf-8') as f:
    f.write(text)
