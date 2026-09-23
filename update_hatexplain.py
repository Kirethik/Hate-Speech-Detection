import glob, re
import sys

def modify_hatexplain():
    with open('N:/Civitas/converters/hatexplain.py', 'r', encoding='utf-8') as f:
        text = f.read()

    text = text.replace('from label_maps import TARGET_CLASSES, SEVERITY_CLASSES  # noqa: E402', 'from label_maps import TARGET_CLASSES, SEVERITY_CLASSES, HATEXPLAIN_TARGET_MAP  # noqa: E402')
    text = re.sub(r'TARGET_MAP = \{.*?\n\}\n', '', text, flags=re.DOTALL)
    text = text.replace('TARGET_MAP.get', 'HATEXPLAIN_TARGET_MAP.get')
    
    with open('N:/Civitas/converters/hatexplain.py', 'w', encoding='utf-8') as f:
        f.write(text)

modify_hatexplain()
