import glob, re

for file in glob.glob('N:/Civitas/converters/*.py'):
    with open(file, 'r', encoding='utf-8') as f:
        text = f.read()

    changed = False

    # 1. Update imports
    if 'from dataset import TARGET_CLASSES, SEVERITY_CLASSES' in text:
        text = text.replace('from dataset import TARGET_CLASSES, SEVERITY_CLASSES', 'from label_maps import TARGET_CLASSES, SEVERITY_CLASSES')
        changed = True
    elif 'from dataset import SEVERITY_CLASSES' in text:
        text = text.replace('from dataset import SEVERITY_CLASSES', 'from label_maps import TARGET_CLASSES, SEVERITY_CLASSES')
        changed = True
    elif 'from dataset import SEVERITY_CLASSES, TARGET_CLASSES' in text:
        text = text.replace('from dataset import SEVERITY_CLASSES, TARGET_CLASSES', 'from label_maps import TARGET_CLASSES, SEVERITY_CLASSES')
        changed = True

    # 2. Add specific map imports if needed, though they aren't explicitly requested to replace inside converters, wait:
    # "Do the same for ALL other converters — any that define their own label mappings should import from label_maps.py."
    # For hatexplain: update converter to use rom label_maps import HATEXPLAIN_TARGET_MAP, map_target, map_severity and remove local TARGET_MAP
    # Actually, it's safer to just replace TARGET_MAP = {...} and use the new one.
    
    if changed:
        with open(file, 'w', encoding='utf-8') as f:
            f.write(text)

