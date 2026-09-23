import re
with open('N:/Civitas/dataset.py', 'r', encoding='utf-8') as f:
    text = f.read()
text = re.sub(r'# NOTE: "political" is currently a DEAD CLASS.*?checkpoint\'s 8-way target head\.', '# NOTE: "political" is no longer a dead class — it is resolved by implicit-hate data.', text, flags=re.DOTALL)
with open('N:/Civitas/dataset.py', 'w', encoding='utf-8') as f:
    f.write(text)
