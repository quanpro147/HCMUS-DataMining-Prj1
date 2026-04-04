import json

with open('notebooks/02_preprocessing_image.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

count = len(nb['cells'])
print(f'Total cells: {count}')
print()

for i in range(count):
    cell = nb['cells'][i]
    src = cell.get('source', [])
    first_line = (src[0][:100] if src else '(empty)').strip()
    print(f'  Cell {i:2d} [{cell["cell_type"]:8s}]: {first_line}')
