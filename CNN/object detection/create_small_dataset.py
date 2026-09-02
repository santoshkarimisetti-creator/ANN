import os, shutil, random, yaml
from pathlib import Path
from collections import Counter, defaultdict

random.seed(42)

SRC = Path(r'C:\Users\santo\.cache\kagglehub\datasets\rupankarmajumdar\disaster-response-object-detection-dataset\versions\1')
DST = Path(r'd:\Projects\Dp Learning\CNN\disaster_small')

# Target counts
TARGETS = {'train': 700, 'val': 200, 'test': 100}
SRC_MAP = {'train': 'train', 'val': 'val', 'test': 'test'}

def get_classes_in_label(label_path):
    classes = set()
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                classes.add(int(parts[0]))
    return classes

print('Analyzing class coverage...')

for dst_split, count in TARGETS.items():
    src_split = SRC_MAP[dst_split]
    src_imgs = SRC / src_split / 'images'
    src_lbls = SRC / src_split / 'labels'

    img_files = {p.stem: p for p in src_imgs.iterdir() if p.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp')}
    lbl_files = {p.stem: p for p in src_lbls.iterdir() if p.suffix == '.txt'}
    paired_stems = sorted(set(img_files) & set(lbl_files))
    print(f'{dst_split}: {len(paired_stems)} paired files available, need {count}')

    # Build class -> stems index
    class_to_stems = defaultdict(set)
    stem_to_classes = {}
    for stem in paired_stems:
        cls_set = get_classes_in_label(lbl_files[stem])
        stem_to_classes[stem] = cls_set
        for c in cls_set:
            class_to_stems[c].add(stem)

    # Greedy: ensure all 6 classes represented, prioritize rare classes
    selected = set()
    all_classes = sorted(class_to_stems.keys())
    print(f'  Classes present: {all_classes}')

    for c in sorted(all_classes, key=lambda c: len(class_to_stems[c])):
        candidates = class_to_stems[c] - selected
        pick = min(max(5, count // (len(all_classes) * 2)), len(candidates))
        selected.update(random.sample(sorted(candidates), pick))

    # Fill remaining randomly
    remaining = sorted(set(paired_stems) - selected)
    need = count - len(selected)
    if need > 0:
        selected.update(random.sample(remaining, min(need, len(remaining))))
    elif need < 0:
        selected = set(random.sample(sorted(selected), count))

    selected = sorted(selected)[:count]

    # Copy files
    dst_imgs = DST / dst_split / 'images'
    dst_lbls = DST / dst_split / 'labels'
    dst_imgs.mkdir(parents=True, exist_ok=True)
    dst_lbls.mkdir(parents=True, exist_ok=True)

    for stem in selected:
        shutil.copy2(img_files[stem], dst_imgs / img_files[stem].name)
        shutil.copy2(lbl_files[stem], dst_lbls / lbl_files[stem].name)

    # Verify class coverage
    final_classes = Counter()
    for stem in selected:
        for c in stem_to_classes.get(stem, set()):
            final_classes[c] += 1
    print(f'  Selected {len(selected)} | Class counts: {dict(sorted(final_classes.items()))}')

# Create data.yaml
class_names = ['person', 'fire', 'smoke', 'small_vehicle', 'large_vehicle', 'two_wheeler']
data_yaml = {
    'path': str(DST),
    'train': 'train/images',
    'val': 'val/images',
    'test': 'test/images',
    'nc': 6,
    'names': class_names,
}
yaml_path = DST / 'data.yaml'
with open(yaml_path, 'w') as f:
    yaml.dump(data_yaml, f, default_flow_style=False, sort_keys=False)

print(f'\ndata.yaml created at {yaml_path}')

# Final summary
print('\ndisaster_small/')
total = 0
for split in ['train', 'val', 'test']:
    ni = len(list((DST / split / 'images').iterdir()))
    nl = len(list((DST / split / 'labels').iterdir()))
    total += ni
    print(f'   {split}/  ->  {ni} images, {nl} labels')
print(f'\nTotal: {total} images')
print('\nDone!')
