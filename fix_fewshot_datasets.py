# save as: fix_fewshot_datasets.py
import random
import shutil
from pathlib import Path

random.seed(420)

src = Path('C:/Users/XR-Student/Documents/AD-IRAS/datasets/MaskedDataset/Computer')
dst_base = Path('C:/Users/XR-Student/Documents/AD-IRAS/datasets/FewShotCurve')

# Get all training images
train_images = sorted(list((src / 'train' / 'good').glob('*.jpg')) +
                      list((src / 'train' / 'good').glob('*.png')))
print(f"Total training images: {len(train_images)}")

for k in [5, 10, 20, 30, 46]:
    dst = dst_base / f'k{k}' / 'Computer'

    # Wipe and recreate
    if dst.exists():
        shutil.rmtree(dst)

    # Copy test and ground_truth (always same)
    for split in ['test', 'ground_truth']:
        shutil.copytree(src / split, dst / split)

    # Copy only k training images
    dst_train = dst / 'train' / 'good'
    dst_train.mkdir(parents=True, exist_ok=True)

    selected = random.sample(train_images, k)
    for img in selected:
        shutil.copy(img, dst_train / img.name)

    # Verify
    n_train = len(list(dst_train.glob('*')))
    n_test  = len(list((dst / 'test' / 'anomaly').glob('*')))
    print(f"✅ k={k}: train={n_train} | test/anomaly={n_test}")

print("\n✅ All datasets ready!")