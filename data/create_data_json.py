import json
import glob

# dnd_dice/dice_dataset/train/d{4,6,8,10,12,20}/d10_2/d10_wood0234.jpg
import sys

split = sys.argv[1]

out = []

for side in [4, 6, 8, 10, 12, 20]:
    for i in range(1, side + 1):
        matches = glob.glob(f'{split}/d{side}/d{side}_{i}/*.jpg')
        for m in matches:
            out.append({'sides': side, 'chosen': i, 'file': m})

print(json.dumps(out))
out_filename = f"{split}.json"
with open(out_filename, "w") as out_f:
    json.dump(out, out_f, indent=4)
