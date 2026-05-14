from PIL import Image
import os

base = '/home/user1/aniket/Patchcore/dataset/kvasir_patchcore/kvasir'
target_size = (224, 224)

# Check test defect images
print("=== Test defect image sizes ===")
test_defect = os.path.join(base, 'test/defect')
for f in sorted(os.listdir(test_defect)):
    if f.lower().endswith(('.png', '.jpg', '.bmp', '.tif')):
        img = Image.open(os.path.join(test_defect, f))
        print(f"  {f}: {img.size}")

# Check ground truth masks
print("\n=== Ground truth mask sizes ===")
gt_dir = os.path.join(base, 'ground_truth/defect')
for f in sorted(os.listdir(gt_dir)):
    if f.lower().endswith(('.png', '.jpg', '.bmp', '.tif')):
        img = Image.open(os.path.join(gt_dir, f))
        print(f"  {f}: {img.size}")

print("\n=== Resizing everything to 224x224 ===")
for directory in [test_defect, gt_dir]:
    for f in os.listdir(directory):
        if f.lower().endswith(('.png', '.jpg', '.bmp', '.tif')):
            path = os.path.join(directory, f)
            img = Image.open(path)
            if img.size != target_size:
                img = img.resize(target_size, Image.NEAREST)
                img.save(path)
                print(f"  Resized: {directory}/{f}")

print("\nDone!")