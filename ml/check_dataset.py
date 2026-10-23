
from pathlib import Path
from collections import Counter
from hashlib import sha256
from PIL import Image

DATA_DIR = Path(r"D:\BrainTumorAI\data")
SPLITS = ("Training", "Testing")
CLASSES = ("glioma", "meningioma", "notumor", "pituitary")
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def file_hash(path):
    digest = sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def main():
    hash_locations = {}
    corrupt_files = []
    counts = Counter()
    missing_folders = []

    for split in SPLITS:
        print(f"\n===== {split} DATASET =====")

        for class_name in CLASSES:
            folder = DATA_DIR / split / class_name

            if not folder.is_dir():
                print(f"MISSING FOLDER: {folder}")
                missing_folders.append(str(folder))
                continue

            files = sorted(
                p for p in folder.rglob("*")
                if p.is_file() and p.suffix.lower() in EXTENSIONS
            )

            counts[(split, class_name)] = len(files)
            print(f"{class_name}: {len(files)} images")

            for path in files:
                try:
                    # Verify image structure.
                    with Image.open(path) as image:
                        image.verify()

                    # Calculate hash to identify exact duplicate files.
                    digest = file_hash(path)
                    hash_locations.setdefault(digest, []).append(
                        (split, class_name, str(path))
                    )

                except Exception as exc:
                    corrupt_files.append((str(path), str(exc)))

    duplicates = [
        locations
        for locations in hash_locations.values()
        if len(locations) > 1
    ]

    cross_split = [
        locations for locations in duplicates
        if len({item[0] for item in locations}) > 1
    ]

    print("\n===== INTEGRITY REPORT =====")
    print("Total images:", sum(counts.values()))
    print("Unique image hashes:", len(hash_locations))
    print("Corrupt/unreadable images:", len(corrupt_files))
    print("Exact duplicate groups:", len(duplicates))
    print("Duplicate groups across splits:", len(cross_split))
    print("Missing folders:", len(missing_folders))

    if corrupt_files:
        print("\n--- Corrupt files ---")
        for path, error in corrupt_files:
            print(path, "|", error)

    if cross_split:
        print("\n--- Exact duplicates across Training/Testing ---")
        for locations in cross_split:
            for split, class_name, path in locations:
                print(f"{split} | {class_name} | {path}")
            print()

    if missing_folders:
        print("\n--- Missing folders ---")
        for folder in missing_folders:
            print(folder)

    print("\nCheck complete. No files were modified or deleted.")

if __name__ == "__main__":
    main()
