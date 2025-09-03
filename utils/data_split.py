#TODO: finalisieren

import shutil
import random
from pathlib import Path
from typing import Dict, List, Tuple

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

def split_and_copy_images(
    input_paths_by_label: Dict[str, List[str]],
    output_root: str,
    split: Tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 42,
    recursive: bool = True,
    allow_duplicates_with_suffix: bool = True,
) -> None:
    """
    input_paths_by_label: dict mit Keys 'real' und/oder 'fake' und jeweils einer Liste von Ordnerpfaden.
        Beispiel:
        {
            "real": ["data/real_set1", "data/real_set2"],
            "fake": ["data/fake_a", "data/fake_b"]
        }

    output_root: Zielbasisordner. Darunter werden erzeugt:
        train/0_real, train/1_fake, val/0_real, val/1_fake, test/0_real, test/1_fake

    split: (train, val, test) Anteile in Summe ≈ 1.0
    seed: Zufallsseed für reproduzierbares Shufflen
    recursive: True -> rekursiv alle Unterordner scannen
    allow_duplicates_with_suffix: Falls gleiche Dateinamen kollidieren, hänge Suffix an (behält Originalnamen weitgehend).
    """
    # Validierung
    assert abs(sum(split) - 1.0) < 1e-6, "split muss zu 1.0 aufsummieren (z.B. (0.7, 0.15, 0.15))"
    for label in input_paths_by_label:
        if label not in {"real", "fake"}:
            raise ValueError(f"Unbekanntes Label '{label}'. Erlaubt: 'real', 'fake'")

    random.seed(seed)
    output_root = Path(output_root)

    # Zielordner vorbereiten
    subsets = ["train", "val", "test"]
    label_map = {"real": "0_real", "fake": "1_fake"}
    for subset in subsets:
        for k, v in label_map.items():
            (output_root / subset / v).mkdir(parents=True, exist_ok=True)

    def list_images(folder: Path) -> List[Path]:
        if recursive:
            files = [p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
        else:
            files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
        return files

    # Dateien je Label sammeln
    files_by_label = {"real": [], "fake": []}
    for label, folders in input_paths_by_label.items():
        for fpath in folders:
            p = Path(fpath)
            if not p.exists():
                print(f"[WARN] Eingabepfad nicht gefunden: {p}")
                continue
            files = list_images(p)
            files_by_label[label].extend(files)

    # Split & Kopieren pro Label
    def copy_with_collision_handling(src: Path, dst_dir: Path) -> None:
        dst = dst_dir / src.name
        if dst.exists() and allow_duplicates_with_suffix:
            stem, ext = dst.stem, dst.suffix
            i = 1
            while dst.exists():
                dst = dst_dir / f"{stem}__dup{i}{ext}"
                i += 1
        shutil.copy2(src, dst)

    for label in ["real", "fake"]:
        files = files_by_label[label]
        if not files:
            print(f"[INFO] Keine Dateien für Label '{label}' gefunden.")
            continue

        random.shuffle(files)
        n = len(files)
        n_train = int(n * split[0])
        n_val = int(n * split[1])
        n_test = n - n_train - n_val  # restliche Bilder gehen in test

        train_files = files[:n_train]
        val_files = files[n_train:n_train + n_val]
        test_files = files[n_train + n_val:]

        # Zielordner nach Spezifikation
        label_dirname = label_map[label]
        out_train = output_root / "train" / label_dirname
        out_val = output_root / "val" / label_dirname
        out_test = output_root / "test" / label_dirname

        for src in train_files:
            copy_with_collision_handling(src, out_train)
        for src in val_files:
            copy_with_collision_handling(src, out_val)
        for src in test_files:
            copy_with_collision_handling(src, out_test)

        print(
            f"[OK] {label}: {n} Dateien → "
            f"train={len(train_files)}, val={len(val_files)}, test={len(test_files)}"
        )