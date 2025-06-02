#!/usr/bin/env python3
"""
main.py

Runs a 10‐fold evaluation (FF and FP protocols) on the CFP dataset using:
  - DeepFace’s Facenet512 embeddings (via model_name='Facenet512')
  - InsightFace’s buffalo_l with a custom ONNX recognizer

Metrics per fold:
  - ACC (accuracy at the EER threshold)
  - AUC (area under the ROC curve)
  - EER (equal error rate)

Additionally, this script writes out every pair’s similarity score and label
into `all_similarity_results.csv` with columns:
    model,mode,fold,img1,img2,similarity,label

Usage:
  1. Put this script in the same folder where “cfp-dataset/cfp-dataset” resides.
  2. Install dependencies:
       pip install numpy scikit-learn deepface insightface onnxruntime pandas opencv-python
  3. Run:
       python main.py
"""

import os
import time
import cv2
import csv
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc, accuracy_score
from deepface import DeepFace
import insightface
from insightface.app import FaceAnalysis

# ------------------------------------------------------------------------------

def load_map(file_path, base_folder):
    """
    Reads a Pair_list file (index and relative path) and returns a dict:
        { index → absolute_image_path }.
    """
    m = {}
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            idx_str, rel_path = line.split()
            idx = int(idx_str)
            abs_path = os.path.normpath(os.path.join(base_folder, rel_path))
            m[idx] = abs_path
    return m

def get_df_embeddings(all_image_paths):
    """
    Given a list of image paths, runs DeepFace.represent(...) once per image
    (using model_name='Facenet512') and returns a dict {img_path → normalized embedding}.
    """
    embeddings = {}
    for img_path in all_image_paths:
        rep = DeepFace.represent(
            img_path=img_path,
            model_name='Facenet512',
            enforce_detection=False
        )

        # DeepFace.represent sometimes returns a list-of‐dict or a dict
        if isinstance(rep, list) and isinstance(rep[0], dict) and "embedding" in rep[0]:
            emb_vec = np.array(rep[0]["embedding"], dtype=np.float32)
        elif isinstance(rep, dict) and "embedding" in rep:
            emb_vec = np.array(rep["embedding"], dtype=np.float32)
        else:
            raise ValueError(f"Unexpected DeepFace output for {img_path}")

        emb_vec /= np.linalg.norm(emb_vec)
        embeddings[img_path] = emb_vec

    return embeddings

def get_if_embeddings(all_image_paths, insight_app, print_every=200):
    """
    Given a sorted list of image paths, loads each image via OpenCV and runs
    InsightFace.app.FaceAnalysis.get(...) to extract embeddings. Prints progress.

    Returns: dict { img_path → normalized_embedding }.
    """
    embeddings = {}
    total = len(all_image_paths)
    print(f"    [InsightFace] Embedding {total} images ...")
    start_time = time.time()

    for idx, img_path in enumerate(all_image_paths, start=1):
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            print(f"      ⚠️ Could not read {img_path} (skipping)")
            continue

        try:
            faces = insight_app.get(img_bgr)
        except Exception as e:
            print(f"      ⚠️ Skipped {img_path} (InsightFace error: {str(e)})")
            continue

        if not faces:
            continue

        emb_vec = faces[0].embedding.astype(np.float32)
        emb_vec /= np.linalg.norm(emb_vec)
        embeddings[img_path] = emb_vec

        if idx % print_every == 0 or idx == total:
            elapsed = time.time() - start_time
            print(f"      [InsightFace] {idx}/{total} done  (elapsed {elapsed:.1f}s)")

    print(f"    [InsightFace] Completed embeddings: {len(embeddings)}/{total}")
    return embeddings

def load_pairs_for_fold(split_dir, maps_tuple):
    """
    For a given split folder (e.g., .../Split/FF/1), read same.txt and diff.txt,
    then map indices → absolute paths using maps_tuple = (mapA, mapB).
    Returns two lists of (path1, path2):
      same_pairs = [ (pathA_i, pathA_j), ... ]
      diff_pairs = [ (pathA_i, pathB_j), ... ]
    """
    same = []
    diff = []
    for filename, container in [('same.txt', same), ('diff.txt', diff)]:
        full_path = os.path.join(split_dir, filename)
        with open(full_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                i1_str, i2_str = line.split(',')
                i1, i2 = int(i1_str), int(i2_str)
                # maps_tuple = (mapA, mapB)
                p1 = maps_tuple[0][i1]
                p2 = maps_tuple[1][i2]
                container.append((p1, p2))
    return same, diff

def evaluate_fold(model_embs, same_pairs, diff_pairs):
    """
    Given a dict of embeddings {img_path→vector}, plus two lists of pairs,
    compute:
      - sims & labels arrays,
      - ACC, AUC, EER
    Returns (acc, auc_score, eer, sims_array, labels_array, pair_list)
      where pair_list is [(img1,img2), ...] in the same order as sims & labels.
    """
    sims = []
    labels = []
    pair_list = []

    # Process "same" pairs with label=1
    for (p1, p2) in same_pairs:
        if p1 not in model_embs or p2 not in model_embs:
            continue
        sim_val = float(np.dot(model_embs[p1], model_embs[p2]))
        sims.append(sim_val)
        labels.append(1)
        pair_list.append((p1, p2))

    # Process "diff" pairs with label=0
    for (p1, p2) in diff_pairs:
        if p1 not in model_embs or p2 not in model_embs:
            continue
        sim_val = float(np.dot(model_embs[p1], model_embs[p2]))
        sims.append(sim_val)
        labels.append(0)
        pair_list.append((p1, p2))

    sims = np.array(sims, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)

    # If there are no valid pairs, return zeros
    if labels.size == 0:
        return 0.0, 0.0, 0.0, sims, labels, pair_list

    fpr, tpr, thresholds = roc_curve(labels, sims, pos_label=1)
    auc_score = auc(fpr, tpr)

    fnr = 1.0 - tpr
    idx_eer = np.nanargmin(np.abs(fnr - fpr))
    eer = (fpr[idx_eer] + fnr[idx_eer]) / 2.0
    thr_eer = thresholds[idx_eer]

    preds = (sims >= thr_eer).astype(np.int32)
    acc = accuracy_score(labels, preds)
    return acc, auc_score, eer, sims, labels, pair_list

# ------------------------------------------------------------------------------

if __name__ == "__main__":
    # (A) Locate the “cfp-dataset” folder and protocol subfolders
    base = "cfp-dataset"
    if not os.path.isdir(os.path.join(base, "Protocol")):
        nested = os.path.join(base, os.path.basename(base))
        if os.path.isdir(os.path.join(nested, "Protocol")):
            base = nested
    protocol_dir = os.path.join(base, "Protocol")

    # (B) Build index→path maps for frontal (F) and profile (P)
    frontal_map = load_map(
        os.path.join(protocol_dir, "Pair_list_F.txt"),
        protocol_dir
    )
    profile_map = load_map(
        os.path.join(protocol_dir, "Pair_list_P.txt"),
        protocol_dir
    )

    # (C) Gather all distinct image paths from all 10 folds of both modes
    all_paths = set()
    for mode_name, maps_tuple in [("FF", (frontal_map, frontal_map)),
                                  ("FP", (frontal_map, profile_map))]:
        split_folder = os.path.join(protocol_dir, "Split", mode_name)
        for fold_name in sorted(os.listdir(split_folder)):
            fold_dir = os.path.join(split_folder, fold_name)
            same_pairs, diff_pairs = load_pairs_for_fold(fold_dir, maps_tuple)
            for (pA, pB) in (same_pairs + diff_pairs):
                all_paths.add(pA)
                all_paths.add(pB)

    all_paths = sorted(all_paths)
    print(f"Total distinct images to embed: {len(all_paths)}")

    # (D) Extract DeepFace (Facenet512) embeddings
    print("Extracting DeepFace (Facenet512) embeddings...")
    df_embeddings = get_df_embeddings(all_paths)
    print(f"  → Done: {len(df_embeddings)} images embedded by DeepFace.")

    # (E) Initialize InsightFace buffalo_l and swap in custom ONNX
    print("Initializing InsightFace buffalo_l and swapping in custom ONNX…")
    app = FaceAnalysis(
        name="buffalo_l",
        providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
    )
    app.prepare(ctx_id=0, det_size=(288, 288))

    print("Loading custom ONNX recognizer…")
    custom_recognizer = insightface.model_zoo.get_model('model.onnx')
    custom_recognizer.prepare(ctx_id=0)
    # Replace the “recognition” head with our custom ONNX
    for model_name, model_instance in app.models.items():
        if model_instance.taskname == 'recognition':
            app.models[model_name] = custom_recognizer
            break

    # (F) Extract InsightFace (custom ONNX) embeddings
    print("Extracting InsightFace (custom ONNX) embeddings...")
    if_embeddings = get_if_embeddings(all_paths, app)
    print(f"  → Done: {len(if_embeddings)} images embedded by InsightFace.")

    # (G) Prepare to write every similarity score into a CSV
    csv_filename = "all_similarity_results.csv"
    csv_fields = ["model", "mode", "fold", "img1", "img2", "similarity", "label"]
    csv_file = open(csv_filename, "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(csv_file, fieldnames=csv_fields)
    writer.writeheader()

    # (H) Run 10‐fold evaluation for both modes and both models,
    #     saving per‐pair similarities to CSV as we go.
    results = {'deepface': [], 'insight': []}
    for mode_name, maps_tuple, label in [
        ("FF", (frontal_map, frontal_map), "FF"),
        ("FP", (frontal_map, profile_map), "FP")
    ]:
        print(f"\nRunning 10‐fold evaluation on mode = '{label}' …")
        split_folder = os.path.join(protocol_dir, "Split", mode_name)

        for fold_name in sorted(os.listdir(split_folder), key=lambda x: int(x)):
            fold_dir = os.path.join(split_folder, fold_name)
            same_pairs, diff_pairs = load_pairs_for_fold(fold_dir, maps_tuple)

            # —— DeepFace fold evaluation + CSV export —— #
            acc_df, auc_df, eer_df, sims_df, labels_df, pairs_df = evaluate_fold(
                df_embeddings, same_pairs, diff_pairs
            )
            # Write every pair + sim score for DeepFace
            for ( (img1, img2), sim_val, lbl ) in zip(pairs_df, sims_df, labels_df):
                writer.writerow({
                    "model": "deepface",
                    "mode": mode_name,
                    "fold": fold_name,
                    "img1": img1,
                    "img2": img2,
                    "similarity": f"{sim_val:.6f}",
                    "label": int(lbl)
                })

            # —— InsightFace fold evaluation + CSV export —— #
            acc_if, auc_if, eer_if, sims_if, labels_if, pairs_if = evaluate_fold(
                if_embeddings, same_pairs, diff_pairs
            )
            for ( (img1, img2), sim_val, lbl ) in zip(pairs_if, sims_if, labels_if):
                writer.writerow({
                    "model": "insight",
                    "mode": mode_name,
                    "fold": fold_name,
                    "img1": img1,
                    "img2": img2,
                    "similarity": f"{sim_val:.6f}",
                    "label": int(lbl)
                })

            results['deepface'].append((acc_df, auc_df, eer_df))
            results['insight'].append((acc_if, auc_if, eer_if))

            print(
                f"  • Fold {fold_name:>2}: "
                f"DeepFace → ACC={acc_df:.4f}, AUC={auc_df:.4f}, EER={eer_df:.4f} | "
                f"Insight → ACC={acc_if:.4f}, AUC={auc_if:.4f}, EER={eer_if:.4f}"
            )

    csv_file.close()
    print(f"\nSaved all pairwise similarities to '{csv_filename}'.\n")

    # (I) Compute and print mean ± std across all 20 folds
    def summarize(metric_list):
        arr = np.array(metric_list, dtype=np.float32)
        return float(np.mean(arr)), float(np.std(arr))

    for model_key in ['deepface', 'insight']:
        arr = np.array(results[model_key], dtype=np.float32)  # shape: (20,3)
        accs = arr[:, 0]
        aucs = arr[:, 1]
        eers = arr[:, 2]
        m_acc, s_acc = summarize(accs)
        m_auc, s_auc = summarize(aucs)
        m_eer, s_eer = summarize(eers)
        print(f"{model_key.upper()} SUMMARY (across all 20 folds):")
        print(f"  → ACC = {m_acc:.4f} ± {s_acc:.4f}")
        print(f"  → AUC = {m_auc:.4f} ± {s_auc:.4f}")
        print(f"  → EER = {m_eer:.4f} ± {s_eer:.4f}")

    print("\nDone.")
