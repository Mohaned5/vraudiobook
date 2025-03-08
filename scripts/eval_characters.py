import os
import re
import glob
import numpy as np
from PIL import Image

# Face recognition for Face Similarity
import face_recognition

# PyTorch
import torch
import torchvision.transforms as T

# LPIPS for perceptual image similarity
import lpips

# CLIP for image similarity
import clip

def load_image_as_tensor(image_path, transform=None):
    """
    Loads image from disk and applies a transform, returning a PyTorch tensor.
    """
    img = Image.open(image_path).convert("RGB")
    if transform:
        return transform(img)
    else:
        # As a fallback, return a standard 224x224 tensor for CLIP, etc.
        transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor()
        ])
        return transform(img)

def cosine_similarity(vec1, vec2):
    """
    Computes cosine similarity between two 1D numpy arrays or PyTorch tensors.
    """
    if isinstance(vec1, torch.Tensor):
        vec1 = vec1.detach().cpu().numpy()
    if isinstance(vec2, torch.Tensor):
        vec2 = vec2.detach().cpu().numpy()
    dot = (vec1 * vec2).sum()
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return dot / (norm1 * norm2 + 1e-9)

def compute_face_similarity(img_path1, img_path2):
    """
    Uses face_recognition to get face encodings and compute
    a cosine similarity measure for the recognized faces.
    """
    img1 = face_recognition.load_image_file(img_path1)
    img2 = face_recognition.load_image_file(img_path2)

    encodings1 = face_recognition.face_encodings(img1)
    encodings2 = face_recognition.face_encodings(img2)

    if len(encodings1) < 1 or len(encodings2) < 1:
        # If either image doesn't have a detectable face, return None or 0
        return None

    # For simplicity, assume one face per image
    encoding1 = encodings1[0]
    encoding2 = encodings2[0]

    # Face recognition encodings are NumPy arrays; compute cosine similarity
    return cosine_similarity(encoding1, encoding2)

def compute_lpips_distance(lpips_model, img_path1, img_path2):
    """
    Compute LPIPS distance (the smaller, the more similar).
    We'll invert it as "similarity" or keep it as "distance."
    """
    img1 = Image.open(img_path1).convert('RGB')
    img2 = Image.open(img_path2).convert('RGB')

    # Convert to normalized torch Tensors
    preprocess = T.Compose([
        T.Resize((256, 256)),  # can adjust size
        T.ToTensor()
    ])
    t1 = preprocess(img1).unsqueeze(0)
    t2 = preprocess(img2).unsqueeze(0)

    with torch.no_grad():
        dist = lpips_model(t1, t2)
    # dist is in [0..1+] range (roughly), lower means more similar
    return dist.item()

def compute_clip_similarity(model, preprocess, img_path1, img_path2, device='cpu'):
    """
    Compute similarity with CLIP image encoder, using cosine similarity.
    """
    img1 = Image.open(img_path1).convert('RGB')
    img2 = Image.open(img_path2).convert('RGB')

    t1 = preprocess(img1).unsqueeze(0).to(device)
    t2 = preprocess(img2).unsqueeze(0).to(device)

    with torch.no_grad():
        emb1 = model.encode_image(t1)
        emb2 = model.encode_image(t2)
        emb1 = emb1 / emb1.norm(dim=-1, keepdim=True)
        emb2 = emb2 / emb2.norm(dim=-1, keepdim=True)
        sim = (emb1 * emb2).sum(dim=-1).item()  # this is cosine similarity
    return sim

def main():
    predict_dir = "predict"
    
    # Identify subfolders that match the pattern "[number]_[number]"
    # e.g. "1_1", "1_2" ... "18_5"
    # We also skip 14 if it doesn't exist.
    pattern = re.compile(r'^(\d+)_(\d+)$')
    
    character_dict = {}  # e.g. { '1': ['predict/1_1/pano.jpg', ...], '2': [...], ... }
    
    for folder_name in os.listdir(predict_dir):
        match = pattern.match(folder_name)
        if match:
            char_id = match.group(1)  # e.g. "1"
            # sub_id = match.group(2)  # Not used, but you could track it if needed
            pano_path = os.path.join(predict_dir, folder_name, "pano.jpg")
            if os.path.isfile(pano_path):
                character_dict.setdefault(char_id, []).append(pano_path)

    # Initialize models

    # 1) Face Recognition - no special init, face_recognition just runs
    # 2) LPIPS
    lpips_model = lpips.LPIPS(net='alex').eval()

    # 3) CLIP
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)

    # For storing overall results across all characters
    face_sims_all = []
    lpips_all = []
    clip_sims_all = []
    # We'll define "trusted face diversity" as the average standard deviation
    # of face encodings within each character, though you can define your own method.
    tfd_all = []

    # Iterate characters
    for char_id, image_paths in character_dict.items():
        if len(image_paths) < 2:
            # Can't compare if there's only one image
            continue

        # Pairwise comparisons
        char_face_sims = []
        char_lpips_dists = []
        char_clip_sims = []

        # For "trusted face diversity", let's store face encodings and
        # measure the standard deviation across them. 
        # (One approach: compute all face_encodings, then compute average
        # pairwise distance or standard deviation.)
        face_encodings = []

        # We make all pairwise comparisons
        for i in range(len(image_paths)):
            for j in range(i + 1, len(image_paths)):
                p1 = image_paths[i]
                p2 = image_paths[j]

                # Face similarity
                f_sim = compute_face_similarity(p1, p2)
                if f_sim is not None:
                    char_face_sims.append(f_sim)

                # LPIPS distance
                lpips_dist = compute_lpips_distance(lpips_model, p1, p2)
                char_lpips_dists.append(lpips_dist)

                # CLIP similarity
                c_sim = compute_clip_similarity(clip_model, clip_preprocess, p1, p2, device)
                char_clip_sims.append(c_sim)

        # If we want to measure "face diversity", gather encodings
        # from each image w.r.t. face_recognition.
        for p in image_paths:
            img = face_recognition.load_image_file(p)
            encs = face_recognition.face_encodings(img)
            if len(encs) > 0:
                face_encodings.append(encs[0])
        
        # Example approach for "Trusted Face Diversity":
        # - compute the mean encoding
        # - measure the Euclidean distance from each encoding to the mean
        # - average or std-dev of that distance
        # Lower values => more consistent; higher => more diverse
        if len(face_encodings) > 1:
            face_encodings = np.array(face_encodings)
            mean_encoding = np.mean(face_encodings, axis=0)
            dists = [np.linalg.norm(e - mean_encoding) for e in face_encodings]
            tfd_value = np.mean(dists)  # you could use np.std(dists) instead
        else:
            tfd_value = 0.0  # or None

        # Average for this character
        face_sim_avg = np.mean(char_face_sims) if len(char_face_sims) else None
        lpips_dist_avg = np.mean(char_lpips_dists) if len(char_lpips_dists) else None
        clip_sim_avg = np.mean(char_clip_sims) if len(char_clip_sims) else None

        # Save for overall
        if face_sim_avg is not None:
            face_sims_all.append(face_sim_avg)
        if lpips_dist_avg is not None:
            lpips_all.append(lpips_dist_avg)
        if clip_sim_avg is not None:
            clip_sims_all.append(clip_sim_avg)
        tfd_all.append(tfd_value)

        print(f"Character {char_id}:")
        print(f"  Face Similarity Avg: {face_sim_avg}")
        print(f"  LPIPS Distance Avg:  {lpips_dist_avg}")
        print(f"  CLIP Similarity Avg: {clip_sim_avg}")
        print(f"  Trusted Face Diversity: {tfd_value}")
        print("")

    # Final averages across all characters
    face_sims_mean = np.mean(face_sims_all) if len(face_sims_all) else 0
    lpips_mean = np.mean(lpips_all) if len(lpips_all) else 0
    clip_sims_mean = np.mean(clip_sims_all) if len(clip_sims_all) else 0
    tfd_mean = np.mean(tfd_all) if len(tfd_all) else 0

    print("==== Final Averages Across All Characters ====")
    print(f"Face Similarity:         {face_sims_mean:.4f}")
    print(f"LPIPS Distance:          {lpips_mean:.4f}")
    print(f"CLIP Similarity:         {clip_sims_mean:.4f}")
    print(f"Trusted Face Diversity:  {tfd_mean:.4f}")

if __name__ == "__main__":
    main()
