import torch
import numpy as np
import yaml
from pathlib import Path
from tqdm import tqdm

def load_dataset(path):
    with open(path, errors='ignore') as f:
        data_dict = yaml.safe_load(f)
    dataset_path = Path(data_dict['train'])
    label_paths = list(dataset_path.rglob('*.txt'))
    shapes = []
    for path in tqdm(label_paths, desc='Reading labels'):
        try:
            with open(path, 'r') as f:
                for line in f.readlines():
                    line = line.strip().split()
                    if len(line) >= 5:
                        w, h = float(line[3]), float(line[4])
                        shapes.append([w, h])
        except:
            continue
    return torch.tensor(shapes)

def kmeans_anchors(data, n=9, img_size=640, generations=1000):
    from scipy.cluster.vq import kmeans
    from sklearn.cluster import KMeans

    print(f"\nRunning kmeans for {n} anchors on {len(data)} boxes...")

    # un-normalize by image size
    data = data * img_size

    # k-means using sklearn
    kmeans = KMeans(n_clusters=n, random_state=0, n_init="auto").fit(data)
    anchors = np.array(sorted(kmeans.cluster_centers_, key=lambda x: x[0]*x[1]))

    anchors_torch = torch.tensor(anchors, dtype=torch.float32)
    ratios = data[:, None] / anchors_torch[None]
    best_possible_recall = torch.min(ratios, 1 / ratios).max(2)[0].gt(0.5).float().mean()
    print(f"\nAnchors:\n{np.round(anchors).astype(int).tolist()}")
    print(f"Best Possible Recall (BPR): {best_possible_recall * 100:.2f}%")

    return anchors

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, help='path to dataset.yaml')
    parser.add_argument('--img', type=int, default=640, help='input image size')
    parser.add_argument('--n', type=int, default=9, help='number of anchors')
    opt = parser.parse_args()

    data = load_dataset(opt.data)
    anchors = kmeans_anchors(data, opt.n, opt.img)
