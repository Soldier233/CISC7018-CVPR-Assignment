import cv2
import numpy as np
import matplotlib.pyplot as plt
import requests
import tarfile
import os
import time


# ============================
# 1. 数据管理模块
# ============================
class DataManager:
    def __init__(self, data_dir="data_graf"):
        self.data_dir = data_dir
        self.url = "https://www.robots.ox.ac.uk/~vgg/research/affine/det_eval_files/graf.tar.gz"
        self.filename = "graf.tar.gz"

    def download_and_extract(self):
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

        file_path = os.path.join(self.data_dir, self.filename)

        # 检查是否已存在关键文件
        if os.path.exists(os.path.join(self.data_dir, "img6.ppm")):
            print(f"[Info] Dataset found in '{self.data_dir}'. Ready to run.")
            return

        print(f"[Info] Downloading dataset...")
        try:
            response = requests.get(self.url, stream=True)
            if response.status_code == 200:
                with open(file_path, 'wb') as f:
                    f.write(response.content)
            else:
                print(f"[Error] Download failed.")
                return
        except Exception as e:
            print(f"[Error] Download error: {e}")
            return

        print("[Info] Extracting...")
        try:
            tar = tarfile.open(file_path, "r:gz")
            tar.extractall(path=self.data_dir)
            tar.close()
        except Exception as e:
            print(f"[Error] Extraction error: {e}")


# ============================
# 2. 核心算法模块
# ============================
class FeatureMatcher:
    def __init__(self, method='SIFT'):
        self.method = method
        if method == 'SIFT':
            self.detector = cv2.SIFT_create()
            self.norm_type = cv2.NORM_L2
        elif method == 'ORB':
            self.detector = cv2.ORB_create(nfeatures=2000)
            self.norm_type = cv2.NORM_HAMMING

    def run_and_draw(self, img1, img2):
        t0 = time.time()
        kp1, des1 = self.detector.detectAndCompute(img1, None)
        kp2, des2 = self.detector.detectAndCompute(img2, None)

        good_matches = []
        if des1 is not None and des2 is not None:
            bf = cv2.BFMatcher(self.norm_type)
            if self.method == 'SIFT':
                matches = bf.knnMatch(des1, des2, k=2)
                for m, n in matches:
                    if m.distance < 0.75 * n.distance:
                        good_matches.append(m)
            else:
                bf = cv2.BFMatcher(self.norm_type, crossCheck=True)
                matches = bf.match(des1, des2)
                # ORB按距离排序，只取前100个避免画面太乱，或者取全部看效果
                good_matches = sorted(matches, key=lambda x: x.distance)[:100]

        t_total = time.time() - t0

        # 绘制连线图
        # flags=2 (NOT_DRAW_SINGLE_POINTS) 只画匹配线，不画孤立点
        res_img = cv2.drawMatches(
            img1, kp1, img2, kp2, good_matches, None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )

        # OpenCV 是 BGR，Matplotlib 是 RGB，这里转换一下颜色以便显示正确
        res_img = cv2.cvtColor(res_img, cv2.COLOR_BGR2RGB)

        return len(good_matches), t_total, res_img


# ============================
# 3. 主程序：批量处理与网格展示
# ============================
def main():
    dm = DataManager()
    dm.download_and_extract()

    img1 = cv2.imread(os.path.join(dm.data_dir, "img1.ppm"), cv2.IMREAD_GRAYSCALE)

    # 存储数据的列表
    stats = {'labels': [], 'sift_count': [], 'orb_count': [], 'sift_time': [], 'orb_time': []}
    # 存储图像的列表
    visuals = []

    print("Processing images... Please wait.")

    # 遍历 img2 到 img6
    for i in range(2, 7):
        img_target = cv2.imread(os.path.join(dm.data_dir, f"img{i}.ppm"), cv2.IMREAD_GRAYSCALE)
        if img_target is None: continue

        pair_label = f"1-{i}"
        stats['labels'].append(pair_label)

        # SIFT
        matcher_sift = FeatureMatcher('SIFT')
        n_s, t_s, img_s = matcher_sift.run_and_draw(img1, img_target)
        stats['sift_count'].append(n_s)
        stats['sift_time'].append(t_s)

        # ORB
        matcher_orb = FeatureMatcher('ORB')
        n_o, t_o, img_o = matcher_orb.run_and_draw(img1, img_target)
        stats['orb_count'].append(n_o)
        stats['orb_time'].append(t_o)

        visuals.append((pair_label, img_s, img_o))
        print(f"Finished pair {pair_label}")

    # --- 窗口 1: 统计折线图 ---
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(stats['labels'], stats['sift_count'], 'b-o', label='SIFT')
    plt.plot(stats['labels'], stats['orb_count'], 'r--x', label='ORB')
    plt.title("Match Count (Robustness)")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(stats['labels'], stats['sift_time'], 'b-o', label='SIFT')
    plt.plot(stats['labels'], stats['orb_time'], 'r--x', label='ORB')
    plt.title("Time Consumption (Efficiency)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # --- 窗口 2: 视觉效果对比网格 (5行 x 2列) ---
    # 创建一个大图
    fig, axes = plt.subplots(5, 2, figsize=(10, 15))
    fig.suptitle("Visual Comparison: SIFT (Left) vs ORB (Right)", fontsize=16)

    for idx, (label, img_s, img_o) in enumerate(visuals):
        # 左列 SIFT
        axes[idx, 0].imshow(img_s)
        axes[idx, 0].set_title(f"SIFT Pair {label} (Matches: {stats['sift_count'][idx]})", fontsize=9)
        axes[idx, 0].axis('off')

        # 右列 ORB
        axes[idx, 1].imshow(img_o)
        axes[idx, 1].set_title(f"ORB Pair {label} (Matches: {stats['orb_count'][idx]})", fontsize=9)
        axes[idx, 1].axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()