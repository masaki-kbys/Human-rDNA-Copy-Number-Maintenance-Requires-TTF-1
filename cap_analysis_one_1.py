import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from skimage import io, filters, morphology, measure, color, exposure, segmentation, feature
from skimage.morphology import disk
from skimage.filters import threshold_li, threshold_otsu
from skimage.transform import resize
from scipy import ndimage as ndi
from scipy.ndimage import gaussian_filter1d
from cellpose import models
import os
import glob
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")


# =============================================================================
# 0. 共通ヘルパー (Scaling)
# =============================================================================

def scaled_params(img, param):
    """画像サイズ(高さ)に応じてパラメータをスケーリング (基準: 1024px)"""
    if img is None: return param

    scale = img.shape[0] / 1024.0

    if (scale < 1):
        print("<1024px image")
    return param * scale


# =============================================================================
# PART 1: Cap Detection Logic
# =============================================================================

def get_masked_data(region, intensity_image, override_mask=None):
    minr, minc, maxr, maxc = region.bbox
    sub_image = intensity_image[minr:maxr, minc:maxc]
    sub_mask = override_mask if override_mask is not None else region.image
    if sub_mask.shape != sub_image.shape:
        return sub_image, region.image, sub_image[region.image]
    return sub_image, sub_mask, sub_image[sub_mask]


def refine_region_mask(region, intensity_image):
    """Li法 + Watershed によるリファイン"""
    sub_image, sub_mask, pixel_values = get_masked_data(region, intensity_image)
    if len(pixel_values) == 0: return sub_mask, region.area

    # Scale parameters
    min_dist_val = max(1, int(scaled_params(intensity_image, 2.0)))

    try:
        try:
            thresh_val = threshold_li(pixel_values)
        except:
            thresh_val = threshold_otsu(pixel_values)
        refined_binary = sub_mask & (sub_image > thresh_val)
        if np.sum(refined_binary) == 0: return sub_mask, region.area

        distance = ndi.distance_transform_edt(refined_binary)
        coords = feature.peak_local_max(distance, min_distance=min_dist_val, labels=refined_binary)
        local_mask = np.zeros(distance.shape, dtype=bool)
        if len(coords) > 0: local_mask[tuple(coords.T)] = True
        markers, _ = ndi.label(local_mask)
        local_ws = segmentation.watershed(-distance, markers, mask=refined_binary)

        props = measure.regionprops(local_ws)
        if not props: return sub_mask, region.area
        main = max(props, key=lambda x: x.area)
        return (local_ws == main.label), main.area
    except:
        return sub_mask, region.area


def calculate_all_features(region, image, g_max, ref_mask):
    """全特徴量を一括計算"""
    sub_img, sub_mask, pixels = get_masked_data(region, image, override_mask=ref_mask)
    if len(pixels) == 0: return None

    # Scale parameters
    sigma_val = scaled_params(image, 0.5)
    min_dist_val = max(1, int(scaled_params(image, 3.0)))

    area = np.sum(ref_mask)
    perim = measure.perimeter(ref_mask)
    circ = (4 * np.pi * area) / (perim ** 2) if perim > 0 else 0

    mask_int = ref_mask.astype(np.uint8)
    props_ref = measure.regionprops(mask_int)
    solidity = props_ref[0].solidity if props_ref else 0

    ratio = np.mean(pixels) / np.max(pixels)
    gri = np.max(pixels) / g_max
    cv = np.std(pixels) / np.mean(pixels)

    smooth = filters.gaussian(sub_img * sub_mask, sigma=sigma_val)
    mean_v = np.mean(pixels)
    peaks = feature.peak_local_max(smooth, min_distance=min_dist_val, threshold_abs=mean_v, exclude_border=False)
    n_peaks = len(peaks)

    return {
        'Area': area, 'Circularity': circ, 'Solidity': solidity,
        'Ratio': ratio, 'GRI': gri, 'CV': cv, 'Peaks': n_peaks,
        'Mean_Int': mean_v, 'Total_Int': np.sum(pixels)
    }


def determine_global_thresholds(all_particles_list):
    """全画像分のデータから閾値を自動決定"""
    if not all_particles_list:
        return 30, 0.2, 0.88, 0.55, 0.70

    areas = [p['Area'] for p in all_particles_list]
    circs = [p['Circularity'] for p in all_particles_list]
    sols = [p['Solidity'] for p in all_particles_list]
    ratios = [p['Ratio'] for p in all_particles_list]
    gris = [p['GRI'] for p in all_particles_list]

    def _find_l2r_area(vals, min_v, cut_off_ratio=0.1):
        """
        左（最小値）から累積して、分布全体の cut_off_ratio (0.0~1.0) に達する値を返す。
        例: 0.1なら下位10%点を閾値とする。0.5なら中央値。
        """
        if not vals: return min_v

        # パーセンタイルを計算 (cut_off_ratio * 100)
        threshold = np.percentile(vals, cut_off_ratio * 100)

        return max(threshold, min_v)

    def _find_l2r_subpeak(vals, min_v, cut_off_ratio=0.05):
        counts, edges = np.histogram(vals, bins=100)
        centers = (edges[:-1] + edges[1:]) / 2
        peak_idx = np.argmax(counts)
        cutoff = counts[peak_idx] * cut_off_ratio
        thresh = centers[-1]
        for i in range(peak_idx + 1, len(counts)):
            if counts[i] < cutoff:
                thresh = centers[i];
                break
        return max(thresh, min_v)

    def _find_r2l_subpeak(vals, min_v, cut_off_ratio=0.05, max_v=1.0):
        counts, edges = np.histogram(vals, bins=100, range=(0, max_v))
        centers = (edges[:-1] + edges[1:]) / 2
        peak_idx = np.argmax(counts)
        mountain = counts[peak_idx] * cut_off_ratio
        thresh = min_v
        for i in range(len(counts) - 1, peak_idx, -1):
            if counts[i] > mountain:
                thresh = centers[i];
                break
        return max(thresh, min_v)

    def find_peak(vals, sigma=4.0):
        if len(vals) == 0: return 0.0
        counts, bin_edges = np.histogram(vals, bins=100)
        centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        smooth_counts = gaussian_filter1d(counts, sigma=sigma)
        peak_idx = np.argmax(smooth_counts)
        return centers[peak_idx]

    th_area = _find_l2r_subpeak(areas, 30, 0.1)
    th_circ = _find_r2l_subpeak(circs, 0.2, 0.1, 0.9)
    th_sol = _find_r2l_subpeak(sols, 0.80, 0.1)
    th_rat = find_peak(sols, 0.80, 0.1)
    th_gri = find_peak(gris)

    return th_area, th_circ, th_sol, th_rat, th_gri


class CapResultCurator:
    def __init__(self, filename, image, ws_labels, particles):
        self.filename = filename
        self.image = image
        self.ws_labels = ws_labels
        self.particles = particles
        self.g_max = np.percentile(image, 99.9)
        self.show_masked_bg = True

        self.ref_map_all = np.zeros_like(image, dtype=bool)
        for p in self.particles:
            if 'mask_local' in p:
                minr, minc, maxr, maxc = p['bbox']
                self.ref_map_all[minr:maxr, minc:maxc] = np.maximum(
                    self.ref_map_all[minr:maxr, minc:maxc], p['mask_local']
                )

        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.fig.canvas.manager.set_window_title(f"Cap Curator: {filename}")
        self.cid_click = self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        self.cid_key = self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.update_plot()

    def update_plot(self):
        self.ax.clear()
        if self.show_masked_bg:
            bg_img = self.image.copy()
            bg_img[~self.ref_map_all] = 0
            title_mode = "View: Masked (Press 'C')"
        else:
            bg_img = self.image
            title_mode = "View: Raw (Press 'C')"
        self.ax.imshow(bg_img, cmap='gray')

        accepted_mask = np.zeros_like(self.image, dtype=bool)
        count = 0
        for p in self.particles:
            if p['status'] in ['auto_cap', 'manual_cap']:
                count += 1
                minr, minc, maxr, maxc = p['bbox']
                accepted_mask[minr:maxr, minc:maxc] = np.maximum(
                    accepted_mask[minr:maxr, minc:maxc], p['mask_local']
                )

        if count > 0:
            overlay = np.zeros((self.image.shape[0], self.image.shape[1], 4))
            overlay[accepted_mask, 0] = 1.0  # R
            overlay[accepted_mask, 3] = 0.35  # Alpha
            self.ax.imshow(overlay)

        self.ax.set_title(f"{self.filename} (Caps: {count})\n{title_mode}\nLeft: Add (Recalc) / Right: Remove")
        self.ax.axis('off')
        plt.draw()

    def on_key(self, event):
        if event.key.lower() == 'c':
            self.show_masked_bg = not self.show_masked_bg
            self.update_plot()

    def onclick(self, event):
        if event.inaxes != self.ax: return
        y, x = event.ydata, event.xdata

        # Scale for click
        click_radius = scaled_params(self.image, 30.0)

        if event.button == 3:  # Remove
            min_dist, target_p = 1e9, None
            for p in self.particles:
                if p['status'] not in ['auto_cap', 'manual_cap']: continue
                minr, minc, maxr, maxc = p['bbox']
                cy, cx = (minr + maxr) / 2, (minc + maxc) / 2
                dist = (cy - y) ** 2 + (cx - x) ** 2
                if dist < min_dist: min_dist, target_p = dist, p

            if target_p and np.sqrt(min_dist) < click_radius:
                target_p['status'] = 'manual_removed'
                self.update_plot()

        elif event.button == 1:  # Add
            iy, ix = int(y), int(x)
            if not (0 <= iy < self.image.shape[0] and 0 <= ix < self.image.shape[1]): return
            label = self.ws_labels[iy, ix]
            if label == 0: return

            existing = next(
                (p for p in self.particles if p['label_id'] == label and p['status'] in ['auto_cap', 'manual_cap']),
                None)
            if existing: return

            target_mask = (self.ws_labels == label).astype(int)
            props = measure.regionprops(target_mask, intensity_image=self.image)
            if not props: return
            r = props[0]

            ref_mask, ref_area = refine_region_mask(r, self.image)
            feats = calculate_all_features(r, self.image, self.g_max, ref_mask)

            if feats:
                target_p = next((p for p in self.particles if p['label_id'] == label), None)
                new_data = {
                    'label_id': label, 'bbox': r.bbox, 'mask_local': ref_mask,
                    **feats, 'manual_add': True, 'status': 'manual_cap'
                }
                if target_p:
                    target_p.update(new_data)
                else:
                    self.particles.append(new_data)

                minr, minc, maxr, maxc = r.bbox
                self.ref_map_all[minr:maxr, minc:maxc] = np.maximum(
                    self.ref_map_all[minr:maxr, minc:maxc], ref_mask
                )
                self.update_plot()


# =============================================================================
# PART 2: Nuclei Integration Logic
# =============================================================================

def run_cellpose_nuclei(img, diameter=None, use_gpu=True):
    # Scale parameter
    if diameter is None: diameter = scaled_params(img, 40)

    try:
        model = models.Cellpose(model_type='nuclei', gpu=use_gpu)
        results = model.eval(img, diameter=diameter, channels=[0, 0])
        masks = results[0]
    except AttributeError:
        # Fallback
        model = models.CellposeModel(model_type='nuclei', gpu=use_gpu)
        results = model.eval(img, diameter=diameter, channels=[0, 0])
        masks = results[0]
    except Exception as e:
        print(f"Cellpose Error: {e}")
        return None
    return masks


def segment_local_nucleus(image, center_y, center_x, radius=None):
    """
    指定座標を中心に局所的なOtsu法で核を検出 (境界チェック付き修正版)
    """
    from skimage import filters
    from scipy import ndimage as ndi

    h, w = image.shape
    scale = h / 1024.0

    # 半径決定
    if radius is None or radius < 2:
        radius_px = int(30 * scale)
    else:
        radius_px = int(radius)

    # 切り出し範囲
    y1 = max(0, int(center_y - radius_px))
    y2 = min(h, int(center_y + radius_px))
    x1 = max(0, int(center_x - radius_px))
    x2 = min(w, int(center_x + radius_px))

    sub_img = image[y1:y2, x1:x2]
    if sub_img.size == 0: return None, None

    # 円形マスク
    local_cy = center_y - y1
    local_cx = center_x - x1
    Y, X = np.ogrid[:sub_img.shape[0], :sub_img.shape[1]]
    dist_from_center = np.sqrt((X - local_cx) ** 2 + (Y - local_cy) ** 2)
    circular_mask = dist_from_center <= radius_px

    # 前処理
    scaled_sigma = 2.0 * scale
    blurred = filters.gaussian(sub_img, sigma=scaled_sigma)
    pixels_in_circle = blurred[circular_mask]

    if len(pixels_in_circle) == 0: return None, None
    try:
        thresh = filters.threshold_otsu(pixels_in_circle)
    except:
        return None, None

    binary = (blurred > thresh) & circular_mask
    binary = ndi.binary_fill_holes(binary)

    labels = measure.label(binary)
    if labels.max() == 0: return None, None

    # ターゲット選択
    icx, icy = int(local_cx), int(local_cy)
    icx = min(max(0, icx), sub_img.shape[1] - 1)
    icy = min(max(0, icy), sub_img.shape[0] - 1)
    target_label = labels[icy, icx]

    if target_label == 0:
        props = measure.regionprops(labels)
        if not props: return None, None
        target_prop = min(props, key=lambda r: (r.centroid[0] - local_cy) ** 2 + (r.centroid[1] - local_cx) ** 2)
    else:
        props = measure.regionprops(labels)
        target_prop = next((r for r in props if r.label == target_label), None)

    if target_prop is None: return None, None

    # --- ★修正: 安全な貼り付け処理 (Safe Paste) ---
    global_mask = np.zeros_like(image, dtype=bool)
    minr, minc, maxr, maxc = target_prop.bbox

    # グローバル座標（計算上の貼り付け位置）
    gr_min = y1 + minr
    gr_max = y1 + maxr
    gc_min = x1 + minc
    gc_max = x1 + maxc

    # 画像範囲内にクリップ (0 〜 h, 0 〜 w に収める)
    valid_r_min = max(0, gr_min)
    valid_r_max = min(h, gr_max)
    valid_c_min = max(0, gc_min)
    valid_c_max = min(w, gc_max)

    # 貼り付けサイズが0なら終了
    if valid_r_max <= valid_r_min or valid_c_max <= valid_c_min:
        return None, None

    # 元画像(target_prop.image)側の切り出し位置を計算
    # (グローバル座標ではみ出した分だけ、元画像の端もカットする)
    src_r_start = valid_r_min - gr_min
    src_r_end = src_r_start + (valid_r_max - valid_r_min)
    src_c_start = valid_c_min - gc_min
    src_c_end = src_c_start + (valid_c_max - valid_c_min)

    # 安全に代入
    global_mask[valid_r_min:valid_r_max, valid_c_min:valid_c_max] = \
        target_prop.image[src_r_start:src_r_end, src_c_start:src_c_end]

    return global_mask, target_prop.area


class NucleiCurator:
    def __init__(self, filename, image, mask):
        self.filename = filename
        self.image = image
        self.mask = mask.copy()
        self.max_id = np.max(self.mask)
        self.show_masked = True
        self.press_start = None
        self.is_dragging = False
        self.history = []

        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.fig.canvas.manager.set_window_title(f"Nuclei Curator: {filename}")
        self.cid_press = self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.cid_release = self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.cid_key = self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.update_plot()

    def save_state(self):
        if len(self.history) > 10: self.history.pop(0)
        self.history.append({'mask': self.mask.copy(), 'max_id': self.max_id})

    def undo(self):
        if not self.history: return
        state = self.history.pop()
        self.mask = state['mask']
        self.max_id = state['max_id']
        print("  Undone.")
        self.update_plot()

    def update_plot(self):
        self.ax.clear()
        bg = self.image.copy()
        if self.show_masked:
            bg[self.mask == 0] = 0
            title_mode = "View: Masked (Press 'C')"
        else:
            title_mode = "View: Raw (Press 'C')"
        self.ax.imshow(bg, cmap='gray')

        if np.max(self.mask) > 0:
            overlay = np.zeros(self.image.shape + (4,))
            overlay[self.mask > 0] = [1, 1, 0, 0.3]  # Yellow
            self.ax.imshow(overlay)
            for c in measure.find_contours(self.mask > 0, 0.5):
                self.ax.plot(c[:, 1], c[:, 0], color='yellow', linewidth=0.5, alpha=0.8)

        count = len(np.unique(self.mask)) - 1
        self.ax.set_title(
            f"{self.filename} (Nuclei: {count})\n{title_mode}\nL-Drag: Add (Radius) / R-Click: Remove / 'Z': Undo")
        self.ax.axis('off')
        plt.draw()

    def on_key(self, event):
        if event.key.lower() == 'c':
            self.show_masked = not self.show_masked
            self.update_plot()
        elif event.key.lower() == 'z':
            self.undo()

    def on_press(self, event):
        if event.inaxes != self.ax: return
        if event.button == 3:  # Remove
            y, x = int(event.ydata), int(event.xdata)
            target_label = self.mask[y, x]
            if target_label > 0:
                self.save_state()
                self.mask[self.mask == target_label] = 0
                print(f"  Removed ID {target_label}")
                self.update_plot()
        elif event.button == 1:  # Start Drag
            self.press_start = (event.xdata, event.ydata)
            self.is_dragging = True

    def on_release(self, event):
        if event.button == 1 and self.is_dragging and self.press_start:
            self.is_dragging = False
            if event.inaxes != self.ax: return
            x1, y1 = self.press_start
            x2, y2 = event.xdata, event.ydata
            radius = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            cy, cx = int(y1), int(x1)

            if self.mask[cy, cx] > 0: return

            print(f"  Detecting... (R={radius:.1f})")
            new_mask, area = segment_local_nucleus(self.image, cy, cx, radius)
            if new_mask is not None:
                overlap = (self.mask > 0) & new_mask
                if np.sum(overlap) / np.sum(new_mask) > 0.5: return
                self.save_state()
                self.max_id += 1
                self.mask[new_mask] = self.max_id
                print(f"  Added ID {self.max_id} (Area: {area})")
                self.update_plot()
            self.press_start = None


# =============================================================================
# 3. 出力用関数 (Outputs)
# =============================================================================

def create_global_stats_dashboard(all_particles, thresholds, output_dir):
    th_area, th_circ, th_sol, th_rat, th_gri = thresholds
    params = ['Area', 'Circularity', 'Solidity', 'Ratio', 'GRI', 'CV', 'Peaks']
    thresholds_dict = {'Area': th_area, 'Circularity': th_circ, 'Solidity': th_sol, 'Ratio': th_rat, 'GRI': th_gri}
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    for i, param in enumerate(params):
        vals = [p[param] for p in all_particles]
        ax = axes[i]
        if param == 'Area':
            ax.hist(vals, bins=50, color='gray', alpha=0.7, log=True)
        else:
            ax.hist(vals, bins=50, color='gray', alpha=0.7)
        if thresholds_dict.get(param):
            ax.axvline(thresholds_dict[param], color='red', linestyle='--')
        ax.set_title(f"Global {param}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "global_stats.png"), dpi=100)
    plt.close()


def generate_cap_outputs(filename, image, ws_labels, particles, thresholds, output_dirs):
    d_img, d_stat, d_mask, d_pol1 = output_dirs
    th_area, th_circ, th_sol, th_rat, th_gri = thresholds

    # 1. Image Dashboard
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    axes[0, 0].imshow(image, cmap='gray');
    axes[0, 0].set_title("Raw");
    axes[0, 0].axis('off')

    masked_v = image.copy();
    masked_v[ws_labels == 0] = 0
    masked_rgb = color.gray2rgb(masked_v)
    bnds = segmentation.find_boundaries(ws_labels, mode='thick')
    masked_rgb[bnds] = [1, 0, 0]
    axes[0, 1].imshow(masked_rgb);
    axes[0, 1].set_title("Segmentation Base");
    axes[0, 1].axis('off')

    # Save Segmentation Base (Pol1 whole area check)
    io.imsave(os.path.join(d_img, f"{os.path.splitext(filename)[0]}_seg_check.png"),
              (masked_rgb * 255).astype(np.uint8), check_contrast=False)

    ref_map = np.zeros_like(image, dtype=bool)
    accepted_caps = []
    for p in particles:
        minr, minc, maxr, maxc = p['bbox']
        ref_map[minr:maxr, minc:maxc] = np.maximum(ref_map[minr:maxr, minc:maxc], p['mask_local'])
        if p['status'] in ['auto_cap', 'manual_cap']: accepted_caps.append(p)
    axes[1, 0].imshow(ref_map, cmap='gray');
    axes[1, 0].set_title("Refined Masks");
    axes[1, 0].axis('off')

    display_map = np.zeros_like(image, dtype=np.uint8)
    for p in particles:
        code = 0
        s = p['status']
        if s == 'shape_ng':
            code = 1
        elif s == 'struct_ng':
            code = 2
        elif s == 'auto_cap':
            code = 3
        elif s == 'manual_cap':
            code = 4
        elif s == 'manual_removed':
            code = 5
        if code > 0:
            minr, minc, maxr, maxc = p['bbox']
            display_map[minr:maxr, minc:maxc] = np.maximum(display_map[minr:maxr, minc:maxc], code * p['mask_local'])

    img_disp = exposure.rescale_intensity(image, out_range='float')
    p_lbls = np.unique(display_map);
    p_lbls = p_lbls[p_lbls != 0]
    cmap = {1: 'blue', 2: 'orange', 3: 'red', 4: 'magenta', 5: 'lime'}
    colors = [cmap[l] for l in p_lbls]
    if colors:
        ov = color.label2rgb(display_map, image=img_disp, bg_label=0, colors=colors, alpha=0.5)
    else:
        ov = color.gray2rgb(img_disp)
    axes[1, 1].imshow(ov);
    axes[1, 1].set_title("Final Result");
    axes[1, 1].axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(d_img, f"dashboard_{filename}.png"), dpi=100)
    plt.close()

    # 2. Local Stats
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    params = ['Area', 'Circularity', 'Solidity', 'Ratio', 'GRI', 'CV', 'Peaks']
    thresholds_dict = {'Area': th_area, 'Circularity': th_circ, 'Solidity': th_sol, 'Ratio': th_rat, 'GRI': th_gri}
    for i, param in enumerate(params):
        vals = [p[param] for p in particles]
        ax = axes[i]
        if param == 'Area':
            ax.hist(vals, bins=20, color='gray', alpha=0.7, log=True)
        else:
            ax.hist(vals, bins=20, color='gray', alpha=0.7)
        if thresholds_dict.get(param): ax.axvline(thresholds_dict[param], color='red', linestyle='--')
        ax.set_title(f"Local {param}")
    plt.tight_layout()
    plt.savefig(os.path.join(d_stat, f"stat_{filename}.png"), dpi=100)
    plt.close()

    # 3. Binary Cap Mask
    binary_mask = np.zeros(image.shape, dtype=np.uint8)
    for p in accepted_caps:
        minr, minc, maxr, maxc = p['bbox']
        sub_mask = binary_mask[minr:maxr, minc:maxc]
        sub_mask[p['mask_local']] = 255
    io.imsave(os.path.join(d_mask, f"{os.path.splitext(filename)[0]}_mask.png"), binary_mask, check_contrast=False)

    # 4. Pol1 Mask (Segmentation Base for Integration)
    pol1_binary = (ws_labels > 0).astype(np.uint8) * 255
    io.imsave(os.path.join(d_pol1, f"{os.path.splitext(filename)[0]}_pol1_mask.png"), pol1_binary, check_contrast=False)

    return accepted_caps


# =============================================================================
# MAIN PIPELINES
# =============================================================================

def run_cap_detection_pipeline(caps_dir, output_root):
    """PART 1: Cap Analysis Pipeline"""
    print("\n--- Starting Cap Detection Pipeline ---")

    # Output Dirs
    d_base = os.path.join(output_root, "cap_analysis")
    d_img = os.path.join(d_base, "image")
    d_stat = os.path.join(d_base, "stat")
    d_mask = os.path.join(d_base, "cap-detected")
    d_pol1 = os.path.join(d_base, "all-pol1")
    for d in [d_img, d_stat, d_mask, d_pol1]: os.makedirs(d, exist_ok=True)

    tophat_radius = 50

    exts = ['*.tif', '*.tiff', '*.jpg', '*.jpeg', '*.png']
    files = []
    for e in exts: files.extend(glob.glob(os.path.join(caps_dir, e)))
    if not files: return None, None

    # Pass 1: Global Stats
    print("Pass 1: Analyzing Images & Global Thresholding...")
    cache = {}
    all_particles = []

    for f in tqdm(files):
        fname = os.path.basename(f)
        img = io.imread(f)
        if img.ndim == 3: img = color.rgb2gray(img)
        g_max = np.percentile(img, 99.9)

        # Segmentation
        scale = img.shape[0] / 1024.0
        blurred = filters.gaussian(img, sigma=scaled_params(img, 1.0))
        tophat = morphology.white_tophat(blurred, disk(int(scaled_params(img, tophat_radius))))
        thresh = filters.threshold_otsu(tophat)
        binary = tophat > thresh
        coords = feature.peak_local_max(tophat, min_distance=int(scaled_params(img, 4)), labels=binary)
        mask = np.zeros(tophat.shape, dtype=bool)
        if len(coords) > 0: mask[tuple(coords.T)] = True
        markers, _ = ndi.label(mask)
        ws_labels = segmentation.watershed(-tophat, markers, mask=binary)

        regions = measure.regionprops(ws_labels, intensity_image=img)
        particles = []
        for r in regions:
            ref_mask, _ = refine_region_mask(r, img)
            feats = calculate_all_features(r, img, g_max, ref_mask)
            if not feats: continue
            particles.append({
                'label_id': r.label, 'bbox': r.bbox, 'mask_local': ref_mask,
                **feats, 'manual_add': False, 'status': 'pending'
            })

        cache[fname] = {'image': img, 'ws_labels': ws_labels, 'particles': particles}
        all_particles.extend(particles)

    thresholds = determine_global_thresholds(all_particles)
    create_global_stats_dashboard(all_particles, thresholds, d_base)
    th_area, th_circ, th_sol, th_rat, th_gri = thresholds
    MAX_CV, MAX_PEAKS = 0.25, 6

    # Pass 2: Curation & Output
    print("\nPass 2: Curation & Output...")
    all_final_data = []

    for fname, data in cache.items():
        # Filter
        for p in data['particles']:
            pass_shape = (p['Area'] >= th_area)
            pass_struct = (p['Ratio'] >= th_rat) and (p['GRI'] >= th_gri) and \
                          (p['Solidity'] >= th_sol) and (p['CV'] <= MAX_CV) and \
                          (p['Peaks'] <= MAX_PEAKS)
            if not pass_shape:
                p['status'] = 'shape_ng'
            elif not pass_struct:
                p['status'] = 'struct_ng'
            else:
                p['status'] = 'auto_cap'

        print(f"Processing: {fname}")
        curator = CapResultCurator(fname, data['image'], data['ws_labels'], data['particles'])
        plt.show()

        accepted = generate_cap_outputs(fname, data['image'], data['ws_labels'],
                                        data['particles'], thresholds, (d_img, d_stat, d_mask, d_pol1))

        for i, p in enumerate(accepted):
            row = {'Filename': fname, 'Cap_ID': i + 1, 'Manual_Add': (p['status'] == 'manual_cap'),
                   **{k: v for k, v in p.items() if
                      k not in ['mask_local', 'bbox', 'label_id', 'status', 'manual_add']},
                   'Centroid_Y': (p['bbox'][0] + p['bbox'][2]) / 2, 'Centroid_X': (p['bbox'][1] + p['bbox'][3]) / 2}
            all_final_data.append(row)

    if all_final_data:
        pd.DataFrame(all_final_data).to_csv(os.path.join(d_base, "final_data.csv"), index=False)

    return d_mask, d_pol1  # Return output dirs for next step


def run_nuclei_integration_pipeline(dapi_dir, cap_mask_dir, pol1_mask_dir, output_root):
    """PART 2: Nuclei Integration Pipeline"""
    print("\n--- Starting Nuclei Integration Pipeline ---")

    d_base = os.path.join(output_root, "integration_analysis")
    d_nuc_mask = os.path.join(d_base, "nuclei_masks")
    d_vis = os.path.join(d_base, "visualizations")
    d_temp = os.path.join(d_base, "temp_cellpose")
    for d in [d_nuc_mask, d_vis, d_temp]: os.makedirs(d, exist_ok=True)


    files = sorted(glob.glob(os.path.join(dapi_dir, "*.tif")) + glob.glob(os.path.join(dapi_dir, "*.jpg")))
    if not files: return

    # Phase 1: Cellpose Batch
    print("Phase 1: Running Cellpose...")
    file_mask_pairs = []
    for f in tqdm(files):
        fname = os.path.basename(f)
        temp_path = os.path.join(d_temp, f"{os.path.splitext(fname)[0]}_auto.npy")
        if os.path.exists(temp_path):
            file_mask_pairs.append((fname, f, temp_path))
            continue

        img = io.imread(f)
        if img.ndim == 3: img = color.rgb2gray(img)
        masks = run_cellpose_nuclei(img, diameter=None, use_gpu=True)

        if masks is not None:
            np.save(temp_path, masks)
            file_mask_pairs.append((fname, f, temp_path))
        else:
            print(f"Skipping {fname} due to Cellpose error.")

    # Phase 2: Curation
    print("\nPhase 2: Manual Curation...")
    curated_masks = {}
    for fname, img_path, mask_path in file_mask_pairs:
        print(f"Curating: {fname}")
        img = io.imread(img_path)

        if img.ndim == 3: img = color.rgb2gray(img)
        mask = np.load(mask_path)

        curator = NucleiCurator(fname, img, mask)
        plt.show()

        curated_masks[fname] = curator.mask
        io.imsave(os.path.join(d_nuc_mask, f"{os.path.splitext(fname)[0]}_nuclei.tif"),
                  curator.mask.astype(np.uint16), check_contrast=False)

    # Phase 3: Integration (Pol1 Filter: 1px overlap)
    print("\nPhase 3: Integration Analysis (Pol1 Filter)...")
    summary_data = []

    for fname, img_path, _ in file_mask_pairs:
        dapi_img = io.imread(img_path)
        if dapi_img.ndim == 3: dapi_img = color.rgb2gray(dapi_img)
        nuc_mask = curated_masks[fname]

        target_shape = dapi_img.shape # 基準サイズを取得
        base_name = os.path.splitext(fname)[0]

        # --- 1. Load Cap Mask & Resize ---
        # ファイル検索
        cap_path = os.path.join(cap_mask_dir, f"{base_name}_mask.png")
        if not os.path.exists(cap_path): cap_path = os.path.join(cap_mask_dir, f"{base_name}_mask.tif")

        if os.path.exists(cap_path):
            cap_mask = io.imread(cap_path) > 0

            # ★追加: サイズ不一致ならリサイズ (Nearest Neighborで二値性を維持)
            if cap_mask.shape != target_shape:
                # print(f"  Resizing Cap Mask: {cap_mask.shape} -> {target_shape}")
                cap_mask = resize(cap_mask, target_shape, order=0, preserve_range=True, anti_aliasing=False).astype(
                    bool)
        else:
            # print(f"Warning: Cap mask not found for {fname}")
            cap_mask = np.zeros(target_shape, dtype=bool)

        # --- 2. Load Pol1 Mask & Resize ---
        pol1_path = os.path.join(pol1_mask_dir, f"{base_name}_pol1_mask.png")

        if os.path.exists(pol1_path):
            pol1_mask_all = io.imread(pol1_path) > 0

            # ★追加: サイズ不一致ならリサイズ
            if pol1_mask_all.shape != target_shape:
                # print(f"  Resizing Pol1 Mask: {pol1_mask_all.shape} -> {target_shape}")
                pol1_mask_all = resize(pol1_mask_all, target_shape, order=0, preserve_range=True,
                                       anti_aliasing=False).astype(bool)
        else:
            print(f"Warning: Pol1 mask not found for {fname}. Using blank.")
            pol1_mask_all = np.zeros(target_shape, dtype=bool)

        # Analyze
        total_caps = measure.label(cap_mask).max()
        nuclei_props = measure.regionprops(nuc_mask)
        pol1_pos_count = 0
        cap_pos_count = 0
        valid_ids, cap_ids = [], []

        for nuc in nuclei_props:
            minr, minc, maxr, maxc = nuc.bbox
            p_sub = pol1_mask_all[minr:maxr, minc:maxc]
            overlap_pol1 = p_sub & nuc.image

            if np.sum(overlap_pol1) > 0:  # Pol1 Filter (1px overlap)
                pol1_pos_count += 1
                valid_ids.append(nuc.label)

                c_sub = cap_mask[minr:maxr, minc:maxc]
                if np.sum(c_sub & nuc.image) > 0:
                    cap_pos_count += 1
                    cap_ids.append(nuc.label)

        ratio = (cap_pos_count / pol1_pos_count) if pol1_pos_count > 0 else 0

        summary_data.append({
            'Filename': fname,
            'Total_Nuclei': len(nuclei_props),
            'Pol1_Pos_Nuclei': pol1_pos_count,
            'Cap_Pos_Nuclei': cap_pos_count,
            'Ratio': ratio,
            'Total_Caps': total_caps
        })

        # Visualization
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(dapi_img, cmap='gray')
        if np.sum(pol1_mask_all) > 0:
            ov = np.zeros(dapi_img.shape + (4,))
            ov[pol1_mask_all] = [1, 1, 0, 0.15]
            ax.imshow(ov)
        if np.sum(cap_mask) > 0:
            ov = np.zeros(dapi_img.shape + (4,))
            ov[cap_mask] = [0, 1, 0, 0.6]
            ax.imshow(ov)

        for nuc in nuclei_props:
            if nuc.label in cap_ids:
                color, lw = 'red', 2.0
            elif nuc.label in valid_ids:
                color, lw = 'cyan', 1.0
            else:
                color, lw = 'gray', 0.5
            for c in measure.find_contours(nuc_mask == nuc.label, 0.5):
                ax.plot(c[:, 1], c[:, 0], color=color, linewidth=lw)

        ax.set_title(f"{fname}\nPol1+: {pol1_pos_count}, Cap+: {cap_pos_count} ({ratio:.1%})")
        ax.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(d_vis, f"result_{fname}.jpg"), dpi=100)
        plt.close()

    if summary_data:
        pd.DataFrame(summary_data).to_csv(os.path.join(d_base, "summary_per_image.csv"), index=False)


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main(input_imageset_dir):
    print(f"Target Directory: {input_imageset_dir}")

    caps_dir = os.path.join(input_imageset_dir, "caps")
    dapi_dir = os.path.join(input_imageset_dir, "dapi")
    output_root = os.path.join(input_imageset_dir, "output")

    if not os.path.exists(caps_dir) or not os.path.exists(dapi_dir):
        print("Error: 'caps' or 'dapi' folder not found in target directory.")
        return

    # Run Pipeline 1: Cap Analysis
    # Returns the paths to generated masks for Pipeline 2
    mask_dir, pol1_dir = run_cap_detection_pipeline(caps_dir, output_root)

    # Run Pipeline 2: Integration
    if mask_dir and pol1_dir:
        run_nuclei_integration_pipeline(dapi_dir, mask_dir, pol1_dir, output_root)

    print("\n--- All Processes Completed Successfully ---")


if __name__ == "__main__":
    # ここに親ディレクトリのパスを指定してください
    target_directory = r"C:/put/the/data/here"
    main(target_directory)