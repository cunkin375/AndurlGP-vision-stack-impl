import os
import sys
import cv2
import numpy as np
import torch
import yaml
from pathlib import Path
from PIL import Image

# Import our models directly since we are in vision-pipeline-impl
from gate_vision.model import GOAT
from gate_vision.detect import decode_keypoints
from gate_vision.utils import pick_hardware, preprocess_image
from ultralytics import YOLO

def load_camera_info(path):
    if not os.path.exists(path):
        print(f"Warning: Camera info {path} not found. Using defaults.")
        camera_matrix = np.array([[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]], dtype=np.float32)
        dist_coeffs = np.zeros((5, 1), dtype=np.float32)
        return camera_matrix, dist_coeffs

    with open(path, 'r') as f:
        calib = yaml.safe_load(f)
        camera_matrix = np.array(calib['camera_matrix']['data'], dtype=np.float32).reshape(3, 3)
        dist_coeffs = np.array(calib['distortion_coefficients']['data'], dtype=np.float32)
        return camera_matrix, dist_coeffs

def square_crop_bounds(box, full_w, full_h, pad=0.20):
    x1, y1, x2, y2 = box
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    half = max(bw, bh) * (1.0 + pad) / 2.0
    cx1 = int(max(0, np.floor(cx - half)))
    cy1 = int(max(0, np.floor(cy - half)))
    cx2 = int(min(full_w, np.ceil(cx + half)))
    cy2 = int(min(full_h, np.ceil(cy + half)))
    return cx1, cy1, cx2, cy2

def main(dataset_dir):
    device = pick_hardware()
    print(f"Using device: {device}")

    # Paths
    base_dir = Path(__file__).resolve().parent
    gates_model_path = base_dir / 'models' / 'gates_model.pt'
    corners_model_path = base_dir / 'models' / 'best_model.pth'
    camera_info_path = base_dir / 'config' / 'camera_info.yaml'
    
    output_dir = base_dir / 'test_output'
    output_dir.mkdir(exist_ok=True)

    # Load models
    gates_model = YOLO(str(gates_model_path))
    
    corners_model = GOAT(num_corners=4).to(device)
    corners_model.load_state_dict(torch.load(str(corners_model_path), map_location=device))
    corners_model.eval()

    # Camera + World 
    camera_matrix, dist_coeffs = load_camera_info(str(camera_info_path))
    
    w, h = 1.5, 1.5
    object_points = np.array([
        [-w/2, -h/2, 0.0], # TL
        [ w/2, -h/2, 0.0], # TR
        [-w/2,  h/2, 0.0], # BL
        [ w/2,  h/2, 0.0]  # BR
    ], dtype=np.float32)

    SIZE = 256
    CORNER_THRESH = 0.5
    corner_bgr = {0: (255, 255, 0), 1: (0, 255, 255), 2: (0, 255, 0), 3: (0, 0, 255)}

    if not dataset_dir.exists():
        print(f"Dataset dir not found: {dataset_dir}")
        return

    images = list(dataset_dir.glob('*.jpg')) + list(dataset_dir.glob('*.png'))
    print(f"Found {len(images)} images to process.")

    for img_path in images:
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            continue

        out = bgr.copy()
        h_img, w_img = bgr.shape[:2]

        # 1. Detect Gates
        results = gates_model(bgr, verbose=False)[0]
        boxes = []
        if results.boxes is not None:
            for box in results.boxes.xyxy.cpu().numpy():
                boxes.append(tuple(map(int, box[:4])))

        pnp_results = []

        for (x1, y1, x2, y2) in boxes:
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w_img - 1, x2), min(h_img - 1, y2)
            if x2 <= x1 or y2 <= y1:
                continue

            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # 2. Crop and Detect Corners
            cx1, cy1, cx2, cy2 = square_crop_bounds((x1, y1, x2, y2), w_img, h_img)
            if cx2 - cx1 < 4 or cy2 - cy1 < 4:
                continue

            crop = bgr[cy1:cy2, cx1:cx2]
            pil_img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            tensor = preprocess_image(pil_img, SIZE).to(device)

            with torch.no_grad():
                heatmaps = corners_model(tensor).cpu()

            decoded = decode_keypoints(heatmaps, CORNER_THRESH)[0]

            cw, ch = (cx2 - cx1), (cy2 - cy1)
            gate_corners = {
                c: (cx_n * cw + cx1, cy_n * ch + cy1, score)
                for c, (cx_n, cy_n, score) in decoded.items()
            }

            # Draw Corners
            for c_type, (cx, cy, score) in gate_corners.items():
                px, py = int(cx), int(cy)
                cv2.circle(out, (px, py), 4, corner_bgr.get(c_type, (255, 0, 0)), -1)

            # 3. Solve PnP
            if len(gate_corners) == 4:
                image_points = np.array([
                    [gate_corners[0][0], gate_corners[0][1]], # TL
                    [gate_corners[1][0], gate_corners[1][1]], # TR
                    [gate_corners[2][0], gate_corners[2][1]], # BL
                    [gate_corners[3][0], gate_corners[3][1]]  # BR
                ], dtype=np.float32)

                success, rvec, tvec = cv2.solvePnP(object_points, image_points, camera_matrix, dist_coeffs)

                if success:
                    # Draw 3D axes
                    cv2.drawFrameAxes(out, camera_matrix, dist_coeffs, rvec, tvec, 0.5)

                    x_c, y_c, z_c = tvec.flatten()
                    rx, ry, rz = rvec.flatten()
                    text = f"Pose (Z-fwd): x={x_c:.2f}, y={y_c:.2f}, z={z_c:.2f}"
                    cv2.putText(out, text, (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    pnp_results.append(
                        f"Gate Box: ({x1}, {y1}, {x2}, {y2})\n"
                        f"Translation (x,y,z): {x_c:.4f}, {y_c:.4f}, {z_c:.4f}\n"
                        f"Rotation vec (rx,ry,rz): {rx:.4f}, {ry:.4f}, {rz:.4f}\n"
                    )

        out_path = output_dir / img_path.name
        cv2.imwrite(str(out_path), out)
        
        txt_path = output_dir / (img_path.stem + ".txt")
        if pnp_results:
            with open(txt_path, "w") as f:
                f.write("\n".join(pnp_results))
                
        print(f"Processed {img_path.name} -> {out_path} (PNP: {len(pnp_results)} gates)")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test PNP on gate images')
    parser.add_argument('--dataset_dir', type=str, required=True, help='Path to dataset directory')
    args = parser.parse_args()
    if not args.dataset_dir:
        print("usage: python test_pnp.py --dataset_dir <path_to_dataset>")
        sys.exit(-1)
    main(Path(args.dataset_dir))
