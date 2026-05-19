#!/usr/bin/env python3

import os
import cv2
import numpy as np
import torch
import yaml
from PIL import Image

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image as SensorImage
from geometry_msgs.msg import PoseWithCovarianceStamped
from cv_bridge import CvBridge

try:
    from ament_index_python.packages import get_package_share_directory
except ImportError:
    pass # Fallback handled below

from ultralytics import YOLO

from gate_vision.model import GOAT
from gate_vision.detect import decode_keypoints
from gate_vision.utils import pick_hardware, preprocess_image

class GateLocalizationNode(Node):
    def __init__(self):
        super().__init__('gate_localization_node')
        self.get_logger().info("Initializing GateLocalizationNode...")

        self.bridge = CvBridge()
        self.device = pick_hardware()

        # Load models
        try:
            pkg_share = get_package_share_directory('gate_vision')
            gates_model_path = os.path.join(pkg_share, 'models', 'gates_model.pt')
            corners_model_path = os.path.join(pkg_share, 'models', 'best_model.pth')
            camera_info_path = os.path.join(pkg_share, 'config', 'camera_info.yaml')
        except Exception:
            pkg_share = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
            gates_model_path = os.path.join(pkg_share, 'models', 'gates_model.pt')
            corners_model_path = os.path.join(pkg_share, 'models', 'best_model.pth')
            camera_info_path = os.path.join(pkg_share, 'config', 'camera_info.yaml')

        self.get_logger().info(f"Loading YOLO from {gates_model_path}")
        self.gates_model = YOLO(gates_model_path)
        
        self.get_logger().info(f"Loading GOAT from {corners_model_path}")
        self.corners_model = GOAT(num_corners=4).to(self.device)
        self.corners_model.load_state_dict(torch.load(corners_model_path, map_location=self.device))
        self.corners_model.eval()

        # Parameters for pipeline
        self.CROP_PAD = 0.20
        self.SIZE = 256
        self.CORNER_THRESH = 0.5
        
        # Load Camera Calibration
        self._load_camera_info(camera_info_path)
        
        # 3D World Model
        # markers at (± w/2, ± h/2, 0)
        w, h = 1.5, 1.5
        self.object_points = np.array([
            [-w/2, -h/2, 0.0], # 0: TL
            [ w/2, -h/2, 0.0], # 1: TR
            [-w/2,  h/2, 0.0], # 2: BL
            [ w/2,  h/2, 0.0]  # 3: BR
        ], dtype=np.float32)

        # Publishers
        self.pose_pub = self.create_publisher(PoseWithCovarianceStamped, 'gate_pose', 10)
        self.debug_pub = self.create_publisher(SensorImage, 'gate_debug_image', 10)

        # Subscriber
        self.img_sub = self.create_subscription(SensorImage, 'camera/image_raw', self.image_callback, 10)

        self.get_logger().info("GateLocalizationNode is ready.")

    def _load_camera_info(self, path):
        if not os.path.exists(path):
            self.get_logger().warn(f"Camera info {path} not found. Using defaults.")
            self.camera_matrix = np.array([[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]], dtype=np.float32)
            self.dist_coeffs = np.zeros((5, 1), dtype=np.float32)
            return

        with open(path, 'r') as f:
            calib = yaml.safe_load(f)
            self.camera_matrix = np.array(calib['camera_matrix']['data'], dtype=np.float32).reshape(3, 3)
            self.dist_coeffs = np.array(calib['distortion_coefficients']['data'], dtype=np.float32)

    def _square_crop_bounds(self, box, full_w, full_h):
        x1, y1, x2, y2 = box
        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        half = max(bw, bh) * (1.0 + self.CROP_PAD) / 2.0
        cx1 = int(max(0, np.floor(cx - half)))
        cy1 = int(max(0, np.floor(cy - half)))
        cx2 = int(min(full_w, np.ceil(cx + half)))
        cy2 = int(min(full_h, np.ceil(cy + half)))
        return cx1, cy1, cx2, cy2

    def detect_gates(self, bgr_frame):
        results = self.gates_model(bgr_frame, verbose=False)[0]
        boxes = []
        if results.boxes is not None:
            for box in results.boxes.xyxy.cpu().numpy():
                x1, y1, x2, y2 = map(int, box[:4])
                boxes.append((x1, y1, x2, y2))
        return boxes

    @torch.no_grad()
    def detect_corners_in_box(self, bgr_frame, box):
        h, w = bgr_frame.shape[:2]
        cx1, cy1, cx2, cy2 = self._square_crop_bounds(box, w, h)
        if cx2 - cx1 < 4 or cy2 - cy1 < 4:
            return {}

        crop = bgr_frame[cy1:cy2, cx1:cx2]
        pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        tensor = preprocess_image(pil, self.SIZE).to(self.device)
        heatmaps = self.corners_model(tensor).cpu()
        
        decoded = decode_keypoints(heatmaps, self.CORNER_THRESH)[0]

        cw, ch = (cx2 - cx1), (cy2 - cy1)
        return {
            c: (cx_n * cw + cx1, cy_n * ch + cy1, score)
            for c, (cx_n, cy_n, score) in decoded.items()
        }

    def image_callback(self, msg):
        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"Failed to convert image: {e}")
            return

        out = bgr.copy()
        h, w = bgr.shape[:2]
        boxes = self.detect_gates(bgr)

        best_gate = None
        best_corners = None
        best_score = -1

        for g_idx, (x1, y1, x2, y2) in enumerate(boxes):
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)
            if x2 <= x1 or y2 <= y1:
                continue

            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            gate_corners = self.detect_corners_in_box(bgr, (x1, y1, x2, y2))
            
            corner_bgr = {0: (255, 255, 0), 1: (0, 255, 255), 2: (0, 255, 0), 3: (0, 0, 255)}
            for c_type, (cx, cy, score) in gate_corners.items():
                px, py = int(cx), int(cy)
                cv2.circle(out, (px, py), 4, corner_bgr.get(c_type, (255, 0, 0)), -1)
                
            if len(gate_corners) == 4:
                avg_score = sum(s for _, _, s in gate_corners.values()) / 4.0
                if avg_score > best_score:
                    best_score = avg_score
                    best_gate = (x1, y1, x2, y2)
                    best_corners = gate_corners

        if best_corners is not None and len(best_corners) == 4:
            image_points = np.array([
                [best_corners[0][0], best_corners[0][1]], # TL
                [best_corners[1][0], best_corners[1][1]], # TR
                [best_corners[2][0], best_corners[2][1]], # BL
                [best_corners[3][0], best_corners[3][1]]  # BR
            ], dtype=np.float32)

            success, rvec, tvec = cv2.solvePnP(self.object_points, image_points, self.camera_matrix, self.dist_coeffs)
            
            if success:
                x_c, y_c, z_c = tvec.flatten()
                
                pose_msg = PoseWithCovarianceStamped()
                pose_msg.header.stamp = self.get_clock().now().to_msg()
                pose_msg.header.frame_id = "camera_link"
                pose_msg.pose.pose.position.x = float(z_c)
                pose_msg.pose.pose.position.y = float(-x_c)
                pose_msg.pose.pose.position.z = float(-y_c)
                
                pose_msg.pose.covariance[0] = 0.1
                pose_msg.pose.covariance[7] = 0.1
                pose_msg.pose.covariance[14] = 0.1
                
                self.pose_pub.publish(pose_msg)
                
                cv2.drawFrameAxes(out, self.camera_matrix, self.dist_coeffs, rvec, tvec, 0.5)

        try:
            debug_msg = self.bridge.cv2_to_imgmsg(out, encoding="bgr8")
            self.debug_pub.publish(debug_msg)
        except Exception as e:
            self.get_logger().error(f"Failed to publish debug image: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = GateLocalizationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
