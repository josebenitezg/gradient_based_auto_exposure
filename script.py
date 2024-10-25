import cv2
import numpy as np
from enum import Enum
from typing import List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AutoExposureMethod(Enum):
    FIXED = 0
    PERCENTILE = 1
    SWEEP = 2
    SHIM = 3
    MEAN_INTENSITY = 4
    PATCH_SCORE = 5
    BLUR_PATCH_SCORE = 6

class AutoCameraOptions:
    def __init__(self):
        self.ae_method = AutoExposureMethod.MEAN_INTENSITY
        self.use_parallel_thread = False
        self.pyramid_lvl = 3
        self.max_exp_time_us = 15000
        self.min_exp_time_us = 20
        self.auto_gain = True
        self.max_gain = 4.0
        self.min_gain = 1.0
        self.gain_change_step = 0.5
        self.dec_gain_exp_us = 2000
        self.inc_gain_exp_us = 10000
        self.target_percentile = 90
        self.target_percentile_value = 200

class Frame:
    def __init__(self, exp_time_us: int, gain_x: float, img: np.ndarray, img_pyr_lvl: int = 3):
        self.exp_time_us = exp_time_us
        self.gain_x = gain_x
        self.img_pyr = self.create_img_pyramid(img, img_pyr_lvl)
        
    @staticmethod
    def create_img_pyramid(img: np.ndarray, img_pyr_lvl: int) -> List[np.ndarray]:
        img_pyr = [img]
        for i in range(1, img_pyr_lvl):
            img_pyr.append(cv2.pyrDown(img_pyr[-1]))
        return img_pyr

class AutoCameraSettings:
    def __init__(self, options: AutoCameraOptions):
        self.options = options
        self.desired_exposure_time_us = options.min_exp_time_us
        self.desired_gain_x = options.min_gain
        
    def set_next_frame(self, frame: Frame):
        if self.options.ae_method == AutoExposureMethod.MEAN_INTENSITY:
            self.compute_desired_exposure_time_intensity(frame)
        elif self.options.ae_method == AutoExposureMethod.PERCENTILE:
            self.compute_desired_exposure_weighted_gradient(frame)
        else:
            print(f"Unsupported auto exposure method: {self.options.ae_method}")
        
        self.update_gain()
        
    def compute_desired_exposure_time_intensity(self, frame: Frame):
        mean_intensity = np.mean(frame.img_pyr[0])
        damping = 0.2
        
        # Handle the case where mean_intensity is very close to or equal to zero
        if mean_intensity < 1e-6:
            # If the image is too dark, increase exposure time significantly
            self.desired_exposure_time_us = min(frame.exp_time_us * 2, self.options.max_exp_time_us)
        else:
            target_intensity = 255.0 * 0.5  # Middle gray
            ratio = target_intensity / mean_intensity
            # Limit the ratio to avoid extreme changes
            ratio = max(0.5, min(2.0, ratio))
            
            exp_delta = int(frame.exp_time_us * (ratio - 1.0))
            self.desired_exposure_time_us = int(frame.exp_time_us + damping * exp_delta)
        
        self.desired_exposure_time_us = self.clamp_exposure_time(self.desired_exposure_time_us)
        
    def compute_desired_exposure_weighted_gradient(self, frame: Frame):
        img = frame.img_pyr[0]
        height, width = img.shape
        
        # Compute gradients
        grad_x = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        
        # Compute weights
        weights = np.minimum(grad_mag, 30.0) / 30.0
        
        # Sort pixel intensities
        flat_img = img.flatten()
        flat_weights = weights.flatten()
        sorted_indices = np.argsort(flat_img)
        cumsum_weights = np.cumsum(flat_weights[sorted_indices])
        
        # Find target percentile
        target_sum = cumsum_weights[-1] * self.options.target_percentile / 100.0
        target_idx = np.searchsorted(cumsum_weights, target_sum)
        target_value = flat_img[sorted_indices[target_idx]]
        
        # Compute desired exposure time
        exposure_ratio = self.options.target_percentile_value / target_value
        self.desired_exposure_time_us = int(frame.exp_time_us * exposure_ratio)
        self.desired_exposure_time_us = self.clamp_exposure_time(self.desired_exposure_time_us)
    
    def update_gain(self):
        if self.options.auto_gain:
            if self.desired_exposure_time_us > self.options.inc_gain_exp_us:
                self.desired_gain_x = min(self.desired_gain_x + self.options.gain_change_step, self.options.max_gain)
            elif self.desired_exposure_time_us < self.options.dec_gain_exp_us:
                self.desired_gain_x = max(self.desired_gain_x - self.options.gain_change_step, self.options.min_gain)
    
    def clamp_exposure_time(self, exp_time: int) -> int:
        return max(min(exp_time, self.options.max_exp_time_us), self.options.min_exp_time_us)

    def get_desired_exposure_time_us(self) -> int:
        return self.desired_exposure_time_us

    def get_desired_gain_x(self) -> float:
        return self.desired_gain_x

class Camera:
    def __init__(self, camera_id: int = 0):
        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera with id {camera_id}")
        
        self.exp_time_us = 10000  # Initial exposure time
        self.gain_x = 1.0  # Initial gain
        
    def set_exposure(self, exp_time_us: int):
        self.exp_time_us = exp_time_us
        self.cap.set(cv2.CAP_PROP_EXPOSURE, exp_time_us / 1000)  # OpenCV uses milliseconds
        
    def set_gain(self, gain_x: float):
        self.gain_x = gain_x
        self.cap.set(cv2.CAP_PROP_GAIN, gain_x)
        
    def capture_frame(self) -> Optional[np.ndarray]:
        ret, frame = self.cap.read()
        if ret:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return None
    
    def release(self):
        self.cap.release()
        
def main():
    options = AutoCameraOptions()
    auto_camera = AutoCameraSettings(options)
    camera = Camera()

    try:
        while True:
            img = camera.capture_frame()
            if img is None:
                logger.warning("Failed to capture frame")
                continue

            frame = Frame(exp_time_us=camera.exp_time_us, gain_x=camera.gain_x, img=img)
            
            try:
                auto_camera.set_next_frame(frame)
            except Exception as e:
                logger.error(f"Error in auto exposure calculation: {e}")
                continue

            new_exp_time = auto_camera.get_desired_exposure_time_us()
            new_gain = auto_camera.get_desired_gain_x()

            logger.info(f"New settings - Exposure: {new_exp_time}us, Gain: {new_gain:.2f}")

            camera.set_exposure(new_exp_time)
            camera.set_gain(new_gain)

            cv2.putText(img, f"Exp: {new_exp_time}us, Gain: {new_gain:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.imshow("Auto Exposure Control", img)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
    finally:
        camera.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
