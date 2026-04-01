import collections
import collections.abc
collections.MutableMapping = collections.abc.MutableMapping


import logging
import time
import cv2
import random
import sys
import traceback
import argparse
import numpy as np
from dronekit import VehicleMode
import drone_lib
import object_tracking as obj_track
import pex03_utils



IS_IN_SITL = True

# Various mission states
MISSION_MODE_SEEK = 0
MISSION_MODE_CONFIRM = 1
MISSION_MODE_TARGET = 2
MISSION_MODE_DELIVER = 4
MISSION_MODE_RTL = 8

DEFAULT_UPDATE_RATE = 1  
DEFAULT_TARGET_RADIUS_MULTI = 1.0  
DEFAULT_TARGET_RADIUS = 5
DEFAULT_IMG_WRITE_RATE = 4  
DEFAULT_MAX_CONFIRM_ATTEMPTS = 8

IMG_FONT = cv2.FONT_HERSHEY_SIMPLEX
IMG_SNAPSHOT_PATH = '/home/usafa/Desktop/student_pex03_oop/cam_pex'

class SITLDashboard:
    """A clickable OpenCV control panel for SITL simulation monitoring."""
    def __init__(self, drone, mission_controller):
        self.drone = drone
        self.mission = mission_controller
        self.width = 450
        self.height = 350
        self.window_name = "SITL Control Panel"
        
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.click_event)

        # Button bounding boxes: (x1, y1, x2, y2)
        self.btn_rtl = (20, 280, 200, 330)
        self.btn_land = (230, 280, 430, 330)

    def click_event(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Check RTL Button
            if self.btn_rtl[0] < x < self.btn_rtl[2] and self.btn_rtl[1] < y < self.btn_rtl[3]:
                self.mission.log_info(">>> UI OVERRIDE: RTL Commanded <<<")
                self.drone.mode = VehicleMode("RTL")
                self.mission.mission_mode = MISSION_MODE_RTL
                
            # Check Land/Abort Button
            elif self.btn_land[0] < x < self.btn_land[2] and self.btn_land[1] < y < self.btn_land[3]:
                self.mission.log_info(">>> UI OVERRIDE: Emergency Land Commanded <<<")
                self.drone.mode = VehicleMode("LAND")
                self.mission.mission_mode = MISSION_MODE_RTL

    def update_and_draw(self):
        # Create dark gray background
        img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        img[:] = (40, 40, 40) 

        # --- Telemetry Data ---
        cv2.putText(img, "PEX 03 SITL DASHBOARD", (20, 30), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)
        cv2.line(img, (20, 40), (430, 40), (100, 100, 100), 2)

        mode_color = (0, 255, 0) if self.drone.mode.name in ["AUTO", "GUIDED"] else (0, 165, 255)
        cv2.putText(img, f"Drone Mode: {self.drone.mode.name}", (20, 75), IMG_FONT, 0.6, mode_color, 1)
        cv2.putText(img, f"Armed: {self.drone.armed}", (20, 105), IMG_FONT, 0.6, (255, 255, 255), 1)
        cv2.putText(img, f"Altitude: {max(0, self.drone.location.global_relative_frame.alt):.2f}m", (20, 135), IMG_FONT, 0.6, (255, 255, 255), 1)
        
        # --- Mission State ---
        state_map = {0: "SEEK", 1: "CONFIRM", 2: "TARGET", 4: "DELIVER", 8: "RTL/DONE"}
        current_state = state_map.get(self.mission.mission_mode, "UNKNOWN")
        cv2.putText(img, f"Mission State: {current_state}", (20, 180), IMG_FONT, 0.7, (255, 200, 0), 2)
        
        tgt_color = (0, 255, 0) if self.mission.object_identified else (0, 0, 255)
        cv2.putText(img, f"Target Acquired: {self.mission.object_identified}", (20, 215), IMG_FONT, 0.6, tgt_color, 1)

        # --- Buttons ---
        cv2.rectangle(img, (self.btn_rtl[0], self.btn_rtl[1]), (self.btn_rtl[2], self.btn_rtl[3]), (0, 100, 200), -1)
        cv2.putText(img, "FORCE RTL", (self.btn_rtl[0] + 35, self.btn_rtl[1] + 32), IMG_FONT, 0.6, (255, 255, 255), 2)

        cv2.rectangle(img, (self.btn_land[0], self.btn_land[1]), (self.btn_land[2], self.btn_land[3]), (0, 0, 200), -1)
        cv2.putText(img, "EMERG LAND", (self.btn_land[0] + 25, self.btn_land[1] + 32), IMG_FONT, 0.6, (255, 255, 255), 2)

        cv2.imshow(self.window_name, img)


class DroneMission:

    def __init__(self, device,
                 virtual_mode=False,
                 update_rate=DEFAULT_UPDATE_RATE,
                 target_radius=DEFAULT_TARGET_RADIUS,
                 target_multiplier=DEFAULT_TARGET_RADIUS_MULTI,
                 image_log_rate=DEFAULT_IMG_WRITE_RATE,
                 log_write_path=pex03_utils.DEFAULT_LOG_PATH,
                 max_confirm_attempts=DEFAULT_MAX_CONFIRM_ATTEMPTS,
                 min_target_radius=10,
                 mission_start_mode=MISSION_MODE_SEEK):

        self.drone = device
        self.virtual_mode = virtual_mode
        self.update_rate = update_rate
        self.target_radius = target_radius
        self.target_multiplier = target_multiplier
        self.image_log_rate = image_log_rate
        self.log_path = log_write_path
        self.max_confirm_attempts = max_confirm_attempts
        self.target_radius = min_target_radius

        self.mission_mode = mission_start_mode
        self.target_locate_attempts = 0
        self.refresh_counter = 0
        self.confirm_attempts = 0
        self.direction_x = "unknown"
        self.direction_y = "unknown"
        self.inside_circle = False
        self.object_identified = False

        self.init_obj_lon = None
        self.init_obj_lat = None
        self.init_obj_alt = None
        self.init_obj_heading = None

        self.last_lon_pos = -1.0
        self.last_lat_pos = -1.0
        self.last_alt_pos = -1.0
        self.last_heading_pos = -1.0

        # Initialize UI if in Virtual Mode
        self.dashboard = SITLDashboard(self.drone, self) if self.virtual_mode else None

    @staticmethod
    def log_info(msg):
        log.info(msg)

    def target_is_centered(self, target_point, frame_write=None):
        dx = float(target_point[0]) - obj_track.FRAME_HORIZONTAL_CENTER
        dy = obj_track.FRAME_VERTICAL_CENTER - float(target_point[1])

        self.log_info(f"Current alignment with target center: dx={dx}, dy={dy}")

        x, y = target_point

        if frame_write is not None:
            cv2.line(frame_write, target_point,
                     (int(obj_track.FRAME_HORIZONTAL_CENTER),
                      int(obj_track.FRAME_VERTICAL_CENTER)),
                     (0, 0, 255), 5)
            cv2.circle(frame_write, (int(x), int(y)), int(self.target_radius), (0, 0, 255), 2)
            cv2.circle(frame_write, (int(x), int(y)), int(self.target_radius * self.target_multiplier),
                       (255, 255, 0), 2)
            cv2.circle(frame_write, target_point, 5, (0, 255, 255), -1)

        return self.check_in_circle(target_point)

    def check_in_circle(self, target_point):
        return (int(target_point[0]) - obj_track.FRAME_HORIZONTAL_CENTER) ** 2 \
               + (int(target_point[1]) - obj_track.FRAME_VERTICAL_CENTER) ** 2 \
               <= self.target_radius ** 2

    def switch_mission_to_confirm_mode(self):
        self.mission_mode = MISSION_MODE_CONFIRM 
        self.confirm_attempts = 0
        self.log_info(f"Need to confirm target. Switching drone to GUIDED...")
        drone_lib.change_device_mode(device=self.drone, mode="GUIDED")

        drone_lib.goto_point(self.drone, self.init_obj_lat, self.init_obj_lon, 2.5, self.init_obj_alt)
        drone_lib.condition_yaw(self.drone, self.last_heading_pos)
        time.sleep(4)

    def confirm_objective(self, frame_write=None):
        if self.mission_mode == MISSION_MODE_CONFIRM:
            if self.object_identified:
                self.log_info(f"Target CONFIRMED.")
                self.mission_mode = MISSION_MODE_TARGET
                return True
            else:
                if self.confirm_attempts >= self.max_confirm_attempts:
                    self.mission_mode = MISSION_MODE_SEEK
                    self.log_info(f"Exceeded attempts. Switching back to AUTO...")
                    drone_lib.change_device_mode(device=self.drone, mode="AUTO")
                    return False
                else:
                    self.log_info("Re-acquiring target...")
                    if frame_write is not None:
                        cv2.putText(frame_write, "Re-acquiring target...", (10, 250), IMG_FONT, 1, (255, 0, 0), 2, cv2.LINE_AA)

                    drone_lib.goto_point(self.drone, self.init_obj_lat, self.init_obj_lon, 2.5, self.init_obj_alt)
                    drone_lib.condition_yaw(self.drone, random.random() * 180)
                    time.sleep(2)
                    self.confirm_attempts += 1
        return False

    def adjust_to_target_center(self, target_point, frame_write=None):
        dx = float(target_point[0]) - obj_track.FRAME_HORIZONTAL_CENTER
        dy = obj_track.FRAME_VERTICAL_CENTER - float(target_point[1])

        pixel_forgiveness = 15  

        if self.mission_mode == MISSION_MODE_TARGET:
            if target_point is not None and self.object_identified:
                self.direction_x = "L" if dx < -pixel_forgiveness else "R" if dx > pixel_forgiveness else "C"
                self.direction_y = "B" if dy < -pixel_forgiveness else "F" if dy > pixel_forgiveness else "C"

                if (self.direction_y == "C" and self.direction_x == "C") or self.inside_circle:
                    self.mission_mode = MISSION_MODE_DELIVER
                    drone_lib.log_activity("Time to deliver package!")
                else:
                    pixel_distance_threshold = 20
                    xv = 0.5 if abs(dx) > pixel_distance_threshold else 0.0
                    yv = 0.5 if abs(dy) > pixel_distance_threshold else 0.0

                    if self.direction_y != "C":  
                        if self.direction_y == "F":
                            drone_lib.small_move_forward(self.drone, velocity=yv, duration=1)
                        else:
                            drone_lib.small_move_back(self.drone, velocity=yv, duration=1)

                    if self.direction_x != "C":  
                        if self.direction_x == "R":
                            drone_lib.small_move_right(self.drone, velocity=xv, duration=1)
                        else:
                            drone_lib.small_move_left(self.drone, velocity=xv, duration=1)
            else:
                self.log_info("Cannot target object; switching back to seek mode...")
                self.mission_mode = MISSION_MODE_SEEK

    def deliver_package(self, frame_write=None):
        self.log_info("Delivering package...")

        location = self.drone.location.global_relative_frame
        alt = location.alt
        dist_to_object = pex03_utils.get_avg_distance_to_obj(5.0, self.drone, self.virtual_mode)

        if dist_to_object > 0:
            safe_distance_meters = 3.0  
            new_lat, new_lon = pex03_utils.calc_new_location(location.lat, location.lon, self.drone.heading, safe_distance_meters)
            drone_lib.goto_point(self.drone, new_lat, new_lon, speed=2.0, alt=alt)

            if self.virtual_mode:
                self.log_info("Time to land... (Virtual Mode Drop Simulation)")
                self.mission_mode = MISSION_MODE_RTL
                drone_lib.device_land(self.drone)
            else:
                self.log_info("Lowering package....")
                while self.drone.location.global_relative_frame.alt > 5.0:
                    if self.drone.mode in ["RTL", "LAND"] or self.mission_mode == MISSION_MODE_RTL: break
                    drone_lib.small_move_down(self.drone, velocity=0.6, duration=1)

                while self.drone.location.global_relative_frame.alt > 3.20:
                    if self.drone.mode in ["RTL", "LAND"] or self.mission_mode == MISSION_MODE_RTL: break
                    drone_lib.small_move_down(self.drone, velocity=0.2, duration=1)

            self.log_info("Releasing package now....")
            pex03_utils.release_grip(self.drone, seconds=2)
            time.sleep(2)
            
        self.mission_mode = MISSION_MODE_RTL
        drone_lib.return_to_launch(self.drone)

    def determine_action(self, target_point, frame_write=None):
        if self.drone.mode in ["RTL", "LAND"] or self.mission_mode == MISSION_MODE_RTL: return

        if self.mission_mode == MISSION_MODE_SEEK:
            if self.object_identified:
                self.switch_mission_to_confirm_mode()
            elif self.drone.mode != "AUTO":
                drone_lib.change_device_mode(device=self.drone, mode="AUTO")
        else:
            if target_point is not None and self.object_identified:  
                self.inside_circle = self.target_is_centered(target_point, frame_write)

            if self.mission_mode == MISSION_MODE_CONFIRM:
                self.confirm_objective(frame_write)

        if self.mission_mode == MISSION_MODE_TARGET:
            self.adjust_to_target_center(target_point, frame_write)

        if self.mission_mode == MISSION_MODE_DELIVER:
            self.deliver_package(frame_write)

    def conduct_mission(self):
        self.log_info("Mission started...")
        self.object_identified = False

        while self.drone.armed:

            if self.drone.mode in ["RTL", "LAND"] or self.mission_mode == MISSION_MODE_RTL:
                self.log_info("RTL/LAND mode activated. Mission ended.")
                break

            timer = cv2.getTickCount()
            frame = obj_track.get_cur_frame()
            if frame is None: continue

            location = self.drone.location.global_relative_frame
            self.last_lat_pos, self.last_lon_pos, self.last_alt_pos = location.lat, location.lon, location.alt
            self.last_heading_pos = self.drone.heading

            frm_display = frame.copy()

            if not self.object_identified:
                # Use in_debug=True dynamically based on virtual mode to track cars in SITL
                center, confidence, corner, radius, frm_display, bbox = obj_track.check_for_initial_target(
                    frame, frm_display, show_img=self.virtual_mode, in_debug=self.virtual_mode)

                if confidence is not None and confidence > 0.50:
                    self.object_identified = True
                    obj_track.set_object_to_track(frame, bbox)
                    self.init_obj_lat, self.init_obj_lon, self.init_obj_alt = location.lat, location.lon, location.alt
                    self.init_obj_heading = self.drone.heading
            else:
                center, confidence, corner, radius, frm_display, bbox = obj_track.track_with_confirm(
                    frame, frm_display, show_img=self.virtual_mode, debug_mode=self.virtual_mode)

                if not confidence:
                    self.object_identified = False

            # --- Update the SITL UI Dashboard if Active ---
            if self.virtual_mode and self.dashboard:
                self.dashboard.update_and_draw()

            if self.virtual_mode:
                cv2.imshow("Tracking Feed", frm_display)
                if cv2.waitKey(1) & 0xFF == ord("q"): break

            if (self.refresh_counter % self.update_rate) == 0 or self.mission_mode != MISSION_MODE_SEEK:
                self.determine_action(center, frame_write=frm_display)

            self.refresh_counter += 1


if __name__ == '__main__':
    # ---------------------------------------------------------
    # Parse CLI arguments for the Toggle switch
    # ---------------------------------------------------------
    parser = argparse.ArgumentParser(description="PEX 03 Autonomous Mission Script")
    parser.add_argument('--sitl', action='store_true', help="Run the mission in Virtual SITL Mode with GUI")
    args = parser.parse_args()
    
    IS_VIRTUAL = IS_IN_SITL

    pex03_utils.backup_prev_experiment(IMG_SNAPSHOT_PATH)
    log_file = time.strftime(IMG_SNAPSHOT_PATH + "/Cam_PEX03_%Y%m%d-%H%M%S") + ".log"

    # Standard logging configuration
    log_level = logging.DEBUG if IS_VIRTUAL else logging.INFO
    logging.basicConfig(level=log_level, handlers=[logging.FileHandler(log_file), logging.StreamHandler()])
    log = logging.getLogger(__name__)

    log.info(f"PEX 03 test program start. Virtual Mode: {IS_VIRTUAL}")
    obj_track.start_camera_stream()
    obj_track.load_visdrone_network()

    # Connection Logic
    if IS_VIRTUAL:
        log.info("Connecting to local SITL (127.0.0.1:14550)...")
        drone = drone_lib.connect_device("127.0.0.1:14550")
    else:
        log.info("Connecting to physical UAV (/dev/ttyACM0)...")
        drone = drone_lib.connect_device("/dev/ttyACM0", baud=115200)

        # Safety Check for IRL Mode
        if drone.rangefinder.distance is None:
            log.info("Rangefinder not detected. Mission aborting for safety.")
            exit(99)

    drone.commands.download()
    time.sleep(1)

    if drone.commands.count < 1:
        log.info("No mission waypoints loaded! Please load a search pattern.")
        exit(99)

    drone_lib.arm_device(drone, log=log)
    drone_lib.device_takeoff(drone, 15, log=log)

    try:
        drone_lib.change_device_mode(drone, "AUTO", log=log)
        drone_mission = DroneMission(device=drone, virtual_mode=IS_VIRTUAL, log_write_path=IMG_SNAPSHOT_PATH)
        drone_mission.conduct_mission()

        log.info("Disarming and closing connection...")
        drone.armed = False
        drone.close()
    except Exception as e:
        log.info(f"Program exception: {traceback.format_exception(*sys.exc_info())}")
        raise