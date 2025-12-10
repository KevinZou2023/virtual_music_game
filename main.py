import sys
import time
import csv
import os
from collections import deque

import cv2
import numpy as np
from scipy import signal
from PyQt5 import QtCore, QtGui, QtWidgets
import pygame
import mediapipe as mp

import random

# Audio init
pygame.mixer.pre_init(44100, -16, 2, 512)
pygame.mixer.init()
pygame.init()
pygame.mixer.set_num_channels(64) # Allow more simultaneous sounds

SOUNDS = {}

def load_sound(filename, volume=0.8):
    """Load a sound file from assets folder with error handling.
    
    Args:
        filename: Name of the WAV file (without path)
        volume: Volume level (0.0 to 1.0)
    
    Returns:
        pygame.mixer.Sound object or None if loading fails
    """
    filepath = os.path.join('assets', filename)
    try:
        if not os.path.exists(filepath):
            print(f"[Warning] Sound file not found: {filepath}")
            return None
        sound = pygame.mixer.Sound(filepath)
        sound.set_volume(volume)
        return sound
    except Exception as e:
        print(f"[Warning] Failed to load sound '{filename}': {e}")
        return None

def generate_piano_string(freq, duration=2.5, sr=44100):
    """Generate piano string sound using basic synthesis"""
    n_samples = int(sr * duration)
    t = np.linspace(0, duration, n_samples, False)
    
    wave = np.zeros(n_samples)
    # More harmonics for richer piano tone
    harmonics = [1, 2, 3, 4, 5, 6, 7]
    weights = [1.0, 0.6, 0.4, 0.25, 0.15, 0.1, 0.05]
    
    for h, w in zip(harmonics, weights):
        wave += w * np.sin(2 * np.pi * freq * h * t)
        # Detuned for chorus
        wave += (w*0.4) * np.sin(2 * np.pi * (freq * h * 1.003) * t)
        
    # Piano envelope with longer sustain
    env = np.exp(-1.2 * t)
    wave = wave * env
    
    if np.max(np.abs(wave)) > 0:
        wave = wave / np.max(np.abs(wave)) * 0.75
        
    stereo = np.column_stack((wave, wave))
    return pygame.sndarray.make_sound((stereo * 32767).astype(np.int16))

# Init Sounds - Load drums from files
print("加载打击乐音色...")
SOUNDS['Kick'] = load_sound('171104__dwsd__kick_gettinglaid.wav', volume=0.8)
SOUNDS['Boom'] = load_sound('33637__herbertboland__cinematicboomnorm.wav', volume=0.8)
SOUNDS['Tom'] = load_sound('37215__simon_lacelle__ba-da-dum.wav', volume=0.8)

# Piano - Generated using synthesis (no samples available yet)
print("生成钢琴音色...")
PIANO_NOTES = {'C': 261.63, 'D': 293.66, 'E': 329.63, 'F': 349.23, 'G': 392.00, 'A': 440.00, 'B': 493.88}
for note, freq in PIANO_NOTES.items():
    SOUNDS[f'Piano_{note}'] = generate_piano_string(freq)
print("音色加载完成！")

# MediaPipe 手部追踪初始化（延迟到应用启动时）
print("初始化 MediaPipe...")
mp_hands = None
mp_drawing = None
hands = None

try:
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,  # 支持双手
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )
    print("MediaPipe 初始化成功！")
except Exception as e:
    print(f"[Warning] MediaPipe 初始化失败: {e}")
    print("将仅使用深度图追踪模式")
    hands = None

class HandTracker:
    """MediaPipe 手部追踪封装"""
    def __init__(self, hands_detector, mp_drawing_utils=None):
        self.hands = hands_detector
        self.mp_drawing = mp_drawing_utils
        self.hand_landmarks = None
        self.handedness = None
        self.available = hands_detector is not None
    
    def update(self, frame):
        """更新手部追踪结果
        
        Args:
            frame: BGR 格式的图像帧
        
        Returns:
            frame: 标注了手部关键点的图像
        """
        if not self.available:
            return frame
        
        try:
            h, w, c = frame.shape
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)
            
            self.hand_landmarks = results.multi_hand_landmarks
            self.handedness = results.multi_handedness
            
            # 绘制手部关键点、连接和视觉准星
            if self.hand_landmarks and self.mp_drawing:
                for hand_landmarks in self.hand_landmarks:
                    # 绘制手部骨骼连接
                    self.mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                        self.mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
                    )
                    
                    # 提取食指指尖坐标（Landmark 8）
                    index_finger_tip = hand_landmarks.landmark[8]
                    
                    # 将归一化坐标转换为像素坐标
                    index_x = int(index_finger_tip.x * w)
                    index_y = int(index_finger_tip.y * h)
                    
                    # 绘制视觉准星：亮黄色实心圆 + 白色描边
                    # 亮黄色实心圆（半径 15）
                    cv2.circle(frame, (index_x, index_y), 15, (0, 255, 255), -1)
                    # 白色描边（半径 15，线宽 2）
                    cv2.circle(frame, (index_x, index_y), 15, (255, 255, 255), 2)
        except Exception as e:
            print(f"[Warning] Hand tracking error: {e}")
        
        return frame
    
    def get_hand_positions(self):
        """获取两只手的中指尖位置（用于击鼓）
        
        Returns:
            list: 包含 (x, y) 位置的列表，范围 [0, 1]
        """
        positions = []
        if not self.available or not self.hand_landmarks:
            return positions
        
        try:
            for hand_lm in self.hand_landmarks:
                # 中指尖是第 8 个关键点
                middle_finger_tip = hand_lm.landmark[8]
                positions.append((middle_finger_tip.x, middle_finger_tip.y))
        except Exception as e:
            print(f"[Warning] Error getting hand positions: {e}")
        return positions


class VirtualDrum:
    """独立的虚拟鼓实例，具备自主判定和施密特触发器逻辑"""
    
    # 状态定义
    STATE_IDLE = 0
    STATE_PRESSED = 1
    
    def __init__(self, name, rect_norm, sound, trigger_depth=500, release_depth=540, 
                 cooldown_ms=100, min_valid_pixels=100):
        """
        Args:
            name: 鼓的名称
            rect_norm: 归一化的矩形 (x, y, w, h)，范围 0-1
            sound: pygame Sound 对象
            trigger_depth: 触发深度阈值 (mm)
            release_depth: 释放深度阈值 (mm)
            cooldown_ms: 冷却时间 (ms)
            min_valid_pixels: 最小有效像素数，低于此值视为噪点
        """
        self.name = name
        self.rect_norm = rect_norm  # (x, y, w, h) in [0, 1]
        self.sound = sound
        self.trigger_depth = trigger_depth
        self.release_depth = release_depth
        self.cooldown_ms = cooldown_ms
        self.min_valid_pixels = min_valid_pixels
        
        # 状态机
        self.state = self.STATE_IDLE
        self.last_trigger_time = 0.0  # 时间戳 (ms)
        
        # 深度历史记录（用于速度检测，避免误触）
        self.depth_history = []  # 存储最近的深度值
        self.max_history_size = 3  # 保留最近3帧
        
        # 缓存的 ROI 坐标（整数像素坐标）
        self.roi_rect = None  # (x, y, w, h)
        
    def set_frame_size(self, frame_width, frame_height):
        """根据帧大小计算 ROI 的像素坐标"""
        x_norm, y_norm, w_norm, h_norm = self.rect_norm
        self.roi_rect = (
            int(x_norm * frame_width),
            int(y_norm * frame_height),
            int(w_norm * frame_width),
            int(h_norm * frame_height)
        )
    
    def update(self, depth_frame, current_time_ms):
        """
        更新鼓的状态，检测触发事件
        
        Args:
            depth_frame: 深度帧 (H, W)，单位 mm
            current_time_ms: 当前时间 (ms)
        
        Returns:
            True 如果触发了声音，False 否则
        """
        if self.roi_rect is None:
            return False
        
        x, y, w, h = self.roi_rect
        
        # 安全检查，防止越界
        if x < 0 or y < 0 or x + w > depth_frame.shape[1] or y + h > depth_frame.shape[0]:
            return False
        
        # 裁剪 ROI
        roi_depth = depth_frame[y:y+h, x:x+w]
        
        # 创建 Mask：保留有效深度范围内的像素
        # 有效条件：深度 > 0 且 深度 < release_depth
        mask = (roi_depth > 0) & (roi_depth < self.release_depth)
        
        # 计算有效像素数量
        valid_pixel_count = np.count_nonzero(mask)
        
        # 误触过滤：如果有效像素太少，视为噪点
        if valid_pixel_count < self.min_valid_pixels:
            # 如果处于 PRESSED 状态，转回 IDLE
            if self.state == self.STATE_PRESSED:
                self.state = self.STATE_IDLE
            return False
        
        # 在 Mask 区域内计算最小深度
        valid_depths = roi_depth[mask]
        min_depth = int(np.min(valid_depths))
        
        # 施密特触发器逻辑
        triggered = False
        
        if self.state == self.STATE_IDLE:
            # 从 IDLE 到 PRESSED：检查是否低于触发阈值
            if min_depth < self.trigger_depth:
                # 检查冷却时间
                if current_time_ms - self.last_trigger_time >= self.cooldown_ms:
                    self.state = self.STATE_PRESSED
                    self.last_trigger_time = current_time_ms
                    triggered = True
                    print(f"[DRUM] {self.name} TRIGGERED - depth: {min_depth}mm, valid_pixels: {valid_pixel_count}")
        
        elif self.state == self.STATE_PRESSED:
            # 从 PRESSED 到 IDLE：检查是否高于释放阈值
            if min_depth > self.release_depth:
                self.state = self.STATE_IDLE
                print(f"[DRUM] {self.name} RELEASED - depth: {min_depth}mm")
        
        return triggered
    
    def play(self):
        """播放鼓声"""
        if self.sound:
            self.sound.play()
    
    def update_hand_skeleton(self, hand_positions, current_time_ms, mirror=False):
        """
        基于手部骨骼位置的判定（替代深度判定）
        
        Args:
            hand_positions: 手部关键点位置列表，[(x, y), ...] 范围 [0, 1]
            current_time_ms: 当前时间 (ms)
            mirror: 是否镜像显示
        
        Returns:
            True 如果触发了声音，False 否则
        """
        triggered = False
        
        # 检查是否有手部在鼓的区域内
        x_norm, y_norm, w_norm, h_norm = self.rect_norm
        
        # 如果启用镜像，调整鼓的位置
        if mirror:
            x_norm = 1.0 - x_norm - w_norm
        
        drum_rect = (x_norm, y_norm, x_norm + w_norm, y_norm + h_norm)
        
        hand_in_roi = False
        for hand_x, hand_y in hand_positions:
            # 如果启用镜像，调整手部坐标
            if mirror:
                hand_x = 1.0 - hand_x
            # 检查手部关键点是否在鼓的矩形范围内
            if drum_rect[0] <= hand_x <= drum_rect[2] and drum_rect[1] <= hand_y <= drum_rect[3]:
                hand_in_roi = True
                break
        
        # 施密特触发器逻辑
        if self.state == self.STATE_IDLE:
            if hand_in_roi:
                # 从 IDLE 到 PRESSED
                if current_time_ms - self.last_trigger_time >= self.cooldown_ms:
                    self.state = self.STATE_PRESSED
                    self.last_trigger_time = current_time_ms
                    triggered = True
                    print(f"[HAND] {self.name} TRIGGERED via hand skeleton")
        
        elif self.state == self.STATE_PRESSED:
            if not hand_in_roi:
                # 从 PRESSED 到 IDLE
                self.state = self.STATE_IDLE
                print(f"[HAND] {self.name} RELEASED")
        
        return triggered
    
    def update_hand_with_depth(self, hand_positions, depth_frame, current_time_ms, mirror=False):
        """
        基于手部骨骼位置 + 深度判定（结合两个条件）
        
        Args:
            hand_positions: 手部关键点位置列表，[(x, y), ...] 范围 [0, 1]
            depth_frame: 深度帧 (H, W)，单位 mm
            current_time_ms: 当前时间 (ms)
            mirror: 是否镜像显示
        
        Returns:
            True 如果触发了声音，False 否则
        """
        if self.roi_rect is None:
            return False
        
        triggered = False
        
        # 检查是否有手部在鼓的区域内
        x_norm, y_norm, w_norm, h_norm = self.rect_norm
        drum_rect = (x_norm, y_norm, x_norm + w_norm, y_norm + h_norm)
        
        # 检查手部是否在 ROI 内，并获取该位置的深度
        hand_in_roi = False
        hand_depth = None
        
        # 注意：depth_frame 已经被镜像过，手部坐标也是镜像后的，所以直接使用
        for hand_x, hand_y in hand_positions:
            check_x = hand_x
            
            # 检查手部关键点是否在鼓的矩形范围内
            if drum_rect[0] <= check_x <= drum_rect[2] and drum_rect[1] <= hand_y <= drum_rect[3]:
                hand_in_roi = True
                
                # 获取该位置的深度值
                x, y, w, h = self.roi_rect
                
                # 计算食指在 ROI 内的相对位置
                roi_x = int((check_x - drum_rect[0]) / w_norm * w)
                roi_y = int((hand_y - drum_rect[1]) / h_norm * h)
                
                # 确保在 ROI 范围内
                roi_x = max(0, min(roi_x, w - 1))
                roi_y = max(0, min(roi_y, h - 1))
                
                # 计算实际像素坐标并确保不越界
                px_start = x
                py_start = y
                depth_h, depth_w = depth_frame.shape[:2]
                actual_y = min(py_start + roi_y, depth_h - 1)
                actual_x = min(px_start + roi_x, depth_w - 1)
                
                # 在食指尖周围采样 5x5 区域，取最小深度（最接近相机的点）
                # 这样可以避免手掌根部误触发
                sample_radius = 2  # 5x5 区域
                y_start = max(0, actual_y - sample_radius)
                y_end = min(depth_h, actual_y + sample_radius + 1)
                x_start = max(0, actual_x - sample_radius)
                x_end = min(depth_w, actual_x + sample_radius + 1)
                
                depth_region = depth_frame[y_start:y_end, x_start:x_end]
                valid_depths = depth_region[(depth_region > 0) & (depth_region < 2000)]
                
                if valid_depths.size > 0:
                    # 取最小深度值（最接近相机的点）
                    hand_depth = int(np.min(valid_depths))
                
                break
        
        # 更新深度历史（只在手在ROI内时更新）
        if hand_in_roi and hand_depth is not None:
            self.depth_history.append(hand_depth)
            if len(self.depth_history) > self.max_history_size:
                self.depth_history.pop(0)
        else:
            # 手不在ROI内，清空历史
            self.depth_history.clear()
        
        # 计算深度变化速度（如果有足够的历史数据）
        depth_velocity = 0
        if len(self.depth_history) >= 2:
            # 速度 = 当前深度 - 上一帧深度（负值表示靠近相机）
            depth_velocity = self.depth_history[-1] - self.depth_history[-2]
        
        # 施密特触发器逻辑（手部位置 + 深度，与钢琴逻辑一致）
        if self.state == self.STATE_IDLE:
            # 触发条件：
            # 1. 手在ROI内
            # 2. 深度小于阈值
            if hand_in_roi and hand_depth is not None and hand_depth < self.trigger_depth:
                # 从 IDLE 到 PRESSED
                if current_time_ms - self.last_trigger_time >= self.cooldown_ms:
                    self.state = self.STATE_PRESSED
                    self.last_trigger_time = current_time_ms
                    triggered = True
                    print(f"[HAND+DEPTH] {self.name} TRIGGERED - depth: {hand_depth}mm, velocity: {depth_velocity}mm/frame")
        
        elif self.state == self.STATE_PRESSED:
            if not hand_in_roi or (hand_depth is not None and hand_depth > self.release_depth):
                # 从 PRESSED 到 IDLE
                self.state = self.STATE_IDLE
                self.depth_history.clear()  # 清除历史，准备下次触发
                print(f"[HAND+DEPTH] {self.name} RELEASED")
        
        return triggered



class DrumPad:
    """UI 辅助类，用于前端显示和键盘输入"""
    def __init__(self, name, rect, key=None, color=(0, 255, 0), mode='Trigger'):
        self.name = name
        self.rect = rect  # x,y,w,h (normalized 0-1)
        self.key = key
        self.color = color
        self.mode = mode
        self.state = 'Idle'
        self.draw_rect = None  # Pixel coordinates
        
        # 钢琴所需的属性
        self.cooldown_until = 0.0
        self.is_playing = False
        self.channel = None
        self.last_trigger_time = 0.0
        self.hover_start_time = 0.0


class AirDrumApp(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('AirDrum 3D Pro - MVP')
        self.resize(960, 720)

        # Controls
        self.btnStart = QtWidgets.QPushButton('Start')
        self.btnCalib = QtWidgets.QPushButton('Auto Calibrate')
        self.sldSens = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.sldSens.setMinimum(200)
        self.sldSens.setMaximum(1200)
        self.sldSens.setValue(500)
        self.lblSens = QtWidgets.QLabel('Sensitivity: 500mm')
        self.sldVol = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.sldVol.setMinimum(0)
        self.sldVol.setMaximum(100)
        self.sldVol.setValue(80)
        self.lblVol = QtWidgets.QLabel('Volume: 80%')
        self.chkDebug = QtWidgets.QCheckBox('Debug Mode')
        self.chkHandSkeleton = QtWidgets.QCheckBox('Hand Skeleton')
        self.chkHandSkeleton.setChecked(True)  # 默认开启手部骨骼追踪
        self.chkMirror = QtWidgets.QCheckBox('Mirror')
        self.chkMirror.setChecked(True)  # 默认开启镜像
        self.btnExit = QtWidgets.QPushButton('Exit')
        self.lblFps = QtWidgets.QLabel('FPS: 0')
        self.lblDist = QtWidgets.QLabel('Center Depth: N/A')

        top = QtWidgets.QHBoxLayout()
        top.addWidget(self.lblFps)
        top.addWidget(self.lblDist)
        top.addStretch(1)

        bottom = QtWidgets.QHBoxLayout()
        bottom.addWidget(self.btnStart)
        bottom.addWidget(self.btnCalib)
        bottom.addWidget(self.lblSens)
        bottom.addWidget(self.sldSens)
        bottom.addWidget(self.lblVol)
        bottom.addWidget(self.sldVol)
        bottom.addWidget(self.chkDebug)
        bottom.addWidget(self.chkHandSkeleton)
        bottom.addWidget(self.chkMirror)
        bottom.addStretch(1)
        bottom.addWidget(self.btnExit)

        self.canvas = QtWidgets.QLabel()
        self.canvas.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.canvas.setStyleSheet('background:black')

        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(top)
        layout.addWidget(self.canvas, 1)
        layout.addLayout(bottom)

        # MediaPipe 手部追踪器初始化
        self.hand_tracker = HandTracker(hands, mp_drawing)
        if hands is None:
            self.chkHandSkeleton.setEnabled(False)
            self.chkHandSkeleton.setToolTip("MediaPipe not available")
            self.use_hand_skeleton = False
        else:
            self.use_hand_skeleton = True  # 开启手部骨骼追踪模式

        # Orbbec RGB-D Camera initialization using OpenCV CAP_OBSENSOR
        self.use_orbbec = False
        self.cap = None
        
        # Try to open Orbbec camera via OpenCV's built-in support
        self.cap = cv2.VideoCapture(0, cv2.CAP_OBSENSOR)
        if self.cap.isOpened():
            self.use_orbbec = True
            print("Orbbec camera initialized successfully via CAP_OBSENSOR.")
        else:
            # Fallback to regular webcam with pseudo-depth
            self.cap = cv2.VideoCapture(0)
            if self.cap.isOpened():
                print("Using fallback webcam with pseudo-depth mode.")
            else:
                print("Failed to open any camera!")
        
        self.running = False
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.tick)
        self.last_tick = time.time()
        self.fps_hist = deque(maxlen=30)

        # Initialize Virtual Drums - Drums (Top half)
        # 三个打击乐鼓，支持双打（独立判定）
        self.drums = [
            VirtualDrum('Kick', (0.14, 0.1, 0.12, 0.2), SOUNDS['Kick'], 
                       trigger_depth=500, release_depth=540, cooldown_ms=100, min_valid_pixels=100),
            VirtualDrum('Boom', (0.34, 0.1, 0.12, 0.2), SOUNDS['Boom'],
                       trigger_depth=500, release_depth=540, cooldown_ms=100, min_valid_pixels=100),
            VirtualDrum('Tom', (0.54, 0.1, 0.12, 0.2), SOUNDS['Tom'],
                       trigger_depth=500, release_depth=540, cooldown_ms=100, min_valid_pixels=100),
        ]
        
        # 保留旧的 pads 用于 UI 绘制和键盘输入
        self.pads = []
        
        # 添加鼓垫的 UI 定义
        drum_colors = [(0, 255, 255), (255, 255, 0), (255, 0, 255)]
        drum_keys = ['A', 'S', 'D']
        for drum, color, key in zip(self.drums, drum_colors, drum_keys):
            pad = DrumPad(drum.name, drum.rect_norm, key=key, color=color, mode='Trigger')
            self.pads.append(pad)
        
        # Piano (Bottom half) - C Major Scale
        keys = ['C', 'D', 'E', 'F', 'G', 'A', 'B']
        kb_keys = ['Z', 'X', 'C', 'V', 'B', 'N', 'M'] # Keyboard mapping for piano
        
        # Calculate piano key layout with gaps - 增大间隔
        n_keys = len(keys)
        key_w = 0.10  # 增大宽度 (from 0.08)
        key_h = 0.28  # 增大高度 (from 0.25)
        key_y = 0.60  # 向上移动 (from 0.65)
        
        # Center the keyboard
        gap = 0.035  # 增大间隔 (from 0.02)
        total_span = (n_keys * key_w) + ((n_keys - 1) * gap)
        start_x = (1.0 - total_span) / 2.0

        for i, note in enumerate(keys):
            px = start_x + i * (key_w + gap)
            self.pads.append(DrumPad(
                f'Piano_{note}', 
                (px, key_y, key_w, key_h), 
                key=kb_keys[i], 
                color=(255, 255, 255),
                mode='Hold'  # Enable Sustain Mode
            ))


        self.Z_trigger_mm = 500  # 用于钢琴
        self.depth_center_mm = None
        self.cooldown_ms = 200  # Increased cooldown to reduce multi-trigger
        self.hover_threshold_ms = 150  # Hover delay before sound starts
        self.min_trigger_interval_ms = 100  # Minimum time between triggers (anti-ghosting)

        # CSV logging
        self.csv_file = open('events.csv', 'a', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(['Timestamp','Event','Instrument','Hit_Depth','Confidence'])

        # Signals
        self.btnStart.clicked.connect(self.toggle_start)
        self.btnCalib.clicked.connect(self.auto_calibrate)
        self.sldSens.valueChanged.connect(self.on_sens_change)
        self.sldVol.valueChanged.connect(self.on_vol_change)
        self.chkDebug.toggled.connect(lambda _: None)
        self.chkHandSkeleton.toggled.connect(self.on_hand_skeleton_toggle)
        self.btnExit.clicked.connect(self.close)

        self.on_vol_change(self.sldVol.value())

    def closeEvent(self, e):
        try:
            self.timer.stop()
            if self.cap is not None:
                self.cap.release()
            self.csv_file.close()
        except Exception:
            pass
        e.accept()

    def on_sens_change(self, v):
        """更新灵敏度 - 同时影响鼓和钢琴"""
        self.Z_trigger_mm = int(v)
        self.lblSens.setText(f'Sensitivity: {v}mm')
        
        # 同时更新所有鼓的触发阈值
        for drum in self.drums:
            drum.trigger_depth = int(v)
            drum.release_depth = int(v) + 40  # 保持 40mm 的迟滞距离

    def on_vol_change(self, v):
        pygame.mixer.music.set_volume(v/100.0)
        self.lblVol.setText(f'Volume: {v}%')

    def on_hand_skeleton_toggle(self, checked):
        """切换手部骨骼追踪模式"""
        self.use_hand_skeleton = checked
        mode = "Hand Skeleton Tracking" if checked else "Depth-based Tracking"
        print(f"[Mode] Switched to {mode}")

    def toggle_start(self):
        self.running = not self.running
        self.btnStart.setText('Stop' if self.running else 'Start')
        if self.running:
            self.timer.start(0)
        else:
            self.timer.stop()

    def auto_calibrate(self):
        # Countdown 3..2..1 per PRD
        for i in [3, 2, 1]:
            self.lblDist.setText(f'Calibrating: {i}..')
            QtWidgets.QApplication.processEvents()
            time.sleep(1)
        
        # Get depth data for calibration
        depth_mm = None
        
        if self.cap is not None and self.cap.grab():
            if self.use_orbbec:
                # Get depth from Orbbec camera
                ret_depth, depth_map = self.cap.retrieve(None, cv2.CAP_OBSENSOR_DEPTH_MAP)
                if ret_depth and depth_map is not None:
                    h, w = depth_map.shape
                    cx0, cy0 = w // 2 - 20, h // 2 - 20
                    roi = depth_map[cy0:cy0+40, cx0:cx0+40]
                    valid = roi[(roi > 0) & (roi < 2000)]
                    if valid.size > 0:
                        depth_mm = int(np.mean(valid))
            else:
                # Fallback to pseudo-depth
                ret, frame = self.cap.retrieve()
                if ret:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    h, w = gray.shape
                    cx0, cy0 = w // 2 - 20, h // 2 - 20
                    roi = gray[cy0:cy0+40, cx0:cx0+40]
                    D_avg = int(np.mean(roi))
                    depth_mm = int(2000 * (1 - D_avg / 255.0))
        
        if depth_mm is not None:
            # Set trigger plane at D_avg - 150mm per PRD
            calibrated_depth = max(200, depth_mm - 150)
            self.Z_trigger_mm = calibrated_depth
            self.sldSens.setValue(calibrated_depth)
            
            # 同时更新所有鼓的触发阈值
            for drum in self.drums:
                drum.trigger_depth = calibrated_depth
                drum.release_depth = calibrated_depth + 40
            
            QtWidgets.QToolTip.showText(
                self.mapToGlobal(QtCore.QPoint(50, 50)),
                f'Calibrated at {self.Z_trigger_mm}mm'
            )
        else:
            QtWidgets.QToolTip.showText(
                self.mapToGlobal(QtCore.QPoint(50, 50)),
                'Calibration failed - no depth data'
            )

    def handle_hit(self, pad, dmin, conf):
        now = time.time()
        now_ms = now * 1000
        
        # Anti-ghosting: Check minimum interval between triggers
        if (now_ms - pad.last_trigger_time) < self.min_trigger_interval_ms:
            return  # Ignore rapid re-triggers
        
        # Determine behavior based on mode
        if pad.mode == 'Trigger':
            # One-shot trigger with cooldown
            if now_ms < pad.cooldown_until:
                return  # Still in cooldown
                
            pad.state = 'Hit'
            pad.cooldown_until = now_ms + self.cooldown_ms
            pad.last_trigger_time = now_ms
            snd = SOUNDS.get(pad.name)
            if snd:
                snd.play()
            # Reset state after a short visual feedback time
            QtCore.QTimer.singleShot(100, lambda: self._reset_pad_state(pad))
            self.csv_writer.writerow([int(now_ms), 'Hit', pad.name, dmin, round(conf, 3)])
            
        elif pad.mode == 'Hold':
            # Continuous Hold (Note On) with hover delay
            if not pad.is_playing:
                # Check if hover duration is sufficient
                if pad.hover_start_time == 0.0:
                    pad.hover_start_time = now_ms
                    return  # Start hover timer
                
                hover_duration = now_ms - pad.hover_start_time
                if hover_duration < self.hover_threshold_ms:
                    return  # Not enough hover time yet
                
                # Trigger sound
                pad.state = 'Hit'
                pad.is_playing = True
                pad.last_trigger_time = now_ms
                snd = SOUNDS.get(pad.name)
                if snd:
                    # Play on a specific channel to allow stopping
                    ch = pygame.mixer.find_channel()
                    if ch:
                        ch.play(snd, loops=-1) # Loop indefinitely
                        pad.channel = ch
                self.csv_writer.writerow([int(now_ms), 'NoteOn', pad.name, dmin, round(conf, 3)])
    
    def _reset_pad_state(self, pad):
        """Reset drum pad state back to Idle after visual feedback"""
        if pad.mode == 'Trigger':
            pad.state = 'Idle'
    
    def handle_release(self, pad):
        now = time.time()
        now_ms = now * 1000
        
        # Reset hover timer
        pad.hover_start_time = 0.0
        
        if pad.mode == 'Hold' and pad.is_playing:
            pad.state = 'Idle'
            pad.is_playing = False
            if pad.channel:
                pad.channel.fadeout(100) # Quick fadeout
                pad.channel = None
            self.csv_writer.writerow([int(now_ms), 'NoteOff', pad.name, 0, 0])

    def keyPressEvent(self, event):
        key = event.text().upper()
        # Find if any pad matches this key
        target_pad = None
        for pad in self.pads:
            if pad.key == key:
                target_pad = pad
                break
        
        if target_pad:
            if not event.isAutoRepeat():
                self.handle_hit(target_pad, 0, 1.0)
        
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        key = event.text().upper()
        target_pad = None
        for pad in self.pads:
            if pad.key == key:
                target_pad = pad
                break
        
        if target_pad and target_pad.mode == 'Hold':
            if not event.isAutoRepeat():
                self.handle_release(target_pad)
                
        super().keyReleaseEvent(event)

    def tick(self):
        now = time.time()
        now_ms = now * 1000
        frame = None
        depth_map = None
        
        if self.cap is None or not self.cap.grab():
            self.lblDist.setText('Camera Disconnected')
            return
        
        if self.use_orbbec:
            # Get BGR and depth from Orbbec camera via CAP_OBSENSOR
            ret_bgr, frame = self.cap.retrieve(None, cv2.CAP_OBSENSOR_BGR_IMAGE)
            ret_depth, depth_map = self.cap.retrieve(None, cv2.CAP_OBSENSOR_DEPTH_MAP)
            
            if not ret_bgr or frame is None:
                self.lblDist.setText('No Color Frame')
                return
            
            if not ret_depth or depth_map is None:
                # Use pseudo-depth if depth not available
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                depth_map = (2000 * (1 - gray.astype(np.float32) / 255.0))
        else:
            # Fallback: use regular webcam
            ret, frame = self.cap.retrieve()
            if not ret or frame is None:
                self.lblDist.setText('Camera Disconnected')
                return
            # Pseudo-depth from grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            depth_map = (2000 * (1 - gray.astype(np.float32) / 255.0))
        
        h, w = frame.shape[:2]
        
        # Resize depth to match color if needed
        if depth_map.shape[:2] != (h, w):
            depth_map = cv2.resize(depth_map.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
        
        # 将深度图转换为 uint16 或保持 float32（确保数据类型一致）
        depth_map = depth_map.astype(np.float32)
        
        # Compute center depth
        ch, cw = h // 2, w // 2
        center_roi = depth_map[ch-20:ch+20, cw-20:cw+20]
        valid_center = center_roi[(center_roi > 0) & (center_roi < 2000)]
        if valid_center.size > 0:
            self.depth_center_mm = int(np.mean(valid_center))
        else:
            self.depth_center_mm = 0
        self.lblDist.setText(f'Center Depth: {self.depth_center_mm}mm')
        
        # Background display
        if self.chkDebug.isChecked():
            # Colorized depth map
            depth_display = np.clip(depth_map / 2000.0 * 255, 0, 255).astype(np.uint8)
            bg = cv2.applyColorMap(depth_display, cv2.COLORMAP_JET)
        else:
            bg = frame

        # ========== 新的独立区域判定逻辑 ==========
        canvas = bg.copy()
        
        # 镜像显示（如果启用）
        mirror_enabled = self.chkMirror.isChecked()
        if mirror_enabled:
            canvas = cv2.flip(canvas, 1)  # 1 表示水平翻转
            depth_map = cv2.flip(depth_map, 1)  # 同步镜像深度图
        
        # 更新手部骨骼追踪
        canvas = self.hand_tracker.update(canvas)
        hand_positions = self.hand_tracker.get_hand_positions()
        
        # 为所有鼓设置帧大小（用于计算 ROI 像素坐标）
        for drum in self.drums:
            drum.set_frame_size(w, h)
        
        # 鼓的触发逻辑：只使用手势识别模式（食指 + 深度 + 速度）
        if hand_positions:
            for drum in self.drums:
                triggered = drum.update_hand_with_depth(hand_positions, depth_map, now_ms, mirror=mirror_enabled)
                if triggered:
                    drum.play()
                    self.csv_writer.writerow([int(now_ms), 'Hit', drum.name, 0, 1.0])
        
        # ========== 钢琴逻辑（保持原有的 Winner-Takes-All）==========
        # 为钢琴键设置帧大小（UI已经通过镜像画面自动调整）
        for pad in self.pads:
            if 'Piano' in pad.name:
                x, y, pw, ph = pad.rect
                rx, ry, rw, rh = int(x*w), int(y*h), int(pw*w), int(ph*h)
                pad.draw_rect = (rx, ry, rw, rh)
        
        # 钢琴的 Winner-Takes-All 逻辑
        piano_candidates = []
        
        # 钢琴触发逻辑：只使用手势识别模式
        if hand_positions:
            for pad in self.pads:
                if 'Piano' in pad.name:
                    x, y, pw, ph = pad.rect
                    piano_rect = (x, y, x + pw, y + ph)
                    
                    # 检查手部是否在钢琴键区域内（已经镜像过）
                    for hand_x, hand_y in hand_positions:
                        check_x = hand_x
                        
                        if piano_rect[0] <= check_x <= piano_rect[2] and piano_rect[1] <= hand_y <= piano_rect[3]:
                            # 获取该位置的深度值
                            # 将归一化坐标直接转换为像素坐标
                            px = int(check_x * w)
                            py = int(hand_y * h)
                            
                            # 确保在范围内
                            px = max(0, min(px, w - 1))
                            py = max(0, min(py, h - 1))
                            
                            hand_depth = depth_map[py, px]
                            
                            # 如果深度满足条件，添加到候选
                            if hand_depth < self.Z_trigger_mm:
                                piano_candidates.append({
                                    'pad': pad,
                                    'dmin': hand_depth,
                                    'conf': 1.0
                                })
                            break
        
        # Piano: 多音和弦模式 (Polyphonic / Multi-Key Support)
        # 所有满足条件的键都可以同时按下
        active_piano_keys = set()
        for candidate in piano_candidates:
            pad = candidate['pad']
            active_piano_keys.add(pad)
            # Hit/Hold 该键
            self.handle_hit(pad, candidate['dmin'], candidate['conf'])
        
        # 3. Process Releases (Piano Only)
        # 释放所有不在 active 列表中的键
        for pad in self.pads:
            if pad.mode == 'Hold':
                if pad not in active_piano_keys:
                    self.handle_release(pad)

        # ========== 绘制 UI ==========
        # 绘制鼓（UI已经通过镜像画面自动调整，不需要再次调整坐标）
        for drum, pad in zip(self.drums, self.pads[:len(self.drums)]):
            x, y, pw, ph = pad.rect
            rx, ry, rw, rh = int(x*w), int(y*h), int(pw*w), int(ph*h)
            pad.draw_rect = (rx, ry, rw, rh)
            
            color = pad.color
            # 根据鼓的状态改变颜色
            if drum.state == drum.STATE_PRESSED:
                color = (0, 0, 255)  # Red for pressed
                cv2.rectangle(canvas, (rx, ry), (rx+rw, ry+rh), color, -1)  # Fill
            else:
                cv2.rectangle(canvas, (rx, ry), (rx+rw, ry+rh), color, 2)  # Outline

            # Label with key
            label = f"{pad.name} [{pad.key}]" if pad.key else pad.name
            cv2.putText(canvas, label, (rx+5, ry+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
        
        # 绘制钢琴键
        for pad in self.pads[len(self.drums):]:
            if pad.draw_rect:
                rx, ry, rw, rh = pad.draw_rect
                
                color = pad.color
                if pad.state == 'Hit':
                    color = (0, 0, 255)  # Red for hit
                    cv2.rectangle(canvas, (rx, ry), (rx+rw, ry+rh), color, -1)  # Fill
                else:
                    cv2.rectangle(canvas, (rx, ry), (rx+rw, ry+rh), color, 2)  # Outline

                # Label with key
                label = f"{pad.name} [{pad.key}]" if pad.key else pad.name
                cv2.putText(canvas, label, (rx+5, ry+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

        # FPS
        dt = now - self.last_tick
        self.last_tick = now
        if dt > 0:
            fps = 1.0/dt
            self.fps_hist.append(fps)
            avg_fps = sum(self.fps_hist)/len(self.fps_hist)
            self.lblFps.setText(f'FPS: {avg_fps:.1f}')
            if avg_fps < 20:
                self.lblFps.setStyleSheet('color:red')
            else:
                self.lblFps.setStyleSheet('color:white')

        # Draw to Qt
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        qimg = QtGui.QImage(rgb.data, w, h, rgb.strides[0], QtGui.QImage.Format_RGB888)
        pix = QtGui.QPixmap.fromImage(qimg)
        self.canvas.setPixmap(pix.scaled(self.canvas.size(), QtCore.Qt.IgnoreAspectRatio))


def main():
    app = QtWidgets.QApplication(sys.argv)
    w = AirDrumApp()
    w.showMaximized()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
