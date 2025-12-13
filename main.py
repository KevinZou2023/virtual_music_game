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

# 导入节奏大师模块
try:
    from rhythm_master import RhythmMasterGame
    RHYTHM_MASTER_AVAILABLE = True
except ImportError:
    RHYTHM_MASTER_AVAILABLE = False
    print("[Warning] rhythm_master.py not found - Rhythm Master mode disabled")

# Audio init
pygame.mixer.pre_init(44100, -16, 2, 512)
pygame.mixer.init()
pygame.init()
pygame.mixer.set_num_channels(64)  # Allow more simultaneous sounds

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
        wave += (w * 0.4) * np.sin(2 * np.pi * (freq * h * 1.003) * t)

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


# ========== 粒子特效系统 ==========
class Particle:
    """单个粒子"""

    def __init__(self, x, y, color, velocity, gravity=0.3, lifespan=30):
        self.x = x
        self.y = y
        self.color = color  # (B, G, R)
        self.vx, self.vy = velocity
        self.gravity = gravity
        self.lifespan = lifespan
        self.max_lifespan = lifespan
        self.size = random.randint(3, 8)

    def update(self):
        """更新粒子位置和状态"""
        self.x += self.vx
        self.y += self.vy
        self.vy += self.gravity  # 重力影响
        self.lifespan -= 1

    def is_alive(self):
        return self.lifespan > 0

    def get_alpha(self):
        """透明度随寿命衰减"""
        return max(0, self.lifespan / self.max_lifespan)

    def draw(self, canvas):
        """绘制粒子"""
        if not self.is_alive():
            return
        alpha = self.get_alpha()
        # 根据透明度调整颜色亮度
        color = tuple(int(c * alpha) for c in self.color)
        size = int(self.size * alpha)
        if size > 0:
            cv2.circle(canvas, (int(self.x), int(self.y)), size, color, -1)


class ParticleSystem:
    """粒子系统管理器"""

    def __init__(self):
        self.particles = []

    def emit(self, x, y, base_color, count=18):
        """在指定位置发射粒子

        Args:
            x, y: 发射位置（像素坐标）
            base_color: 基础颜色 (B, G, R)
            count: 粒子数量 (15-20)
        """
        for _ in range(count):
            # 随机颜色变化（基于鼓的颜色）
            color_variation = 50
            color = tuple(
                max(0, min(255, c + random.randint(-color_variation, color_variation)))
                for c in base_color
            )
            # 随机速度向量（向四周扩散）
            angle = random.uniform(0, 2 * 3.14159)
            speed = random.uniform(3, 10)
            vx = speed * np.cos(angle)
            vy = speed * np.sin(angle) - random.uniform(2, 5)  # 向上偏移

            particle = Particle(x, y, color, (vx, vy),
                                gravity=random.uniform(0.2, 0.5),
                                lifespan=random.randint(20, 40))
            self.particles.append(particle)

    def update(self):
        """更新所有粒子"""
        for particle in self.particles:
            particle.update()
        # 移除死亡粒子
        self.particles = [p for p in self.particles if p.is_alive()]

    def draw(self, canvas):
        """绘制所有粒子"""
        for particle in self.particles:
            particle.draw(canvas)


# ========== 手势音量推杆 ==========
class VolumeSlider:
    """手势音量推杆"""

    def __init__(self, x_norm=0.90, width_norm=0.045, height_norm=0.72, y_norm=0.15):
        """
        Args:
            x_norm: 推杆X位置（归一化）
            width_norm: 推杆宽度（归一化）
            height_norm: 推杆高度（归一化）
            y_norm: 推杆Y起始位置（归一化）
        """
        self.x_norm = x_norm
        self.width_norm = width_norm
        self.height_norm = height_norm
        self.y_norm = y_norm

        self.volume = 0.8  # 当前音量 (0.0-1.0)
        self.is_pinching = False
        self.pinch_threshold = 0.05  # 归一化距离阈值（约30像素 / 600像素）
        self.is_locked = False  # 是否锁定到推杆

    def update(self, hand_landmarks, frame_width, frame_height):
        """更新推杆状态

        Args:
            hand_landmarks: MediaPipe 手部关键点
            frame_width, frame_height: 帧尺寸

        Returns:
            volume: 当前音量值 (0.0-1.0)
        """
        if hand_landmarks is None:
            self.is_pinching = False
            self.is_locked = False
            return self.volume

        for hand_lm in hand_landmarks:
            # 获取拇指指尖(4)和小指指尖(20) - 用于音量调节
            thumb_tip = hand_lm.landmark[4]
            pinky_tip = hand_lm.landmark[20]

            # 计算欧氏距离（归一化）
            dx = thumb_tip.x - pinky_tip.x
            dy = thumb_tip.y - pinky_tip.y
            distance = np.sqrt(dx * dx + dy * dy)

            # 检测捏合状态
            self.is_pinching = distance < self.pinch_threshold

            if self.is_pinching:
                # 计算手的中心位置（拇指和小指的中点）
                hand_x = (thumb_tip.x + pinky_tip.x) / 2
                hand_y = (thumb_tip.y + pinky_tip.y) / 2

                # 检查是否在推杆区域内
                slider_left = self.x_norm - self.width_norm / 2
                slider_right = self.x_norm + self.width_norm / 2
                slider_top = self.y_norm
                slider_bottom = self.y_norm + self.height_norm

                in_slider_area = (slider_left <= hand_x <= slider_right and
                                  slider_top <= hand_y <= slider_bottom)

                if in_slider_area or self.is_locked:
                    self.is_locked = True
                    # 根据Y轴位置计算音量
                    # Y轴向下增加，所以需要反转
                    relative_y = (hand_y - slider_top) / self.height_norm
                    self.volume = 1.0 - max(0.0, min(1.0, relative_y))
            else:
                self.is_locked = False

        return self.volume

    def draw(self, canvas, frame_width, frame_height):
        """绘制推杆UI"""
        # 计算像素坐标
        x = int(self.x_norm * frame_width)
        y = int(self.y_norm * frame_height)
        w = int(self.width_norm * frame_width)
        h = int(self.height_norm * frame_height)

        # 绘制背景轨道
        track_x = x - w // 2
        cv2.rectangle(canvas, (track_x, y), (track_x + w, y + h), (55, 60, 75), -1)
        cv2.rectangle(canvas, (track_x, y), (track_x + w, y + h), (120, 104, 83), 2)

        # 绘制音量填充
        fill_height = int(h * self.volume)
        fill_y = y + h - fill_height
        # 根据手势状态微调填充亮度
        base_color = (200, 224, 234)
        color = tuple(min(255, c + 20) for c in base_color) if self.is_pinching else base_color
        cv2.rectangle(canvas, (track_x + 2, fill_y), (track_x + w - 2, y + h - 2), color, -1)

        # 绘制滑块
        slider_y = y + h - fill_height
        cv2.rectangle(canvas, (track_x - 6, slider_y), (track_x + w + 6, slider_y + 4), (235, 245, 255), -1)

        # 绘制音量百分比
        vol_text = f"{int(self.volume * 100)}%"
        cv2.putText(canvas, vol_text, (track_x + w + 12, y + h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 245, 255), 2)

        # 绘制标题
        cv2.putText(canvas, "VOL", (track_x, y - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 225, 240), 1)

        # 捏合状态指示
        if self.is_pinching:
            cv2.putText(canvas, "PINCH", (track_x - 10, y - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)


# ========== 节奏游戏模式 ==========
class Note:
    """下落音符"""

    def __init__(self, lane, spawn_time, speed=5, note_type='short', duration=None):
        self.lane = lane  # 轨道索引 (0, 1, 2)
        self.spawn_time = spawn_time
        self.y = 0  # 当前Y位置（像素）
        self.speed = speed
        self.judged = False
        self.judge_result = None  # 'Perfect', 'Good', 'Miss'
        
        # 长短音符机制
        self.note_type = note_type  # 'short' 或 'long'
        self.duration = duration  # 长音符的持续时间（毫秒）
        # 一次性计算长音符的像素长度（范围：150-300 像素）
        self.note_length = (duration * speed / 80) if (note_type == 'long' and duration) else 0
        self.is_held = False  # 长音符是否被按住
        self.hold_start_y = None  # 长按开始时的Y位置

    def update(self):
        """更新音符位置"""
        self.y += self.speed

    def draw(self, canvas, lane_x, lane_width, color, outline_color=None):
        """绘制音符"""
        if self.judged and self.judge_result != 'Miss':
            return

        # 边界使用更深的颜色，便于和填充区分
        def darken(col, factor=0.55):
            return tuple(max(0, int(c * factor)) for c in col)
        outline = outline_color if outline_color is not None else darken(color)

        if self.note_type == 'short':
            # 绘制短音符（点状）
            radius = int(lane_width * 0.3)
            cv2.circle(canvas, (lane_x, int(self.y)), radius, color, -1)
            cv2.circle(canvas, (lane_x, int(self.y)), radius, outline, 2)
        else:
            # 绘制长音符（长条状）
            half_width = int(lane_width * 0.4)
            top_y = int(self.y)
            bottom_y = int(self.y + self.note_length)
            
            # 长条颜色：如果被按住则为绿色，否则为原色
            bar_color = (0, 255, 0) if self.is_held else color
            bar_outline = outline if outline_color is not None else darken(bar_color)
            cv2.rectangle(canvas, (lane_x - half_width, top_y), 
                         (lane_x + half_width, bottom_y), bar_color, -1)
            cv2.rectangle(canvas, (lane_x - half_width, top_y), 
                         (lane_x + half_width, bottom_y), bar_outline, 3)
            
            # 绘制起点和终点的强调圆圈
            cv2.circle(canvas, (lane_x, top_y), 9, color, -1)
            cv2.circle(canvas, (lane_x, top_y), 9, outline, 2)
            cv2.circle(canvas, (lane_x, bottom_y), 9, color, -1)
            cv2.circle(canvas, (lane_x, bottom_y), 9, outline, 2)


class RhythmGame:
    """节奏游戏管理器"""

    def __init__(self, drums):
        self.drums = drums[:2]  # 只保留前两个鼓
        self.active = False
        self.paused = False  # 暂停状态
        self.pause_time = 0  # 暂停时的时间戳
        self.notes = []
        self.score = 0
        self.combo = 0
        self.max_combo = 0

        # 判定区域（底部）
        self.judge_line_y = 0.80  # 归一化Y位置（上移避免与底部UI重叠）
        self.judge_perfect = 80  # 像素（扩大）
        self.judge_good = 150  # 像素（扩大）

        # 音符生成
        self.note_speed = 6
        self.spawn_interval = 800  # 毫秒
        self.last_spawn_time = 0

        # 反馈显示
        self.feedback_text = ""
        self.feedback_time = 0
        self.feedback_duration = 500  # 毫秒

        # 统计
        self.perfect_count = 0
        self.good_count = 0
        self.miss_count = 0

        # 轨道颜色（只保留前两个鼓的颜色）
        # 使用 RGB 提供的颜色转换为 BGR：
        # Lane0 (原蓝)：内填充 RGB(250,235,215) -> BGR(215,235,250)，外描边 RGB(204,136,153) -> BGR(153,136,204)
        # Lane1 (原黄)：内填充 RGB(244,194,194) -> BGR(194,194,244)，外描边 RGB(85,94,80) -> BGR(80,94,85)
        self.lane_colors = [(215, 235, 250), (194, 194, 244)]
        self.lane_outline_colors = [(153, 136, 204), (80, 94, 85)]

        # 拖拽状态
        self.is_dragging = False
        self.dragging_drum_index = -1
        self.pinch_threshold = 0.05  # 捏合阈值（和音量推杆一样）
        
        # 长按手势状态
        self.long_press_lanes = set()  # 正在长按的轨道集合
        self.pinch_duration = 0  # 捏合持续时间
        
        # 谱面模式
        self.chart_mode = False  # 是否使用谱面模式
        self.chart_notes = []  # 从谱面加载的音符列表
        self.chart_start_time = 0  # 谱面开始时间
        self.next_note_index = 0  # 下一个要生成的音符索引

    def toggle(self):
        """切换游戏模式"""
        self.active = not self.active
        if self.active:
            self.reset()
        return self.active

    def toggle_pause(self):
        """暂停/继续游戏"""
        if not self.active:
            return False
        self.paused = not self.paused
        self.pause_time = time.time() * 1000  # 记录暂停时间
        return self.paused

    def reset(self):
        """重置游戏"""
        self.notes = []
        self.score = 0
        self.combo = 0
        self.max_combo = 0
        self.perfect_count = 0
        self.good_count = 0
        self.miss_count = 0
        self.feedback_text = ""
        self.paused = False  # 重置暂停状态
        self.chart_start_time = 0  # 重置谱面开始时间
        self.next_note_index = 0  # 重置音符索引

    def detect_long_press(self, hand_landmarks, drums, frame_width, frame_height, current_time_ms):
        """检测长按手势（捏合状态维持）
        
        Args:
            hand_landmarks: MediaPipe 手部关键点
            drums: VirtualDrum 列表
            frame_width, frame_height: 帧尺寸
            current_time_ms: 当前时间（毫秒）
        """
        if hand_landmarks is None:
            self.long_press_lanes.clear()
            self.pinch_duration = 0
            return
        
        for hand_lm in hand_landmarks:
            # 获取拇指指尖(4)和食指指尖(8)
            thumb_tip = hand_lm.landmark[4]
            index_tip = hand_lm.landmark[8]
            
            # 计算欧氏距离（归一化）
            dx = thumb_tip.x - index_tip.x
            dy = thumb_tip.y - index_tip.y
            distance = (dx * dx + dy * dy) ** 0.5
            
            # 检测捏合状态
            is_pinching = distance < self.pinch_threshold
            
            if is_pinching:
                # 计算手的中心位置
                hand_x = (thumb_tip.x + index_tip.x) / 2
                hand_y = (thumb_tip.y + index_tip.y) / 2
                
                # 增加捏合持续时间
                self.pinch_duration += 16  # 假设 ~60FPS
                
                # 判断该位置属于哪个轨道
                for i, drum in enumerate(drums[:2]):
                    x, y, w, h = drum.rect_norm
                    if x <= hand_x <= x + w and y <= hand_y <= y + h:
                        self.long_press_lanes.add(i)
                        # 在长条音符上维持捏合，不会有新的Perfect/Good判定
                        # 而是让长条音符持续被"按住"
            else:
                # 释放捏合
                self.long_press_lanes.clear()
                self.pinch_duration = 0
    
    def update_dragging(self, hand_landmarks, drums, frame_width, frame_height):
        """更新拖拽状态 - 使用拇指与小指指尖捏合

        Args:
            hand_landmarks: MediaPipe 手部关键点
            drums: VirtualDrum 列表
            frame_width, frame_height: 帧尺寸
        """
        if hand_landmarks is None:
            self.is_dragging = False
            self.dragging_drum_index = -1
            for drum in drums[:2]:
                drum.is_being_dragged = False
            return

        for hand_lm in hand_landmarks:
            # 获取拇指指尖(4)和小指指尖(20)
            thumb_tip = hand_lm.landmark[4]
            pinky_tip = hand_lm.landmark[20]

            # 计算欧氏距离（归一化）
            dx = thumb_tip.x - pinky_tip.x
            dy = thumb_tip.y - pinky_tip.y
            distance = (dx * dx + dy * dy) ** 0.5

            # 检测捏合状态
            is_pinching = distance < self.pinch_threshold

            if is_pinching:
                # 计算手的中心位置（拇指和小指的中点）
                hand_x = (thumb_tip.x + pinky_tip.x) / 2
                hand_y = (thumb_tip.y + pinky_tip.y) / 2

                # 检查是否在某个鼓的范围内
                for i, drum in enumerate(drums[:2]):
                    x, y, w, h = drum.rect_norm
                    drum_center_x = x + w / 2
                    drum_center_y = y + h / 2

                    # 检查距离
                    dist_to_drum = ((hand_x - drum_center_x) ** 2 +
                                    (hand_y - drum_center_y) ** 2) ** 0.5

                    # 拖拽触发范围
                    drag_threshold = max(w, h) / 2 + 0.05

                    if dist_to_drum < drag_threshold or self.dragging_drum_index == i:
                        self.is_dragging = True
                        self.dragging_drum_index = i
                        drum.is_being_dragged = True

                        # 直接更新鼓的位置
                        new_x = hand_x - w / 2
                        new_y = hand_y - h / 2
                        drum.set_position(new_x, new_y)
                        return
            else:
                self.is_dragging = False
                self.dragging_drum_index = -1
                for drum in drums[:2]:
                    drum.is_being_dragged = False

    def load_chart(self, chart_file):
        """加载谱面文件"""
        import json
        try:
            with open(chart_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.chart_notes = []
            for note_data in data.get('notes', []):
                note_info = {
                    'time_ms': note_data['time'],
                    'lane': note_data['lane'],
                    'note_type': note_data.get('type', 'short'),
                    'duration': note_data.get('duration', 0)
                }
                self.chart_notes.append(note_info)
            
            # 按时间排序
            self.chart_notes.sort(key=lambda n: n['time_ms'])
            self.chart_mode = True
            self.next_note_index = 0
            print(f"[RhythmGame] 已加载谱面: {chart_file}, 共 {len(self.chart_notes)} 个音符")
            return True
        except Exception as e:
            print(f"[RhythmGame] 加载谱面失败: {e}")
            return False
    
    def update(self, current_time_ms, frame_height):
        """更新游戏状态"""
        if not self.active or self.paused:
            return

        judge_y_px = int(self.judge_line_y * frame_height)

        # 根据模式生成音符
        if self.chart_mode:
            # 谱面模式：按照谱面时间生成音符
            if self.chart_start_time == 0:
                self.chart_start_time = current_time_ms
            
            elapsed_time = current_time_ms - self.chart_start_time
            
            # 提前生成音符（提前2秒），让音符有足够时间下落
            spawn_ahead_time = 2000
            
            while self.next_note_index < len(self.chart_notes):
                note_info = self.chart_notes[self.next_note_index]
                if note_info['time_ms'] - elapsed_time <= spawn_ahead_time:
                    # 生成音符
                    if note_info['note_type'] == 'long':
                        note = Note(note_info['lane'], current_time_ms, self.note_speed, 
                                  note_type='long', duration=note_info['duration'])
                    else:
                        note = Note(note_info['lane'], current_time_ms, self.note_speed, 
                                  note_type='short')
                    self.notes.append(note)
                    self.next_note_index += 1
                else:
                    break
        else:
            # 随机模式：原来的逻辑
            if current_time_ms - self.last_spawn_time > self.spawn_interval:
                lane = random.randint(0, 1)
                has_active_note = any(n.lane == lane and not n.judged for n in self.notes)
                
                if not has_active_note:
                    if random.random() < 0.5:
                        duration = random.randint(1500, 2500)
                        note = Note(lane, current_time_ms, self.note_speed, note_type='long', duration=duration)
                    else:
                        note = Note(lane, current_time_ms, self.note_speed, note_type='short')
                    
                    self.notes.append(note)
                    self.last_spawn_time = current_time_ms

        # 更新音符位置和长按状态
        for note in self.notes:
            if not note.judged:
                note.update()
                # 如果是长音符且该轨道正在长按，标记为被按住
                if note.note_type == 'long' and note.lane in self.long_press_lanes:
                    note.is_held = True
                else:
                    # 如果长按结束，重置被按住状态
                    if note.note_type == 'long':
                        note.is_held = False

        # 判定长音符（通过捏合手势）
        self.judge_long_note(frame_height, self.drums, current_time_ms)

        # 检查 Miss（飞出屏幕）
        for note in self.notes:
            if not note.judged:
                # 短音符：在判定线后100像素判定为Miss
                if note.note_type == 'short' and note.y > judge_y_px + 100:
                    note.judged = True
                    note.judge_result = 'Miss'
                    self.miss_count += 1
                    self.combo = 0
                    self.show_feedback("Miss", current_time_ms)
                # 长音符：在长音符完整通过判定线后还未被判定，才标记为Miss
                elif note.note_type == 'long' and note.y + note.note_length > judge_y_px + self.judge_good:
                    # 长音符的终点已经完全通过了判定线的Good范围
                    note.judged = True
                    note.judge_result = 'Miss'
                    self.miss_count += 1
                    self.combo = 0
                    self.show_feedback("Miss", current_time_ms)

        # 移除已判定的音符或飞出屏幕的音符
        self.notes = [n for n in self.notes if not n.judged and n.y < frame_height + 100]

    def judge_hit(self, lane, current_time_ms, frame_height, drums):
        """判定击打（仅用于短音符）

        Args:
            lane: 轨道索引
            current_time_ms: 当前时间
            frame_height: 帧高度
            drums: VirtualDrum 列表

        Returns:
            str: 判定结果 ('Perfect', 'Good', None)
        """
        if not self.active:
            return None

        # 使用实际鼓的 Y 位置
        if lane < len(drums):
            drum = drums[lane]
            x, y, w, h = drum.rect_norm
            judge_y_px = int((y + h / 2) * frame_height)
        else:
            judge_y_px = int(self.judge_line_y * frame_height)

        # 查找该轨道上最近的未判定短音符（只判定短音符）
        closest_note = None
        closest_distance = float('inf')

        for note in self.notes:
            if note.lane == lane and not note.judged and note.note_type == 'short':
                distance = abs(note.y - judge_y_px)
                if distance < closest_distance:
                    closest_distance = distance
                    closest_note = note

        if closest_note is None:
            return None

        # 判定范围
        if closest_distance < self.judge_perfect:
            closest_note.judged = True
            closest_note.judge_result = 'Perfect'
            self.perfect_count += 1
            self.score += 100
            self.combo += 1
            self.max_combo = max(self.max_combo, self.combo)
            self.show_feedback("Perfect!", current_time_ms)
            # 播放命中音效
            self.play_hit_sound('Perfect', lane)
            return 'Perfect'
        elif closest_distance < self.judge_good:
            closest_note.judged = True
            closest_note.judge_result = 'Good'
            self.good_count += 1
            self.score += 50
            self.combo += 1
            self.max_combo = max(self.max_combo, self.combo)
            self.show_feedback("Good", current_time_ms)
            # 播放命中音效
            self.play_hit_sound('Good', lane)
            return 'Good'

        return None

    def play_hit_sound(self, judge_result, lane):
        """播放命中音效
        
        Args:
            judge_result: 判定结果 ('Perfect' 或 'Good')
            lane: 轨道索引
        """
        if lane == 0 and SOUNDS['Kick']:
            SOUNDS['Kick'].play()
        elif lane == 1 and SOUNDS['Tom']:
            SOUNDS['Tom'].play()
        
        # 为 Perfect 额外播放钢琴音
        if judge_result == 'Perfect':
            piano_notes = ['C', 'D', 'E', 'F']
            note_name = f'Piano_{piano_notes[lane % len(piano_notes)]}'
            if note_name in SOUNDS and SOUNDS[note_name]:
                SOUNDS[note_name].play()

    def judge_long_note(self, frame_height, drums, current_time_ms):
        """判定长音符（通过捏合手势）
        
        判定逻辑：
        - 长音符从起点到终点通过判定线的过程中，如果一直处于捏合状态，判定为Perfect
        - 如果有间断捏合，或完全没有捏合，判定为Miss
        """
        # 使用实际鼓的 Y 位置计算判定线
        for lane, drum in enumerate(self.drums):
            if lane >= len(drums):
                judge_y_px = int(self.judge_line_y * frame_height)
            else:
                drum_info = drums[lane]
                x, y, w, h = drum_info.rect_norm
                judge_y_px = int((y + h / 2) * frame_height)
            
            # 查找该轨道上未被判定的长音符
            for note in self.notes:
                if note.lane == lane and not note.judged and note.note_type == 'long':
                    note_start_y = note.y
                    note_end_y = note.y + note.note_length
                    
                    # 判断长音符的起点是否已经到达判定线
                    if note_start_y >= judge_y_px and note.hold_start_y is None:
                        # 长音符起点刚到达判定线，检查是否正在捏合
                        if lane in self.long_press_lanes:
                            # 开始追踪：标记起点y值
                            note.hold_start_y = note_start_y
                            note.is_held = True
                        else:
                            # 没有捏合，开始追踪但标记为未被按住
                            note.hold_start_y = note_start_y
                            note.is_held = False
                    
                    # 检查长音符的终点是否已完全通过判定线
                    if note_end_y > judge_y_px and note.hold_start_y is not None:
                        # 长音符终点已通过判定线
                        
                        # 追踪：在整个过程中，检查是否一直被按住
                        if not (lane in self.long_press_lanes):
                            # 如果在过程中失去了捏合，标记为未被按住
                            note.is_held = False
                        
                        # 检查是否已完全通过判定线（终点已经下降了judge_good距离）
                        if note_end_y > judge_y_px + self.judge_good:
                            # 长音符已完全通过判定线
                            if note.is_held:
                                # 全程都被按住了，判定为Perfect
                                note.judged = True
                                note.judge_result = 'Perfect'
                                self.perfect_count += 1
                                self.score += 100
                                self.combo += 1
                                self.max_combo = max(self.max_combo, self.combo)
                                self.show_feedback("Perfect!", current_time_ms)
                                self.play_hit_sound('Perfect', lane)
                            # 如果is_held为False，会由Miss检查来处理

    def show_feedback(self, text, current_time_ms):
        """显示反馈文字"""
        self.feedback_text = text
        self.feedback_time = current_time_ms

    def draw(self, canvas, frame_width, frame_height, drum_pads, drums):
        """绘制游戏元素"""
        if not self.active:
            return

        # 绘制轨道（使用实际鼓的位置）
        for i in range(2):
            if i < len(drums):
                drum = drums[i]
                x, y, w, h = drum.rect_norm
                lane_center_x = int((x + w / 2) * frame_width)
                lane_width = int(w * frame_width)

                # 半透明轨道背景（从顶部到鼓的位置）
                overlay = canvas.copy()
                lane_left = lane_center_x - lane_width // 2
                drum_y_px = int((y + h / 2) * frame_height)
                cv2.rectangle(overlay, (lane_left, 0), (lane_left + lane_width, drum_y_px + 50),
                              (50, 50, 50), -1)
                cv2.addWeighted(overlay, 0.3, canvas, 0.7, 0, canvas)

                # 轨道边框
                cv2.rectangle(canvas, (lane_left, 0), (lane_left + lane_width, drum_y_px + 50),
                              self.lane_outline_colors[i], 2)

        # 绘制音符（使用实际鼓的位置）
        for note in self.notes:
            if note.lane < 2 and note.lane < len(drums):
                drum = drums[note.lane]
                x, y, w, h = drum.rect_norm
                lane_x = int((x + w / 2) * frame_width)
                lane_width = int(w * frame_width)
                note.draw(canvas, lane_x, lane_width, self.lane_colors[note.lane], self.lane_outline_colors[note.lane])

        # 拖拽提示
        if self.is_dragging:
            cv2.putText(canvas, "DRAGGING (pinch)", (frame_width - 200, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 200), 2)

        # 绘制分数、连击和统计信息
        cv2.putText(canvas, f"SCORE: {self.score}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(canvas, f"COMBO: {self.combo}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        # 绘制 Perfect/Good/Miss 统计
        perf_inner = (178, 202, 172)   # RGB(172,202,178) -> BGR
        good_inner = (209, 181, 113)   # RGB(113,181,209) -> BGR
        miss_inner = (95, 86, 199)     # RGB(199,86,95)  -> BGR
        cv2.putText(canvas, f"Perfect: {self.perfect_count}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, perf_inner, 1)
        cv2.putText(canvas, f"Good: {self.good_count}", (10, 115),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, good_inner, 1)
        cv2.putText(canvas, f"Miss: {self.miss_count}", (10, 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, miss_inner, 1)

        # 绘制反馈文字
        if self.feedback_text:
            # 计算反馈显示时间
            elapsed = time.time() * 1000 - self.feedback_time
            if elapsed < self.feedback_duration:
                alpha = 1.0 - (elapsed / self.feedback_duration)
                color = (0, 255, 0) if 'Perfect' in self.feedback_text else \
                    (0, 255, 255) if 'Good' in self.feedback_text else (0, 0, 255)
                # 放大显示
                scale = 1.5 + (1.0 - alpha) * 0.5
                text_size = cv2.getTextSize(self.feedback_text, cv2.FONT_HERSHEY_SIMPLEX, scale, 3)[0]
                text_x = (frame_width - text_size[0]) // 2
                text_y = frame_height // 2
                cv2.putText(canvas, self.feedback_text, (text_x, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, scale, color, 3)
            else:
                self.feedback_text = ""

        # 游戏模式标识
        cv2.putText(canvas, "RHYTHM MODE", (frame_width - 180, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # 绘制暂停菜单
        if self.paused:
            # 绘制半透明黑色遮罩
            overlay = canvas.copy()
            cv2.rectangle(overlay, (0, 0), (frame_width, frame_height), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, canvas, 0.4, 0, canvas)

            # 绘制暂停文字和提示
            cv2.putText(canvas, "PAUSED", (frame_width // 2 - 100, frame_height // 2 - 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
            cv2.putText(canvas, "Press SPACE to Resume", (frame_width // 2 - 150, frame_height // 2 + 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(canvas, "Press ESC to Quit", (frame_width // 2 - 130, frame_height // 2 + 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)


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
        self.rect_norm = list(rect_norm)  # (x, y, w, h) in [0, 1] - 改为 list 以便修改
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

        # 拖拽状态
        self.is_being_dragged = False

    def set_position(self, x_norm, y_norm):
        """设置鼓的位置（归一化坐标）"""
        self.rect_norm[0] = x_norm
        self.rect_norm[1] = y_norm

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
        roi_depth = depth_frame[y:y + h, x:x + w]

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
                    print(
                        f"[HAND+DEPTH] {self.name} TRIGGERED - depth: {hand_depth}mm, velocity: {depth_velocity}mm/frame")

        elif self.state == self.STATE_PRESSED:
            if not hand_in_roi or (hand_depth is not None and hand_depth > self.release_depth):
                # 从 PRESSED 到 IDLE
                self.state = self.STATE_IDLE
                self.depth_history.clear()  # 清除历史，准备下次触发
                print(f"[HAND+DEPTH] {self.name} RELEASED")

        return triggered



class DifficultyDialog(QtWidgets.QDialog):
    """游戏难度选择对话框 - 美化版"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('选择游戏难度')
        self.setModal(True)
        self.setFixedSize(500, 450)
        self.selected_difficulty = 'normal'
        
        # 设置整体样式
        self.setStyleSheet("""
            QDialog {
                background-color: #2b2b2b;
                color: #ffffff;
            }
            QLabel {
                color: #ffffff;
                font-family: 'Segoe UI', Arial, sans-serif;
            }
            QPushButton {
                border: none;
                border-radius: 8px;
                padding: 15px;
                text-align: left;
                font-family: 'Segoe UI', Arial, sans-serif;
            }
        """)
        
        # 主布局
        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setContentsMargins(30, 30, 30, 30)
        main_layout.setSpacing(20)
        
        # 标题部分
        title_label = QtWidgets.QLabel("DIFFICULTY SELECT")
        title_label.setStyleSheet("font-size: 24px; font-weight: bold; letter-spacing: 2px; color: #00e5ff;")
        title_label.setAlignment(QtCore.Qt.AlignCenter)
        main_layout.addWidget(title_label)
        
        subtitle_label = QtWidgets.QLabel("请选择您的挑战等级")
        subtitle_label.setStyleSheet("font-size: 14px; color: #aaaaaa; margin-bottom: 10px;")
        subtitle_label.setAlignment(QtCore.Qt.AlignCenter)
        main_layout.addWidget(subtitle_label)
        
        # 难度按钮辅助函数
        def create_difficulty_btn(name, desc, color_base, color_hover, diff_code):
            btn = QtWidgets.QPushButton()
            btn.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
            btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: {color_base};
                    color: white;
                    border-left: 5px solid {color_hover};
                }}
                QPushButton:hover {{
                    background-color: {color_hover};
                    padding-left: 20px; /* 简单的动画效果 */
                }}
                QPushButton:pressed {{
                    background-color: {color_base};
                }}
            """)
            
            # 按钮内部布局
            btn_layout = QtWidgets.QVBoxLayout(btn)
            btn_layout.setContentsMargins(15, 10, 15, 10)
            
            lbl_name = QtWidgets.QLabel(name)
            lbl_name.setStyleSheet("font-size: 18px; font-weight: bold; background: transparent;")
            
            lbl_desc = QtWidgets.QLabel(desc)
            lbl_desc.setStyleSheet("font-size: 12px; color: rgba(255,255,255,0.8); background: transparent;")
            
            btn_layout.addWidget(lbl_name)
            btn_layout.addWidget(lbl_desc)
            
            btn.clicked.connect(lambda: self.select_difficulty(diff_code))
            return btn

        # 添加按钮
        # 简单：青绿色
        btn_easy = create_difficulty_btn(
            "EASY  |  简单", 
            "适合初学者，音符速度较慢", 
            "#2d4a3e", "#27ae60", 'easy'
        )
        main_layout.addWidget(btn_easy)
        
        # 普通：深蓝色
        btn_normal = create_difficulty_btn(
            "NORMAL  |  普通", 
            "标准的节奏体验，适中的速度", 
            "#2c3e50", "#2980b9", 'normal'
        )
        main_layout.addWidget(btn_normal)
        
        # 困难：深红色
        btn_hard = create_difficulty_btn(
            "HARD  |  困难", 
            "极速挑战，考验你的反应极限", 
            "#502c2c", "#c0392b", 'hard'
        )
        main_layout.addWidget(btn_hard)
        
        main_layout.addStretch()
        
        # 底部提示
        footer = QtWidgets.QLabel("Press ESC to Cancel")
        footer.setStyleSheet("color: #666666; font-size: 10px;")
        footer.setAlignment(QtCore.Qt.AlignCenter)
        main_layout.addWidget(footer)

    def center_on_screen(self):
        """将对话框移动到屏幕中心"""
        desktop = QtWidgets.QDesktopWidget()
        screen_rect = desktop.screenGeometry()
        dialog_rect = self.geometry()
        
        # 计算中心位置
        x = (screen_rect.width() - dialog_rect.width()) // 2
        y = (screen_rect.height() - dialog_rect.height()) // 2
        
        self.move(x, y)
    
    def select_difficulty(self, difficulty):
        """选择难度并关闭对话框"""
        self.selected_difficulty = difficulty
        set_difficulty(difficulty)
        self.accept()
    
    def get_selected_difficulty(self):
        """获取选中的难度"""
        return self.selected_difficulty


class Difficulty:
    EASY = {'speed': 1, 'density': 1}
    NORMAL = {'speed': 2, 'density': 2}
    HARD = {'speed': 3, 'density': 3}


current_difficulty = Difficulty.NORMAL


def set_difficulty(level):
    global current_difficulty
    if level == 'easy':
        current_difficulty = Difficulty.EASY
    elif level == 'normal':
        current_difficulty = Difficulty.NORMAL
    elif level == 'hard':
        current_difficulty = Difficulty.HARD
    print(f"Difficulty set to: {level}")


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
        self.btnStart.setProperty('btnRole', 'primary')
        self.btnCalib = QtWidgets.QPushButton('Auto Calibrate')
        self.btnCalib.setProperty('btnRole', 'accent')
        self.sldSens = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.sldSens.setMinimum(200)
        self.sldSens.setMaximum(1200)
        self.sldSens.setValue(500)
        self.sldSens.setFixedHeight(18)
        self.sldSens.setFixedWidth(140)
        self.sldSens.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        self.lblSens = QtWidgets.QLabel('Sensitivity: 500mm')
        self.sldVol = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.sldVol.setMinimum(0)
        self.sldVol.setMaximum(100)
        self.sldVol.setValue(80)
        self.sldVol.setFixedHeight(18)
        self.sldVol.setFixedWidth(140)
        self.sldVol.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        self.lblVol = QtWidgets.QLabel('Volume: 80%')
        self.chkDebug = QtWidgets.QCheckBox('Debug Mode')
        self.chkHandSkeleton = QtWidgets.QCheckBox('Hand Skeleton')
        self.chkHandSkeleton.setChecked(True)  # 默认开启手部骨骼追踪
        self.chkMirror = QtWidgets.QCheckBox('Mirror')
        self.chkMirror.setChecked(True)  # 默认开启镜像
        
        # 游戏模式按钮
        self.btnGameMode = QtWidgets.QPushButton('🎮 Game Mode: OFF')
        self.btnGameMode.setCheckable(True)
        self.btnGameMode.setProperty('btnRole', 'info')
        
        # 选择谱面按钮
        self.btnSelectChart = QtWidgets.QPushButton('📋 选择谱面')
        self.btnSelectChart.setProperty('btnRole', 'secondary')
        self.btnSelectChart.setEnabled(RHYTHM_MASTER_AVAILABLE)
        if not RHYTHM_MASTER_AVAILABLE:
            self.btnSelectChart.setToolTip("需要 rhythm_master.py")
        
        self.btnExit = QtWidgets.QPushButton('Exit')
        self.btnExit.setProperty('btnRole', 'danger')
        self.lblFps = QtWidgets.QLabel('FPS: 0')
        self.lblDist = QtWidgets.QLabel('Center Depth: N/A')

        # Bump font sizes for bottom controls to improve readability
        primary_btn_font = QtGui.QFont(self.font())
        primary_btn_font.setPointSize(primary_btn_font.pointSize() + 4)
        primary_btn_font.setBold(True)
        for btn in [
            self.btnStart,
            self.btnCalib,
            self.btnSelectChart,
            self.btnGameMode,
            self.btnExit,
        ]:
            btn.setFont(primary_btn_font)

        secondary_font = QtGui.QFont(primary_btn_font)
        secondary_font.setPointSize(primary_btn_font.pointSize() - 2)
        for control in [
            self.chkDebug,
            self.chkHandSkeleton,
            self.chkMirror,
            self.lblSens,
            self.lblVol,
        ]:
            control.setFont(secondary_font)

        top = QtWidgets.QHBoxLayout()
        top.addWidget(self.lblFps)
        top.addWidget(self.lblDist)
        top.addStretch(1)

        bottom = QtWidgets.QHBoxLayout()
        bottom.setContentsMargins(12, 10, 12, 14)
        bottom.setSpacing(8)
        bottom.addWidget(self.btnStart)
        bottom.addWidget(self.btnCalib)
        # 分组布局让标签贴近滑条
        sens_layout = QtWidgets.QHBoxLayout()
        sens_layout.setContentsMargins(0, 0, 0, 0)
        sens_layout.setSpacing(6)
        sens_layout.setAlignment(QtCore.Qt.AlignLeft)
        sens_layout.addWidget(self.lblSens)
        sens_layout.addWidget(self.sldSens)
        bottom.addLayout(sens_layout)
        bottom.addSpacing(10)

        vol_layout = QtWidgets.QHBoxLayout()
        vol_layout.setContentsMargins(0, 0, 0, 0)
        vol_layout.setSpacing(6)
        vol_layout.setAlignment(QtCore.Qt.AlignLeft)
        vol_layout.addWidget(self.lblVol)
        vol_layout.addWidget(self.sldVol)
        bottom.addLayout(vol_layout)
        bottom.addWidget(self.chkDebug)
        bottom.addWidget(self.chkHandSkeleton)
        bottom.addWidget(self.chkMirror)
        bottom.addWidget(self.btnSelectChart)  # 选择谱面按钮
        bottom.addWidget(self.btnGameMode)  # 游戏模式按钮
        bottom.addWidget(self.btnExit)

        self.canvas = QtWidgets.QLabel()
        # Ignored size policy to prevent pixmap sizeHint from撑大窗口；内容自适应缩放
        self.canvas.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Ignored)
        self.canvas.setScaledContents(True)
        self.canvas.setStyleSheet('background:black')
        self.canvas.setAlignment(QtCore.Qt.AlignCenter)

        # 预置背景图（Start 前显示）
        self.bg_pixmap = None
        bg_path = os.path.join('assets', 'UI', 'background.png')
        if os.path.exists(bg_path):
            self.bg_pixmap = QtGui.QPixmap(bg_path)

        # Dialog background (for all pop-ups)
        self.dialog_bg_path = os.path.join('assets', 'UI', 'bg.png')
        self.dialog_bg_url = self.dialog_bg_path.replace('\\', '/') if os.path.exists(self.dialog_bg_path) else None
        self._install_dialog_stylesheet()

        # 运行标记提前初始化，供背景逻辑使用
        self.running = False

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)
        layout.addLayout(top)
        layout.addWidget(self.canvas, 1)
        layout.addLayout(bottom)

        # 如果有背景图，初始显示
        self._set_background_pixmap()

        self.setStyleSheet("""
            QWidget {
                background-color: #1b1b1b;
            }
            QPushButton {
                min-height: 42px;
                padding: 10px 18px;
                border-radius: 10px;
                border: 1px solid rgba(255,255,255,0.12);
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #3a3a3a, stop:1 #2d2d2d);
                color: #f5f5f5;
                font-weight: 600;
                letter-spacing: 0.2px;
            }
            QPushButton:hover {
                border-color: rgba(255,255,255,0.25);
                transform: translateY(-1px);
            }
            QPushButton:pressed {
                transform: translateY(0);
                background: #252525;
            }
            QPushButton[btnRole="primary"] {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #2ecc71, stop:1 #27ae60);
                border-color: #1f8a4f;
                color: #0a1d12;
            }
            QPushButton[btnRole="accent"] {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #1abc9c, stop:1 #16a085);
                border-color: #12806a;
                color: #07221c;
            }
            QPushButton[btnRole="secondary"] {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #8e44ad, stop:1 #7d3c98);
                border-color: #5e2d74;
                color: #f7f0ff;
            }
            QPushButton[btnRole="info"] {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #3498db, stop:1 #2980b9);
                border-color: #1f6392;
                color: #e8f4ff;
            }
            QPushButton[btnRole="danger"] {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #e74c3c, stop:1 #c0392b);
                border-color: #992d22;
                color: #2c0c09;
            }
            QPushButton:checked {
                border: 2px solid rgba(255,255,255,0.35);
                box-shadow: 0 0 0 3px rgba(255,255,255,0.08);
            }
            QSlider::groove:horizontal {
                height: 8px;
                margin: 6px 0;
                background: #2e2e2e;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #1abc9c;
                border: 1px solid #0f7a67;
                width: 18px;
                height: 18px;
                margin: -5px 0;
                border-radius: 9px;
            }
            QSlider::sub-page:horizontal {
                background: #20d0b0;
                border-radius: 4px;
            }
            QLabel, QCheckBox {
                color: #f0f0f0;
            }
        """)

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
        # 鼓的颜色，按给定 RGB 转为 BGR： (53,139,138)->(138,139,53), (249,211,210)->(210,211,249), (196,233,226)->(226,233,196)
        drum_colors = [(138, 139, 53), (210, 211, 249), (226, 233, 196)]
        drum_keys = ['A', 'S', 'D']
        for drum, color, key in zip(self.drums, drum_colors, drum_keys):
            pad = DrumPad(drum.name, drum.rect_norm, key=key, color=color, mode='Trigger')
            self.pads.append(pad)

        # Piano (Bottom half) - C Major Scale
        keys = ['C', 'D', 'E', 'F', 'G', 'A', 'B']
        kb_keys = ['Z', 'X', 'C', 'V', 'B', 'N', 'M']  # Keyboard mapping for piano

        # Calculate piano key layout - 左移且更紧凑，避免靠近右侧音量推杆
        n_keys = len(keys)
        key_w = 0.09
        key_h = 0.28
        key_y = 0.60
        gap = 0.02
        start_x = 0.08  # 固定左侧起点，留出右侧空间给手势音量条

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
        self.csv_writer.writerow(['Timestamp', 'Event', 'Instrument', 'Hit_Depth', 'Confidence'])

        # Signals
        self.btnStart.clicked.connect(self.toggle_start)
        self.btnCalib.clicked.connect(self.auto_calibrate)
        self.sldSens.valueChanged.connect(self.on_sens_change)
        self.sldVol.valueChanged.connect(self.on_vol_change)
        self.chkDebug.toggled.connect(self.on_debug_toggle)
        self.chkHandSkeleton.toggled.connect(self.on_hand_skeleton_toggle)
        self.btnSelectChart.clicked.connect(self.select_chart)  # 连接选择谱面按钮
        self.btnGameMode.clicked.connect(self.toggle_game_mode)  # 连接游戏模式按钮
        self.btnExit.clicked.connect(self.close)

        self.on_vol_change(self.sldVol.value())

        # 初始化粒子系统
        self.particle_system = ParticleSystem()

        # 初始化音量推杆
        self.volume_slider = VolumeSlider()
        # 确保初始音量正确，防止声音丢失
        for sound in SOUNDS.values():
            if sound:
                sound.set_volume(0.8)

        # 初始化节奏游戏系统
        self.rhythm_game = None  # 不再使用旧的随机模式
        self.current_chart_path = None  # 当前选择的谱面路径
        
        if RHYTHM_MASTER_AVAILABLE:
            self.rhythm_game = RhythmMasterGame(self.drums)
            # 默认加载小星星谱面
            default_chart = 'assets/charts/twinkle_twinkle.json'
            if os.path.exists(default_chart):
                if self.rhythm_game.load_chart(default_chart):
                    self.current_chart_path = default_chart
                    print(f"✓ 默认加载谱面: {self.rhythm_game.current_chart.title}")
        
        # 保留旧的rhythm_master引用以兼容
        self.rhythm_master = self.rhythm_game
    
    def select_chart(self):
        """选择谱面"""
        if not RHYTHM_MASTER_AVAILABLE or not self.rhythm_game:
            self._show_dialog("错误", "节奏大师模块未安装！", QtWidgets.QMessageBox.Warning)
            return
        
        # 打开文件选择对话框
        chart_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, 
            "选择谱面文件", 
            "assets/charts",
            "JSON 文件 (*.json)"
        )
        
        if not chart_path:
            return
        
        # 加载谱面
        if self.rhythm_game.load_chart(chart_path):
            self.current_chart_path = chart_path
            
            self._show_dialog(
                "谱面已加载",
                (
                    f"<h3>{self.rhythm_game.current_chart.title}</h3>"
                    f"<p><b>曲师:</b> {self.rhythm_game.current_chart.artist}</p>"
                    f"<p><b>难度:</b> {'★' * self.rhythm_game.current_chart.difficulty}</p>"
                    f"<p><b>BPM:</b> {self.rhythm_game.current_chart.bpm}</p>"
                    f"<p><b>音符数:</b> {len(self.rhythm_game.current_chart.notes)}</p>"
                    f"<hr>"
                    f"<p>点击 <b>Game Mode</b> 按钮开始游戏</p>"
                ),
                QtWidgets.QMessageBox.Information,
                rich=True
            )
        else:
            self._show_dialog("错误", "谱面加载失败！请检查文件格式。", QtWidgets.QMessageBox.Critical)


    
    def toggle_game_mode(self):
        """切换游戏模式 - 启动/停止谱面游戏"""
        if not self.rhythm_game:
            self._show_dialog("错误", "游戏系统未初始化！", QtWidgets.QMessageBox.Warning)
            return
        
        if not self.running:
            self._show_dialog("提示", "请先点击 'Start' 按钮启动摄像头！", QtWidgets.QMessageBox.Warning)
            self.btnGameMode.setChecked(False)
            return
        
        # 如果游戏正在运行，停止它
        if self.rhythm_game.active:
            self.rhythm_game.stop_game()
            self.btnGameMode.setChecked(False)
            self.btnGameMode.setText('🎮 Game Mode: OFF')
            self.btnGameMode.setStyleSheet('')
            self.btnSelectChart.setEnabled(True)
            
            # 恢复鼓的位置
            self.drums[0].set_position(0.14, 0.1)
            self.drums[1].set_position(0.34, 0.1)
            
            # 显示最终分数
            if self.rhythm_game.current_chart:
                total_notes = len(self.rhythm_game.current_chart.notes)
                accuracy = 0
                if total_notes > 0:
                    accuracy = (self.rhythm_game.perfect_count + self.rhythm_game.good_count) / total_notes * 100
                
                self._show_dialog(
                    "游戏结束",
                    (
                        f"<h3>🏆 {self.rhythm_game.current_chart.title}</h3>"
                        f"<p><b>最终得分:</b> {self.rhythm_game.score}</p>"
                        f"<p><b>最大连击:</b> {self.rhythm_game.max_combo}</p>"
                        f"<p><b>Perfect:</b> {self.rhythm_game.perfect_count}</p>"
                        f"<p><b>Good:</b> {self.rhythm_game.good_count}</p>"
                        f"<p><b>Miss:</b> {self.rhythm_game.miss_count}</p>"
                        f"<hr>"
                        f"<p><b>准确率:</b> {accuracy:.1f}%</p>"
                    ),
                    QtWidgets.QMessageBox.Information,
                    rich=True
                )
            return
        
        # 启动游戏
        if not self.rhythm_game.current_chart:
            self._show_dialog(
                "提示", 
                "请先选择谱面！\n\n"
                "点击 '📋 选择谱面' 按钮选择谱面\n"
                "或使用默认的小星星谱面",
                QtWidgets.QMessageBox.Warning
            )
            self.btnGameMode.setChecked(False)
            return
        
        # 先显示对话框（深色卡片 + 大字号）
        chart_name = self.rhythm_game.current_chart.title
        intro = self._create_message_box(
            "开始游戏！",
            f"<div style=\"font-size:28px;font-weight:800;color:#4dd0ff;font-family:'Microsoft YaHei','Segoe UI',sans-serif;\">🎮 {chart_name}</div>"
            f"<div style=\"margin-top:16px;font-size:24px;color:#cfd8e3;font-family:'Microsoft YaHei','Segoe UI',sans-serif;\">准备好！3 秒倒计时后开始。</div>"
            f"<div style=\"margin-top:18px;font-size:24px;color:#e8f1ff;line-height:2.2em;font-family:'Microsoft YaHei','Segoe UI',sans-serif;\">"
            f"键盘 <b>A</b> / <b>D</b> 控左右轨<br>"
            f"或用手势捏合触发<br>"
            f"<b>空格</b> = 暂停"
            f"</div>"
            f"<div style=\"margin-top:16px;font-size:24px;color:#f5dd85;font-family:'Microsoft YaHei','Segoe UI',sans-serif;\">音符到达绿线时击打！</div>",
            QtWidgets.QMessageBox.Information,
            rich=True
        )
        intro.exec_()
        
        # 先设置UI和游戏状态
        self.btnGameMode.setChecked(True)
        self.btnGameMode.setText(f'🎮 Game: {chart_name}')
        self.btnGameMode.setStyleSheet('background-color: #4CAF50; color: white;')
        self.btnSelectChart.setEnabled(False)
        
        # 调整鼓的位置到游戏模式
        w = self.drums[0].rect_norm[2]
        h = self.drums[0].rect_norm[3]
        drum_center_y = self.rhythm_game.judge_line_y  # align centers with judge line
        drum_top_y = drum_center_y - h / 2
        self.drums[0].set_position(0.25 - w / 2, drum_top_y)  # 左鼓
        self.drums[1].set_position(0.75 - w / 2, drum_top_y)  # 右鼓
        
        # 启动游戏但设置为倒计时状态
        if self.rhythm_game.start_game():
            # 设置倒计时状态（游戏暂停，显示倒计时）
            self.rhythm_game.countdown_active = True
            self.rhythm_game.countdown_value = 3
            self.rhythm_game.countdown_start_time = time.time() * 1000
        else:
            self._show_dialog("错误", "游戏启动失败！", QtWidgets.QMessageBox.Critical)
            self.btnGameMode.setChecked(False)
    
    def draw_rhythm_master_ui(self, canvas, w, h, current_time_ms):
        """绘制节奏大师游戏UI"""
        if not self.rhythm_master or not self.rhythm_master.active:
            return
        
        game = self.rhythm_master
        chart = game.current_chart
        
        if not chart:
            return
        
        # 半透明背景
        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.3, canvas, 0.7, 0, canvas)
        
        # 判定线
        judge_y = int(game.judge_line_y * h)
        judge_fill = (103, 181, 157)    # RGB(157,181,103) -> BGR
        judge_outline = (179, 154, 212) # RGB(212,154,179) -> BGR
        cv2.line(canvas, (0, judge_y), (w, judge_y), judge_fill, 4)
        cv2.putText(canvas, "JUDGE LINE", (10, judge_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, judge_outline, 2)
        
        # 倒计时显示（覆盖在游戏界面上）
        if game.countdown_active and game.countdown_value > 0:
            countdown_text = str(game.countdown_value)
            font_scale = 10
            thickness = 20
            text_size = cv2.getTextSize(countdown_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
            text_x = (w - text_size[0]) // 2
            text_y = int(h * 0.42)  # move higher so it's fully visible
            
            # 绘制阴影
            cv2.putText(canvas, countdown_text, (text_x + 5, text_y + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness + 5)
            # 绘制主文字
            cv2.putText(canvas, countdown_text, (text_x, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (178, 202, 172), thickness)
            
            # 显示提示文字
            tip_text = "GET READY!"
            tip_size = cv2.getTextSize(tip_text, cv2.FONT_HERSHEY_SIMPLEX, 2, 3)[0]
            tip_x = (w - tip_size[0]) // 2
            tip_y = text_y + 90  # keep closer to countdown number and within view
            cv2.putText(canvas, tip_text, (tip_x, tip_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 2, (235, 216, 183), 3)
            
            # 倒计时期间直接返回，不绘制音符
            return
        
        def darken(col, factor=0.55):
            return tuple(max(0, int(c * factor)) for c in col)
        
        # 绘制音符
        for note in chart.notes:
            if note.spawned and not note.judged:
                if note.lane < len(game.drums):
                    drum = game.drums[note.lane]
                    x, y, dw, dh = drum.rect_norm
                    lane_x = int((x + dw / 2) * w)
                    lane_width = int(dw * w)
                    
                    # 轨道背景
                    cv2.rectangle(canvas, 
                                (lane_x - lane_width // 2, 0),
                                (lane_x + lane_width // 2, h),
                                (50, 50, 50), 1)
                    
                    # 音符
                    color = game.lane_colors[note.lane]
                    outline = game.lane_outline_colors[note.lane]
                    if note.note_type == 'short':
                        radius = int(lane_width * 0.3)
                        cv2.circle(canvas, (lane_x, int(note.y)), radius, color, -1)
                        cv2.circle(canvas, (lane_x, int(note.y)), radius, outline, 2)
                    else:
                        # 长音符
                        half_width = int(lane_width * 0.4)
                        top_y = int(note.y)
                        bottom_y = int(note.y + note.note_length)
                        bar_color = (0, 255, 0) if note.is_held else color
                        bar_outline = outline
                        cv2.rectangle(canvas, (lane_x - half_width, top_y),
                                    (lane_x + half_width, bottom_y), bar_color, -1)
                        cv2.rectangle(canvas, (lane_x - half_width, top_y),
                                    (lane_x + half_width, bottom_y), bar_outline, 3)
                        cv2.circle(canvas, (lane_x, top_y), 9, color, -1)
                        cv2.circle(canvas, (lane_x, top_y), 9, outline, 2)
                        cv2.circle(canvas, (lane_x, bottom_y), 9, color, -1)
                        cv2.circle(canvas, (lane_x, bottom_y), 9, outline, 2)
        
        # 分数面板
        panel_h = 120
        cv2.rectangle(canvas, (10, 10), (300, panel_h), (0, 0, 0), -1)
        cv2.rectangle(canvas, (10, 10), (300, panel_h), (255, 255, 255), 2)
        
        cv2.putText(canvas, f"SCORE: {game.score}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        combo_color = (255, 215, 0) if game.combo > 10 else (255, 255, 0)
        cv2.putText(canvas, f"COMBO: {game.combo}", (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, combo_color, 2)
        
        stats_text = f"P:{game.perfect_count} G:{game.good_count} M:{game.miss_count}"
        cv2.putText(canvas, stats_text, (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # 判定反馈
        if game.feedback_text and current_time_ms - game.feedback_time < game.feedback_duration:
            feedback_size = 2.5 if 'Perfect' in game.feedback_text else 2.0
            if 'Perfect' in game.feedback_text:
                inner = (178, 202, 172)    # RGB(172,202,178)
                outline = (179, 154, 212)  # RGB(212,154,179)
            elif 'Good' in game.feedback_text:
                inner = (209, 181, 113)    # RGB(113,181,209)
                outline = (92, 161, 248)   # RGB(248,161,92)
            else:  # Miss
                inner = (95, 86, 199)      # RGB(199,86,95)
                outline = (201, 208, 213)  # RGB(213,208,201)
            
            text_size = cv2.getTextSize(game.feedback_text, cv2.FONT_HERSHEY_SIMPLEX, 
                                        feedback_size, 4)[0]
            text_x = (w - text_size[0]) // 2
            text_y = h // 2
            
            cv2.putText(canvas, game.feedback_text, (text_x + 3, text_y + 3),
                       cv2.FONT_HERSHEY_SIMPLEX, feedback_size, outline, 6)
            cv2.putText(canvas, game.feedback_text, (text_x, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, feedback_size, inner, 4)
        
        # 进度条
        progress = game.get_progress()
        bar_width = w - 40
        bar_x = 20
        bar_y = h - 32  # push slightly lower to separate from judge line
        bar_height = 16  # thinner bar

        cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), 
                     (50, 50, 50), -1)
        cv2.rectangle(canvas, (bar_x, bar_y), 
                     (bar_x + int(bar_width * progress), bar_y + bar_height), 
                     (230, 240, 242), -1)
        cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), 
                     (75, 0, 139), 2)
        
        progress_text = f"{int(progress * 100)}%"
        cv2.putText(canvas, progress_text, (bar_x + bar_width + 10, bar_y + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # OpenCV 默认字体不支持中文，非 ASCII 会显示为问号，做一次 ASCII 友好化
        def ascii_safe(text):
            cleaned = ''.join(ch if ord(ch) < 128 else ' ' for ch in text).strip()
            return cleaned or "Song"

        title_txt = ascii_safe(chart.title)
        artist_txt = ascii_safe(chart.artist)
        cv2.putText(canvas, f"{title_txt} - {artist_txt}", (bar_x, bar_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

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
        pygame.mixer.music.set_volume(v / 100.0)
        self.lblVol.setText(f'Volume: {v}%')

    def on_debug_toggle(self, checked):
        """仅在摄像头开启后允许进入 Debug 模式，并在无摄像头时提示"""
        if not checked:
            return

        if not self.running:
            self._show_thematic_info(
                "提示",
                "请先点击 <b>Start</b> 启动摄像头，再开启 Debug 模式。"
            )
            self._reset_debug_checkbox()
            return

        if self.cap is None or not self.cap.isOpened():
            self._show_thematic_info(
                "提示",
                "未检测到可用摄像头，无法进入 Debug 模式。"
            )
            self._reset_debug_checkbox()
            return

    def _reset_debug_checkbox(self):
        """关闭 Debug 复选框且不触发信号"""
        self.chkDebug.blockSignals(True)
        self.chkDebug.setChecked(False)
        self.chkDebug.blockSignals(False)

    def _build_dialog_styles(self):
        """构造统一的弹窗样式，使用 bg.png 作为背景"""
        styles = [
            "QMessageBox {",
            "  background-color: transparent;",
            "  color: #e8f1ff;",
            "  border: none;",
            "  padding: 30px;",
            "  min-width: 900px;",
            "  min-height: 540px;",
            "  font-family: 'Microsoft YaHei', 'Segoe UI', sans-serif;",
        ]
        if self.dialog_bg_url:
            styles.extend([
                f"  background-image: url('{self.dialog_bg_url}');",
                "  background-position: center;",
                "  background-repeat: no-repeat;",
                "  background-attachment: fixed;",
        ])
        styles.extend([
            "}",
            "QMessageBox QLabel { color: #e8f1ff; font-size: 24px; line-height: 2.2em; background-color: transparent; font-family: 'Microsoft YaHei', 'Segoe UI', sans-serif; }",
            "QMessageBox QPushButton {",
            "  min-width: 140px;",
            "  padding: 16px 24px;",
            "  border-radius: 16px;",
            "  background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #22c1c3, stop:1 #4158d0);",
            "  color: #0c1224;",
            "  font-weight: 800;",
            "  border: none;",
            "  font-family: 'Microsoft YaHei', 'Segoe UI', sans-serif;",
        ])
        styles.extend([
            "}",
            "QMessageBox QPushButton:hover {",
            "  background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #2cd6d8, stop:1 #4f69e8);",
            "}",
            "QMessageBox QPushButton:pressed {",
            "  background: #1b2a4f;",
            "}"
        ])
        return "\n".join(styles)

    def _apply_dialog_style(self, box):
        box.setStyleSheet(self._build_dialog_styles())

    def _install_dialog_stylesheet(self):
        """将弹窗样式注入到全局，覆盖所有 QMessageBox"""
        app = QtWidgets.QApplication.instance()
        if not app:
            return
        existing = app.styleSheet() or ""
        extra = self._build_dialog_styles()
        combined = (existing + "\n" + extra).strip()
        app.setStyleSheet(combined)

    def _create_message_box(self, title, text, icon=QtWidgets.QMessageBox.Information, rich=False):
        box = QtWidgets.QMessageBox(self)
        box.setWindowTitle(title)
        box.setIcon(icon)
        text_format = QtCore.Qt.RichText if rich or "<" in text else QtCore.Qt.PlainText
        box.setTextFormat(text_format)
        box.setText(text)
        self._apply_dialog_style(box)
        return box

    def _show_dialog(self, title, text, icon=QtWidgets.QMessageBox.Information, rich=False):
        box = self._create_message_box(title, text, icon, rich)
        box.exec_()

    def _show_thematic_info(self, title, message):
        """带背景图和统一样式的提示框（信息级别）"""
        self._show_dialog(
            title,
            f"<div style=\"line-height:2.0em;font-family:'Microsoft YaHei','Segoe UI',sans-serif;\">{message}</div>",
            QtWidgets.QMessageBox.Information,
            rich=True
        )

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
            self._reset_debug_checkbox()
            self.timer.stop()
            self._stop_all_audio()
            # 停止摄像头后恢复背景图
            self._set_background_pixmap()

    def resizeEvent(self, event):
        """保持背景图在非运行状态下自适应"""
        super().resizeEvent(event)
        if not self.running:
            self._set_background_pixmap()

    def _set_background_pixmap(self):
        """将背景图填充到画布（仅在未运行摄像头时）"""
        if self.bg_pixmap is None or self.running:
            return
        if self.canvas.width() <= 0 or self.canvas.height() <= 0:
            return
        scaled = self.bg_pixmap.scaled(
            self.canvas.size(),
            QtCore.Qt.KeepAspectRatio,
            QtCore.Qt.SmoothTransformation
        )
        self.canvas.setPixmap(scaled)

    def _stop_all_audio(self):
        """停止所有声音并重置按键状态"""
        try:
            pygame.mixer.stop()
            for pad in self.pads:
                pad.state = 'Idle'
                pad.is_playing = False
                pad.hover_start_time = 0.0
                pad.cooldown_until = 0.0
                if getattr(pad, 'channel', None):
                    pad.channel.stop()
                    pad.channel = None
        except Exception:
            pass

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
                    roi = depth_map[cy0:cy0 + 40, cx0:cx0 + 40]
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
                    roi = gray[cy0:cy0 + 40, cx0:cx0 + 40]
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
                        ch.play(snd, loops=-1)  # Loop indefinitely
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
                pad.channel.fadeout(100)  # Quick fadeout
                pad.channel = None
            self.csv_writer.writerow([int(now_ms), 'NoteOff', pad.name, 0, 0])

    def keyPressEvent(self, event):
        key = event.text().upper()
        key_code = event.key()
        
        # 暂停/继续快捷键 (空格)
        if key_code == QtCore.Qt.Key_Space:
            if self.rhythm_game and self.rhythm_game.active:
                self.rhythm_game.toggle_pause()
            return
        
        # 游戏模式的键盘判定
        if self.rhythm_game and self.rhythm_game.active:
            h = self.canvas.height()
            if key == 'A':  # 左轨道
                result = self.rhythm_game.judge_hit(0, time.time() * 1000, h)
                if result:
                    print(f"键盘触发: 左轨道 - {result}")
            elif key == 'D':  # 右轨道
                result = self.rhythm_game.judge_hit(1, time.time() * 1000, h)
                if result:
                    print(f"键盘触发: 右轨道 - {result}")
            return
        
        # ESC 键退出游戏
        if key_code == QtCore.Qt.Key_Escape:
            if self.rhythm_game and self.rhythm_game.paused:
                self.rhythm_game.active = False
                self.rhythm_game.paused = False
                print("已退出游戏")
            return
        
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
        center_roi = depth_map[ch - 20:ch + 20, cw - 20:cw + 20]
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
            # 游戏模式下只处理前两个鼓
            drums_to_process = self.drums[:2] if self.rhythm_game.active else self.drums

            for i, drum in enumerate(drums_to_process):
                triggered = drum.update_hand_with_depth(hand_positions, depth_map, now_ms, mirror=mirror_enabled)
                if triggered:
                    drum.play()
                    self.csv_writer.writerow([int(now_ms), 'Hit', drum.name, 0, 1.0])

                    # 生成粒子特效
                    pad = self.pads[i]
                    x, y, pw, ph = pad.rect
                    particle_x = int((x + pw / 2) * w)
                    particle_y = int((y + ph / 2) * h)
                    self.particle_system.emit(particle_x, particle_y, pad.color, count=random.randint(15, 20))

                    # 节奏游戏判定
                    if self.rhythm_game.active:
                        self.rhythm_game.judge_hit(i, now_ms, h)

        # 更新音量推杆
        if self.hand_tracker.hand_landmarks:
            new_volume = self.volume_slider.update(self.hand_tracker.hand_landmarks, w, h)
            # 只在捏合状态时同步更新 Pygame 音量和 UI 滑块
            if self.volume_slider.is_pinching:
                vol_percent = int(new_volume * 100)
                if vol_percent != self.sldVol.value():
                    self.sldVol.blockSignals(True)
                    self.sldVol.setValue(vol_percent)
                    self.sldVol.blockSignals(False)
                    # 更新所有声音的音量
                    for sound in SOUNDS.values():
                        if sound:
                            sound.set_volume(new_volume)
                    pygame.mixer.music.set_volume(new_volume)

        # 更新游戏模式（谱面系统）
        if self.rhythm_game and self.rhythm_game.active:
            was_active = self.rhythm_game.active
            self.rhythm_game.update(now_ms, h)
            
            # 更新长按检测
            if self.hand_tracker.hand_landmarks:
                self.rhythm_game.detect_long_press(
                    self.hand_tracker.hand_landmarks, w, h
                )
            
            # 检查游戏是否自动结束（所有音符判定完成）
            if was_active and not self.rhythm_game.active:
                # 游戏自动结束，显示最终分数
                self.btnGameMode.setChecked(False)
                self.btnGameMode.setText('🎮 Game Mode: OFF')
                self.btnGameMode.setStyleSheet('')
                self.btnSelectChart.setEnabled(True)
                
                # 恢复鼓的位置
                self.drums[0].set_position(0.14, 0.1)
                self.drums[1].set_position(0.34, 0.1)
                
                # 显示最终分数
                if self.rhythm_game.current_chart:
                    total_notes = len(self.rhythm_game.current_chart.notes)
                    accuracy = 0
                    if total_notes > 0:
                        accuracy = (self.rhythm_game.perfect_count + self.rhythm_game.good_count) / total_notes * 100
                    
                    self._show_dialog(
                        "游戏结束",
                        (
                            f"<h3>🏆 {self.rhythm_game.current_chart.title}</h3>"
                            f"<p><b>最终得分:</b> {self.rhythm_game.score}</p>"
                            f"<p><b>最大连击:</b> {self.rhythm_game.max_combo}</p>"
                            f"<p><b>Perfect:</b> {self.rhythm_game.perfect_count}</p>"
                            f"<p><b>Good:</b> {self.rhythm_game.good_count}</p>"
                            f"<p><b>Miss:</b> {self.rhythm_game.miss_count}</p>"
                            f"<hr>"
                            f"<p><b>准确率:</b> {accuracy:.1f}%</p>"
                        ),
                        QtWidgets.QMessageBox.Information,
                        rich=True
                    )

        # 更新粒子系统
        self.particle_system.update()

        # ========== 钢琴逻辑（保持原有的 Winner-Takes-All）==========
        # 游戏模式下禁用钢琴
        if not (self.rhythm_game and self.rhythm_game.active):
            # 为钢琴键设置帧大小（UI已经通过镜像画面自动调整）
            for pad in self.pads:
                if 'Piano' in pad.name:
                    x, y, pw, ph = pad.rect
                    rx, ry, rw, rh = int(x * w), int(y * h), int(pw * w), int(ph * h)
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

        # 绘制游戏UI（谱面模式）
        if self.rhythm_game and self.rhythm_game.active:
            self.draw_rhythm_master_ui(canvas, w, h, now_ms)

        # 绘制鼓（UI已经通过镜像画面自动调整，不需要再次调整坐标）
        for i, (drum, pad) in enumerate(zip(self.drums, self.pads[:len(self.drums)])):
            # 游戏模式下只绘制前两个鼓
            if self.rhythm_game and self.rhythm_game.active and i >= 2:
                continue

            # 使用实际 VirtualDrum 的位置
            x, y, pw, ph = drum.rect_norm

            rx, ry, rw, rh = int(x * w), int(y * h), int(pw * w), int(ph * h)
            pad.draw_rect = (rx, ry, rw, rh)
            # 同步更新 pad.rect 以便击鼓判定正确
            pad.rect = (x, y, pw, ph)

            color = pad.color
            # 根据鼓的状态改变颜色
            if drum.state == drum.STATE_PRESSED:
                # 暖色高亮，不再用刺眼红色
                color = (220, 200, 120)
                cv2.rectangle(canvas, (rx, ry), (rx + rw, ry + rh), color, -1)  # Fill
            elif drum.is_being_dragged:
                # 正在拖拽时显示高亮边框
                cv2.rectangle(canvas, (rx, ry), (rx + rw, ry + rh), (0, 255, 200), 5)  # Thick border
                cv2.rectangle(canvas, (rx, ry), (rx + rw, ry + rh), pad.color, 3)
            else:
                cv2.rectangle(canvas, (rx, ry), (rx + rw, ry + rh), color, 4)  # Outline

            # Label with key，去掉前缀“Piano_”让字更短
            short_name = pad.name.replace('Piano_', '')
            label = f"{short_name} [{pad.key}]" if pad.key else short_name
            cv2.putText(canvas, label, (rx + 5, ry + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1,
                        cv2.LINE_AA)

        # 绘制钢琴键（游戏模式下隐藏）
        if not (self.rhythm_game and self.rhythm_game.active):
            for pad in self.pads[len(self.drums):]:
                if pad.draw_rect:
                    rx, ry, rw, rh = pad.draw_rect

                    color = pad.color
                    if pad.state == 'Hit':
                        color = (220, 200, 120)  # Warm highlight when hit
                        cv2.rectangle(canvas, (rx, ry), (rx + rw, ry + rh), color, -1)  # Fill
                    else:
                        cv2.rectangle(canvas, (rx, ry), (rx + rw, ry + rh), color, 3)  # Outline

                    # Label with key，去掉前缀“Piano_”
                    short_name = pad.name.replace('Piano_', '')
                    label = f"{short_name} [{pad.key}]" if pad.key else short_name
                    cv2.putText(canvas, label, (rx + 5, ry + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1,
                                cv2.LINE_AA)

        # 绘制粒子特效
        self.particle_system.draw(canvas)

        # 绘制音量推杆
        self.volume_slider.draw(canvas, w, h)

        # FPS
        dt = now - self.last_tick
        self.last_tick = now
        if dt > 0:
            fps = 1.0 / dt
            self.fps_hist.append(fps)
            avg_fps = sum(self.fps_hist) / len(self.fps_hist)
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

def show_difficulty_menu():
    """在游戏开始前显示难度选择菜单"""
    print("\n" + "="*50)
    print("选择游戏难度:")
    print("1. 简单 (Easy)   - 音符下降速度慢，生成密度低")
    print("2. 普通 (Normal) - 标准难度")
    print("3. 困难 (Hard)   - 音符下降速度快，生成密度高")
    print("="*50)

    choice = input("请输入数字选择难度 (默认: 2): ").strip()
    if choice == '1':
        set_difficulty('easy')
        print("✓ 已选择：简单难度")
    elif choice == '3':
        set_difficulty('hard')
        print("✓ 已选择：困难难度")
    else:
        set_difficulty('normal')
        print("✓ 已选择：普通难度")
    print()


def main():
    app = QtWidgets.QApplication(sys.argv)
    
    # 创建主窗口
    w = AirDrumApp()
    w.showMaximized()
    
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
