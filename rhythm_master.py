"""
节奏大师模式 - 打击乐谱面系统
音符按谱面下落，击中时播放鼓声（不播放背景音乐）
"""

import json
import os
import time
import pygame
from typing import List, Dict, Optional


class ChartNote:
    """谱面音符"""
    
    def __init__(self, time_ms: float, lane: int, note_type: str = 'short', duration: float = 0, note_name: str = None):
        """
        Args:
            time_ms: 音符应该出现的时间（毫秒）
            lane: 轨道索引 (0=左, 1=右)
            note_type: 音符类型 ('short' 或 'long')
            duration: 长音符持续时间（毫秒）
            note_name: 音符名称（例如 'C', 'D', 'E' 等，用于播放对应音高）
        """
        self.time_ms = time_ms
        self.lane = lane
        self.note_type = note_type
        self.duration = duration
        self.note_name = note_name  # 新增：音符名称
        self.spawned = False  # 是否已显示在屏幕上
        self.y = 0  # 当前Y位置（像素）
        self.judged = False  # 是否已判定
        self.judge_result = None  # 'Perfect', 'Good', 'Miss'
        self.note_length = 0  # 长音符的像素长度
        self.is_held = False  # 长音符是否正在被按住


class Chart:
    """谱面数据"""
    
    def __init__(self, title: str, artist: str, audio_file: str, bpm: float, difficulty: int):
        self.title = title
        self.artist = artist
        self.audio_file = audio_file
        self.bpm = bpm
        self.difficulty = difficulty  # 1-5
        self.notes: List[ChartNote] = []
        self.duration_ms = 0  # 曲目总时长
    
    def add_note(self, time_ms: float, lane: int, note_type: str = 'short', duration: float = 0, note_name: str = None):
        """添加音符"""
        note = ChartNote(time_ms, lane, note_type, duration, note_name)
        self.notes.append(note)
        self.duration_ms = max(self.duration_ms, time_ms + duration)
    
    def sort_notes(self):
        """按时间排序音符"""
        self.notes.sort(key=lambda n: n.time_ms)
    
    def to_dict(self) -> dict:
        """导出为字典"""
        return {
            'title': self.title,
            'artist': self.artist,
            'audio_file': self.audio_file,
            'bpm': self.bpm,
            'difficulty': self.difficulty,
            'notes': [
                {
                    'time': n.time_ms,
                    'lane': n.lane,
                    'type': n.note_type,
                    'duration': n.duration,
                    'note': n.note_name
                }
                for n in self.notes
            ]
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Chart':
        """从字典加载"""
        chart = cls(
            data['title'],
            data['artist'],
            data['audio_file'],
            data['bpm'],
            data['difficulty']
        )
        for note_data in data['notes']:
            chart.add_note(
                note_data['time'],
                note_data['lane'],
                note_data.get('type', 'short'),
                note_data.get('duration', 0),
                note_data.get('note', None)
            )
        chart.sort_notes()
        return chart
    
    def save(self, filepath: str):
        """保存谱面到文件"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        print(f"谱面已保存: {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'Chart':
        """从文件加载谱面"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls.from_dict(data)


class RhythmMasterGame:
    """节奏大师游戏管理器 - 击中音符时播放鼓声"""
    
    def __init__(self, drums):
        self.drums = drums[:2]  # 只使用前两个鼓（左右两个）
        self.active = False
        self.paused = False
        
        # 谱面（不需要背景音乐）
        self.current_chart: Optional[Chart] = None
        self.game_start_time = 0  # 游戏开始时间（毫秒）
        self.current_time_ms = 0  # 当前游戏时间（毫秒）
        
        # 游戏设置
        self.note_speed = 400  # 音符下落速度（像素/秒）
        self.spawn_ahead_time = 2000  # 提前生成音符的时间（毫秒）
        self.judge_line_y = 0.85  # 判定线位置（归一化，屏幕下方85%）
        self.judge_perfect = 80  # Perfect判定范围（像素）
        self.judge_good = 150  # Good判定范围（像素）
        
        # 分数和统计
        self.score = 0
        self.combo = 0
        self.max_combo = 0
        self.perfect_count = 0
        self.good_count = 0
        self.miss_count = 0
        
        # 反馈
        self.feedback_text = ""
        self.feedback_time = 0
        self.feedback_duration = 500
        
        # 轨道颜色
        self.lane_colors = [(0, 255, 255), (255, 255, 0)]  # 青色和黄色
        
        # 手势状态
        self.long_press_lanes = set()
        self.pinch_threshold = 0.05
        
        # 倒计时状态
        self.countdown_active = False
        self.countdown_value = 3
        self.countdown_start_time = 0
    
    def load_chart(self, chart_path: str) -> bool:
        """加载谱面（不需要音乐文件）"""
        try:
            self.current_chart = Chart.load(chart_path)
            print(f"✓ 已加载谱面: {self.current_chart.title} - {self.current_chart.artist}")
            print(f"  音符数: {len(self.current_chart.notes)}")
            return True
            
        except Exception as e:
            print(f"✗ 加载谱面失败: {e}")
            return False
    
    def start_game(self):
        """开始游戏"""
        if not self.current_chart:
            print("✗ 请先加载谱面")
            return False
        
        self.reset()
        self.active = True
        self.game_start_time = time.time() * 1000  # 记录开始时间
        self.paused = False
        
        print(f"✓ 开始游戏: {self.current_chart.title}")
        return True
    
    def stop_game(self):
        """停止游戏"""
        self.active = False
        print("游戏已停止")
    
    def toggle_pause(self):
        """暂停/继续"""
        if not self.active:
            return
        
        self.paused = not self.paused
        # 不需要暂停音乐，因为没有背景音乐
    
    def reset(self):
        """重置游戏状态"""
        self.score = 0
        self.combo = 0
        self.max_combo = 0
        self.perfect_count = 0
        self.good_count = 0
        self.miss_count = 0
        self.feedback_text = ""
        self.current_time_ms = 0
        
        # 重置所有音符的状态
        if self.current_chart:
            for note in self.current_chart.notes:
                note.spawned = False
                note.judged = False
                note.judge_result = None
                note.y = 0
                note.is_held = False
    
    def update(self, current_time_ms: float, frame_height: int):
        """更新游戏状态"""
        if not self.active or self.paused or not self.current_chart:
            return
        
        # 处理倒计时
        if self.countdown_active:
            elapsed = current_time_ms - self.countdown_start_time
            new_countdown = 3 - int(elapsed / 1000)
            
            if new_countdown != self.countdown_value:
                self.countdown_value = new_countdown
            
            # 倒计时结束
            if self.countdown_value <= 0:
                self.countdown_active = False
                # 重新设置游戏开始时间为倒计时结束时间
                self.game_start_time = current_time_ms
                self.current_time_ms = 0
            else:
                # 倒计时期间不更新音符
                return
        
        # 更新游戏时间
        self.current_time_ms = current_time_ms - self.game_start_time
        
        # 检查游戏是否结束（所有音符都已生成并判定）
        all_spawned = all(note.spawned for note in self.current_chart.notes)
        all_judged = all(note.judged for note in self.current_chart.notes)
        
        if all_spawned and all_judged and len(self.current_chart.notes) > 0:
            print("✓ 曲目完成！")
            self.show_final_score()
            self.active = False
            return
        
        judge_y_px = int(self.judge_line_y * frame_height)
        
        # 生成音符（提前生成，让玩家有时间反应）
        for note in self.current_chart.notes:
            if not note.spawned:
                # 计算音符应该在什么时候出现在屏幕顶部
                time_to_judge = note.time_ms - self.current_time_ms
                time_to_spawn = time_to_judge - self.spawn_ahead_time
                
                if time_to_spawn <= 0:
                    note.spawned = True
                    note.y = 0
                    # 计算长音符的像素长度
                    if note.note_type == 'long':
                        note.note_length = (note.duration / 1000.0) * self.note_speed
        
        # 更新已生成的音符
        for note in self.current_chart.notes:
            if note.spawned and not note.judged:
                # 根据时间计算音符应该在的位置
                time_since_spawn = self.current_time_ms - (note.time_ms - self.spawn_ahead_time)
                target_y = (time_since_spawn / 1000.0) * self.note_speed
                note.y = target_y
                
                # 更新长按状态
                if note.note_type == 'long' and note.lane in self.long_press_lanes:
                    note.is_held = True
                else:
                    if note.note_type == 'long':
                        note.is_held = False
        
        # 检查Miss
        for note in self.current_chart.notes:
            if note.spawned and not note.judged:
                if note.note_type == 'short' and note.y > judge_y_px + 100:
                    self.judge_miss(note, current_time_ms)
                elif note.note_type == 'long' and note.y + note.note_length > judge_y_px + self.judge_good:
                    self.judge_miss(note, current_time_ms)
    
    def judge_hit(self, lane: int, current_time_ms: float, frame_height: int) -> Optional[str]:
        """判定击打（短音符）"""
        if not self.active or not self.current_chart:
            return None
        
        judge_y_px = int(self.judge_line_y * frame_height)
        
        # 查找该轨道最近的未判定短音符
        closest_note = None
        closest_distance = float('inf')
        
        for note in self.current_chart.notes:
            if (note.lane == lane and note.spawned and not note.judged and 
                note.note_type == 'short'):
                distance = abs(note.y - judge_y_px)
                if distance < closest_distance:
                    closest_distance = distance
                    closest_note = note
        
        if not closest_note:
            return None
        
        # 判定
        if closest_distance < self.judge_perfect:
            return self.judge_perfect_hit(closest_note, current_time_ms, lane)
        elif closest_distance < self.judge_good:
            return self.judge_good_hit(closest_note, current_time_ms, lane)
        
        return None
    
    def judge_perfect_hit(self, note: ChartNote, current_time_ms: float, lane: int) -> str:
        """Perfect判定"""
        note.judged = True
        note.judge_result = 'Perfect'
        self.perfect_count += 1
        self.score += 100
        self.combo += 1
        self.max_combo = max(self.max_combo, self.combo)
        self.show_feedback("Perfect!", current_time_ms)
        self.play_hit_sound(note, 'Perfect')
        return 'Perfect'
    
    def judge_good_hit(self, note: ChartNote, current_time_ms: float, lane: int) -> str:
        """Good判定"""
        note.judged = True
        note.judge_result = 'Good'
        self.good_count += 1
        self.score += 50
        self.combo += 1
        self.max_combo = max(self.max_combo, self.combo)
        self.show_feedback("Good", current_time_ms)
        self.play_hit_sound(note, 'Good')
        return 'Good'
    
    def judge_miss(self, note: ChartNote, current_time_ms: float):
        """Miss判定"""
        note.judged = True
        note.judge_result = 'Miss'
        self.miss_count += 1
        self.combo = 0
        self.show_feedback("Miss", current_time_ms)
    
    def play_hit_sound(self, note: ChartNote, judge_result: str):
        """播放打击音效 - 根据音符播放对应的音高"""
        # 从主程序导入音色库
        from main import SOUNDS
        
        # 如果音符有指定音高，播放对应的钢琴音
        if note.note_name and f'Piano_{note.note_name}' in SOUNDS:
            sound = SOUNDS[f'Piano_{note.note_name}']
            if sound:
                sound.play()
        else:
            # 降级方案：播放鼓声
            if note.lane < len(self.drums):
                drum = self.drums[note.lane]
                drum.play()
    
    def show_feedback(self, text: str, current_time_ms: float):
        """显示反馈文字"""
        self.feedback_text = text
        self.feedback_time = current_time_ms
    
    def show_final_score(self):
        """显示最终分数"""
        print("\n" + "="*50)
        print(f"曲目: {self.current_chart.title}")
        print(f"最终得分: {self.score}")
        print(f"最大连击: {self.max_combo}")
        print(f"Perfect: {self.perfect_count}")
        print(f"Good: {self.good_count}")
        print(f"Miss: {self.miss_count}")
        
        # 计算准确率
        total = self.perfect_count + self.good_count + self.miss_count
        if total > 0:
            accuracy = (self.perfect_count + self.good_count) / total * 100
            print(f"准确率: {accuracy:.1f}%")
        print("="*50 + "\n")
    
    def get_progress(self) -> float:
        """获取游戏进度 (0.0 - 1.0)"""
        if not self.current_chart or self.current_chart.duration_ms == 0:
            return 0.0
        return min(1.0, self.current_time_ms / self.current_chart.duration_ms)
    
    def detect_long_press(self, hand_landmarks, frame_width: int, frame_height: int):
        """检测长按手势"""
        if hand_landmarks is None:
            self.long_press_lanes.clear()
            return
        
        for hand_lm in hand_landmarks:
            thumb_tip = hand_lm.landmark[4]
            index_tip = hand_lm.landmark[8]
            
            dx = thumb_tip.x - index_tip.x
            dy = thumb_tip.y - index_tip.y
            distance = (dx * dx + dy * dy) ** 0.5
            
            is_pinching = distance < self.pinch_threshold
            
            if is_pinching:
                hand_x = (thumb_tip.x + index_tip.x) / 2
                hand_y = (thumb_tip.y + index_tip.y) / 2
                
                # 判断属于哪个轨道
                for i, drum in enumerate(self.drums):
                    x, y, w, h = drum.rect_norm
                    if x <= hand_x <= x + w and y <= hand_y <= y + h:
                        self.long_press_lanes.add(i)
            else:
                self.long_press_lanes.clear()


def create_demo_chart() -> Chart:
    """创建示例谱面"""
    chart = Chart(
        title="示例曲目",
        artist="AirDrum",
        audio_file="demo_song.wav",
        bpm=120,
        difficulty=3
    )
    
    # 生成简单的节奏谱面（4/4拍，每拍一个音符）
    beat_interval = 60000 / chart.bpm  # 一拍的毫秒数
    
    for measure in range(8):  # 8小节
        for beat in range(4):  # 每小节4拍
            time_ms = (measure * 4 + beat) * beat_interval
            lane = random.randint(0, 1)
            
            # 60%短音符，40%长音符
            if random.random() < 0.6:
                chart.add_note(time_ms, lane, 'short')
            else:
                duration = beat_interval * 2  # 2拍长
                chart.add_note(time_ms, lane, 'long', duration)
    
    chart.sort_notes()
    return chart


if __name__ == '__main__':
    # 测试：创建并保存示例谱面
    os.makedirs('assets/charts', exist_ok=True)
    os.makedirs('assets/music', exist_ok=True)
    
    demo = create_demo_chart()
    demo.save('assets/charts/demo.json')
    print("示例谱面已创建！")
