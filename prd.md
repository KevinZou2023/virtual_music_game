

# 产品需求文档 (PRD) - AirDrum 3D Pro

| 文档属性 | 详情 |
| :--- | :--- |
| **项目名称** | AirDrum 3D 虚拟打击乐系统 |
| **版本号** | v1.0.0 (MVP - Minimum Viable Product) |
| **文档状态** | Draft (草稿) / **Final (定稿)** |
| **面向读者** | 算法工程师, UI开发, 测试工程师 |
| **最后更新** | 2023-XX-XX |

-----

## 1\. 产品概述 (Product Overview)

### 1.1 产品定义

AirDrum 3D Pro 是一款基于 Orbbec RGB-D 相机的桌面级增强现实（AR）应用。用户无需佩戴任何设备，仅凭双手在空中的动作即可触发虚拟乐器，实现“隔空打鼓”的低延迟交互体验。

### 1.2 用户故事 (User Story)

> “作为一个用户，我希望坐在电脑前，通过挥舞双手就能听到架子鼓的声音，并且屏幕上能实时看到我的手打在哪个鼓上，就像玩《节奏大师》一样，但我不需要触摸屏幕。”

-----

## 2\. 界面交互设计 (UI/UX Specification) —— **重点**

**布局原则：** 采用“HUD（平视显示器）”风格，底层为实时 RGB 视频流，上层覆盖虚拟 UI 控件。

### 2.1 主界面 (Main Workspace)

界面分为三个区域：**中央交互区**、**顶部状态栏**、**底部控制面板**。

#### **区域 A: 中央交互区 (Canvas)**

  * **A1. 背景层：** 实时显示相机的 RGB 彩色画面 (640x480 或更高)。
  * **A2. 虚拟乐器组件 (Virtual Pads):**
      * 画面中预置 3-4 个半透明矩形/圆形区域，代表不同的鼓（Snare, Hi-Hat, Tom, Crash）。
      * **状态样式定义：**
          * `State_Idle` (待机): 绿色边框 (2px)，内部透明度 10%。
          * `State_Hover` (预备): 当手部深度进入 $Threshold + 10cm$ 范围时，边框变黄，内部透明度 30%。
          * `State_Hit` (触发): 当手部穿过虚拟平面时，边框变红，内部填充红色 (Alpha 80%)，持续 0.1秒后恢复。
  * **A3. 深度反馈游标 (Depth Cursor):**
      * 在手部位置实时渲染一个小圆点。
      * 颜色随深度变化：距离越近越红，越远越蓝（辅助用户感知 Z 轴距离）。

#### **区域 B: 顶部状态栏 (Status Bar)**

位于窗口顶部，高度 40px，半透明黑色背景。

  * **B1. 帧率指示器 (`LBL_FPS`):** 显示当前处理帧率 (e.g., "FPS: 29.5")。若低于 20fps 显示红色警告。
  * **B2. 距离探针 (`LBL_DIST`):** 实时显示画面中心点的 Z 轴距离 (e.g., "Center Depth: 650mm")，用于调试。

#### **区域 C: 底部控制面板 (Control Panel)**

位于窗口底部，高度 80px，包含具体的**交互按钮**。

| 控件ID | 控件名称 | 类型 | 默认值 | 功能描述 | 交互逻辑 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **BTN\_01** | **Start/Stop** | Toggle Button | Stop | 开启/暂停相机流与声音触发 | 点击后文字变为 "Stop"，图标变红；再次点击恢复。 |
| **BTN\_02** | **Auto Calibrate** | Push Button | N/A | 自动校准虚拟平面距离 | **点击后：**<br>1. 倒计时3秒 (屏幕中央显示 3..2..1)。<br>2. 读取画面中心区域的平均深度 $D_{avg}$。<br>3. 设定触发平面 $Z_{trigger} = D_{avg} - 150mm$。<br>4. 弹出 Toast 提示 "Calibrated at 500mm"。 |
| **SLD\_01** | **Sensitivity** | Slider | 500mm | 手动调节触发平面的 Z 值 | 拖动滑块时，屏幕显示当前的数值。向左拖动距离变近，向右变远。 |
| **SLD\_02** | **Volume** | Slider | 80% | 全局音量调节 | 调节音频输出增益。 |
| **BTN\_03** | **Debug Mode** | Checkbox | Off | 开启调试视图 | 勾选后，将 RGB 背景替换为伪彩色深度图 (Colorized Depth Map)，方便观察噪点。 |
| **BTN\_04** | **Exit** | Button | N/A | 退出程序 | 点击弹出确认框 "Are you sure?" |

-----

## 3\. 功能需求 (Functional Requirements)

### 3.1 核心检测算法 (Algorithm)

  * **FR-CORE-01 深度数据对齐:** 系统必须使用 SDK 提供的 `D2C` (Depth to Color) 功能，确保深度图与 RGB 图像素坐标严格对应。
  * **FR-CORE-02 区域碰撞检测:**
      * 输入：当前帧深度图 $D$。
      * 逻辑：遍历每个鼓的 ROI 区域 $R_i$。
      * 判据：计算 $R_i$ 内 **非零像素的 P5 (第5百分位)** 深度值 $d_{min}$。
      * 触发：若 $d_{min} < Z_{trigger}$ 且该区域状态为 `Un-triggered`，则判定为 `HIT`。
  * **FR-CORE-03 防抖动机制 (Debounce):**
      * 触发后，该区域进入 `CoolDown` 状态 (0.15秒)，期间忽略所有输入，防止一次挥手触发多次声音。
  * **FR-CORE-04 迟滞复位 (Hysteresis):**
      * 只有当 $d_{min} > Z_{trigger} + 50mm$ 时，才将状态重置为 `Un-triggered`（允许下一次击打）。

### 3.2 音频反馈系统 (Audio)

  * **FR-AUD-01 多通道混音:** 系统必须支持并发播放（即：左手敲鼓A的同时，右手敲鼓B，两个声音必须同时响起，不能被截断）。
  * **FR-AUD-02 延迟要求:** 从检测到 `HIT` 信号到音频输出的系统延迟不得超过 **100ms**。

-----

## 4\. 非功能需求 (Non-Functional Requirements)

### 4.1 鲁棒性 (Robustness)

  * **NFR-01 距离越界处理:** 当探测距离 \< 0mm (无效点) 或 \> 2000mm (太远) 时，应自动忽略，不产生误触。
  * **NFR-02 设备热插拔:** 若相机中途断开，程序不应崩溃，而应弹出提示 "Camera Disconnected" 并尝试重连。

-----

## 5\. 数据埋点与日志 (Logging) —— *加分项*

为了方便写报告时的“实验分析”，系统应记录以下数据到 `.csv` 文件：

  * `Timestamp`: 毫秒级时间戳。
  * `Event`: Hit / Miss。
  * `Instrument`: Snare / Base / Hi-hat。
  * `Hit_Depth`: 触发时的深度值 (mm)。
  * `Confidence`: 触发区域的有效像素比例。

-----

## 6\. 开发环境与依赖 (Tech Stack)

  * **OS:** Windows 10/11
  * **Language:** Python 3.8+
  * **Camera SDK:** Orbbec SDK (PyOrbbecSdk) / OpenNI2
  * **Computer Vision:** OpenCV (`cv2`) 4.x
  * **Data Processing:** NumPy
  * **Audio Engine:** PyGame (`pygame.mixer`)
  * **GUI Framework:** PyQt5 (进阶版，推荐用于实现上述按钮)

-----

