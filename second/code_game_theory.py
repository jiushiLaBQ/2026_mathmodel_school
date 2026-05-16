# -*- coding: utf-8 -*-
"""
环境要求:
    Python >= 3.10  (推荐 Anaconda 或 Miniconda)
    numpy >= 2.0
    scipy >= 1.14
    matplotlib >= 3.9
    pandas >= 2.2
    seaborn >= 0.13

    安装: pip install -r requirements.txt
    运行: 先运行 first/code1_lagrangian.py 生成 results.npz，再运行本文件

CUMCM 2026 Problem B - 代码3: 攻防博弈模型
基于问题一的13种攻击动作动力学数据, 定义22种防御动作,
构建13×22攻防综合效用矩阵, 用极大极小值准则求解纳什均衡,
确定最优攻防组合。

输入: first/results.npz (由code1生成)
输出: 12+张可视化图表
"""

import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.patches import FancyArrowPatch
import seaborn as sns

# ============================================================
# 全局参数与字体设置
# ============================================================
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
RESULT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'first', 'results.npz')
FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')

# 中文字体
rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False
rcParams['font.size'] = 11
rcParams['figure.dpi'] = 150
rcParams['savefig.dpi'] = 200
rcParams['savefig.bbox'] = 'tight'

# 关节索引
IDX = {
    'HIP_PITCH_L': 0, 'HIP_ROLL_L': 1, 'HIP_YAW_L': 2,
    'KNEE_PITCH_L': 3, 'ANKLE_PITCH_L': 4, 'ANKLE_ROLL_L': 5,
    'HIP_PITCH_R': 6, 'HIP_ROLL_R': 7, 'HIP_YAW_R': 8,
    'KNEE_PITCH_R': 9, 'ANKLE_PITCH_R': 10, 'ANKLE_ROLL_R': 11,
    'WAIST_YAW': 12,
    'SHOULDER_PITCH_L': 13, 'SHOULDER_ROLL_L': 14, 'SHOULDER_YAW_L': 15,
    'ELBOW_PITCH_L': 16, 'ELBOW_YAW_L': 17,
    'SHOULDER_PITCH_R': 18, 'SHOULDER_ROLL_R': 19, 'SHOULDER_YAW_R': 20,
    'ELBOW_PITCH_R': 21, 'ELBOW_YAW_R': 22,
}

N_JOINTS = 23

# 攻击动作名称 (与code1一致)
ATTACK_NAMES = [
    '左直拳', '右直拳', '左摆拳', '右摆拳', '左上勾拳', '右上勾拳',
    '左掌击', '右肘击', '左膝击', '右前蹬', '右侧踢', '右回旋踢', '右后踢'
]
ATTACK_NAMES_EN = [
    'L Straight', 'R Straight', 'L Hook', 'R Hook',
    'L Uppercut', 'R Uppercut', 'L Palm', 'R Elbow',
    'L Knee', 'R Front Kick', 'R Side Kick', 'R Spinning Kick', 'R Back Kick'
]

# 攻击类型: 拳法(0-6), 肘法(7), 腿法(8-12)
ATTACK_TYPES = ['拳法']*7 + ['肘法'] + ['腿法']*5
ATTACK_TYPE_COLORS = {'拳法': '#2196F3', '肘法': '#FF9800', '腿法': '#F44336'}

# 攻击目标区域: 上段(头/颈), 中段(躯干), 下段(腿)
# 0=上段, 1=中段, 2=下段
ATTACK_TARGET = [1, 1, 0, 0, 0, 0, 1, 1, 2, 2, 2, 2, 2]
ATTACK_TARGET_NAMES = ['上段', '中段', '下段']
ATTACK_TARGET_COLORS = {'上段': '#E91E63', '中段': '#9C27B0', '下段': '#3F51B5'}


# ============================================================
# 数据加载
# ============================================================
def load_results():
    if not os.path.exists(RESULT_PATH):
        raise FileNotFoundError(f"未找到结果文件: {RESULT_PATH}")
    try:
        data = np.load(RESULT_PATH, allow_pickle=True)
    except Exception as e:
        raise RuntimeError(f"加载结果文件失败: {e}")
    
    """加载code1的仿真结果"""
    n_motions = int(data['n_motions'])
    n_joints = int(data['n_joints'])

    motion_names_en = [str(s) for s in data['motion_names_en']]

    motions = []
    for i in range(n_motions):
        prefix = f'motion_{i}'
        mr = {
            'name': str(data[f'{prefix}_name']),
            'name_en': motion_names_en[i],
            'time': data[f'{prefix}_time'],
            'q': data[f'{prefix}_q'],
            'qdot': data[f'{prefix}_qdot'],
            'qddot': data[f'{prefix}_qddot'],
            'tau': data[f'{prefix}_tau'],
            'p_com': data[f'{prefix}_p_com'],
            'v_ee': data[f'{prefix}_v_ee'],
            'v_speed': data[f'{prefix}_v_speed'],
            'm_e': data[f'{prefix}_m_e'],
            'E_k': data[f'{prefix}_E_k'],
            'p_momentum': data[f'{prefix}_p_momentum'],
            'peak_v_speed': float(data[f'{prefix}_peak_v_speed']),
            'peak_m_e': float(data[f'{prefix}_peak_m_e']),
            'peak_E_k': float(data[f'{prefix}_peak_E_k']),
            'peak_p_momentum': float(data[f'{prefix}_peak_p_momentum']),
            'peak_tau': float(data[f'{prefix}_peak_tau']),
            'ee_link': int(data[f'{prefix}_ee_link']),
            'T_attack': float(data[f'{prefix}_T_attack']),
            'T_return': float(data[f'{prefix}_T_return']),
            'n_steps': int(data[f'{prefix}_n_steps']),
        }
        motions.append(mr)

    return motions, data


# ============================================================
# 22种防御动作定义
# ============================================================
DEFENSE_NAMES = [
    '左上格挡', '右上格挡', '左中格挡', '右中格挡', '双臂交叉格挡',
    '下潜闪避', '左侧闪避', '右侧闪避', '后仰闪避', '前俯闪避',
    '后撤步', '侧滑步', '前压步',
    '左膝防御', '右膝防御',
    '抱架防御', '低姿态防御', '侧身防御',
    '左反击准备', '右反击准备', '腿法反击准备',
    '全力后撤'
]
DEFENSE_NAMES_EN = [
    'L High Block', 'R High Block', 'L Mid Block', 'R Mid Block', 'X Block',
    'Duck', 'L Dodge', 'R Dodge', 'Lean Back', 'Lean Forward',
    'Step Back', 'Side Step', 'Step Forward',
    'L Knee Guard', 'R Knee Guard',
    'Guard Stance', 'Low Stance', 'Side Stance',
    'L Counter Prep', 'R Counter Prep', 'Kick Counter Prep',
    'Full Retreat'
]

# 防御类型: 格挡(0-4), 闪避(5-9), 步法(10-12), 膝防(13-14), 姿态(15-17), 反击(18-20), 后撤(21)
# 将第150行改为：
DEFENSE_TYPES = ['格挡']*5 + ['闪避']*5 + ['步法']*4 + ['膝防']*2 + ['姿态']*3 + ['反击']*3
# 注意：原来最后一个是单独的'步法'，现在统一包含在前4个步法中
DEFENSE_TYPE_COLORS = {
    '格挡': '#4CAF50', '闪避': '#00BCD4', '步法': '#795548',
    '膝防': '#9E9E9E', '姿态': '#607D8B', '反击': '#FF5722'
}

# 防御目标区域: 0=上段, 1=中段, 2=下段, 3=全身
DEFENSE_TARGET = [
    0, 0, 1, 1, 3,  # 格挡：左上格挡、右上格挡、左中格挡、右中格挡、双臂交叉格挡
    3, 3, 3, 3, 3,  # 闪避：全部改为全身防御（原定义全部为0）
    3, 3, 3,        # 步法
    2, 2,           # 膝防
    3, 3, 3,        # 姿态
    3, 3, 3, 3      # 反击 + 全力后撤
]

def get_defense_motions():
    """定义22种防御动作的关节偏移量

    Returns:
        list of dict, 每个防御动作包含:
            'name': 防御名称
            'delta_q': 关节偏移量 (23,)
            'T_defend': 防御持续时间
            'defense_type': 防御类型
            'target_zone': 防御目标区域
    """
    motions = []

    def make_delta(**kwargs):
        d = np.zeros(N_JOINTS)
        for k, v in kwargs.items():
            d[IDX[k]] = v
        return d

    # d01: 左上格挡 - 左臂上举保护头部
    motions.append({
        'name': '左上格挡', 'delta_q': make_delta(
            SHOULDER_PITCH_L=1.5, SHOULDER_ROLL_L=-0.5, ELBOW_PITCH_L=-1.8,
            SHOULDER_YAW_L=0.3),
        'T_defend': 0.2, 'defense_type': '格挡', 'target_zone': 0
    })

    # d02: 右上格挡 - 右臂上举保护头部
    motions.append({
        'name': '右上格挡', 'delta_q': make_delta(
            SHOULDER_PITCH_R=1.5, SHOULDER_ROLL_R=0.5, ELBOW_PITCH_R=-1.8,
            SHOULDER_YAW_R=-0.3),
        'T_defend': 0.2, 'defense_type': '格挡', 'target_zone': 0
    })

    # d03: 左中格挡 - 左臂横挡保护躯干
    motions.append({
        'name': '左中格挡', 'delta_q': make_delta(
            SHOULDER_PITCH_L=0.8, SHOULDER_ROLL_L=-1.2, ELBOW_PITCH_L=-1.5,
            WAIST_YAW=0.2),
        'T_defend': 0.2, 'defense_type': '格挡', 'target_zone': 1
    })

    # d04: 右中格挡 - 右臂横挡保护躯干
    motions.append({
        'name': '右中格挡', 'delta_q': make_delta(
            SHOULDER_PITCH_R=0.8, SHOULDER_ROLL_R=1.2, ELBOW_PITCH_R=-1.5,
            WAIST_YAW=-0.2),
        'T_defend': 0.2, 'defense_type': '格挡', 'target_zone': 1
    })

    # d05: 双臂交叉格挡 - 双臂交叉保护正面
    motions.append({
        'name': '双臂交叉格挡', 'delta_q': make_delta(
            SHOULDER_PITCH_L=1.0, SHOULDER_PITCH_R=1.0,
            SHOULDER_ROLL_L=-0.8, SHOULDER_ROLL_R=0.8,
            ELBOW_PITCH_L=-1.5, ELBOW_PITCH_R=-1.5),
        'T_defend': 0.25, 'defense_type': '格挡', 'target_zone': 3
    })

    # d06: 下潜闪避 - 快速下蹲躲避上段攻击
    motions.append({
        'name': '下潜闪避', 'delta_q': make_delta(
            HIP_PITCH_L=0.5, HIP_PITCH_R=0.5,
            KNEE_PITCH_L=0.8, KNEE_PITCH_R=0.8,
            ANKLE_PITCH_L=-0.3, ANKLE_PITCH_R=-0.3,
            SHOULDER_PITCH_L=0.3, SHOULDER_PITCH_R=0.3),
        'T_defend': 0.2, 'defense_type': '闪避', 'target_zone': 0
    })

    # d07: 左侧闪避 - 向左侧身躲避
    motions.append({
        'name': '左侧闪避', 'delta_q': make_delta(
            WAIST_YAW=0.4, HIP_ROLL_L=0.2, HIP_ROLL_R=-0.2,
            ANKLE_ROLL_L=-0.15, ANKLE_ROLL_R=0.15),
        'T_defend': 0.2, 'defense_type': '闪避', 'target_zone': 0
    })

    # d08: 右侧闪避 - 向右侧身躲避
    motions.append({
        'name': '右侧闪避', 'delta_q': make_delta(
            WAIST_YAW=-0.4, HIP_ROLL_L=-0.2, HIP_ROLL_R=0.2,
            ANKLE_ROLL_L=0.15, ANKLE_ROLL_R=-0.15),
        'T_defend': 0.2, 'defense_type': '闪避', 'target_zone': 0
    })

    # d09: 后仰闪避 - 上身后仰躲避
    motions.append({
        'name': '后仰闪避', 'delta_q': make_delta(
            WAIST_YAW=0.0, SHOULDER_PITCH_L=-0.5, SHOULDER_PITCH_R=-0.5,
            HIP_PITCH_L=-0.2, HIP_PITCH_R=-0.2),
        'T_defend': 0.2, 'defense_type': '闪避', 'target_zone': 0
    })

    # d10: 前俯闪避 - 上身前倾躲避
    motions.append({
        'name': '前俯闪避', 'delta_q': make_delta(
            HIP_PITCH_L=0.4, HIP_PITCH_R=0.4,
            KNEE_PITCH_L=0.3, KNEE_PITCH_R=0.3),
        'T_defend': 0.2, 'defense_type': '闪避', 'target_zone': 0
    })

    # d11: 后撤步 - 后退拉开距离
    motions.append({
        'name': '后撤步', 'delta_q': make_delta(
            HIP_PITCH_L=-0.3, HIP_PITCH_R=-0.3,
            KNEE_PITCH_L=-0.2, KNEE_PITCH_R=-0.2),
        'T_defend': 0.3, 'defense_type': '步法', 'target_zone': 3
    })

    # d12: 侧滑步 - 横向移动闪避
    motions.append({
        'name': '侧滑步', 'delta_q': make_delta(
            HIP_ROLL_L=0.3, HIP_ROLL_R=-0.3,
            ANKLE_ROLL_L=-0.2, ANKLE_ROLL_R=0.2),
        'T_defend': 0.3, 'defense_type': '步法', 'target_zone': 3
    })

    # d13: 前压步 - 向前逼近施压
    motions.append({
        'name': '前压步', 'delta_q': make_delta(
            HIP_PITCH_L=0.3, HIP_PITCH_R=0.3,
            KNEE_PITCH_L=0.2, KNEE_PITCH_R=0.2),
        'T_defend': 0.3, 'defense_type': '步法', 'target_zone': 3
    })

    # d14: 左膝防御 - 提左膝防御下段
    motions.append({
        'name': '左膝防御', 'delta_q': make_delta(
            HIP_PITCH_L=-1.0, KNEE_PITCH_L=1.5,
            SHOULDER_PITCH_L=0.5, SHOULDER_PITCH_R=0.5),
        'T_defend': 0.25, 'defense_type': '膝防', 'target_zone': 2
    })

    # d15: 右膝防御 - 提右膝防御下段
    motions.append({
        'name': '右膝防御', 'delta_q': make_delta(
            HIP_PITCH_R=-1.0, KNEE_PITCH_R=1.5,
            SHOULDER_PITCH_L=0.5, SHOULDER_PITCH_R=0.5),
        'T_defend': 0.25, 'defense_type': '膝防', 'target_zone': 2
    })

    # d16: 抱架防御 - 拳击抱架护头护体
    motions.append({
        'name': '抱架防御', 'delta_q': make_delta(
            SHOULDER_PITCH_L=0.8, SHOULDER_PITCH_R=0.8,
            SHOULDER_ROLL_L=-0.3, SHOULDER_ROLL_R=0.3,
            ELBOW_PITCH_L=-1.2, ELBOW_PITCH_R=-1.2,
            SHOULDER_YAW_L=0.2, SHOULDER_YAW_R=-0.2),
        'T_defend': 0.2, 'defense_type': '姿态', 'target_zone': 3
    })

    # d17: 低姿态防御 - 降低重心稳定防御
    motions.append({
        'name': '低姿态防御', 'delta_q': make_delta(
            HIP_PITCH_L=0.4, HIP_PITCH_R=0.4,
            KNEE_PITCH_L=0.6, KNEE_PITCH_R=0.6,
            ANKLE_PITCH_L=-0.2, ANKLE_PITCH_R=-0.2,
            SHOULDER_PITCH_L=0.5, SHOULDER_PITCH_R=0.5,
            ELBOW_PITCH_L=-1.0, ELBOW_PITCH_R=-1.0),
        'T_defend': 0.25, 'defense_type': '姿态', 'target_zone': 3
    })

    # d18: 侧身防御 - 侧身减少受击面积
    motions.append({
        'name': '侧身防御', 'delta_q': make_delta(
            WAIST_YAW=0.6, HIP_ROLL_L=0.15, HIP_ROLL_R=-0.15,
            SHOULDER_PITCH_L=0.5, SHOULDER_ROLL_L=-0.5),
        'T_defend': 0.2, 'defense_type': '姿态', 'target_zone': 3
    })

    # d19: 左反击准备 - 防御同时准备左手反击
    motions.append({
        'name': '左反击准备', 'delta_q': make_delta(
            SHOULDER_PITCH_L=0.5, ELBOW_PITCH_L=-1.0,
            SHOULDER_PITCH_R=-0.3, WAIST_YAW=0.2),
        'T_defend': 0.25, 'defense_type': '反击', 'target_zone': 3
    })

    # d20: 右反击准备 - 防御同时准备右手反击
    motions.append({
        'name': '右反击准备', 'delta_q': make_delta(
            SHOULDER_PITCH_R=0.5, ELBOW_PITCH_R=-1.0,
            SHOULDER_PITCH_L=-0.3, WAIST_YAW=-0.2),
        'T_defend': 0.25, 'defense_type': '反击', 'target_zone': 3
    })

    # d21: 腿法反击准备 - 防御同时准备腿法反击
    motions.append({
        'name': '腿法反击准备', 'delta_q': make_delta(
            HIP_PITCH_R=-0.3, KNEE_PITCH_R=0.5,
            SHOULDER_PITCH_L=0.3, SHOULDER_PITCH_R=0.3),
        'T_defend': 0.3, 'defense_type': '反击', 'target_zone': 3
    })

    # d22: 全力后撤 - 大幅后退完全脱离
    motions.append({
        'name': '全力后撤', 'delta_q': make_delta(
            HIP_PITCH_L=-0.5, HIP_PITCH_R=-0.5,
            KNEE_PITCH_L=-0.3, KNEE_PITCH_R=-0.3,
            SHOULDER_PITCH_L=-0.3, SHOULDER_PITCH_R=-0.3),
        'T_defend': 0.4, 'defense_type': '步法', 'target_zone': 3
    })

    return motions


# ============================================================
# 效用矩阵计算
# ============================================================
def compute_utility_matrix(motions, defense_motions, attack_data):
    """计算13×22攻防综合效用矩阵

    U_ij = H_ij - C_i - R_ij

    Args:
        motions: 13种攻击动作的仿真结果
        defense_motions: 22种防御动作定义
        attack_data: code1的原始数据

    Returns:
        U: (13, 22) 效用矩阵
        H: (13, 22) 有效伤害矩阵
        C: (13,) 能量代价向量
        R: (13, 22) 反击风险矩阵
    """
    n_attacks = len(motions)
    n_defenses = len(defense_motions)

    # ============================
    # 1. 计算各攻击的能量代价 C_i
    # ============================
    C = np.zeros(n_attacks)
    for i, mr in enumerate(motions):
        tau = mr['tau']  # (n_steps, 23)
        dt = mr['time'][1] - mr['time'][0] if mr['n_steps'] > 1 else 0.001
        # 力矩L2范数的时间积分
        tau_norm = np.sqrt(np.sum(tau**2, axis=1))
        C[i] = np.trapezoid(tau_norm, dx=dt)

    # 归一化到 [0, 1]
    C_min, C_max = C.min(), C.max()
    if C_max - C_min > 1e-12:
        C_norm = (C - C_min) / (C_max - C_min)
    else:
        C_norm = np.ones(n_attacks) * 0.5

    # ============================
    # 2. 计算有效伤害矩阵 H_ij
    # ============================
    # 基础攻击力: 使用问题一的峰值有效动能
    E_k_base = np.array([mr['peak_E_k'] for mr in motions])

    # 攻击方向匹配度 (拳法攻击上中段, 腿法攻击下段)
    # 攻击方向: 0=向前(拳/肘), 1=向前上(上勾拳), 2=向下(膝击), 3=向前下(前蹬/踢)
    attack_direction = [0, 0, 0, 0, 1, 1, 0, 0, 2, 3, 3, 3, 3]

    # 防御削弱系数矩阵 (defense_reduction[i][j])
    # 格挡类: 方向匹配时削弱0.5-0.7, 不匹配时0.2-0.4
    # 闪避类: 削弱0.6-0.9 (取决于闪避时机)
    # 步法类: 距离衰减0.3-0.6
    # 膝防类: 削弱下段0.7-0.9, 其他0.1-0.2
    # 姿态类: 削弱0.3-0.5
    # 反击类: 削弱0.3-0.5 (兼顾防御和反击准备)
    # 后撤: 距离衰减0.7-0.9

    # 基础削弱系数 (按防御类型)
    base_reduction = {
        '格挡': 0.5, '闪避': 0.7, '步法': 0.5,
        '膝防': 0.5, '姿态': 0.4, '反击': 0.4
    }

    # 方向匹配修正系数 (区分同类型内不同防御)
    def direction_match_factor(attack_idx, defense_idx):
        """攻击方向与防御方向的匹配度, 同类型内不同防御有差异化"""
        atk_dir = attack_direction[attack_idx]
        def_target = DEFENSE_TARGET[defense_idx]
        atk_target = ATTACK_TARGET[attack_idx]
        dname = DEFENSE_NAMES[defense_idx]

        # 格挡类: 匹配目标区域时增强, 左右对称但有细微差异
        if DEFENSE_TYPES[defense_idx] == '格挡':
            if def_target == atk_target or def_target == 3:
                base_f = 1.3
            else:
                base_f = 0.6
            # 双臂交叉格挡覆盖全身, 略优于单臂
            if dname == '双臂交叉格挡':
                return base_f * 1.05
            # 左右格挡对左右攻击有微小差异 (同侧格挡略快)
            if '左' in dname and atk_target in [0, 1]:
                return base_f * 1.02  # 左手格挡对左侧/正面攻击略优
            if '右' in dname and atk_target in [0, 1]:
                return base_f * 0.98
            return base_f

        # 闪避类: 不同闪避方向对不同攻击有显著差异
        elif DEFENSE_TYPES[defense_idx] == '闪避':
            if dname == '下潜闪避':
                # 下潜对上段拳法最有效, 对下段腿法几乎无效
                if atk_target == 0:
                    return 1.3   # 上段: 最佳闪避
                elif atk_target == 1:
                    return 1.0   # 中段: 一般
                else:
                    return 0.9   # 下段: 下潜降低重心，对下段攻击也有效（从0.3提升）

            elif dname in ('左侧闪避', '右侧闪避'):
                # 侧闪对直线攻击(直拳)有效, 对弧线攻击(摆拳)较差
                # 攻击方向0=向前(直拳) → 侧闪有效
                # 攻击方向1=向前上(上勾拳) → 侧闪一般
                side_f = 0.85 if atk_target == 0 else 0.6 if atk_target == 1 else 0.4
                # 左侧闪避对右侧来的攻击更有效, 反之亦然 (简化为对称)
                return side_f

            elif dname == '后仰闪避':
                # 后仰对正面直拳有效, 对侧向攻击较差
                if atk_dir == 0:  # 向前攻击
                    return 1.1
                else:
                    return 0.6

            elif dname == '前俯闪避':
                # 前俯对高位攻击有效, 对下段无效
                if atk_target == 0:
                    return 0.9
                elif atk_target == 1:
                    return 0.7
                else:
                    return 0.3
            return 0.7

        # 步法类: 后撤拉开距离对所有有效但幅度不同, 侧滑对直线攻击好
        elif DEFENSE_TYPES[defense_idx] == '步法':
            if dname == '后撤步':
                return 0.9   # 后撤: 距离衰减, 中等效果
            elif dname == '侧滑步':
                # 侧滑对直线攻击好, 对弧线攻击一般
                if atk_dir == 0:
                    return 1.0
                else:
                    return 0.7
            elif dname == '前压步':
                # 前压缩短距离, 增加被击风险, 但可打断对方
                return 0.6
            elif dname == '全力后撤':
                return 1.2   # 全力后撤: 距离最远
            return 0.8

        # 膝防类: 对下段攻击特别有效, 左右膝防对左右攻击有微差
        # 将第510-517行改为：
        elif DEFENSE_TYPES[defense_idx] == '膝防':
            if atk_target == 2:  # 下段
                if dname == '左膝防御':
                    return 1.5
                else:
                    return 1.45
            else:  # 上段/中段：提膝时身体前倾，有一定防护
                return 0.7  # 从0.25提升到0.7

        # 姿态类: 不同姿态效果不同
        elif DEFENSE_TYPES[defense_idx] == '姿态':
            if dname == '抱架防御':
                return 0.85   # 抱架: 全面防御
            elif dname == '低姿态防御':
                if atk_target == 2:
                    return 0.7  # 低姿态对下段: 一般
                else:
                    return 0.9  # 低姿态对上中段: 降低重心好
            elif dname == '侧身防御':
                return 0.75   # 侧身: 减少受击面积
            return 0.8

        # 反击类: 不同反击准备方式效果不同
        elif DEFENSE_TYPES[defense_idx] == '反击':
            if dname == '左反击准备':
                return 0.85
            elif dname == '右反击准备':
                return 0.82
            elif dname == '腿法反击准备':
                return 0.78  # 腿法反击准备: 防御略弱但反击威胁大
            return 0.8

        # 默认情况
        else:
            dir_factor = 1.0
        # =========================
        # 添加防御持续时间惩罚因子
        # =========================
        T_defend = defense_motions[defense_idx]['T_defend']
        max_defense_time = max(d['T_defend'] for d in defense_motions)
        
        # 防御持续时间越长，效果越差（简单线性衰减）
        # 惩罚因子范围: [0.8, 1.0]
        time_penalty = 0.8 + 0.2 * (1.0 - T_defend / max_defense_time)
        
        # 应用时间惩罚
        return dir_factor * time_penalty

    H = np.zeros((n_attacks, n_defenses))
    for i in range(n_attacks):
        for j in range(n_defenses):
            def_type = DEFENSE_TYPES[j]
            base_red = base_reduction[def_type]
            dir_factor = direction_match_factor(i, j)

            # 有效伤害 = 基础攻击力 × (1 - 削弱系数 × 方向匹配修正)
            reduction = min(base_red * dir_factor, 0.95)  # 最大削弱95%
            H[i, j] = E_k_base[i] * (1.0 - reduction)

    # ============================
    # 3. 计算反击风险矩阵 R_ij
    # ============================
    # 反击风险: 防御后对方反击的可能性
    # 反击准备类防御: R较大 (防御后快速反击, 对方风险高)
    # 纯防御类: R较小
    # 步法类: R中等 (拉开距离后反击机会减少)

    # 反击风险基础值 (按防御类型)
    base_risk = {
        '格挡': 0.1, '闪避': 0.15, '步法': 0.2,
        '膝防': 0.15, '姿态': 0.1, '反击': 0.4
    }

    # 攻击暴露度: 攻击动作越大, 防御后反击机会越高
    attack_exposure = np.array([
        0.3, 0.3, 0.4, 0.4, 0.35, 0.35, 0.25, 0.3,
        0.5, 0.5, 0.6, 0.7, 0.5  # 腿法暴露度更高
    ])

    R = np.zeros((n_attacks, n_defenses))
    for i in range(n_attacks):
        for j in range(n_defenses):
            def_type = DEFENSE_TYPES[j]
            base = base_risk[def_type]
            exposure = attack_exposure[i]
            dname = DEFENSE_NAMES[j]

            # 反击风险 = 基础风险 × 攻击暴露度 × 防御类型修正 × 防御个体差异
            if def_type == '反击':
                # 反击准备类: 不同反击方式风险不同
                if dname == '腿法反击准备':
                    R[i, j] = base * (1.1 + exposure)   # 腿法反击: 转换慢, 风险稍高
                elif dname == '左反击准备':
                    R[i, j] = base * (1.0 + exposure * 0.9)
                else:
                    R[i, j] = base * (1.0 + exposure)
            elif def_type == '步法':
                if dname == '全力后撤':
                    R[i, j] = base * (0.5 + 0.1 * exposure)  # 全力后撤: 距离最远, 反击风险最低
                elif dname == '后撤步':
                    R[i, j] = base * (0.7 + 0.15 * exposure)
                elif dname == '前压步':
                    R[i, j] = base * (1.0 + 0.3 * exposure)  # 前压: 距离近, 风险高
                else:
                    R[i, j] = base * (0.8 + 0.2 * exposure)
            elif def_type == '闪避':
                # 闪避后位置有利可反击, 但不同闪避差异大
                if dname == '下潜闪避':
                    R[i, j] = base * exposure * 0.8   # 下潜后低位, 反击困难
                elif dname in ('左侧闪避', '右侧闪避'):
                    R[i, j] = base * exposure * 1.1   # 侧闪后侧面位, 反击机会好
                else:
                    R[i, j] = base * exposure
            elif def_type == '膝防':
                # 单腿站立, 反击能力受限
                R[i, j] = base * exposure * 0.7
            else:
                R[i, j] = base * exposure

    # ============================
    # 4. 计算效用矩阵
    # ============================
    # 对H进行归一化，使三个分量量级匹配
    H_min, H_max = H.min(), H.max()
    H_norm = (H - H_min) / (H_max - H_min + 1e-12)
    
    # 引入权重系数，调整各分量的重要性
    alpha, beta, gamma = 0.6, 0.25, 0.15  # 可根据实际需求调整
    U = alpha * H_norm - beta * C_norm[:, np.newaxis] - gamma * R

    return U, H, C_norm, R, H_norm


# ============================================================
# 博弈求解
# ============================================================
def find_saddle_points(U):
    """寻找纯策略纳什均衡 (鞍点)

    鞍点条件: U[i*, j] <= U[i*, j*] <= U[i, j*] 对所有i, j成立
    即: 行最小值的最大值 == 列最大值的最小值

    Returns:
        saddle_points: list of (i, j) 鞍点位置
        has_saddle: bool 是否存在鞍点
        game_value: float 博弈值
    """
    n_rows, n_cols = U.shape

    # 进攻方: 每行最小值 (最坏情况)
    row_mins = np.min(U, axis=1)
    # 防守方: 每列最大值 (最坏情况)
    col_maxs = np.max(U, axis=0)

    # 极大极小值
    maximin = np.max(row_mins)
    minimax = np.min(col_maxs)

    print(f"\n极大极小值分析:")
    print(f"  进攻方 maximin (max_i min_j U_ij): {maximin:.4f}")
    print(f"  防守方 minimax (min_j max_i U_ij): {minimax:.4f}")

    # 寻找鞍点
    saddle_points = []
    for i in range(n_rows):
        for j in range(n_cols):
            if abs(U[i, j] - maximin) < 1e-10 and abs(U[i, j] - minimax) < 1e-10:
                # 验证鞍点条件
                if (abs(U[i, j] - row_mins[i]) < 1e-10 and
                    abs(U[i, j] - col_maxs[j]) < 1e-10):
                    saddle_points.append((i, j))

    has_saddle = len(saddle_points) > 0
    game_value = maximin if has_saddle else None

    if has_saddle:
        print(f"  博弈值 (鞍点): {game_value:.4f}")
        print(f"  鞍点位置: {[(ATTACK_NAMES[i], DEFENSE_NAMES[j]) for i, j in saddle_points]}")
    else:
        print(f"  未找到纯策略纳什均衡 (鞍点)")

    return saddle_points, has_saddle, game_value


def analyze_minimax(U):
    """极大极小值详细分析

    Returns:
        row_mins: 每个攻击的最坏情况收益
        col_maxs: 每个防御的最大损失
        best_attack_idx: 进攻方最优纯策略
        best_defense_idx: 防守方最优纯策略
    """
    n_rows, n_cols = U.shape

    row_mins = np.min(U, axis=1)
    col_maxs = np.max(U, axis=0)

    best_attack_idx = np.argmax(row_mins)
    best_defense_idx = np.argmin(col_maxs)

    return row_mins, col_maxs, best_attack_idx, best_defense_idx


# ============================================================
# 逐攻击最佳防御分析
# ============================================================
def find_best_defenses_per_attack(U, top_k=5):
    """对每种攻击, 找出防御效果最佳的top_k种防御

    U[i,j] = 进攻方收益, 越小说明防御效果越好

    Returns:
        best_defenses: list of list, best_defenses[i] = [(j, U[i,j]), ...]
    """
    n_attacks = U.shape[0]
    best_defenses = []
    for i in range(n_attacks):
        sorted_idx = np.argsort(U[i, :])  # 升序, 最小的在前
        top_indices = sorted_idx[:top_k]
        best_defenses.append([(j, U[i, j]) for j in top_indices])
    return best_defenses


def analyze_per_attack(U, motions, defense_motions, top_k=5):
    """逐攻击打印最佳防御推荐

    对每种攻击动作, 输出防御效果最好的top_k种防御及其效用值
    """
    n_attacks = len(motions)
    atk_names = [mr['name'] for mr in motions]
    def_names = [d['name'] for d in defense_motions]

    best_defenses = find_best_defenses_per_attack(U, top_k)

    print("\n" + "=" * 70)
    print("逐攻击最佳防御分析")
    print("=" * 70)
    print(f"{'攻击动作':<10} | {'排名':<4} {'最佳防御':<12} {'效用值':<10} {'防御类型':<6}")
    print("-" * 70)

    for i in range(n_attacks):
        print(f"\n  攻击 {i+1:2d}/13: {atk_names[i]}")
        for rank, (j, u_val) in enumerate(best_defenses[i]):
            marker = "★" if rank == 0 else " " if rank == 1 else " "
            def_type = defense_motions[j]['defense_type']
            print(f"    {marker} {rank+1}. {def_names[j]:<10s}  U={u_val:>8.2f}  [{def_type}]")
        print()

    # 汇总表
    print("\n" + "=" * 70)
    print("逐攻击最佳防御汇总表")
    print("=" * 70)
    header = f"{'攻击':<10} | {'最佳防御1':<10} {'(U值)':<10} | {'最佳防御2':<10} {'(U值)':<10} | {'最佳防御3':<10} {'(U值)':<10}"
    print(header)
    print("-" * 90)
    for i in range(n_attacks):
        cols = []
        for rank in range(min(3, len(best_defenses[i]))):
            j, u_val = best_defenses[i][rank]
            cols.append(f"{def_names[j]:<10} {u_val:>7.2f}")
        row = f"{atk_names[i]:<10} | "
        row += " | ".join(cols)
        print(row)

    return best_defenses


# ============================================================
# 可视化
# ============================================================
def create_figures(U, H, C, R, motions, defense_motions, saddle_points,
                   has_saddle, game_value,
                   row_mins, col_maxs, best_attack_idx, best_defense_idx):
    """生成所有可视化图表"""
    os.makedirs(FIG_DIR, exist_ok=True)

    n_attacks = len(motions)
    n_defenses = len(defense_motions)
    atk_names = [mr['name'] for mr in motions]
    def_names = [d['name'] for d in defense_motions]

    # ==========================================
    # 图1: 效用矩阵热力图
    # ==========================================
    fig, ax = plt.subplots(figsize=(16, 10))
    im = ax.imshow(U, cmap='RdYlGn', aspect='auto', vmin=np.min(U), vmax=np.max(U))

    ax.set_xticks(range(n_defenses))
    ax.set_xticklabels(def_names, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(n_attacks))
    ax.set_yticklabels(atk_names, fontsize=10)

    # 标注数值
    for i in range(n_attacks):
        for j in range(n_defenses):
            color = 'white' if U[i, j] < (np.min(U) + np.max(U)) / 2 else 'black'
            ax.text(j, i, f'{U[i, j]:.2f}', ha='center', va='center',
                   fontsize=6, color=color)

    # 标注鞍点
    for si, sj in saddle_points:
        ax.add_patch(plt.Rectangle((sj - 0.5, si - 0.5), 1, 1,
                                    fill=False, edgecolor='red', linewidth=3))
        ax.text(sj, si, '*', ha='center', va='center', fontsize=16,
               color='red', fontweight='bold')

    plt.colorbar(im, ax=ax, label='效用值')
    ax.set_title('13×22 攻防综合效用矩阵 U_ij = H_ij - C_i - R_ij', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, 'fig1_utility_heatmap.png')
    plt.savefig(fig_path)
    plt.close()
    print(f"图1已保存: {fig_path}")

    # ==========================================
    # 图2: 有效伤害矩阵热力图
    # ==========================================
    fig, ax = plt.subplots(figsize=(16, 10))
    im = ax.imshow(H, cmap='YlOrRd', aspect='auto')

    ax.set_xticks(range(n_defenses))
    ax.set_xticklabels(def_names, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(n_attacks))
    ax.set_yticklabels(atk_names, fontsize=10)

    for i in range(n_attacks):
        for j in range(n_defenses):
            color = 'white' if H[i, j] > np.mean(H) else 'black'
            ax.text(j, i, f'{H[i, j]:.2f}', ha='center', va='center',
                   fontsize=6, color=color)

    plt.colorbar(im, ax=ax, label='有效伤害值')
    ax.set_title('有效伤害矩阵 H_ij', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, 'fig2_damage_heatmap.png')
    plt.savefig(fig_path)
    plt.close()
    print(f"图2已保存: {fig_path}")

    # ==========================================
    # 图3: 能量代价柱状图
    # ==========================================
    fig, ax = plt.subplots(figsize=(12, 6))

    colors = [ATTACK_TYPE_COLORS[ATTACK_TYPES[i]] for i in range(n_attacks)]
    bars = ax.bar(range(n_attacks), C, color=colors, edgecolor='black', linewidth=0.5)

    ax.set_xticks(range(n_attacks))
    ax.set_xticklabels(atk_names, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('归一化能量代价', fontsize=12)
    ax.set_title('13种攻击动作的能量代价 C_i', fontsize=14, fontweight='bold')

    # 标注数值
    for bar, val in zip(bars, C):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
               f'{val:.3f}', ha='center', va='bottom', fontsize=8)

    # 图例
    for cat, color in ATTACK_TYPE_COLORS.items():
        ax.bar([], [], color=color, label=cat, edgecolor='black', linewidth=0.5)
    ax.legend(fontsize=10, loc='upper right')

    ax.set_ylim(0, max(C) * 1.15)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, 'fig3_energy_cost.png')
    plt.savefig(fig_path)
    plt.close()
    print(f"图3已保存: {fig_path}")

    # ==========================================
    # 图4: 反击风险热力图
    # ==========================================
    fig, ax = plt.subplots(figsize=(16, 10))
    im = ax.imshow(R, cmap='YlOrRd', aspect='auto')

    ax.set_xticks(range(n_defenses))
    ax.set_xticklabels(def_names, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(n_attacks))
    ax.set_yticklabels(atk_names, fontsize=10)

    for i in range(n_attacks):
        for j in range(n_defenses):
            color = 'white' if R[i, j] > np.mean(R) else 'black'
            ax.text(j, i, f'{R[i, j]:.2f}', ha='center', va='center',
                   fontsize=6, color=color)

    plt.colorbar(im, ax=ax, label='反击风险值')
    ax.set_title('反击风险矩阵 R_ij', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, 'fig4_counter_risk_heatmap.png')
    plt.savefig(fig_path)
    plt.close()
    print(f"图4已保存: {fig_path}")

    # ==========================================
    # 图5: 极大极小值分析
    # ==========================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # 左图: 进攻方视角
    y_pos = np.arange(n_attacks)
    colors_m = ['#F44336' if i == best_attack_idx else '#2196F3' for i in range(n_attacks)]
    ax1.barh(y_pos, row_mins, color=colors_m, edgecolor='black', linewidth=0.5)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(atk_names, fontsize=9)
    ax1.set_xlabel('最坏情况收益 (min_j U_ij)', fontsize=11)
    ax1.set_title('进攻方: 各攻击的最坏情况收益', fontsize=13, fontweight='bold')
    ax1.axvline(x=row_mins[best_attack_idx], color='red', linestyle='--', linewidth=1.5,
               label=f'最优: {atk_names[best_attack_idx]} = {row_mins[best_attack_idx]:.3f}')
    ax1.legend(fontsize=9, loc='lower right')
    ax1.invert_yaxis()
    ax1.grid(True, alpha=0.3, axis='x')

    for i, val in enumerate(row_mins):
        ax1.text(val + 0.002, i, f'{val:.3f}', va='center', fontsize=8)

    # 右图: 防守方视角
    y_pos2 = np.arange(n_defenses)
    colors_d = ['#F44336' if j == best_defense_idx else '#4CAF50' for j in range(n_defenses)]
    ax2.barh(y_pos2, col_maxs, color=colors_d, edgecolor='black', linewidth=0.5)
    ax2.set_yticks(y_pos2)
    ax2.set_yticklabels(def_names, fontsize=8)
    ax2.set_xlabel('最大损失 (max_i U_ij)', fontsize=11)
    ax2.set_title('防守方: 各防御的最大损失', fontsize=13, fontweight='bold')
    ax2.axvline(x=col_maxs[best_defense_idx], color='red', linestyle='--', linewidth=1.5,
               label=f'最优: {def_names[best_defense_idx]} = {col_maxs[best_defense_idx]:.3f}')
    ax2.legend(fontsize=9, loc='lower right')
    ax2.invert_yaxis()
    ax2.grid(True, alpha=0.3, axis='x')

    for j, val in enumerate(col_maxs):
        ax2.text(val + 0.002, j, f'{val:.3f}', va='center', fontsize=8)

    plt.suptitle('极大极小值准则分析', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, 'fig5_minimax_analysis.png')
    plt.savefig(fig_path)
    plt.close()
    print(f"图5已保存: {fig_path}")

    # ==========================================
    # 图6: 纯策略均衡标注图
    # ==========================================
    fig, ax = plt.subplots(figsize=(16, 10))
    im = ax.imshow(U, cmap='RdYlGn', aspect='auto', vmin=np.min(U), vmax=np.max(U))

    ax.set_xticks(range(n_defenses))
    ax.set_xticklabels(def_names, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(n_attacks))
    ax.set_yticklabels(atk_names, fontsize=10)

    # 标注鞍点 (红色方框)
    for si, sj in saddle_points:
        ax.add_patch(plt.Rectangle((sj - 0.5, si - 0.5), 1, 1,
                                    fill=False, edgecolor='red', linewidth=4))
        ax.annotate(f'鞍点\nU={U[si, sj]:.3f}',
                   xy=(sj, si), xytext=(sj + 2, si - 1),
                   fontsize=10, fontweight='bold', color='red',
                   arrowprops=dict(arrowstyle='->', color='red', lw=2),
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.8))

    if not has_saddle:
        # 如果没有鞍点, 标注极大极小解
        ax.add_patch(plt.Rectangle((best_defense_idx - 0.5, best_attack_idx - 0.5),
                                    1, 1, fill=False, edgecolor='blue', linewidth=3,
                                    linestyle='--'))
        ax.annotate(f'极大极小解\n({atk_names[best_attack_idx]}, {def_names[best_defense_idx]})\n'
                   f'U={U[best_attack_idx, best_defense_idx]:.3f}',
                   xy=(best_defense_idx, best_attack_idx),
                   xytext=(best_defense_idx + 3, best_attack_idx - 2),
                   fontsize=10, fontweight='bold', color='blue',
                   arrowprops=dict(arrowstyle='->', color='blue', lw=2),
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))

    plt.colorbar(im, ax=ax, label='效用值')
    ax.set_title('纯策略纳什均衡 (鞍点) 分析', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, 'fig6_saddle_point.png')
    plt.savefig(fig_path)
    plt.close()
    print(f"图6已保存: {fig_path}")

    # ==========================================
    # 图7: TOP10最优攻防组合
    # ==========================================
    fig, ax = plt.subplots(figsize=(12, 8))

    # 展平效用矩阵并排序
    U_flat = U.flatten()
    top10_idx = np.argsort(-U_flat)[:10]

    labels = []
    values = []
    bar_colors = []
    for idx in top10_idx:
        i = idx // n_defenses
        j = idx % n_defenses
        labels.append(f'{atk_names[i]} vs\n{def_names[j]}')
        values.append(U_flat[idx])
        bar_colors.append(ATTACK_TYPE_COLORS[ATTACK_TYPES[i]])

    y_pos = np.arange(10)
    bars = ax.barh(y_pos, values, color=bar_colors, edgecolor='black', linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel('效用值 U_ij', fontsize=12)
    ax.set_title('TOP10 最优攻防组合 (进攻方视角)', fontsize=14, fontweight='bold')
    ax.invert_yaxis()

    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
               f'{val:.3f}', va='center', fontsize=9)

    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, 'fig8_top10_combinations.png')
    plt.savefig(fig_path)
    plt.close()
    print(f"图8已保存: {fig_path}")

    # ==========================================
    # 图9: 防御类型对比图
    # ==========================================
    fig, ax = plt.subplots(figsize=(12, 7))

    defense_type_names = ['格挡', '闪避', '步法', '膝防', '姿态', '反击']
    type_avg_utility = []
    type_counts = []
    type_colors = [DEFENSE_TYPE_COLORS[t] for t in defense_type_names]

    for dt in defense_type_names:
        mask = np.array([DEFENSE_TYPES[j] == dt for j in range(n_defenses)])
        if np.any(mask):
            avg = np.mean(U[:, mask])
            type_avg_utility.append(avg)
            type_counts.append(np.sum(mask))
        else:
            type_avg_utility.append(0)
            type_counts.append(0)

    bars = ax.bar(defense_type_names, type_avg_utility, color=type_colors,
                  edgecolor='black', linewidth=0.5, alpha=0.8)

    # 标注数量
    for bar, count, val in zip(bars, type_counts, type_avg_utility):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
               f'n={count}\n{val:.3f}', ha='center', va='bottom', fontsize=9)

    ax.set_ylabel('平均效用值', fontsize=12)
    ax.set_title('各防御类型的平均效用对比 (进攻方视角)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, 'fig9_defense_type_comparison.png')
    plt.savefig(fig_path)
    plt.close()
    print(f"图9已保存: {fig_path}")

    # ==========================================
    # 图10: 攻击类型对防御类型的平均效用
    # ==========================================
    fig, ax = plt.subplots(figsize=(10, 8))

    attack_type_names = ['拳法', '肘法', '腿法']
    defense_group_names = ['格挡', '闪避', '步法', '膝防', '姿态', '反击']

    matrix = np.zeros((len(attack_type_names), len(defense_group_names)))
    for ai, at in enumerate(attack_type_names):
        for di, dt in enumerate(defense_group_names):
            atk_mask = np.array([ATTACK_TYPES[i] == at for i in range(n_attacks)])
            def_mask = np.array([DEFENSE_TYPES[j] == dt for j in range(n_defenses)])
            if np.any(atk_mask) and np.any(def_mask):
                matrix[ai, di] = np.mean(U[np.ix_(atk_mask, def_mask)])

    im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto')

    ax.set_xticks(range(len(defense_group_names)))
    ax.set_xticklabels(defense_group_names, fontsize=11)
    ax.set_yticks(range(len(attack_type_names)))
    ax.set_yticklabels(attack_type_names, fontsize=11)

    for i in range(len(attack_type_names)):
        for j in range(len(defense_group_names)):
            color = 'white' if matrix[i, j] < np.mean(matrix) else 'black'
            ax.text(j, i, f'{matrix[i, j]:.3f}', ha='center', va='center',
                   fontsize=12, fontweight='bold', color=color)

    plt.colorbar(im, ax=ax, label='平均效用值')
    ax.set_title('攻击类型 × 防御类型 平均效用矩阵', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, 'fig10_type_heatmap.png')
    plt.savefig(fig_path)
    plt.close()
    print(f"图10已保存: {fig_path}")

    # ==========================================
    # 图11: 各攻击的最佳防御 vs 最差防御效用差
    # ==========================================
    fig, ax = plt.subplots(figsize=(12, 7))

    best_vals = np.min(U, axis=1)
    worst_vals = np.max(U, axis=1)
    diff_vals = worst_vals - best_vals

    x = np.arange(n_attacks)
    width = 0.35

    bars1 = ax.bar(x - width/2, best_vals, width, label='最佳防御效用 (min_j U)',
                   color='#4CAF50', edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x + width/2, worst_vals, width, label='最差防御效用 (max_j U)',
                   color='#F44336', edgecolor='black', linewidth=0.5)

    # 标注差值
    for i in range(n_attacks):
        ax.annotate(f'Δ={diff_vals[i]:.0f}', xy=(x[i], max(best_vals[i], worst_vals[i]) + 5),
                   ha='center', fontsize=7, color='#333')

    ax.set_xticks(x)
    ax.set_xticklabels(atk_names, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('效用值', fontsize=11)
    ax.set_title('各攻击动作: 最佳防御 vs 最差防御的效用对比', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, 'fig11_best_vs_worst_defense.png')
    plt.savefig(fig_path)
    plt.close()
    print(f"图11已保存: {fig_path}")

    # ==========================================
    # 图12: 效用矩阵分量堆叠柱状图
    # ==========================================
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # H (伤害)
    H_avg = np.mean(H, axis=1)
    axes[0].bar(range(n_attacks), H_avg, color='#F44336', edgecolor='black', linewidth=0.5)
    axes[0].set_xticks(range(n_attacks))
    axes[0].set_xticklabels(atk_names, rotation=45, ha='right', fontsize=8)
    axes[0].set_ylabel('平均有效伤害', fontsize=11)
    axes[0].set_title('H_ij: 平均有效伤害', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='y')

    # C (能量代价)
    axes[1].bar(range(n_attacks), C, color='#2196F3', edgecolor='black', linewidth=0.5)
    axes[1].set_xticks(range(n_attacks))
    axes[1].set_xticklabels(atk_names, rotation=45, ha='right', fontsize=8)
    axes[1].set_ylabel('归一化能量代价', fontsize=11)
    axes[1].set_title('C_i: 能量代价', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')

    # R (反击风险)
    R_avg = np.mean(R, axis=1)
    axes[2].bar(range(n_attacks), R_avg, color='#FF9800', edgecolor='black', linewidth=0.5)
    axes[2].set_xticks(range(n_attacks))
    axes[2].set_xticklabels(atk_names, rotation=45, ha='right', fontsize=8)
    axes[2].set_ylabel('平均反击风险', fontsize=11)
    axes[2].set_title('R_ij: 平均反击风险', fontsize=12, fontweight='bold')
    axes[2].grid(True, alpha=0.3, axis='y')

    plt.suptitle('效用矩阵分量分析', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, 'fig12_utility_components.png')
    plt.savefig(fig_path)
    plt.close()
    print(f"图12已保存: {fig_path}")

    # ==========================================
    # 图13: 逐攻击最佳防御热力图 (核心图)
    # ==========================================
    best_defenses = find_best_defenses_per_attack(U, top_k=3)

    fig, ax = plt.subplots(figsize=(18, 10))
    im = ax.imshow(U, cmap='RdYlGn', aspect='auto', vmin=np.min(U), vmax=np.max(U))

    ax.set_xticks(range(n_defenses))
    ax.set_xticklabels(def_names, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(n_attacks))
    ax.set_yticklabels(atk_names, fontsize=10)

    # 标注数值
    for i in range(n_attacks):
        for j in range(n_defenses):
            color = 'white' if U[i, j] < (np.min(U) + np.max(U)) / 2 else 'black'
            ax.text(j, i, f'{U[i, j]:.0f}', ha='center', va='center',
                   fontsize=5, color=color)

    # 用红框标注每行TOP3最小值 (最佳防御)
    for i in range(n_attacks):
        for rank, (j, u_val) in enumerate(best_defenses[i]):
            linewidth = 3 if rank == 0 else 2 if rank == 1 else 1.5
            edgecolor = 'red' if rank == 0 else 'orange' if rank == 1 else 'yellow'
            ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                        fill=False, edgecolor=edgecolor, linewidth=linewidth))
            if rank == 0:
                ax.text(j, i, '★', ha='center', va='center', fontsize=12,
                       color='red', fontweight='bold')

    # 图例
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='red', linewidth=3, label='最佳防御 ★'),
        Line2D([0], [0], color='orange', linewidth=2, label='次佳防御'),
        Line2D([0], [0], color='yellow', linewidth=1.5, label='第三佳防御'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10,
             fancybox=True, framealpha=0.9)

    plt.colorbar(im, ax=ax, label='效用值 (越小=防御越好)')
    ax.set_title('13种攻击动作 × 22种防御动作 效用矩阵 (红框=每种攻击的最佳防御)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, 'fig13_per_attack_best_defense.png')
    plt.savefig(fig_path)
    plt.close()
    print(f"图13已保存: {fig_path}")

    # ==========================================
    # 图14: 13子图 — 逐攻击防御效用排名
    # ==========================================
    fig, axes = plt.subplots(4, 4, figsize=(18, 16))
    axes_flat = axes.flatten()

    for i in range(n_attacks):
        ax = axes_flat[i]
        row = U[i, :]
        sorted_idx = np.argsort(row)

        colors = []
        for j in range(n_defenses):
            if j == sorted_idx[0]:
                colors.append('#F44336')  # 最佳: 红色
            elif j == sorted_idx[1]:
                colors.append('#FF9800')  # 次佳: 橙色
            elif j == sorted_idx[2]:
                colors.append('#FFC107')  # 第三: 黄色
            else:
                colors.append('#90CAF9')  # 其他: 浅蓝

        ax.barh(range(n_defenses), row, color=colors, edgecolor='gray', linewidth=0.3, height=0.8)
        ax.set_yticks(range(n_defenses))
        ax.set_yticklabels([def_names[j][:4] for j in range(n_defenses)], fontsize=5)
        ax.set_xlabel('U值', fontsize=7)
        ax.set_title(f'{atk_names[i]}', fontsize=10, fontweight='bold')
        ax.invert_yaxis()
        ax.tick_params(axis='x', labelsize=6)

        # 标注TOP3
        for rank in range(3):
            j = sorted_idx[rank]
            ax.text(row[j] + 1, j, f'{row[j]:.0f}', va='center', fontsize=5,
                   fontweight='bold', color='red' if rank == 0 else 'black')

    # 隐藏第14个子图 (4x4=16, 只有13个攻击)
    for k in range(n_attacks, 16):
        axes_flat[k].set_visible(False)

    plt.suptitle('逐攻击防御效用排名 (红色=最佳防御, 橙色=次佳, 黄色=第三佳)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, 'fig14_per_attack_defense_ranking.png')
    plt.savefig(fig_path)
    plt.close()
    print(f"图14已保存: {fig_path}")

    # ==========================================
    # 图15: 攻击-最佳防御汇总表
    # ==========================================
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.axis('off')

    # 表格数据
    cell_text = []
    for i in range(n_attacks):
        row_data = [atk_names[i]]
        for rank in range(3):
            j, u_val = best_defenses[i][rank]
            row_data.append(f'{def_names[j]}')
            row_data.append(f'{u_val:.1f}')
        cell_text.append(row_data)

    col_labels = ['攻击动作', '最佳防御1', 'U值', '最佳防御2', 'U值', '最佳防御3', 'U值']

    table = ax.table(cellText=cell_text, colLabels=col_labels,
                     cellLoc='center', loc='center',
                     colWidths=[0.12, 0.14, 0.08, 0.14, 0.08, 0.14, 0.08])

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.8)

    # 表头样式
    for j in range(len(col_labels)):
        table[0, j].set_facecolor('#2196F3')
        table[0, j].set_text_props(color='white', fontweight='bold')

    # 最佳防御1列用红色背景
    for i in range(1, n_attacks + 1):
        table[i, 1].set_facecolor('#FFEBEE')
        table[i, 2].set_facecolor('#FFEBEE')

    ax.set_title('13种攻击动作的最佳防御推荐 (按效用值升序排列)', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, 'fig15_attack_defense_summary_table.png')
    plt.savefig(fig_path)
    plt.close()
    print(f"图15已保存: {fig_path}")

    return {
        'fig1': '效用矩阵热力图',
        'fig2': '有效伤害矩阵热力图',
        'fig3': '能量代价柱状图',
        'fig4': '反击风险热力图',
        'fig5': '极大极小值分析',
        'fig6': '纯策略均衡标注图',
        'fig7': 'TOP10最优攻防组合',
        'fig9': '防御类型对比',
        'fig10': '类型交叉热力图',
        'fig11': '策略稳定性分析',
        'fig12': '效用矩阵分量',
        'fig13': '逐攻击最佳防御热力图',
        'fig14': '逐攻击防御效用排名',
        'fig15': '攻击-最佳防御汇总表',
    }


# ============================================================
# 主程序
# ============================================================
def main():
    print("=" * 70)
    print("CUMCM 2026 Problem B - 攻防博弈模型")
    print("=" * 70)

    # 1. 加载问题一结果
    print("\n[1/6] 加载问题一仿真结果...")
    motions, raw_data = load_results()
    print(f"  已加载 {len(motions)} 种攻击动作")

    # 2. 定义22种防御动作
    print("\n[2/6] 定义22种防御动作...")
    defense_motions = get_defense_motions()
    print(f"  已定义 {len(defense_motions)} 种防御动作")
    for d in defense_motions:
        print(f"    {d['name']:8s} [{d['defense_type']}]")

    # 3. 计算效用矩阵
    print("\n[3/6] 计算13×22攻防综合效用矩阵...")
    U, H, C, R, H_norm = compute_utility_matrix(motions, defense_motions, raw_data)
    print(f"  效用矩阵形状: {U.shape}")
    print(f"  效用值范围: [{U.min():.4f}, {U.max():.4f}]")
    print(f"  平均效用: {U.mean():.4f}")

    # 4. 纯策略纳什均衡
    print("\n[4/6] 求解纯策略纳什均衡...")
    saddle_points, has_saddle, game_value = find_saddle_points(U)

    # 5. 极大极小值分析
    print("\n  极大极小值详细分析:")
    row_mins, col_maxs, best_attack_idx, best_defense_idx = analyze_minimax(U)
    print(f"  进攻方最优纯策略: {ATTACK_NAMES[best_attack_idx]} (min_j = {row_mins[best_attack_idx]:.4f})")
    print(f"  防守方最优纯策略: {DEFENSE_NAMES[best_defense_idx]} (max_i = {col_maxs[best_defense_idx]:.4f})")

    # 6. 逐攻击最佳防御分析 (核心输出)
    print("\n[5/6] 逐攻击最佳防御分析...")
    best_defenses = analyze_per_attack(U, motions, defense_motions, top_k=5)

    # 7. 生成可视化
    print("\n[6/6] 生成可视化图表...")
    fig_dict = create_figures(U, H, C, R, motions, defense_motions,
                               saddle_points, has_saddle, game_value,
                               row_mins, col_maxs, best_attack_idx, best_defense_idx)

    # 总结
    print("\n" + "=" * 70)
    print("分析完成!")
    print("=" * 70)
    print(f"\n关键结果:")
    print(f"  效用矩阵: 13×22 (攻击×防御)")
    if has_saddle:
        print(f"  纯策略纳什均衡: {[(ATTACK_NAMES[i], DEFENSE_NAMES[j]) for i, j in saddle_points]}")
    else:
        print(f"  无纯策略纳什均衡")
    print(f"  进攻方最优纯策略: {ATTACK_NAMES[best_attack_idx]}")
    print(f"  防守方最优纯策略: {DEFENSE_NAMES[best_defense_idx]}")

    # 逐攻击最佳防御汇总
    print(f"\n  --- 逐攻击最佳防御汇总 ---")
    for i in range(len(ATTACK_NAMES)):
        top3 = [DEFENSE_NAMES[j] for j, _ in best_defenses[i][:3]]
        print(f"  {ATTACK_NAMES[i]:<8s} → {top3[0]} / {top3[1]} / {top3[2]}")

    print(f"\n图表已保存至: {FIG_DIR}/")

    # 保存结果摘要
    summary_path = os.path.join(FIG_DIR, 'game_theory_summary.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("CUMCM 2026 Problem B - 攻防博弈模型 结果摘要\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"效用矩阵形状: {U.shape}\n")
        f.write(f"效用值范围: [{U.min():.4f}, {U.max():.4f}]\n\n")

        if has_saddle:
            f.write("纯策略纳什均衡:\n")
            for i, j in saddle_points:
                f.write(f"  攻击: {ATTACK_NAMES[i]}, 防御: {DEFENSE_NAMES[j]}, U={U[i,j]:.4f}\n")
        else:
            f.write("无纯策略纳什均衡\n")

        f.write(f"\n极大极小解:\n")
        f.write(f"  进攻方: {ATTACK_NAMES[best_attack_idx]}, min_j U = {row_mins[best_attack_idx]:.4f}\n")
        f.write(f"  防守方: {DEFENSE_NAMES[best_defense_idx]}, max_i U = {col_maxs[best_defense_idx]:.4f}\n")

        f.write(f"\n逐攻击最佳防御:\n")
        for i in range(len(ATTACK_NAMES)):
            top3 = [(DEFENSE_NAMES[j], u_val) for j, u_val in best_defenses[i][:3]]
            f.write(f"  {ATTACK_NAMES[i]:<8s}: {top3[0][0]} (U={top3[0][1]:.2f}), "
                    f"{top3[1][0]} (U={top3[1][1]:.2f}), {top3[2][0]} (U={top3[2][1]:.2f})\n")

    print(f"结果摘要已保存至: {summary_path}")


if __name__ == '__main__':
    main()
