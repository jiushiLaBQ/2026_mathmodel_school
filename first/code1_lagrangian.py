# -*- coding: utf-8 -*-
"""
环境要求:
    Python >= 3.10  (推荐 Anaconda 或 Miniconda)
    numpy >= 2.0
    scipy >= 1.14
    matplotlib >= 3.9
    pandas >= 2.2

    安装: pip install -r requirements.txt
    运行: python code1_lagrangian.py

代码1: 拉格朗日动力学建模与攻击力指标计算
人形机器人PM01商业版 (23 DOF, ~40.1kg, 无颈部) 13种攻击动作的动力学仿真

核心方法: 预定义轨迹分析法
- 定义关节角度随时间变化的函数(五次多项式插值)
- 在每个时间步计算正运动学、质量矩阵、科里奥利力、重力项
- 计算末端执行器速度、等效质量、有效动能、有效动量
- 输出: results.npz (供code2使用)
"""

import numpy as np
import pandas as pd
import os
from scipy.spatial import ConvexHull
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

# ============================================================
# 全局参数
# ============================================================
G = 9.81  # 重力加速度 m/s^2
DT = 0.001  # 时间步长 s
N_JOINTS = 23
N_LINKS = 28
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')

# 连杆索引映射
LINK_NAMES = [
    'LINK_BASE', 'LINK_HIP_PITCH_L', 'LINK_HIP_ROLL_L', 'LINK_HIP_YAW_L',
    'LINK_KNEE_PITCH_L', 'LINK_ANKLE_PITCH_L', 'LINK_ANKLE_ROLL_L', 'LINK_FOOT_L',
    'LINK_HIP_PITCH_R', 'LINK_HIP_ROLL_R', 'LINK_HIP_YAW_R',
    'LINK_KNEE_PITCH_R', 'LINK_ANKLE_PITCH_R', 'LINK_ANKLE_ROLL_R', 'LINK_FOOT_R',
    'LINK_TORSO_YAW',
    'LINK_SHOULDER_PITCH_L', 'LINK_SHOULDER_ROLL_L', 'LINK_SHOULDER_YAW_L',
    'LINK_ELBOW_PITCH_L', 'LINK_ELBOW_YAW_L', 'LINK_ELBOW_END_L',
    'LINK_SHOULDER_PITCH_R', 'LINK_SHOULDER_ROLL_R', 'LINK_SHOULDER_YAW_R',
    'LINK_ELBOW_PITCH_R', 'LINK_ELBOW_YAW_R', 'LINK_ELBOW_END_R',
]

# 关节索引映射 (便于阅读)
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


# ============================================================
# 数据结构
# ============================================================
@dataclass
class Link:
    name: str
    parent: int
    joint_idx: int
    mass: float
    com_offset: np.ndarray
    inertia_diag: np.ndarray
    joint_axis: np.ndarray


@dataclass
class Robot:
    links: List[Link]
    n_joints: int = N_JOINTS
    n_links: int = N_LINKS
    joint_names: List[str] = field(default_factory=list)
    joint_limits_low: np.ndarray = field(default_factory=lambda: np.zeros(N_JOINTS))
    joint_limits_high: np.ndarray = field(default_factory=lambda: np.zeros(N_JOINTS))
    motor_type: List[str] = field(default_factory=list)
    motor_inertia: np.ndarray = field(default_factory=lambda: np.zeros(N_JOINTS))
    motor_torque_limit: np.ndarray = field(default_factory=lambda: np.zeros(N_JOINTS))
    T_fixed: np.ndarray = field(default_factory=lambda: np.zeros((N_LINKS, 4, 4)))


# ============================================================
# CSV数据读取与机器人构建
# ============================================================
def parse_tuple(s):
    """解析 '(x, y, z)' 格式的字符串为numpy数组"""
    s = s.strip().strip('()"')
    parts = [float(x.strip()) for x in s.split(',')]
    return np.array(parts)


def build_robot():
    """从CSV文件构建机器人模型"""
    links = [None] * N_LINKS

    # --- 读取连杆质量与质心 ---
    mass_df = pd.read_csv(os.path.join(DATA_DIR, '各连杆质量与质心.csv'))
    mass_dict = {}
    com_dict = {}
    for _, row in mass_df.iterrows():
        name = row.iloc[0].replace('\\', '')
        mass = float(row.iloc[1])
        com = parse_tuple(str(row.iloc[2]))
        mass_dict[name] = mass
        com_dict[name] = com

    # --- 读取转动惯量 ---
    inertia_df = pd.read_csv(os.path.join(DATA_DIR, '转动惯量矩阵（相对于质心，diaginertia = [Ixx, Iyy, Izz]）.csv'))
    inertia_dict = {}
    for _, row in inertia_df.iterrows():
        name = row.iloc[0].replace('\\', '')
        inertia_dict[name] = np.array([float(row.iloc[1]), float(row.iloc[2]), float(row.iloc[3])])

    # --- 读取关节参数 ---
    joint_df = pd.read_csv(os.path.join(DATA_DIR, '关节参数（全部为旋转关节 RevoluteJoint）.csv'))
    joint_names = []
    joint_axes = []
    joint_limits_low = np.zeros(N_JOINTS)
    joint_limits_high = np.zeros(N_JOINTS)
    motor_types = []
    for _, row in joint_df.iterrows():
        name = str(row.iloc[1]).replace('\\', '')
        if name == 'HEAD_YAW':
            continue  # PM01商业版无颈部自由度
        joint_names.append(name)
        axis = parse_tuple(str(row.iloc[2]))
        joint_axes.append(axis / np.linalg.norm(axis))
        # 解析角度范围
        range_str = str(row.iloc[3]).replace('\\', '').strip('[] ')
        parts = range_str.split(',')
        joint_limits_low[len(joint_names)-1] = float(parts[0].strip())
        joint_limits_high[len(joint_names)-1] = float(parts[1].strip())
        motor_types.append(str(row.iloc[5]).strip())

    # --- 读取电机参数 ---
    motor_df = pd.read_csv(os.path.join(DATA_DIR, '电机参数.csv'))
    gear_ratio = 25
    q90_rotor = 0.0453
    q25_rotor = 0.0067
    q90_torque = 145.0
    q25_torque = 50.0

    motor_inertia = np.zeros(N_JOINTS)
    motor_torque = np.zeros(N_JOINTS)
    for i in range(N_JOINTS):
        if motor_types[i] == 'Q90':
            motor_inertia[i] = gear_ratio**2 * q90_rotor
            motor_torque[i] = q90_torque
        else:
            motor_inertia[i] = gear_ratio**2 * q25_rotor
            motor_torque[i] = q25_torque

    # --- 构建运动学树 ---
    # (link_name, parent_link_idx, joint_idx, joint_axis_key)
    tree_def = [
        ('LINK_BASE', -1, -1, None),
        ('LINK_HIP_PITCH_L', 0, 0, 'HIP_PITCH_L'),
        ('LINK_HIP_ROLL_L', 1, 1, 'HIP_ROLL_L'),
        ('LINK_HIP_YAW_L', 2, 2, 'HIP_YAW_L'),
        ('LINK_KNEE_PITCH_L', 3, 3, 'KNEE_PITCH_L'),
        ('LINK_ANKLE_PITCH_L', 4, 4, 'ANKLE_PITCH_L'),
        ('LINK_ANKLE_ROLL_L', 5, 5, 'ANKLE_ROLL_L'),
        ('LINK_FOOT_L', 6, -1, None),
        ('LINK_HIP_PITCH_R', 0, 6, 'HIP_PITCH_R'),
        ('LINK_HIP_ROLL_R', 8, 7, 'HIP_ROLL_R'),
        ('LINK_HIP_YAW_R', 9, 8, 'HIP_YAW_R'),
        ('LINK_KNEE_PITCH_R', 10, 9, 'KNEE_PITCH_R'),
        ('LINK_ANKLE_PITCH_R', 11, 10, 'ANKLE_PITCH_R'),
        ('LINK_ANKLE_ROLL_R', 12, 11, 'ANKLE_ROLL_R'),
        ('LINK_FOOT_R', 13, -1, None),
        ('LINK_TORSO_YAW', 0, 12, 'WAIST_YAW'),
        ('LINK_SHOULDER_PITCH_L', 15, 13, 'SHOULDER_PITCH_L'),
        ('LINK_SHOULDER_ROLL_L', 16, 14, 'SHOULDER_ROLL_L'),
        ('LINK_SHOULDER_YAW_L', 17, 15, 'SHOULDER_YAW_L'),
        ('LINK_ELBOW_PITCH_L', 18, 16, 'ELBOW_PITCH_L'),
        ('LINK_ELBOW_YAW_L', 19, 17, 'ELBOW_YAW_L'),
        ('LINK_ELBOW_END_L', 20, -1, None),
        ('LINK_SHOULDER_PITCH_R', 15, 18, 'SHOULDER_PITCH_R'),
        ('LINK_SHOULDER_ROLL_R', 22, 19, 'SHOULDER_ROLL_R'),
        ('LINK_SHOULDER_YAW_R', 23, 20, 'SHOULDER_YAW_R'),
        ('LINK_ELBOW_PITCH_R', 24, 21, 'ELBOW_PITCH_R'),
        ('LINK_ELBOW_YAW_R', 25, 22, 'ELBOW_YAW_R'),
        ('LINK_ELBOW_END_R', 26, -1, None),
    ]

    for i, (name, parent, jidx, jname) in enumerate(tree_def):
        mass = mass_dict.get(name, 0.001)
        com = com_dict.get(name, np.zeros(3))
        iner = inertia_dict.get(name, np.array([1e-6, 1e-6, 1e-6]))
        axis = joint_axes[jidx] if jidx >= 0 else np.zeros(3)
        links[i] = Link(name=name, parent=parent, joint_idx=jidx,
                        mass=mass, com_offset=com, inertia_diag=iner, joint_axis=axis)

    robot = Robot(links=links)
    robot.joint_names = joint_names
    robot.joint_limits_low = joint_limits_low
    robot.joint_limits_high = joint_limits_high
    robot.motor_type = motor_types
    robot.motor_inertia = motor_inertia
    robot.motor_torque_limit = motor_torque

    # --- 构建固定变换矩阵 T_fixed ---
    # 由连杆长度推导各关节间的固定偏移
    _build_fixed_transforms(robot)

    return robot


def _build_fixed_transforms(robot):
    """根据连杆长度和人体工学构建固定变换矩阵"""
    T_fixed = np.zeros((N_LINKS, 4, 4))

    # 基础连杆: 无变换
    T_fixed[0] = np.eye(4)

    # --- 左腿链 ---
    # HIP_PITCH_L (link 1): 从BASE出发，轻微偏移
    T_fixed[1] = _make_transform(0, 0, 0)
    # HIP_ROLL_L (link 2): 横向偏移(髋宽的一半)
    T_fixed[2] = _make_transform(0, 0.045, 0)
    # HIP_YAW_L (link 3): 小偏移
    T_fixed[3] = _make_transform(0, 0, 0)
    # KNEE_PITCH_L (link 4): 大腿长度 0.27m (z方向向下)
    T_fixed[4] = _make_transform(0, 0, -0.27)
    # ANKLE_PITCH_L (link 5): 小腿长度 0.38m
    T_fixed[5] = _make_transform(0, 0, -0.38)
    # ANKLE_ROLL_L (link 6): 小偏移
    T_fixed[6] = _make_transform(0, 0, 0)
    # FOOT_L (link 7): 脚掌长度 0.04m
    T_fixed[7] = _make_transform(0.08, 0, -0.04)

    # --- 右腿链 (对称, y取反) ---
    T_fixed[8] = _make_transform(0, 0, 0)
    T_fixed[9] = _make_transform(0, -0.045, 0)
    T_fixed[10] = _make_transform(0, 0, 0)
    T_fixed[11] = _make_transform(0, 0, -0.27)
    T_fixed[12] = _make_transform(0, 0, -0.38)
    T_fixed[13] = _make_transform(0, 0, 0)
    T_fixed[14] = _make_transform(0.08, 0, -0.04)

    # --- 躯干 ---
    T_fixed[15] = _make_transform(0, 0, 0.18)

    # --- 左臂链 ---
    T_fixed[16] = _make_transform(0, 0.18, 0)    # 肩宽偏移
    T_fixed[17] = _make_transform(0, 0, 0)
    T_fixed[18] = _make_transform(0, 0, 0)
    T_fixed[19] = _make_transform(0, 0, -0.17)   # 上臂长
    T_fixed[20] = _make_transform(0, 0, 0)
    T_fixed[21] = _make_transform(0, 0, -0.16)   # 前臂长(末端)

    # --- 右臂链 (对称) ---
    T_fixed[22] = _make_transform(0, -0.18, 0)
    T_fixed[23] = _make_transform(0, 0, 0)
    T_fixed[24] = _make_transform(0, 0, 0)
    T_fixed[25] = _make_transform(0, 0, -0.17)
    T_fixed[26] = _make_transform(0, 0, 0)
    T_fixed[27] = _make_transform(0, 0, -0.16)

    robot.T_fixed = T_fixed


def _make_transform(dx, dy, dz):
    """创建平移齐次变换矩阵"""
    T = np.eye(4)
    T[:3, 3] = [dx, dy, dz]
    return T


# ============================================================
# 运动学核心函数
# ============================================================
def axis_angle_to_rotation(axis, theta):
    """Rodrigues旋转公式: 绕任意轴旋转"""
    if abs(theta) < 1e-12:
        return np.eye(3)
    k = axis / (np.linalg.norm(axis) + 1e-12)
    K = np.array([
        [0, -k[2], k[1]],
        [k[2], 0, -k[0]],
        [-k[1], k[0], 0]
    ])
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)


def forward_kinematics(robot, q):
    """正运动学: 计算所有连杆的世界坐标系变换和质心位置

    Args:
        robot: 机器人模型
        q: 关节角度 (23,)

    Returns:
        T_world: 每个连杆的世界变换 (28, 4, 4)
        p_com: 每个连杆质心的世界坐标 (28, 3)
    """
    T_world = np.zeros((N_LINKS, 4, 4))
    p_com = np.zeros((N_LINKS, 3))

    T_world[0] = np.eye(4)

    for i in range(1, N_LINKS):
        parent = robot.links[i].parent
        j = robot.links[i].joint_idx

        if j >= 0:
            R_joint = axis_angle_to_rotation(robot.links[i].joint_axis, q[j])
            T_joint = np.eye(4)
            T_joint[:3, :3] = R_joint
        else:
            T_joint = np.eye(4)

        T_local = robot.T_fixed[i] @ T_joint
        T_world[i] = T_world[parent] @ T_local

        # 计算质心位置
        com_local = np.append(robot.links[i].com_offset, 1.0)
        p_com[i] = (T_world[i] @ com_local)[:3]

    return T_world, p_com


def compute_body_jacobians(robot, q, T_world, p_com):
    """计算每个连杆的体雅可比矩阵 (世界坐标系)

    Returns:
        J_v: 线速度雅可比列表, 每个元素 (3, 23)
        J_omega: 角速度雅可比列表, 每个元素 (3, 23)
    """
    J_v = [np.zeros((3, N_JOINTS)) for _ in range(N_LINKS)]
    J_omega = [np.zeros((3, N_JOINTS)) for _ in range(N_LINKS)]

    for i in range(N_LINKS):
        current = i
        while current != 0:
            j = robot.links[current].joint_idx
            if j >= 0:
                R_world = T_world[current][:3, :3]
                z_j = R_world @ robot.links[current].joint_axis
                p_joint = T_world[current][:3, 3]
                J_v[i][:, j] = np.cross(z_j, p_com[i] - p_joint)
                J_omega[i][:, j] = z_j
            current = robot.links[current].parent

    return J_v, J_omega


def compute_mass_matrix(robot, q, T_world, J_v, J_omega):
    """计算质量矩阵 M(q) (23x23)

    M_ij = Σ_k [m_k·J_v_k^T·J_v_k + J_ω_k^T·R_k·I_k·R_k^T·J_ω_k] + motor_inertia·δ_ij
    """
    M = np.zeros((N_JOINTS, N_JOINTS))

    for k in range(N_LINKS):
        m_k = robot.links[k].mass
        R_k = T_world[k][:3, :3]
        I_k = np.diag(robot.links[k].inertia_diag)
        I_world_k = R_k @ I_k @ R_k.T

        # 平动贡献
        M += m_k * (J_v[k].T @ J_v[k])
        # 转动贡献
        M += J_omega[k].T @ I_world_k @ J_omega[k]

    # 添加电机等效惯量 (对角线)
    for j in range(N_JOINTS):
        M[j, j] += robot.motor_inertia[j]

    # 强制对称
    M = 0.5 * (M + M.T)
    return M


def compute_gravity(robot, J_v):
    """计算重力矩向量 g(q) (23,)

    g_i = -Σ_k m_k · g^T · J_v_k[:,i]
    """
    g_world = np.array([0.0, 0.0, -G])
    g = np.zeros(N_JOINTS)

    for i in range(N_JOINTS):
        for k in range(N_LINKS):
            g[i] -= robot.links[k].mass * g_world @ J_v[k][:, i]

    return g


def compute_coriolis_times_qdot(robot, q, qdot, delta=1e-5):
    """计算科里奥利力与离心力 C(q,qdot)·qdot

    使用克里斯托弗尔符号: c_ijk = 0.5(∂M_ij/∂q_k + ∂M_ik/∂q_j - ∂M_jk/∂q_i)
    """
    # 先计算各关节扰动下的质量矩阵
    dM_dq = []
    for k in range(N_JOINTS):
        q_plus = q.copy()
        q_plus[k] += delta
        q_minus = q.copy()
        q_minus[k] -= delta

        T_wp, p_cp = forward_kinematics(robot, q_plus)
        J_vp, J_op = compute_body_jacobians(robot, q_plus, T_wp, p_cp)
        M_plus = compute_mass_matrix(robot, q_plus, T_wp, J_vp, J_op)

        T_wm, p_cm = forward_kinematics(robot, q_minus)
        J_vm, J_om = compute_body_jacobians(robot, q_minus, T_wm, p_cm)
        M_minus = compute_mass_matrix(robot, q_minus, T_wm, J_vm, J_om)

        dM_dq.append((M_plus - M_minus) / (2 * delta))

    # 计算 C·qdot
    C_qdot = np.zeros(N_JOINTS)
    for i in range(N_JOINTS):
        for j in range(N_JOINTS):
            c_ij = 0.0
            for k in range(N_JOINTS):
                c_ijk = 0.5 * (dM_dq[k][i, j] + dM_dq[j][i, k] - dM_dq[i][j, k])
                c_ij += c_ijk * qdot[k]
            C_qdot[i] += c_ij * qdot[j]

    return C_qdot


# ============================================================
# 攻击力指标计算
# ============================================================
def compute_attack_metrics(robot, q, qdot, T_world, p_com, J_v, J_omega, M):
    """计算单个时间步的攻击力指标

    Returns:
        v_ee: 末端执行器速度 (3,)
        v_speed: 速度大小
        m_e: 等效质量
        E_k: 有效动能
        p_momentum: 有效动量
    """
    # 末端执行器速度
    v_ee = J_v @ qdot
    v_speed = np.linalg.norm(v_ee)

    if v_speed < 1e-10:
        return v_ee, 0.0, 0.0, 0.0, 0.0

    # 打击方向单位向量
    u = v_ee / v_speed

    # 笛卡尔惯量 (操作空间惯量)
    M_inv = np.linalg.inv(M)
    Lambda_inv = J_v @ M_inv @ J_v.T  # (3, 3)

    # 等效质量: m_e = 1 / (u^T · Lambda_inv · u)
    denom = u @ Lambda_inv @ u
    if abs(denom) < 1e-12:
        m_e = 0.0
    else:
        m_e = 1.0 / denom

    E_k = 0.5 * m_e * v_speed**2
    p_momentum = m_e * v_speed

    return v_ee, v_speed, m_e, E_k, p_momentum


# ============================================================
# 13种攻击动作定义
# ============================================================
def quintic(tau):
    """五次多项式插值 s(tau) = 10τ³ - 15τ⁴ + 6τ⁵, tau ∈ [0, 1]"""
    tau = np.clip(tau, 0.0, 1.0)
    return 10 * tau**3 - 15 * tau**4 + 6 * tau**5


def get_neutral_pose():
    """中立站立姿态"""
    q = np.zeros(N_JOINTS)
    # 腿部微屈
    q[IDX['KNEE_PITCH_L']] = 0.1
    q[IDX['KNEE_PITCH_R']] = 0.1
    q[IDX['ANKLE_PITCH_L']] = -0.05
    q[IDX['ANKLE_PITCH_R']] = -0.05
    # 手臂自然下垂微前
    q[IDX['SHOULDER_PITCH_L']] = -0.3
    q[IDX['SHOULDER_PITCH_R']] = -0.3
    q[IDX['SHOULDER_ROLL_L']] = 0.1
    q[IDX['SHOULDER_ROLL_R']] = -0.1
    q[IDX['ELBOW_PITCH_L']] = -0.5
    q[IDX['ELBOW_PITCH_R']] = -0.5
    return q


def get_attack_poses():
    """定义13种攻击动作的关节偏移量和时间参数

    Returns:
        list of dict, 每个动作包含:
            'name': 动作名称
            'delta_q': 关节偏移量 (24,)
            'T_attack': 攻击时间
            'T_return': 返回时间
            'ee_link': 末端执行器连杆索引
    """
    q_neutral = get_neutral_pose()
    motions = []

    def make_delta(**kwargs):
        d = np.zeros(N_JOINTS)
        for k, v in kwargs.items():
            d[IDX[k]] = v
        return d

    # 1. 左直拳
    motions.append({
        'name': '左直拳', 'name_en': 'Left Straight Punch',
        'delta_q': make_delta(WAIST_YAW=0.3, SHOULDER_PITCH_L=1.2, ELBOW_PITCH_L=1.5,
                              SHOULDER_ROLL_L=-0.2, SHOULDER_YAW_L=0.1,
                              SHOULDER_PITCH_R=-0.3),
        'T_attack': 0.3, 'T_return': 0.2, 'ee_link': 21  # ELBOW_END_L
    })

    # 2. 右直拳
    motions.append({
        'name': '右直拳', 'name_en': 'Right Straight Punch',
        'delta_q': make_delta(WAIST_YAW=-0.3, SHOULDER_PITCH_R=1.2, ELBOW_PITCH_R=1.5,
                              SHOULDER_ROLL_R=-0.2, SHOULDER_YAW_R=-0.1,
                              SHOULDER_PITCH_L=-0.3),
        'T_attack': 0.3, 'T_return': 0.2, 'ee_link': 27  # ELBOW_END_R
    })

    # 3. 左摆拳
    motions.append({
        'name': '左摆拳', 'name_en': 'Left Hook',
        'delta_q': make_delta(WAIST_YAW=0.5, SHOULDER_PITCH_L=0.6, SHOULDER_ROLL_L=1.0,
                              ELBOW_PITCH_L=1.2, SHOULDER_YAW_L=0.4),
        'T_attack': 0.35, 'T_return': 0.2, 'ee_link': 21
    })

    # 4. 右摆拳
    motions.append({
        'name': '右摆拳', 'name_en': 'Right Hook',
        'delta_q': make_delta(WAIST_YAW=-0.5, SHOULDER_PITCH_R=0.6, SHOULDER_ROLL_R=-1.0,
                              ELBOW_PITCH_R=1.2, SHOULDER_YAW_R=-0.4),
        'T_attack': 0.35, 'T_return': 0.2, 'ee_link': 27
    })

    # 5. 左上勾拳
    motions.append({
        'name': '左上勾拳', 'name_en': 'Left Uppercut',
        'delta_q': make_delta(SHOULDER_PITCH_L=0.8, SHOULDER_ROLL_L=0.3, ELBOW_PITCH_L=1.0,
                              KNEE_PITCH_L=-0.2, KNEE_PITCH_R=-0.2,
                              WAIST_YAW=0.2),
        'T_attack': 0.3, 'T_return': 0.2, 'ee_link': 21
    })

    # 6. 右上勾拳
    motions.append({
        'name': '右上勾拳', 'name_en': 'Right Uppercut',
        'delta_q': make_delta(SHOULDER_PITCH_R=0.8, SHOULDER_ROLL_R=-0.3, ELBOW_PITCH_R=1.0,
                              KNEE_PITCH_R=-0.2, KNEE_PITCH_L=-0.2,
                              WAIST_YAW=-0.2),
        'T_attack': 0.3, 'T_return': 0.2, 'ee_link': 27
    })

    # 7. 左掌击
    motions.append({
        'name': '左掌击', 'name_en': 'Left Palm Strike',
        'delta_q': make_delta(SHOULDER_PITCH_L=1.0, ELBOW_PITCH_L=0.3, SHOULDER_ROLL_L=-0.1,
                              WAIST_YAW=0.2),
        'T_attack': 0.25, 'T_return': 0.15, 'ee_link': 21
    })

    # 8. 右肘击
    motions.append({
        'name': '右肘击', 'name_en': 'Right Elbow Strike',
        'delta_q': make_delta(WAIST_YAW=-0.6, SHOULDER_PITCH_R=0.4, ELBOW_PITCH_R=1.5,
                              SHOULDER_ROLL_R=0.3),
        'T_attack': 0.2, 'T_return': 0.15, 'ee_link': 25  # ELBOW_PITCH_R
    })

    # 9. 左膝击
    motions.append({
        'name': '左膝击', 'name_en': 'Left Knee Strike',
        'delta_q': make_delta(HIP_PITCH_L=1.5, KNEE_PITCH_L=-1.5, HIP_ROLL_L=0.1,
                              ANKLE_PITCH_L=0.3, WAIST_YAW=0.1,
                              SHOULDER_PITCH_L=0.3, SHOULDER_PITCH_R=0.3),
        'T_attack': 0.35, 'T_return': 0.2, 'ee_link': 4  # KNEE_PITCH_L
    })

    # 10. 右前蹬
    motions.append({
        'name': '右前蹬', 'name_en': 'Right Front Push Kick',
        'delta_q': make_delta(HIP_PITCH_R=1.0, KNEE_PITCH_R=-1.0, ANKLE_PITCH_R=-0.3,
                              WAIST_YAW=-0.1, SHOULDER_PITCH_L=0.2, SHOULDER_PITCH_R=0.2),
        'T_attack': 0.4, 'T_return': 0.2, 'ee_link': 14  # FOOT_R
    })

    # 11. 右侧踢
    motions.append({
        'name': '右侧踢', 'name_en': 'Right Side Kick',
        'delta_q': make_delta(HIP_PITCH_R=0.5, HIP_ROLL_R=-1.0, KNEE_PITCH_R=-1.2,
                              HIP_YAW_R=-0.3, ANKLE_ROLL_R=0.3,
                              WAIST_YAW=-0.2, SHOULDER_ROLL_L=0.3),
        'T_attack': 0.4, 'T_return': 0.2, 'ee_link': 14
    })

    # 12. 右回旋踢
    motions.append({
        'name': '右回旋踢', 'name_en': 'Right Spinning Kick',
        'delta_q': make_delta(WAIST_YAW=-2.0, HIP_PITCH_R=0.8, HIP_ROLL_R=-0.5,
                              KNEE_PITCH_R=-0.8, HIP_YAW_R=-0.5,
                              SHOULDER_PITCH_L=0.3, SHOULDER_PITCH_R=0.3),
        'T_attack': 0.5, 'T_return': 0.3, 'ee_link': 14
    })

    # 13. 右后踢
    motions.append({
        'name': '右后踢', 'name_en': 'Right Back Kick',
        'delta_q': make_delta(HIP_PITCH_R=-1.0, KNEE_PITCH_R=0.3, ANKLE_PITCH_R=0.2,
                              WAIST_YAW=-0.1, SHOULDER_PITCH_L=0.2, SHOULDER_PITCH_R=0.2),
        'T_attack': 0.4, 'T_return': 0.2, 'ee_link': 14
    })

    return motions


def motion_trajectory(t, motion, q_neutral):
    """生成某个动作在时刻t的关节角度

    前向阶段: t ∈ [0, T_attack], s = quintic(t/T_attack)
    返回阶段: t ∈ [T_attack, T_attack+T_return], s = 1 - quintic((t-T_attack)/T_return)
    静止阶段: t > T_attack + T_return, s = 0
    """
    T_a = motion['T_attack']
    T_r = motion['T_return']
    delta_q = motion['delta_q']

    if t <= T_a:
        s = quintic(t / T_a) if T_a > 0 else 1.0
    elif t <= T_a + T_r:
        s = 1.0 - quintic((t - T_a) / T_r) if T_r > 0 else 0.0
    else:
        s = 0.0

    return q_neutral + delta_q * s


def motion_trajectory_deriv(t, motion, q_neutral):
    """生成某个动作在时刻t的关节角度一阶导数(速度)"""
    T_a = motion['T_attack']
    T_r = motion['T_return']
    delta_q = motion['delta_q']

    if t <= T_a and T_a > 0:
        tau = t / T_a
        ds_dt = (30 * tau**2 - 60 * tau**3 + 30 * tau**4) / T_a
    elif t <= T_a + T_r and T_r > 0:
        tau = (t - T_a) / T_r
        ds_dt = -(30 * tau**2 - 60 * tau**3 + 30 * tau**4) / T_r
    else:
        ds_dt = 0.0

    return delta_q * ds_dt


def motion_trajectory_deriv2(t, motion, q_neutral):
    """生成某个动作在时刻t的关节角度二阶导数(加速度)"""
    T_a = motion['T_attack']
    T_r = motion['T_return']
    delta_q = motion['delta_q']

    if t <= T_a and T_a > 0:
        tau = t / T_a
        d2s_dt2 = (60 * tau - 180 * tau**2 + 120 * tau**3) / T_a**2
    elif t <= T_a + T_r and T_r > 0:
        tau = (t - T_a) / T_r
        d2s_dt2 = -(60 * tau - 180 * tau**2 + 120 * tau**3) / T_r**2
    else:
        d2s_dt2 = 0.0

    return delta_q * d2s_dt2


# ============================================================
# 主仿真循环
# ============================================================
def run_simulation():
    """运行13个动作的拉格朗日动力学仿真"""
    print("=" * 60)
    print("拉格朗日动力学仿真 - PM01人形机器人")
    print("=" * 60)

    # 构建机器人
    robot = build_robot()
    print(f"机器人构建完成: {robot.n_links}个连杆, {robot.n_joints}个关节")

    # 验证站立姿态
    q_standing = get_neutral_pose()
    T_world, p_com = forward_kinematics(robot, q_standing)
    total_mass = sum(link.mass for link in robot.links)
    com_height = sum(link.mass * p_com[i, 2] for i, link in enumerate(robot.links)) / total_mass
    print(f"站立质心高度: {com_height:.3f} m (期望 ~0.45m)")

    # 获取动作定义
    motions = get_attack_poses()
    q_neutral = get_neutral_pose()

    # 存储结果
    results = {
        'motion_names': [m['name'] for m in motions],
        'motion_names_en': [m['name_en'] for m in motions],
        'q_neutral': q_neutral,
        'total_mass': total_mass,
        'com_height_standing': com_height,
    }

    # 每个动作的详细数据
    motion_results = []

    for mi, motion in enumerate(motions):
        T_total = motion['T_attack'] + motion['T_return']
        n_steps = int(T_total / DT) + 1
        time_array = np.linspace(0, T_total, n_steps)

        print(f"\n--- 动作 {mi+1}/13: {motion['name']} ({motion['name_en']}) ---")
        print(f"    攻击时间: {motion['T_attack']}s, 返回时间: {motion['T_return']}s")

        # 分配数组
        q_traj = np.zeros((n_steps, N_JOINTS))
        qdot_traj = np.zeros((n_steps, N_JOINTS))
        qddot_traj = np.zeros((n_steps, N_JOINTS))
        p_com_traj = np.zeros((n_steps, N_LINKS, 3))
        T_world_traj = np.zeros((n_steps, N_LINKS, 4, 4))
        tau_traj = np.zeros((n_steps, N_JOINTS))

        # 攻击力指标时间序列
        v_ee_traj = np.zeros((n_steps, 3))
        v_speed_traj = np.zeros(n_steps)
        m_e_traj = np.zeros(n_steps)
        E_k_traj = np.zeros(n_steps)
        p_mom_traj = np.zeros(n_steps)

        for ti, t in enumerate(time_array):
            # 关节角度/速度/加速度
            q = motion_trajectory(t, motion, q_neutral)
            qdot = motion_trajectory_deriv(t, motion, q_neutral)
            qddot = motion_trajectory_deriv2(t, motion, q_neutral)

            q_traj[ti] = q
            qdot_traj[ti] = qdot
            qddot_traj[ti] = qddot

            # 正运动学
            T_world, p_com = forward_kinematics(robot, q)
            T_world_traj[ti] = T_world
            p_com_traj[ti] = p_com

            # 雅可比矩阵
            J_v, J_omega = compute_body_jacobians(robot, q, T_world, p_com)

            # 质量矩阵
            M = compute_mass_matrix(robot, q, T_world, J_v, J_omega)

            # 重力矩
            g_vec = compute_gravity(robot, J_v)

            # 科里奥利力 (用简化方法: 仅在关键时间步计算)
            if ti % 10 == 0:
                C_qdot = compute_coriolis_times_qdot(robot, q, qdot)
            else:
                C_qdot = np.zeros(N_JOINTS)

            # 所需力矩
            tau = M @ qddot + C_qdot + g_vec
            tau_traj[ti] = tau

            # 攻击力指标
            ee_link = motion['ee_link']
            v_ee, v_speed, m_e, E_k, p_mom = compute_attack_metrics(
                robot, q, qdot, T_world, p_com, J_v[ee_link], J_omega[ee_link], M)
            v_ee_traj[ti] = v_ee
            v_speed_traj[ti] = v_speed
            m_e_traj[ti] = m_e
            E_k_traj[ti] = E_k
            p_mom_traj[ti] = p_mom

        # 记录峰值指标
        T_a = motion['T_attack']
        attack_idx = int(T_a / DT)
        if attack_idx >= n_steps:
            attack_idx = n_steps - 1

        # 前向阶段的最大速度
        forward_mask = time_array <= T_a + 0.01
        peak_speed_idx = np.argmax(v_speed_traj[forward_mask])

        print(f"    峰值末端速度: {np.max(v_speed_traj):.3f} m/s")
        print(f"    峰值等效质量: {np.max(m_e_traj):.3f} kg")
        print(f"    峰值有效动能: {np.max(E_k_traj):.3f} J")
        print(f"    峰值有效动量: {np.max(p_mom_traj):.3f} kg·m/s")
        print(f"    最大力矩: {np.max(np.abs(tau_traj)):.1f} Nm")

        motion_results.append({
            'name': motion['name'],
            'name_en': motion['name_en'],
            'ee_link': motion['ee_link'],
            'T_attack': motion['T_attack'],
            'T_return': motion['T_return'],
            'n_steps': n_steps,
            'time': time_array,
            'q': q_traj,
            'qdot': qdot_traj,
            'qddot': qddot_traj,
            'tau': tau_traj,
            'p_com': p_com_traj,
            'T_world': T_world_traj,
            'v_ee': v_ee_traj,
            'v_speed': v_speed_traj,
            'm_e': m_e_traj,
            'E_k': E_k_traj,
            'p_momentum': p_mom_traj,
            'peak_v_speed': np.max(v_speed_traj),
            'peak_m_e': np.max(m_e_traj),
            'peak_E_k': np.max(E_k_traj),
            'peak_p_momentum': np.max(p_mom_traj),
            'peak_tau': np.max(np.abs(tau_traj)),
        })

    results['motions'] = motion_results

    # --- 保存结果 ---
    # 将motion_results中的numpy数组保存
    save_dict = {
        'motion_names': np.array([m['name'] for m in motions]),
        'motion_names_en': np.array([m['name_en'] for m in motions]),
        'q_neutral': q_neutral,
        'total_mass': total_mass,
        'com_height_standing': com_height,
    }

    # 按动作保存时间序列数据
    for i, mr in enumerate(motion_results):
        prefix = f'motion_{i}'
        save_dict[f'{prefix}_name'] = np.array(mr['name'])
        save_dict[f'{prefix}_time'] = mr['time']
        save_dict[f'{prefix}_q'] = mr['q']
        save_dict[f'{prefix}_qdot'] = mr['qdot']
        save_dict[f'{prefix}_qddot'] = mr['qddot']
        save_dict[f'{prefix}_tau'] = mr['tau']
        save_dict[f'{prefix}_p_com'] = mr['p_com']
        save_dict[f'{prefix}_v_ee'] = mr['v_ee']
        save_dict[f'{prefix}_v_speed'] = mr['v_speed']
        save_dict[f'{prefix}_m_e'] = mr['m_e']
        save_dict[f'{prefix}_E_k'] = mr['E_k']
        save_dict[f'{prefix}_p_momentum'] = mr['p_momentum']
        save_dict[f'{prefix}_peak_v_speed'] = mr['peak_v_speed']
        save_dict[f'{prefix}_peak_m_e'] = mr['peak_m_e']
        save_dict[f'{prefix}_peak_E_k'] = mr['peak_E_k']
        save_dict[f'{prefix}_peak_p_momentum'] = mr['peak_p_momentum']
        save_dict[f'{prefix}_peak_tau'] = mr['peak_tau']
        save_dict[f'{prefix}_ee_link'] = mr['ee_link']
        save_dict[f'{prefix}_T_attack'] = mr['T_attack']
        save_dict[f'{prefix}_T_return'] = mr['T_return']
        save_dict[f'{prefix}_n_steps'] = mr['n_steps']

    # 保存T_world (太大, 只保存末端执行器的)
    for i, mr in enumerate(motion_results):
        prefix = f'motion_{i}'
        ee = mr['ee_link']
        n = mr['n_steps']
        T_ee = mr['T_world'][:, ee, :, :]  # (n_steps, 4, 4)
        save_dict[f'{prefix}_T_world_ee'] = T_ee

    # 保存机器人参数摘要
    save_dict['n_motions'] = len(motions)
    save_dict['n_joints'] = N_JOINTS
    save_dict['n_links'] = N_LINKS
    save_dict['g'] = G
    save_dict['dt'] = DT

    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results.npz')
    np.savez_compressed(save_path, **save_dict)
    print(f"\n结果已保存至: {save_path}")

    return results


# ============================================================
# 可视化 (code1自带基础可视化)
# ============================================================
def plot_basic_results(results):
    """绘制基础结果图"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import rcParams

    # 中文字体设置
    rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    rcParams['axes.unicode_minus'] = False
    rcParams['font.size'] = 10
    rcParams['figure.dpi'] = 150
    rcParams['savefig.dpi'] = 200
    rcParams['savefig.bbox'] = 'tight'

    motions = results['motions']
    n_motions = len(motions)
    attack_motions = get_attack_poses()

    # --- 图1: 13个动作的关节轨迹 ---
    fig, axes = plt.subplots(4, 4, figsize=(20, 16))
    axes = axes.flatten()
    for i, mr in enumerate(motions):
        ax = axes[i]
        t = mr['time']
        # 找出活动关节 (偏移量较大的)
        delta_q = attack_motions[i]['delta_q']
        active_joints = np.where(np.abs(delta_q) > 0.05)[0]
        if len(active_joints) == 0:
            active_joints = [IDX['SHOULDER_PITCH_L'], IDX['ELBOW_PITCH_L']]
        for j in active_joints[:5]:
            ax.plot(t, np.degrees(mr['q'][:, j]), label=f'J{j:02d}')
        ax.set_title(f"{mr['name']}", fontsize=11)
        ax.set_xlabel('时间 (s)')
        ax.set_ylabel('角度 (°)')
        ax.legend(fontsize=7, loc='best')
        ax.grid(True, alpha=0.3)
    for i in range(n_motions, len(axes)):
        axes[i].set_visible(False)
    plt.suptitle('13种攻击动作的关节轨迹', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fig1_joint_trajectories.png')
    plt.savefig(fig_path)
    plt.close()
    print(f"图1已保存: {fig_path}")

    # --- 图2: 末端执行器速度曲线 ---
    fig, ax = plt.subplots(figsize=(12, 7))
    colors = plt.cm.tab20(np.linspace(0, 1, n_motions))
    for i, mr in enumerate(motions):
        ax.plot(mr['time'], mr['v_speed'], color=colors[i], linewidth=1.5, label=mr['name'])
        # 标记峰值
        peak_idx = np.argmax(mr['v_speed'])
        ax.plot(mr['time'][peak_idx], mr['v_speed'][peak_idx], 'o', color=colors[i], markersize=5)
    ax.set_xlabel('时间 (s)', fontsize=12)
    ax.set_ylabel('末端执行器速度 (m/s)', fontsize=12)
    ax.set_title('13种攻击动作的末端执行器速度曲线', fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fig2_ee_velocity.png')
    plt.savefig(fig_path)
    plt.close()
    print(f"图2已保存: {fig_path}")

    # --- 图3: 有效动能曲线 ---
    fig, ax = plt.subplots(figsize=(12, 7))
    for i, mr in enumerate(motions):
        ax.plot(mr['time'], mr['E_k'], color=colors[i], linewidth=1.5, label=mr['name'])
    ax.set_xlabel('时间 (s)', fontsize=12)
    ax.set_ylabel('有效动能 (J)', fontsize=12)
    ax.set_title('13种攻击动作的有效动能曲线', fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fig3_kinetic_energy.png')
    plt.savefig(fig_path)
    plt.close()
    print(f"图3已保存: {fig_path}")

    # --- 图4: 峰值指标对比柱状图 ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    names = [mr['name'] for mr in motions]
    x = np.arange(n_motions)

    metrics = [
        ([mr['peak_v_speed'] for mr in motions], '峰值末端速度 (m/s)'),
        ([mr['peak_m_e'] for mr in motions], '峰值等效质量 (kg)'),
        ([mr['peak_E_k'] for mr in motions], '峰值有效动能 (J)'),
        ([mr['peak_p_momentum'] for mr in motions], '峰值有效动量 (kg·m/s)'),
    ]

    for ax, (values, ylabel) in zip(axes.flatten(), metrics):
        bars = ax.bar(x, values, color=colors, edgecolor='black', linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')
        # 标注数值
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                   f'{val:.2f}', ha='center', va='bottom', fontsize=7)

    plt.suptitle('13种攻击动作的攻击力指标对比', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fig4_attack_metrics_bar.png')
    plt.savefig(fig_path)
    plt.close()
    print(f"图4已保存: {fig_path}")

    # --- 图5: 力矩曲线 ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    key_joints = [IDX['HIP_PITCH_L'], IDX['KNEE_PITCH_L'],
                  IDX['SHOULDER_PITCH_L'], IDX['ELBOW_PITCH_L']]
    key_names = ['HIP_PITCH_L', 'KNEE_PITCH_L', 'SHOULDER_PITCH_L', 'ELBOW_PITCH_L']

    for ax, jidx, jname in zip(axes.flatten(), key_joints, key_names):
        for i, mr in enumerate(motions):
            ax.plot(mr['time'], mr['tau'][:, jidx], color=colors[i], linewidth=1, label=mr['name'])
        ax.set_title(f'关节 {jname} 力矩', fontsize=11)
        ax.set_xlabel('时间 (s)')
        ax.set_ylabel('力矩 (Nm)')
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)

    plt.suptitle('关键关节的力矩曲线', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fig5_torque_profiles.png')
    plt.savefig(fig_path)
    plt.close()
    print(f"图5已保存: {fig_path}")


# ============================================================
# 入口
# ============================================================
if __name__ == '__main__':
    results = run_simulation()
    plot_basic_results(results)
    print("\n" + "=" * 60)
    print("代码1完成! 请运行 code2_zmp_entropy.py 进行ZMP分析和排序")
    print("=" * 60)
