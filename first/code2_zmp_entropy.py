# -*- coding: utf-8 -*-
"""
环境要求:
    Python >= 3.10
    numpy >= 2.0
    scipy >= 1.14
    matplotlib >= 3.9
    pandas >= 2.2
    seaborn >= 0.13

    安装: pip install -r requirements.txt
    运行: 先运行 code1_lagrangian.py 生成 results.npz，再运行本文件

CUMCM 2026 Problem B - 代码2: ZMP稳定性分析、熵权TOPSIS排序与可视化
读取code1的拉格朗日动力学结果, 计算ZMP轨迹和稳定性指标,
使用熵权TOPSIS法对13种攻击动作进行综合排序。

输入: results.npz (由code1生成)
输出: 12+张可视化图表, 排序结果表格
"""

import numpy as np
import os
from scipy.spatial import ConvexHull
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.patches import Polygon as MplPolygon

# ============================================================
# 全局参数与字体设置
# ============================================================
G = 9.81
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
RESULT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results.npz')
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


# ============================================================
# 数据加载
# ============================================================
def load_results():
    """加载code1的仿真结果"""
    data = np.load(RESULT_PATH, allow_pickle=True)
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
# ZMP计算
# ============================================================
def compute_zmp_trajectory(motion, height_offset=0.85):
    """计算单个动作的ZMP轨迹

    ZMP公式:
    x_zmp = [Σ m_i(z̈_i+g)·x_i - Σ m_i·ẍ_i·z_i - Σ I_i·α_i_y] / [Σ m_i(z̈_i+g)]
    y_zmp = [Σ m_i(z̈_i+g)·y_i - Σ m_i·ÿ_i·z_i + Σ I_i·α_i_x] / [Σ m_i(z̈_i+g)]

    使用有限差分计算CoM加速度和角加速度。
    """
    n_steps = motion['n_steps']
    p_com = motion['p_com']  # (n_steps, 29, 3)
    dt = motion['time'][1] - motion['time'][0] if n_steps > 1 else 0.001

    # 将质心位置上移(骨盆在地面上方约0.85m)
    p_com_shifted = p_com.copy()
    p_com_shifted[:, :, 2] += height_offset

    # 用有限差分计算CoM加速度
    p_com_ddot = np.zeros_like(p_com_shifted)
    for i in range(n_steps):
        if i == 0:
            p_com_ddot[i] = (p_com_shifted[1] - p_com_shifted[0]) / dt**2
        elif i == n_steps - 1:
            p_com_ddot[i] = (p_com_shifted[-1] - p_com_shifted[-2]) / dt**2
        else:
            p_com_ddot[i] = (p_com_shifted[i+1] - 2*p_com_shifted[i] + p_com_shifted[i-1]) / dt**2

    # 连杆质量 (从运动数据推断)
    # 使用总质量40.95kg按比例分配
    total_mass = 40.95
    n_links = p_com.shape[1]
    # 简化: 假设均匀质量分布
    link_masses = np.ones(n_links) * total_mass / n_links

    # 重心位置较大的连杆赋予更大质量
    # 基于p_com的z坐标范围来估计质量分布
    mass_weights = np.array([
        4.086, 1.686, 0.633, 1.858, 4.295, 0.115, 0.718, 0.001,  # 左腿
        1.680, 0.637, 1.831, 4.292, 0.115, 0.716, 0.001,          # 右腿
        9.014,                                                       # 躯干
        0.932, 0.511, 0.909, 1.381, 0.468, 0.001,                  # 左臂
        0.929, 0.510, 0.909, 1.382, 0.467, 0.001,                  # 右臂
        0.845                                                        # 头
    ])
    link_masses = mass_weights

    # 惯量 (简化为球形近似: I ≈ m*r², r取质心偏移的模)
    link_inertia = np.zeros(n_links)

    # 计算ZMP
    zmp_x = np.zeros(n_steps)
    zmp_y = np.zeros(n_steps)
    den = np.zeros(n_steps)

    for t in range(n_steps):
        num_x = 0.0
        num_y = 0.0
        d = 0.0

        for k in range(n_links):
            m_k = link_masses[k]
            x_k, y_k, z_k = p_com_shifted[t, k]
            z_ddot_k = p_com_ddot[t, k, 2]
            x_ddot_k = p_com_ddot[t, k, 0]
            y_ddot_k = p_com_ddot[t, k, 1]

            # 简化惯量贡献 (角加速度近似为0对于快速近似)
            I_k = link_inertia[k]

            denom_k = m_k * (z_ddot_k + G)
            num_x += denom_k * x_k - m_k * x_ddot_k * z_k
            num_y += denom_k * y_k - m_k * y_ddot_k * z_k
            d += denom_k

        if abs(d) > 1e-6:
            zmp_x[t] = num_x / d
            zmp_y[t] = num_y / d
        else:
            zmp_x[t] = 0.0
            zmp_y[t] = 0.0

        den[t] = d

    return zmp_x, zmp_y, den


def compute_support_polygon(height_offset=0.85):
    """计算双脚支撑多边形

    脚掌近似为矩形: 前后0.16m, 左右0.10m
    左脚中心: (0, 0.045, 0), 右脚中心: (0, -0.045, 0)
    """
    # 脚掌尺寸
    foot_len = 0.16  # 前后
    foot_wid = 0.10  # 左右
    hip_width = 0.045  # 髋关节横向偏移

    # 左脚四个角点
    left_foot = np.array([
        [foot_len/2, hip_width + foot_wid/2, 0],
        [-foot_len/2, hip_width + foot_wid/2, 0],
        [-foot_len/2, hip_width - foot_wid/2, 0],
        [foot_len/2, hip_width - foot_wid/2, 0],
    ])

    # 右脚四个角点
    right_foot = np.array([
        [foot_len/2, -hip_width + foot_wid/2, 0],
        [-foot_len/2, -hip_width + foot_wid/2, 0],
        [-foot_len/2, -hip_width - foot_wid/2, 0],
        [foot_len/2, -hip_width - foot_wid/2, 0],
    ])

    # 合并两脚的所有角点, 求凸包
    all_points = np.vstack([left_foot[:, :2], right_foot[:, :2]])
    hull = ConvexHull(all_points)
    polygon = all_points[hull.vertices]

    return polygon, left_foot[:, :2], right_foot[:, :2]


def zmp_margin(zmp_x, zmp_y, polygon):
    """计算ZMP到支撑多边形边界的最小距离

    正值: ZMP在多边形内部
    负值: ZMP在多边形外部
    """
    n = len(polygon)
    min_dist = np.inf

    for i in range(n):
        p1 = polygon[i]
        p2 = polygon[(i + 1) % n]

        # 线段p1-p2
        edge = p2 - p1
        edge_len = np.linalg.norm(edge)
        if edge_len < 1e-12:
            continue
        edge_unit = edge / edge_len
        normal = np.array([-edge_unit[1], edge_unit[0]])  # 外法线

        # ZMP到线段的距离
        for zx, zy in zip(zmp_x, zmp_y):
            to_point = np.array([zx, zy]) - p1
            # 沿边的投影
            proj = np.dot(to_point, edge_unit)
            # 法向距离
            dist_normal = np.dot(to_point, normal)

            # 点到线段的最短距离
            if proj < 0:
                dist = np.linalg.norm(to_point)
            elif proj > edge_len:
                dist = np.linalg.norm(np.array([zx, zy]) - p2)
            else:
                dist = abs(dist_normal)

            # 判断内外: 用叉积
            cross = edge[0] * (zy - p1[1]) - edge[1] * (zx - p1[0])
            if cross < 0:  # 在边的外侧
                dist = -dist

            min_dist = min(min_dist, dist)

    return min_dist


def compute_stability_metrics(motion, height_offset=0.85):
    """计算稳定性指标"""
    # ZMP轨迹
    zmp_x, zmp_y, den = compute_zmp_trajectory(motion, height_offset)

    # 支撑多边形
    polygon, left_foot, right_foot = compute_support_polygon(height_offset)

    # ZMP裕度 (全周期最小值)
    margin_min = zmp_margin(zmp_x, zmp_y, polygon)

    # 躯干倾角 (从q轨迹估算)
    # 简化: 用WAIST_YAW的关节角度变化来近似躯干倾斜
    q = motion['q']
    # 躯干倾角: 用髋关节和膝关节角度来估算
    # 实际上应该是躯干z轴与世界z轴的夹角
    # 这里用各时刻的q来近似
    n_steps = motion['n_steps']

    # 角动量变化率 (简化计算)
    # 用qddot的L2范数来近似角动量变化率
    qddot = motion['qddot']
    angular_momentum_rate = np.sqrt(np.sum(qddot**2, axis=1))

    return {
        'zmp_x': zmp_x,
        'zmp_y': zmp_y,
        'zmp_margin_min': margin_min,
        'angular_momentum_rate': angular_momentum_rate,
        'peak_angular_momentum_rate': np.max(angular_momentum_rate),
        'polygon': polygon,
        'left_foot': left_foot,
        'right_foot': right_foot,
    }


# ============================================================
# 熵权TOPSIS法
# ============================================================
def entropy_weight_topsis(indicator_matrix, indicator_names, motion_names):
    """熵权TOPSIS综合评价

    Args:
        indicator_matrix: (n_motions, n_indicators) 原始指标矩阵
        indicator_names: 指标名称列表
        motion_names: 动作名称列表

    Returns:
        dict with weights, scores, ranking
    """
    n, m = indicator_matrix.shape

    print("\n" + "=" * 60)
    print("熵权TOPSIS综合评价")
    print("=" * 60)

    # Step 1: 极差标准化
    X = indicator_matrix.copy().astype(float)
    for j in range(m):
        col_min = np.min(X[:, j])
        col_max = np.max(X[:, j])
        if col_max - col_min > 1e-12:
            X[:, j] = (X[:, j] - col_min) / (col_max - col_min)
        else:
            X[:, j] = 1.0 / n

    print(f"\n标准化矩阵:\n{np.array2string(X, precision=3, suppress_small=True)}")

    # Step 2: 计算比重矩阵
    P = np.zeros_like(X)
    for j in range(m):
        col_sum = np.sum(X[:, j])
        if col_sum > 1e-12:
            P[:, j] = X[:, j] / col_sum
        else:
            P[:, j] = 1.0 / n

    # Step 3: 计算信息熵
    E = np.zeros(m)
    for j in range(m):
        for i in range(n):
            if P[i, j] > 1e-12:
                E[j] -= P[i, j] * np.log(P[i, j])
        E[j] /= np.log(n)

    print(f"\n信息熵 E: {np.array2string(E, precision=4)}")

    # Step 4: 计算熵权
    D = 1 - E  # 差异系数
    if np.sum(D) > 1e-12:
        W = D / np.sum(D)
    else:
        W = np.ones(m) / m

    print(f"差异系数 D: {np.array2string(D, precision=4)}")
    print(f"熵权 W: {np.array2string(W, precision=4)}")

    for j in range(m):
        print(f"  {indicator_names[j]}: w = {W[j]:.4f}")

    # Step 5: 加权标准化矩阵
    V = X * W  # 广播

    # Step 6: 正负理想解
    V_plus = np.max(V, axis=0)
    V_minus = np.min(V, axis=0)

    # Step 7: 计算距离
    D_plus = np.sqrt(np.sum((V - V_plus)**2, axis=1))
    D_minus = np.sqrt(np.sum((V - V_minus)**2, axis=1))

    # Step 8: 相对接近度
    C = D_minus / (D_plus + D_minus + 1e-12)

    # 排序
    ranking = np.argsort(-C)  # 降序

    print(f"\n{'排名':<4} {'动作名称':<12} {'D+':<8} {'D-':<8} {'C_i':<8}")
    print("-" * 48)
    for rank, idx in enumerate(ranking):
        print(f"{rank+1:<4} {motion_names[idx]:<12} {D_plus[idx]:<8.4f} {D_minus[idx]:<8.4f} {C[idx]:<8.4f}")

    return {
        'weights': W,
        'entropy': E,
        'diversity': D,
        'scores': C,
        'D_plus': D_plus,
        'D_minus': D_minus,
        'ranking': ranking,
        'normalized_matrix': X,
        'weighted_matrix': V,
        'V_plus': V_plus,
        'V_minus': V_minus,
    }


# ============================================================
# 可视化
# ============================================================
def create_figures(motions, stability_results, topsis_results):
    """生成所有可视化图表"""
    os.makedirs(FIG_DIR, exist_ok=True)
    n_motions = len(motions)
    names = [mr['name'] for mr in motions]
    names_en = [mr['name_en'] for mr in motions]

    colors = plt.cm.tab20(np.linspace(0, 1, n_motions))
    categories = ['拳法', '拳法', '拳法', '拳法', '拳法', '拳法', '拳法', '肘法',
                  '腿法', '腿法', '腿法', '腿法', '腿法']
    cat_colors = {'拳法': '#2196F3', '肘法': '#FF9800', '腿法': '#F44336'}

    # ==========================================
    # 图6: ZMP轨迹 + 支撑多边形
    # ==========================================
    fig, ax = plt.subplots(figsize=(10, 8))

    # 绘制支撑多边形
    polygon = stability_results[0]['polygon']
    poly_patch = MplPolygon(polygon, closed=True, facecolor='#E0E0E0',
                           edgecolor='black', linewidth=2, alpha=0.5, label='支撑多边形')
    ax.add_patch(poly_patch)

    # 绘制双脚轮廓
    for si in range(n_motions):
        lf = stability_results[si]['left_foot']
        rf = stability_results[si]['right_foot']
        if si == 0:
            ax.fill(lf[:, 0], lf[:, 1], alpha=0.3, color='#90CAF9', label='左脚')
            ax.fill(rf[:, 0], rf[:, 1], alpha=0.3, color='#90CAF9', label='右脚')
        else:
            ax.fill(lf[:, 0], lf[:, 1], alpha=0.3, color='#90CAF9')
            ax.fill(rf[:, 0], rf[:, 1], alpha=0.3, color='#90CAF9')

    # 绘制各动作的ZMP轨迹
    for i in range(n_motions):
        zx = stability_results[i]['zmp_x']
        zy = stability_results[i]['zmp_y']
        line, = ax.plot(zx, zy, color=colors[i], linewidth=1.5, label=names[i])
        # 标记起点和终点
        ax.plot(zx[0], zy[0], 'o', color=colors[i], markersize=4)
        ax.plot(zx[-1], zy[-1], 's', color=colors[i], markersize=4)

    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_title('ZMP轨迹与支撑多边形', fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.plot([], [], 'ok', markersize=4, label='起点 (o)')
    ax.plot([], [], 'sk', markersize=4, label='终点 (s)')
    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, 'fig6_zmp_trajectory.png')
    plt.savefig(fig_path)
    plt.close()
    print(f"图6已保存: {fig_path}")

    # ==========================================
    # 图7: 帕累托前沿 (有效动能 vs ZMP裕度)
    # ==========================================
    fig, ax = plt.subplots(figsize=(10, 7))

    E_k_vals = [mr['peak_E_k'] for mr in motions]
    zmp_margins = [stability_results[i]['zmp_margin_min'] for i in range(n_motions)]

    # 分类绘制
    for i in range(n_motions):
        cat = categories[i]
        ax.scatter(E_k_vals[i], zmp_margins[i], c=cat_colors[cat], s=120,
                  edgecolors='black', linewidth=0.5, zorder=5)
        ax.annotate(f'{names[i]}', (E_k_vals[i], zmp_margins[i]),
                   textcoords="offset points", xytext=(8, 5), fontsize=8)

    # 找帕累托前沿
    pareto_mask = np.ones(n_motions, dtype=bool)
    for i in range(n_motions):
        for j in range(n_motions):
            if i != j:
                if E_k_vals[j] >= E_k_vals[i] and zmp_margins[j] >= zmp_margins[i]:
                    if E_k_vals[j] > E_k_vals[i] or zmp_margins[j] > zmp_margins[i]:
                        pareto_mask[i] = False
                        break

    pareto_indices = np.where(pareto_mask)[0]
    pareto_E = [E_k_vals[i] for i in pareto_indices]
    pareto_margin = [zmp_margins[i] for i in pareto_indices]

    # 按E_k排序连接帕累托前沿
    sort_idx = np.argsort(pareto_E)
    pareto_E_sorted = [pareto_E[i] for i in sort_idx]
    pareto_margin_sorted = [pareto_margin[i] for i in sort_idx]

    ax.plot(pareto_E_sorted, pareto_margin_sorted, 'r--', linewidth=2, alpha=0.7, label='帕累托前沿')
    ax.scatter(pareto_E_sorted, pareto_margin_sorted, c='red', s=200, marker='*',
              zorder=6, label='帕累托最优点')

    # 安全阈值线
    ax.axhline(y=0, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    ax.axhline(y=0.02, color='green', linestyle=':', linewidth=1, alpha=0.5, label='安全阈值 (0.02m)')

    # 图例
    for cat, color in cat_colors.items():
        ax.scatter([], [], c=color, s=60, label=cat, edgecolors='black', linewidth=0.5)

    ax.set_xlabel('峰值有效动能 (J)', fontsize=12)
    ax.set_ylabel('最小ZMP稳定裕度 (m)', fontsize=12)
    ax.set_title('攻击力-稳定性帕累托前沿', fontsize=14, fontweight='bold')
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, 'fig7_pareto_front.png')
    plt.savefig(fig_path)
    plt.close()
    print(f"图7已保存: {fig_path}")

    # ==========================================
    # 图8: TOPSIS排名条形图
    # ==========================================
    fig, ax = plt.subplots(figsize=(10, 7))

    ranking = topsis_results['ranking']
    scores = topsis_results['scores']
    ranked_names = [names[i] for i in ranking]
    ranked_scores = [scores[i] for i in ranking]
    ranked_colors = [cat_colors[categories[i]] for i in ranking]

    y_pos = np.arange(n_motions)
    bars = ax.barh(y_pos, ranked_scores, color=ranked_colors, edgecolor='black', linewidth=0.5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(ranked_names, fontsize=10)
    ax.set_xlabel('综合评价值 C_i', fontsize=12)
    ax.set_title('熵权TOPSIS综合排名', fontsize=14, fontweight='bold')
    ax.invert_yaxis()

    # 标注排名和分数
    for i, (bar, score) in enumerate(zip(bars, ranked_scores)):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
               f'#{i+1}  {score:.4f}', va='center', fontsize=9)

    # 图例
    for cat, color in cat_colors.items():
        ax.scatter([], [], c=color, s=60, label=cat, edgecolors='black', linewidth=0.5)
    ax.legend(fontsize=9, loc='lower right')

    ax.set_xlim(0, max(ranked_scores) * 1.15)
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, 'fig8_topsis_ranking.png')
    plt.savefig(fig_path)
    plt.close()
    print(f"图8已保存: {fig_path}")

    # ==========================================
    # 图9: 指标热力图
    # ==========================================
    fig, ax = plt.subplots(figsize=(12, 8))

    indicator_names = ['有效动能', '有效动量', '末端速度', 'ZMP裕度', '角动量率']
    X_norm = topsis_results['normalized_matrix']
    W = topsis_results['weights']

    im = ax.imshow(X_norm, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)

    ax.set_xticks(range(len(indicator_names)))
    ax.set_xticklabels([f'{n}\n(w={W[i]:.3f})' for i, n in enumerate(indicator_names)],
                       fontsize=9, rotation=0)
    ax.set_yticks(range(n_motions))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_title('指标热力图 (标准化值)', fontsize=14, fontweight='bold')

    # 在每个格子中标注数值
    for i in range(n_motions):
        for j in range(len(indicator_names)):
            ax.text(j, i, f'{X_norm[i, j]:.2f}', ha='center', va='center',
                   fontsize=8, color='black' if X_norm[i, j] < 0.6 else 'white')

    plt.colorbar(im, ax=ax, label='标准化值')
    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, 'fig9_indicator_heatmap.png')
    plt.savefig(fig_path)
    plt.close()
    print(f"图9已保存: {fig_path}")

    # ==========================================
    # 图10: 顶部动作雷达图
    # ==========================================
    top_n = min(5, n_motions)
    top_indices = ranking[:top_n]

    fig, axes = plt.subplots(1, top_n, figsize=(4*top_n, 4), subplot_kw=dict(polar=True))
    if top_n == 1:
        axes = [axes]

    indicator_labels = ['动能', '动量', '速度', 'ZMP', '稳定性']

    for plot_idx, motion_idx in enumerate(top_indices):
        ax = axes[plot_idx]
        values = X_norm[motion_idx].tolist()
        values += values[:1]  # 闭合

        angles = np.linspace(0, 2 * np.pi, len(indicator_labels), endpoint=False).tolist()
        angles += angles[:1]

        ax.fill(angles, values, alpha=0.25, color=cat_colors[categories[motion_idx]])
        ax.plot(angles, values, color=cat_colors[categories[motion_idx]], linewidth=2)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(indicator_labels, fontsize=8)
        ax.set_title(f'#{ranking.tolist().index(motion_idx)+1} {names[motion_idx]}',
                    fontsize=10, fontweight='bold', pad=15)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(['0.25', '0.5', '0.75', '1.0'], fontsize=7)

    plt.suptitle('TOP 5动作指标雷达图', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, 'fig10_radar_top5.png')
    plt.savefig(fig_path)
    plt.close()
    print(f"图10已保存: {fig_path}")

    # ==========================================
    # 图11: 角动量变化率曲线
    # ==========================================
    fig, ax = plt.subplots(figsize=(12, 7))
    for i in range(n_motions):
        ax.plot(motions[i]['time'], stability_results[i]['angular_momentum_rate'],
               color=colors[i], linewidth=1.5, label=names[i])
    ax.set_xlabel('时间 (s)', fontsize=12)
    ax.set_ylabel('角动量变化率 ||dH/dt|| (Nm)', fontsize=12)
    ax.set_title('角动量变化率曲线', fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, 'fig11_angular_momentum_rate.png')
    plt.savefig(fig_path)
    plt.close()
    print(f"图11已保存: {fig_path}")

    # ==========================================
    # 图12: TOP4动作力矩曲线对比
    # ==========================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    top4 = ranking[:4]
    key_joints = [IDX['HIP_PITCH_L'], IDX['KNEE_PITCH_L'],
                  IDX['SHOULDER_PITCH_L'], IDX['ELBOW_PITCH_L']]
    key_names = ['HIP_PITCH_L', 'KNEE_PITCH_L', 'SHOULDER_PITCH_L', 'ELBOW_PITCH_L']

    for ax, jidx, jname in zip(axes.flatten(), key_joints, key_names):
        for mi in top4:
            ax.plot(motions[mi]['time'], motions[mi]['tau'][:, jidx],
                   linewidth=1.5, label=f'#{ranking.tolist().index(mi)+1} {names[mi]}')
        ax.set_title(f'关节 {jname} 力矩', fontsize=11)
        ax.set_xlabel('时间 (s)')
        ax.set_ylabel('力矩 (Nm)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle('TOP 4动作的关键关节力矩对比', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, 'fig12_top4_torques.png')
    plt.savefig(fig_path)
    plt.close()
    print(f"图12已保存: {fig_path}")

    # ==========================================
    # 图13: ZMP裕度随时间变化
    # ==========================================
    fig, ax = plt.subplots(figsize=(12, 7))
    for i in range(n_motions):
        zx = stability_results[i]['zmp_x']
        zy = stability_results[i]['zmp_y']
        # 计算每个时刻的裕度
        polygon = stability_results[i]['polygon']
        n_t = len(zx)
        margins_t = np.zeros(n_t)
        for t in range(n_t):
            margins_t[t] = zmp_margin(np.array([zx[t]]), np.array([zy[t]]), polygon)
        ax.plot(motions[i]['time'], margins_t, color=colors[i], linewidth=1.5, label=names[i])

    ax.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5, label='稳定边界')
    ax.set_xlabel('时间 (s)', fontsize=12)
    ax.set_ylabel('ZMP稳定裕度 (m)', fontsize=12)
    ax.set_title('ZMP稳定裕度随时间变化', fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, 'fig13_zmp_margin_time.png')
    plt.savefig(fig_path)
    plt.close()
    print(f"图13已保存: {fig_path}")

    # ==========================================
    # 图14: 攻击力指标汇总表格
    # ==========================================
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('off')

    col_labels = ['动作名称', '峰值速度\n(m/s)', '等效质量\n(kg)', '有效动能\n(J)',
                  '有效动量\n(kg·m/s)', 'ZMP裕度\n(m)', 'TOPSIS\n评分', '排名']
    table_data = []
    ranking_list = ranking.tolist()
    for i in range(n_motions):
        row = [
            names[i],
            f'{motions[i]["peak_v_speed"]:.2f}',
            f'{motions[i]["peak_m_e"]:.1f}',
            f'{motions[i]["peak_E_k"]:.1f}',
            f'{motions[i]["peak_p_momentum"]:.1f}',
            f'{stability_results[i]["zmp_margin_min"]:.4f}',
            f'{topsis_results["scores"][i]:.4f}',
            f'#{ranking_list.index(i)+1}'
        ]
        table_data.append(row)

    table = ax.table(cellText=table_data, colLabels=col_labels, loc='center',
                    cellLoc='center', colLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)

    # 着色: TOP3高亮
    for i in range(n_motions):
        rank = ranking_list.index(i)
        if rank < 3:
            for j in range(len(col_labels)):
                table[i+1, j].set_facecolor('#C8E6C9')
        # 按类别着色名称列
        table[i+1, 0].set_facecolor(cat_colors[categories[i]])
        table[i+1, 0].set_text_props(color='white', fontweight='bold')

    ax.set_title('13种攻击动作综合评价结果', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, 'fig14_summary_table.png')
    plt.savefig(fig_path)
    plt.close()
    print(f"图14已保存: {fig_path}")

    # ==========================================
    # 图15: 熵权分布饼图
    # ==========================================
    fig, ax = plt.subplots(figsize=(8, 8))
    W = topsis_results['weights']
    indicator_names_full = ['有效动能 E_k', '有效动量 p', '末端速度 v', 'ZMP裕度', '角动量率']

    wedges, texts, autotexts = ax.pie(W, labels=indicator_names_full, autopct='%1.2f%%',
                                      colors=plt.cm.Set3(np.linspace(0, 1, len(W))),
                                      startangle=90, textprops={'fontsize': 10})
    for autotext in autotexts:
        autotext.set_fontsize(10)
        autotext.set_fontweight('bold')
    ax.set_title('熵权分布', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, 'fig15_entropy_weights.png')
    plt.savefig(fig_path)
    plt.close()
    print(f"图15已保存: {fig_path}")

    # ==========================================
    # 图16: D+/D-距离对比
    # ==========================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ranked_names_list = [names[i] for i in ranking]
    D_plus_ranked = [topsis_results['D_plus'][i] for i in ranking]
    D_minus_ranked = [topsis_results['D_minus'][i] for i in ranking]
    ranked_cat_colors = [cat_colors[categories[i]] for i in ranking]

    y_pos = np.arange(n_motions)

    axes[0].barh(y_pos, D_plus_ranked, color=ranked_cat_colors, edgecolor='black', linewidth=0.5)
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(ranked_names_list, fontsize=9)
    axes[0].set_xlabel('D+ (到正理想解距离)', fontsize=11)
    axes[0].set_title('到正理想解的距离', fontsize=12, fontweight='bold')
    axes[0].invert_yaxis()
    axes[0].grid(True, alpha=0.3, axis='x')

    axes[1].barh(y_pos, D_minus_ranked, color=ranked_cat_colors, edgecolor='black', linewidth=0.5)
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels(ranked_names_list, fontsize=9)
    axes[1].set_xlabel('D- (到负理想解距离)', fontsize=11)
    axes[1].set_title('到负理想解的距离', fontsize=12, fontweight='bold')
    axes[1].invert_yaxis()
    axes[1].grid(True, alpha=0.3, axis='x')

    plt.suptitle('TOPSIS距离分析', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, 'fig16_topsis_distances.png')
    plt.savefig(fig_path)
    plt.close()
    print(f"图16已保存: {fig_path}")


# ============================================================
# 主函数
# ============================================================
def main():
    print("=" * 60)
    print("ZMP稳定性分析 + 熵权TOPSIS排序")
    print("=" * 60)

    # 加载数据
    motions, raw_data = load_results()
    n_motions = len(motions)
    print(f"加载了 {n_motions} 个动作的数据")

    # 计算稳定性指标
    print("\n--- 计算ZMP轨迹和稳定性指标 ---")
    stability_results = []
    for i, mr in enumerate(motions):
        print(f"  计算 {mr['name']} ...", end=' ')
        stab = compute_stability_metrics(mr)
        stability_results.append(stab)
        print(f"ZMP裕度: {stab['zmp_margin_min']:.4f} m")

    # 构建TOPSIS指标矩阵
    print("\n--- 构建评价指标矩阵 ---")
    indicator_names = ['有效动能', '有效动量', '末端速度', 'ZMP裕度', '角动量率(反)']

    indicator_matrix = np.zeros((n_motions, 5))
    for i in range(n_motions):
        indicator_matrix[i, 0] = motions[i]['peak_E_k']       # 有效动能 (效益型)
        indicator_matrix[i, 1] = motions[i]['peak_p_momentum'] # 有效动量 (效益型)
        indicator_matrix[i, 2] = motions[i]['peak_v_speed']    # 末端速度 (效益型)
        indicator_matrix[i, 3] = stability_results[i]['zmp_margin_min']  # ZMP裕度 (效益型)
        indicator_matrix[i, 4] = -stability_results[i]['peak_angular_momentum_rate']  # 角动量率取负 (成本→效益)

    print(f"\n指标矩阵 (13×5):")
    print(f"{'动作':<10} {'动能(J)':<10} {'动量':<10} {'速度(m/s)':<10} {'ZMP(m)':<10} {'角动量(反)':<10}")
    for i in range(n_motions):
        print(f"{motions[i]['name']:<10} {indicator_matrix[i,0]:<10.2f} {indicator_matrix[i,1]:<10.2f} "
              f"{indicator_matrix[i,2]:<10.2f} {indicator_matrix[i,3]:<10.4f} {indicator_matrix[i,4]:<10.2f}")

    # 熵权TOPSIS
    motion_names = [mr['name'] for mr in motions]
    topsis_results = entropy_weight_topsis(indicator_matrix, indicator_names, motion_names)

    # 生成可视化
    print("\n--- 生成可视化图表 ---")
    create_figures(motions, stability_results, topsis_results)

    # 最终结论
    print("\n" + "=" * 60)
    print("最终结论")
    print("=" * 60)
    ranking = topsis_results['ranking']
    print("\n推荐的优先动作 (TOP 5):")
    for i in range(min(5, n_motions)):
        idx = ranking[i]
        print(f"  #{i+1}: {motions[idx]['name']} ({motions[idx]['name_en']})")
        print(f"       有效动能: {motions[idx]['peak_E_k']:.1f} J, "
              f"ZMP裕度: {stability_results[idx]['zmp_margin_min']:.4f} m, "
              f"TOPSIS评分: {topsis_results['scores'][idx]:.4f}")

    print(f"\n所有图表已保存至: {FIG_DIR}")
    print("=" * 60)


if __name__ == '__main__':
    main()
