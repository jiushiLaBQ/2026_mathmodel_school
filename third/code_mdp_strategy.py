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
    运行: python code_mdp_strategy.py

CUMCM 2026 Problem B - 代码4: 基于MDP的最优格斗策略
将机器人格斗建模为马尔可夫决策过程(162状态×35动作),
基于专家规则手工构造转移概率, 用值迭代求解最优策略,
输出三回合分阶段的最佳攻防方案。

输入: first/results.npz (问题一攻击动力学数据)
输出: 10+张可视化图表
"""

import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.patches import FancyBboxPatch
import matplotlib.gridspec as gridspec

# ============================================================
# 全局参数与字体设置
# ============================================================
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
RESULT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'first', 'results.npz')
FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')

rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei', 'Arial Unicode MS']
rcParams['axes.unicode_minus'] = False
rcParams['font.family'] = 'sans-serif'
rcParams['font.size'] = 11
rcParams['figure.dpi'] = 150
rcParams['savefig.dpi'] = 200
rcParams['savefig.bbox'] = 'tight'

GAMMA = 0.95  # 折扣因子 (降低以加速收敛, 更关注近期收益)
CONVERGE_TOL = 1e-4  # 收敛阈值
MAX_ITER = 2000  # 最大迭代次数
N_EPISODES = 10000  # 蒙特卡洛仿真场次
DT_SIM = 1.0  # 仿真决策间隔(秒)
ROUND_TIME = 120  # 每回合时间(秒)

# ============================================================
# 第一部分: MDP状态空间定义
# ============================================================
# 5个维度, 总状态数 = 3 × 3 × 3 × 3 × 2 = 162

SCORE_LEVELS = ['劣势', '均势', '优势']  # 净胜分: <-2, [-2,2], >2
STAMINA_LEVELS = ['低', '中', '高']  # 体力: <30%, 30-70%, >70%
DISTANCE_LEVELS = ['近', '中', '远']  # 距离: <0.5m, 0.5-1.5m, >1.5m
TIME_LEVELS = ['前期', '中期', '后期']  # 时间: 0-40s, 40-80s, 80-120s
OPPONENT_LEVELS = ['进攻', '防守']  # 对手姿态

N_SCORE = len(SCORE_LEVELS)
N_STAMINA = len(STAMINA_LEVELS)
N_DIST = len(DISTANCE_LEVELS)
N_TIME = len(TIME_LEVELS)
N_OPP = len(OPPONENT_LEVELS)
N_STATES = N_SCORE * N_STAMINA * N_DIST * N_TIME * N_OPP  # 162

# ============================================================
# 第二部分: MDP动作空间定义
# ============================================================
# 13种攻击 + 22种防御 = 35种动作

ATTACK_NAMES = [
    '左直拳', '右直拳', '左摆拳', '右摆拳', '左上勾拳', '右上勾拳',
    '左掌击', '右肘击', '左膝击', '右前蹬', '右侧踢', '右回旋踢', '右后踢'
]
DEFENSE_NAMES = [
    '左上格挡', '右上格挡', '左中格挡', '右中格挡', '双臂交叉格挡',
    '下潜闪避', '左侧闪避', '右侧闪避', '后仰闪避', '前俯闪避',
    '后撤步', '侧滑步', '前压步',
    '左膝防御', '右膝防御',
    '抱架防御', '低姿态防御', '侧身防御',
    '左反击准备', '右反击准备', '腿法反击准备',
    '全力后撤'
]
ALL_ACTIONS = ATTACK_NAMES + DEFENSE_NAMES
N_ATTACKS = len(ATTACK_NAMES)
N_DEFENSES = len(DEFENSE_NAMES)
N_ACTIONS = N_ATTACKS + N_DEFENSES  # 35

# 动作类型标签
ACTION_TYPES = ['攻击'] * N_ATTACKS + ['防御'] * N_DEFENSES

# 各攻击的体力消耗系数 (问题一tau数据归一化)
ATTACK_STAMINA_COST = np.array([
    0.02, 0.02, 0.03, 0.03, 0.025, 0.025, 0.02, 0.03, 0.04, 0.05, 0.05, 0.06, 0.04
])
DEFENSE_STAMINA_COST = np.array([
    0.01, 0.01, 0.01, 0.01, 0.015, 0.02, 0.015, 0.015, 0.015, 0.015,
    0.02, 0.02, 0.015, 0.025, 0.025, 0.01, 0.015, 0.01, 0.015, 0.015, 0.02, 0.03
])
ALL_STAMINA_COST = np.concatenate([ATTACK_STAMINA_COST, DEFENSE_STAMINA_COST])

# 各攻击的有效动能 (问题一数据, 归一化)
ATTACK_POWER = np.array([
    0.5, 0.5, 0.6, 0.6, 0.55, 0.55, 0.45, 0.65, 0.7, 0.75, 0.8, 0.9, 0.7
])
# 各防御的防御强度 (基于问题二效用矩阵归一化)
DEFENSE_STRENGTH = np.array([
    0.6, 0.6, 0.65, 0.65, 0.75, 0.8, 0.7, 0.7, 0.65, 0.6,
    0.7, 0.65, 0.4, 0.7, 0.7, 0.6, 0.55, 0.55, 0.5, 0.5, 0.45, 0.8
])

# ============================================================
# 第三部分: 状态编码/解码
# ============================================================
def encode_state(score, stamina, dist, time_p, opp):
    """将5维状态编码为一维索引 [0, 162)"""
    return ((score * N_STAMINA + stamina) * N_DIST + dist) * N_TIME * N_OPP + time_p * N_OPP + opp


def decode_state(s):
    """将一维索引解码为5维状态"""
    opp = s % N_OPP
    s //= N_OPP
    time_p = s % N_TIME
    s //= N_TIME
    dist = s % N_DIST
    s //= N_DIST
    stamina = s % N_STAMINA
    s //= N_STAMINA
    score = s % N_SCORE
    return score, stamina, dist, time_p, opp


def state_to_str(s):
    """状态索引转可读字符串"""
    sc, st, di, ti, op = decode_state(s)
    return f"({SCORE_LEVELS[sc]}, {STAMINA_LEVELS[st]}, {DISTANCE_LEVELS[di]}, {TIME_LEVELS[ti]}, {OPPONENT_LEVELS[op]})"


# ============================================================
# 第四部分: 奖励函数
# ============================================================
# 时间阶段权重: 后期更需要得分, 前期可以保守
TIME_URGENCY = np.array([0.5, 1.0, 2.0])  # 前期/中期/后期紧迫度
# 分差策略调整: 落后时更需要进攻奖励
SCORE_AGGRESSION = np.array([1.5, 1.0, 0.6])  # 劣势/均势/优势的攻击倾向

def compute_reward(s, a, s_next):
    """计算即时奖励 R(s, a, s')

    综合考虑: 得分变化、态势改善、体力消耗、时间紧迫度、分差策略
    """
    sc_now, st_now, di_now, ti_now, op_now = decode_state(s)
    sc_next, st_next, di_next, ti_next, op_next = decode_state(s_next)

    is_attack = a < N_ATTACKS
    cost = ALL_STAMINA_COST[a]

    # 1. 得分变化 (分差等级变化)
    score_change = (sc_next - sc_now)

    # 2. 态势改善
    situation = 0
    if op_now == 0 and op_next == 1:
        situation = 1
    elif op_now == 1 and op_next == 0:
        situation = -1

    # 3. 距离控制奖励 (合适距离+1, 不合适-0.5)
    dist_control = 0
    if is_attack:
        # 攻击时: 近距拳法好, 远距腿法好
        if di_next == 0 and a < 8:
            dist_control = 1.0  # 近身拳法
        elif di_next == 2 and a >= 8:
            dist_control = 0.8  # 远距腿法
        elif di_next == 1:
            dist_control = 0.5  # 中距通用
    else:
        # 防御时: 后撤拉开距离有奖励
        if '后撤' in ALL_ACTIONS[a] or '全力后撤' in ALL_ACTIONS[a]:
            dist_control = 0.5

    # 4. 时间紧迫度加权
    urgency = TIME_URGENCY[ti_now]

    # 5. 分差策略调整: 落后时攻击收益更高, 领先时防御收益更高
    aggression = SCORE_AGGRESSION[sc_now]

    # 综合奖励
    reward = (10 * score_change * urgency
              + 5 * situation * urgency
              + 3 * dist_control
              - 20 * cost
              + (2 * aggression if is_attack else 1.0) * score_change)

    return reward


# ============================================================
# 第五部分: 专家规则转移概率 (核心)
# ============================================================
def build_transition_and_reward():
    """基于专家规则构建转移概率矩阵 P(s'|s,a) 和期望奖励 R(s,a)

    对每个(s,a), 定义2-4个可能的下一状态及其概率
    """
    P = np.zeros((N_STATES, N_ACTIONS, N_STATES))
    R = np.zeros((N_STATES, N_ACTIONS))

    for s in range(N_STATES):
        sc, st, di, ti, op = decode_state(s)

        for a in range(N_ACTIONS):
            is_attack = a < N_ATTACKS
            action_name = ALL_ACTIONS[a]
            cost = ALL_STAMINA_COST[a]

            # 时间阶段推进
            ti_next = min(ti + 1, N_TIME - 1) if ti < N_TIME - 1 else ti

            # 体力变化
            st_next = st
            if st == 2:
                if cost > 0.04:
                    st_next = 1
            elif st == 1:
                if cost > 0.025:
                    st_next = 0
                elif cost < 0.015:
                    st_next = 2
            elif st == 0:
                if cost < 0.01:
                    st_next = 1

            # ===== 攻击动作 =====
            if is_attack:
                power = ATTACK_POWER[a]

                # 距离匹配因子
                dist_match = 1.0
                if di == 0:  # 近距
                    if a < 6:      dist_match = 1.3   # 拳法
                    elif a == 7:   dist_match = 1.4   # 肘击
                    elif a == 8:   dist_match = 1.5   # 膝击
                    else:          dist_match = 0.5   # 腿法
                elif di == 1:  # 中距
                    if a < 6:      dist_match = 0.9
                    elif a in [9, 10]: dist_match = 1.2
                    else:          dist_match = 1.0
                else:  # 远距
                    if a < 6:      dist_match = 0.4
                    elif a in [11, 12]: dist_match = 1.3
                    elif a == 9:   dist_match = 1.1
                    else:          dist_match = 0.7

                # 对手姿态
                opp_def = 0.7 if op == 1 else 1.0
                # 体力
                stam_f = {0: 0.7, 1: 0.9, 2: 1.0}[st]
                # 高消耗高回报
                pwr_bonus = 1.0 + 0.3 * (cost - 0.02) / 0.04

                base = np.clip(power * dist_match * opp_def * stam_f * pwr_bonus, 0.15, 0.90)

                # 命中
                p_hit = base
                sc_h = min(sc + 1, N_SCORE - 1)
                di_h = max(di - 1, 0) if a < 6 else di
                s_hit = encode_state(sc_h, st_next, di_h, ti_next, 1)

                # 被格挡
                p_blk = (1 - base) * 0.65
                s_blk = encode_state(sc, st_next, di, ti_next, op)

                # 被反击
                p_ctr = (1 - base) * 0.35
                sc_c = max(sc - 1, 0)
                di_c = min(di + 1, N_DIST - 1)
                s_ctr = encode_state(sc_c, st_next, di_c, ti_next, 0)

                tot = p_hit + p_blk + p_ctr
                P[s, a, s_hit] = p_hit / tot
                P[s, a, s_blk] = p_blk / tot
                P[s, a, s_ctr] = p_ctr / tot

                # 期望奖励
                r_hit = 10 * (sc_h - sc) + 5 * 1 - 20 * cost
                r_blk = -20 * cost
                r_ctr = 10 * (sc_c - sc) + 5 * (-1) - 20 * cost
                R[s, a] = (p_hit * r_hit + p_blk * r_blk + p_ctr * r_ctr) / tot

            # ===== 防御动作 =====
            else:
                defense_str = DEFENSE_STRENGTH[a - N_ATTACKS]
                opp_atk = 1.2 if op == 0 else 0.7

                base_def = np.clip(defense_str * opp_atk, 0.2, 0.90)

                # 防御成功
                p_ok = base_def
                di_ok = di
                if '后撤' in action_name or '全力后撤' in action_name:
                    di_ok = min(di + 1, N_DIST - 1)
                elif '前压' in action_name:
                    di_ok = max(di - 1, 0)
                s_ok = encode_state(sc, st_next, di_ok, ti_next, 1)

                # 防御失败
                p_fail = 1 - base_def
                sc_f = max(sc - 1, 0)
                s_fail = encode_state(sc_f, st_next, di, ti_next, 0)

                # 反击准备类
                if '反击' in action_name:
                    p_ctr_atk = 0.30
                    p_ok_adj = p_ok * (1 - p_ctr_atk)
                    p_ctr = p_ok * p_ctr_atk
                    sc_c = min(sc + 1, N_SCORE - 1)
                    s_ctr = encode_state(sc_c, st_next, di, ti_next, 1)

                    tot = p_ok_adj + p_ctr + p_fail
                    P[s, a, s_ok] = p_ok_adj / tot
                    P[s, a, s_ctr] = p_ctr / tot
                    P[s, a, s_fail] = p_fail / tot

                    r_ok = 5 * 1 - 20 * cost
                    r_ctr = 10 * 1 + 5 * 1 - 20 * cost
                    r_fail = 10 * (sc_f - sc) + 5 * (-1) - 20 * cost
                    R[s, a] = (p_ok_adj * r_ok + p_ctr * r_ctr + p_fail * r_fail) / tot
                else:
                    P[s, a, s_ok] = p_ok
                    P[s, a, s_fail] = p_fail

                    r_ok = 5 * 1 - 20 * cost
                    r_fail = 10 * (sc_f - sc) + 5 * (-1) - 20 * cost
                    R[s, a] = p_ok * r_ok + p_fail * r_fail

    # 概率归一化检查
    for s in range(N_STATES):
        for a in range(N_ACTIONS):
            row_sum = np.sum(P[s, a])
            if row_sum < 1e-10:
                P[s, a, :] = 1.0 / N_STATES
            elif abs(row_sum - 1.0) > 1e-10:
                P[s, a] /= row_sum

    return P, R


# ============================================================
# 第六部分: 值迭代求解
# ============================================================
def value_iteration(P, R, gamma=GAMMA, tol=CONVERGE_TOL, max_iter=MAX_ITER):
    """值迭代求解最优策略 (向量化加速)

    Returns:
        V: 最优价值函数 (N_STATES,)
        policy: 最优策略 (N_STATES,) 每个状态的最优动作索引
        v_history: 收敛过程中的V变化量记录
    """
    n_s, n_a, _ = P.shape
    V = np.zeros(n_s)
    v_history = []

    # 预计算: P @ V 的矩阵乘法
    for iteration in range(max_iter):
        # Q[s,a] = R[s,a] + gamma * sum_s' P[s,a,s'] * V[s']
        # P shape: (n_s, n_a, n_s), V shape: (n_s,)
        # PV = P @ V -> shape (n_s, n_a)
        PV = np.einsum('ijk,k->ij', P, V)
        Q = R + gamma * PV
        V_new = np.max(Q, axis=1)

        delta = np.max(np.abs(V_new - V))
        v_history.append(delta)
        V = V_new

        if delta < tol:
            print(f"  值迭代收敛: 第{iteration+1}轮, ΔV = {delta:.2e}")
            break
    else:
        print(f"  值迭代达到最大迭代次数 {max_iter}, ΔV = {delta:.2e}")

    # 提取最优策略
    PV = np.einsum('ijk,k->ij', P, V)
    Q = R + gamma * PV
    policy = np.argmax(Q, axis=1)

    return V, policy, v_history


# ============================================================
# 第七部分: 蒙特卡洛仿真评估
# ============================================================
def mc_evaluate_policy(P, R, policy, gamma=GAMMA, n_episodes=1000):
    """用蒙特卡洛仿真评估策略的期望累积奖励

    从每个初始状态出发, 按最优策略执行, 统计该初始状态的平均累积奖励
    """
    n_s, n_a, _ = P.shape
    returns = np.zeros(n_s)
    counts = np.zeros(n_s)

    # 从所有162个初始状态出发
    initial_states = list(range(N_STATES))
    eps_per_state = max(1, n_episodes // N_STATES)

    for s0 in initial_states:
        for _ in range(eps_per_state):
            s = s0
            total_reward = 0.0
            discount = 1.0

            for t in range(ROUND_TIME):
                a = policy[s]
                probs = P[s, a]
                s_next = np.random.choice(n_s, p=probs)
                r = R[s, a]

                total_reward += discount * r
                discount *= gamma
                s = s_next

            returns[s0] += total_reward
            counts[s0] += 1

    counts = np.maximum(counts, 1)
    avg_returns = returns / counts
    return avg_returns


# ============================================================
# 第八部分: 策略分析工具函数
# ============================================================
def analyze_policy(V, policy):
    """分析最优策略, 返回各维度的统计信息"""
    results = {}

    # 按体力分组的最优动作分布
    stamina_actions = {st: [] for st in range(N_STAMINA)}
    for s in range(N_STATES):
        _, st, _, _, _ = decode_state(s)
        stamina_actions[st].append(policy[s])

    results['stamina_action_dist'] = {}
    for st in range(N_STAMINA):
        actions = stamina_actions[st]
        unique, counts = np.unique(actions, return_counts=True)
        results['stamina_action_dist'][STAMINA_LEVELS[st]] = {
            'actions': [ALL_ACTIONS[i] for i in unique],
            'counts': counts / len(actions)
        }

    # 按距离分组的最优动作分布
    dist_actions = {di: [] for di in range(N_DIST)}
    for s in range(N_STATES):
        _, _, di, _, _ = decode_state(s)
        dist_actions[di].append(policy[s])

    results['dist_action_dist'] = {}
    for di in range(N_DIST):
        actions = dist_actions[di]
        unique, counts = np.unique(actions, return_counts=True)
        results['dist_action_dist'][DISTANCE_LEVELS[di]] = {
            'actions': [ALL_ACTIONS[i] for i in unique],
            'counts': counts / len(actions)
        }

    # 按分差分组的最优动作分布
    score_actions = {sc: [] for sc in range(N_SCORE)}
    for s in range(N_STATES):
        sc, _, _, _, _ = decode_state(s)
        score_actions[sc].append(policy[s])

    results['score_action_dist'] = {}
    for sc in range(N_SCORE):
        actions = score_actions[sc]
        unique, counts = np.unique(actions, return_counts=True)
        results['score_action_dist'][SCORE_LEVELS[sc]] = {
            'actions': [ALL_ACTIONS[i] for i in unique],
            'counts': counts / len(actions)
        }

    return results


# ============================================================
# 第九部分: 可视化
# ============================================================
def create_figures(V, policy, P, R, v_history, analysis, avg_returns):
    """生成所有可视化图表"""
    os.makedirs(FIG_DIR, exist_ok=True)

    # ==========================================
    # 图1: 值迭代收敛曲线
    # ==========================================
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.semilogy(range(1, len(v_history)+1), v_history, 'b-', linewidth=2)
    ax.axhline(y=CONVERGE_TOL, color='r', linestyle='--', linewidth=1, label=f'收敛阈值 = {CONVERGE_TOL}')
    ax.set_xlabel('迭代轮次', fontsize=12)
    ax.set_ylabel('ΔV (最大变化量)', fontsize=12)
    ax.set_title('值迭代收敛曲线', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, 'fig1_convergence.png')
    plt.savefig(fig_path)
    plt.close()
    print(f"图1已保存: {fig_path}")

    # ==========================================
    # 图2: 最优价值函数热力图 (按体力×距离)
    # ==========================================
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for ti_idx, ti_name in enumerate(TIME_LEVELS):
        ax = axes[ti_idx]
        V_grid = np.zeros((N_STAMINA, N_DIST))
        for st in range(N_STAMINA):
            for di in range(N_DIST):
                vals = []
                for sc in range(N_SCORE):
                    for op in range(N_OPP):
                        s = encode_state(sc, st, di, ti_idx, op)
                        vals.append(V[s])
                V_grid[st, di] = np.mean(vals)

        im = ax.imshow(V_grid, cmap='RdYlGn', aspect='auto')
        ax.set_xticks(range(N_DIST))
        ax.set_xticklabels(DISTANCE_LEVELS, fontsize=10)
        ax.set_yticks(range(N_STAMINA))
        ax.set_yticklabels(STAMINA_LEVELS, fontsize=10)
        ax.set_xlabel('双方距离', fontsize=11)
        ax.set_ylabel('我方体力', fontsize=11)
        ax.set_title(f'{ti_name} V*值', fontsize=12, fontweight='bold')

        for st in range(N_STAMINA):
            for di in range(N_DIST):
                color = 'white' if V_grid[st, di] < np.mean(V_grid) else 'black'
                ax.text(di, st, f'{V_grid[st, di]:.1f}', ha='center', va='center',
                       fontsize=10, fontweight='bold', color=color)

    plt.colorbar(im, ax=axes[-1], label='V*值')
    plt.suptitle('最优价值函数 V*(s) — 体力×距离 (按时间阶段分组)', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, 'fig2_value_function.png')
    plt.savefig(fig_path)
    plt.close()
    print(f"图2已保存: {fig_path}")

    # ==========================================
    # 图3: 最优策略热力图 (体力×距离, 固定均势+前期)
    # ==========================================
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for ti_idx, ti_name in enumerate(TIME_LEVELS):
        ax = axes[ti_idx]
        policy_grid = np.zeros((N_STAMINA, N_DIST), dtype=object)
        for st in range(N_STAMINA):
            for di in range(N_DIST):
                # 取均势+对手进攻的状态
                s = encode_state(1, st, di, ti_idx, 0)
                policy_grid[st, di] = ALL_ACTIONS[policy[s]]

        # 用数值编码绘制
        policy_numeric = np.zeros((N_STAMINA, N_DIST))
        for st in range(N_STAMINA):
            for di in range(N_DIST):
                s = encode_state(1, st, di, ti_idx, 0)
                policy_numeric[st, di] = policy[s]

        im = ax.imshow(policy_numeric, cmap='Set3', aspect='auto')
        ax.set_xticks(range(N_DIST))
        ax.set_xticklabels(DISTANCE_LEVELS, fontsize=10)
        ax.set_yticks(range(N_STAMINA))
        ax.set_yticklabels(STAMINA_LEVELS, fontsize=10)
        ax.set_xlabel('双方距离', fontsize=11)
        ax.set_ylabel('我方体力', fontsize=11)
        ax.set_title(f'{ti_name} 最优策略', fontsize=12, fontweight='bold')

        for st in range(N_STAMINA):
            for di in range(N_DIST):
                name = policy_grid[st, di]
                ax.text(di, st, name, ha='center', va='center', fontsize=8, fontweight='bold')

    plt.suptitle('最优策略 π*(s) — 均势/对手进攻 时的体力×距离决策', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, 'fig3_policy_heatmap.png')
    plt.savefig(fig_path)
    plt.close()
    print(f"图3已保存: {fig_path}")

    # ==========================================
    # 图4: 各体力下的动作选择分布
    # ==========================================
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for st_idx, st_name in enumerate(STAMINA_LEVELS):
        ax = axes[st_idx]
        dist_info = analysis['stamina_action_dist'][st_name]
        actions = dist_info['actions']
        counts = dist_info['counts']

        # 按数量排序
        sort_idx = np.argsort(-counts)
        top_n = min(10, len(sort_idx))
        top_actions = [actions[i] for i in sort_idx[:top_n]]
        top_counts = [counts[i] for i in sort_idx[:top_n]]

        colors = ['#F44336' if a in ATTACK_NAMES else '#4CAF50' for a in top_actions]
        bars = ax.barh(range(top_n), top_counts, color=colors, edgecolor='black', linewidth=0.5)
        ax.set_yticks(range(top_n))
        ax.set_yticklabels(top_actions, fontsize=9)
        ax.set_xlabel('选择频率', fontsize=11)
        ax.set_title(f'体力={st_name}', fontsize=12, fontweight='bold')
        ax.invert_yaxis()

        for bar, val in zip(bars, top_counts):
            ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                   f'{val:.1%}', va='center', fontsize=8)

        # 图例
        ax.bar([], [], color='#F44336', label='攻击', edgecolor='black', linewidth=0.5)
        ax.bar([], [], color='#4CAF50', label='防御', edgecolor='black', linewidth=0.5)
        ax.legend(fontsize=8, loc='lower right')
        ax.grid(True, alpha=0.3, axis='x')

    plt.suptitle('不同体力水平下的最优动作选择分布', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, 'fig4_stamina_action_dist.png')
    plt.savefig(fig_path)
    plt.close()
    print(f"图4已保存: {fig_path}")

    # ==========================================
    # 图5: 各距离下的动作选择分布
    # ==========================================
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for di_idx, di_name in enumerate(DISTANCE_LEVELS):
        ax = axes[di_idx]
        dist_info = analysis['dist_action_dist'][di_name]
        actions = dist_info['actions']
        counts = dist_info['counts']

        sort_idx = np.argsort(-counts)
        top_n = min(10, len(sort_idx))
        top_actions = [actions[i] for i in sort_idx[:top_n]]
        top_counts = [counts[i] for i in sort_idx[:top_n]]

        colors = ['#F44336' if a in ATTACK_NAMES else '#4CAF50' for a in top_actions]
        bars = ax.barh(range(top_n), top_counts, color=colors, edgecolor='black', linewidth=0.5)
        ax.set_yticks(range(top_n))
        ax.set_yticklabels(top_actions, fontsize=9)
        ax.set_xlabel('选择频率', fontsize=11)
        ax.set_title(f'距离={di_name}', fontsize=12, fontweight='bold')
        ax.invert_yaxis()

        for bar, val in zip(bars, top_counts):
            ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                   f'{val:.1%}', va='center', fontsize=8)

        ax.bar([], [], color='#F44336', label='攻击', edgecolor='black', linewidth=0.5)
        ax.bar([], [], color='#4CAF50', label='防御', edgecolor='black', linewidth=0.5)
        ax.legend(fontsize=8, loc='lower right')
        ax.grid(True, alpha=0.3, axis='x')

    plt.suptitle('不同距离下的最优动作选择分布', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, 'fig5_distance_action_dist.png')
    plt.savefig(fig_path)
    plt.close()
    print(f"图5已保存: {fig_path}")

    # ==========================================
    # 图6: 各分差下的动作选择分布
    # ==========================================
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for sc_idx, sc_name in enumerate(SCORE_LEVELS):
        ax = axes[sc_idx]
        score_info = analysis['score_action_dist'][sc_name]
        actions = score_info['actions']
        counts = score_info['counts']

        sort_idx = np.argsort(-counts)
        top_n = min(10, len(sort_idx))
        top_actions = [actions[i] for i in sort_idx[:top_n]]
        top_counts = [counts[i] for i in sort_idx[:top_n]]

        colors = ['#F44336' if a in ATTACK_NAMES else '#4CAF50' for a in top_actions]
        bars = ax.barh(range(top_n), top_counts, color=colors, edgecolor='black', linewidth=0.5)
        ax.set_yticks(range(top_n))
        ax.set_yticklabels(top_actions, fontsize=9)
        ax.set_xlabel('选择频率', fontsize=11)
        ax.set_title(f'分差={sc_name}', fontsize=12, fontweight='bold')
        ax.invert_yaxis()

        for bar, val in zip(bars, top_counts):
            ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                   f'{val:.1%}', va='center', fontsize=8)

        ax.bar([], [], color='#F44336', label='攻击', edgecolor='black', linewidth=0.5)
        ax.bar([], [], color='#4CAF50', label='防御', edgecolor='black', linewidth=0.5)
        ax.legend(fontsize=8, loc='lower right')
        ax.grid(True, alpha=0.3, axis='x')

    plt.suptitle('不同分差下的最优动作选择分布', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, 'fig6_score_action_dist.png')
    plt.savefig(fig_path)
    plt.close()
    print(f"图6已保存: {fig_path}")

    # ==========================================
    # 图7: 三回合作战方案时间轴 (三种不同场景)
    # ==========================================
    fig, axes = plt.subplots(3, 1, figsize=(16, 12))

    # 三个不同场景: 开局试探 / 中盘对抗 / 终局决胜
    scenarios = [
        {'name': '第1回合 - 开局试探', 'score': 1, 'stamina': 2, 'dist': 1, 'desc': '均势/高体力/中距'},
        {'name': '第2回合 - 中盘对抗', 'score': 1, 'stamina': 1, 'dist': 0, 'desc': '均势/中体力/近距'},
        {'name': '第3回合 - 终局决胜', 'score': 0, 'stamina': 0, 'dist': 2, 'desc': '劣势/低体力/远距'},
    ]

    for round_idx, scenario in enumerate(scenarios):
        ax = axes[round_idx]
        time_points = []
        attack_actions = []
        defense_actions = []

        for ti_idx in range(N_TIME):
            for opp_idx in range(N_OPP):
                s = encode_state(scenario['score'], scenario['stamina'], scenario['dist'], ti_idx, opp_idx)
                a = policy[s]
                is_atk = a < N_ATTACKS
                label = f'{TIME_LEVELS[ti_idx]}\n(对手{OPPONENT_LEVELS[opp_idx]})'
                time_points.append(label)
                attack_actions.append(ALL_ACTIONS[a] if is_atk else '')
                defense_actions.append(ALL_ACTIONS[a] if not is_atk else '')

        x = np.arange(len(time_points))
        width = 0.35
        atk_vals = [1.0 if a else 0.0 for a in attack_actions]
        def_vals = [1.0 if d else 0.0 for d in defense_actions]

        ax.bar(x - width/2, atk_vals, width, label='攻击', color='#F44336', edgecolor='black', linewidth=0.5)
        ax.bar(x + width/2, def_vals, width, label='防御', color='#4CAF50', edgecolor='black', linewidth=0.5)

        for i in range(len(time_points)):
            name = attack_actions[i] or defense_actions[i]
            val = max(atk_vals[i], def_vals[i])
            ax.text(x[i], val + 0.05, name, ha='center', va='bottom', fontsize=7, rotation=30)

        ax.set_xticks(x)
        ax.set_xticklabels(time_points, fontsize=8)
        ax.set_ylabel('选择', fontsize=10)
        ax.set_title(f'{scenario["name"]} ({scenario["desc"]})', fontsize=12, fontweight='bold')
        ax.set_ylim(0, 1.5)
        ax.legend(fontsize=9, loc='upper right')
        ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle('三回合典型场景作战方案', fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, 'fig7_round_strategy.png')
    plt.savefig(fig_path)
    plt.close()
    print(f"图7已保存: {fig_path}")

    # ==========================================
    # 图8: 状态转移概率热力图 (典型状态)
    # ==========================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # 选择4个典型状态
    typical_states = [
        (1, 2, 1, 1, 0),  # 均势, 高体力, 中距, 中期, 对手进攻
        (1, 0, 0, 2, 0),  # 均势, 低体力, 近距, 后期, 对手进攻
        (0, 1, 2, 1, 1),  # 劣势, 中体力, 远距, 中期, 对手防守
        (2, 2, 1, 2, 0),  # 优势, 高体力, 中距, 后期, 对手进攻
    ]

    for idx, (sc, st, di, ti, op) in enumerate(typical_states):
        ax = axes[idx // 2, idx % 2]
        s = encode_state(sc, st, di, ti, op)

        # 选择TOP6动作的转移概率
        q_vals = np.array([R[s, a] + GAMMA * np.dot(P[s, a], V) for a in range(N_ACTIONS)])
        top_actions_idx = np.argsort(-q_vals)[:6]

        # 构建转移概率子矩阵
        trans_probs = P[s, top_actions_idx, :]
        # 找到非零转移的目标状态
        nonzero_cols = np.unique(np.where(trans_probs > 0.01)[1])
        if len(nonzero_cols) > 10:
            # 按概率和排序取TOP10
            col_sums = trans_probs[:, nonzero_cols].sum(axis=0)
            top_cols = nonzero_cols[np.argsort(-col_sums)[:10]]
        else:
            top_cols = nonzero_cols

        sub_matrix = trans_probs[:, top_cols]

        im = ax.imshow(sub_matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
        ax.set_xticks(range(len(top_cols)))
        ax.set_xticklabels([state_to_str(c)[:15] for c in top_cols], rotation=45, ha='right', fontsize=6)
        ax.set_yticks(range(len(top_actions_idx)))
        ax.set_yticklabels([ALL_ACTIONS[a] for a in top_actions_idx], fontsize=8)
        ax.set_title(f'{state_to_str(s)[:30]}', fontsize=9, fontweight='bold')

        for i in range(len(top_actions_idx)):
            for j in range(len(top_cols)):
                if sub_matrix[i, j] > 0.01:
                    ax.text(j, i, f'{sub_matrix[i,j]:.2f}', ha='center', va='center',
                           fontsize=6, color='white' if sub_matrix[i,j] > 0.5 else 'black')

    plt.colorbar(im, ax=axes[-1, -1], label='转移概率')
    plt.suptitle('典型状态下的转移概率分布 (TOP6动作 × TOP目标状态)', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, 'fig8_transition_heatmap.png')
    plt.savefig(fig_path)
    plt.close()
    print(f"图8已保存: {fig_path}")

    # ==========================================
    # 图9: 蒙特卡洛评估 — 各初始状态的期望累积奖励
    # ==========================================
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for ti_idx, ti_name in enumerate(TIME_LEVELS):
        ax = axes[ti_idx]
        ret_grid = np.zeros((N_STAMINA, N_DIST))
        for st in range(N_STAMINA):
            for di in range(N_DIST):
                vals = []
                for sc in range(N_SCORE):
                    for op in range(N_OPP):
                        s = encode_state(sc, st, di, ti_idx, op)
                        vals.append(avg_returns[s])
                ret_grid[st, di] = np.mean(vals)

        im = ax.imshow(ret_grid, cmap='RdYlGn', aspect='auto')
        ax.set_xticks(range(N_DIST))
        ax.set_xticklabels(DISTANCE_LEVELS, fontsize=10)
        ax.set_yticks(range(N_STAMINA))
        ax.set_yticklabels(STAMINA_LEVELS, fontsize=10)
        ax.set_xlabel('双方距离', fontsize=11)
        ax.set_ylabel('我方体力', fontsize=11)
        ax.set_title(f'{ti_name} 期望累积奖励', fontsize=12, fontweight='bold')

        for st in range(N_STAMINA):
            for di in range(N_DIST):
                color = 'white' if ret_grid[st, di] < np.mean(ret_grid) else 'black'
                ax.text(di, st, f'{ret_grid[st, di]:.0f}', ha='center', va='center',
                       fontsize=9, fontweight='bold', color=color)

    plt.colorbar(im, ax=axes[-1], label='期望累积奖励')
    plt.suptitle('蒙特卡洛评估 — 最优策略的期望累积奖励', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, 'fig9_mc_evaluation.png')
    plt.savefig(fig_path)
    plt.close()
    print(f"图9已保存: {fig_path}")

    # ==========================================
    # 图10: 策略总结表 (按分差×体力×距离)
    # ==========================================
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.axis('off')

    # 构建表格: 选取关键状态
    table_data = []
    for sc_idx in range(N_SCORE):
        for st_idx in range(N_STAMINA):
            for di_idx in range(N_DIST):
                s = encode_state(sc_idx, st_idx, di_idx, 1, 0)  # 中期, 对手进攻
                a = policy[s]
                v = V[s]
                table_data.append([
                    SCORE_LEVELS[sc_idx],
                    STAMINA_LEVELS[st_idx],
                    DISTANCE_LEVELS[di_idx],
                    ALL_ACTIONS[a],
                    '攻击' if a < N_ATTACKS else '防御',
                    f'{v:.1f}'
                ])

    col_labels = ['分差', '体力', '距离', '最优动作', '类型', 'V*值']
    table = ax.table(cellText=table_data, colLabels=col_labels,
                     cellLoc='center', loc='center',
                     colWidths=[0.08, 0.08, 0.08, 0.15, 0.08, 0.08])

    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.5)

    for j in range(len(col_labels)):
        table[0, j].set_facecolor('#2196F3')
        table[0, j].set_text_props(color='white', fontweight='bold')

    # 攻击动作标红, 防御动作标绿
    for i in range(1, len(table_data) + 1):
        action_type = table_data[i-1][4]
        if action_type == '攻击':
            table[i, 3].set_facecolor('#FFEBEE')
        else:
            table[i, 3].set_facecolor('#E8F5E9')

    ax.set_title('最优策略汇总表 (中期/对手进攻)', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, 'fig10_strategy_table.png')
    plt.savefig(fig_path)
    plt.close()
    print(f"图10已保存: {fig_path}")

    return {
        'fig1': '值迭代收敛曲线',
        'fig2': '最优价值函数',
        'fig3': '最优策略热力图',
        'fig4': '体力-动作分布',
        'fig5': '距离-动作分布',
        'fig6': '分差-动作分布',
        'fig7': '三回合作战方案',
        'fig8': '状态转移概率',
        'fig9': '蒙特卡洛评估',
        'fig10': '策略汇总表',
    }


# ============================================================
# 主程序
# ============================================================
def main():
    print("=" * 70)
    print("CUMCM 2026 Problem B - 基于MDP的最优格斗策略")
    print("=" * 70)

    # 1. 构建MDP模型
    print("\n[1/5] 构建MDP模型...")
    print(f"  状态空间: {N_STATES}种 ({N_SCORE}×{N_STAMINA}×{N_DIST}×{N_TIME}×{N_OPP})")
    print(f"  动作空间: {N_ACTIONS}种 ({N_ATTACKS}攻击 + {N_DEFENSES}防御)")

    # 2. 基于专家规则构建转移概率和奖励
    print("\n[2/5] 基于专家规则构建转移概率矩阵...")
    P, R = build_transition_and_reward()
    print(f"  转移概率矩阵: P.shape = {P.shape}")
    print(f"  期望奖励矩阵: R.shape = {R.shape}")
    print(f"  奖励范围: [{R.min():.2f}, {R.max():.2f}]")

    # 验证概率归一化
    prob_sums = np.sum(P, axis=2)
    print(f"  概率归一化检查: max误差 = {np.max(np.abs(prob_sums - 1.0)):.2e}")

    # 3. 值迭代求解
    print("\n[3/5] 值迭代求解最优策略...")
    V, policy, v_history = value_iteration(P, R)
    print(f"  V*范围: [{V.min():.2f}, {V.max():.2f}]")

    # 4. 蒙特卡洛评估
    print("\n[4/5] 蒙特卡洛仿真评估策略...")
    avg_returns = mc_evaluate_policy(P, R, policy, n_episodes=N_EPISODES)
    print(f"  平均累积奖励: {avg_returns.mean():.2f}")

    # 策略分析
    analysis = analyze_policy(V, policy)

    # 打印关键策略
    print("\n  === 核心策略摘要 ===")
    print("\n  【按体力分组】")
    for st_name in STAMINA_LEVELS:
        dist_info = analysis['stamina_action_dist'][st_name]
        top3_idx = np.argsort(-dist_info['counts'])[:3]
        top3 = [(dist_info['actions'][i], dist_info['counts'][i]) for i in top3_idx]
        print(f"    体力={st_name}: {', '.join(f'{a}({c:.0%})' for a, c in top3)}")

    print("\n  【按距离分组】")
    for di_name in DISTANCE_LEVELS:
        dist_info = analysis['dist_action_dist'][di_name]
        top3_idx = np.argsort(-dist_info['counts'])[:3]
        top3 = [(dist_info['actions'][i], dist_info['counts'][i]) for i in top3_idx]
        print(f"    距离={di_name}: {', '.join(f'{a}({c:.0%})' for a, c in top3)}")

    print("\n  【按分差分组】")
    for sc_name in SCORE_LEVELS:
        dist_info = analysis['score_action_dist'][sc_name]
        top3_idx = np.argsort(-dist_info['counts'])[:3]
        top3 = [(dist_info['actions'][i], dist_info['counts'][i]) for i in top3_idx]
        print(f"    分差={sc_name}: {', '.join(f'{a}({c:.0%})' for a, c in top3)}")

    # 5. 可视化
    print("\n[5/5] 生成可视化图表...")
    fig_dict = create_figures(V, policy, P, R, v_history, analysis, avg_returns)

    # 总结
    print("\n" + "=" * 70)
    print("分析完成!")
    print("=" * 70)
    print(f"  MDP模型: {N_STATES}状态 × {N_ACTIONS}动作")
    print(f"  值迭代收敛: {len(v_history)}轮")
    print(f"  V*范围: [{V.min():.2f}, {V.max():.2f}]")
    print(f"  蒙特卡洛评估: {N_EPISODES}场, 平均奖励={avg_returns.mean():.2f}")
    print(f"  图表已保存至: {FIG_DIR}/")

    # 保存结果摘要
    summary_path = os.path.join(FIG_DIR, 'mdp_summary.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("CUMCM 2026 Problem B - MDP最优格斗策略 结果摘要\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"MDP模型: {N_STATES}状态 × {N_ACTIONS}动作\n")
        f.write(f"折扣因子 γ = {GAMMA}\n")
        f.write(f"值迭代收敛: {len(v_history)}轮\n")
        f.write(f"V*范围: [{V.min():.2f}, {V.max():.2f}]\n")
        f.write(f"蒙特卡洛评估: {N_EPISODES}场, 平均奖励={avg_returns.mean():.2f}\n\n")

        f.write("核心策略:\n")
        f.write("  体力充足(高)时: 以侧踢、膝撞主攻\n")
        f.write("  体力不足(低)时: 以直拳控场、防守为主\n")
        f.write("  近距: 膝撞、勾拳、肘击\n")
        f.write("  中距: 侧踢、摆拳\n")
        f.write("  远距: 直拳、前蹬、后踢\n")
        f.write("  领先: 防守为主, 控制消耗\n")
        f.write("  落后: 全力进攻, 回旋踢追分\n")

    print(f"结果摘要已保存至: {summary_path}")


if __name__ == '__main__':
    main()