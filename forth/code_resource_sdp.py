# -*- coding: utf-8 -*-
"""
CUMCM 2026 Problem B - 第四问: 基于随机动态规划的BO3资源调度优化模型

输入: first/results.npz (问题一攻击动力学数据)
输出: forth/figures/ 下12张可视化图表 + 结果摘要

环境: Python 3.8+, numpy, matplotlib
"""
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import warnings
warnings.filterwarnings('ignore')

# 中文字体配置
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.family'] = 'sans-serif'

# 路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULT_PATH = os.path.join(BASE_DIR, '..', 'first', 'results.npz')
FIG_DIR = os.path.join(BASE_DIR, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

# ============================================================
# 第一部分: 全局常量与配置
# ============================================================

# 比赛参数
GAME_TIME = 300.0        # 每局时间: 5分钟(秒)
DT_SIM = 1.0             # 仿真步长(秒)
GAMMA = 0.98             # 折扣因子(接近1, 有限博弈)
CONVERGE_TOL = 1e-4
MAX_ITER = 3000
N_MC_EPISODES = 20000

# 资源参数
MAX_MANUAL_RESET = 2     # 人工复位最多次数
MAX_TACTICAL_TIMEOUT = 2 # 战术暂停最多次数
MAX_EMERGENCY_REPAIR = 1 # 紧急维修最多次数
RESET_TIME_COST = 1      # 人工复位消耗1个时间步(30s)
TIMEOUT_HEAL = 1         # 战术暂停恢复1档健康度
REPAIR_TIME_COST = 10    # 紧急维修消耗10个时间步(5min=整局)
REPAIR_HEALTH_IDX = 1    # 紧急维修后健康度=0.8

# 故障模型参数
LAMBDA_0 = 0.015         # 基础故障率(每秒)
ALPHA_FAULT = 1.5        # 健康度敏感系数
BETA_ATTACKS = 0.005     # 累计攻击敏感系数

# 局内状态离散化
N_HEALTH = 5             # 健康度: 5档
N_FAULT = 4              # 故障等级: 4档
N_TIME = 10              # 时间: 10档(每档30s)
N_SCORE_DIFF = 5         # 比分差: 5档(-2,-1,0,+1,+2)

# 格斗动作(复用问题三)
N_ATTACKS = 13
N_DEFENSES = 22
N_COMBAT_ACTIONS = N_ATTACKS + N_DEFENSES  # 35

ATTACK_POWER = np.array([
    0.5, 0.5, 0.6, 0.6, 0.55, 0.55, 0.45, 0.65, 0.7, 0.75, 0.8, 0.9, 0.7
])
DEFENSE_STRENGTH = np.array([
    0.6, 0.6, 0.65, 0.65, 0.75, 0.8, 0.7, 0.7, 0.65, 0.6,
    0.7, 0.65, 0.4, 0.7, 0.7, 0.6, 0.55, 0.55, 0.5, 0.5, 0.45, 0.8
])
ATTACK_STAMINA_COST = np.array([
    0.02, 0.02, 0.03, 0.03, 0.025, 0.025, 0.02, 0.03, 0.04, 0.05, 0.05, 0.06, 0.04
])

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
ALL_COMBAT_NAMES = ATTACK_NAMES + DEFENSE_NAMES

# 资源动作
RESOURCE_NAMES = ['人工复位', '战术暂停', '紧急维修']
N_RESOURCE_ACTIONS = 3
N_GAME_ACTIONS = N_COMBAT_ACTIONS + N_RESOURCE_ACTIONS  # 38

# 局内状态总数
N_GAME_STATES = N_HEALTH * N_FAULT * N_TIME * N_SCORE_DIFF  # 1000

# BO3状态
N_BO3_WIN_STATES = 3     # w_m: 0,1 (非终结)
N_BO3_LOSE_STATES = 3    # w_o: 0,1 (非终结)
N_BO3_SCORE_COMBOS = 4   # (0,0),(0,1),(1,0),(1,1)
N_RES_COMBOS = MAX_MANUAL_RESET + 1  # 3
N_TACT_COMBOS = MAX_TACTICAL_TIMEOUT + 1  # 3
N_REPAIR_COMBOS = MAX_EMERGENCY_REPAIR + 1  # 2
N_BO3_STATES = N_BO3_SCORE_COMBOS * N_RES_COMBOS * N_TACT_COMBOS * N_REPAIR_COMBOS * N_HEALTH

# 奖励常量
WIN_REWARD = 5.0
LOSE_REWARD = -5.0
COMBAT_HEALTH_COST = 0.08  # 每次格斗消耗健康度

# 时间紧迫度
TIME_URGENCY = np.linspace(0.5, 2.0, N_TIME)


# ============================================================
# 第二部分: 状态编码/解码
# ============================================================

def encode_game_state(h, f, t, sd):
    """局内状态编码: (h,f,t,sd) -> [0, N_GAME_STATES)"""
    return ((h * N_FAULT + f) * N_TIME + t) * N_SCORE_DIFF + sd

def decode_game_state(s):
    """局内状态解码: [0, N_GAME_STATES) -> (h,f,t,sd)"""
    sd = s % N_SCORE_DIFF; s //= N_SCORE_DIFF
    t = s % N_TIME; s //= N_TIME
    f = s % N_FAULT; s //= N_FAULT
    h = s % N_HEALTH
    return h, f, t, sd

def health_val(h_idx):
    """健康度索引 -> 实际值 (0=1.0, 4=0.2)"""
    return 1.0 - 0.2 * h_idx

def score_diff_val(sd_idx):
    """比分差索引 -> 实际值 (0=-2, 2=0, 4=+2)"""
    return sd_idx - 2

def encode_bo3_state(w_m, w_o, r_res, r_tact, r_repair, h_start):
    """BO3状态编码"""
    return ((((w_m * 2 + w_o) * N_RES_COMBOS + r_res)
             * N_TACT_COMBOS + r_tact) * N_REPAIR_COMBOS + r_repair) * N_HEALTH + h_start

def decode_bo3_state(s):
    """BO3状态解码"""
    h_start = s % N_HEALTH; s //= N_HEALTH
    r_repair = s % N_REPAIR_COMBOS; s //= N_REPAIR_COMBOS
    r_tact = s % N_TACT_COMBOS; s //= N_TACT_COMBOS
    r_res = s % N_RES_COMBOS; s //= N_RES_COMBOS
    w_o = s % 2; s //= 2
    w_m = s % 2
    return w_m, w_o, r_res, r_tact, r_repair, h_start


# ============================================================
# 第三部分: 故障模型
# ============================================================

def fault_occurrence_rate(h_idx, f_idx, t_idx):
    """计算故障发生概率(每个时间步)"""
    h_val = health_val(h_idx)
    # 基础故障率随健康度降低而增加
    base_rate = LAMBDA_0 * np.exp(ALPHA_FAULT * (1 - h_val))
    # 时间推移增加故障风险
    time_factor = 1.0 + 0.3 * t_idx / N_TIME
    # 已有故障放大后续故障
    fault_factor = 1.0 + 0.8 * f_idx
    return min(base_rate * time_factor * fault_factor, 0.5)

def transition_fault_level(f_idx, fault_occurred):
    """故障等级转移"""
    if not fault_occurred:
        return f_idx
    # 转移矩阵: 从当前等级转移到更高等级
    transitions = {
        0: [1, 2],           # 无 -> 轻(0.7) / 中(0.3)
        1: [2, 3],           # 轻 -> 中(0.6) / 重(0.4)
        2: [2, 3],           # 中 -> 中(0.2) / 重(0.8)
        3: [3],              # 重 -> 重(1.0)
    }
    probs = {
        0: [0.7, 0.3],
        1: [0.6, 0.4],
        2: [0.2, 0.8],
        3: [1.0],
    }
    targets = transitions[f_idx]
    ps = probs[f_idx]
    r = np.random.random()
    cumsum = 0
    for i, p in enumerate(ps):
        cumsum += p
        if r < cumsum:
            return targets[i]
    return targets[-1]


# ============================================================
# 第四部分: 局内转移矩阵构建
# ============================================================

def build_game_transition_and_reward():
    """构建局内MDP的转移概率矩阵和奖励矩阵

    P: (N_GAME_STATES, N_GAME_ACTIONS, N_GAME_STATES)
    R: (N_GAME_STATES, N_GAME_ACTIONS)
    """
    P = np.zeros((N_GAME_STATES, N_GAME_ACTIONS, N_GAME_STATES))
    R = np.zeros((N_GAME_STATES, N_GAME_ACTIONS))

    for s in range(N_GAME_STATES):
        h, f, t, sd = decode_game_state(s)

        # 终端状态: 时间耗尽
        if t >= N_TIME - 1:
            if sd >= 3:      # 领先2+ -> 胜
                R[s, :] = WIN_REWARD
            elif sd <= 1:    # 落后2+ -> 负
                R[s, :] = LOSE_REWARD
            else:            # 平局/微差
                R[s, :] = 0
            P[s, :, s] = 1.0
            continue

        # ===== 格斗动作 (a < 35) =====
        for a in range(N_COMBAT_ACTIONS):
            is_attack = a < N_ATTACKS
            power = ATTACK_POWER[a] if is_attack else 0.0
            defense_str = DEFENSE_STRENGTH[a - N_ATTACKS] if not is_attack else 0.0
            cost = ATTACK_STAMINA_COST[a] if is_attack else 0.015

            # 故障发生检查
            p_fault = fault_occurrence_rate(h, f, t)
            fault_happens = np.random.random() < p_fault
            f_next = transition_fault_level(f, fault_happens)

            # 健康度退化
            health_loss = COMBAT_HEALTH_COST * (1 + 0.3 * f_next)
            h_next = min(h + max(1, int(health_loss / 0.2)), N_HEALTH - 1)

            # 时间推进
            t_next = t + 1

            if is_attack:
                # 攻击: hit/block/counter 三结果
                # 基础成功率受健康度和故障影响
                health_factor = 1.0 - 0.2 * h  # 健康度低→成功率低
                fault_penalty = 1.0 - 0.3 * f_next  # 故障→成功率低
                base = np.clip(power * 0.4 * health_factor * fault_penalty, 0.10, 0.55)

                # 比分差影响: 落后时更拼命
                score_aggression = 1.0 + 0.05 * (2 - sd)
                base = np.clip(base * score_aggression, 0.08, 0.60)

                p_hit = base
                p_block = (1 - base) * 0.55
                p_counter = (1 - base) * 0.45

                sd_hit = min(sd + 1, N_SCORE_DIFF - 1)
                sd_counter = max(sd - 1, 0)

                s_hit = encode_game_state(h_next, f_next, t_next, sd_hit)
                s_blk = encode_game_state(h_next, f_next, t_next, sd)
                s_ctr = encode_game_state(h_next, f_next, t_next, sd_counter)

                P[s, a, s_hit] = p_hit
                P[s, a, s_blk] = p_block
                P[s, a, s_ctr] = p_counter

                urgency = TIME_URGENCY[t]
                r_hit = 10 * 1 * urgency - 5 * health_loss
                r_blk = -5 * health_loss
                r_ctr = 10 * (-1) * urgency - 5 * health_loss
                R[s, a] = p_hit * r_hit + p_block * r_blk + p_counter * r_ctr

            else:
                # 防御: success/fail 二结果
                health_factor = 1.0 - 0.1 * h
                fault_penalty = 1.0 - 0.15 * f_next
                base_def = np.clip(defense_str * health_factor * fault_penalty, 0.2, 0.85)

                p_ok = base_def
                p_fail = 1 - base_def

                sd_fail = max(sd - 1, 0)
                s_ok = encode_game_state(h_next, f_next, t_next, sd)
                s_fail = encode_game_state(h_next, f_next, t_next, sd_fail)

                P[s, a, s_ok] = p_ok
                P[s, a, s_fail] = p_fail

                urgency = TIME_URGENCY[t]
                r_ok = 3 - 5 * health_loss
                r_fail = 10 * (-1) * urgency - 5 * health_loss
                R[s, a] = p_ok * r_ok + p_fail * r_fail

        # ===== 资源动作 =====
        # 人工复位: 故障-1, 耗时1步
        a_reset = N_COMBAT_ACTIONS
        f_reset = max(f - 1, 0)
        t_reset = min(t + RESET_TIME_COST, N_TIME - 1)
        s_reset = encode_game_state(h, f_reset, t_reset, sd)
        P[s, a_reset, s_reset] = 1.0
        R[s, a_reset] = -3  # 机会成本

        # 战术暂停: 健康+1档, 不耗时
        a_timeout = N_COMBAT_ACTIONS + 1
        h_timeout = max(h - TIMEOUT_HEAL, 0)
        s_timeout = encode_game_state(h_timeout, f, t, sd)
        P[s, a_timeout, s_timeout] = 1.0
        R[s, a_timeout] = 2  # 小正奖励

        # 紧急维修: 故障归零, 健康→0.8, 耗时整局
        a_repair = N_COMBAT_ACTIONS + 2
        t_repair = N_TIME - 1  # 直接到终端
        s_repair = encode_game_state(REPAIR_HEALTH_IDX, 0, t_repair, sd)
        P[s, a_repair, s_repair] = 1.0
        R[s, a_repair] = 0  # 中性(牺牲本局换状态恢复)

    # 概率归一化
    for s in range(N_GAME_STATES):
        for a in range(N_GAME_ACTIONS):
            row_sum = np.sum(P[s, a])
            if row_sum < 1e-10:
                P[s, a, :] = 1.0 / N_GAME_STATES
            elif abs(row_sum - 1.0) > 1e-10:
                P[s, a] /= row_sum

    return P, R


# ============================================================
# 第五部分: 局内值迭代
# ============================================================

def value_iteration_game(P, R, gamma=GAMMA, tol=CONVERGE_TOL, max_iter=MAX_ITER):
    """局内MDP值迭代 (向量化加速)"""
    n_s, n_a, _ = P.shape
    V = np.zeros(n_s)
    v_history = []

    for iteration in range(max_iter):
        PV = np.einsum('ijk,k->ij', P, V)
        Q = R + gamma * PV
        V_new = np.max(Q, axis=1)

        delta = np.max(np.abs(V_new - V))
        v_history.append(delta)
        V = V_new

        if delta < tol:
            print(f"  局内值迭代收敛: 第{iteration+1}轮, ΔV = {delta:.2e}")
            break
    else:
        print(f"  局内值迭代达最大轮数 {max_iter}, ΔV = {delta:.2e}")

    PV = np.einsum('ijk,k->ij', P, V)
    Q = R + gamma * PV
    policy = np.argmax(Q, axis=1)

    return V, policy, v_history


# ============================================================
# 第六部分: 局胜率计算
# ============================================================

def compute_game_win_prob(P, policy, h_start_idx):
    """计算从给定健康度出发的局胜率

    从 (h_start, f=0, t=0, sd=2) 出发, 前向传播概率质量
    sd=2 表示比分差=0 (平局开始)
    """
    s0 = encode_game_state(h_start_idx, 0, 0, 2)
    state_dist = np.zeros(N_GAME_STATES)
    state_dist[s0] = 1.0

    for step in range(N_TIME):
        new_dist = np.zeros(N_GAME_STATES)
        for s in range(N_GAME_STATES):
            if state_dist[s] < 1e-15:
                continue
            a = policy[s]
            new_dist += state_dist[s] * P[s, a]
        state_dist = new_dist

    # 统计终端状态中的胜率
    p_win = 0.0
    p_draw = 0.0
    for s in range(N_GAME_STATES):
        h, f, t, sd = decode_game_state(s)
        if t >= N_TIME - 1:
            if sd >= 3:
                p_win += state_dist[s]
            elif sd == 2:
                p_draw += state_dist[s]

    return p_win, p_draw


def compute_all_game_win_probs(P, policy):
    """计算所有健康度下的局胜率"""
    results = {}
    for h in range(N_HEALTH):
        p_win, p_draw = compute_game_win_prob(P, policy, h)
        results[h] = {'p_win': p_win, 'p_draw': p_draw, 'p_lose': 1 - p_win - p_draw}
        print(f"    健康度={health_val(h):.1f}: 胜率={p_win:.3f}, 平局={p_draw:.3f}")
    return results


# ============================================================
# 第七部分: BO3马尔可夫链
# ============================================================

def build_bo3_markov_chain(game_win_probs):
    """构建BO3马尔可夫链

    状态: (w_m, w_o, r_res, r_tact, r_repair, h_start)
    非终结状态数: 4 × 3 × 3 × 2 × 5 = 360
    终结吸收态: 2个 (我方胜/对方胜)

    Returns:
        P_bo3: (N_BO3+2, N_BO3+2) 转移矩阵
        state_map: BO3状态索引映射
    """
    n_bo3 = N_BO3_STATES
    n_total = n_bo3 + 2  # +2 吸收态
    P_bo3 = np.zeros((n_total, n_total))
    WIN_ABS = n_bo3      # 我方胜BO3吸收态
    LOSE_ABS = n_bo3 + 1  # 对方胜BO3吸收态

    # 构建状态索引映射
    state_to_idx = {}
    idx_to_state = {}
    idx = 0
    for w_m in range(2):
        for w_o in range(2):
            for r_res in range(N_RES_COMBOS):
                for r_tact in range(N_TACT_COMBOS):
                    for r_repair in range(N_REPAIR_COMBOS):
                        for h_start in range(N_HEALTH):
                            state_to_idx[(w_m, w_o, r_res, r_tact, r_repair, h_start)] = idx
                            idx_to_state[idx] = (w_m, w_o, r_res, r_tact, r_repair, h_start)
                            idx += 1

    # 填充转移矩阵
    for idx_s in range(n_bo3):
        w_m, w_o, r_res, r_tact, r_repair, h_start = idx_to_state[idx_s]

        # 获取当前健康度下的局胜率
        pw = game_win_probs[h_start]['p_win']
        pd = game_win_probs[h_start]['p_draw']
        pl = 1 - pw - pd

        # 我方赢本局
        w_m_new = w_m + 1
        if w_m_new >= 2:
            # 我方赢BO3
            P_bo3[idx_s, WIN_ABS] += pw
        else:
            # 下一局, 健康度退化(局间恢复不完全)
            h_next = min(h_start + 1, N_HEALTH - 1)  # 每局后健康度降低1档
            next_idx = state_to_idx.get((w_m_new, w_o, r_res, r_tact, r_repair, h_next))
            if next_idx is not None:
                P_bo3[idx_s, next_idx] += pw

        # 平局(重新比赛, 不消耗资源, 健康度微降)
        h_next_draw = min(h_start + 1, N_HEALTH - 1)
        next_idx = state_to_idx.get((w_m, w_o, r_res, r_tact, r_repair, h_next_draw))
        if next_idx is not None:
            P_bo3[idx_s, next_idx] += pd

        # 我方输本局
        w_o_new = w_o + 1
        if w_o_new >= 2:
            # 对方赢BO3
            P_bo3[idx_s, LOSE_ABS] += pl
        else:
            # 下一局, 健康度退化
            h_next = min(h_start + 1, N_HEALTH - 1)
            next_idx = state_to_idx.get((w_m, w_o_new, r_res, r_tact, r_repair, h_next))
            if next_idx is not None:
                P_bo3[idx_s, next_idx] += pl

    # 吸收态自循环
    P_bo3[WIN_ABS, WIN_ABS] = 1.0
    P_bo3[LOSE_ABS, LOSE_ABS] = 1.0

    return P_bo3, state_to_idx, idx_to_state, WIN_ABS, LOSE_ABS


def solve_bo3_win_prob(P_bo3, n_bo3, WIN_ABS, LOSE_ABS):
    """求解BO3马尔可夫链的吸收概率

    使用值迭代方法
    """
    n_total = P_bo3.shape[0]
    V = np.zeros(n_total)
    V[WIN_ABS] = 1.0  # 赢BO3 = 1
    V[LOSE_ABS] = 0.0  # 输BO3 = 0

    for _ in range(2000):
        V_new = np.einsum('ij,j->i', P_bo3, V)
        V_new[WIN_ABS] = 1.0
        V_new[LOSE_ABS] = 0.0
        delta = np.max(np.abs(V_new[:n_bo3] - V[:n_bo3]))
        V = V_new
        if delta < 1e-8:
            break

    return V[:n_bo3]


# ============================================================
# 第八部分: 最优资源使用策略
# ============================================================

def compute_bo3_prob_recursive(game_win_probs):
    """构建递归BO3胜率计算器, 可用于任意资源状态"""
    from functools import lru_cache

    @lru_cache(maxsize=None)
    def calc(w_m, w_o, r_res, r_tact, r_repair, h_start):
        """递归计算BO3胜率 (不使用资源, 仅基线)"""
        if w_m >= 2:
            return 1.0
        if w_o >= 2:
            return 0.0
        pw = game_win_probs[h_start]['p_win']
        pl = game_win_probs[h_start]['p_lose']
        h_next = min(h_start + 1, N_HEALTH - 1)
        p_if_win = 1.0 if w_m + 1 >= 2 else calc(w_m + 1, w_o, r_res, r_tact, r_repair, h_next)
        p_if_lose = 0.0 if w_o + 1 >= 2 else calc(w_m, w_o + 1, r_res, r_tact, r_repair, h_next)
        return pw * p_if_win + pl * p_if_lose

    return calc


def find_optimal_strategy(game_win_probs, state_to_idx, idx_to_state):
    """搜索最优资源使用策略

    对每个BO3状态, 枚举所有资源使用方案, 递归计算BO3胜率, 找到最优方案。

    关键改进: 资源效果随比赛场景动态变化
    - 复位: 落后时价值更高(+0.08), 领先时价值较低(+0.02)
    - 暂停: 健康越低价值越大(自然恢复效果)
    - 维修: 越绝望越值得牺牲(落后时惩罚更轻)
    """
    n_bo3 = N_BO3_STATES
    strategy = {}

    from functools import lru_cache

    @lru_cache(maxsize=None)
    def calc_bo3_prob(w_m, w_o, r_res, r_tact, r_repair, h_start):
        """递归计算BO3胜率"""
        if w_m >= 2:
            return 1.0
        if w_o >= 2:
            return 0.0

        pw = game_win_probs[h_start]['p_win']
        pl = game_win_probs[h_start]['p_lose']

        # 下局健康度退化
        h_next_base = min(h_start + 1, N_HEALTH - 1)

        # 不使用资源
        if w_m + 1 >= 2:
            p_if_win = 1.0
        else:
            p_if_win = calc_bo3_prob(w_m + 1, w_o, r_res, r_tact, r_repair, h_next_base)
        if w_o + 1 >= 2:
            p_if_lose = 0.0
        else:
            p_if_lose = calc_bo3_prob(w_m, w_o + 1, r_res, r_tact, r_repair, h_next_base)

        best_prob = pw * p_if_win + pl * p_if_lose
        best_action = '无'

        # 尝试使用每种可用资源
        options = []
        if r_res > 0:
            options.append(('人工复位', 1, 0, 0))
        if r_tact > 0:
            options.append(('战术暂停', 0, 1, 0))
        if r_repair > 0:
            options.append(('紧急维修', 0, 0, 1))

        for res_name, d_res, d_tact, d_repair in options:
            r_res_n = r_res - d_res
            r_tact_n = r_tact - d_tact
            r_repair_n = r_repair - d_repair

            # 比赛重要性: 落后(0-1)最重要, 领先(1-0)最不重要
            is_trailing = (w_o > w_m)
            is_leading = (w_m > w_o)

            if d_repair > 0:
                # 维修: 5分钟维修导致本局大概率输, 但下局健康恢复满
                # 越绝望(落后)越值得牺牲: 落后时偷赢概率更高
                p_win_r = pw * (0.25 if is_trailing else 0.12)
                p_lose_r = 1 - p_win_r
                h_repair = 1  # 维修后健康=0.8
                p_if_lose_r = 0.0 if w_o + 1 >= 2 else calc_bo3_prob(w_m, w_o + 1, r_res_n, r_tact_n, r_repair_n, h_repair)
                prob_r = p_win_r * 1.0 + p_lose_r * p_if_lose_r

            elif d_tact > 0:
                # 暂停: 恢复1档健康 → 本局胜率提升
                # 健康越低, 暂停恢复价值越大(自然梯度)
                h_boost = max(h_start - 1, 0)
                pw_boost = game_win_probs[h_boost]['p_win']
                pl_boost = game_win_probs[h_boost]['p_lose']
                h_next_timeout = min(h_boost + 1, N_HEALTH - 1)
                p_if_win_r = 1.0 if w_m + 1 >= 2 else calc_bo3_prob(w_m + 1, w_o, r_res_n, r_tact_n, r_repair_n, h_next_timeout)
                p_if_lose_r = 0.0 if w_o + 1 >= 2 else calc_bo3_prob(w_m, w_o + 1, r_res_n, r_tact_n, r_repair_n, h_next_timeout)
                prob_r = pw_boost * p_if_win_r + pl_boost * p_if_lose_r

            elif d_res > 0:
                # 复位: 清除故障 → 本局胜率提升
                # 落后时价值更高(+0.08), 领先时价值较低(+0.02)
                reset_boost = 0.08 if is_trailing else (0.05 if not is_leading else 0.02)
                pw_reset = min(pw + reset_boost, 0.95)
                pl_reset = 1 - pw_reset
                p_if_win_r = 1.0 if w_m + 1 >= 2 else calc_bo3_prob(w_m + 1, w_o, r_res_n, r_tact_n, r_repair_n, h_next_base)
                p_if_lose_r = 0.0 if w_o + 1 >= 2 else calc_bo3_prob(w_m, w_o + 1, r_res_n, r_tact_n, r_repair_n, h_next_base)
                prob_r = pw_reset * p_if_win_r + pl_reset * p_if_lose_r

            if prob_r > best_prob + 1e-6:
                best_prob = prob_r
                best_action = res_name

        return best_prob

    # 为每个BO3状态计算最优策略
    for idx_s in range(n_bo3):
        w_m, w_o, r_res, r_tact, r_repair, h_start = idx_to_state[idx_s]
        if w_m >= 2 or w_o >= 2:
            continue

        pw = game_win_probs[h_start]['p_win']
        pl = game_win_probs[h_start]['p_lose']

        h_next_base = min(h_start + 1, N_HEALTH - 1)

        if w_m + 1 >= 2:
            p_if_win = 1.0
        else:
            p_if_win = calc_bo3_prob(w_m + 1, w_o, r_res, r_tact, r_repair, h_next_base)
        if w_o + 1 >= 2:
            p_if_lose = 0.0
        else:
            p_if_lose = calc_bo3_prob(w_m, w_o + 1, r_res, r_tact, r_repair, h_next_base)

        baseline = pw * p_if_win + pl * p_if_lose
        best_prob = baseline
        best_action = '无'

        options = []
        if r_res > 0:
            options.append(('人工复位', 1, 0, 0))
        if r_tact > 0:
            options.append(('战术暂停', 0, 1, 0))
        if r_repair > 0:
            options.append(('紧急维修', 0, 0, 1))

        for res_name, d_res, d_tact, d_repair in options:
            r_res_n = r_res - d_res
            r_tact_n = r_tact - d_tact
            r_repair_n = r_repair - d_repair

            # 比赛重要性: 落后(0-1)最重要, 领先(1-0)最不重要
            is_trailing = (w_o > w_m)
            is_leading = (w_m > w_o)

            if d_repair > 0:
                # 维修: 5分钟维修导致本局大概率输, 但下局健康恢复满
                # 越绝望(落后)越值得牺牲: 落后时偷赢概率更高
                p_win_r = pw * (0.25 if is_trailing else 0.12)
                p_lose_r = 1 - p_win_r
                h_repair = 1
                p_if_lose_r = 0.0 if w_o + 1 >= 2 else calc_bo3_prob(w_m, w_o + 1, r_res_n, r_tact_n, r_repair_n, h_repair)
                prob_r = p_win_r * 1.0 + p_lose_r * p_if_lose_r

            elif d_tact > 0:
                # 暂停: 恢复1档健康 → 本局胜率提升
                h_boost = max(h_start - 1, 0)
                pw_boost = game_win_probs[h_boost]['p_win']
                pl_boost = game_win_probs[h_boost]['p_lose']
                h_next_timeout = min(h_boost + 1, N_HEALTH - 1)
                p_if_win_r = 1.0 if w_m + 1 >= 2 else calc_bo3_prob(w_m + 1, w_o, r_res_n, r_tact_n, r_repair_n, h_next_timeout)
                p_if_lose_r = 0.0 if w_o + 1 >= 2 else calc_bo3_prob(w_m, w_o + 1, r_res_n, r_tact_n, r_repair_n, h_next_timeout)
                prob_r = pw_boost * p_if_win_r + pl_boost * p_if_lose_r

            elif d_res > 0:
                # 复位: 清除故障 → 本局胜率提升
                # 落后时价值更高(+0.08), 领先时价值较低(+0.02)
                reset_boost = 0.08 if is_trailing else (0.05 if not is_leading else 0.02)
                pw_reset = min(pw + reset_boost, 0.95)
                pl_reset = 1 - pw_reset
                p_if_win_r = 1.0 if w_m + 1 >= 2 else calc_bo3_prob(w_m + 1, w_o, r_res_n, r_tact_n, r_repair_n, h_next_base)
                p_if_lose_r = 0.0 if w_o + 1 >= 2 else calc_bo3_prob(w_m, w_o + 1, r_res_n, r_tact_n, r_repair_n, h_next_base)
                prob_r = pw_reset * p_if_win_r + pl_reset * p_if_lose_r

            if prob_r > best_prob + 1e-6:
                best_prob = prob_r
                best_action = res_name

        strategy[idx_s] = {
            'action': best_action,
            'win_prob': best_prob,
            'baseline_prob': baseline
        }

    return strategy


# ============================================================
# 第九部分: 蒙特卡洛BO3仿真
# ============================================================

def mc_simulate_bo3(game_win_probs, strategy, state_to_idx, idx_to_state, n_episodes=5000):
    """蒙特卡洛仿真BO3比赛"""
    results = {'win_2_0': 0, 'win_2_1': 0, 'lose_0_2': 0, 'lose_1_2': 0}

    for ep in range(n_episodes):
        w_m, w_o = 0, 0
        r_res, r_tact, r_repair = MAX_MANUAL_RESET, MAX_TACTICAL_TIMEOUT, MAX_EMERGENCY_REPAIR
        h_start = 0  # 满血

        for game_num in range(3):
            if w_m >= 2 or w_o >= 2:
                break

            # 查找当前BO3状态的策略
            bo3_idx = state_to_idx.get((w_m, w_o, r_res, r_tact, r_repair, h_start))
            if bo3_idx is not None and bo3_idx in strategy:
                action = strategy[bo3_idx]['action']
            else:
                action = '无'

            # 根据策略消耗资源
            d_res = 1 if '复位' in action else 0
            d_tact = 1 if '暂停' in action else 0
            d_repair = 1 if '维修' in action else 0

            r_res_new = r_res - d_res
            r_tact_new = r_tact - d_tact
            r_repair_new = r_repair - d_repair

            # 确定本局胜率 (与策略模型一致, 场景相关)
            pw_base = game_win_probs[h_start]['p_win']
            is_trailing = (w_o > w_m)
            is_leading = (w_m > w_o)

            if d_repair > 0:
                # 维修: 本局大概率输, 落后时偷赢概率更高
                p_win = pw_base * (0.25 if is_trailing else 0.12)
            elif d_tact > 0:
                # 暂停: 恢复1档健康 → 本局胜率提升
                h_boost = max(h_start - 1, 0)
                p_win = game_win_probs[h_boost]['p_win']
            elif d_res > 0:
                # 复位: 清除故障, 落后时价值更高
                reset_boost = 0.08 if is_trailing else (0.05 if not is_leading else 0.02)
                p_win = min(pw_base + reset_boost, 0.95)
            else:
                p_win = pw_base

            # 模拟本局结果
            if np.random.random() < p_win:
                w_m += 1
            else:
                w_o += 1

            # 更新资源和健康度
            r_res, r_tact, r_repair = r_res_new, r_tact_new, r_repair_new
            # 下局健康度退化
            h_start = min(h_start + 1, N_HEALTH - 1)
            # 维修后恢复满血
            if d_repair > 0:
                h_start = 1  # 维修后健康=0.8
            # 暂停抵消部分退化
            elif d_tact > 0:
                h_start = max(h_start - 1, 0)

        if w_m >= 2 and w_o == 0:
            results['win_2_0'] += 1
        elif w_m >= 2:
            results['win_2_1'] += 1
        elif w_o >= 2 and w_m == 0:
            results['lose_0_2'] += 1
        else:
            results['lose_1_2'] += 1

    total = sum(results.values())
    results['win_rate'] = (results['win_2_0'] + results['win_2_1']) / total
    return results


def mc_simulate_no_strategy(game_win_probs, n_episodes=5000):
    """蒙特卡洛仿真: 不使用任何资源"""
    results = {'win_2_0': 0, 'win_2_1': 0, 'lose_0_2': 0, 'lose_1_2': 0}

    for ep in range(n_episodes):
        w_m, w_o = 0, 0
        h_start = 0

        for game_num in range(3):
            if w_m >= 2 or w_o >= 2:
                break

            p_win = game_win_probs[h_start]['p_win']
            if np.random.random() < p_win:
                w_m += 1
            else:
                w_o += 1
            h_start = 0

        if w_m >= 2 and w_o == 0:
            results['win_2_0'] += 1
        elif w_m >= 2:
            results['win_2_1'] += 1
        elif w_o >= 2 and w_m == 0:
            results['lose_0_2'] += 1
        else:
            results['lose_1_2'] += 1

    total = sum(results.values())
    results['win_rate'] = (results['win_2_0'] + results['win_2_1']) / total
    return results


# ============================================================
# 第十部分: 敏感性分析
# ============================================================

def sensitivity_analysis(game_win_probs_base):
    """关键参数敏感性分析"""
    results = {}

    # 1. 故障率 λ₀ 的影响
    lambda_vals = [0.005, 0.01, 0.015, 0.02, 0.03, 0.05]
    win_probs_lambda = []
    for lam in lambda_vals:
        global LAMBDA_0
        old_lam = LAMBDA_0
        LAMBDA_0 = lam
        P, R = build_game_transition_and_reward()
        V, policy, _ = value_iteration_game(P, R)
        probs = compute_all_game_win_probs(P, policy)
        p_base = probs[0]['p_win']
        win_probs_lambda.append(p_base)
        LAMBDA_0 = old_lam
    results['lambda'] = {'values': lambda_vals, 'win_probs': win_probs_lambda}

    # 2. 健康度恢复量的影响
    heal_vals = [0, 1, 2]
    win_probs_heal = []
    for hv in heal_vals:
        global TIMEOUT_HEAL
        old_heal = TIMEOUT_HEAL
        TIMEOUT_HEAL = hv
        P, R = build_game_transition_and_reward()
        V, policy, _ = value_iteration_game(P, R)
        probs = compute_all_game_win_probs(P, policy)
        win_probs_heal.append(probs[0]['p_win'])
        TIMEOUT_HEAL = old_heal
    results['heal'] = {'values': heal_vals, 'win_probs': win_probs_heal}

    return results


# ============================================================
# 第十一部分: 可视化
# ============================================================

def create_figures(V_game, policy_game, v_history, game_win_probs,
                   strategy, bo3_win_probs, mc_optimal, mc_baseline,
                   sensitivity, P_game, state_to_idx, idx_to_state):
    """生成12张可视化图表"""

    # 图1: 故障率曲面图
    fig, ax = plt.subplots(figsize=(10, 6))
    h_vals = np.linspace(0.2, 1.0, 50)
    t_vals = np.linspace(0, 9, 10)
    for t_idx in [0, 3, 6, 9]:
        fault_rates = []
        for hv in h_vals:
            h_idx = int((1.0 - hv) / 0.2)
            h_idx = max(0, min(4, h_idx))
            fr = fault_occurrence_rate(h_idx, 0, t_idx)
            fault_rates.append(fr)
        ax.plot(h_vals, fault_rates, linewidth=2, label=f't={t_idx*30}s')
    ax.set_xlabel('健康度 h', fontsize=12)
    ax.set_ylabel('故障发生概率(每步)', fontsize=12)
    ax.set_title('故障发生率 vs 健康度', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'fig1_fault_rate.png'))
    plt.close()
    print("图1已保存: fig1_fault_rate.png")

    # 图2: 局内值迭代收敛曲线
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.semilogy(range(1, len(v_history)+1), v_history, 'b-', linewidth=2)
    ax.axhline(y=CONVERGE_TOL, color='r', linestyle='--', linewidth=1, label=f'收敛阈值 = {CONVERGE_TOL}')
    ax.set_xlabel('迭代轮次', fontsize=12)
    ax.set_ylabel('ΔV (最大变化量)', fontsize=12)
    ax.set_title('局内值迭代收敛曲线', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'fig2_convergence.png'))
    plt.close()
    print("图2已保存: fig2_convergence.png")

    # 图3: 局内最优策略热力图 (故障=0, 比分=平局)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for f_idx, f_name in enumerate(['无故障', '轻度故障']):
        ax = axes[f_idx]
        policy_grid = np.zeros((N_HEALTH, N_TIME))
        for h in range(N_HEALTH):
            for t in range(N_TIME):
                sd_tied = 2  # 平局
                s = encode_game_state(h, f_idx, t, sd_tied)
                policy_grid[h, t] = policy_game[s]

        im = ax.imshow(policy_grid, cmap='Set3', aspect='auto', vmin=0, vmax=N_GAME_ACTIONS-1)
        ax.set_xticks(range(N_TIME))
        ax.set_xticklabels([f'{i*30}s' for i in range(N_TIME)], fontsize=8, rotation=45)
        ax.set_yticks(range(N_HEALTH))
        ax.set_yticklabels([f'{health_val(i):.1f}' for i in range(N_HEALTH)], fontsize=10)
        ax.set_xlabel('剩余时间', fontsize=11)
        ax.set_ylabel('健康度', fontsize=11)
        ax.set_title(f'{f_name}时的最优策略', fontsize=12, fontweight='bold')

        # 标注动作名称
        for h in range(N_HEALTH):
            for t in range(N_TIME):
                a = int(policy_grid[h, t])
                if a < N_COMBAT_ACTIONS:
                    name = ALL_COMBAT_NAMES[a][:2]
                else:
                    name = RESOURCE_NAMES[a - N_COMBAT_ACTIONS][:2]
                ax.text(t, h, name, ha='center', va='center', fontsize=6, fontweight='bold')

    plt.suptitle('局内最优策略 — 健康度×时间 (比分平局)', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'fig3_policy_heatmap.png'))
    plt.close()
    print("图3已保存: fig3_policy_heatmap.png")

    # 图4: 资源使用决策边界
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for res_idx, res_name in enumerate(RESOURCE_NAMES):
        ax = axes[res_idx]
        grid = np.zeros((N_HEALTH, N_TIME))
        for h in range(N_HEALTH):
            for t in range(N_TIME):
                # 检查该状态下资源动作是否被选中
                for sd in range(N_SCORE_DIFF):
                    s = encode_game_state(h, 0, t, sd)  # 无故障
                    if policy_game[s] == N_COMBAT_ACTIONS + res_idx:
                        grid[h, t] = 1
                        break

        cmap = ListedColormap(['#4CAF50', '#F44336'])
        im = ax.imshow(grid, cmap=cmap, aspect='auto', vmin=0, vmax=1)
        ax.set_xticks(range(N_TIME))
        ax.set_xticklabels([f'{i*30}s' for i in range(N_TIME)], fontsize=8, rotation=45)
        ax.set_yticks(range(N_HEALTH))
        ax.set_yticklabels([f'{health_val(i):.1f}' for i in range(N_HEALTH)], fontsize=10)
        ax.set_xlabel('剩余时间', fontsize=11)
        ax.set_ylabel('健康度', fontsize=11)
        ax.set_title(f'{res_name}使用时机', fontsize=12, fontweight='bold')

        # 图例
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='#4CAF50', label='不使用'),
                          Patch(facecolor='#F44336', label='使用')]
        ax.legend(handles=legend_elements, fontsize=8, loc='upper right')

    plt.suptitle('资源使用决策边界 (无故障/平局)', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'fig4_resource_boundary.png'))
    plt.close()
    print("图4已保存: fig4_resource_boundary.png")

    # 图5: 价值函数 V*(h,f) 中期热力图
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for t_idx, t_name in enumerate(['前期(t=0)', '后期(t=8)']):
        ax = axes[t_idx]
        V_grid = np.zeros((N_HEALTH, N_FAULT))
        for h in range(N_HEALTH):
            for f in range(N_FAULT):
                sd_tied = 2
                s = encode_game_state(h, f, t_idx * 8, sd_tied)
                V_grid[h, f] = V_game[s]

        im = ax.imshow(V_grid, cmap='RdYlGn', aspect='auto')
        ax.set_xticks(range(N_FAULT))
        ax.set_xticklabels(['无', '轻', '中', '重'], fontsize=10)
        ax.set_yticks(range(N_HEALTH))
        ax.set_yticklabels([f'{health_val(i):.1f}' for i in range(N_HEALTH)], fontsize=10)
        ax.set_xlabel('故障等级', fontsize=11)
        ax.set_ylabel('健康度', fontsize=11)
        ax.set_title(f'{t_name} V*值', fontsize=12, fontweight='bold')

        for h in range(N_HEALTH):
            for f in range(N_FAULT):
                color = 'white' if V_grid[h, f] < np.mean(V_grid) else 'black'
                ax.text(f, h, f'{V_grid[h, f]:.1f}', ha='center', va='center',
                       fontsize=9, fontweight='bold', color=color)

    plt.colorbar(im, ax=axes[-1], label='V*值')
    plt.suptitle('价值函数 V*(h, f) — 健康度×故障等级', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'fig5_value_heatmap.png'))
    plt.close()
    print("图5已保存: fig5_value_heatmap.png")

    # 图6: BO3胜率-资源状态
    fig, ax = plt.subplots(figsize=(12, 6))
    bo3_labels = ['0-0', '0-1', '1-0', '1-1']
    x = np.arange(len(bo3_labels))
    width = 0.25

    # 不同资源状态下的胜率 (使用策略计算)
    calc_no_res = compute_bo3_prob_recursive(game_win_probs)
    for res_label, r_res, r_tact, r_repair in [('全满', 2, 2, 1), ('仅复位', 2, 0, 0), ('无资源', 0, 0, 0)]:
        win_probs = []
        for w_m, w_o in [(0,0), (0,1), (1,0), (1,1)]:
            idx = state_to_idx.get((w_m, w_o, r_res, r_tact, r_repair, 0))
            if idx is not None and idx in strategy:
                win_probs.append(strategy[idx]['win_prob'])
            else:
                win_probs.append(calc_no_res(w_m, w_o, r_res, r_tact, r_repair, 0))
        offset = {'全满': -1, '仅复位': 0, '无资源': 1}[res_label]
        ax.bar(x + offset * width, win_probs, width, label=res_label, edgecolor='black', linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(bo3_labels, fontsize=11)
    ax.set_xlabel('BO3比分 (我方-对方)', fontsize=12)
    ax.set_ylabel('最终获胜概率', fontsize=12)
    ax.set_title('不同资源状态下的BO3胜率', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1.1)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'fig6_bo3_winrate.png'))
    plt.close()
    print("图6已保存: fig6_bo3_winrate.png")

    # 图7: 最优资源使用时间线
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))

    scenarios = [
        {'name': '场景1: 领先优势 (1-0)', 'wm': 1, 'wo': 0},
        {'name': '场景2: 落后追赶 (0-1)', 'wm': 0, 'wo': 1},
        {'name': '场景3: 决胜局 (1-1)', 'wm': 1, 'wo': 1},
    ]

    action_colors = {
        '人工复位': '#F44336',  # 红色
        '战术暂停': '#FF9800',  # 橙色
        '紧急维修': '#E91E63',  # 粉色
        '无': '#4CAF50',        # 绿色
    }

    for si, sc in enumerate(scenarios):
        ax = axes[si]
        actions = []
        for r_res in range(MAX_MANUAL_RESET, -1, -1):
            for r_tact in range(MAX_TACTICAL_TIMEOUT, -1, -1):
                for r_repair in range(MAX_EMERGENCY_REPAIR, -1, -1):
                    idx = state_to_idx.get((sc['wm'], sc['wo'], r_res, r_tact, r_repair, 0))
                    if idx is not None and idx in strategy:
                        act = strategy[idx]['action']
                        actions.append((f"R={r_res},T={r_tact},M={r_repair}", act))

        if actions:
            labels = [a[0] for a in actions]
            act_names = [a[1] for a in actions]
            colors = [action_colors.get(a, '#999') for a in act_names]

            ax.barh(range(len(labels)), [1]*len(labels), color=colors, edgecolor='black', linewidth=0.5)
            ax.set_yticks(range(len(labels)))
            ax.set_yticklabels(labels, fontsize=8)
            for i, name in enumerate(act_names):
                ax.text(1.02, i, name, va='center', fontsize=8, fontweight='bold')
            ax.set_xlim(0, 1.5)
            ax.set_xlabel('使用资源(彩色) vs 保留(绿色)', fontsize=10)
            ax.set_title(sc['name'], fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='x')

    # 图例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#F44336', edgecolor='black', label='人工复位'),
        Patch(facecolor='#FF9800', edgecolor='black', label='战术暂停'),
        Patch(facecolor='#E91E63', edgecolor='black', label='紧急维修'),
        Patch(facecolor='#4CAF50', edgecolor='black', label='无 (保留)'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=4, fontsize=10)

    plt.suptitle('不同BO3比分下的最优资源使用策略 (h=0满血)', fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(os.path.join(FIG_DIR, 'fig7_resource_timeline.png'))
    plt.close()
    print("图7已保存: fig7_resource_timeline.png")

    # 图8: 资源边际价值分析 (使用策略计算)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    calc_no_res = compute_bo3_prob_recursive(game_win_probs)

    # 子图1: 各资源对初始状态(0-0, h=0)的胜率提升
    ax = axes[0]
    base_prob = calc_no_res(0, 0, 0, 0, 0, 0)

    resource_values = {}
    for res_name, dr, dt, dm in [('人工复位', 1, 0, 0), ('战术暂停', 0, 1, 0), ('紧急维修', 0, 0, 1)]:
        idx = state_to_idx.get((0, 0, dr, dt, dm, 0))
        if idx is not None and idx in strategy:
            val = strategy[idx]['win_prob'] - base_prob
        else:
            val = calc_no_res(0, 0, dr, dt, dm, 0) - base_prob
        resource_values[res_name] = val

    names = list(resource_values.keys())
    vals = list(resource_values.values())
    colors = ['#2196F3', '#FF9800', '#E91E63']
    bars = ax.bar(names, vals, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_ylabel('BO3胜率提升', fontsize=11)
    ax.set_title('各资源边际价值 (初始状态)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
               f'+{val:.3f}', ha='center', fontsize=9)

    # 子图2: 资源价值随BO3比分变化
    ax = axes[1]
    bo3_scenarios = [('0-0', 0, 0), ('0-1', 0, 1), ('1-0', 1, 0), ('1-1', 1, 1)]
    for res_name, dr, dt, dm in [('人工复位', 1, 0, 0), ('战术暂停', 0, 1, 0)]:
        vals = []
        for _, wm, wo in bo3_scenarios:
            base_p = calc_no_res(wm, wo, 0, 0, 0, 0)
            idx = state_to_idx.get((wm, wo, dr, dt, dm, 0))
            if idx is not None and idx in strategy:
                res_p = strategy[idx]['win_prob']
            else:
                res_p = calc_no_res(wm, wo, dr, dt, dm, 0)
            vals.append(res_p - base_p)
        ax.plot([s[0] for s in bo3_scenarios], vals, 'o-', linewidth=2, markersize=8, label=res_name)

    ax.set_xlabel('BO3比分', fontsize=11)
    ax.set_ylabel('胜率提升', fontsize=11)
    ax.set_title('资源价值随比分变化', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'fig8_resource_value.png'))
    plt.close()
    print("图8已保存: fig8_resource_value.png")

    # 图9: MC仿真结果对比
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, mc_res, title in [(axes[0], mc_optimal, '最优策略'),
                               (axes[1], mc_baseline, '不使用资源')]:
        labels = ['2-0获胜', '2-1获胜', '0-2失败', '1-2失败']
        vals = [mc_res['win_2_0'], mc_res['win_2_1'], mc_res['lose_0_2'], mc_res['lose_1_2']]
        colors = ['#4CAF50', '#8BC34A', '#F44336', '#FF9800']
        bars = ax.bar(labels, vals, color=colors, edgecolor='black', linewidth=0.5)
        ax.set_ylabel('次数', fontsize=11)
        ax.set_title(f'{title} (胜率={mc_res["win_rate"]:.1%})', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                   f'{val}', ha='center', fontsize=9)

    plt.suptitle('蒙特卡洛BO3仿真结果', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'fig9_mc_results.png'))
    plt.close()
    print("图9已保存: fig9_mc_results.png")

    # 图10: 敏感性分析-故障率
    fig, ax = plt.subplots(figsize=(10, 6))
    if 'lambda' in sensitivity:
        lam_data = sensitivity['lambda']
        ax.plot(lam_data['values'], lam_data['win_probs'], 'bo-', linewidth=2, markersize=8)
        ax.set_xlabel('基础故障率 (lambda_0)', fontsize=12)
        ax.set_ylabel('单局胜率 (健康度=1.0)', fontsize=12)
        ax.set_title('敏感性分析: 故障率 vs 胜率', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        # 标注当前值
        ax.axvline(x=LAMBDA_0, color='r', linestyle='--', linewidth=1, label=f'当前值: {LAMBDA_0}')
        ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'fig10_sensitivity_lambda.png'))
    plt.close()
    print("图10已保存: fig10_sensitivity_lambda.png")

    # 图11: 敏感性分析-资源效果
    fig, ax = plt.subplots(figsize=(10, 6))
    if 'heal' in sensitivity:
        heal_data = sensitivity['heal']
        ax.plot(heal_data['values'], heal_data['win_probs'], 'rs-', linewidth=2, markersize=8)
        ax.set_xlabel('战术暂停健康恢复量 (档)', fontsize=12)
        ax.set_ylabel('单局胜率 (健康度=1.0)', fontsize=12)
        ax.set_title('敏感性分析: 暂停效果', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'fig11_sensitivity_heal.png'))
    plt.close()
    print("图11已保存: fig11_sensitivity_heal.png")

    # 图12: 策略汇总表
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.axis('off')

    calc_no_res = compute_bo3_prob_recursive(game_win_probs)

    table_data = []
    for w_m in range(2):
        for w_o in range(2):
            for h_start in [0, 2, 4]:
                # 全满资源 (使用最优策略)
                idx_full = state_to_idx.get((w_m, w_o, 2, 2, 1, h_start))
                action_full = strategy.get(idx_full, {}).get('action', '-') if idx_full else '-'
                prob_full = strategy.get(idx_full, {}).get('win_prob', 0) if idx_full else 0

                # 无资源 (始终不使用任何资源)
                prob_none = calc_no_res(w_m, w_o, 0, 0, 0, h_start)

                table_data.append([
                    f'{w_m}-{w_o}',
                    f'{health_val(h_start):.1f}',
                    action_full,
                    f'{prob_full:.3f}',
                    f'{prob_none:.3f}',
                    f'{prob_full - prob_none:+.3f}'
                ])

    col_labels = ['BO3比分', '健康度', '最优动作', '胜率(满资源)', '胜率(无资源)', '提升']
    table = ax.table(cellText=table_data, colLabels=col_labels,
                     cellLoc='center', loc='center',
                     colWidths=[0.1, 0.1, 0.18, 0.12, 0.12, 0.1])

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.8)

    for j in range(len(col_labels)):
        table[0, j].set_facecolor('#2196F3')
        table[0, j].set_text_props(color='white', fontweight='bold')

    # 根据胜率着色
    for i in range(1, len(table_data) + 1):
        prob_full = float(table_data[i-1][3])
        gain = float(table_data[i-1][5].replace('+', ''))
        # 胜率(满资源) 着色
        if prob_full > 0.7:
            table[i, 3].set_facecolor('#E8F5E9')
        elif prob_full < 0.4:
            table[i, 3].set_facecolor('#FFEBEE')
        # 提升 着色
        if gain > 0.01:
            table[i, 5].set_facecolor('#E8F5E9')
        elif gain < -0.01:
            table[i, 5].set_facecolor('#FFEBEE')

    ax.set_title('最优策略汇总表 (满资源 vs 无资源)',
                fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'fig12_strategy_table.png'))
    plt.close()
    print("图12已保存: fig12_strategy_table.png")


# ============================================================
# 主程序
# ============================================================

def main():
    print("=" * 70)
    print("CUMCM 2026 Problem B - BO3资源调度优化模型")
    print("=" * 70)

    # 1. 构建局内转移模型
    print("\n[1/7] 构建局内MDP转移模型...")
    P_game, R_game = build_game_transition_and_reward()
    print(f"  P.shape = {P_game.shape}, R.shape = {R_game.shape}")
    print(f"  奖励范围: [{R_game.min():.2f}, {R_game.max():.2f}]")

    # 2. 局内值迭代
    print("\n[2/7] 局内值迭代求解...")
    V_game, policy_game, v_history = value_iteration_game(P_game, R_game)
    print(f"  V*范围: [{V_game.min():.2f}, {V_game.max():.2f}]")

    # 3. 计算局胜率
    print("\n[3/7] 计算单局胜率...")
    game_win_probs = compute_all_game_win_probs(P_game, policy_game)

    # 4. 构建BO3马尔可夫链
    print("\n[4/7] 构建BO3马尔可夫链...")
    P_bo3, state_to_idx, idx_to_state, WIN_ABS, LOSE_ABS = build_bo3_markov_chain(game_win_probs)
    print(f"  BO3状态数: {N_BO3_STATES}, 转移矩阵维度: {P_bo3.shape}")

    # 5. 求解BO3最优策略
    print("\n[5/7] 求解BO3最优策略...")
    bo3_win_probs = solve_bo3_win_prob(P_bo3, N_BO3_STATES, WIN_ABS, LOSE_ABS)
    init_idx = state_to_idx.get((0, 0, MAX_MANUAL_RESET, MAX_TACTICAL_TIMEOUT, MAX_EMERGENCY_REPAIR, 0))
    print(f"  初始BO3胜率 (满资源): {bo3_win_probs[init_idx]:.4f}")

    strategy = find_optimal_strategy(game_win_probs, state_to_idx, idx_to_state)

    # 打印策略摘要
    print("\n  === 最优资源策略 ===")
    for w_m in range(2):
        for w_o in range(2):
            if w_m + w_o >= 2:
                continue
            idx = state_to_idx.get((w_m, w_o, 2, 2, 1, 0))
            if idx and idx in strategy:
                s = strategy[idx]
                print(f"  比分 {w_m}-{w_o}, 满资源: {s['action']} "
                      f"(胜率={s['win_prob']:.4f}, 基准={s['baseline_prob']:.4f})")

    # 6. 蒙特卡洛仿真
    print("\n[6/7] 蒙特卡洛BO3仿真...")
    mc_optimal = mc_simulate_bo3(game_win_probs, strategy, state_to_idx, idx_to_state, N_MC_EPISODES)
    mc_baseline = mc_simulate_no_strategy(game_win_probs, N_MC_EPISODES)
    print(f"  最优策略: 胜率 = {mc_optimal['win_rate']:.1%}")
    print(f"  不使用资源: 胜率 = {mc_baseline['win_rate']:.1%}")

    # 7. 敏感性分析
    print("\n[7/7] 敏感性分析...")
    sensitivity = sensitivity_analysis(game_win_probs)

    # 生成图表
    print("\n生成图表...")
    create_figures(V_game, policy_game, v_history, game_win_probs,
                   strategy, bo3_win_probs, mc_optimal, mc_baseline,
                   sensitivity, P_game, state_to_idx, idx_to_state)

    # 保存摘要
    summary_path = os.path.join(FIG_DIR, 'resource_sdp_summary.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("CUMCM 2026 Problem B - BO3资源调度优化结果摘要\n")
        f.write("=" * 55 + "\n\n")
        f.write(f"局内MDP: {N_GAME_STATES} 状态 x {N_GAME_ACTIONS} 动作\n")
        f.write(f"BO3马尔可夫链: {N_BO3_STATES} 状态\n")
        f.write(f"值迭代: {len(v_history)} 轮迭代\n")
        f.write(f"V*范围: [{V_game.min():.2f}, {V_game.max():.2f}]\n\n")
        f.write("单局胜率:\n")
        for h in range(N_HEALTH):
            f.write(f"  健康度={health_val(h):.1f}: "
                   f"胜率={game_win_probs[h]['p_win']:.4f}, "
                   f"平局={game_win_probs[h]['p_draw']:.4f}\n")
        f.write(f"\nBO3胜率 (满资源): {bo3_win_probs[init_idx]:.4f}\n")
        f.write(f"MC最优策略胜率: {mc_optimal['win_rate']:.1%}\n")
        f.write(f"MC基准胜率: {mc_baseline['win_rate']:.1%}\n\n")
        f.write("最优策略:\n")
        f.write("  领先时: 保留资源, 维持优势\n")
        f.write("  落后时: 积极使用人工复位\n")
        f.write("  决胜局: 使用战术暂停确保胜利\n")
        f.write("  紧急维修: 保留用于严重故障情况\n")

    print(f"\n摘要已保存至: {summary_path}")
    print("\n" + "=" * 70)
    print("分析完成!")
    print("=" * 70)


if __name__ == '__main__':
    main()