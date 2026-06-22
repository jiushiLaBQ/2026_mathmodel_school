# -*- coding: utf-8 -*-
"""
附录: 核心算法伪代码与关键公式
CUMCM 2026 Problem B - 众擎 PM01 人形机器人格斗分析

本文件整理四道题目的核心算法, 可直接嵌入论文附录。
"""

# ============================================================================
# 附录A: 问题一 - 拉格朗日动力学与攻击力指标
# ============================================================================

"""
A.1 正运动学算法
----------------
输入: 关节角度 q ∈ R^23, 机器人模型 Robot
输出: 各连杆世界变换 T_world ∈ R^{28×4×4}, 质心位置 p_com ∈ R^{28×3}

算法 1: 正运动学递推
    T_world[0] ← I_{4×4}                    // 基座固定
    for i = 1 to 27:
        parent ← links[i].parent
        j ← links[i].joint_idx
        if j ≥ 0:
            R_joint ← AxisAngleToRotation(links[i].joint_axis, q[j])
        else:
            R_joint ← I_{3×3}
        T_joint ← [R_joint, 0; 0^T, 1]
        T_world[i] ← T_world[parent] · T_fixed[i] · T_joint
        p_com[i] ← (T_world[i] · [com_offset; 1])_{1:3}
    return T_world, p_com
"""

"""
A.2 质量矩阵 M(q) 计算
-----------------------
公式: M_{ij} = Σ_{k=1}^{28} [m_k · J_{v,k}^T · J_{v,k} + J_{ω,k}^T · R_k · I_k · R_k^T · J_{ω,k}] + J_{m,j} · δ_{ij}

其中:
  m_k: 第k个连杆质量
  J_{v,k}: 第k个连杆线速度雅可比 (3×23)
  J_{ω,k}: 第k个连杆角速度雅可比 (3×23)
  R_k: 第k个连杆世界坐标系旋转矩阵
  I_k: 第k个连杆体坐标系转动惯量对角阵
  J_{m,j}: 第j个关节电机等效惯量

算法 2: 质量矩阵组装
    M ← zeros(23, 23)
    for k = 0 to 27:
        R_k ← T_world[k][:3, :3]
        I_world_k ← R_k · diag(inertia_diag[k]) · R_k^T
        M ← M + m_k · (J_v[k]^T · J_v[k])
        M ← M + J_omega[k]^T · I_world_k · J_omega[k]
    for j = 0 to 22:
        M[j,j] ← M[j,j] + motor_inertia[j]
    M ← (M + M^T) / 2                    // 强制对称
    return M
"""

"""
A.3 重力矩向量 g(q) 计算
-------------------------
公式: g_i = -Σ_{k=1}^{28} m_k · g^T · J_{v,k}[:,i]

其中 g = [0, 0, -9.81]^T 为重力加速度向量。

算法 3: 重力矩计算
    g_world ← [0, 0, -9.81]^T
    g ← zeros(23)
    for i = 0 to 22:
        for k = 0 to 27:
            g[i] ← g[i] - m_k · g_world^T · J_v[k][:, i]
    return g
"""

"""
A.4 科里奥利力与离心力 C(q,q̇)·q̇ 计算
----------------------------------------
使用克里斯托弗尔符号:
  c_{ijk} = 0.5 · (∂M_{ij}/∂q_k + ∂M_{ik}/∂q_j - ∂M_{jk}/∂q_i)
  C·q̇_i = Σ_j Σ_k c_{ijk} · q̇_j · q̇_k

算法 4: 数值微分法计算科里奥利力
    for k = 0 to 22:
        q_plus ← q; q_plus[k] += δ
        q_minus ← q; q_minus[k] -= δ
        M_plus ← ComputeMassMatrix(q_plus)
        M_minus ← ComputeMassMatrix(q_minus)
        dM_dq[k] ← (M_plus - M_minus) / (2δ)

    C_qdot ← zeros(23)
    for i = 0 to 22:
        for j = 0 to 22:
            c_ij ← 0
            for k = 0 to 22:
                c_ijk ← 0.5 · (dM_dq[k][i,j] + dM_dq[j][i,k] - dM_dq[i][j,k])
                c_ij ← c_ij + c_ijk · q̇[k]
            C_qdot[i] ← C_qdot[i] + c_ij · q̇[j]
    return C_qdot
"""

"""
A.5 攻击力指标计算
-------------------
输入: 关节状态 (q, q̇), 质量矩阵 M, 雅可比 J_v
输出: 末端速度 v_ee, 等效质量 m_e, 有效动能 E_k, 有效动量 p_m

算法 5: 攻击力指标
    v_ee ← J_v · q̇                               // 末端执行器速度
    v_speed ← ‖v_ee‖                              // 速度大小
    u ← v_ee / v_speed                            // 打击方向单位向量

    M_inv ← M^{-1}
    Λ^{-1} ← J_v · M_inv · J_v^T                 // 操作空间惯量逆 (3×3)
    m_e ← 1 / (u^T · Λ^{-1} · u)                // 等效质量

    E_k ← 0.5 · m_e · v_speed^2                  // 有效动能
    p_m ← m_e · v_speed                          // 有效动量
    return v_ee, v_speed, m_e, E_k, p_m
"""


# ============================================================================
# 附录B: 问题二 - 攻防博弈与纳什均衡
# ============================================================================

"""
B.1 攻防效用矩阵构建
---------------------
公式: U_{ij} = H_{ij} - C_i - R_{ij}

其中:
  H_{ij}: 有效伤害 = E_k^{peak} · (1 - defense_reduction_{ij})
  C_i: 能量代价 = ∫‖τ‖dt (归一化)
  R_{ij}: 反击风险 (反击类防御时非零)

算法 6: 效用矩阵计算
    // 1. 能量代价
    for i = 0 to 12:
        τ_norm ← ‖τ_i(t)‖₂ (各时间步)
        C[i] ← TrapezoidIntegral(τ_norm, dt)
    C_norm ← Normalize(C, [0,1])

    // 2. 有效伤害矩阵
    E_k_base ← [peak_E_k for each attack]
    for i = 0 to 12:
        for j = 0 to 21:
            reduction ← BaseReduction[defense_type[j]]
            reduction ← reduction · DirectionMatchFactor(i, j)
            H[i,j] ← E_k_base[i] · (1 - reduction)

    // 3. 反击风险
    for i = 0 to 12:
        for j = 0 to 21:
            if defense_type[j] == '反击':
                R[i,j] ← CounterAttackRisk(i, j)

    U ← H - C_norm[:, None] - R
    return U, H, C_norm, R
"""

"""
B.2 纯策略纳什均衡 (鞍点) 搜索
-------------------------------
鞍点条件: U[i*, j] ≤ U[i*, j*] ≤ U[i, j*] 对所有 i, j 成立

算法 7: 鞍点搜索
    row_mins ← min_j(U[i,j]) for each i        // 进攻方最坏情况
    col_maxs ← max_i(U[i,j]) for each j        // 防守方最坏情况

    maximin ← max_i(row_mins)                   // 极大极小值
    minimax ← min_j(col_maxs)                   // 极小极大值

    saddle_points ← []
    if |maximin - minimax| < ε:
        for each (i,j):
            if |U[i,j] - maximin| < ε and |U[i,j] - row_mins[i]| < ε
               and |U[i,j] - col_maxs[j]| < ε:
                saddle_points.append((i,j))

    return saddle_points, (maximin == minimax)
"""

"""
B.3 混合策略纳什均衡 (线性规划)
-------------------------------
当纯策略均衡不存在时, 求解混合策略:

设 x ∈ R^{13} 为进攻方策略概率, y ∈ R^{22} 为防守方策略概率

进攻方问题:
  max  v
  s.t. U^T · x ≥ v · 1_{22}
       1^T · x = 1, x ≥ 0

防守方问题:
  min  w
  s.t. U · y ≤ w · 1_{13}
       1^T · y = 1, y ≥ 0

算法 8: 线性规划求解混合策略
    // 进攻方
    c ← [-1, 0, ..., 0]                        // 目标函数系数
    A_ub ← [-1_{22}, -U^T]                     // 不等式约束
    b_ub ← zeros(22)
    A_eq ← [0, 1_{13}^T]                       // 等式约束
    b_eq ← [1]
    bounds ← [(0, None)] + [(0, 1)] * 13
    x_opt, v ← linprog(c, A_ub, b_ub, A_eq, b_eq, bounds)

    return x_opt[1:], v                         // 最优混合策略, 博弈值
"""


# ============================================================================
# 附录C: 问题三 - MDP最优格斗策略
# ============================================================================

"""
C.1 状态空间与动作空间
-----------------------
状态: s = (score, stamina, distance, time_phase, opponent) ∈ S
  score ∈ {0,1,2}: 净胜分等级 (劣势/均势/优势)
  stamina ∈ {0,1,2}: 体力等级 (低/中/高)
  distance ∈ {0,1,2}: 距离等级 (近/中/远)
  time_phase ∈ {0,1,2}: 时间阶段 (前/中/后)
  opponent ∈ {0,1}: 对手姿态 (进攻/防守)
  |S| = 3 × 3 × 3 × 3 × 2 = 162

动作: a ∈ A = A_attack ∪ A_defense
  A_attack: 13种攻击动作 (索引 0-12)
  A_defense: 22种防御动作 (索引 13-34)
  |A| = 35
"""

"""
C.2 奖励函数 R(s, a, s')
-------------------------
公式: R(s,a,s') = 10·Δscore·τ_t + 5·Δsituation·τ_t + 3·d_ctrl - 20·cost_a + aggr(s)·Δscore

其中:
  Δscore: 分差等级变化 (s'的score - s的score)
  τ_t: 时间紧迫度权重 [0.5, 1.0, 2.0]
  Δsituation: 对手姿态变化 (+1对手转防守, -1对手转进攻)
  d_ctrl: 距离控制奖励 (攻击时距离匹配+1, 不匹配-0.5)
  cost_a: 动作体力消耗
  aggr(s): 分差策略系数 [1.5, 1.0, 0.8] (落后时更激进)

算法 9: 奖励计算
    R ← 10 × (score_next - score_now) × τ[time_phase]
         + 5 × situation_change × τ[time_phase]
         + 3 × distance_control
         - 20 × stamina_cost[action]
         + aggression[score_now] × (score_next - score_now)
    return R
"""

"""
C.3 专家规则转移概率 P(s'|s,a)
-------------------------------
对每个 (s,a), 基于专家规则定义 2-4 个可能的下一状态及概率。

算法 10: 转移概率构建
    P ← zeros(162, 35, 162)
    for s = 0 to 161:
        (sc, st, di, ti, op) ← DecodeState(s)
        for a = 0 to 34:
            if is_attack(a):
                power ← ATTACK_POWER[a]
                dist_match ← DistanceMatchFactor(di, a)
                p_hit ← BaseHitProb × power × dist_match × (1 + 0.1×st)

                // 命中: 分差提升
                s_hit ← EncodeState(min(sc+1,2), st_next, di_next, ti_next, op_next)
                P[s,a,s_hit] += p_hit

                // 未命中: 体力消耗
                s_miss ← EncodeState(sc, st_next, di, ti_next, op)
                P[s,a,s_miss] += (1 - p_hit)
            else:
                defense ← DEFENSE_STRENGTH[a-13]
                p_block ← BaseBlockProb × defense × (1 + 0.05×st)

                // 成功防御: 对手转防守
                s_block ← EncodeState(sc, st_next, di_next, ti_next, 1)
                P[s,a,s_block] += p_block

                // 防御失败: 保持原态
                s_fail ← EncodeState(sc, st_next, di, ti_next, op)
                P[s,a,s_fail] += (1 - p_block)
    return P
"""

"""
C.4 值迭代求解最优策略
-----------------------
Bellman最优方程: V*(s) = max_a [R(s,a) + γ · Σ_{s'} P(s'|s,a) · V*(s')]

算法 11: 向量化值迭代
    V ← zeros(|S|)
    for iter = 1 to MAX_ITER:
        // 向量化Q值计算
        PV ← einsum('ijk,k->ij', P, V)         // P @ V, 形状 (162, 35)
        Q ← R + γ × PV                          // Q值矩阵
        V_new ← max_a(Q, axis=1)                // 最优价值

        δ ← ‖V_new - V‖_∞
        V ← V_new
        if δ < tol: break

    // 提取最优策略
    PV ← einsum('ijk,k->ij', P, V)
    Q ← R + γ × PV
    π*(s) ← argmax_a(Q[s,a])                    // 最优策略
    return V, π*
"""

"""
C.5 蒙特卡洛策略评估
---------------------
算法 12: MC仿真评估
    returns ← zeros(|S|)
    counts ← zeros(|S|)
    for s0 = 0 to 161:
        for ep = 1 to N:
            s ← s0, total_reward ← 0, discount ← 1
            for t = 1 to ROUND_TIME:
                a ← π*(s)
                s' ← SampleFromDistribution(P[s,a,:])
                r ← R[s,a]
                total_reward += discount × r
                discount *= γ
                s ← s'
            returns[s0] += total_reward
            counts[s0] += 1
    V_mc ← returns / counts
    return V_mc
"""


# ============================================================================
# 附录D: 问题四 - BO3资源调度优化
# ============================================================================

"""
D.1 两层层次化模型架构
-----------------------
第一层: 局内MDP (单局比赛内的格斗与资源决策)
  状态: s_game = (h, f, t, sd) ∈ S_game
    h ∈ {0,...,4}: 健康度等级 (1.0, 0.8, 0.6, 0.4, 0.2)
    f ∈ {0,...,3}: 故障等级 (无/轻/中/重)
    t ∈ {0,...,9}: 剩余时间 (每档30s)
    sd ∈ {0,...,4}: 比分差 (-2,-1,0,+1,+2)
    |S_game| = 5 × 4 × 10 × 5 = 1000

  动作: a ∈ A_game = A_combat ∪ A_resource
    A_combat: 35种格斗动作 (索引 0-34)
    A_resource: 3种资源动作 (人工复位/战术暂停/紧急维修, 索引 35-37)
    |A_game| = 38

第二层: BO3马尔可夫链 (跨局资源分配)
  状态: s_bo3 = (w_m, w_o, r_res, r_tact, r_repair, h_start)
    w_m ∈ {0,1}: 我方胜局数
    w_o ∈ {0,1}: 对方胜局数
    r_res ∈ {0,1,2}: 剩余人工复位次数
    r_tact ∈ {0,1,2}: 剩余战术暂停次数
    r_repair ∈ {0,1}: 剩余紧急维修次数
    h_start ∈ {0,...,4}: 本局起始健康度
    |S_bo3| = 4 × 3 × 3 × 2 × 5 = 360
"""

"""
D.2 故障发生模型 (泊松过程)
----------------------------
公式: λ(t,h) = λ₀ · e^{α·(1-h_val)} · (1 + 0.3·t/N_TIME) · (1 + 0.8·f)

其中:
  λ₀ = 0.015: 基础故障率
  α = 1.5: 健康度敏感系数
  h_val: 健康度实际值 (0.2~1.0)
  t: 当前时间档
  f: 当前故障等级

算法 13: 故障等级转移
    p_fault ← min(λ₀ × exp(α×(1-h_val)) × time_factor × fault_factor, 0.5)
    if random() < p_fault:
        // 故障发生, 等级提升
        if f == 0: f_next ← Bernoulli(0.7) ? 1 : 2
        elif f == 1: f_next ← Bernoulli(0.6) ? 2 : 3
        elif f == 2: f_next ← Bernoulli(0.8) ? 3 : 2
        else: f_next ← 3
    else:
        f_next ← f
    return f_next
"""

"""
D.3 局内值迭代
---------------
与问题三相同的值迭代框架, 但状态空间扩大到1000。

算法 14: 局内值迭代
    V_game ← zeros(1000)
    for iter = 1 to 3000:
        PV ← einsum('ijk,k->ij', P_game, V_game)    // (1000, 38)
        Q ← R_game + 0.98 × PV
        V_new ← max_a(Q, axis=1)
        δ ← ‖V_new - V_game‖_∞
        V_game ← V_new
        if δ < 1e-4: break
    π_game*(s) ← argmax_a(Q[s,a])
    return V_game, π_game*
"""

"""
D.4 局胜率计算 (前向概率传播)
-------------------------------
从初始状态 (h, f=0, t=0, sd=0) 出发, 按最优策略传播概率。

算法 15: 局胜率计算
    function ComputeGameWinProb(h_start):
        s0 ← EncodeGameState(h_start, 0, 0, 2)      // sd=2表示平局开始
        dist ← zeros(1000)
        dist[s0] ← 1.0

        for step = 1 to 10:
            new_dist ← zeros(1000)
            for s = 0 to 999:
                if dist[s] < 1e-15: continue
                a ← π_game*(s)
                new_dist += dist[s] × P_game[s, a, :]
            dist ← new_dist

        // 统计终端状态
        p_win ← 0
        for s = 0 to 999:
            (h, f, t, sd) ← DecodeGameState(s)
            if t ≥ 9 and sd ≥ 3:
                p_win += dist[s]
        return p_win
"""

"""
D.5 BO3递归胜率计算
--------------------
公式: P_win(w_m, w_o, R, h) = pw·P_win(w_m+1, w_o, R', h') + pl·P_win(w_m, w_o+1, R', h')

其中 pw, pl 为当前健康度下的局胜率, R'为资源消耗后的状态, h'为下局健康度。

算法 16: 递归BO3胜率 (带记忆化)
    @lru_cache(maxsize=None)
    function CalcBo3Prob(w_m, w_o, r_res, r_tact, r_repair, h_start):
        if w_m ≥ 2: return 1.0                     // 我方胜BO3
        if w_o ≥ 2: return 0.0                     // 对方胜BO3

        pw ← GameWinProb[h_start].p_win
        pl ← GameWinProb[h_start].p_lose
        h_next ← min(h_start + 1, 4)               // 局间健康退化

        // 基线: 不使用资源
        p_win ← pw × CalcBo3Prob(w_m+1, w_o, R..., h_next)
              + pl × CalcBo3Prob(w_m, w_o+1, R..., h_next)

        // 尝试使用每种可用资源
        for each available resource:
            R' ← R with resource consumed
            if resource == 复位:
                pw_r ← min(pw + boost[scenario], 0.95)    // 场景相关boost
            elif resource == 暂停:
                pw_r ← GameWinProb[max(h-1, 0)].p_win     // 健康恢复
            elif resource == 维修:
                pw_r ← pw × penalty[scenario]              // 牺牲本局
                h_repair ← 1                               // 下局满血

            p_r ← ComputeBo3ProbWithResource(...)
            if p_r > p_win + ε:
                p_win ← p_r
                best_action ← resource

        return p_win
"""

"""
D.6 最优资源使用策略搜索
-------------------------
算法 17: 枚举搜索最优策略
    strategy ← {}
    for each BO3 state (w_m, w_o, R, h):
        if w_m ≥ 2 or w_o ≥ 2: continue

        // 计算基线胜率
        baseline ← CalcBo3Prob(w_m, w_o, R, h)
        best_prob ← baseline
        best_action ← '无'

        // 枚举所有可用资源
        for each available resource in R:
            R' ← R with resource consumed
            prob ← CalcBo3ProbWithResource(w_m, w_o, R', h, resource)
            if prob > best_prob + ε:
                best_prob ← prob
                best_action ← resource

        strategy[state] ← {action: best_action, win_prob: best_prob}
    return strategy
"""

"""
D.7 蒙特卡洛BO3仿真验证
-------------------------
算法 18: MC BO3仿真
    results ← {win_2_0: 0, win_2_1: 0, lose_0_2: 0, lose_1_2: 0}

    for ep = 1 to 20000:
        w_m ← 0, w_o ← 0
        R ← (2, 2, 1)                           // 满资源
        h ← 0                                     // 满血

        for game = 1 to 3:
            if w_m ≥ 2 or w_o ≥ 2: break

            // 查找当前状态的最优策略
            action ← strategy[(w_m, w_o, R, h)]

            // 根据资源类型调整本局胜率
            pw ← GameWinProb[h].p_win
            if action == '维修':
                pw ← pw × 0.15                    // 牺牲本局
            elif action == '暂停':
                pw ← GameWinProb[max(h-1,0)].p_win // 健康恢复
            elif action == '复位':
                pw ← min(pw + boost, 0.95)        // 清除故障

            // 模拟本局结果
            if random() < pw:
                w_m += 1
            else:
                w_o += 1

            // 更新资源和健康度
            R ← R - resource_used
            h ← min(h + 1, 4)                     // 退化
            if action == '维修': h ← 1            // 维修恢复
            elif action == '暂停': h ← max(h-1, 0) // 暂停抵消退化

        // 记录结果
        if w_m == 2: results[win_type] += 1
        else: results[lose_type] += 1

    win_rate ← (results[win_2_0] + results[win_2_1]) / 20000
    return results, win_rate
"""


# ============================================================================
# 附录E: 关键参数汇总表
# ============================================================================

"""
表E.1: 全局参数汇总
+-------------------+-------+-------------------------------+
| 参数              | 值    | 含义                          |
+-------------------+-------+-------------------------------+
| N_JOINTS          | 23    | 关节自由度数                  |
| N_LINKS           | 28    | 连杆数量                      |
| DT                | 0.001 | 仿真时间步长 (s)              |
| GAMMA (P3)        | 0.95  | MDP折扣因子                   |
| GAMMA (P4)        | 0.98  | 局内MDP折扣因子               |
| LAMBDA_0          | 0.015 | 基础故障率 (每秒)             |
| ALPHA_FAULT       | 1.5   | 健康度故障敏感系数            |
| MAX_MANUAL_RESET  | 2     | 人工复位最大次数              |
| MAX_TACTICAL_TIMEOUT| 2   | 战术暂停最大次数              |
| MAX_EMERGENCY_REPAIR| 1   | 紧急维修最大次数              |
| N_MC_EPISODES     | 20000 | 蒙特卡洛仿真场次              |
+-------------------+-------+-------------------------------+

表E.2: 13种攻击动作有效动能 (问题一结果)
+----+------------+---------+--------+--------+--------+
| 序号| 动作名称  | E_k(J)  | m_e(kg)| v(m/s) | p(N·s) |
+----+------------+---------+--------+--------+--------+
| 1  | 左直拳     | 12.3    | 2.1    | 3.4    | 7.2    |
| 2  | 右直拳     | 13.8    | 2.3    | 3.5    | 8.0    |
| 3  | 左摆拳     | 15.2    | 2.0    | 3.9    | 7.8    |
| 4  | 右摆拳     | 16.1    | 2.1    | 3.9    | 8.2    |
| 5  | 左上勾拳   | 14.5    | 2.2    | 3.6    | 8.0    |
| 6  | 右上勾拳   | 15.0    | 2.3    | 3.6    | 8.3    |
| 7  | 左掌击     | 10.8    | 1.9    | 3.4    | 6.5    |
| 8  | 右肘击     | 18.5    | 2.5    | 3.8    | 9.5    |
| 9  | 左膝击     | 22.3    | 3.2    | 3.7    | 11.8   |
| 10 | 右前蹬     | 25.1    | 3.5    | 3.8    | 13.3   |
| 11 | 右侧踢     | 28.7    | 3.8    | 3.9    | 14.8   |
| 12 | 右回旋踢   | 35.2    | 4.1    | 4.1    | 16.8   |
| 13 | 右后踢     | 30.5    | 3.9    | 3.9    | 15.2   |
+----+------------+---------+--------+--------+--------+
(注: 数值为典型值, 实际结果见 results.npz)

表E.3: 问题四资源效果模型
+----------+----------------+----------------+----------------+
| 资源     | 领先(1-0)      | 平局(0-0)      | 落后(0-1)      |
+----------+----------------+----------------+----------------+
| 人工复位 | +0.02 胜率提升 | +0.05 胜率提升 | +0.08 胜率提升 |
| 战术暂停 | 恢复1档健康    | 恢复1档健康    | 恢复1档健康    |
| 紧急维修 | 本局×0.12      | 本局×0.12      | 本局×0.25      |
+----------+----------------+----------------+----------------+
"""


if __name__ == '__main__':
    print("附录算法文件 - 可直接嵌入论文附录")
    print("包含: 问题一(拉格朗日动力学), 问题二(博弈论), 问题三(MDP), 问题四(BO3调度)")
