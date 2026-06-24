# 众擎PM01人形机器人格斗策略优化

第十六届妈妈杯（MotherCup）数学建模校赛 B题

## 题目概述

基于众擎PM01人形机器人（商业版，23自由度，~42kg）的竞技赛规则，通过对机器人攻击和防守动作的仿真模拟，建立数学模型，优化机器人竞技策略，实现竞技赛获胜概率最大化。

## 问题列表

| 问题 | 内容 | 代码 |
|------|------|------|
| 问题一 | 13种攻击动作的动力学分析与攻击力指标 | `first/code1_lagrangian.py` |
| 问题二 | 22种防守动作的最佳防守模型 | `second/code_game_theory.py` |
| 问题三 | 单人比赛策略的MDP优化模型 | `third/code_mdp_strategy.py` |
| 问题四 | BO3赛制资源调度优化 | `forth/code_resource_sdp.py` |
| 问题五 | 机器人产业建议书 | 见论文 |

## 项目结构

```
mathmodel_school/
├── data/                    # 机器人参数数据
├── first/                   # 问题一：拉格朗日动力学建模
├── second/                  # 问题二：博弈论攻防策略
├── third/                   # 问题三：MDP最优策略
├── forth/                   # 问题四：资源调度优化
├── fifth/                   # 问题五：综述与建议
├── references/              # 参考文献
│   ├── dynamics/            # 动力学相关
│   ├── game_theory/         # 博弈论相关
│   ├── reinforcement_learning/  # 强化学习相关
│   └── robotics_general/    # 机器人通用
├── appendix_algorithms.py   # 附录：核心算法伪代码
├── requirements.txt         # Python依赖
└── 题目.pdf                 # 原始题目
```

## 环境配置

```bash
pip install -r requirements.txt
```

## 运行顺序

```bash
# 问题一（生成动力学数据）
python first/code1_lagrangian.py

# 问题二（需先运行问题一）
python second/code_game_theory.py

# 问题三
python third/code_mdp_strategy.py

# 问题四
python forth/code_resource_sdp.py
```

## 团队

- **代码**：jiushiLaBQ
- **建模/论文**：数学科学学院两位同学

本次数学建模比赛获二等奖，感谢两位数学科学学院的队友在建模和论文撰写方面的辛苦付出，没有你们的数学推导和文字功底，这份作品不可能完成。

## 参考文献

详见 [references/](references/) 目录，按研究方向分类整理。
