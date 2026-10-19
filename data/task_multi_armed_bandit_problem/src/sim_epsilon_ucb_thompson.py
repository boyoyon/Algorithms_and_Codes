import numpy as np
import matplotlib.pyplot as plt

# シミュレーションの設定
# np.random.seed(42)
n_arms = 3
true_probs = [0.2, 0.5, 0.8]  # 各腕の真の成功確率（一番右が正解）
n_steps = 5000

# 0. Greedy法
def run_greedy():
    n_trials = int(n_steps / 500)

    counts = np.zeros(n_arms)   # 各腕を引いた回数
    values = np.zeros(n_arms)   # 各腕の報酬の平均（期待値）
    total_reward = 0

    greedy = []
    for i in range(n_trials):
        arm = np.random.choice(n_arms)

        # 腕を引いて報酬(0か1)を確定
        reward = 1 if np.random.rand() < true_probs[arm] else 0
        counts[arm] += 1
        # 期待値の逐次更新
        values[arm] += (reward - values[arm]) / counts[arm]
        total_reward += reward
        greedy.append(total_reward/(i+1))

    arm = np.argmax(values)
    for t in range(n_trials, n_steps):
        reward = 1 if np.random.rand() < true_probs[arm] else 0
        total_reward += reward
        greedy.append(total_reward/t)

    return greedy

# 1. Epsilon-Greedy法
def run_epsilon_greedy(eps=0.1):
    counts = np.zeros(n_arms)   
    values = np.zeros(n_arms)  
    total_reward = 0
    
    epsilon = []
    for i in range(n_steps):
        # 確率epsでランダム探索、1-epsで現在最善の腕を活用
        if np.random.rand() < eps:
            arm = np.random.choice(n_arms)
        else:
            arm = np.argmax(values)
            
        # 腕を引いて報酬(0か1)を確定
        reward = 1 if np.random.rand() < true_probs[arm] else 0
        counts[arm] += 1
        # 期待値の逐次更新
        values[arm] += (reward - values[arm]) / counts[arm]
        total_reward += reward
        epsilon.append(total_reward/(i+1))

    return epsilon

# 2. UCB1法
def run_ucb1():
    counts = np.zeros(n_arms)
    values = np.zeros(n_arms)
    total_reward = 0
    
    # 最初にすべての腕を1回ずつ試す（UCBの初期化に必要なステップ）
    nr = 1
    ucb = []
    for arm in range(n_arms):
        reward = 1 if np.random.rand() < true_probs[arm] else 0
        counts[arm] += 1
        values[arm] = reward
        total_reward += reward
        ucb.append(total_reward/nr)
        nr += 1

    # UCBスコアに基づく選択
    for t in range(n_arms, n_steps):
        # UCBスコア = 期待値 + 不確実さのボーナス
        ucb_values = values + np.sqrt(2 * np.log(t) / counts)
        arm = np.argmax(ucb_values)
        
        reward = 1 if np.random.rand() < true_probs[arm] else 0
        counts[arm] += 1
        values[arm] += (reward - values[arm]) / counts[arm]
        total_reward += reward

        ucb.append(total_reward/nr)
        nr += 1

    return ucb

# 3. Thompson Sampling（トンプソンサンプリング）
def run_thompson_sampling():
    alphas = np.ones(n_arms)  # 成功回数+1 (Beta分布のパラメータ)
    betas = np.ones(n_arms)   # 失敗回数+1 (Beta分布のパラメータ)
    total_reward = 0
    
    tompson = []
    for i in range(n_steps):
        # 各腕のBeta分布から確率を1つずつサンプリング
        samples = [np.random.beta(alphas[i], betas[i]) for i in range(n_arms)]
        arm = np.argmax(samples)
        
        reward = 1 if np.random.rand() < true_probs[arm] else 0
        if reward == 1:
            alphas[arm] += 1
        else:
            betas[arm] += 1
        total_reward += reward

        tompson.append(total_reward/(i+1))

    return tompson

# 実行と結果表示

greedy = run_greedy()
plt.plot(greedy, label='greedy')

epsilon = run_epsilon_greedy()
plt.plot(epsilon, label='epsilon')

ucb  = run_ucb1()
plt.plot(ucb, label='usb1')

tompson = run_thompson_sampling()
plt.plot(tompson, label='tompson')

plt.legend()
plt.show()
