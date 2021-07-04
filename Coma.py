# Подключаем библиотеки
from smac.env import StarCraft2Env
import numpy as np
from numpy import arange
import pandas as pd
from collections import deque
import matplotlib.pyplot as plt
import time
import torch.nn.functional as F
import torch
import torch.nn as nn

# Флаг вывода массива целиком
np.set_printoptions(threshold=np.inf)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loss_funk_actor  = []
loss_funk_critic =[]
# Определяем архитектуру нейронной сети
# все параметры запомнинаем и храним в классе Memory
class Memory:
    def __init__(self, agent_num, action_dim):
        self.agent_num = agent_num
        self.action_dim = action_dim

        self.actions = []
        self.observations = []
        self.pi = [[] for _ in range(agent_num)]
        self.reward = []
        self.done =[]

    def get(self):
        actions = torch.tensor(self.actions)
        observations = self.observations

        pi = []
        for i in range(self.agent_num):
            pi.append(torch.cat(self.pi[i]).view(len(self.pi[i]), self.action_dim))

        reward = torch.tensor(self.reward)
        done = self.done

        return actions, observations, pi, reward, done

    def clear(self):
        self.actions = []
        self.observations = []
        self.pi = [[] for _ in range(self.agent_num)]
        self.reward = []
        self.done = []

# класс агентов
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()

        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.GRU(64, 64,2)
        #self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)
        self.act1 = nn.ReLU()

    def forward(self, x):

        x = self.fc1(x)
     #   print("x", x)
        x = self.act1(x)
      #  print("x2", x)
        x = torch.FloatTensor(x)
        x = x.view(-1, 1, 64)
        #x = self.fc2(x)
        # print("x3", x)
        x = torch.FloatTensor(x[0])
        x = x.view(-1, 64)
        x = self.act1(x)
        return F.softmax(self.fc3(x), dim=-1)

# класс критика
class Critic(nn.Module):
    def __init__(self, agent_num, state_dim, action_dim):
        super().__init__()

        input_dim = 1 + state_dim * agent_num + agent_num

        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class COMA:
    def __init__(self, agent_num, state_dim, action_dim, lr_c, lr_a, gamma, target_update_steps):
        self.agent_num = agent_num
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.gamma = gamma

        self.target_update_steps = target_update_steps

        self.memory = Memory(agent_num, action_dim)

        self.actors = [Actor(state_dim, action_dim) for _ in range(agent_num)]
        self.critic = Critic(agent_num, state_dim, action_dim)

        self.critic_target = Critic(agent_num, state_dim, action_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actors_optimizer = [torch.optim.Adam(self.actors[i].parameters(), lr=lr_a) for i in range(agent_num)]
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_c)

        self.count = 0

    # выбираем действия
    def get_actions(self, observations,avail_agent_actions):

        actions = []

        for i in range(self.agent_num):
            # получаем вероятноть всех действий определенного агента
            dist = self.actors[i](observations[i])
            # возможные действия агентов
            avail_actions = avail_agent_actions[i]
            avail_actions_ind = np.nonzero(avail_actions)[0]
            # Выбираем возможное действие агента с учетом
            # максимальной вероятноси
            action = select_actionFox(dist, avail_actions_ind, 0.5)
            if action is None: action = np.random.choice(avail_actions_ind)
            # запоминаем вероятности
            self.memory.pi[i].append(dist)
            # запоминаем действия
            actions.append(action.item())
        # запоминаем наблюдения
        self.memory.observations.append(observations)
        # запоминаем все действия агентов
        self.memory.actions.append(actions)

        return actions

    def train(self):
        actor_optimizer = self.actors_optimizer
        critic_optimizer = self.critic_optimizer

        actions, observations, pi, reward, done = self.memory.get()
        input_critic = self.build_input_critic(observations, actions)
        crit = 0
        actor = 0
        for i in range(self.agent_num):
            # обучаем агентов
            # расчитываем значение Q - функции с помощью критика
            Q_target = self.critic_target(input_critic).detach()

            action_taken = actions.type(torch.long)[:, i].reshape(-1, 1)
            # вычисляем смещение
            baseline = torch.sum(pi[i] * Q_target, dim=1).detach()
            Q_taken_target = torch.gather(Q_target, dim=1, index=action_taken).squeeze()
            #  рассчитываем функцию полезности
            advantage = Q_taken_target - baseline
            #  находим логарифм от стратегии
            log_pi = torch.log(torch.gather(pi[i], dim=1, index=action_taken).squeeze())
            #  в качестве функции потерь используется произведение фунции полезности и логарифм от стратегии
            actor_loss = - torch.mean(advantage * log_pi)
            actor+=torch.mean(advantage * log_pi).detach().numpy()
            #  само обучение
            actor_optimizer[i].zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actors[i].parameters(), 5)
            actor_optimizer[i].step()

            #  обучение критика

            Q = self.critic(input_critic)

            action_taken = actions.type(torch.long)[:, i].reshape(-1, 1)
            Q_taken = torch.gather(Q, dim=1, index=action_taken).squeeze()

            # рассчитываем TD

            r = torch.zeros(len(reward))
            for t in range(len(reward)):
                if done[t]:
                    r[t] = reward[t]
                else:
                    r[t] = reward[t] + self.gamma * Q_taken_target[t + 1]
            # функция потерь критика
            critic_loss = torch.mean((r - Q_taken) ** 2)
            crit +=torch.mean((r - Q_taken) ** 2).detach().numpy()
            # само обучение
            critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 5)
            critic_optimizer.step()


        loss_funk_critic.append(self.agent_num/crit)
        loss_funk_actor.append(actor/self.agent_num)

        print('loss_funk_actor', loss_funk_actor)
        print('loss_funk_critic', loss_funk_critic)


        if self.count == self.target_update_steps:
            self.critic_target.load_state_dict(self.critic.state_dict())
            self.count = 0
        else:
            self.count += 1
        # отчистка памяти
        self.memory.clear()

    # входные данные для критика требуют определенную форму
    def build_input_critic(self, observations, actions):
        batch_size = len(observations)

        ids = (torch.ones(batch_size) * 1).view(-1, 1)

        for i in range (len(observations)):
            observations[i] = torch.cat((observations[i][0],observations[i][1],observations[i][2]))

        observations = torch.cat(observations).view(batch_size, self.state_dim * self.agent_num)
        input_critic = torch.cat([observations.type(torch.float32), actions.type(torch.float32)], dim=-1)
        input_critic = torch.cat([ids, input_critic], dim=-1)

        return input_critic



# Выбираем возможное действие с максимальным Q-значением в зависимости от эпсилон
def select_actionFox(action_probabilities, avail_actions_ind, epsilon):
    action_probabilities = action_probabilities.detach().numpy()

    for ia in action_probabilities:
        action = np.argmax(action_probabilities)

        if action in avail_actions_ind:
            return action
        else:
            action_probabilities[0][action] = 0



# Основная функция программы
def main():
    # Загружаем среду Starcraft II, карту, сложность противника и расширенную награду
    env = StarCraft2Env(map_name="3ps1zgWallFOX", reward_only_positive=False, reward_scale_rate=200, difficulty="1")
    # Получаем и выводим на печать информацию о среде
    env_info = env.get_env_info()
    print('env_info=', env_info)
    # Получаем и выводим на печать размер наблюдений агента
    obs_size = env_info.get('obs_shape')
    print("obs_size=", obs_size)
    # Количество действий агента
    n_actions = env_info["n_actions"]
    # количество дружественных агентов
    n_agents = env_info["n_agents"]

    # Определяем основные параметры нейросетевого обучения
    ###########################################################################

    # Основные переходы в алгоритме IQL зависят от управляющих параметров
    global_step = 0  # подсчитываем общее количество шагов в игре
    # Общее количество эпизодов игры
    n_episodes = 350
    # Параметр дисконтирования
    gamma = 0.99
    # Скорость обучения
    lr_a = 0.0001
    lr_c = 0.005
    target_update_steps = 10
    ###########################################################################

    episodes_reward = []
    Reward_History = []
    winrate_history = []

    agents = COMA(n_agents, obs_size, n_actions, lr_c, lr_a, gamma, target_update_steps)

    obs_agent = np.zeros([n_agents], dtype=object)
    action_agent = np.zeros([n_agents], dtype=object)
    # Основной цикл по эпизодам игры

    ################_цикл for по эпизодам_#####################################
    for e in range(n_episodes):
        # Перезагружаем среду
        env.reset()
        # Флаг окончания эпизода
        done_n = False
        # Награда за эпизод
        episode_reward = 0
        while not done_n:

            # Храним историю состояний среды один шаг для разных агентов
            obs_agent = np.zeros([n_agents], dtype=object)
            action_agent = np.zeros([n_agents], dtype=object)

            for agent_id in range(n_agents):
                obs_agent[agent_id] = torch.FloatTensor([env.get_obs_agent(agent_id)]).to(device)
                action_agent[agent_id] = env.get_avail_agent_actions(agent_id)
            # Получаем действия агентов
            actions = agents.get_actions(obs_agent,action_agent)
            reward, done_n, _ = env.step(actions)
            agents.memory.reward.append(reward)
            agents.memory.done.append(done_n)

           # Суммируем награды за этот шаг для вычисления награды за эпизод
            episode_reward += reward


        episodes_reward.append(episode_reward)
        # Обучаем агентов
        agents.train()
        # Обновляем счетчик общего количества шагов
        global_step += 1

        ######################_конец цикла while###############################
    #Выводим счетчик шагов игры и общую награду за эпизод
        print('global_step=', global_step, "Total reward in episode {} = {}".format(e, episode_reward))

        # Собираем данные для графиков
        Reward_History.append(episode_reward)
        status = env.get_stats()
        winrate_history.append(status["win_rate"])

    ################_конец цикла по эпизодам игры_############################################

    # Закрываем среду StarCraft II
    env.close()

# Точка входа в программу
if __name__ == "__main__":
    start_time = time.time()
    main()
    # Время обучения
    print("--- %s минут ---" % ((time.time() - start_time) / 60))
