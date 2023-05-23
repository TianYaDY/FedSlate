import torch
import random

class ReplayMemory():
    def __init__(self, capacity, state_shape, action_shape):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.capacity = capacity
        self.state_memory = torch.zeros((capacity,) + state_shape, device=self.device)
        self.action_memory = torch.zeros((capacity,) + action_shape, device=self.device)
        self.reward_memory = torch.zeros((capacity,), device=self.device)
        self.next_state_memory = torch.zeros((capacity,) + state_shape, device=self.device)
        self.terminals_memory = torch.zeros((capacity,), dtype=torch.bool, device=self.device)
        self.click_memory = torch.zeros((capacity,) + action_shape, dtype=torch.int, device=self.device)
        self.position = 0
        self.full = False

    def push(self, state, action, reward, click, next_state, done):
        self.state_memory[self.position] = state
        self.action_memory[self.position] = action
        self.reward_memory[self.position] = reward
        self.click_memory[self.position] = click
        self.next_state_memory[self.position] = next_state
        self.terminals_memory[self.position] = done
        self.position = (self.position + 1) % self.capacity
        self.full = self.full or self.position == 0

    def recall(self, indices):
        states = self.state_memory[indices]
        actions = self.action_memory[indices]
        rewards = self.reward_memory[indices]
        clicks = self.click_memory[indices]
        next_states = self.next_state_memory[indices]
        terminals = self.terminals_memory[indices]

        return states, actions, rewards, clicks, next_states, terminals

    def __len__(self):
        return self.capacity if self.full else self.position


if __name__ == "__main__":
    # 创建一个容量为 100 的 ReplayMemory 实例
    memory = ReplayMemory(capacity=2, state_shape=(4,), action_shape=(1,))

    # 向 ReplayMemory 中添加数据
    memory.push(torch.tensor([10, 0, 0, 0]), torch.tensor([0]), torch.tensor(1), torch.tensor([1]),
                torch.tensor([1, 1, 1, 1]), torch.tensor(False))
    memory.push(torch.tensor([0, 20, 0, 0]), torch.tensor([0]), torch.tensor(1), torch.tensor([0]),
                torch.tensor([1, 1, 1, 1]), torch.tensor(False))
    memory.push(torch.tensor([0, 30, 0, 0]), torch.tensor([0]), torch.tensor(1), torch.tensor([0]),
                torch.tensor([1, 1, 1, 1]), torch.tensor(False))
    memory.push(torch.tensor([0, 30, 0, 650]), torch.tensor([0]), torch.tensor(1), torch.tensor([0]),
                torch.tensor([1, 1, 1, 1]), torch.tensor(False))
    print(len(memory))

    # 获取指定索引的数据
    random_list = [0,1]
    print(memory.recall(random_list))
