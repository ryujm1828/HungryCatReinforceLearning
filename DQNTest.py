import numpy as np
import random
import copy
import datetime
import platform
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from collections import deque
from mlagents_envs.environment import UnityEnvironment, ActionTuple
from mlagents_envs.side_channel.engine_configuration_channel\
    import EngineConfigurationChannel

state_size = 6
action_size = 2

load_model = True
train_mode = True

batch_size = 32
mem_maxlen = 10000
discount_factor = 0.9
learning_rate = 0.0005

run_step = 10000 if train_mode else 0
test_step = 500
train_start_step = 10
target_update_step = 10

print_interval = 10
save_interval = 500

epsilon_eval = 0.05
epsilon_init = 1.0 if train_mode else epsilon_eval
epsilon_min = 0.01
explore_step = run_step * 0.8
epsilon_delta = (epsilon_init - epsilon_min)/explore_step if train_mode else 0

game = "HungryCat"
env_name = "HungryCat/HungryCat"

loaddate = ""

date_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
save_path = f"./saved_models/{game}/DQN/{date_time}"
load_path = f"./saved_models/{game}/DQN/20230803122154"

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

class DQN(torch.nn.Module):
    def __init__(self,state_size,action_size):
        super(DQN,self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_size,128),
            nn.ReLU(),
            nn.Linear(128,128),
            nn.ReLU(),
            nn.Linear(128,action_size)
        )
    def forward(self,x):
        return self.fc(x)
    
class DQNAgent:
    def __init__(self):
        self.network = DQN(state_size,action_size).to(device)
        self.target_network = copy.deepcopy(self.network)
        self.optimizer = torch.optim.Adam(self.network.parameters(),lr=learning_rate)
        self.memory = deque(maxlen=mem_maxlen)
        self.epsilon = epsilon_init
        self.writer = SummaryWriter(save_path)
        
        if load_model == True:
            print(f"... Load Model from {load_path}/ckpt.pt...")
            checkpoint = torch.load(load_path+'/ckpt')
            self.network.load_state_dict(checkpoint["network"])
            self.target_network.load_state_dict(checkpoint["network"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            
        
    def get_action(self,state,training=True):
        self.network.train(training)
        epsilon = self.epsilon if training else epsilon_eval
        
        if epsilon > random.random():
            action = np.random.randint(0,2,size=(1,1))
        else:
            q = self.network(torch.FloatTensor(state).to(device))
            action = torch.argmax(q, axis=-1, keepdim=True).data.cpu().numpy()
        
        return action
    
    def append_sample(self,state,action,reward,next_state,done):
        self.memory.append((state,action,reward,next_state,done))
    
    def train_model(self):
        batch = random.sample(self.memory,batch_size)
        state = np.stack([b[0] for b in batch], axis=0)
        action = np.stack([b[1] for b in batch], axis=0)
        reward = np.stack([b[2] for b in batch], axis=0)
        next_state = np.stack([b[3] for b in batch], axis=0)
        done = np.stack([b[4] for b in batch], axis=0)
        
        state, action, reward, next_state, done = map(lambda x: torch.FloatTensor(x).to(device),[state,action,reward,next_state,done])
        eye = torch.eye(action_size).to(device)
        one_hot_action = eye[action.view(-1).long()]
        q = (self.network(state) * one_hot_action).sum(1,keepdims=True)
        
        with torch.no_grad():
            next_q = self.target_network(next_state)
            target_q = reward + next_q.max(1, keepdims=True).values * ((1- done) * discount_factor)
            
        loss = F.smooth_l1_loss(q,target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        #self.epsilon = max(epsilon_min, self.epsilon - epsilon_delta)
        self.epsilon = 100/(episode+1)
        
        return loss.item()
    
    def update_target(self):
        self.target_network.load_state_dict(self.network.state_dict())
        
    def save_model(self):
        print(f"Save model to {save_path}/ckpt")
        torch.save({
            "network" : self.network.state_dict(),
            "optimizer" : self.optimizer.state_dict()
        },save_path+'/ckpt')
        
    def write_summary(self,score,loss,epsilon,step):
        self.writer.add_scalar("run/score", score, step)
        self.writer.add_scalar("model/loss",loss,step)
        self.writer.add_scalar("model/epsilon",epsilon,step)
        
    
if __name__ == '__main__':
    engine_configuration_channel = EngineConfigurationChannel()
    
    env = UnityEnvironment(file_name=env_name,side_channels=[engine_configuration_channel])
    
    env.reset()
    
    behavior_name = list(env.behavior_specs)[0]
    
    spec = env.behavior_specs[behavior_name]
    engine_configuration_channel.set_configuration_parameters(time_scale=1.0)
    
    dec,term = env.get_steps(behavior_name)
    
    agent = DQNAgent()
    
    losses, scores, episode, score = [],[],0,0
    step = 0
    while step <= run_step + test_step :
        if step == run_step:
            if train_mode:
                agent.save_model()
            print("TEST START")
            train_mode = False
            engine_configuration_channel.set_configuration_parameters(time_scale=1.0)
            
        state = dec.obs[0]
        action = agent.get_action(state,train_mode)
        action_tuple = ActionTuple()
        action_tuple.add_discrete(action)
        env.set_actions(behavior_name,action_tuple)
        env.step()
        dec,term = env.get_steps(behavior_name)
        done = len(term.agent_id) > 0
        reward = dec.reward
        if(done):
            next_state = term.obs[0]
        else:
            next_state = dec.obs[0]
        score += reward[0]  
        
        if train_mode:
            agent.append_sample(state[0],action[0],reward,next_state[0],[done])
            
        if train_mode and step > train_start_step:
            loss = agent.train_model()
            losses.append(loss)
            
        if step % target_update_step == 0:
            agent.update_target()
            
        if done:
            episode += 1
            step += 1
            scores.append(score)
            score = 0
            
            if train_mode and episode % save_interval == 0:
                agent.save_model()
            
            if episode % print_interval == 0 and episode != 0:
                mean_score = np.mean(scores)
                mean_loss = np.mean(losses)
                agent.write_summary(mean_score,mean_loss,agent.epsilon,step)
                losses,scores = [], []
                
                print(f"{episode} Episode / Step: {step} / Score: {mean_score:.2f} / " + \
                    f"Loss: {mean_loss:.4f} / Epsilon: {agent.epsilon:.4f}")
            
            
        
            
    env.close()