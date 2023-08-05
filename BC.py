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
from mlagents.trainers.demo_loader import demo_to_buffer
from mlagents.trainers.buffer import BufferKey, ObservationKeyPrefix

state_size = 6
action_size = 2

load_model = True
train_mode = True

batch_size = 128
discount_factor = 0.9
learning_rate = 0.0005

train_epoch = 5000
test_epoch = 500

print_interval = 100
save_interval = 500

game = "HungryCat"
env_name = "HungryCat/HungryCat"

loaddate = "20230804132829"

date_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
save_path = f"./saved_models/{game}/BC/{date_time}"
load_path = f"./saved_models/{game}/DQN/{loaddate}"

demo_path = "./demo/HungryCatRecord.demo"

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

class Actor(torch.nn.Module):
    def __init__(self):
        super(Actor,self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_size,128),
            nn.ReLU(),
            nn.Linear(128,128),
            nn.ReLU(),
            nn.Linear(128,action_size)
        )
    def forward(self,state):
        return self.fc(state)

class BCAgent():
    def __init__(self):
        self.actor = Actor().to(device)
        self.optimizer = torch.optim.Adam(self.actor.parameters(),lr=learning_rate)
        self.writer = SummaryWriter(save_path)
        
        if load_model == True:
            print(f"Load Model from {load_path}/ckpt")
            checkpoint = torch.load(load_path+'/ckpt', map_location=device)
            self.actor.load_state_dict(checkpoint["network"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            
    def get_action(self,state,trainig=False):
        self.actor.train(trainig)
        q = self.actor(torch.FloatTensor(state).to(device))
        action = torch.argmax(q, axis=-1, keepdim=True).data.cpu().numpy()
        return action
    
    def train_model(self,state,action):
        losses = []
        rand_idx = torch.randperm(len(state))
        for iter in range(int(np.ceil(len(state)/batch_size))):
            _state = state[rand_idx[iter*batch_size: (iter+1)*batch_size]]
            _action = action[rand_idx[iter*batch_size: (iter+1)*batch_size]]
            
            action_pred = self.actor(_state)
            loss = F.mse_loss(_action,action_pred).mean()
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())
            
        return np.mean(losses)
    
    def save_model(self):
        print(f"Save Model to {save_path}/ckpt")
        torch.save({
            "network" : self.actor.state_dict(),
            "optimizer" : self.optimizer.state_dict(),
        },save_path+"/ckpt")
        
    def write_summray(self,loss,epoch):
        self.writer.add_scalar("model/loss",loss,epoch)
        
if __name__ == '__main__':
    agent = BCAgent()
    
    if train_mode:
        behavior_spec, demo_buffer = demo_to_buffer(demo_path,1)
        print(demo_buffer._fields.keys())
        demo_to_tensor = lambda key: torch.FloatTensor(demo_buffer[key]).to(device)
        state = demo_to_tensor((ObservationKeyPrefix.OBSERVATION,0))
        action = demo_to_tensor(BufferKey.DISCRETE_ACTION)
        reward = demo_to_tensor(BufferKey.ENVIRONMENT_REWARDS)
        done = demo_to_tensor(BufferKey.DONE)
        

        ret = reward.clone()
        for t in reversed(range(len(ret)-1)):
            ret[t] += (1. - done[t]) * (discount_factor * ret[t+1])
        
        losses = []
        for epoch in range(1, train_epoch+1):
            loss = agent.train_model(state,action)
            losses.append(loss)
            
            if epoch % print_interval == 0:
                mean_loss = np.mean(losses)
                print(f"{epoch} Epoch / Loss: {mean_loss:.8f}")
                agent.write_summray(mean_loss,epoch)
                losses = []
            if epoch % save_interval == 0:
                agent.save_model()
    
    print("PLAY START")

    engine_configuration_channel = EngineConfigurationChannel()

    env = UnityEnvironment(file_name=env_name,side_channels=[engine_configuration_channel])
    env.reset()
    behavior_name = list(env.behavior_specs)[0]

    spec = env.behavior_specs[behavior_name]
    engine_configuration_channel.set_configuration_parameters(time_scale=1.0)

    dec,term = env.get_steps(behavior_name)

        
    losses, scores, episode, score = [],[],0,0
    step = 0
    
    episode,score = 0,0
    while(episode < test_epoch):
            
        state = dec.obs[0]
        action = agent.get_action(state,train_mode)
        action_tuple = ActionTuple()
        action_tuple.add_discrete(action)
        env.set_actions(behavior_name,action_tuple)
        env.step()
        dec,term = env.get_steps(behavior_name)
        done = len(term.agent_id) > 0
        
        
        if(done):
            next_state = term.obs[0]
            reward = term.reward
        else:
            next_state = dec.obs[0]
            reward = dec.reward
            
        score += reward[0]  
            
        if done:
            if episode % print_interval == 0 and episode != 0:
                print(f"{episode} Episode / Step: {step} / Score: {score:.2f} /")
            
            episode += 1
            step += 1
            score = 0
            
           
    
    env.close()