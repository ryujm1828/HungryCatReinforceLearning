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

state_size = 4
action_size = 2

load_model = False
train_mode = True

discount_factor = 0.8
learning_rate = 0.0000002

run_step = 50000 if train_mode else 0
test_step = 500

print_interval = 100
save_interval = 100

game = "HungryCat"
env_name = "HungryCat/HungryCat"

loaddate = ""

date_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
save_path = f"./saved_models/{game}/A2C/{date_time}"
load_path = f"./saved_models/{game}/A2C/{loaddate}"

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

class A2C(torch.nn.Module):
    def __init__(self,**kwargs):
        super(A2C,self).__init__(**kwargs)
        self.d1 = torch.nn.Linear(state_size,256)
        self.d2 = torch.nn.Linear(256,256)
        self.pi = torch.nn.Linear(256,action_size)
        self.v = torch.nn.Linear(256,1)
        
    def forward(self,x):
        x = F.relu(self.d1(x))
        x = F.relu(self.d2(x))
        #print(F.softmax(self.pi(x),dim=1))
        return F.softmax(self.pi(x),dim=1),self.v(x)
    
class A2CAgent:
    def __init__(self):
        self.a2c = A2C().to(device)
        self.optimizer = torch.optim.Adam(self.a2c.parameters(),lr=learning_rate)
        self.writer = SummaryWriter(save_path)
        
        if load_model == True:
            print(f"Load model from {load_path}/ckpt ...")
            checkpoint = torch.load(load_path+'/ckpt', map_location=device)
            self.a2c.load_state_dict(checkpoint["network"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            
    def get_action(self,state,trainig=True):
        self.a2c.train(trainig)
        pi, _ = self.a2c(torch.FloatTensor(state).to(device))
        action = torch.multinomial(pi,num_samples=1).cpu().numpy()
        return action
    
    def train_model(self,state,action,reward,next_state,done):
        state,action,reward,next_state,done = map(lambda x: torch.FloatTensor(x).to(device),[state,action,reward,next_state,done])
        pi, value = self.a2c(state)
        print(state)
        with torch.no_grad():
            _,next_value = self.a2c(next_state)
            target_value = reward + discount_factor * next_value
        
        critic_loss = F.mse_loss(target_value,value)
        
        eye = torch.eye(action_size).to(device)
        one_hot_action = eye[action.view(-1).long()]
        advantage = (target_value - value).detach()
        actor_loss = -(torch.log((one_hot_action*pi).sum(1))*advantage).mean()
        total_loss = critic_loss + actor_loss
        
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        return actor_loss.item(),critic_loss.item()

    
    def save_model(self):
        print(f"Save Model to {save_path}/ckpt")
        torch.save({
            "network" : self.a2c.state_dict(),
            "optimizer" : self.optimizer.state_dict(),
        },save_path+'/ckpt')
        
    def write_summary(self,score,actor_loss,critic_loss,step):
        self.writer.add_scalar("run/score",score,step)
        self.writer.add_scalar("model/acotr_loss",actor_loss,step)
        self.writer.add_scalar("model/critic_loss",critic_loss,step)
        
    
if __name__ == '__main__':
    engine_configuration_channel = EngineConfigurationChannel()
    env = UnityEnvironment(file_name=env_name,
                           side_channels=[engine_configuration_channel])

    env.reset()
    
    behavior_name = list(env.behavior_specs.keys())[0]
    spec = env.behavior_specs[behavior_name]
    engine_configuration_channel.set_configuration_parameters(time_scale=4.0)
    dec, term = env.get_steps(behavior_name)
    
    agent = A2CAgent()
    actor_losses, critic_losses, scores, episode, score = [],[],[],0,0
    step = 0
    while(step <= run_step + test_step):
        if step == run_step:
            if train_mode:
                agent.save_model()
            print("TEST START")
            train_mode= False
            engine_configuration_channel.set_configuration_parameters(time_scale=1.0)
        
        #print(dec.obs)
        state = dec.obs[0]
        action = agent.get_action(state,train_mode)
        #print(action)
        action_tuple = ActionTuple()
        action_tuple.add_discrete(action)
        env.set_actions(behavior_name,action_tuple)
        env.step()
        
        dec,term = env.get_steps(behavior_name)
        done = len(term.agent_id)>0
        reward = term.reward if done else dec.reward

        if done:
            next_state = term.obs[0]       #여기도 수정해야됨
        else:
            next_state = dec.obs[0]
        score += reward[0]
        
        if train_mode:
            actor_loss, critic_loss = agent.train_model(state,action[0],[reward],next_state,[done])
            actor_losses.append(actor_loss)
            critic_losses.append(critic_loss)
            
        if done:
            print(f"Episode : {episode}, socre : {score:.1f}")
            episode += 1
            step += 1
            
            scores.append(score)
            score = 0
            
            if episode % print_interval == 0 and episode != 0:
                mean_score = np.mean(scores)
                mean_actor_loss = np.mean(actor_losses) if len(actor_losses) > 0 else 0
                mean_critic_loss = np.mean(critic_losses) if len(critic_losses) > 0 else 0
                agent.write_summary(mean_score,mean_actor_loss,mean_critic_loss,step)
                actor_losses, critic_losses, scores = [],[],[]
                
                print(f"{episode} Episode / Step: {step} / Score: {mean_score:.2f}/ " + \
                    f"Actor loss: {mean_actor_loss:.2f} / Critic loss: {mean_critic_loss:.4f}")
                
            if train_mode and episode % save_interval == 0:
                agent.save_model()
                
    print("End")   
    env.close()
        
        