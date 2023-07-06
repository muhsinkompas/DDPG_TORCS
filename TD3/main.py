import torch
from torch.autograd import Variable
import numpy as np
import random
from gym_torcs import TorcsEnv
import argparse
import collections
#import ipdb

from ReplayBuffer import ReplayBuffer
from ActorNetwork import ActorNetwork
from CriticNetwork import CriticNetwork
from OU import OU
import rwfile

state_size = 29
action_size = 3
LRA = 0.0001
LRC = 0.001
BUFFER_SIZE = 100000  #to change
BATCH_SIZE = 32


GAMMA = 0.95
EXPLORE = 100000.
epsilon = 1
train_indicator = 1   # train or not
TAU = 0.001

VISION = False

relaunch_le = 15

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OU = OU()

def init_weights(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.normal_(m.weight, 0, 1e-4)
        m.bias.data.fill_(0.0)

file_path = 'Best/bestlaptime.csv'
best_lap_time = rwfile.RW.read_float_from_file(file_path)



actor = ActorNetwork(state_size).to(device)
actor.apply(init_weights)
critic = CriticNetwork(state_size, action_size).to(device)

#load model
print("loading model")
try:

    actor.load_state_dict(torch.load('actormodel.pth'))
    actor.eval()
    critic.load_state_dict(torch.load('criticmodel.pth'))
    critic.eval()
    print("model load successfully")
except:
    print("cannot find the model")

#critic.apply(init_weights)
buff = ReplayBuffer(BUFFER_SIZE)

target_actor = ActorNetwork(state_size).to(device)
target_critic = CriticNetwork(state_size, action_size).to(device)
target_actor.load_state_dict(actor.state_dict())
target_actor.eval()
target_critic.load_state_dict(critic.state_dict())
target_critic.eval()

criterion_critic = torch.nn.MSELoss(reduction='sum')

optimizer_actor = torch.optim.Adam(actor.parameters(), lr=LRA)
optimizer_critic = torch.optim.Adam(critic.parameters(), lr=LRC)

#env environment
env = TorcsEnv(vision=VISION, throttle=True, gear_change=False)

file_reward = open("results/results2.txt","w") 
file_distances = open("results/distances2.txt","w") 
file_states = open("results/states2.txt", "w")
if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor') 
steps = 0
for i in range(2000):

    if np.mod(i, relaunch_le) == 0:
        ob, distFromStart = env.reset(relaunch = True)
    else:
        ob, distFromStart = env.reset()

    s_t = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY, ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm))
    
    for j in range(100000):
        loss = 0
        epsilon -= 1.0 / EXPLORE
        a_t = np.zeros([1, action_size])
        noise_t = np.zeros([1, action_size])
        #ipdb.set_trace() 
        a_t_original = actor(torch.tensor(s_t.reshape(1, s_t.shape[0]), device=device).float())

        if torch.cuda.is_available():
            a_t_original = a_t_original.data.cpu().numpy()
        else:
            a_t_original = a_t_original.data.numpy()
        #print(type(a_t_original[0][0]))

        noise_t[0][0] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][0], 0.0, 0.60, 0.30)
        noise_t[0][1] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][1], 0.5, 1.00, 0.10)
        noise_t[0][2] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][2], -0.1, 1.00, 0.05)

        #stochastic brake
        # if random.random() <= 0.1:
        #     print("apply the brake")
        #     noise_t[0][2] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][2], 0.2, 1.00, 0.10)
        
        a_t[0][0] = a_t_original[0][0] + noise_t[0][0]
        a_t[0][1] = a_t_original[0][1] + noise_t[0][1]
        a_t[0][2] = a_t_original[0][2] + noise_t[0][2]

        ob, distFromStart, r_t, done, info, end_type = env.step(a_t[0])
        
        ### LAST LAP TIME ###
        if ob.lastLapTime > 0:
            print("lap time is : ",ob.lastLapTime)
            if (ob.lastLapTime < best_lap_time) and (train_indicator):
                best_lap_time = ob.lastLapTime
                rwfile.RW.write_float_to_file(file_path, best_lap_time)
                print("Best Lap Time is updated.")
                print("saving Best model")
                torch.save(actor.state_dict(), 'Best/actormodel.pth')
                torch.save(critic.state_dict(), 'Best/criticmodel.pth')
                
                
        s_t1 = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY, ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm))

        #add to replay buffer
        buff.add(s_t, a_t[0], r_t, s_t1, done)

        batch = buff.getBatch(BATCH_SIZE)

        states = torch.tensor(np.asarray([e[0] for e in batch]), device=device).float()    #torch.cat(batch[0])
        actions = torch.tensor(np.asarray([e[1] for e in batch]), device=device).float()
        rewards = torch.tensor(np.asarray([e[2] for e in batch]), device=device).float()
        new_states = torch.tensor(np.asarray([e[3] for e in batch]), device=device).float()
        dones = np.asarray([e[4] for e in batch])
        y_t = torch.tensor(np.asarray([e[1] for e in batch]), device=device).float()
        
        
        target_q_values1, target_q_values2 = target_critic(new_states, target_actor(new_states))
        target_q_values = torch.min(target_q_values1,target_q_values2)
        
        for k in range(len(batch)):
            if dones[k]:
                y_t[k] = rewards[k]
            else:
                y_t[k] = rewards[k] + GAMMA * target_q_values[k]

        if(train_indicator):
            
            # Get current Q estimates
            q_values1, q_values2 = critic(states, actions)

            # Compute critic loss
            loss = criterion_critic(y_t, q_values1) +  criterion_critic(y_t, q_values2)  

            # Optimize the critic
            optimizer_critic.zero_grad()
            loss.backward(retain_graph=True)                         ##for param in critic.parameters(): param.grad.data.clamp(-1, 1)
            optimizer_critic.step()
            #print("before actor1")
            a_for_grad = actor(states)
            #print("after actor1", a_for_grad)
            a_for_grad.requires_grad_()    #enables the requires_grad of a_for_grad
            q_values_for_grad = critic.Q1(states, a_for_grad)
            critic.zero_grad()
            q_mean = q_values_for_grad.mean()
            q_mean.backward(retain_graph=True)

            grads = torch.autograd.grad(q_mean, a_for_grad) #a_for_grad is not a leaf node  
            #grads is a tuple, while grads[0] is what we want

            #grads[0] = -grads[0]
            #print(grads)   

            act = actor(states)
            actor.zero_grad()
            act.backward(-grads[0])
            optimizer_actor.step()

            #soft update for target network
            #actor_params = list(actor.parameters())
            #critic_params = list(critic.parameters())
            #print("soft updates target network")
            new_actor_state_dict = collections.OrderedDict()
            new_critic_state_dict = collections.OrderedDict()
            for var_name in target_actor.state_dict():
                new_actor_state_dict[var_name] = TAU * actor.state_dict()[var_name] + (1-TAU) * target_actor.state_dict()[var_name]
            target_actor.load_state_dict(new_actor_state_dict)

            for var_name in target_critic.state_dict():
                new_critic_state_dict[var_name] = TAU * critic.state_dict()[var_name] + (1-TAU) * target_critic.state_dict()[var_name]
            target_critic.load_state_dict(new_critic_state_dict)
        
        s_t = s_t1
        steps +=1
        #print("---Episode ", i , "  Action:", a_t, "  Reward:", r_t, "  Loss:", int(loss))
        print("="*100)
        print("--- Episode : {:<4}\tActions ".format(i)+ np.array2string(a_t, formatter={'float_kind': '{0:.3f}'.format})+"\tReward : {:8.4f}\tLoss : {:<8}".format(r_t,int(loss))+" ---")
        print("="*100)
        file_reward.write(str(i) + " "+ str(steps) + " "+ str(distFromStart) + " "+ str(r_t) +" "+ str(a_t[0][0]) +" "+ str(a_t[0][1]) +" "+ str(a_t[0][2]) + " "+ str(int(loss))+"\n") 
        file_states.write(str(i) + " "+ str(steps) + " "+ str(ob.angle)  + " "+ str(ob.trackPos)+ " "+ str(ob.speedX)+ " "+ str(ob.speedY)+ " "+ str(ob.speedZ)+ "\n")
        
        #print(str(i) + " "+ str(steps) + " "+ str(ob.angle)  + " "+ str(ob.trackPos)+ " "+ str(ob.speedX)+ " "+ str(ob.speedY)+ " "+ str(ob.speedZ)+ "\n")
        
        
        if done:
            print("I'm dead")
            break
    file_distances.write(str(i) + " "+ str(distFromStart) +"\n")
    if np.mod(i, 3) == 0:
        if (train_indicator):
            print("saving model")
            torch.save(actor.state_dict(), 'actormodel.pth')
            torch.save(critic.state_dict(), 'criticmodel.pth')

file_reward.close()   
file_distances.close()
file_states.close()
env.end()
print("Finish.")

#for param in critic.parameters(): param.grad.data.clamp(-1, 1)

