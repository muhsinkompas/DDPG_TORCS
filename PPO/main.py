import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import gym
from ppo import *
from gym_torcs import TorcsEnv
import rwfile

#----------------------------------------------------------------------------------------

EP_MAX = 2000
EP_LEN = 1000
GAMMA = 0.99


A_LR = 1*1e-4
C_LR = 1*1e-3

BATCH = 32 
A_UPDATE_STEPS = 10
C_UPDATE_STEPS = 10
S_DIM, A_DIM = 29, 3
METHOD = [
    dict(name='kl_pen', kl_target=0.01, lam=1.0),   # KL penalty; lam is actually beta from the PPO paper
    dict(name='clip', epsilon=0.10),           # Clipped surrogate objective, find this is better
][1]        # choose the method for optimization
#old eps =0.1

# train_test = 0 for train; =1 for test
train_test = 0

# irestart = 0 for fresh restart; =1 for restart from file
irestart = 0

iter_num = 0

relaunch_le = 15 #relaunch for leak memory error.

if (irestart == 0):
    iter_num = 0

#----------------------------------------------------------------------------------------

sess = tf.Session()

ppo = PPO(sess, S_DIM, A_DIM, A_LR, C_LR, A_UPDATE_STEPS, C_UPDATE_STEPS, METHOD)

saver = tf.train.Saver()

env = TorcsEnv(vision=False, throttle=True, gear_change=False)

#----------------------------------------------------------------------------------------

file_path = 'Best/bestlaptime.csv'
best_lap_time = rwfile.RW.read_float_from_file(file_path)


#----------------------------------------------------------------------------------------

if (train_test == 0 and irestart == 0):
    sess.run(tf.global_variables_initializer())
else:
    saver.restore(sess, "weights/model")  


for ep in range(iter_num, EP_MAX):

    #print("-"*50)
    #print("episode: ", ep)

    if np.mod(ep, relaunch_le) == 0:
        ob = env.reset(relaunch=True)   #relaunch TORCS every N episode because of the memory leak error
    else:
        ob = env.reset()

    s = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY, ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm))

    buffer_s, buffer_a, buffer_r = [], [], []
    ep_r = 0

    for t in range(EP_LEN):    # in one episode
        a = ppo.choose_action(s)
        #print(a)
        a[0] = np.clip(a[0],-1.0,1.0)
        a[1] = np.clip(a[1],0.0,1.0)
        a[2] = np.clip(a[2],0.0,1.0)  
        #print(a)
        ob, r, done, _, end_type = env.step(a)
        
        ### LAST LAP TIME ###
        if ob.lastLapTime > 0:
            print("lap time is : ",ob.lastLapTime)
            if (ob.lastLapTime < best_lap_time) and (train_test==0):
                best_lap_time = ob.lastLapTime
                rwfile.RW.write_float_to_file(file_path, best_lap_time)
                saver.save(sess, "Best/model")
                print("Best Lap Time is updated.")
                print("saving Best model")
        
        s_ = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY, ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm))  
        if (train_test == 0):
            buffer_s.append(s)
            buffer_a.append(a)
            buffer_r.append(r)    
        s = s_
        ep_r += r
        if (train_test == 0):
            with open("results/results_ppo0.txt", "a") as myfile:
                myfile.write(str(ep) + " " + str(t) + " " + str(r) +" " +  str(r) + " " + str(a[0]) +" " + str(a[1]) +" " + str(a[2]) + " "+ str(ob.distRaced)+str("\n"))
        
        if (train_test == 0):
            with open("results/states_ppo0.txt", "a") as file_states:
                file_states.write(str(ep) + " "+ str(t) + " "+ str(ob.angle)  + " "+ str(ob.trackPos)+ " "+ str(ob.speedX)+ " "+ str(ob.speedY)+ " "+ str(ob.speedZ)+ "\n")

            if (t+1) % BATCH == 0 or t == EP_LEN-1 or done == True:
                v_s_ = ppo.get_v(s_)
                discounted_r = []
                for r in buffer_r[::-1]:
                    v_s_ = r + GAMMA * v_s_
                    discounted_r.append(v_s_)
                discounted_r.reverse()
                bs = np.array(np.vstack(buffer_s))
                ba = np.array(np.vstack(buffer_a))  
                br = np.array(discounted_r)[:, np.newaxis]
                buffer_s, buffer_a, buffer_r = [], [], []

                #print("ppo update")              
                ppo.update(bs, ba, br)
                
        print('Ep: %i' % ep,"|Ep_r: %i" % ep_r,("|Lam: %.4f" % METHOD['lam']) if METHOD['name'] == 'kl_pen' else '',)
        
        print("="*100)
        print("--- Episode : {:<4}\tActions ".format(ep)+ np.array2string(a, formatter={'float_kind': '{0:.3f}'.format})+"\tReward : {:8.4f}".format(ep_r)+" ---")
        print("="*100)
        
        if (done  == True):
            break
        	
    if (train_test == 0 and ep%25 == 0):
        saver.save(sess, "weights/model")

