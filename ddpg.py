from gym_torcs import TorcsEnv
import numpy as np
import random
import argparse
from keras.models import model_from_json, Model
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam
import tensorflow as tf
from keras.engine.training import collect_trainable_weights
import json

from ReplayBuffer import ReplayBuffer
from ActorNetwork import ActorNetwork
from CriticNetwork import CriticNetwork
from OU import OU
import timeit
import rwfile
import wOutputToCsv as w_Out
import os


OU = OU()       #Ornstein-Uhlenbeck Process

def playGame(train_indicator=0):    #1 means Train, 0 means simply Run
    BUFFER_SIZE = 100000
    BATCH_SIZE = 32
    GAMMA = 0.99
    TAU = 0.001     #Target Network HyperParameters
    LRA = 0.0001    #Learning rate for Actor
    LRC = 0.001     #Lerning rate for Critic

    action_dim = 3  #Steering/Acceleration/Brake
    state_dim = 29  #of sensors input

    np.random.seed(1337)

    vision = False

    EXPLORE = 100000.
    episode_count = 4000
    max_steps = 100000
    reward = 0
    done = False
    step = 0
    epsilon = 1
    indicator = 0

    #Tensorflow GPU optimization
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    from keras import backend as K
    K.set_session(sess)
    
    #----------------------------------------------------------------------------------------
    file_path = 'Best/bestlaptime.csv'
    r_w = rwfile.RW(file_path)
    best_lap_time = r_w.read_numpy_array_from_csv()
    print(best_lap_time)
    
    w_csv = w_Out.OW(csv_path = 'OutputCsv/output.csv',headers = ['ep', 'step', 'a_1', 'a_2', 'a_3' , 'reward', 
                                                              's_1', 's_2', 's_3', 's_4', 's_5', 's_6', 's_7', 's_8', 's_9', 's_10',
                                                              's_11', 's_12', 's_13', 's_14', 's_15', 's_16', 's_17', 's_18', 's_19', 's_20',
                                                              's_21', 's_22', 's_23', 's_24', 's_25', 's_26', 's_27', 's_28', 's_29', 
                                                              'end_type', 'f_1', 'f_2', 'f_3', 'f_4', 'f_5', 'distRaced', 'distFromStart',
                                                              'curLapTime', 'lastLapTime', 'loss'])
    w_total_csv = w_Out.OW(csv_path = 'OutputCsv/output_total.csv',headers = ['ep', 'step', 'end_type', 'col_count', 'oot_count', 'np_count', 
                                                                          'wrong_direction', 'speedX', 'distRaced', 'distFromStart', 'last_lap_distance', 
                                                                          'curLapTime', 'lastLapTime', 'total_reward'])
    w_event_csv = w_Out.OW(csv_path = 'OutputCsv/event_history.csv',headers = ['ep', 'step', 'col_count', 'oot_count', 'np_count', 
                                                                          'wrong_direction', 'distFromStart'])

    #----------------------------------------------------------------------------------------
    
    actor = ActorNetwork(sess, state_dim, action_dim, BATCH_SIZE, TAU, LRA)
    critic = CriticNetwork(sess, state_dim, action_dim, BATCH_SIZE, TAU, LRC)
    buff = ReplayBuffer(BUFFER_SIZE)    #Create replay buffer

    # Generate a Torcs environment
    env = TorcsEnv(vision=vision, throttle=True,gear_change=False)

    #Now load the weight
    print("Now we load the weight")
    try:
        actor.model.load_weights("actormodel.h5")
        critic.model.load_weights("criticmodel.h5")
        actor.target_model.load_weights("actormodel.h5")
        critic.target_model.load_weights("criticmodel.h5")
        print("Weight load successfully")
    except:
        print("Cannot find the weight")

    print("TORCS Experiment Start.")
    for i in range(episode_count):

        print("Episode : " + str(i) + " Replay Buffer " + str(buff.count()))

        if np.mod(i, 20) == 0:
            ob = env.reset(relaunch=True)   #relaunch TORCS every 3 episode because of the memory leak error
        else:
            ob = env.reset()

        s_t = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY,  ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm))
        last_lap_distance = 0
        total_reward = 0.
        event_counts = np.array([0, 0, 0, 0])
        event_list = np.array([0, 0, 0, 0, 0, 0, 0])
        for j in range(max_steps):
            loss = 0 
            epsilon -= 1.0 / EXPLORE
            a_t = np.zeros([1,action_dim])
            noise_t = np.zeros([1,action_dim])
            
            a_t_original = actor.model.predict(s_t.reshape(1, s_t.shape[0]))
            noise_t[0][0] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][0],  0.0 , 0.60, 0.30)
            noise_t[0][1] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][1],  0.5 , 1.00, 0.10)
            noise_t[0][2] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][2], -0.1 , 1.00, 0.05)

            #The following code do the stochastic brake
            #if random.random() <= 0.1:
            #    print("********Now we apply the brake***********")
            #    noise_t[0][2] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][2],  0.2 , 1.00, 0.10)

            a_t[0][0] = a_t_original[0][0] + noise_t[0][0]
            a_t[0][1] = a_t_original[0][1] + noise_t[0][1]
            a_t[0][2] = a_t_original[0][2] + noise_t[0][2]

            ob, r_t, done, info , end_type, event_buff = env.step(a_t[0])
            event_counts = event_counts + event_buff
            if np.sum(event_buff) > 0:
                event_list_buff = np.hstack((i, j, event_buff, ob.distFromStart))
                event_list = np.vstack((event_list, event_list_buff))

            
            ### LAST LAP TIME & saving best model----------------------------------------###
            if ob.lastLapTime > 0:
                print("lap time is : ",ob.lastLapTime)
                if (ob.lastLapTime < best_lap_time) and (train_indicator==1):
                    best_lap_time = ob.lastLapTime
                    r_w.write_numpy_array_to_csv(best_lap_time)
                    print("Now we save model")
                    actor.model.save_weights("Best/actormodel.h5", overwrite=True)
                    with open("Best/actormodel.json", "w") as outfile:
                        json.dump(actor.model.to_json(), outfile)

                    critic.model.save_weights("Best/criticmodel.h5", overwrite=True)
                    with open("Best/criticmodel.json", "w") as outfile:
                        json.dump(critic.model.to_json(), outfile)
                    print("Best Lap Time is updated.")
                    print("saving Best model")
            ###------------------------------------------------------------------------###
            
            s_t1 = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY, ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm))
        
            buff.add(s_t, a_t[0], r_t, s_t1, done)      #Add replay buffer
            
            #Do the batch update
            batch = buff.getBatch(BATCH_SIZE)
            states = np.asarray([e[0] for e in batch])
            actions = np.asarray([e[1] for e in batch])
            rewards = np.asarray([e[2] for e in batch])
            new_states = np.asarray([e[3] for e in batch])
            dones = np.asarray([e[4] for e in batch])
            y_t = np.asarray([e[1] for e in batch])

            target_q_values = critic.target_model.predict([new_states, actor.target_model.predict(new_states)])  
           
            for k in range(len(batch)):
                if dones[k]:
                    y_t[k] = rewards[k]
                else:
                    y_t[k] = rewards[k] + GAMMA*target_q_values[k]
       
            if (train_indicator):
                loss += critic.model.train_on_batch([states,actions], y_t) 
                a_for_grad = actor.model.predict(states)
                grads = critic.gradients(states, a_for_grad)
                actor.train(states, grads)
                actor.target_train()
                critic.target_train()

            total_reward += r_t
            s_t = s_t1
        
            print("="*100)
            print("--- Episode : {:<4}\tActions ".format(i)+ np.array2string(a_t, formatter={'float_kind': '{0:.3f}'.format})+"\tReward : {:8.4f}\tLoss : {:<8}".format(r_t,int(loss))+" ---")
            print("="*100)
        
            if ob.distFromStart > last_lap_distance:
                last_lap_distance = ob.distFromStart
            
            #----------------------------------------------------------------------------------------------------------------
            # Saving outputs to csv file
            #print("saving csv")
            output_csv = np.hstack((i, j, a_t[0], r_t, s_t, end_type, ob.focus, ob.distRaced, ob.distFromStart , ob.curLapTime, ob.lastLapTime, loss))
            w_csv.append_numpy_array_to_csv(output_csv)
            #----------------------------------------------------------------------------------------------------------------
            
            
            step += 1
            if done:
                break

        if np.mod(i, 4) == 0 or i==episode_count:
            if (train_indicator):
                print("saving model")
                actor.model.save_weights("actormodel.h5", overwrite=True)
                with open("actormodel.json", "w") as outfile:
                    json.dump(actor.model.to_json(), outfile)

                critic.model.save_weights("criticmodel.h5", overwrite=True)
                with open("criticmodel.json", "w") as outfile:
                    json.dump(critic.model.to_json(), outfile)
                    
        ### Saving models each 100 episodes --------------------------------------------- ###         
        if np.mod(i,100)==99:
            if (train_indicator):
                print("saving model ")
                file_name = str(i+1)
                file_name = "Models/"+file_name
                if os.path.isdir(file_name)==False:
                    os.makedirs(file_name)
                actor_name = file_name+"/actormodel.h5"
                t_actor_name = file_name+"/actormodel.json"
                critic_model = file_name+"/criticmodel.h5"
                t_critic_model = file_name+"/criticmodel.json"
                actor.model.save_weights(actor_name, overwrite=True)
                with open(t_actor_name, "w") as outfile:
                    json.dump(actor.model.to_json(), outfile)

                critic.model.save_weights(critic_model, overwrite=True)
                with open(t_critic_model, "w") as outfile:
                    json.dump(critic.model.to_json(), outfile)
        ### ----------------------------------------------------------------------------- ###
        
        ### Saving total outputs for each episode --------------------------------------- ###
        # EDIT HERE AFTER
        output_total_csv = np.hstack((i, j, end_type, event_counts, ob.speedX, ob.distRaced, ob.distFromStart, last_lap_distance, ob.curLapTime, ob.lastLapTime, total_reward))
        w_total_csv.append_numpy_array_to_csv(output_total_csv)
        w_event_csv.append_numpy_array_to_csv(event_list)
        ### ----------------------------------------------------------------------------- ###
        
        print("TOTAL REWARD @ " + str(i) +"-th Episode  : Reward " + str(total_reward))
        print("Total Step: " + str(step))
        print("")

    env.end()  # This is for shutting down TORCS
    print("Finish.")

if __name__ == "__main__":
    playGame(train_indicator=1)
