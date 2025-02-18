import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import torch
from VAE import VAE_dataset,VariationalAutoEncoder
from SAC_3 import SACAgent,ReplayBufferHER
from PIL import Image
from torchvision import transforms
import cv2
import random


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vae_model = VariationalAutoEncoder()
vae_model.load_state_dict(torch.load('vae_model_new.pth'))
vae_model.eval() 

env = gym.make(
    'Pusher-v5',
    render_mode="rgb_array",
    default_camera_config={
        "distance": 2.2,
        "azimuth": 0.0,
        "elevation": 90.0,
        "lookat": [0.2, 0, 0]
    }
)

def reparameterize(mean, log_var):
    std = torch.exp(0.5 * log_var)
    eps = torch.randn_like(std)
    return mean + eps * std


def proximity_reward(mask,EE_position):
    #I'll use the ratio found in the inizio.py file to translate the image centroid position to the reference system of the End effector
    M = cv2.moments(mask)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    cX_ratio = cX*0.01412576
    cY_ratio = cY*(-0.00123944)
    cZ = 0
    c = np.array([cX_ratio,cY_ratio,cZ])
    return np.linalg.norm(EE_position - c)
# def image_processing(image):
#     image = Image.fromarray(starting_frame)#.transform.ToTensor().Resize(64,64)
#     transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Resize((64, 64))
#     ])
    
#     return transform(image)


def image_processing(image):
    # Ridimensiona l'immagine a 64x64 pixel mantenendo il formato NumPy (BGR)
    image_resized = cv2.resize(image, (64, 64))
    return image_resized

def goal_func(next_state):
    """ Estrae un goal alternativo dallo stato successivo per il relabeling di HER. """
    #noise = torch.randn_like(next_state[:, 3:9]) * 0.01  # Deviazione standard 0.01
    #new_state = next_state[:, 3:9] + noise
    return next_state[:, 3:9]


epochs = 3
cycles = 50
episodes = 16
steps = 200 #defaultfor the environment
#actor_net = Actor(state_dim=5) #3 per EE e 2 per le due distribuzioni
#critic_net = Critic(state_action_dim=5+7) #state+action
replay_buffer = ReplayBufferHER(capacity=1000000, goal_func=goal_func)
agent = SACAgent(state_dim=15,action_dim=7)
agent.replay_buffer = replay_buffer
episode_rewards = []
mask_extraction = VAE_dataset(image_paths=[], target_colors=[])
for epoch in range(epochs):
    episode_reward = []
    for cycle in range(cycles):
        batch_episode_transitions = []
        for episode in range(episodes):
            episode_transitions = []
            state_pusher, _ = env.reset()
            starting_frame = env.render()
            image = image_processing(starting_frame)
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            input_goal = mask_extraction.mask_extraction(image_bgr,(0, 0, 255))
            # plt.imshow(input_goal)
            # plt.show()
            input_goal = torch.tensor(input_goal, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            with torch.no_grad():
                goal_mean, goal_log_var = vae_model.encoder(input_goal)
                #goal_latent = reparameterize(goal_mean, goal_log_var)
            input_object = mask_extraction.mask_extraction(image,(0, 255, 0))
            input_object = torch.tensor(input_object, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            with torch.no_grad():
                object_mean, object_log_var = vae_model.encoder(input_object)
                #object_latent = reparameterize(object_mean, object_log_var)
            state_EE = state_pusher[14:17]
            state_EE = torch.tensor(state_EE, dtype=torch.float32).unsqueeze(0)
            state = torch.cat([state_EE, object_mean, goal_mean], dim=1)#.latent
            state = (state - state.mean(dim=1, keepdim=True)) / (state.std(dim=1, keepdim=True) + 1e-8)
            #object_states = []
            # state_object = state_pusher[17:19] ############
            # state_object = np.append(state_object, -0.275) 
            #object_states.append(state_object) 

            for step in range(steps):
                if random.randint(1, 100) < 20:
                    action = env.action_space.sample()  # Campionamento casuale
                # else:
                action_tensor,_ = agent.policy.get_action(state)
                action = action_tensor.detach().cpu().numpy().flatten()
                # action = action*2
                state_pusher, _, _, _, _ = env.step(action)
                # Compute next state
                frame = env.render()
                state_EE = state_pusher[14:17]
                # state_object = state_pusher[17:19] ############
                # state_object = np.append(state_object, -0.275) 
                #object_states.append(state_EE) 
                input_image = image_processing(frame)
                input_image_bgr = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)
                input_object = mask_extraction.mask_extraction(input_image_bgr,(0, 255, 0))
                # plt.imshow(input_object)
                # plt.show()
                # first_reward = proximity_reward(input_object,state_EE)
                # first_reward = np.linalg.norm(state_EE - state_object)
                # if first_reward < 0.20:  # Soglia arbitraria (es. 10 cm)
                #     #terminated = 4
                #     done = False
                #     # print("MIIIIIIIIII")
                # if first_reward < 0.15:  # Soglia arbitraria (es. 10 cm)
                #     terminated = 8
                #     done = False
                #     print("MEEEEEEEEE")
                # if first_reward < 0.10:  # Soglia arbitraria (es. 10 cm)
                #     terminated = 20
                #     done = False
                #     print("MAROOO")
                # if first_reward < 0.05:  # Soglia arbitraria (es. 10 cm)
                #     terminated = 30
                #     done = False
                #     #print("Bingoooooo")
                # else:
                #     terminated = 0
                
                input_object = torch.tensor(input_object, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                with torch.no_grad():
                    object_mean, object_log_var = vae_model.encoder(input_object)
                    #object_latent = reparameterize(object_mean, object_log_var)
                # Compute reward based on latent space difference
                second_reward = -torch.norm(object_mean-goal_mean).numpy()#latent


                if second_reward < -0.20:
                    done = False
                    reward = -1
                else:
                    done = True
                    #print("assurdo")
                # if step != 511:
                #     done = False
                # else:
                #     done = True
                episode_reward.append(reward)
                state_EE = torch.tensor(state_EE, dtype=torch.float32).unsqueeze(0)
                next_state = torch.cat([state_EE, object_mean, goal_mean], dim=1)#latent
                next_state = (next_state - next_state.mean(dim=1, keepdim=True)) / (next_state.std(dim=1, keepdim=True) + 1e-8)
                episode_transitions.append((state, action, reward, next_state, done, goal_mean))
                # print("Primo", episode_transitions[0][0][:,3:9])
                # print("Ultimo", episode_transitions[-1][3][:,3:9])
                # print((episode_transitions[0][0][:,3:9] - episode_transitions[-1][3][:,3:9]).norm().item())
                if step == 199 and ((episode_transitions[0][0][:,3:9] - episode_transitions[-1][3][:,3:9]).norm().item() < 0.1) :
                    episode_transitions = []
                    episode_reward = []
                    print("helo")
                state = next_state
                if done:
                    break
            
            episode_rewards.append(sum(episode_reward))
            batch_episode_transitions.extend(episode_transitions)
            #ep_mean = np.mean(episode_reward)
            
   
        
        agent.replay_buffer.add_episode(batch_episode_transitions)
        agent.update()
        agent.soft_update_targets()
        avg_reward = np.mean(episode_rewards)

        # if episode % 10 == 0:
        print(f"Cycle {cycle}: Average Reward: {avg_reward:.2f}")
    
    # if episode % 300 == 0:
    # save_filename1 = f'sac_policy_{epoch}_HER.pt'
    save_filename1 = f'/kaggle/working/sac_policy_{epoch}_HER_leggero.pt'
    save_filename2 = f'/kaggle/working/sac_q1_{epoch}_HER.pt'
    save_filename3 = f'/kaggle/working/sac_q2_{epoch}_HER.pt'
    torch.save(agent.policy.state_dict(), save_filename1)
    torch.save(agent.q1.state_dict(), save_filename2)
    torch.save(agent.q2.state_dict(), save_filename3)
    print(f"Model saved after {epoch} episodes")
env.close()
            







