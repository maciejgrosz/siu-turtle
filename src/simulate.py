import turtlesim_env_single
from dqn_single import DqnSingle
import tensorflow as tf
import numpy as np

NUMBER_OF_AGENTS = 1
ROUTES_FILENAME = '/root/siu/routes.csv'
MODEL_FILEPATH = '/root/siu/models/Edycja1/train-Gr5_Cr200_Sw0.5_Sv-10.0_Sf-10.0_Dr2.0_Oo-10_Cd1.5_Ms20_Pb6_D0.9_E0.99_e0.05_M20000_m400_B32_U20_P4000_T4.h5'


def main():
    env = turtlesim_env_single.provide_env()
    env.setup(routes_fname=ROUTES_FILENAME, agent_cnt=NUMBER_OF_AGENTS)

    agents = env.reset()
    tname = list(agents.keys())[0]

    dqn = DqnSingle(env, 'simulate')
    dqn.model = tf.keras.models.load_model(MODEL_FILEPATH)

    current_state = agents[tname].map
    last_state = [i.copy() for i in current_state]
    for step in range(1000):
        control = np.argmax(dqn.decision(dqn.model, last_state, current_state))
        last_state = current_state
        current_state, reward, done = env.step({tname: dqn.ctl2act(int(control))})
        # print(done)

if __name__ == "__main__":
    main()
