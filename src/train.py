import turtlesim_env_single
from dqn_single import DqnSingle

NUMBER_OF_AGENTS = 1
ROUTES_FILENAME = '/root/siu/routes.csv'

LOAD_MODEL = False
MODEL_FILEPATH = '/root/siu/models/model.h5'


def main():
    env = turtlesim_env_single.provide_env()
    env.setup(routes_fname=ROUTES_FILENAME, agent_cnt=NUMBER_OF_AGENTS)

    agents = env.reset()
    tname = list(agents.keys())[0]

    dqn = DqnSingle(env, 'train')
    if LOAD_MODEL:
        dqn.model.load(MODEL_FILEPATH)
    else:
        dqn.make_model()

    dqn.train_main(tname, save_model=True)


if __name__ == "__main__":
    main()
