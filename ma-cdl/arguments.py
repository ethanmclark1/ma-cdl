import argparse

def get_arguments():
    parser = argparse.ArgumentParser(description='Teach a multi-agent system to create its own context-dependent language.')
    parser.add_argument('--num_agents', metavar='NUM_AGENTS', type=int, default=1,
                        help='number of agents in the environment {default_val: 1}')
    parser.add_argument('--num_large_obstacles', metavar='NUM_LARGE_OBSTACLES', type=int, default=4,
                        help='number of large obstacles in the environment (no more than 4) {default_val: 4}')
    parser.add_argument('--num_small_obstacles', metavar='NUM_SMALL_OBSTACLES', type=int, default=10,
                        help='number of small obstacles in the environment {default_val: 10}')
    parser.add_argument('--render_mode', metavar='RENDER', type=str, default='None', choices=['human', 'rgb_array', 'None'], 
                        help='mode of visualization {default_val: None, choices: [%(choices)s]}')
    args = parser.parse_args()
        
    return args.num_agents, args.num_large_obstacles, args.num_small_obstacles, args.render_mode