import argparse

def get_arguments():
    parser = argparse.ArgumentParser(
        description='Teach a multi-agent system to create its own context-dependent language.'
        )
    
    parser.add_argument(
        '--num_agents', 
        type=int, 
        default=1,
        help='Number of agents in the environment {default_val: 1}'
        )
    
    parser.add_argument(
        '--num_large_obstacles',
        type=int, 
        default=10,
        help='Number of large obstacles in the environment (no more than 16) {default_val: 6}'
        )
    
    parser.add_argument(
        '--num_small_obstacles', 
        type=int, 
        default=4,
        help='Number of small obstacles in the environment {default_val: 10}'
        )
    
    parser.add_argument(
        '--render_mode', 
        type=str, 
        default='None', 
        choices=['human', 'rgb_array', 'None'], 
        help='Mode of visualization {default_val: None, choices: [%(choices)s]}'
        )
    
    args = parser.parse_args()
        
    return args.num_agents, args.num_large_obstacles, args.num_small_obstacles, args.render_mode