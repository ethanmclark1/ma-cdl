import argparse

def get_arguments():
    parser = argparse.ArgumentParser(description='Teach a multi-agent system to create its own context-dependent language.')
    parser.add_argument('--obs_size', metavar='SIZE', type=float, default=0.2,
                        help='size/radius of obstacles {default_val: %(default)s}')
    parser.add_argument('--problem', metavar='PROBLEM', type=str, default='cluster', 
                        choices=['cluster', 'L-shaped', 'vertical', 'horizontal', 'left', 'right', 'up', 'down', 'random'],
                        help='choose problem setup for obstacles {default_val: %(default)s, choices: [%(choices)s]}')
    parser.add_argument('--render_mode', metavar='RENDER', type=str, default='human', choices=['human', 'rgb_array', 'None'], 
                        help='mode of visualization {default_val: None, choices: [%(choices)s]}')
    args = parser.parse_args()
        
    return args