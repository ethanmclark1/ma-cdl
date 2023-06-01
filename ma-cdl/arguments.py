import argparse

def get_arguments():
    parser = argparse.ArgumentParser(description='Teach a multi-agent system to create its own context-dependent language.')
    parser.add_argument('--num_agents', metavar='NUM_AGENTS', type=int, default=1)
    parser.add_argument('--render_mode', metavar='RENDER', type=str, default='None', choices=['human', 'rgb_array', 'None'], 
                        help='mode of visualization {default_val: None, choices: [%(choices)s]}')
    args = parser.parse_args()
        
    return args