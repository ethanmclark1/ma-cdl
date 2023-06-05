import argparse

def get_arguments():
    parser = argparse.ArgumentParser(description='Teach a multi-agent system to create its own context-dependent language.')
    parser.add_argument('--problem_type', metavar='PROB_TYPE', type=str, default='disaster_response', choices=['disaster_response', 'precision_farming'],
                        help='type of problem {default_val: disaster_response, choices: [%(choices)s]}')
    parser.add_argument('--num_agents', metavar='NUM_AGENTS', type=int, default=1,
                        help='number of agents in the environment {default_val: 1}')
    parser.add_argument('--render_mode', metavar='RENDER', type=str, default='None', choices=['human', 'rgb_array', 'None'], 
                        help='mode of visualization {default_val: None, choices: [%(choices)s]}')
    args = parser.parse_args()
        
    return args.problem_type, args.num_agents, args.render_mode