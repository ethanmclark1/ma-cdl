import argparse

def get_arguments():
    parser = argparse.ArgumentParser(
        description='Teach a multi-agent system to create its own context-dependent language.')
    parser.add_argument('--start_constr', metavar='START', type=int, nargs='+', default=[3], choices=[0,1,2,3,4],
                        help='quadrant(s) to constrain start position to {default_val: %(default)s, choices: [%(choices)s]}')
    parser.add_argument('--goal_constr', metavar='GOAL', type=int, nargs='+', default=[4], choices=[0,1,2,3,4], 
                        help='quadrant(s) to constrain goal position to {default_val: %(default)s, choices: [%(choices)s]}')
    parser.add_argument('--num_obs', metavar='N', type=int, default=1,
                        help='number of obstacles in environment {default_val: %(default)s}')
    parser.add_argument('--obs_constr', metavar='OBS', type=int, nargs='+', default=[1], choices=[0,1,2,3,4],
                        help='quadrant(s) to constrain obstacle position(s) to {default_val: %(default)s, choices: [%(choices)s]}')
    parser.add_argument('--obs_size', metavar='SIZE', type=float, default=0.02,
                        help='size of obstacles in environment {default_val: %(default)s}')
    parser.add_argument('--render_mode', metavar='MODE', type=str, default='None', choices=['human', 'rgb_array', 'None'], 
                        help='mode of visualization {default_val: None, choices: [%(choices)s]}')
    args = parser.parse_args()
    
    pos_constr = [(1, 1), (1, -1), (-1, -1), (-1, 1)]
    args.start_constr = pos_constr if 0 in args.start_constr else list(map(lambda x: pos_constr[x-1], args.start_constr))
    args.goal_constr = pos_constr if 0 in args.goal_constr else list(map(lambda x: pos_constr[x-1], args.goal_constr))
    args.obs_constr = pos_constr if 0 in args.obs_constr else list(map(lambda x: pos_constr[x-1], args.obs_constr))

    return args