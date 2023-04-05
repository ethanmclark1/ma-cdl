import argparse

def get_problem_info(problem):
    problem_types = {
        'cluster': {
            'start': ((-1, -0.80), (-0.25, 0.25)),
            'goal': ((0.80, 1), (-0.25, 0.25)),
            'obs': ((-0.15, 0.15), (-0.15, 0.15))
        },
        'L-shaped': {
            'start': ((-1, -0.80), (-0.25, 0.25)),
            'goal': ((0.4, 0.6), (0.4, 0.6)),
            'obs': [((-0.1, 0.1), (0, 0.6)), ((0.1, 0.5), (0, 0.25))]
        },
        'vertical': {
            'start': ((-1, -0.80), (-1, 1)),
            'goal': ((0.80, 1), (-1, 1)),
            'obs': ((-0.075, 0.075), (-0.6, 0.6))
        },
        'horizontal': {
            'start': ((-1, 1), (-1, -0.80)),
            'goal': ((-1, 1), (0.80, 1)),
            'obs': ((-0.6, 0.6), (-0.075, 0.75))
        },
        'left': {
            'start': ((0, 1), (-1, -0.80)),
            'goal': ((0, 1), (0.80, 1)),
            'obs': ((-1, 0), (-1, 1))
        },
        'right': {
            'start': ((-1, 0), (-1, -0.80)),
            'goal': ((-1, 0), (0.80, 1)),
            'obs': ((0, 1), (-1, 1))
        },
        'up': {
            'start': ((-1, 0.80), (-1, 0)),
            'goal': ((0.80, 1), (-1, 0)),
            'obs': ((-1, 1), (0, 1))
        },
        'down': {
            'start': ((-1, 0.80), (0, 1)),
            'goal': ((0.80, 1), (0, 1)),
            'obs': ((-1, 1), (-1, 0))
        },
        'random': {
          'start': ((-1, 1), (-1, 1)),
          'goal': ((-1, 1), (-1, 1)),
          'obs': ((-1, 1), (-1, 1))  
        }
    }
    problem_info = problem_types[problem]
    start = problem_info['start']
    goal = problem_info['goal']
    obs = problem_info['obs']
    return start, goal, obs

def get_arguments():
    parser = argparse.ArgumentParser(
        description='Teach a multi-agent system to create its own context-dependent language.')
    parser.add_argument('--obs_size', metavar='SIZE', type=float, default=0.2,
                        help='size/radius of obstacles {default_val: %(default)s}')
    parser.add_argument('--problem', metavar='PROBLEM', type=str, default='cluster', 
                        choices=['cluster', 'L-shaped', 'vertical', 'horizontal', 'left', 'right', 'up', 'down', 'random'],
                        help='choose problem setup for obstacles {default_val: %(default)s, choices: [%(choices)s]}')
    parser.add_argument('--render_mode', metavar='RENDER', type=str, default='human', choices=['human', 'rgb_array', 'None'], 
                        help='mode of visualization {default_val: None, choices: [%(choices)s]}')
    args = parser.parse_args()

    start_constr, goal_constr, obs_constr = get_problem_info(args.problem)
    
    setattr(args, 'start_constr', start_constr)
    setattr(args, 'goal_constr', goal_constr)
    setattr(args, 'obs_constr', obs_constr)
    return args