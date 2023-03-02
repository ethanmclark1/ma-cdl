import argparse

def get_arguments():
    parser = argparse.ArgumentParser(
        description='Teach a multi-agent system to create its own context-dependent language.')
    parser.add_argument('--num_obstacles', metavar='N', type=int, default=1,
                        help='number of obstacles in environment (default: %(default)s)')
    parser.add_argument('--obstacle_size', metavar='SIZE', type=float, default=0.02,
                        help='size of obstacles in environment (default: %(default)s)')
    parser.add_argument('--num_languages', metavar='K', type=int, default=8,
                        help='number of possible languages to search through (default: %(default)s)')
    parser.add_argument('--render_mode', metavar='MODE', type=str, default='None', choices=[
                        'human', 'rgb_array'], help='mode of visualization (choices: %(choices)s)')
    args = parser.parse_args()
    return args