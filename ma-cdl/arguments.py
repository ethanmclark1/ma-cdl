import argparse

def get_arguments():
    parser = argparse.ArgumentParser(
        description='Teach a multi-agent system to create its own context-dependent language.')
    parser.add_argument('--env_shape', metavar='ENV', type=str, default='Square', choices=[
                        'square', 'triangle', 'circle'], help='shape of environment space (choices: %(choices)s)')
    parser.add_argument('--num_obstacles', metavar='N', type=int, default=2,
                        help='number of obstacles in environment (default: %(default)s)')
    parser.add_argument('--min_symbol', metavar='MIN', type=int, default=2,
                        help='min number of rows/columns a language can represent (default: %(default)s)')
    parser.add_argument('--max_symbol', metavar='MAX', type=int, default=5,
                        help='max number of rows/columns a language can represent (default: %(default)s)')
    parser.add_argument('--render_mode', metavar='R', type=str, default='None', choices=[
                        'human', 'rgb_array'], help='mode of visualization (choices: %(choices)s)')
    args = parser.parse_args()
    return args