import argparse
import nashpy
import numpy as np

max_iteration = 100

parser = argparse.ArgumentParser()
parser.add_argument("--attack", default="PGD", help="The adversarial attack")
parser.add_argument("--max_do_iteration", type=int, default=100, help="The max number of iterations of DOAT")
parser.add_argument("--attack_iteration", type=int, default=100, help="The max number of iteration for PGD attacks")
parser.add_argument("--convergence", type=float, default=0.05, help="The convergence threshold")


def run(args):
    predictor_list = []
    dis_predictor_list = []
    dis_perturbator_list = []

    payoff_matrix = np.zeros((2, 3))
    print(payoff_matrix)

    for i in range(args.max_do_iteration):
        print(payoff_matrix)


if __name__ == '__main__':
    args = parser.parse_args()
    run(args)
