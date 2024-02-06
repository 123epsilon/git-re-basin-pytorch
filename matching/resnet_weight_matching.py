from utils.weight_matching import weight_matching, apply_permutation, resnet20_permutation_spec, resnet50_permutation_spec
from utils.utils import  lerp
from utils.plot import plot_interp_acc
import argparse
import torch
from models.resnet import ResNet
from torchvision import datasets, transforms
from utils.training import test
from utils.data import get_cifar10, get_cifar100, get_mnist
from tqdm import tqdm
import copy
import matplotlib.pyplot as plt
import os


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--model_a", type=str, required=True)
	parser.add_argument("--model_b", type=str, required=True)
	parser.add_argument('--tag', type=str, required=True)
	parser.add_argument("--seed", type=int, default=0, help="Random seed")
	parser.add_argument('--width-multiplier', type=int, default=2)
	parser.add_argument('--depth', type=int, default=22)
	parser.add_argument('--output-dir', type=str, default="./figs")
	parser.add_argument('--dataset', type=str, default='cifar100', choices=['cifar10', 'cifar100', 'mnist'])
	
	args = parser.parse_args()

	# load dataset
	if args.dataset == "cifar10":
		trainset, testset = get_cifar10()
		n_classes = 10
	elif args.dataset == "cifar100":
		trainset, testset = get_cifar100()
		n_classes = 100
	elif args.dataset == "mnist":
		trainset, testset = get_mnist()
		n_classes = 10
	else:
		raise Exception(f"Invalid Dataset: {args.dataset}")

	# load models
	model_a = ResNet(args.depth, args.width_multiplier, 0, num_classes=n_classes)
	model_b = ResNet(args.depth, args.width_multiplier, 0, num_classes=n_classes)
	checkpoint = torch.load(args.model_a)
	model_a.load_state_dict(checkpoint)   
	checkpoint_b = torch.load(args.model_b)
	model_b.load_state_dict(checkpoint_b)

	if args.depth == 22:
		permutation_spec = resnet20_permutation_spec()
	elif args.depth == 52:
		permutation_spec = resnet50_permutation_spec()
	else:
		print("invalid depth")
		return

	final_permutation = weight_matching(permutation_spec,
										model_a.state_dict(), model_b.state_dict())
			  

	updated_params = apply_permutation(permutation_spec, final_permutation, model_b.state_dict())



	train_loader = torch.utils.data.DataLoader(trainset, batch_size=1024,
											shuffle=True, num_workers=2)

	test_loader = torch.utils.data.DataLoader(testset, batch_size=1024,
											shuffle=False, num_workers=2)

	lambdas = torch.linspace(0, 1, steps=25)

	test_acc_interp_clever = []
	test_acc_interp_naive = []
	train_acc_interp_clever = []
	train_acc_interp_naive = []
	# naive
	model_b.load_state_dict(checkpoint_b)
	model_a_dict = copy.deepcopy(model_a.state_dict())
	model_b_dict = copy.deepcopy(model_b.state_dict())
	for lam in tqdm(lambdas):
		naive_p = lerp(lam, model_a_dict, model_b_dict)
		model_b.load_state_dict(naive_p)
		test_loss, acc = test(model_b.cuda(), 'cuda', test_loader, True)
		test_acc_interp_naive.append(acc)
		train_loss, acc = test(model_b.cuda(), 'cuda', train_loader, True)
		train_acc_interp_naive.append(acc)

	# smart
	model_b.load_state_dict(updated_params)
	model_b.cuda()
	model_a.cuda()
	model_a_dict = copy.deepcopy(model_a.state_dict())
	model_b_dict = copy.deepcopy(model_b.state_dict())
	for lam in tqdm(lambdas):
		naive_p = lerp(lam, model_a_dict, model_b_dict)
		model_b.load_state_dict(naive_p)
		test_loss, acc = test(model_b.cuda(), 'cuda', test_loader, True)
		test_acc_interp_clever.append(acc)
		train_loss, acc = test(model_b.cuda(), 'cuda', train_loader, True)
		train_acc_interp_clever.append(acc)

	fig = plot_interp_acc(lambdas, train_acc_interp_naive, test_acc_interp_naive,
					train_acc_interp_clever, test_acc_interp_clever)
	
	fig_path = os.path.join(args.output_dir, args.tag)
	os.makedirs(fig_path, exist_ok=True)
	fig_fp = os.path.join(fig_path, f"{args.dataset}_resnet{str(args.depth)}_{str(args.width_multiplier)}_weight_matching_interp_accuracy_epoch.png")

	plt.savefig(fig_fp, dpi=300)

if __name__ == "__main__":
  main()