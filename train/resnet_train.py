import argparse
import torch
from models.resnet import ResNet
import torch.optim as optim
from utils.training import train, test
from utils.data import get_mnist, get_cifar10, get_cifar100
from utils.utils import save_model

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--seed', type=int, default=1)
	parser.add_argument('--batch_size', type=int, default=100)
	parser.add_argument('--opt', type=str, default='sgd', choices=['sgd', 'adam'])
	parser.add_argument('--dataset', type=str, default='cifar100', choices=['cifar10', 'cifar100', 'mnist'])
	parser.add_argument('--depth', type=int, default=22)
	parser.add_argument('--width-multiplier', type=int, default=2)
	parser.add_argument('--epochs', type=int, default=250)
	parser.add_argument("--lr", type=float, default=1e-3)
	parser.add_argument('--log-interval', type=int, default=10, metavar='N',help='how many batches to wait before logging training status')
	args = parser.parse_args()

	# Get data
	args = parser.parse_args()
	use_cuda = torch.cuda.is_available()

	torch.manual_seed(args.seed)

	device = torch.device("cuda" if use_cuda else "cpu")


	# load split datasets
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

	trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)
	testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=2)
	
	model = ResNet(args.depth, args.width_multiplier, 0, num_classes=n_classes).to(device)
	if args.opt == "adam":
		optimizer = optim.Adam(model.parameters(), lr=args.lr)
	else:
		optimizer = optim.SGD(model.parameters(), momentum=0.9, lr=args.lr, weight_decay=1e-4)
	scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

	best_acc = 0.0
	for epoch in range(1, args.epochs + 1):
		train_acc = train(args, model, device, trainloader, optimizer, epoch, True)
		test(model, device, testloader, True)
		scheduler.step()

		if train_acc > best_acc:
			best_acc = train_acc
			save_model(model, 
						checkpoint_name=f"{args.dataset}_{str(args.seed)}_lr_{str(args.lr)}_resnet_depth_{str(args.depth)}_{str(args.width_multiplier)}.pt",
						experiment_name=f"resnet{str(args.depth)}_{args.dataset}"
						)


if __name__ == "__main__":
  main()