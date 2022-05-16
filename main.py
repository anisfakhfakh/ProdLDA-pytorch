from argparse import ArgumentParser

from utils import *
import pickle
from torch.optim import Adam

from models import ProdLDA

parser = ArgumentParser(description='Paper Implementation')
parser.add_argument('--vocab_size', type=int, default=2000, help="Vocabulary size")
parser.add_argument('--hidden_size', type=int, default=100, help="Number of hidden units")
parser.add_argument('--num_topics', type=int, default=50, help="Number of topics")
parser.add_argument('--num_epochs', type=int, default=500, help="Number of epochs")
parser.add_argument('--batch_size', type=int, default=32, help="Batch size")
parser.add_argument('--dropout', type=float, default=0.2, help="Dropout rate")
parser.add_argument('--learning_rate', type=float, default=1e-3, help="Learning rate")
parser.add_argument('--dir', type=str, default='20newsdata/', help="Data directory")
parser.add_argument('--device', type=str, default='cpu', help="cuda to use GPU")

args = parser.parse_args()


def main(args):
    train_data, vocab_size = data_transform(args.dir, args.vocab_size, "train")
    val_data, _ = data_transform(args.dir, args.vocab_size, "test")
    with open(args.dir+"vocab.pkl", 'rb') as f:
        vocab = pickle.load(f)
    train_data.to(args.device)
    val_data.to(args.device)
    model = ProdLDA(vocab_size, args.hidden_size, args.num_topics, args.dropout).to(args.device)
    optimizer = Adam(model.parameters(), lr=args.learning_rate)
    model = train(model, optimizer, train_data, val_data, args.num_epochs, args.batch_size)
    test(model, val_data)

if __name__ == '__main__':
    main(args)