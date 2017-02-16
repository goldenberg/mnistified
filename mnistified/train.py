import argparse
from mnistified.model import CNNModel

def main():
    parser = argparse.ArgumentParser(description='Train an MNIST model and serialize the model weights.')
    parser.add_argument(
        '--weights',
        help='HDF5 output file for the weights'
    )
    parser.add_argument(
        '--num-epochs',
        default=12,
        type=int,
        help='Number of epochs to train for.',
    )
    parser.add_argument(
        '--batch-size',
        default=128,
        type=int,
        help='Mini batch size for training.',
    )
    args = parser.parse_args()


    model = CNNModel()
    model.train(num_epochs=args.num_epochs, batch_size=args.batch_size)
    model.serialize(args.weights)


if __name__ == '__main__':
    main()