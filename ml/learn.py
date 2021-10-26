import torch
from torch import nn
from torch.utils.data import DataLoader

from ml.DirectionFindDataset import DirectionFindDataset
from ml.DirectionFindNetwork import DirectionFindNetwork
from ml.PlusDataset import PlusDataset
from ml.PlusNetwork import PlusNetwork

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))


class Learner(object):

    def learnDirectionFindNetwork(self):
        model = DirectionFindNetwork().to(device)
        loss_fn = nn.L1Loss()
        # optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        # optimizer = torch.optim.Adamax(model.parameters(), lr=1e-3)
        optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-3)

        batch_size = 64

        training_data = DirectionFindDataset("angle-calc-training-data2.csv")
        test_data = DirectionFindDataset("angle-calc-test-data2.csv")
        train_dataloader = DataLoader(training_data, batch_size=batch_size)
        test_dataloader = DataLoader(test_data, batch_size=batch_size)

        for X, y in test_dataloader:
            print("Shape of X [N, C, H, W]: ", X.shape)
            print("Shape of y: ", y.shape, y.dtype)
            break

        epochs = 5000
        for t in range(epochs):
            print(f"Epoch {t + 1}\n-------------------------------")
            self.train(train_dataloader, model, loss_fn, optimizer)
            self.test(test_dataloader, model, loss_fn)
        print("Done!")

        torch.save(model.state_dict(), "direction_find_model.pth")
        print("Saved PyTorch Model State to direction_find_model.pth")

    def learnPlusNetwork(self):
        model = PlusNetwork().to(device)
        loss_fn = nn.L1Loss()
        # optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
        # optimizer = torch.optim.Adadelta(model.parameters(), lr=1e-3)
        optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-3)

        batch_size = 32

        training_data = PlusDataset("plus-training-data.csv")
        test_data = PlusDataset("plus-test-data.csv")
        train_dataloader = DataLoader(training_data, batch_size=batch_size)
        test_dataloader = DataLoader(test_data, batch_size=batch_size)

        for X, y in test_dataloader:
            print("Shape of X [N, C, H, W]: ", X.shape)
            print("Shape of y: ", y.shape, y.dtype)
            break

        epochs = 10000
        for t in range(epochs):
            print(f"Epoch {t + 1}\n-------------------------------")
            self.train(train_dataloader, model, loss_fn, optimizer)
            self.test(test_dataloader, model, loss_fn)
        print("Done!")

        torch.save(model.state_dict(), "direction_find_model.pth")
        print("Saved PyTorch Model State to direction_find_model.pth")

    def train(self, dataloader, model, loss_fn, optimizer):
        size = len(dataloader.dataset)
        model.train()
        for batch, (X, y) in enumerate(dataloader):
            # y.type(torch.LongTensor)
            X, y = X.to(device), y.to(device)

            # Compute prediction error
            pred = model(X)
            loss = loss_fn(pred, y)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch % 100 == 0:
                loss, current = loss.item(), batch * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    def test(self, dataloader, model, loss_fn):
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        model.eval()
        test_loss, correct = 0, 0
        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(device), y.to(device)
                pred = model(X)
                test_loss += loss_fn(pred, y).item()
                n_digits = 3
                roundedPred = torch.round(pred * 10 ** n_digits) / (10 ** n_digits)
                roundedY =  torch.round(y * 10 ** n_digits) / (10 ** n_digits)
                correct += (roundedPred.argmax(1) == roundedY).type(torch.float).sum().item()
        test_loss /= num_batches
        correct /= size
        print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


learner = Learner()
learner.learnDirectionFindNetwork()
# learner.learnPlusNetwork()
