import torch
from utils.utils import get_loss_function, get_optimizer, get_scheduler
from .TextLoader import TextLoader
import torch.nn.functional as F

class RNNTrainer:
    def __init__(
            self, 
            model,
            data: TextLoader,
            loss_name, 
            optimizer_name, 
            device, 
            lr,
            optimizer_params,
            scheduler_name=None):
        self.model = model.to(device)
        self.loss = get_loss_function(loss_name)
        self.optimizer = get_optimizer(optimizer_name, model.parameters(), lr, **optimizer_params)
        self.data = data

        self.device = device

        self.scheduler = get_scheduler(self.optimizer, scheduler_name)

        self.history = {
            "train_loss": [],
            "val_loss": [], 
            "val_err": []
        }

    def train_epoch(self):
        self.model.train() # switch to train mode

        running_loss = 0.0
        num_batches = 0

        for idx, (X_train, y_train) in enumerate(self.data.get_batches()):

            X_train, y_train = torch.tensor(X_train, device=self.device), \
                               torch.tensor(y_train, device=self.device)
            
            num_batches += 1

            logits, _ = self.model(X_train) # batch_size x num_steps x vocab_size

            y_hat = logits.reshape(-1, logits.shape[-1]) # (batch_size * num_steps) x vocab_size
            y_train = y_train.reshape(-1) # batch_size * num_steps

            loss = self.loss(y_hat, y_train)

            self.optimizer.zero_grad()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            running_loss += loss.item()
        
        return running_loss / num_batches
    
    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        running_loss = 0.0
        total_samples = 0
        correct = 0
        curr_state = None
        num_batches = 0

        for X_test, y_test in self.data.get_batches(train=False):


            X_test, y_test = torch.tensor(X_test, device=self.device), \
                               torch.tensor(y_test, device=self.device)
            
            num_batches += 1

            logits, state = self.model(X_test)

            y_hat = logits.reshape(-1, logits.shape[-1]) # (batch_size * num_steps) x vocab_size
            y_test = y_test.reshape(-1) # batch_size * num_steps

            loss = self.loss(y_hat, y_test)

            running_loss += loss.item()
            y_hat = torch.argmax(y_hat, dim=1)

            correct += (y_hat == y_test).sum().item()
            total_samples += y_hat.size(0)

        loss = running_loss / num_batches
        val_err = 1 - (correct / total_samples) 

        return loss, val_err
    
    def train(self, max_epochs=10):
            for epoch in range(1, max_epochs + 1):
                train_loss = self.train_epoch()
                test_loss, val_err = self.evaluate()

                if self.scheduler:
                    if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(test_loss)
                    else:
                        self.scheduler.step()
                
                current_lr = self.optimizer.param_groups[0]['lr']

                self.history["train_loss"].append(train_loss)
                self.history["val_loss"].append(test_loss)
                self.history["val_err"].append(val_err)

                print(f"Epoch [{epoch}/{max_epochs}] | LR: {current_lr:.6f} | "
                    f"Train Loss: {train_loss:.4f} | Val Loss: {test_loss:.4f}")
    
    @torch.no_grad()
    def make_predictions(self):
        self.model.eval()
        total_samples = 0
        correct = 0
        total_y_hat = []
        total_y_true = []
        total_probs = []

        for X_test, y_test in self.data.get_batches(train=False):
            X_test, y_test = torch.tensor(X_test, device=self.device), \
                               torch.tensor(y_test, device=self.device)
            
            logits, state = self.model(X_test)

            logits = logits.reshape(-1, logits.shape[-1]) # (batch_size * num_steps) x vocab_size
            y_test = y_test.reshape(-1) # batch_size * num_steps

            softmax_probs = F.softmax(logits, dim=1)

            probs, y_hat = torch.max(softmax_probs, dim=1)

            total_y_hat.append(y_hat)
            total_y_true.append(y_test)
            total_probs.append(probs)

        total_y_hat = torch.cat(total_y_hat)
        total_y_true = torch.cat(total_y_true)
        total_probs = torch.cat(total_probs)

        correct += (total_y_hat == total_y_true).sum().item()
        total_samples = total_y_hat.shape[0]
        val_err = 1 - (correct / total_samples) 

        return total_y_hat, total_y_true, val_err, total_probs