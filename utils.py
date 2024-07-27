import os
import sys
import time
import pickle
import torch
import numpy as np
#from tqdm import tqdm
from sklearn.metrics import r2_score, roc_auc_score
import dataset as Dataset


def make_progress_bar(counter, iterations, suffix=''):
    bar_length = 50
    filled_up_Length = int(round(bar_length* counter / float(iterations)))
    percentage = round(100.0 * counter/float(iterations),1)
    bar = '=' * filled_up_Length + '-' * (bar_length - filled_up_Length)
    sys.stdout.write('[%s] %s%s %s\n' %(bar, percentage, '%', suffix))
    sys.stdout.flush()






class DataTools:
    
    def __init__(self) -> None:
        pass
    
    @staticmethod
    def split_x_y(df, target):
        X = df.drop(target, axis=1)
        y = df[target]
        return X.values, y.values
    
    @staticmethod
    def numpy_to_tensor(*args, device=None, dtype):
        if device is None:
            return [torch.tensor(arg, dtype=dtype) for arg in args]
        else:
            return [torch.tensor(arg, dtype=dtype).to(device) for arg in args]
    
    @staticmethod
    def train_val_test_split(df, train_size=0.7, val_size=0.15, test_size=0.15):
        assert train_size + val_size + test_size == 1, 'The sum of the sizes must be equal to 1'
        assert train_size > 0 and val_size > 0 and test_size > 0, 'The sizes must be greater than 0'
        
        split_idx = int(train_size * len(df))
        train = df.iloc[:split_idx].copy()
        split_idx += int(val_size * len(df))
        val = df.iloc[split_idx-int(val_size * len(df)):split_idx].copy()
        test = df.iloc[split_idx:].copy()
        
        return train, val, test
    

    # Function which returns the cross fold validation sets for the fi-2010 dataset
    # 0 -> 1, 2
    # 1 -> (1+2), 3
    # 2 -> (1+2+3), 4
    # ...
    @staticmethod
    def cross_fold_timeseries(df):
        # create generator
        days = df['Day'].unique().tolist()
        for i,day in enumerate(days[1:]):
            print('Day:', day)
            train_set, val_set, test_set = None, None, None
            
            if i == 0:
                # Case in which there are only two days
                test = df[df['Day'] == day]
                tmp = df[df['Day'] < day]
                idx_split = int(0.7 * len(tmp))
                train = tmp.iloc[:idx_split]
                val = tmp.iloc[idx_split:]
                
                train_set = train
                val_set = val
                test_set = test            
            else:
                # Case in which there are more than two days
                test = df[df['Day'] == day]
                tmp = df[df['Day'] < day-1]
                middle_day = df[df['Day'] == day-1]
                idx_split = int(0.7 * len(middle_day))
                train = pd.concat([tmp, middle_day.iloc[:idx_split]])
                val = middle_day.iloc[idx_split:]
                
                train_set = train
                val_set = val
                test_set = test
            
            yield train_set, val_set, test_set




class ModelTools:
    
    def __init__(self) -> None:
        pass
    
    @staticmethod
    def save_model(model, path, full=False):
        if full:
            torch.save(model, path)
        else:
            torch.save(model.state_dict(), path)

    @staticmethod
    def build_model(model_class, params):
        model = model_class(**params)
        return model

    @staticmethod
    def ht_train_wrapper(dataset_train, dataset_val, model_class, params, device, criterion):
        batch_size = params['batch_size']
        n_epochs = params['n_epochs']
        lr = params['lr']
        # Delete the keys
        del params['batch_size']
        del params['n_epochs']
        del params['lr']
        # Build the data loaders
        train_loader = Dataset.TimeSeriesLoader(
            dataset = dataset_train,
            batch_size = batch_size,
            shuffle=True
        )
        val_loader = Dataset.TimeSeriesLoader(
            dataset = dataset_val,
            batch_size = batch_size,
            shuffle=False
        )
        # Build the model
        model = ModelTools.build_model(model_class, params).to(device)
        # Build the optimizer and the loss function
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        # Train the model
        results = ModelTools.train(
            model_id = 'ht_model',
            train_loader = train_loader,
            val_loader = val_loader,
            model = model,
            optimizer = optimizer,
            criterion = criterion,
            device = device,
            n_epochs = n_epochs,
            save=False
        )
        # Average the last 5 losses on the validation set
        avg_loss = np.mean(results['val_loss'][-5:])
        
        # Return negative loss, since the bayesian optimization is a maximization problem
        return -avg_loss

    @staticmethod
    def train(model_id, model, criterion, optimizer, train_loader, val_loader, n_epochs, save:bool, device):
        training_info = {
            'train_loss': [],
            'val_loss': [],
        }
        best_val_loss = np.inf
        best_val_epoch = np.inf
        print("Start training...\n")

        for epoch in range(n_epochs):
            train_loss = []
            train_accuracy = []
            make_progress_bar(epoch, n_epochs, suffix=f' starting epoch {epoch + 1}')
            
            # train the model
            model.train()
            
            start = time.time()
            for inputs, labels in train_loader:
                # move data to GPU, if available
                inputs, labels = inputs.to(device), labels.to(device)
                # zero the parameter gradients
                optimizer.zero_grad()
                # Forward pass
                outputs = model(inputs)
                # calculate the loss
                loss = criterion(outputs, labels)
                # Backward and optimize
                loss.backward()
                optimizer.step()
                # store the performances on the train set directly during the training epochs
                train_loss.append(loss.item())
            end = time.time()
            print(f'Duration of training epoch {epoch + 1}: {end - start}, seconds')
            # Get train loss and val loss
            train_loss = np.mean(train_loss, dtype=np.float64)
            

            # evaluate the model after an epoch of training
            model.eval()
            val_loss = []
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss.append(loss.item())
            val_loss = np.mean(val_loss, dtype=np.float64)

            # save the results after each training epoch
            training_info['train_loss'].append(train_loss)
            training_info['val_loss'].append(val_loss)

            # save the model if the validation loss is the best so far
            saving_path = './saved_models/' + model_id
            if (val_loss < best_val_loss and save):
                torch.save(model, os.path.join(saving_path, 'best_model.pt'))
                best_val_loss = val_loss
                best_val_epoch = epoch
                print('model saved')

            print(f'Epoch {epoch + 1}/{n_epochs}, '
                f'Train Loss: {train_loss:.8f}, Val Loss: {val_loss: .8f}, ')
            
            # stat_file = open('./stats/' + model_id + '.txt', 'a')
            # stat_file.write(f'{epoch+1},{train_loss},{val_loss}')
            # stat_file.close()
        
        # save the final results after the training
        if save:
            torch.save({
                'epoch': n_epochs,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss,
            }, os.path.join(saving_path, 'checkpoint.pt'))

            with open(os.path.join(saving_path, 'training_process.pkl'), 'wb') as f:
                pickle.dump(training_info, f)

        return training_info
    
    def evaluate(dataloader, model, accname, loss, regression:bool, device):
        # torch.no_grad() is used only for the assessing/validation phase only. 
        # When you only perform validation, gradients are not needed for backward phase.
        true = []
        pred = []
        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                
                true.append(inputs)
                pred.append(outputs)
        
        true = np.array(true)
        pred = np.array(pred)

        return true, pred




class Metrics:
    
    def __init__(self) -> None:
        pass
    
    @staticmethod
    def sMAPE(y_true, y_pred):
        return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))
