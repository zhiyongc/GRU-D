# -*- coding: utf-8 -*-
"""
Created on Sat May 12 16:49:49 2018

@author: Zhiyong
"""

from GRUD import * 

def PrepareDataset(speed_matrix, \
                   BATCH_SIZE = 40, \
                   seq_len = 10, \
                   pred_len = 1, \
                   train_propotion = 0.7, \
                   valid_propotion = 0.2, \
                   masking = False, \
                   mask_ones_proportion = 0.8):
    """ Prepare training and testing datasets and dataloaders.
    
    Convert speed/volume/occupancy matrix to training and testing dataset. 
    The vertical axis of speed_matrix is the time axis and the horizontal axis 
    is the spatial axis.
    
    Args:
        speed_matrix: a Matrix containing spatial-temporal speed data for a network
        seq_len: length of input sequence
        pred_len: length of predicted sequence
    Returns:
        Training dataloader
        Testing dataloader
    """
    time_len = speed_matrix.shape[0]
    
    speed_matrix = speed_matrix.clip(0, 100)
    
    max_speed = speed_matrix.max().max()
    speed_matrix =  speed_matrix / max_speed
    
    speed_sequences, speed_labels = [], []
    for i in range(time_len - seq_len - pred_len):
        speed_sequences.append(speed_matrix.iloc[i:i+seq_len].values)
        speed_labels.append(speed_matrix.iloc[i+seq_len:i+seq_len+pred_len].values)
    speed_sequences, speed_labels = np.asarray(speed_sequences), np.asarray(speed_labels)
    
    # using zero-one mask to randomly set elements to zeros
    if masking:
        print('Split Speed finished. Start to generate Mask, Delta, Last_observed_X ...')
        np.random.seed(1024)
        Mask = np.random.choice([0,1], size=(speed_sequences.shape), p = [1 - mask_ones_proportion, mask_ones_proportion])
        speed_sequences = np.multiply(speed_sequences, Mask)
        
        # temporal information
        interval = 5 # 5 minutes
        S = np.zeros_like(speed_sequences) # time stamps
        for i in range(S.shape[1]):
            S[:,i,:] = interval * i

        Delta = np.zeros_like(speed_sequences) # time intervals
        for i in range(1, S.shape[1]):
            Delta[:,i,:] = S[:,i,:] - S[:,i-1,:]

        missing_index = np.where(Mask == 0)

        X_last_obsv = np.copy(speed_sequences)
        for idx in range(missing_index[0].shape[0]):
            i = missing_index[0][idx] 
            j = missing_index[1][idx]
            k = missing_index[2][idx]
            if j != 0 and j != 9:
                Delta[i,j+1,k] = Delta[i,j+1,k] + Delta[i,j,k]
            if j != 0:
                X_last_obsv[i,j,k] = X_last_obsv[i,j-1,k] # last observation
        Delta = Delta / Delta.max() # normalize
    
    # shuffle and split the dataset to training and testing datasets
    print('Generate Mask, Delta, Last_observed_X finished. Start to shuffle and split dataset ...')
    sample_size = speed_sequences.shape[0]
    index = np.arange(sample_size, dtype = int)
    np.random.seed(1024)
    np.random.shuffle(index)
    
    speed_sequences = speed_sequences[index]
    speed_labels = speed_labels[index]
    
    if masking:
        X_last_obsv = X_last_obsv[index]
        Mask = Mask[index]
        Delta = Delta[index]
        speed_sequences = np.expand_dims(speed_sequences, axis=1)
        X_last_obsv = np.expand_dims(X_last_obsv, axis=1)
        Mask = np.expand_dims(Mask, axis=1)
        Delta = np.expand_dims(Delta, axis=1)
        dataset_agger = np.concatenate((speed_sequences, X_last_obsv, Mask, Delta), axis = 1)
        
    train_index = int(np.floor(sample_size * train_propotion))
    valid_index = int(np.floor(sample_size * ( train_propotion + valid_propotion)))
    
    if masking:
        train_data, train_label = dataset_agger[:train_index], speed_labels[:train_index]
        valid_data, valid_label = dataset_agger[train_index:valid_index], speed_labels[train_index:valid_index]
        test_data, test_label = dataset_agger[valid_index:], speed_labels[valid_index:]
    else:
        train_data, train_label = speed_sequences[:train_index], speed_labels[:train_index]
        valid_data, valid_label = speed_sequences[train_index:valid_index], speed_labels[train_index:valid_index]
        test_data, test_label = speed_sequences[valid_index:], speed_labels[valid_index:]
    
    train_data, train_label = torch.Tensor(train_data), torch.Tensor(train_label)
    valid_data, valid_label = torch.Tensor(valid_data), torch.Tensor(valid_label)
    test_data, test_label = torch.Tensor(test_data), torch.Tensor(test_label)
    
    train_dataset = utils.TensorDataset(train_data, train_label)
    valid_dataset = utils.TensorDataset(valid_data, valid_label)
    test_dataset = utils.TensorDataset(test_data, test_label)
    
    train_dataloader = utils.DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle=True, drop_last = True)
    valid_dataloader = utils.DataLoader(valid_dataset, batch_size = BATCH_SIZE, shuffle=True, drop_last = True)
    test_dataloader = utils.DataLoader(test_dataset, batch_size = BATCH_SIZE, shuffle=True, drop_last = True)
    
    X_mean = np.mean(speed_sequences, axis = 0)
    
    print('Finished')
    
    return train_dataloader, valid_dataloader, test_dataloader, max_speed, X_mean



def Train_Model(model, train_dataloader, valid_dataloader, num_epochs = 300, patience = 10, min_delta = 0.00001):
    
    print('Model Structure: ', model)
    print('Start Training ... ')
    
    model.cuda()
    
    if (type(model) == nn.modules.container.Sequential):
        output_last = model[-1].output_last
        print('Output type dermined by the last layer')
    else:
        output_last = model.output_last
        print('Output type dermined by the model')
        
    loss_MSE = torch.nn.MSELoss()
    loss_L1 = torch.nn.L1Loss()
    
    learning_rate = 0.0001
    optimizer = torch.optim.RMSprop(model.parameters(), lr = learning_rate, alpha=0.99)
    use_gpu = torch.cuda.is_available()
    
    interval = 100
    losses_train = []
    losses_valid = []
    losses_epochs_train = []
    losses_epochs_valid = []
    
    cur_time = time.time()
    pre_time = time.time()
    
    # Variables for Early Stopping
    is_best_model = 0
    patient_epoch = 0
    for epoch in range(num_epochs):
        
        trained_number = 0
        
        valid_dataloader_iter = iter(valid_dataloader)
        
        losses_epoch_train = []
        losses_epoch_valid = []
        
        for data in train_dataloader:
            inputs, labels = data

            if inputs.shape[0] != batch_size:
                continue

            if use_gpu:
                inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            else: 
                inputs, labels = Variable(inputs), Variable(labels)
            
            model.zero_grad()

            outputs = model(inputs)
            
            if output_last:
                loss_train = loss_MSE(torch.squeeze(outputs), torch.squeeze(labels))
            else:
                full_labels = torch.cat((inputs[:,1:,:], labels), dim = 1)
                loss_train = loss_MSE(outputs, full_labels)
        
            losses_train.append(loss_train.data)
            losses_epoch_train.append(loss_train.data)
            
            optimizer.zero_grad()
            
            loss_train.backward()
            
            optimizer.step()
            
             # validation 
            try: 
                inputs_val, labels_val = next(valid_dataloader_iter)
            except StopIteration:
                valid_dataloader_iter = iter(valid_dataloader)
                inputs_val, labels_val = next(valid_dataloader_iter)
            
            if use_gpu:
                inputs_val, labels_val = Variable(inputs_val.cuda()), Variable(labels_val.cuda())
            else: 
                inputs_val, labels_val = Variable(inputs_val), Variable(labels_val)
                
            model.zero_grad()
            
            outputs_val = model(inputs_val)
            
            if output_last:
                loss_valid = loss_MSE(torch.squeeze(outputs_val), torch.squeeze(labels_val))
            else:
                full_labels_val = torch.cat((inputs_val[:,1:,:], labels_val), dim = 1)
                loss_valid = loss_MSE(outputs_val, full_labels_val)

            losses_valid.append(loss_valid.data)
            losses_epoch_valid.append(loss_valid.data)
            
            # output
            trained_number += 1
            
        avg_losses_epoch_train = sum(losses_epoch_train).cpu().numpy() / float(len(losses_epoch_train))
        avg_losses_epoch_valid = sum(losses_epoch_valid).cpu().numpy() / float(len(losses_epoch_valid))
        losses_epochs_train.append(avg_losses_epoch_train)
        losses_epochs_valid.append(avg_losses_epoch_valid)
        
        # Early Stopping
        if epoch == 0:
            is_best_model = 1
            best_model = model
            min_loss_epoch_valid = 10000.0
            if avg_losses_epoch_valid < min_loss_epoch_valid:
                min_loss_epoch_valid = avg_losses_epoch_valid
        else:
            if min_loss_epoch_valid - avg_losses_epoch_valid > min_delta:
                is_best_model = 1
                best_model = model
                min_loss_epoch_valid = avg_losses_epoch_valid 
                patient_epoch = 0
            else:
                is_best_model = 0
                patient_epoch += 1
                if patient_epoch >= patience:
                    print('Early Stopped at Epoch:', epoch)
                    break
        
        # Print training parameters
        cur_time = time.time()
        print('Epoch: {}, train_loss: {}, valid_loss: {}, time: {}, best model: {}'.format( \
                    epoch, \
                    np.around(avg_losses_epoch_train, decimals=8),\
                    np.around(avg_losses_epoch_valid, decimals=8),\
                    np.around([cur_time - pre_time] , decimals=2),\
                    is_best_model) )
        pre_time = cur_time
                
    return best_model, [losses_train, losses_valid, losses_epochs_train, losses_epochs_valid]


def Test_Model(model, test_dataloader, max_speed):
    
    if (type(model) == nn.modules.container.Sequential):
        output_last = model[-1].output_last
    else:
        output_last = model.output_last
    
    inputs, labels = next(iter(test_dataloader))
    [batch_size, type_size, step_size, fea_size] = inputs.size()

    cur_time = time.time()
    pre_time = time.time()
    
    use_gpu = torch.cuda.is_available()
    
    loss_MSE = torch.nn.MSELoss()
    loss_L1 = torch.nn.MSELoss()
    
    tested_batch = 0
    
    losses_mse = []
    losses_l1 = [] 
    MAEs = []
    MAPEs = []

    
    for data in test_dataloader:
        inputs, labels = data
        
        if inputs.shape[0] != batch_size:
            continue
    
        if use_gpu:
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        else: 
            inputs, labels = Variable(inputs), Variable(labels)
        
        outputs = model(inputs)
    
        loss_MSE = torch.nn.MSELoss()
        loss_L1 = torch.nn.L1Loss()
        
        if output_last:
            loss_mse = loss_MSE(torch.squeeze(outputs), torch.squeeze(labels))
            loss_l1 = loss_L1(torch.squeeze(outputs), torch.squeeze(labels))
            MAE = torch.mean(torch.abs(torch.squeeze(outputs) - torch.squeeze(labels)))
            MAPE = torch.mean(torch.abs(torch.squeeze(outputs) - torch.squeeze(labels)) / torch.squeeze(labels))
        else:
            loss_mse = loss_MSE(outputs[:,-1,:], labels)
            loss_l1 = loss_L1(outputs[:,-1,:], labels)
            MAE = torch.mean(torch.abs(outputs[:,-1,:] - torch.squeeze(labels)))
            MAPE = torch.mean(torch.abs(outputs[:,-1,:] - torch.squeeze(labels)) / torch.squeeze(labels))
            
        losses_mse.append(loss_mse.data)
        losses_l1.append(loss_l1.data)
        MAEs.append(MAE.data)
        MAPEs.append(MAPE.data)
        
        tested_batch += 1
    
        if tested_batch % 1000 == 0:
            cur_time = time.time()
            print('Tested #: {}, loss_l1: {}, loss_mse: {}, time: {}'.format( \
                  tested_batch * batch_size, \
                  np.around([loss_l1.data[0]], decimals=8), \
                  np.around([loss_mse.data[0]], decimals=8), \
                  np.around([cur_time - pre_time], decimals=8) ) )
            pre_time = cur_time
    losses_l1 = np.array(losses_l1)
    losses_mse = np.array(losses_mse)
    MAEs = np.array(MAEs)
    MAPEs = np.array(MAPEs)
    
    mean_l1 = np.mean(losses_l1) * max_speed
    std_l1 = np.std(losses_l1) * max_speed
    MAE_ = np.mean(MAEs) * max_speed
    MAPE_ = np.mean(MAPEs) * 100
    
    print('Tested: L1_mean: {}, L1_std: {}, MAE: {} MAPE: {}'.format(mean_l1, std_l1, MAE_, MAPE_))
    return [losses_l1, losses_mse, mean_l1, std_l1]


if __name__ == "__main__":
    
    data = 'loop'
    if data == 'inrix':
        speed_matrix =  pd.read_pickle('../Data_Warehouse/Data_network_traffic/inrix_seattle_speed_matrix_2012')
    elif data == 'loop':
        speed_matrix =  pd.read_pickle('../Data_Warehouse/Data_network_traffic//speed_matrix_2015')
        
    train_dataloader, valid_dataloader, test_dataloader, max_speed, X_mean = PrepareDataset(speed_matrix, BATCH_SIZE = 64, masking = True)
    
    inputs, labels = next(iter(train_dataloader))
    [batch_size, type_size, step_size, fea_size] = inputs.size()
    input_dim = fea_size
    hidden_dim = fea_size
    output_dim = fea_size
    
    grud = GRUD(input_dim, hidden_dim, output_dim, X_mean, output_last = True)
    best_grud, losses_grud = Train_Model(grud, train_dataloader, valid_dataloader)
    [losses_l1, losses_mse, mean_l1, std_l1] = Test_Model(best_grud, test_dataloader, max_speed)