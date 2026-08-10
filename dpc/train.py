import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import pandas as pd
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from scipy.stats import spearmanr
from scipy.stats import pearsonr
from data_process.encode import base_one_hot_encoding_8v_dif, genotype_to_dataframe, MICSelector
from model.model_1v_depth import PhenotypePredictor_1v
from model.early_stop import EarlyStopping

def initialize_model(num_chromosomes, new_snp_counts):
    return PhenotypePredictor_1v(num_chromosomes, new_snp_counts, input_dim=8).to(device)

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(f'output'):
        os.makedirs(f'output')

    np.random.seed(42)
    dim = 8

    output_file_path = '/public/home/knliu25/lkn/dpc_model_test/chr12345678910_hybrid_PH_BJ_extracted.csv'

    merged_df = pd.read_csv(output_file_path, sep='\t', chunksize=100)

    one_hot_genotype_list = []
    phenotypes_list = []
    i = 0
    for chunk in merged_df:
        print(f"chunk {i} has processed!")
        genotype_data = chunk['genotype']
        phenotype_data = chunk['phenotype']
        genotype_df = genotype_to_dataframe(genotype_data)
        one_hot_genotype = base_one_hot_encoding_8v_dif(genotype_df).values
        one_hot_genotype_list.append(one_hot_genotype)
        phenotypes_list.append(phenotype_data.values)
        i += 1

    one_hot_genotype_array = np.concatenate(one_hot_genotype_list, axis=0)
    phenotypes_array = np.concatenate(phenotypes_list, axis=0)

    scaler = StandardScaler()
    phenotypes_array = scaler.fit_transform(phenotypes_array.reshape(-1, 1)).flatten()

    map_file_path = '/public/home/knliu25/lkn/dpc_model_test/chr12345678910_merged_data_ped.map'
    map_df = pd.read_csv(map_file_path, sep='\t', header=None, names=['chromosome', 'snp_id', 'map', 'position'])

    snp_count_per_chromosome = map_df.groupby('chromosome').size().to_dict()
    chromosomes = list(snp_count_per_chromosome.keys())
    num_chromosomes = len(chromosomes)
    snp_counts = list(snp_count_per_chromosome.values())

    print(f"Number of chromosomes: {num_chromosomes}")
    print(f"SNP counts per chromosome: {snp_count_per_chromosome}")
    print(f"Starting get split")

    selected_snp_count = 1000
    chromosome_data_list = []
    start_idx = 0
    i = 0

    snp_encoding_map = {
        "AA": 0, "AT": 1, "TA": 1, "AC": 2, "CA": 2,
        "AG": 3, "GA": 3, "TT": 4, "TC": 5, "CT": 5,
        "TG": 6, "GT": 6, "CC": 7, "CG": 8,"GC": 8, "GG": 9,
        "00": -1, "A0": -1, "0A": -1, "T0": -1, "0T": -1,
        "C0": -1, "0C": -1, "G0": -1, "0G": -1
    }

    base_mapping = {0: 'A', 1: 'T', 2: 'C', 3: 'G', 4: '0', 5: '0', 6: '0', 7: '0'}

    for count in snp_counts:
        i += 1
        end_idx = start_idx + count
        data_3d = one_hot_genotype_array[:, start_idx * 8:end_idx * 8].reshape(-1, count, 8)

        snp_index_features = np.zeros((data_3d.shape[0], data_3d.shape[1]))
        for j in range(data_3d.shape[0]):
            for k in range(data_3d.shape[1]):
                snp_encoding = data_3d[j, k, :]
                indices = np.where(snp_encoding == 1)[0]

                if len(indices) == 2:
                    base1 = base_mapping.get(indices[0], '0')
                    base2 = base_mapping.get(indices[1], '0')
                    base_pair = ''.join(sorted([base1, base2]))
                    if base_pair == 'AT':
                        base_pair = 'AT' if indices[0] < indices[1] else 'TA'
                    elif base_pair == 'AC':
                        base_pair = 'AC' if indices[0] < indices[1] else 'CA'
                    elif base_pair == 'AG':
                        base_pair = 'AG' if indices[0] < indices[1] else 'GA'
                    elif base_pair == 'TC':
                        base_pair = 'TC' if indices[0] < indices[1] else 'CT'
                    elif base_pair == 'TG':
                        base_pair = 'TG' if indices[0] < indices[1] else 'GT'
                    elif base_pair == 'CG':
                        base_pair = 'CG' if indices[0] < indices[1] else 'GC'

                    snp_index_features[j, k] = snp_encoding_map.get(base_pair, 16)

                elif len(indices) == 1:
                    base = base_mapping.get(indices[0], '0')
                    if indices[0] < 4:
                        if np.argmax(snp_encoding[:4]) == indices[0]:
                            base_pair = base + '0'
                        else:
                            base_pair = '0' + base
                    else:
                        base_pair = '00'
                    snp_index_features[j, k] = snp_encoding_map.get(base_pair, 16)
                else:
                    snp_index_features[j, k] = 16

        mic_selector = MICSelector(k=selected_snp_count)
        mic_selector.fit(snp_index_features, phenotypes_array)
        selected_snp_indices = mic_selector.top_k_indices_

        selected_data = data_3d[:, selected_snp_indices, :]

        print(f"selected_data shape before padding: {selected_data.shape}")
        max_snps = selected_snp_count
        current_snps = selected_data.shape[1]
        padding_size = max_snps - current_snps
        if padding_size > 0:
            padding = np.zeros((selected_data.shape[0], padding_size, selected_data.shape[2]))
            selected_data = np.concatenate((selected_data, padding), axis=1)

        chromosome_data_list.append(selected_data)
        start_idx = end_idx
        print(f"Chromosome {i} has processed!")

    chromosome_tensors = [torch.tensor(data, dtype=torch.float32) for data in chromosome_data_list]
    chromosome_tensors = torch.stack(chromosome_tensors, dim=1)
    print(f"chromosome_tensors shape: {chromosome_tensors.shape}")

    phenotypes_tensor = torch.tensor(phenotypes_array, dtype=torch.float32).reshape(-1, 1)
    print(f"phenotypes_tensor: {phenotypes_tensor.shape}")

    print(f"Starting get DataLoader")
    dataset = TensorDataset(chromosome_tensors, phenotypes_tensor)
    test_ratio = 0.1
    test_size = int(test_ratio * len(dataset))
    train_val_size = len(dataset) - test_size

    indices = torch.randperm(len(dataset))
    test_indices = indices[:test_size]
    print(f"len(test_indices):{len(test_indices)}")
    train_val_indices = indices[test_size:]
    print(f"len(train_val_indices):{len(train_val_indices)}")

    test_subdataset = torch.utils.data.Subset(dataset, test_indices)
    k = 10
    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    cv_val_losses = []
    cv_val_spearman = []
    cv_val_pearsonr = []
    cv_test_losses = []
    cv_test_spearson = []
    cv_test_pearsonr = []
    num_samples = len(dataset)
    num_features = chromosome_tensors.shape[1]

    if not os.path.exists(f'output/saved_models'):
        os.makedirs(f'output/saved_models')

    best_test_pearson = -1.0
    best_test_fold = -1
    best_test_spearman = -1.0
    best_test_loss = float('inf')

    for fold, (train_indices, val_indices) in enumerate(kf.split(train_val_indices)):
        print(f"Fold {fold + 1}/{k}")
        current_train_indices = train_val_indices[train_indices]
        current_val_indices = train_val_indices[val_indices]

        train_subdataset = torch.utils.data.Subset(dataset, current_train_indices)
        print(f"len(train_subdataset):{len(train_subdataset)}")
        val_subdataset = torch.utils.data.Subset(dataset, current_val_indices)
        print(f"len(val_subdataset):{len(val_subdataset)}")

        train_dataloader = DataLoader(train_subdataset, batch_size=64, shuffle=True)
        val_dataloader = DataLoader(val_subdataset, batch_size=64, shuffle=False)
        test_dataloader = DataLoader(test_subdataset, batch_size=64, shuffle=False)

        model = initialize_model(num_features, [1000] * num_features)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
        early_stopping = EarlyStopping(20)
        log_file = f"output/training_log_fold{fold + 1}.txt"

        with open(log_file, "w") as f:
            f.write("Epoch,Train Loss,Val Loss,Val Spearman Corr\n")

        best_val_loss = float('inf')
        best_spearman = 0.0
        best_pearsonr = 0.0
        best_model_state = None

        train_losses = []
        val_losses = []

        num_epochs = 300
        for epoch in range(num_epochs):
            model.train()
            train_loss = 0.0
            for batch in train_dataloader:
                inputs, targets = batch
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * inputs.size(0)
            train_loss = train_loss / len(train_subdataset)
            train_losses.append(train_loss)

            model.eval()
            val_loss = 0.0
            val_preds = []
            val_targets = []
            with torch.no_grad():
                for batch in val_dataloader:
                    inputs, targets = batch
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    outputs = outputs.squeeze(1)
                    targets = targets.squeeze(1)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item() * inputs.size(0)
                    val_preds.extend(outputs.cpu().numpy())
                    val_targets.extend(targets.cpu().numpy())

            val_loss = val_loss / len(val_subdataset)
            val_pearsonr, p_value = pearsonr(val_preds, val_targets)
            combined_matrix = np.column_stack((val_preds, val_targets))
            val_spearman, _ = spearmanr(combined_matrix)
            val_losses.append(val_loss)

            early_stopping(val_loss)
            if val_loss < best_val_loss:
                best_val_loss, best_spearman, best_pearsonr, best_model_state = val_loss, val_spearman, val_pearsonr, model.state_dict()
            if early_stopping.early_stop:
                print("Early stopping")
                break
            scheduler.step(val_loss)

            print(
                f'Fold {fold + 1}, Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, '
                f'Val Pearsonr: {val_pearsonr:.4f}, Val Spearman Corr: {val_spearman:.4f}')

            with open(log_file, "a") as f:
                f.write(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, '
                        f'Val Pearsonr: {val_pearsonr:.4f}, Val Spearman Corr: {val_spearman:.4f}\n')

        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Training and Validation Loss - Fold {fold + 1}')
        plt.legend()
        save_path = f'output/loss_png/fold{fold + 1}'
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(os.path.join(save_path, f'{model.__class__.__name__}_loss_plot.png'))
        plt.close()

        if best_model_state:
            model_save_name = f'output/saved_models/{model.__class__.__name__}_fold{fold + 1}_model.pth'
            torch.save(best_model_state, model_save_name)
            print(f'Fold {fold + 1} best model saved with Val Loss: {best_val_loss:.4f}, '
                  f'Best Pearsonr: {best_pearsonr:.4f}, Best Spearman: {best_spearman:.4f}')

        model.load_state_dict(best_model_state)
        model.eval()
        test_loss = 0.0
        test_preds = []
        test_targets = []
        with torch.no_grad():
            for batch in test_dataloader:
                inputs, targets = batch
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                outputs = outputs.squeeze(1)
                targets = targets.squeeze(1)
                loss = criterion(outputs, targets)
                test_loss += loss.item() * inputs.size(0)
                test_preds.extend(outputs.cpu().numpy())
                test_targets.extend(targets.cpu().numpy())

        test_loss = test_loss / len(test_subdataset)
        test_pearsonr, p_value = pearsonr(test_preds, test_targets)
        combined_matrix = np.column_stack((test_preds, test_targets))
        test_spearman, _ = spearmanr(combined_matrix)

        print(f'Fold {fold + 1} Test Loss: {test_loss:.4f}, Test Pearsonr: {test_pearsonr:.4f}, '
              f'Test Spearman Corr: {test_spearman:.4f}')

        cv_val_losses.append(best_val_loss)
        cv_val_spearman.append(best_spearman)
        cv_val_pearsonr.append(best_pearsonr)
        cv_test_losses.append(test_loss)
        cv_test_pearsonr.append(test_pearsonr)
        cv_test_spearson.append(test_spearman)

        if test_pearsonr > best_test_pearson:
            best_test_pearson = test_pearsonr
            best_test_spearman = test_spearman
            best_test_loss = test_loss
            best_test_fold = fold + 1
        print("ln")
        print("\nCross-Validation Results:")
        print(f"Average Best Val Loss: {np.mean(cv_val_losses):.4f} 卤 {np.std(cv_val_losses):.4f}")
        print(f"Average Best Val Pearsonr: {np.mean(cv_val_pearsonr):.4f} 卤 {np.std(cv_val_pearsonr):.4f}")
        print(f"Average Best Val Spearman Corr: {np.mean(cv_val_spearman):.4f} 卤 {np.std(cv_val_spearman):.4f}")
        print(f"Average Test Loss: {np.mean(cv_test_losses):.4f} 卤 {np.std(cv_test_losses):.4f}")
        print(f"Average Test Pearsonr: {np.mean(cv_test_pearsonr):.4f} 卤 {np.std(cv_test_pearsonr):.4f}")
        print(f"Average Test Spearman Corr: {np.mean(cv_test_spearson):.4f} 卤 {np.std(cv_test_spearson):.4f}")

        print("\nBest Test Performance Across All Folds:")
        print(f"Best Test Pearsonr: {best_test_pearson:.4f} (Achieved in Fold {best_test_fold})")
        print(f"Corresponding Test Spearman Corr: {best_test_spearman:.4f}")
        print(f"Corresponding Test Loss: {best_test_loss:.4f}")

        with open("output/cross_validation_results.txt", "w") as f:
            f.write("Cross-Validation Results:\n")
            f.write(f"Average Best Val Loss: {np.mean(cv_val_losses):.4f} 卤 {np.std(cv_val_losses):.4f}\n")
            f.write(f"Average Best Val Pearsonr: {np.mean(cv_val_pearsonr):.4f} 卤 {np.std(cv_val_pearsonr):.4f}\n")
            f.write(f"Average Best Val Spearman Corr: {np.mean(cv_val_spearman):.4f} 卤 {np.std(cv_val_spearman):.4f}\n")
            f.write(f"Average Test Loss: {np.mean(cv_test_losses):.4f} 卤 {np.std(cv_test_losses):.4f}\n")
            f.write(f"Average Test Pearsonr: {np.mean(cv_test_pearsonr):.4f} 卤 {np.std(cv_test_pearsonr):.4f}\n")
            f.write(f"Average Test Spearman Corr: {np.mean(cv_test_spearson):.4f} 卤 {np.std(cv_test_spearson):.4f}\n")

            f.write("\nBest Test Performance Across All Folds:\n")
            f.write(f"Best Test Pearsonr: {best_test_pearson:.4f} (Achieved in Fold {best_test_fold})\n")
            f.write(f"Corresponding Test Spearman Corr: {best_test_spearman:.4f}\n")
            f.write(f"Corresponding Test Loss: {best_test_loss:.4f}\n")
