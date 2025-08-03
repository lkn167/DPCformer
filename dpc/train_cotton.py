import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from torch.utils.data import DataLoader, TensorDataset, Subset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

from data_process.encode import base_one_hot_encoding_8v_dif, genotype_to_dataframe, MICSelector
from model.model_1v_depth import PhenotypePredictor_1v
from model.early_stop import EarlyStopping


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def preprocess_data(output_file_path, map_file_path):

    merged_df_chunks = pd.read_csv(output_file_path, sep='\t', chunksize=100)

    one_hot_genotype_list = []
    phenotypes_list = []
    print("Starting data preprocessing...")
    for i, chunk in enumerate(merged_df_chunks):
        print(f"Processing chunk {i + 1}...")
        genotype_data = chunk['genotype']
        phenotype_data = chunk['phenotype']

        genotype_df = genotype_to_dataframe(genotype_data)
        one_hot_genotype = base_one_hot_encoding_8v_dif(genotype_df).values

        one_hot_genotype_list.append(one_hot_genotype)
        phenotypes_list.append(phenotype_data.values)

    if not one_hot_genotype_list or not phenotypes_list:
        raise ValueError("Data lists are empty! Check data preprocessing steps and file paths.")

    one_hot_genotype_array = np.concatenate(one_hot_genotype_list, axis=0)
    phenotypes_array = np.concatenate(phenotypes_list, axis=0)

    scaler = StandardScaler()
    phenotypes_array = scaler.fit_transform(phenotypes_array.reshape(-1, 1)).flatten()

    map_df = pd.read_csv(map_file_path, sep='\t', header=None, names=['chromosome', 'snp_id', 'map', 'position'])
    snp_count_per_chromosome = map_df.groupby('chromosome').size().to_dict()

    print("Data preprocessing finished.")
    return one_hot_genotype_array, phenotypes_array, snp_count_per_chromosome, scaler


def create_chromosome_groups(snp_count_per_chromosome):

    chromosome_groups = []
    sorted_chromosomes = sorted(
        [ch for ch in snp_count_per_chromosome.keys() if isinstance(ch, int) or str(ch).isdigit()])
    max_chr = int(sorted_chromosomes[-1]) if sorted_chromosomes else 0

    for i in range(1, (max_chr // 2) + 2):
        first_chr = i
        second_chr = i + 13
        group = []
        if first_chr in snp_count_per_chromosome: group.append(first_chr)
        if second_chr in snp_count_per_chromosome: group.append(second_chr)
        if group: chromosome_groups.append(tuple(group))

    return chromosome_groups


def prepare_grouped_data(one_hot_genotype_array, phenotypes_array, chromosome_groups, snp_count_per_chromosome,
                         selected_snp_count):

    print("Preparing grouped data with MIC selection...")
    grouped_chromosome_data_list = []

    base_map = {0: 'A', 1: 'T', 2: 'C', 3: 'G', 4: '0', 5: '0', 6: '0', 7: '0'}
    genotype_to_int_map = {
        ''.join(sorted(g)): i for i, g in enumerate([
            "AA", "AT", "AC", "AG", "TT", "TC", "TG", "CC", "CG", "GG", "A0", "T0", "C0", "G0", "00"
        ])
    }

    sorted_keys = sorted(snp_count_per_chromosome.keys())
    snp_indices = np.cumsum([0] + [snp_count_per_chromosome[c] for c in sorted_keys])
    snp_positions = {c: (snp_indices[i], snp_indices[i + 1]) for i, c in enumerate(sorted_keys)}

    for group in chromosome_groups:
        group_data_list_8d = []
        for chr_num in group:
            if chr_num not in snp_count_per_chromosome: continue
            count = snp_count_per_chromosome[chr_num]
            start_idx, end_idx = snp_positions[chr_num]
            data_8d = one_hot_genotype_array[:, start_idx * 8: end_idx * 8].reshape(-1, count, 8)
            group_data_list_8d.append(data_8d)

        if not group_data_list_8d: continue

        concatenated_group_data_8d = np.concatenate(group_data_list_8d, axis=1)
        print(f"Group {group}: combined shape {concatenated_group_data_8d.shape}")

        n_samples, n_snps, _ = concatenated_group_data_8d.shape
        snp_1d_features = np.zeros((n_samples, n_snps), dtype=int)

        for i in range(n_samples):
            for j in range(n_snps):
                indices = np.where(concatenated_group_data_8d[i, j, :] == 1)[0]
                if len(indices) == 2:
                    genotype = ''.join(sorted((base_map[indices[0]], base_map[indices[1]])))
                elif len(indices) == 1:
                    genotype = ''.join(sorted((base_map[indices[0]], '0')))
                else:
                    genotype = "00"
                snp_1d_features[i, j] = genotype_to_int_map.get(genotype, -1)

        mic_selector = MICSelector(k=selected_snp_count)
        mic_selector.fit(snp_1d_features, phenotypes_array)
        selected_snp_indices = mic_selector.top_k_indices_

        selected_data_8d = concatenated_group_data_8d[:, selected_snp_indices, :]
        print(f"  - Selected {selected_data_8d.shape[1]} SNPs via MIC.")

        padding_size = selected_snp_count - selected_data_8d.shape[1]
        if padding_size > 0:
            padding_array = np.zeros((n_samples, padding_size, 8))
            selected_data_8d = np.concatenate((selected_data_8d, padding_array), axis=1)

        grouped_chromosome_data_list.append(selected_data_8d)

    if not grouped_chromosome_data_list: return None

    grouped_chromosome_tensors = np.stack(grouped_chromosome_data_list, axis=1)
    print(f"Final stacked tensor shape: {grouped_chromosome_tensors.shape}")

    return torch.tensor(grouped_chromosome_tensors, dtype=torch.float32)


def initialize_model_and_optimizer(num_groups, group_snps_list, input_dim, lr):

    model = PhenotypePredictor_1v(num_groups, group_snps_list, input_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    return model, optimizer


if __name__ == '__main__':

    output_dir = 'output_cv_modified'
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'saved_models'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'loss_plots'), exist_ok=True)

    output_file_path = 'data/Cotton_FL.csv'
    map_file_path = 'data/Cotton.map'

    selected_snp_count = 1000
    input_dim = 8
    num_epochs = 300
    batch_size = 64
    learning_rate = 0.001
    patience = 50
    k_folds = 10
    test_ratio = 0.1


    one_hot_genotype_array, phenotypes_array, snp_count_per_chromosome, scaler = preprocess_data(output_file_path,
                                                                                                 map_file_path)
    chromosome_groups = create_chromosome_groups(snp_count_per_chromosome)
    grouped_chromosome_tensors = prepare_grouped_data(
        one_hot_genotype_array, phenotypes_array, chromosome_groups, snp_count_per_chromosome, selected_snp_count
    )
    if grouped_chromosome_tensors is None: exit()
    phenotypes_tensor = torch.tensor(phenotypes_array, dtype=torch.float32).reshape(-1, 1)


    dataset = TensorDataset(grouped_chromosome_tensors, phenotypes_tensor)


    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(test_ratio * dataset_size))
    np.random.shuffle(indices)
    train_val_indices, test_indices = indices[split:], indices[:split]

    test_dataset = Subset(dataset, test_indices)
    train_val_dataset = Subset(dataset, train_val_indices)

    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)


    cv_test_losses, cv_test_pearson, cv_test_spearman = [], [], []


    best_test_pearson_overall = -1.0
    best_fold_info = {}


    print(f"\nStarting {k_folds}-fold cross-validation...")
    for fold, (train_idx, val_idx) in enumerate(kf.split(train_val_dataset)):
        print(f"\n{'=' * 20} FOLD {fold + 1}/{k_folds} {'=' * 20}")


        train_subsampler = Subset(train_val_dataset, train_idx)
        val_subsampler = Subset(train_val_dataset, val_idx)

        train_dataloader = DataLoader(train_subsampler, batch_size=batch_size, shuffle=True)
        val_dataloader = DataLoader(val_subsampler, batch_size=batch_size, shuffle=False)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


        num_groups = len(chromosome_groups)
        group_snps_list = [selected_snp_count] * num_groups
        model, optimizer = initialize_model_and_optimizer(num_groups, group_snps_list, input_dim, learning_rate)
        criterion = nn.MSELoss()
        early_stopping = EarlyStopping(patience=patience)

        log_file = os.path.join(output_dir, f"training_log_fold_{fold + 1}.txt")
        with open(log_file, "w") as f:
            f.write("Epoch,Train Loss,Val Loss,Val Pearson,Val Spearman\n")

        best_val_pearson = -1.0


        for epoch in range(num_epochs):
            model.train()
            total_train_loss = 0
            for inputs, targets in train_dataloader:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item() * inputs.size(0)
            avg_train_loss = total_train_loss / len(train_subsampler)

            model.eval()
            total_val_loss = 0
            val_preds, val_targets = [], []
            with torch.no_grad():
                for inputs, targets in val_dataloader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    total_val_loss += loss.item() * inputs.size(0)
                    val_preds.extend(outputs.cpu().numpy().flatten())
                    val_targets.extend(targets.cpu().numpy().flatten())

            avg_val_loss = total_val_loss / len(val_subsampler)
            val_pearson, _ = pearsonr(val_preds, val_targets)
            val_spearman, _ = spearmanr(val_preds, val_targets)

            print(f"Epoch {epoch + 1}/{num_epochs} | Val Loss: {avg_val_loss:.4f} | Val Pearson: {val_pearson:.4f}")
            with open(log_file, "a") as f:
                f.write(f"{epoch + 1},{avg_train_loss:.4f},{avg_val_loss:.4f},{val_pearson:.4f},{val_spearman:.4f}\n")

            if val_pearson > best_val_pearson:
                best_val_pearson = val_pearson
                torch.save(model.state_dict(),
                           os.path.join(output_dir, 'saved_models', f'best_model_fold_{fold + 1}.pth'))

            early_stopping(avg_val_loss)
            if early_stopping.early_stop: break


        print(f"--- Evaluating Fold {fold + 1} on Test Set ---")
        model.load_state_dict(torch.load(os.path.join(output_dir, 'saved_models', f'best_model_fold_{fold + 1}.pth')))
        model.eval()
        total_test_loss = 0
        test_preds, test_targets = [], []
        with torch.no_grad():
            for inputs, targets in test_dataloader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                total_test_loss += loss.item() * inputs.size(0)
                test_preds.extend(outputs.cpu().numpy().flatten())
                test_targets.extend(targets.cpu().numpy().flatten())

        avg_test_loss = total_test_loss / len(test_dataset)
        test_pearson, _ = pearsonr(test_preds, test_targets)
        test_spearman, _ = spearmanr(test_preds, test_targets)

        print(
            f"Fold {fold + 1} Test Results: Loss={avg_test_loss:.4f}, Pearson={test_pearson:.4f}, Spearman={test_spearman:.4f}")

        cv_test_losses.append(avg_test_loss)
        cv_test_pearson.append(test_pearson)
        cv_test_spearman.append(test_spearman)

        if test_pearson > best_test_pearson_overall:
            best_test_pearson_overall = test_pearson
            best_fold_info = {
                'fold': fold + 1,
                'loss': avg_test_loss,
                'pearson': test_pearson,
                'spearman': test_spearman
            }


    print(f"\n{'=' * 20} CROSS-VALIDATION SUMMARY {'=' * 20}")
    print(f"Average Test Loss: {np.mean(cv_test_losses):.4f} ± {np.std(cv_test_losses):.4f}")
    print(f"Average Test Pearson Correlation: {np.mean(cv_test_pearson):.4f} ± {np.std(cv_test_pearson):.4f}")
    print(f"Average Test Spearman Correlation: {np.mean(cv_test_spearman):.4f} ± {np.std(cv_test_spearman):.4f}")

    print(f"\n{'=' * 20} BEST PERFORMANCE ACROSS ALL FOLDS {'=' * 20}")
    print(f"Highest Test Pearson Correlation was achieved in Fold {best_fold_info['fold']}:")
    print(f"  - Best Pearson: {best_fold_info['pearson']:.4f}")
    print(f"  - Corresponding Spearman: {best_fold_info['spearman']:.4f}")
    print(f"  - Corresponding Loss: {best_fold_info['loss']:.4f}")


    summary_file = os.path.join(output_dir, "cross_validation_summary.txt")
    with open(summary_file, "w") as f:
        f.write("Cross-Validation Summary:\n")
        f.write(f"Average Test Loss: {np.mean(cv_test_losses):.4f} ± {np.std(cv_test_losses):.4f}\n")
        f.write(f"Average Test Pearson Correlation: {np.mean(cv_test_pearson):.4f} ± {np.std(cv_test_pearson):.4f}\n")
        f.write(
            f"Average Test Spearman Correlation: {np.mean(cv_test_spearman):.4f} ± {np.std(cv_test_spearman):.4f}\n\n")
        f.write("Best Performance Across All Folds:\n")
        f.write(f"Fold: {best_fold_info['fold']}\n")
        f.write(f"  - Best Pearson: {best_fold_info['pearson']:.4f}\n")
        f.write(f"  - Corresponding Spearman: {best_fold_info['spearman']:.4f}\n")
        f.write(f"  - Corresponding Loss: {best_fold_info['loss']:.4f}\n")

    print(f"\nCode execution finished. Check the '{output_dir}' directory for all results.")
