import os
import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset, Subset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from scipy.stats import pearsonr, spearmanr
from data_process.encode import genotype_to_dataframe, base_one_hot_encoding_1v, base_one_hot_encoding_8v_dif, MICSelector
from model.model_1v_depth import PhenotypePredictor_1v
from model.early_stop import EarlyStopping


def main():

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    os.makedirs('output/saved_models', exist_ok=True)
    os.makedirs('output/loss_png', exist_ok=True)

    merged_df = pd.read_csv('data/Maize_PH_HN.csv', sep='\t')
    genotype_data = merged_df['genotype']
    phenotype_data = merged_df['phenotype'].values


    genotype_df = genotype_to_dataframe(genotype_data)
    one_hot_8v = base_one_hot_encoding_8v_dif(genotype_df).values
    one_hot_1v = base_one_hot_encoding_1v(genotype_df).values  


    phenotypes = StandardScaler().fit_transform(phenotype_data.reshape(-1, 1)).flatten()


    map_df = pd.read_csv('data/Mazie.map', sep='\t', header=None,
                         names=['chromosome', 'snp_id', 'map', 'position'])
    snp_counts = map_df.groupby('chromosome').size().tolist()
    num_chromosomes = len(snp_counts)


    chromosome_data_list = []
    start_idx = 0
    selected_snp_count = 1000

    for count in snp_counts:
        end_idx = start_idx + count


        chrom_1v = one_hot_1v[:, start_idx:end_idx]
        chrom_8v = one_hot_8v[:, start_idx * 8:end_idx * 8].reshape(-1, count, 8)


        mic_selector = MICSelector(k=selected_snp_count)
        mic_selector.fit(chrom_1v, phenotypes)
        selected_indices = mic_selector.top_k_indices_


        selected_data = chrom_8v[:, selected_indices, :]


        padding = np.zeros((selected_data.shape[0], selected_snp_count - selected_data.shape[1], 8))
        padded_data = np.concatenate((selected_data, padding), axis=1)

        chromosome_data_list.append(padded_data)
        start_idx = end_idx


    chromosome_tensors = torch.tensor(np.stack(chromosome_data_list, axis=1), dtype=torch.float32)
    phenotypes_tensor = torch.tensor(phenotypes, dtype=torch.float32).reshape(-1, 1)
    dataset = TensorDataset(chromosome_tensors, phenotypes_tensor)


    test_size = int(0.1 * len(dataset))
    indices = torch.randperm(len(dataset))
    test_indices = indices[:test_size]
    train_val_indices = indices[test_size:]
    test_dataset = Subset(dataset, test_indices)


    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    cv_results = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(train_val_indices)):
        print(f"\n=== Fold {fold + 1}/10 ===")


        train_dataset = Subset(dataset, train_val_indices[train_idx])
        val_dataset = Subset(dataset, train_val_indices[val_idx])

        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


        model = PhenotypePredictor_1v(
            num_chromosomes,
            [selected_snp_count] * num_chromosomes,
            input_dim=8
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.MSELoss()
        early_stop = EarlyStopping(patience=20, verbose=True)


        train_losses, val_losses = [], []
        best_val_loss = float('inf')

        for epoch in range(300):

            model.train()
            epoch_train_loss = 0
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                epoch_train_loss += loss.item() * inputs.size(0)

            train_loss = epoch_train_loss / len(train_dataset)
            train_losses.append(train_loss)


            model.eval()
            val_loss, val_preds, val_targets = 0, [], []
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item() * inputs.size(0)
                    val_preds.append(outputs.cpu())
                    val_targets.append(targets.cpu())

            val_loss = val_loss / len(val_dataset)
            val_losses.append(val_loss)
            val_pearson = pearsonr(
                torch.cat(val_preds).numpy().flatten(),
                torch.cat(val_targets).numpy().flatten()
            )[0]


            print(f"Epoch {epoch + 1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Pearson={val_pearson:.4f}")


            early_stop(val_loss, model)
            if early_stop.early_stop:
                print("Early stopping triggered")
                break


        model.load_state_dict(torch.load('checkpoint.pt'))
        test_loss, test_preds, test_targets = 0, [], []
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                test_loss += criterion(outputs, targets).item() * inputs.size(0)
                test_preds.append(outputs.cpu())
                test_targets.append(targets.cpu())

        test_loss /= len(test_dataset)
        test_pearson = pearsonr(
            torch.cat(test_preds).numpy().flatten(),
            torch.cat(test_targets).numpy().flatten()
        )[0]


        cv_results.append({
            'fold': fold + 1,
            'best_val_loss': min(val_losses),
            'test_loss': test_loss,
            'test_pearson': test_pearson
        })


        torch.save(model.state_dict(), f'output/saved_models/model_fold_{fold + 1}.pth')


    print("\n=== Cross-Validation Results ===")
    for res in cv_results:
        print(f"Fold {res['fold']}: Test Loss={res['test_loss']:.4f}, Pearson={res['test_pearson']:.4f}")

    avg_pearson = np.mean([res['test_pearson'] for res in cv_results])
    print(f"\nAverage Pearson Correlation: {avg_pearson:.4f}")


if __name__ == '__main__':
    main()