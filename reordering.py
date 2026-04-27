import pandas as pd

def reorder_dataset_by_snr(metrics_csv, mixture_csv):

    metrics_df = pd.read_csv(metrics_csv)
    mixture_df = pd.read_csv(mixture_csv)

    mask = metrics_df['source_2_SNR'] > metrics_df['source_1_SNR']

    temp_s1 = mixture_df.loc[mask, 'source_1_path'].copy()

    mixture_df.loc[mask, 'source_1_path'] = mixture_df.loc[mask, 'source_2_path']
    mixture_df.loc[mask, 'source_2_path'] = temp_s1

    mixture_df.to_csv(mixture_csv.replace('.csv', '_ordered.csv'), index=False)

    print(f"Reordered {mask.sum()} samples based on SNR.")

train_csv = r"C:\Users\umroot\Projects\Speech\LibriMix\train-data\Libri2Mix\wav16k\min\metadata\mixture_train-100_mix_clean.csv"
test_csv = r"C:\Users\umroot\Projects\Speech\LibriMix\train-data\Libri2Mix\wav16k\min\metadata\mixture_test_mix_clean.csv"
dev_csv = r"C:\Users\umroot\Projects\Speech\LibriMix\train-data\Libri2Mix\wav16k\min\metadata\mixture_dev_mix_clean.csv"

reorder_dataset_by_snr(r"C:\Users\umroot\Projects\Speech\LibriMix\train-data\Libri2Mix\wav16k\min\metadata\metrics_train-100_mix_clean.csv", train_csv)
reorder_dataset_by_snr(r"C:\Users\umroot\Projects\Speech\LibriMix\train-data\Libri2Mix\wav16k\min\metadata\metrics_test_mix_clean.csv", test_csv)
reorder_dataset_by_snr(r"C:\Users\umroot\Projects\Speech\LibriMix\train-data\Libri2Mix\wav16k\min\metadata\metrics_dev_mix_clean.csv", dev_csv)

