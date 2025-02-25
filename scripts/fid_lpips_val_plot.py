import pandas as pd
import numpy as np
import wandb
from scipy.signal import savgol_filter

# Load CSV Data
loss_df = pd.read_csv("valloss.csv")
lpips_df = pd.read_csv("lpips.csv")
fid_df   = pd.read_csv("fid.csv")

# Rename Columns for Consistency
loss_df = loss_df.rename(columns={'Step': 'Step', 'Panime - val/loss': 'Loss'})
lpips_df = lpips_df.rename(columns={'LPIPS-Per-Epoch - average_lpips': 'LPIPS'})
fid_df = fid_df.rename(columns={'FID-Per-Epoch - fid': 'FID'})

# Merge DataFrames on 'Step'
merged_df = pd.merge(loss_df[['Step', 'Loss']], lpips_df[['Step', 'LPIPS']], on='Step', how='outer')
merged_df = pd.merge(merged_df, fid_df[['Step', 'FID']], on='Step', how='outer')
merged_df = merged_df.sort_values('Step')
merged_df = merged_df.dropna(subset=['Step', 'Loss', 'LPIPS', 'FID'])
merged_df['Step'] = merged_df['Step'].astype(float)

# Normalize the Metrics
def normalize(series):
    return (series - series.min()) / (series.max() - series.min())

merged_df['Loss_norm']   = normalize(merged_df['Loss'])
merged_df['LPIPS_norm']  = normalize(merged_df['LPIPS'])
merged_df['FID_norm']    = normalize(merged_df['FID'])

# Compute the average using only LPIPS_norm and FID_norm.
merged_df['Average_norm'] = merged_df[['LPIPS_norm', 'FID_norm']].mean(axis=1)

# Compute Smoothed Averages
merged_df['Average_norm_ema'] = merged_df['Average_norm'].ewm(span=5, adjust=False).mean()
window_length = 7  # must be odd
polyorder = 2
merged_df['Average_norm_savgol'] = savgol_filter(merged_df['Average_norm'], window_length, polyorder)

# Smooth the raw validation loss using Savitzky–Golay filter.
merged_df['Loss_savgol'] = savgol_filter(merged_df['Loss'], 11, polyorder)

# Initialize WandB and log the merged data as a table.
wandb.init(project="Find Epoch", name="normalized_and_smoothed_metrics_vs_step")
table = wandb.Table(data=merged_df.values.tolist(), columns=merged_df.columns.tolist())
wandb.log({"merged_metrics_table": table})
x_data = merged_df['Step'].tolist()

# Graph 1: Normalized Metrics vs Step (Loss_norm, LPIPS_norm, FID_norm)
y_data_normalized = [
    merged_df['Loss_norm'].tolist(),
    merged_df['LPIPS_norm'].tolist(),
    merged_df['FID_norm'].tolist()
]
keys_normalized = ["Loss_norm", "LPIPS_norm", "FID_norm"]
wandb.log({
    "normalized_metrics_vs_step": wandb.plot.line_series(
        xs=x_data,
        ys=y_data_normalized,
        keys=keys_normalized,
        title="Normalized Metrics vs Step",
        xname="Step"
    )
})

# Graph 2: Raw Average (of LPIPS_norm and FID_norm) vs Step
y_data_average = [ merged_df['Average_norm'].tolist() ]
keys_average = ["Average_norm"]
wandb.log({
    "average_normalized_metric_vs_step": wandb.plot.line_series(
        xs=x_data,
        ys=y_data_average,
        keys=keys_average,
        title="Raw Average (LPIPS & FID) Normalized vs Step",
        xname="Step"
    )
})

# Graph 3: EMA Smoothed Average vs Step
y_data_ema = [ merged_df['Average_norm_ema'].tolist() ]
keys_ema = ["Average_norm_ema"]
wandb.log({
    "ema_smoothed_average_normalized_metric_vs_step": wandb.plot.line_series(
        xs=x_data,
        ys=y_data_ema,
        keys=keys_ema,
        title="EMA Smoothed Average (LPIPS & FID) vs Step",
        xname="Step"
    )
})

# Graph 4: Savitzky–Golay Smoothed Average vs Step
y_data_savgol = [ merged_df['Average_norm_savgol'].tolist() ]
keys_savgol = ["Average_norm_savgol"]
wandb.log({
    "savgol_smoothed_average_normalized_metric_vs_step": wandb.plot.line_series(
        xs=x_data,
        ys=y_data_savgol,
        keys=keys_savgol,
        title="Savitzky–Golay Smoothed Average (LPIPS & FID) vs Step",
        xname="Step"
    )
})

# Graph 5: Smoothed Validation Loss vs Step using Savitzky–Golay filter
y_data_loss = [ merged_df['Loss_savgol'].tolist() ]
keys_loss = ["Loss_savgol"]
wandb.log({
    "validation_loss_smoothed_vs_step": wandb.plot.line_series(
        xs=x_data,
        ys=y_data_loss,
        keys=keys_loss,
        title="Smoothed Validation Loss vs Step",
        xname="Step"
    )
})

wandb.finish()
