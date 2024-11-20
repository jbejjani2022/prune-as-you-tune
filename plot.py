"""Plot eval accuracy and eval loss using data from different fine-tuning methods."""

import matplotlib.pyplot as plt
import pandas as pd
   
data_dir = 'experiments/distilbert-imdb-full'

# csv files of interest, in `data_dir`
files = ['full_finetune.csv', 'lora_finetune.csv', 'lora_prune_finetune.csv', 'lora_prune_kd_finetune.csv']

datasets = [pd.read_csv(f'{data_dir}/{f}') for f in files]
labels = ['full fine-tuning', 'rsLoRA fine-tuning', 'rsLoRA fine-tuning, 5% pruning / epoch', 'rsLoRA fine-tuning, 5% pruning / epoch, KD loss']

# Create figure and subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Plot eval_accuracy and eval_loss
for i, df in enumerate(datasets):
    ax1.plot(df['epoch'][1:], df['eval_accuracy'][1:], label=labels[i])
    ax2.plot(df['epoch'][1:], df['eval_loss'][1:], label=labels[i])

# Set titles, labels, and legend
ax1.set_title('Evaluation Accuracy')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Evaluation Accuracy')
ax1.legend(title="Methods", loc='lower right')
ax1.set_xticks(datasets[0]['epoch'][1:])

ax2.set_title('Evaluation Loss')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Evaluation Loss')
ax2.set_xticks(datasets[0]['epoch'][1:])

# Set the overall title
fig.suptitle('distilbert-base-uncased (67M) fine-tuning on IMDb')

# Save the plot
plt.tight_layout()
plt.savefig('plot')
