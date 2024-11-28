"""Plot eval accuracy and eval loss using data from different fine-tuning methods."""

import matplotlib.pyplot as plt
import pandas as pd
   
data_dir = 'experiments/bert-yelp-review-full/5epochs-25pct-sparsity'

# csv files of interest, in `data_dir`
files = ['lora_prune_kd_finetune_not_rs.csv', 'lora_prune_kd_finetune.csv']

datasets = [pd.read_csv(f'{data_dir}/{f}') for f in files]
labels = ['LoRA', 'rsLoRA']

# Create figure and subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Plot eval_accuracy and eval_loss
for i, df in enumerate(datasets):
    ax1.plot(df['epoch'], df['eval_accuracy'], label=labels[i])
    ax2.plot(df['epoch'], df['eval_loss'], label=labels[i])

# Set titles, labels, and legend
ax1.set_title('Evaluation Accuracy')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Evaluation Accuracy')
ax1.legend(title="Methods", loc='lower right')
ax1.set_xticks(datasets[0]['epoch'])

ax2.set_title('Evaluation Loss')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Evaluation Loss')
ax2.set_xticks(datasets[0]['epoch'])

# Set the overall title
fig.suptitle('bert-base-uncased (110M) fine-tuning on yelp_review_full (5% pruning / epoch, KD loss)')

# Save the plot
plt.tight_layout()
plt.savefig('plot')
