"""Plot eval accuracy and eval loss using data from different fine-tuning methods."""

import matplotlib.pyplot as plt

dataset_1 = [
    {'eval_loss': 0.5721017122268677, 'eval_accuracy': 0.8024, 'eval_runtime': 21.7974, 'eval_samples_per_second': 114.693, 'eval_steps_per_second': 14.36, 'epoch': 1.0},                                                                                                                                          
    {'eval_loss': 0.3710387945175171, 'eval_accuracy': 0.8972, 'eval_runtime': 19.6785, 'eval_samples_per_second': 127.042, 'eval_steps_per_second': 15.906, 'epoch': 2.0},                                                                                                                                                                                                   
    {'eval_loss': 0.4334213435649872, 'eval_accuracy': 0.9032, 'eval_runtime': 19.8803, 'eval_samples_per_second': 125.753, 'eval_steps_per_second': 15.744, 'epoch': 3.0}
]
dataset_2 = [
    {'eval_loss': 0.39401495456695557, 'eval_accuracy': 0.8528, 'eval_runtime': 31.6572, 'eval_samples_per_second': 78.971, 'eval_steps_per_second': 9.887, 'epoch': 1.0},                                                                                                                                                                                                     
    {'eval_loss': 0.2803560495376587, 'eval_accuracy': 0.9024, 'eval_runtime': 29.5265, 'eval_samples_per_second': 84.67, 'eval_steps_per_second': 10.601, 'epoch': 2.0},                                                                                                                                                                                                       
    {'eval_loss': 0.30256250500679016, 'eval_accuracy': 0.904, 'eval_runtime': 29.5639, 'eval_samples_per_second': 84.563, 'eval_steps_per_second': 10.587, 'epoch': 3.0}
]
dataset_3 = [
    {'eval_loss': 0.28556424379348755, 'eval_accuracy': 0.8908, 'eval_runtime': 32.0795, 'eval_samples_per_second': 77.931, 'eval_steps_per_second': 9.757, 'epoch': 1.0},                                                                                                                                                                                                        
    {'eval_loss': 0.2750439941883087, 'eval_accuracy': 0.9064, 'eval_runtime': 29.9789, 'eval_samples_per_second': 83.392, 'eval_steps_per_second': 10.441, 'epoch': 2.0},                                                                                                                                                                                                                         
    {'eval_loss': 0.3003833591938019, 'eval_accuracy': 0.9048, 'eval_runtime': 30.1074, 'eval_samples_per_second': 83.036, 'eval_steps_per_second': 10.396, 'epoch': 3.0}       
]

dataset_4 = [
    {'eval_loss': 0.3683100938796997, 'eval_accuracy': 0.8472, 'eval_runtime': 50.586, 'eval_samples_per_second': 49.421, 'eval_steps_per_second': 6.187, 'epoch': 1.0},
    {'eval_loss': 0.3090507686138153, 'eval_accuracy': 0.902, 'eval_runtime': 48.4634, 'eval_samples_per_second': 51.585, 'eval_steps_per_second': 6.458, 'epoch': 2.0},
    {'eval_loss': 0.3165464997291565, 'eval_accuracy': 0.9036, 'eval_runtime': 48.4573, 'eval_samples_per_second': 51.592, 'eval_steps_per_second': 6.459, 'epoch': 3.0}
]      

# Data preparation
datasets = [dataset_1, dataset_2, dataset_3, dataset_4]
labels = ['full fine-tuning', 'rsLoRA fine-tuning', 'rsLoRA fine-tuning, 5% pruning / epoch', 'rsLoRA fine-tuning, 5% pruning / epoch, KD loss']

# Create figure and subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Plot eval_accuracy and eval_loss
for i, data in enumerate(datasets):
    epochs = [entry['epoch'] for entry in data]
    eval_accuracy = [entry['eval_accuracy'] for entry in data]
    eval_loss = [entry['eval_loss'] for entry in data]
    
    ax1.plot(epochs, eval_accuracy, label=labels[i])
    ax2.plot(epochs, eval_loss, label=labels[i])

# Set titles, labels, and legend
ax1.set_title('Evaluation Accuracy')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Evaluation Accuracy')
ax1.set_xticks([1, 2, 3])

ax2.set_title('Evaluation Loss')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Evaluation Loss')
ax2.legend(title="Methods", loc='upper right')
ax2.set_xticks([1, 2, 3])

# Set the overall title
fig.suptitle('distilbert-base-uncased (88M) fine-tuning on (10% of) IMDb')

# Save the plot
plt.tight_layout()
plt.savefig('plot')
