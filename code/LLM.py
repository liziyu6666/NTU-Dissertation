import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import os
from collections import Counter
from code import sol, num_agents, faulty_agent  # 引用系统代码中的仿真数据

# ================== 生成训练数据 ==================
num_samples = sol.y.shape[1]  # 获取仿真数据的时间步长
data = []
labels = []

def format_signal(agent_id, t_index):
    """ 将系统状态转换为文本描述 """
    state = sol.y[agent_id * 6:(agent_id + 1) * 6, t_index]
    text_input = f"Node {agent_id}: State={state.tolist()}"
    return text_input

for agent_id in range(num_agents):
    for t_index in range(num_samples):
        data.append(format_signal(agent_id, t_index))
        labels.append(1 if agent_id == faulty_agent else 0)

# ================== 训练集划分 ==================
train_texts, test_texts, train_labels, test_labels = train_test_split(data, labels, test_size=0.2, random_state=42)

# 数据集分布检查
print("Training label distribution:", Counter(train_labels))
print("Test label distribution:", Counter(test_labels))

# ================== 处理文本数据 ==================
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

class SignalDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=128)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

train_dataset = SignalDataset(train_texts, train_labels)
test_dataset = SignalDataset(test_texts, test_labels)

# ================== 加载 BERT 模型 ==================
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# ================== 计算评估指标 ==================
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "eval_accuracy": accuracy_score(labels, predictions),
        "precision": precision_score(labels, predictions),
        "recall": recall_score(labels, predictions),
        "f1": f1_score(labels, predictions)
    }

# ================== 训练参数 ==================
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=3,  # 训练轮次
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    logging_dir="./logs",
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="eval_accuracy",
    learning_rate=2e-5,
    fp16=True,
    save_total_limit=2,  # 只保留最近2个最佳模型
    greater_is_better=True
)

# ================== 训练模型 ==================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)

trainer.train()

# ================== 保存模型 ==================
model.save_pretrained("./saved_model")
tokenizer.save_pretrained("./saved_model")
print("Model saved to ./saved_model")

# ================== 可视化训练过程 ==================
metrics = trainer.state.log_history
train_loss = [m["loss"] for m in metrics if "loss" in m]
eval_acc = [m["eval_accuracy"] for m in metrics if "eval_accuracy" in m]

def plot_metrics():
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_loss, label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(eval_acc, label='Eval Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Evaluation Accuracy Curve')
    plt.legend()
    
    plt.show()

plot_metrics()
