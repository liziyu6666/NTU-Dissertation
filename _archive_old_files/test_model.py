import torch
from transformers import BertTokenizer, BertForSequenceClassification
from code import sol, faulty_agent  # 引用系统代码中的仿真数据

# ================== 加载已训练好的模型 ==================
def load_model():
    model = BertForSequenceClassification.from_pretrained("./saved_model")
    tokenizer = BertTokenizer.from_pretrained("./saved_model")
    return model, tokenizer

model, tokenizer = load_model()
print("Model loaded from ./saved_model")

# ================== 生成测试数据 ==================
def format_signal(agent_id, t_index):
    """ 将系统状态转换为文本描述 """
    state = sol.y[agent_id * 6:(agent_id + 1) * 6, t_index]
    text_input = f"Node {agent_id}: State={state.tolist()}"
    return text_input

def predict(text):
    """ 预测节点是否为拜占庭节点 """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    outputs = model(**inputs)
    prediction = torch.argmax(outputs.logits, dim=-1).item()
    return "Byzantine" if prediction == 1 else "Normal"

# ================== 进行预测 ==================
test_text = format_signal(faulty_agent, 0)
prediction_result = predict(test_text)
print(f"Test Node Prediction: {prediction_result}")
