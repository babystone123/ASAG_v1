"""
简答题评分，模型部分
"""
import os
import time
import csv
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import BertTokenizer, BertModel
from documents_preprocess import load_con_dataset
import model_judge


# load model
model_name = 'D:\\myAImodel\\bert-base-chinese'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bert_tokenizer = BertTokenizer.from_pretrained(model_name)

# 参数设置
epochs = [30, 12]
initial_learning_rates = [0.00002, 0.0002]  # 三个学习率，分别对应三个模型
learning_rate_decays = [0.5, 0.4]  # 学习率衰减因子，可以根据需要调整
weight_decay = [0.001, 0.002]  # 池化参数
Step_decay_parameter = 4  # 隔多少步衰减

loss_save_name = "modal_all_loss.csv"
whether_model_train = [1, 1]  # 控制模型训练


# 文本压缩，若文本过长可选使用
class TextCompression:
    def __init__(self):
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        modal_name = "D:\\myAImodel\\t5-finetune-cnndaily-news"
        self.t5_tokenizer = AutoTokenizer.from_pretrained(modal_name)
        self.t5_model = AutoModelForSeq2SeqLM.from_pretrained(modal_name).cuda()

    def compression(self, src):
        tokenized_text = self.t5_tokenizer.encode(src, return_tensors="pt").cuda()
        self.t5_model.eval()
        summary_ids = self.t5_model.generate(tokenized_text, max_length=128)
        output = self.t5_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return output


# 定义数据集
class EntailmentDataset(Dataset):
    def __init__(self, data, max_length=128):
        self.tokenizer = bert_tokenizer
        self.dataset = data
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        student_answer = self.dataset[index][1]
        teacher_answer = self.dataset[index][0]
        label = torch.tensor(self.dataset[index][2])

        # 使用BertTokenizer编码学生和老师答案
        encoded_inputs = self.tokenizer(student_answer, teacher_answer,
                                        padding='max_length',
                                        truncation='do_not_truncate',
                                        max_length=self.max_length,
                                        return_tensors='pt'
                                        )

        input_ids = encoded_inputs['input_ids'].squeeze()
        attention_mask = encoded_inputs['attention_mask'].squeeze()
        token_type_ids = encoded_inputs['token_type_ids'].squeeze()

        return input_ids, attention_mask, token_type_ids, label


def collate_fn_entailment(batch):
    # 从batch中获取input_ids、attention_mask、token_type_ids和labels
    input_ids = [item[0] for item in batch]
    attention_mask = [item[1] for item in batch]
    token_type_ids = [item[2] for item in batch]
    labels = torch.tensor([item[3] for item in batch], dtype=torch.float32)

    # 计算batch中的最大长度
    max_length = max(len(ids) for ids in input_ids)

    # 动态填充句子，保证batch中的所有句子都具有相同长度（最大长度为128）
    padded_input_ids = [torch.cat([ids, torch.zeros(max_length - len(ids), dtype=torch.long)])
                        for ids in input_ids]
    padded_attention_mask = [torch.cat([mask, torch.zeros(max_length - len(mask), dtype=torch.long)])
                             for mask in attention_mask]
    padded_token_type_ids = [torch.cat([ids, torch.zeros(max_length - len(ids), dtype=torch.long)])
                             for ids in token_type_ids]

    # 将填充后的张量转换为tensor并返回
    return torch.stack(padded_input_ids), torch.stack(padded_attention_mask), torch.stack(padded_token_type_ids), labels


class SimilarDataset(Dataset):
    def __init__(self, data, max_length=128):
        self.tokenizer = bert_tokenizer
        self.dataset = data
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        student_answer = self.dataset[index][1]
        teacher_answer = self.dataset[index][0]
        label = torch.tensor(self.dataset[index][2])

        # 使用BertTokenizer编码学生和老师答案
        encoder_student_answer = self.tokenizer(student_answer,
                                                padding='max_length',
                                                truncation='do_not_truncate',
                                                max_length=self.max_length,
                                                return_tensors='pt'
                                                )
        encoder_teacher_answer = self.tokenizer(teacher_answer,
                                                padding='max_length',
                                                truncation='do_not_truncate',
                                                max_length=self.max_length,
                                                return_tensors='pt'
                                                )

        input_ids_student = encoder_student_answer['input_ids'].squeeze()
        attention_mask_student = encoder_student_answer['attention_mask'].squeeze()
        input_ids_teacher = encoder_teacher_answer['input_ids'].squeeze()
        attention_mask_teacher = encoder_teacher_answer['attention_mask'].squeeze()

        return input_ids_student, attention_mask_student, input_ids_teacher, attention_mask_teacher, label


def collate_fn_similar(batch):
    # 从batch中获取input_ids_student, attention_mask_student, input_ids_teacher, attention_mask_teacher和label
    input_ids_student = [item[0] for item in batch]
    attention_mask_student = [item[1] for item in batch]
    input_ids_teacher = [item[2] for item in batch]
    attention_mask_teacher = [item[3] for item in batch]
    labels = torch.tensor([item[4] for item in batch], dtype=torch.float32)

    # 计算batch中的最大长度
    max_length = max(len(ids) for ids in input_ids_student + input_ids_teacher)

    # 动态填充句子，保证batch中的所有句子都具有相同长度（最大长度为128）
    padded_input_ids_student = [torch.cat([ids, torch.zeros(max_length - len(ids), dtype=torch.long)])
                                for ids in input_ids_student]
    padded_attention_mask_student = [torch.cat([mask, torch.zeros(max_length - len(mask), dtype=torch.long)])
                                     for mask in attention_mask_student]
    padded_input_ids_teacher = [torch.cat([ids, torch.zeros(max_length - len(ids), dtype=torch.long)])
                                for ids in input_ids_teacher]
    padded_attention_mask_teacher = [torch.cat([mask, torch.zeros(max_length - len(mask), dtype=torch.long)])
                                     for mask in attention_mask_teacher]

    # 将填充后的张量转换为tensor并返回
    return torch.stack(padded_input_ids_student), \
        torch.stack(padded_attention_mask_student), \
        torch.stack(padded_input_ids_teacher), \
        torch.stack(padded_attention_mask_teacher), \
        labels


# 定义模型
class EntailmentModel(nn.Module):
    def __init__(self, input_size=768, hidden_size=256, output_size=1):
        super().__init__()
        self.model = BertModel.from_pretrained(model_name).to(device)
        # 添加两个全连接层作为感知机，作为文本蕴含输出
        self.predictor = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),  # 将输出维度设为1，因为是回归任务
        )
        # 初始化感知机部分的权重
        for layer in self.predictor:
            if isinstance(layer, nn.Linear):
                if layer.bias is not None:
                    init.zeros_(layer.bias)
                if layer.weight is not None:
                    if isinstance(layer, nn.Linear):
                        init.xavier_uniform_(layer.weight)

    def forward(self, input_ids, attention_mask, token_type_ids):
        Bert_outputs = self.model(input_ids=input_ids,
                                  attention_mask=attention_mask,
                                  token_type_ids=token_type_ids)
        # 取[CLS]对应的隐藏状态向量作为输出
        mlp_outputs_cls_hidden_state = Bert_outputs.last_hidden_state[:, 0, :]
        mlp_output = self.predictor(mlp_outputs_cls_hidden_state).squeeze()
        return mlp_output


class SimilarModel(nn.Module):
    def __init__(self, input_size=768, hidden_size=256, output_size=64):
        super().__init__()
        self.model = BertModel.from_pretrained(model_name).to(device)

        self.pooling = nn.AdaptiveAvgPool1d(1)  # 池化层，用于对所有时间步进行池化
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        # 初始化GRU的权重
        for name, par in self.gru.named_parameters():
            if 'weight' in name:
                init.xavier_uniform_(par)

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, sentence1_ids, sentence1_mask, sentence2_ids, sentence2_mask):
        Bert_sentence1_outputs = self.model(input_ids=sentence1_ids, attention_mask=sentence1_mask)
        Bert_sentence2_outputs = self.model(input_ids=sentence2_ids, attention_mask=sentence2_mask)
        # 取[CLS]对应的隐藏状态向量作为输出
        sentence1_output_cls_hidden_state = Bert_sentence1_outputs.last_hidden_state
        sentence2_output_cls_hidden_state = Bert_sentence2_outputs.last_hidden_state

        # 先将形状改为（batch_size, hidden_size, sequence_length），再对所有时间步进行池化
        pool_GRU_output1 = self.pooling(sentence1_output_cls_hidden_state.permute(0, 2, 1))
        pool_GRU_output1 = pool_GRU_output1.squeeze()  # 将形状改为（batch_size, hidden_size）
        pool_GRU_output2 = self.pooling(sentence2_output_cls_hidden_state.permute(0, 2, 1))
        pool_GRU_output2 = pool_GRU_output2.squeeze()

        GRU_out1, _ = self.gru(pool_GRU_output1.unsqueeze(1))
        GRU_out2, _ = self.gru(pool_GRU_output2.unsqueeze(1))
        out1 = self.fc(GRU_out1[:, -1, :])
        out2 = self.fc(GRU_out2[:, -1, :])

        similarity_input = torch.cosine_similarity(out1, out2, dim=1)  # 计算相似度

        return 2.5 * similarity_input + 2.5


# 数据集加载
def models_dataset_loader3():
    # 加载并准备数据集
    dataset = load_con_dataset()  # 若需要文本压缩则在这步
    dataset_rem = []
    for i in range(len(dataset[2])):
        dataset_rem.append([dataset[0][i], dataset[1][i], dataset[2][i]])

    # 定义训练集和验证集的比例
    train_ratio = 0.9
    # 计算数据集的样本数量
    dataset_size = len(dataset_rem)
    train_size = int(train_ratio * dataset_size)
    val_size = dataset_size - train_size
    train_dataset, val_dataset = random_split(dataset_rem, [train_size, val_size])

    # 划分训练集和验证集
    mlp_train_dataset, mlp_val_dataset = EntailmentDataset(train_dataset), EntailmentDataset(val_dataset)
    gru_train_dataset, gru_val_dataset = SimilarDataset(train_dataset), SimilarDataset(val_dataset)

    mlp_train_dataloader = DataLoader(mlp_train_dataset, batch_size=4,
                                      collate_fn=collate_fn_entailment, drop_last=True)
    mlp_val_dataloader = DataLoader(mlp_val_dataset, batch_size=4,
                                    collate_fn=collate_fn_entailment, drop_last=True)

    gru_train_dataloader = DataLoader(gru_train_dataset, batch_size=4,
                                      collate_fn=collate_fn_similar, drop_last=True)
    gru_val_dataloader = DataLoader(gru_val_dataset, batch_size=4,
                                    collate_fn=collate_fn_similar, drop_last=True)

    return [mlp_train_dataloader, gru_train_dataloader,
            mlp_val_dataloader, gru_val_dataloader, val_dataset]


# model train
def model_train():
    start_time = time.time()  # 开始时间
    save_infor_to_csv(True, ['train and val loss', 'unm epoch loss', 'others\' data about order'])  # 清空文件，用于记录新训练损失
    # 加载模型
    models = [EntailmentModel().to(device),
              SimilarModel().to(device)]
    # 加载数据
    datasets = models_dataset_loader3()

    print("模型和数据加载成功，开始训练")

    # 定义优化器和损失
    optimizers = []  # 存储三个优化器
    for i in range(2):
        model = models[i]
        optimizer = torch.optim.NAdam(model.parameters(),
                                      lr=initial_learning_rates[i],
                                      weight_decay=weight_decay[i])
        optimizers.append(optimizer)
    criterion = nn.MSELoss()  # 使用均方误差损失（MSE Loss）

    # 开始训练
    if whether_model_train[0]:
        best_loss = 100
        # MLP模型训练
        mlp_tra_total_lo = []  # 记录模型每个训练步的损失之和
        mlp_val_total_lo = []
        for epoch in range(epochs[0]):
            # 训练
            train_losses = 0  # 分别记录三个模型的损失之和
            mlp_model = models[0].train()
            for mlp_batch_idx, mlp_batch in enumerate(datasets[0], 1):
                # 解包数据批次
                mlp_input_ids, mlp_attention_mask, mlp_token_type_ids, mlp_label = mlp_batch

                # 将数据放入显卡中
                mlp_input_ids, mlp_attention_mask, mlp_token_type_ids, \
                    mlp_label = data_to_cuda(mlp_input_ids, mlp_attention_mask, mlp_token_type_ids, mlp_label)

                # 模型输出
                output = mlp_model(mlp_input_ids, mlp_attention_mask, mlp_token_type_ids)

                # 计算损失，反向传播，更新参数
                loss = criterion(output, mlp_label)  # 需根据实际情况修改

                # 反向传播和参数优化
                optimizers[0].zero_grad()
                loss.backward()
                optimizers[0].step()  # 密集梯度

                train_losses += loss.item()

                # 每100个批次打印一次损失
                if mlp_batch_idx % 100 == 0:
                    print(
                        f"Epoch [{epoch + 1}/{epochs[0]}] - Batch [{mlp_batch_idx}/{len(datasets[0])}] - MLP: train - "
                        f"Now Loss: {round(loss.item(), 3)} - Average Loss:{round(train_losses / mlp_batch_idx, 3)}")
                    # 储存损失
                    save_infor_to_csv(False,
                                      f"Epoch [{epoch + 1}/{epochs[0]}] - Batch [{mlp_batch_idx}/{len(datasets[0])}] - "
                                      f"MLP: train - Now Loss: {round(loss.item(), 3)} - "
                                      f"Average Loss:{round(train_losses / mlp_batch_idx, 3)}", 0)

                # 测试
                # if mlp_batch_idx >= 50 and epoch == 2:
                #     print('开始测试')

            # 验证
            val_losses = 0
            mlp_model.eval()
            for mlp_batch_idx, mlp_batch in enumerate(datasets[2], 1):
                # 解包数据批次
                mlp_input_ids, mlp_attention_mask, mlp_token_type_ids, mlp_label = mlp_batch

                # 将数据放入显卡中
                mlp_input_ids, mlp_attention_mask, mlp_token_type_ids, \
                    mlp_label = data_to_cuda(mlp_input_ids, mlp_attention_mask, mlp_token_type_ids, mlp_label)
                # 分别计算三个模型的输出
                # 模型输出
                output = mlp_model(mlp_input_ids, mlp_attention_mask, mlp_token_type_ids)

                # 计算损失
                loss = criterion(output, mlp_label)  # 需根据实际情况修改
                val_losses += loss.item()
                # 每10个批次打印一次损失
                if mlp_batch_idx % 10 == 0:
                    print(
                        f"Epoch [{epoch + 1}/{epochs[0]}] - Batch [{mlp_batch_idx}/{len(datasets[2])}] MLP:"
                        f" val - "
                        f"Now Loss: {round(loss.item(), 3)} - Average Loss:{round(val_losses / mlp_batch_idx, 3)}")

            # 记录损失
            mlp_tra_total_lo.append(train_losses / len(datasets[0]))
            mlp_val_total_lo.append(val_losses / len(datasets[2]))
            print(f"Epoch [{epoch + 1}/{epochs[0]}] MLP: Train Average Losses: {mlp_tra_total_lo[-1]:.4f}"
                  f"- Val Average Loss: {mlp_val_total_lo[-1]:.4f}")
            # 储存损失
            save_infor_to_csv(False,
                              f"Epoch [{epoch + 1}/{epochs[0]}] MLP: Train Average Losses: {mlp_tra_total_lo[-1]:.4f}"
                              f"- Val Average Loss: {mlp_val_total_lo[-1]:.4f}", 1)

            # 储存模型
            if mlp_val_total_lo[-1] <= best_loss:
                best_loss = mlp_val_total_lo[-1]
                torch.save(models[0],
                           os.path.join('D:\\myAImodel\\my_automatic_grading\\output_models_ls', "mlp_model.pt"))

        # 绘制图像
        # plot_loss(mlp_tra_total_lo, mlp_val_total_lo, 'MLP train and validation losses')

    if whether_model_train[1]:
        # gru模型训练
        best_loss = 100
        gru_tra_total_lo = []  # 记录模型每个训练步的损失之和
        gru_val_total_lo = []
        for epoch in range(epochs[1]):
            # 训练
            train_losses = 0  # 分别记录三个模型的损失之和
            gru_model = models[1].train()
            for gru_batch_idx, gru_batch in enumerate(datasets[1], 1):
                # 解包数据批次
                gru_input_ids_student, gru_attention_mask_student, gru_input_ids_teacher, gru_attention_mask_teacher, \
                    gru_label = gru_batch

                # 将数据放入显卡中
                gru_input_ids_student, gru_attention_mask_student, gru_input_ids_teacher, gru_attention_mask_teacher, \
                    gru_label = data_to_cuda(gru_input_ids_student, gru_attention_mask_student,
                                             gru_input_ids_teacher, gru_attention_mask_teacher, gru_label)

                # 模型输出
                output = gru_model(gru_input_ids_student, gru_attention_mask_student,
                                   gru_input_ids_teacher, gru_attention_mask_teacher)

                # 计算损失，反向传播，更新参数
                loss = criterion(output, gru_label)  # 需根据实际情况修改

                # 反向传播和参数优化
                optimizers[1].zero_grad()
                loss.backward()
                optimizers[1].step()  # 密集梯度

                train_losses += loss.item()

                # 每100个批次打印一次损失
                if gru_batch_idx % 100 == 0:
                    print(
                        f"Epoch:[{epoch + 1}/{epochs[1]}] - Batch:[{gru_batch_idx}/{len(datasets[1])}] - GRU: train - "
                        f"Now Loss: {round(loss.item(), 3)} - Average Loss:{round(train_losses / gru_batch_idx, 3)}")
                    # 储存损失
                    save_infor_to_csv(False,
                                      f"Epoch:[{epoch + 1}/{epochs[1]}] - Batch:[{gru_batch_idx}/{len(datasets[1])}] - "
                                      f"GRU: train - Now Loss: {round(loss.item(), 3)} - "
                                      f"Average Loss:{round(train_losses / gru_batch_idx, 3)}", 0)

                # 测试
                # if gru_batch_idx >= 50 and epoch == 2:
                #     print('开始测试')

            # 验证
            val_losses = 0
            gru_model.eval()
            for gru_batch_idx, gru_batch in enumerate(datasets[3], 1):
                # 解包数据批次
                gru_input_ids_student, gru_attention_mask_student, \
                    gru_input_ids_teacher, gru_attention_mask_teacher, gru_label = gru_batch

                # 将数据放入显卡中
                gru_input_ids_student, gru_attention_mask_student, gru_input_ids_teacher, gru_attention_mask_teacher, \
                    gru_label = data_to_cuda(gru_input_ids_student, gru_attention_mask_student,
                                             gru_input_ids_teacher, gru_attention_mask_teacher, gru_label)
                # 分别计算三个模型的输出
                # 模型输出
                output = gru_model(gru_input_ids_student, gru_attention_mask_student,
                                   gru_input_ids_teacher, gru_attention_mask_teacher)

                # 计算损失
                loss = criterion(output, gru_label)  # 需根据实际情况修改
                val_losses += loss.item()
                # 每10个批次打印一次损失
                if gru_batch_idx % 10 == 0:
                    print(
                        f"Epoch [{epoch + 1}/{epochs[1]}] - Batch [{gru_batch_idx}/{len(datasets[3])}] GRU:"
                        f"val - Now Loss: {round(loss.item(), 3)} - "
                        f"Average Loss:{round(val_losses / gru_batch_idx, 3)}")

            # 记录损失
            gru_tra_total_lo.append(train_losses / len(datasets[1]))
            gru_val_total_lo.append(val_losses / len(datasets[3]))
            print(f"Epoch [{epoch + 1}/{epochs[1]}] -GRU: Train Avg Losses: {gru_tra_total_lo[-1]:.4f} "
                  f"- Val Average Loss: {gru_val_total_lo[-1]:.4f}")
            # 储存损失
            save_infor_to_csv(False,
                              f"Epoch [{epoch + 1}/{epochs[1]}] -GRU: Train Average Losses: {gru_tra_total_lo[-1]:.4f}"
                              f" - Val Average Loss: {gru_val_total_lo[-1]:.4f}", 1)

            # 储存模型
            if gru_val_total_lo[-1] <= best_loss:
                best_loss = gru_val_total_lo[-1]
                torch.save(models[1],
                           os.path.join('D:\\myAImodel\\my_automatic_grading\\output_models_ls', "gru_model.pt"))

        # 绘制图像
        # plot_loss(gru_tra_total_lo, gru_val_total_lo, 'GRU train and validation losses')

    end_time = time.time()  # 结束时间
    # 计算执行时间
    execution_time_seconds = end_time - start_time
    hours = int(execution_time_seconds // 3600)
    minutes = int((execution_time_seconds % 3600) // 60)
    seconds = int(execution_time_seconds % 60)
    print(f"模型训练总共用时 {execution_time_seconds :.6f} 秒, 约{hours}时{minutes}分{seconds}秒")


# 整合模型的输出
def model_voting(models, test_data):
    for model in models:
        model.eval()  # 设置为评估模式

    mlp_dataloader = DataLoader(EntailmentDataset(test_data), batch_size=4,
                                collate_fn=collate_fn_entailment, drop_last=True)
    gru_dataloader = DataLoader(SimilarDataset(test_data), batch_size=4,
                                collate_fn=collate_fn_similar, drop_last=True)

    # 用模型计算结果
    mlp_predictions = []
    for i, [mlp_input_ids, mlp_attention_mask, mlp_token_type_ids, _] in enumerate(mlp_dataloader):
        mlp_input_ids, mlp_attention_mask, mlp_token_type_ids = \
            data_to_cuda(mlp_input_ids, mlp_attention_mask, mlp_token_type_ids)
        # 禁用梯度
        with torch.no_grad():
            mlp_output = models[0](mlp_input_ids, mlp_attention_mask, mlp_token_type_ids)
            mlp_predictions.append(mlp_output)
        torch.cuda.empty_cache()  # 清除缓存
    mlp_outputs_tensor = torch.cat(mlp_predictions, dim=0)

    gru_predictions = []
    for i, [gru_input_ids_student, gru_attention_mask_student, gru_input_ids_teacher, gru_attention_mask_teacher,
            _] in enumerate(gru_dataloader):
        gru_input_ids_student, gru_attention_mask_student, gru_input_ids_teacher, gru_attention_mask_teacher = \
            data_to_cuda(gru_input_ids_student, gru_attention_mask_student,
                         gru_input_ids_teacher, gru_attention_mask_teacher)
        # 禁用梯度
        with torch.no_grad():
            gru_output = models[1](gru_input_ids_student, gru_attention_mask_student,
                                   gru_input_ids_teacher, gru_attention_mask_teacher)
            gru_predictions.append(gru_output)
        torch.cuda.empty_cache()  # 清除缓存
    gru_outputs_tencer = torch.cat(gru_predictions)

    # 将三个模型的预测结果按列合并成一个数组
    stacked_preds = torch.stack([mlp_outputs_tensor, gru_outputs_tencer], dim=0)

    # 投票法计算结果
    fused_predictions, _ = torch.median(stacked_preds, dim=0)
    fused_predictions = map_to_specific_value(fused_predictions)

    return fused_predictions


# 输出的映射
def map_to_specific_value(input_tensor):
    # 定义映射规则
    conditions = [
        (input_tensor <= 0.25, torch.tensor(0.0)),
        ((input_tensor > 0.25) & (input_tensor <= 0.75), torch.tensor(0.5)),
        ((input_tensor > 0.75) & (input_tensor <= 1.25), torch.tensor(1.0)),
        ((input_tensor > 1.25) & (input_tensor <= 1.75), torch.tensor(1.5)),
        ((input_tensor > 1.75) & (input_tensor <= 2.25), torch.tensor(2.0)),
        ((input_tensor > 2.25) & (input_tensor <= 2.75), torch.tensor(2.5)),
        ((input_tensor > 2.75) & (input_tensor <= 3.25), torch.tensor(3.0)),
        ((input_tensor > 3.25) & (input_tensor <= 3.75), torch.tensor(3.5)),
        ((input_tensor > 3.75) & (input_tensor <= 4.25), torch.tensor(4.0)),
        ((input_tensor > 4.25) & (input_tensor <= 4.75), torch.tensor(4.5)),
        (input_tensor > 4.75, torch.tensor(5.0))
    ]

    # 使用torch.where函数将输入张量映射到特定的值
    output_tensor = torch.where(conditions[0][0], conditions[0][1], input_tensor)
    for i in range(1, len(conditions)):
        output_tensor = torch.where(conditions[i][0], conditions[i][1], output_tensor)

    return output_tensor


# 将数据放入CUDA
def data_to_cuda(*datas):
    return [data.to(device) for data in datas]


# 加载.pt模型
def load_pt_model(pt_model_name):
    model_path = 'D:\\myAImodel\\my_automatic_grading\\output_models_ls\\'
    return torch.load(model_path + pt_model_name)


# 绘制损失图像
def plot_loss(train_loss, val_loss, epochs_len, title):
    epoch = list(range(1, epochs_len + 1))
    plt.plot(epoch, train_loss, label='Training Loss')
    plt.plot(epoch, val_loss, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title(title)
    plt.show(block=False)


# 将损失储存
def save_list_to_csv(list1, list2, list1_name, list2_name, file_path="modal_loss.csv"):
    with open(file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([list1_name, list2_name])
        for item1, item2 in zip(list1, list2):
            writer.writerow([item1, item2])


# 将一些信息写入某个文件
def save_infor_to_csv(obliterate, data, index=0, file_path=loss_save_name):
    """
    用于储存数据，主要是储存损失
    :param obliterate: 是否清空文件内容，清空后会将data数据写到第一行
    :param data: 需要写入的数据
    :param index: 写入的位置，第几列
    :param file_path: 文件名称
    :return:
    """
    if obliterate:
        with open(file_path, 'w', encoding='utf-8', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(data)
    else:
        # 读取现有CSV文件内容
        existing_data = []
        try:
            with open(file_path, 'r', encoding='utf-8', newline='') as csv_file:
                reader = csv.reader(csv_file)
                existing_data = list(reader)
        except FileNotFoundError:
            pass

        # 扩展现有行数，以确保每一行有足够的列
        for row in existing_data:
            while len(row) <= index:
                row.append("")

        data_append = False
        # 在指定列的最后一行添加数据
        for i in range(len(existing_data)):
            if not existing_data[i][index]:
                existing_data[i][index] = data
                data_append = True
                break
        if not data_append:
            new_row = [""] * (index + 1)
            new_row[index] = data
            existing_data.append(new_row)

        # 将修改后的数据写回到文件
        with open(file_path, 'w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerows(existing_data)


# 模型融合后的结果测试
def model_test():
    if whether_model_train != [0, 0]:
        model_train()
    else:
        datasets = models_dataset_loader3()
        test_data = datasets[4]

        # 加载模型
        models = [load_pt_model('mlp_model.pt'), load_pt_model('gru_model.pt')]

        fused_predictions = model_voting(models, test_data)  # 融合
        test_labels = [row[2] for row in test_data]

        test_output = model_judge.classify_judge(fused_predictions, test_labels)
        accuracy = model_judge.accuracy_rate(fused_predictions, test_labels)
        test_output_zoom = model_judge.regression_judge(fused_predictions, test_labels)
        print("测试集模型评价数据[mse,rmse,mae,mape]:{}".format(test_output))
        print("测试集准确率:{}\n与人工评分相差0.5分：{}\n错误率为：{}".format(accuracy[0], accuracy[1], accuracy[3]))
        print(f"测试集模型缩放后评估数据：{len(test_output_zoom[-3])}\n"
              f"f1_weighted:{test_output_zoom[2]}\n"
              f"precision_per_class:{test_output_zoom[3]}\n"
              f"recall_per_class:{test_output_zoom[4]}\n"
              f"kappa:{test_output_zoom[5]}\n"
              f"confusion:{test_output_zoom[6]}")


def main():

    model_test()  # 使用训练后的三个模型


if __name__ == "__main__":
    main()
    print("*" * 50)
