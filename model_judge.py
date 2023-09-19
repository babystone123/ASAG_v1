"""模型评估"""
import math
from sklearn.metrics import f1_score, mean_squared_error, mean_absolute_error,\
    precision_score, recall_score, cohen_kappa_score, confusion_matrix


def classify_judge(model_forecasts, true_labels):
    """
    用于计算模型评估的一些值
    :param model_forecasts: 模型预测值
    :param true_labels: 实际标签
    :return:
    """
    model_forecasts = model_forecasts.to('cpu').numpy()
    true_labels = true_labels[:len(model_forecasts)]

    mse = mean_squared_error(true_labels, model_forecasts)
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(true_labels, model_forecasts)

    model_forecasts_clear = [10 if x == 0 else x for x in model_forecasts]
    true_labels_clear = [10 if x == 0 else x for x in true_labels]
    ape = sum([abs(x - y) / x for x, y in zip(true_labels_clear, model_forecasts_clear)]) * 100
    mape = ape / len(model_forecasts)

    return mse, rmse, mae, mape


# 准确率
def accuracy_rate(model_forecasts, true_labels):
    model_forecasts = model_forecasts.to('cpu').numpy()
    accuracy = 0
    discrepancy1 = 0
    discrepancy2 = 0
    errors = 0
    for i in range(len(model_forecasts)):
        if abs(model_forecasts[i] - true_labels[i]) <= 0.25:
            accuracy += 1
        elif abs(model_forecasts[i] - true_labels[i]) <= 0.5:
            discrepancy1 += 1
        elif abs(model_forecasts[i] - true_labels[i]) <= 1:
            discrepancy2 += 1
        else:
            errors += 1

    return [accuracy / len(model_forecasts),
            discrepancy1 / len(model_forecasts),
            discrepancy2 / len(model_forecasts),
            errors / len(model_forecasts)]


# f1score，计算模型的准确率
def regression_judge(model_forecasts, true_labels):
    """
    计算模型f1分数，精准度&召回率，kappa系数，混淆矩阵
    :param model_forecasts: 模型预测列表
    :param true_labels:真实数据标签
    num_samples: 样本数量，验证集大小
    num_classes: 标签类别数量，0~5=11
    :return:

    返回的三个参数有不同含义
    F1-score (Macro):计算每个类别的 F1-score 并取平均得到的值。每个类别的 F1-score 被平等地对待，不考虑类别的样本数量。
        适用于在关注每个类别的平均表现时，特别是当类别之间的样本数量差异较大时。
    F1-score (Micro)：计算 F1-score 之前将所有类别的真正例、假正例和假负例的数量加总，然后计算整体的精确度和召回率，并基于这些整体值计算 F1-score。
        适用于在关注总体分类性能的同时，平衡了不同类别的权重。
    F1-score (Weighted):每个类别的 F1-score 进行加权平均，权重是类别在数据集中的样本数量占比。
        适用于不均衡数据集，更关注样本量较大的类别。
    """
    model_forecasts = model_forecasts.to('cpu')
    model_forecasts = [int(x * 2) for x in model_forecasts]
    true_labels = true_labels[:len(model_forecasts)]
    true_labels = [int(x * 2) for x in true_labels]

    f1_macro = f1_score(true_labels, model_forecasts, average='macro')
    f1_micro = f1_score(true_labels, model_forecasts, average='micro')
    f1_weighted = f1_score(true_labels, model_forecasts, average='weighted')

    precision_per_class = precision_score(true_labels, model_forecasts, average=None, zero_division=1)  # 计算精确率
    recall_per_class = recall_score(true_labels, model_forecasts, average=None, zero_division=1)  # 计算召回率（Recall）
    kappa = cohen_kappa_score(true_labels, model_forecasts, weights='quadratic')
    confusion = confusion_matrix(true_labels, model_forecasts)

    return f1_macro, f1_micro, f1_weighted, precision_per_class, recall_per_class, kappa, confusion


# 直接输出
def model_judge_output(fused_predictions, test_labels):
    test_output = classify_judge(fused_predictions, test_labels)
    accuracy = accuracy_rate(fused_predictions, test_labels)
    test_output_zoom = regression_judge(fused_predictions, test_labels)
    print("测试集模型评价数据[mse,rmse,mae,mape]:{}".format(test_output))
    print("测试集准确率:{}\n与人工评分相差0.5分：{}\n错误率为：{}".format(accuracy[0], accuracy[1], accuracy[3]))
    print(f"测试集模型缩放后评估数据：{len(test_output_zoom[-3])}\n"
          f"f1_weighted:{test_output_zoom[2]}\n"
          f"precision_per_class:{test_output_zoom[3]}\n"
          f"recall_per_class:{test_output_zoom[4]}\n"
          f"kappa:{test_output_zoom[5]}\n"
          f"confusion:{test_output_zoom[6]}")
