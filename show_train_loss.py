# coding=utf-8
import json
import os.path
import matplotlib.pyplot as plt


def show_train_loss(loss_file: str):
    with open(loss_file) as _file_:
        loss_data_list = []
        for line in _file_:
            loss_data = json.loads(line)
            loss_data_list.append(loss_data)
    loss_value_list = [item['loss'] for item in loss_data_list]
    epoch_value_list = [item['epoch'] for item in loss_data_list]
    step_value_list = [item['current_steps'] for item in loss_data_list]
    plt.plot(epoch_value_list, loss_value_list)
    plt.xlabel("Epoch")
    plt.ylabel('Train Loss')

    for epoch_value, loss_value, step_value in zip(epoch_value_list, loss_value_list, step_value_list):
        if step_value % 1000 == 0:
            plt.text(epoch_value, loss_value + 0.03, str(step_value), ha='center', va="bottom")
    plt.show()


if __name__ == '__main__':
    train_folder_path = "./tmp_data/1031_trainer_log.jsonl"
    show_train_loss(train_folder_path)
