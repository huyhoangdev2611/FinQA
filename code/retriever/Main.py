#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Main script cho quá trình train model.
"""

import os
# Báo cho transformers không import TensorFlow để tránh xung đột với torch-xla.
os.environ["TRANSFORMERS_NO_TF"] = "1"

# Lọc cảnh báo FutureWarning từ traitlets (nếu bạn muốn ẩn các cảnh báo này)
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="traitlets")

from tqdm import tqdm
import json
from datetime import datetime
import time
import logging
import shutil  # Dùng để xóa thư mục checkpoint cũ

from utils import *
from config import parameters as conf
from torch import nn
import torch
import torch.optim as optim

from Model import Bert_model

# Khởi tạo tokenizer và cấu hình model dựa trên lựa chọn pretrained_model trong conf
if conf.pretrained_model == "bert":
    from transformers import BertTokenizer, BertConfig
    tokenizer = BertTokenizer.from_pretrained(conf.model_size)
    model_config = BertConfig.from_pretrained(conf.model_size)
elif conf.pretrained_model == "roberta":
    from transformers import RobertaTokenizer, RobertaConfig
    tokenizer = RobertaTokenizer.from_pretrained(conf.model_size)
    model_config = RobertaConfig.from_pretrained(conf.model_size)

# Tạo đường dẫn lưu model, kết quả và log
if conf.mode == "train":
    model_dir_name = conf.model_save_name + "_" + datetime.now().strftime("%Y%m%d%H%M%S")
    model_dir = os.path.join(conf.output_path, model_dir_name)
    results_path = os.path.join(model_dir, "results")
    saved_model_path = os.path.join(model_dir, "saved_model")
    os.makedirs(saved_model_path, exist_ok=False)
    os.makedirs(results_path, exist_ok=False)
    log_file = os.path.join(results_path, 'log.txt')
else:
    saved_model_path = os.path.join(conf.output_path, conf.saved_model_path)
    model_dir_name = datetime.now().strftime("%Y%m%d%H%M%S")
    model_dir = os.path.join(conf.output_path, 'inference_only_' + model_dir_name)
    results_path = os.path.join(model_dir, "results")
    os.makedirs(results_path, exist_ok=False)
    log_file = os.path.join(results_path, 'log.txt')

# Đọc danh sách các token và constant từ file
op_list = read_txt(conf.op_list_file, log_file)
op_list = [op + '(' for op in op_list]
op_list = ['EOF', 'UNK', 'GO', ')'] + op_list
const_list = read_txt(conf.const_list_file, log_file)
const_list = [const.lower().replace('.', '_') for const in const_list]
reserved_token_size = len(op_list) + len(const_list)

# Đọc data từ file train, valid, test
train_data, train_examples, op_list, const_list = read_examples(
    input_path=conf.train_file, tokenizer=tokenizer, op_list=op_list, const_list=const_list, log_file=log_file)

valid_data, valid_examples, op_list, const_list = read_examples(
    input_path=conf.valid_file, tokenizer=tokenizer, op_list=op_list, const_list=const_list, log_file=log_file)

test_data, test_examples, op_list, const_list = read_examples(
    input_path=conf.test_file, tokenizer=tokenizer, op_list=op_list, const_list=const_list, log_file=log_file)

kwargs = {
    "examples": train_examples,
    "tokenizer": tokenizer,
    "option": conf.option,
    "is_training": True,
    "max_seq_length": conf.max_seq_length,
}

train_features = convert_examples_to_features(**kwargs)
kwargs["examples"] = valid_examples
kwargs["is_training"] = False
valid_features = convert_examples_to_features(**kwargs)
kwargs["examples"] = test_examples
test_features = convert_examples_to_features(**kwargs)


def cleanup_checkpoints():
    """
    Xóa các checkpoint cũ, chỉ giữ lại 3 checkpoint mới nhất trong thư mục saved_model_path/loads.
    """
    checkpoint_dir = os.path.join(saved_model_path, 'loads')
    if not os.path.exists(checkpoint_dir):
        return
    # Lấy danh sách các thư mục checkpoint (giả sử tên thư mục là số)
    checkpoint_list = [d for d in os.listdir(checkpoint_dir)
                       if os.path.isdir(os.path.join(checkpoint_dir, d))]
    if len(checkpoint_list) > 3:
        # Sắp xếp theo thứ tự số (đảm bảo tên thư mục có thể ép kiểu thành int)
        checkpoint_list = sorted(checkpoint_list, key=lambda x: int(x))
        # Xóa các checkpoint cũ nhất
        for folder in checkpoint_list[:-3]:
            full_path = os.path.join(checkpoint_dir, folder)
            shutil.rmtree(full_path)
            write_log(log_file, "Đã xóa checkpoint cũ: " + full_path)


def train():
    """
    Hàm train chính, theo dõi loss, lưu checkpoint định kỳ và đánh giá trên tập validation.
    Hỗ trợ tiếp tục training từ checkpoint nếu conf.resume_model_path không rỗng.
    """
    # Đọc file train.json và tính số lượng phần tử (num_examples)
    with open(conf.train_file, 'r') as f:
        data = json.load(f)
    num_examples = len(data)
    print(f"Số lượng phần tử trong train.json: {num_examples}")

    # Tính số bước mỗi epoch (steps/epoch)
    steps_per_epoch = (num_examples + conf.batch_size - 1) // conf.batch_size  # Lấy ceiling của phép chia
    print(f"Số bước mỗi epoch: {steps_per_epoch}")

    # Tính max_steps cho 16 epoch
    max_steps = steps_per_epoch * conf.epoch
    print(f"Số bước tối đa (max_steps) cho {conf.epoch} epoch: {max_steps}")

    write_log(log_file, "####################INPUT PARAMETERS###################")
    for attr in conf.__dict__:
        value = conf.__dict__[attr]
        write_log(log_file, f"{attr} = {value}")
    write_log(log_file, "#######################################################")

    model = Bert_model(hidden_size=model_config.hidden_size,
                       dropout_rate=conf.dropout_rate)
    model = nn.DataParallel(model)
    model.to(conf.device)

    optimizer = optim.Adam(model.parameters(), conf.learning_rate)
    criterion = nn.CrossEntropyLoss(reduction='none', ignore_index=-1)
    model.train()

    k = 0  # Số bước training toàn cục (global_step)
    if conf.resume_model_path != "":
        print("Tiếp tục training từ checkpoint:", conf.resume_model_path)
        write_log(log_file, "Tiếp tục training từ checkpoint: " + conf.resume_model_path)
        checkpoint = torch.load(conf.resume_model_path, map_location=torch.device(conf.device))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        k = checkpoint.get('global_step', 0)
        write_log(log_file, f"Đã load global_step = {k}")

    train_iterator = DataLoader(is_training=True, data=train_features,
                                batch_size=conf.batch_size, shuffle=True)
    record_k = 0
    record_loss = 0.0
    start_time = time.time()

    for _ in range(conf.epoch):
        train_iterator.reset()
        for x in train_iterator:
            # Chuyển đổi dữ liệu về tensor và đưa vào device
            input_ids = torch.tensor(x['input_ids']).to(conf.device)
            input_mask = torch.tensor(x['input_mask']).to(conf.device)
            segment_ids = torch.tensor(x['segment_ids']).to(conf.device)
            label = torch.tensor(x['label']).to(conf.device)

            model.zero_grad()
            optimizer.zero_grad()

            this_logits = model(True, input_ids, input_mask, segment_ids, device=conf.device)
            this_loss = criterion(this_logits.view(-1, this_logits.shape[-1]), label.view(-1))
            this_loss = this_loss.sum()

            record_loss += this_loss.item() * 100
            record_k += 1
            k += 1

            this_loss.backward()
            optimizer.step()

            if k > 1 and k % conf.report_loss == 0:
                avg_loss = record_loss / record_k
                write_log(log_file, f"{k} : loss = {avg_loss:.3f}")
                record_loss = 0.0
                record_k = 0

            if k > 1 and k % conf.report == 0:
                print("Round:", k / conf.report)
                model.eval()
                cost_time = time.time() - start_time
                write_log(log_file, f"{k // conf.report} : time = {cost_time:.3f}")
                start_time = time.time()

                if k // conf.report >= 1:
                    print("Val test")
                    # Lưu model checkpoint với thông tin optimizer và global step
                    saved_model_path_cnt = os.path.join(saved_model_path, 'loads', str(k // conf.report))
                    os.makedirs(saved_model_path_cnt, exist_ok=True)
                    checkpoint = {
                        'global_step': k,  # Lưu số bước training đã thực hiện
                        'optimizer_state_dict': optimizer.state_dict(),  # Lưu trạng thái optimizer
                        'model_state_dict': model.state_dict()  # Lưu trạng thái của model
                    }
                    torch.save(checkpoint, os.path.join(saved_model_path_cnt, "model.pt"))

                    # Dọn dẹp các checkpoint cũ, chỉ giữ lại 3 mới nhất
                    cleanup_checkpoints()

                    # Đánh giá trên tập validation và lưu kết quả
                    results_path_cnt = os.path.join(results_path, 'loads', str(k // conf.report))
                    os.makedirs(results_path_cnt, exist_ok=True)
                    evaluate(valid_examples, valid_features, model, results_path_cnt, mode='valid')

                model.train()

            # Điều kiện dừng nếu đạt số bước tối đa
            if k >= max_steps:
                print("Dừng huấn luyện sau khi train xong")
                write_log(log_file, "Dừng huấn luyện sau khi đạt số bước tối đa")
                break
        # Kiểm tra lại điều kiện dừng sau mỗi epoch
        if k >= max_steps:
            break


def evaluate(data_ori, data, model, ksave_dir, mode='valid'):
    """
    Đánh giá model trên tập dữ liệu đã cho và lưu kết quả dự đoán.
    """
    ksave_dir_mode = os.path.join(ksave_dir, mode)
    os.makedirs(ksave_dir_mode, exist_ok=True)

    data_iterator = DataLoader(is_training=False, data=data,
                               batch_size=conf.batch_size_test, shuffle=False)
    all_logits = []
    all_filename_id = []
    all_ind = []

    with torch.no_grad():
        for x in tqdm(data_iterator):
            input_ids = x['input_ids']
            input_mask = x['input_mask']
            segment_ids = x['segment_ids']
            filename_id = x["filename_id"]
            ind = x["ind"]

            ori_len = len(input_ids)
            for each_item in [input_ids, input_mask, segment_ids]:
                if ori_len < conf.batch_size_test:
                    each_len = len(each_item[0])
                    pad_x = [0] * each_len
                    each_item += [pad_x] * (conf.batch_size_test - ori_len)

            input_ids = torch.tensor(input_ids).to(conf.device)
            input_mask = torch.tensor(input_mask).to(conf.device)
            segment_ids = torch.tensor(segment_ids).to(conf.device)

            logits = model(True, input_ids, input_mask, segment_ids, device=conf.device)
            all_logits.extend(logits.tolist())
            all_filename_id.extend(filename_id)
            all_ind.extend(ind)

    output_prediction_file = os.path.join(ksave_dir_mode, "predictions.json")

    if mode == "valid":
        print_res = retrieve_evaluate(all_logits, all_filename_id, all_ind,
                                      output_prediction_file, conf.valid_file, topn=conf.topn)
    else:
        print_res = retrieve_evaluate(all_logits, all_filename_id, all_ind,
                                      output_prediction_file, conf.test_file, topn=conf.topn)

    write_log(log_file, print_res)
    print(print_res)
    return


if __name__ == '__main__':
    train()
