# coding=utf-8
import json
import os.path
from llmtuner import ChatModel
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")


def main():
    chat_model = ChatModel()

    model_name_or_path = chat_model.model_args.model_name_or_path
    checkpoint_info = os.path.basename(model_name_or_path)
    model_info = os.path.basename(os.path.dirname(model_name_or_path))
    checkpoint = checkpoint_info.split('-')[1]
    output_folder = "./benchmark_results/{}-{}".format(model_info, checkpoint)
    if os.path.exists(output_folder):
        print("Error: target directory '{}' exists.".format(output_folder))
        exit(-1)
    os.makedirs(output_folder)

    benchmark_file_path = "./tmp_data/sql_benchmark_20230927_v2.json"
    with open(benchmark_file_path) as _file_:
        benchmark_data = json.load(_file_)

    for data_item in tqdm(benchmark_data, desc="Run benchmark"):
        # load the model input and extract the instruction
        model_input: str = data_item['model_input']
        model_input = model_input.split("### Instruction:\n")[1]
        instruction = model_input.split('### Response:')[0].strip()
        print(">> 输入的指令：")
        print(instruction)
        response, _ = chat_model.chat(instruction)
        response = response[0]
        print(">> 返回的输出：")
        print(response)

        # save the input
        output_file = os.path.join(output_folder, data_item['filename'])
        with open(output_file, 'w') as _file_:
            _file_.write(data_item['model_input'])

        # save the result
        output_file_name = data_item['filename'].split('_')[0]
        output_file = os.path.join(
            output_folder, "{}_response.md".format(output_file_name)
        )
        with open(output_file, 'w') as _file_:
            _file_.write(response)


if __name__ == '__main__':
    main()
