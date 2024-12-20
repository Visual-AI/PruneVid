import os, json
import argparse
import requests

root_dir = 'test_results/pllava-7b-lora14-threshold0.8-layer10-alpha0.4-temporal-segment-ratio-0.25-cluster-ratio-0.5/egoschema'

def extract_and_convert(label_string):
    # 创建一个字典来映射字母到数字
    mapping = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}
    
    # 提取字符串中的第一个字符
    first_char = label_string[1]
    
    # 确保字符在映射范围内
    if first_char in mapping:
        return mapping[first_char]
    else:
        raise ValueError("Input string does not start with a valid label (A-E).")

def send_post_request(data):
    """
    Sends a POST request to the specified URL with the given JSON file.

    Parameters:
    - json_file (str): Path to the JSON file to be used in the request body.

    Returns:
    - Response object containing server's response.
    """

    url = "https://validation-server.onrender.com/api/upload/"
    headers = {
        "Content-Type": "application/json"
    }

    response = requests.post(url, headers=headers, json=data)
    
    return response

predition_jsonls = [f for f in os.listdir(root_dir) if 'all_results' in f]

result_dict = {}

for pred_jsonl in predition_jsonls:
    data_list = json.load(open(os.path.join(root_dir, pred_jsonl), 'r'))['result_list']
    for data in data_list:
        pred = data['pred']
        pred = extract_and_convert(pred)
        vid = data['video_path'].split('/')[-1].split('.')[0]
        result_dict[vid] = pred
    # with open(os.path.join(root_dir, pred_jsonl), 'r') as f:
    #     lines = f.readlines()
    #     for line in lines:
    #         data = json.loads(line)
    #         result_dict[data['vid']] = extract_and_convert(data['text']['prediction'])
print(result_dict)
response = send_post_request(result_dict)
print(f"Response Status Code: {response.status_code}")
print(f"Response Content:\n{response.text}")