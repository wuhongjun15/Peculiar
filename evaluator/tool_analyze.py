import json
import re
from tqdm import tqdm
import os


def get_contract_name(contract):
    # e.g. \ncontract SafeMath {\n
    contract_name = re.findall("\\ncontract\ [\s\S]*?{", contract)
    try:
        contract_name = contract_name[0].split(" ")[1]
        return contract_name
    except IndexError:
        print("get contract name error!!! please check")
        return None


def read_test(path):
    url_to_label = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            url1, label = line.split('\t')
            url_to_label[url1] = label
    return url_to_label


def tool_result_analyze(tool, url_to_code, url_to_address, url_to_label, sol_map_contracts):
    address_to_tool = {}
    compare_result = []
    with open(sol_map_contracts) as g:
        for line in g:
            line = line.strip()
            js = json.loads(line)
            address_to_tool[js["address"]] = js["tools"]
    for url in tqdm(url_to_label.keys(), desc="analyzing %s" % tool):
        name = get_contract_name(url_to_code[url])
        address = url_to_address[url]
        test_flag = "-1"
        for contract in address_to_tool[address][tool]:
            if name == contract["name"]:
                test_flag = contract["flag"]
                break
        compare_result.append([url, test_flag])
    return compare_result


def create_tool_test_result(data_jsonl, sol_map_contracts_jsonl, test_path, evaluate_data_path):
    url_to_code = {}
    url_to_address = {}
    with open(data_jsonl) as f:
        for line in tqdm(f, desc="prepare to create tools' test txt"):
            line = line.strip()
            js = json.loads(line)
            url_to_code[js['idx']] = js['contract']
            url_to_address[js['idx']] = js['address']
    tools = ["mythril", "slither", "manticore", "osiris",
             "oyente", "securify", "smartcheck", "honeybadger"]
    url_to_label = read_test(test_path)

    if not os.path.exists(evaluate_data_path):
        os.makedirs(evaluate_data_path, exist_ok=True)
    for tool in tools:
        with open(os.path.join(evaluate_data_path, tool+"_test.txt"), "w") as t:
            ana_res = tool_result_analyze(
                tool, url_to_code, url_to_address, url_to_label, sol_map_contracts_jsonl)
            for line in tqdm(ana_res, desc="write %s_test.txt" % tool):
                for i in range(len(line)):
                    t.write(str(line[i]))
                    if i < len(line)-1:
                        t.write("\t")
                    else:
                        t.write("\n")


if __name__ == "__main__":
    create_tool_test_result(
        "dataset/data.jsonl", "dataset/sol_map_contracts.jsonl", "dataset/test.txt", "evaluator/tool_analysis_result")
