import argparse
import re

def get_api_info(path):
    pattern_fallback = '(?<=missing sdaa kernel: )\s*\w+'
    pattern_api = '(?<=Finish AD API: )\s*\w+'
    all_api_set = set()
    fallback_api_set = set()
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    for line in lines:
        if 'Finish AD API: ' in line:
            api = re.search(pattern_api, line).group()
            all_api_set.add(api)
        
        if 'missing sdaa kernel: ' in line:
            api = re.search(pattern_fallback, line).group()
            fallback_api_set.add(api)
    return sorted(list(all_api_set)), sorted(list(fallback_api_set))
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='算子覆盖率统计')
    parser.add_argument('--path', type=str, help='日志文件路径', required=True)
    args = parser.parse_args()
    all_api_set, fallback_api_set = get_api_info(args.path)
    print(f"all api: {all_api_set}, total: {len(all_api_set)}\n")
    print(f"fallback op: {fallback_api_set}, total: {len(fallback_api_set)}\n")
    print(f"coverage rate: {(1 - len(fallback_api_set) / len(all_api_set)) * 100:.2f}%")