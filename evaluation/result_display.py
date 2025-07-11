import os

def display_metric_results(metric_result_folder):
    # metric_result 폴더 내 모든 하위 폴더 탐색
    for root, dirs, _ in os.walk(metric_result_folder):
        for dir_name in dirs:
            folder_path = os.path.join(root, dir_name)
            result_file = os.path.join(folder_path, "result.txt")
            
            if os.path.exists(result_file):
                print(f"Results for {folder_path}:")
                with open(result_file, 'r', encoding='utf-8') as f:
                    print(f.read())
                print("-" * 50)  # 구분선 출력

# 실행
if __name__ == "__main__":
    metric_result_folder = "metric_result"  # 결과가 저장된 폴더 경로
    display_metric_results(metric_result_folder)