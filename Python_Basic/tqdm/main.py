import argparse

from tqdm import tqdm
import time
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser(description="tqdm types")
    parser.add_argument("--tqdm", type=str, default="basic", help="select type of tqdm")

    return parser.parse_args()

def basic(): 
    # 기본 실행
    for i in tqdm(range(100)):
        time.sleep(0.1)  # 작업을 흉내내기 위해 잠시 대기


def custom_pbar():
    # 커스텀 진행바 설정
    for i in tqdm(range(100), desc="Custom Pbar Processing"):
        time.sleep(0.1)

def custom_pbar_unit():
    # 단위 설정하는데 유용
    # 파일 다운로드 크기나 처리된 데이터 양 표시할 때 사용
    for i in tqdm(range(1000000), desc="Downloading", unit="B", unit_scale=True, unit_divisor=1024):
        time.sleep(0.0001)

def update_range():
    # 루프의 길이를 알 수 없을  때 수동으로 전체 작업 수(total) 지정
    total_tasks = 1000
    with tqdm(total=total_tasks, desc="Processing") as pbar:
        for i in range(100):
            time.sleep(0.1)
            pbar.update(10)  # 진행률을 수동으로 업데이트

def leave_pbar():
    for i in tqdm(range(100), leave=False, desc="Working"):
        time.sleep(0.05)

def colorbar():
    for i in tqdm(range(100), colour="blue", desc="Simulation"):
        time.sleep(0.05)

def tqdm_apply():
    # pandas에서 tqdm 사용
    tqdm.pandas()

    # DataFrame 생성
    df = pd.DataFrame({'numbers': range(1000)})

    # progress_apply로 적용 함수의 진행상황을 표시
    df['squared'] = df['numbers'].progress_apply(lambda x: x ** 2)

def nested():
    # 중첩된 루프에서 tqdm 사용
    for i in tqdm(range(3), desc="Outer Loop"):
        for j in tqdm(range(100), desc="Inner Loop", leave=False):
            time.sleep(0.01)

def manual():
    # 수동 업데이트 모드
    progress_bar = tqdm(total=100, desc="Manual Update")
    for i in range(10):
        time.sleep(0.5)
        progress_bar.update(10)  # 수동으로 10씩 업데이트
    progress_bar.close()


if __name__=='__main__':
    args = parse_args()

    if args.tqdm == 'basic':
        basic()
    elif args.tqdm == 'custom_pbar':
        custom_pbar()
    elif args.tqdm == 'custom_pbar_unit':
        custom_pbar()
    elif args.tqdm == 'update_range': # usefule
        update_range()
    elif args.tqdm == 'leave_pbar':
        leave_pbar()
    elif args.tqdm == 'colorbar': # useful
        colorbar()
    elif args.tqdm == 'tqdm_apply':
        tqdm_apply()
    elif args.tqdm == 'nested': # useful
        nested()
    elif args.tqdm == 'manual': # useful
        manual()
    
    