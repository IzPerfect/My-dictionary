# Logging in Python
파이썬의 기본 내장 라이브러리인 logging을 사용 
<br>로깅이란 소프트웨어를 실행했을 때 발생하는 다양한 이벤트를 추적하는 수단
* 이벤트: 변수를 포함할 수 있는 설명 메시지
* 로깅 레벨: 이벤트에 부여되는 중요도를 의미

## 로깅 레벨
총 5가지로 분류 DEBUG-INFO-WARNING-ERROR-CRITICAL

| 레벨 | 중요도 level| 설명 |
|:--------:|:--------:|:--------:|
| DEBUG | 1 | 프로그램이 작동하는 지 진단할 때 사용, <br>INFO보다 상세한 정보 출력 |
| INFO | 2 | 프로그램이 예상대로 작동하는지 확인할 때 사용, <br>정상작동 시 이벤트 추적 |
| WARNING | 3 | 예상하지 못한 일이 발생했거나 가까운 미래에 발생할 문제가 있을 때 보고 |
| ERROR | 4 | 심각한 문제로 소프트웨어 일부 기능이 작동하지 못할 때,<br> 예외를 일으키지 않으면서 에러 보고 |
| CRITICAL | 5 | 심각한 에러로 프로그램 자체가 계속 실행될 수 없을 때, <br>예외를 일으키지 않으면서 에러를 보고 |

## 기본 사용 
logging 모듈을 사용하여 로그를 설정하고 기록
```bash
import logging as log

logger = log.getLogger()
# INO 레벨 이상의 로그 메시지만 기록(제일 낮은 DEBUG는 기록X)
logger.setLevel(log.INFO) 
# filename을 지정해 저장함 - append됨
log.basicConfig(filename="example.log", level=log.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S') 


logger.info("Sample info message")
```

다음과 같이 기록
```bash
2024-10-09 15:30:45 - INFO - Sample info message
```





## Reference
* [파이썬으로 logging하기](https://king-rabbit.github.io/python/python-logging/)
* [Logger 사용법](https://jh-bk.tistory.com/40)
