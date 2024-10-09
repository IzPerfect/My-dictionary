'''
logging 라이브러리는 크게 4가지 클래스가 있음
* Logger: 로깅 관련 인터페이스 제공
* Handler: 이벤트 출력을 특정 파일, 콘솔, 네트워크 소켓 등 적절한 곳으로 보냄
* Filter: 더 심화된 로그 필터링
* Formatter: 로그 이벤트를 사람이 읽을 수 있는 형태로 변환
'''



import logging

# 로거 인스턴스 생성
# getLogger(__name__) 형식인데, 비워놓으면 root 로거가 생성
# 모듈 별로 관리해주기 위해 __name__ 설정하기도 함. 예를 들어) postprocessor.skeleton - postprocessor의 skeleton모듈에서의 로깅
logger_preprocess = logging.getLogger("preprocess")
logger_postprocess = logging.getLogger("postprocess")

# 포매터, 필터 설정
formatter = logging.Formatter("%(asctime)s %(levelname)s:%(message)s") # 핸들러마다 다른 형식으로 출력가능
filter = logging.Filter("postprocess") # postprocess만 처리하도록 필터링

# 핸들러 설정
handler = logging.StreamHandler()
handler.setFormatter(formatter)
handler.addFilter(filter)

# 핸들러 추가
logger_preprocess.addHandler(handler)
logger_postprocess.addHandler(handler)

logger_preprocess.critical("Critical from preprocess.")
logger_postprocess.critical("Critical from postprocess.")