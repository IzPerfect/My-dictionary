import logging as log

logger = log.getLogger()
# INO 레벨 이상의 로그 메시지만 기록(제일 낮은 DEBUG는 기록X)
logger.setLevel(log.INFO) 
# filename을 지정해 저장함 - append됨
log.basicConfig(filename="example.log", level=log.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S') 


logger.info("Sample info message")