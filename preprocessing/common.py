import numpy as np
import pandas as pd
import re

'''
    모듈 import 방법 ex) 
    import sys
    file_path = "{경로}"
    sys.path.append(file_path)
    from module import *
'''
        


def extract_number(payload: str): 
    ''' 
    String 에서 숫자만 추출
        Args:
            payload(str): 숫자 포함 string
        Returns:
            float: float형 숫자 
    return 숫자형
    '''
    payload = str(payload)
    return float(re.sub(r'[^0-9]', '', payload))
    
    
def mapping_string_to_int(mapping_dict: dict, string_list: list):
    '''
    string을 int로 매핑
        Args:
            mapping_dict(dict): {k: v} 형태 k는 string, v는 매핑할 int
            string_list(list): String 형태의 리스트
        Returns:
            list: int형 리스트
    '''
    int_list = []
    for sl in string_list:
        if sl in mapping_dict.keys():
            int_list.append(mapping_dict[sl])
        else:
            int_list.append(None)
        
    return int_list
            
def pd_empty_fill(df, columns, fill_value):
    df_copy = df.copy()
    
    return df[columns].fillna(fill_value)


def print_ex(*inputs, func):
    print('Function Name: ' + func.__name__)
    for i, input in enumerate(inputs):
        print(f'input{[i]}: ' + str(input) + '\t')

    print('output: ' + str(func(*inputs)))
    print('')


    
if __name__ == '__main__':
    print('=== examples ===')
    print_ex('test123', func=extract_number)
    print_ex({'test':1, 'a': 2}, ['a', 'b', 'test'], func=mapping_string_to_int)
    
    
    

