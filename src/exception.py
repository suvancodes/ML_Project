import sys
import logging


def error_massage_del(error,error_del:sys):
    _,_,exc_tb = error_del.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename      # it is give the file name where error came
    error_massage = 'error came in python script name [{0}] line no [{1}] error massage [{2}]'.format(
        file_name,exc_tb.tb_lineno,str(error)
    )
    
    return error_massage

class CustomException(Exception):
    def __init__(self, error_massage,error_del):
        super().__init__(error_massage)
        self.error_massage=error_massage_del(error_massage,error_del = error_del)
        
    def __str__(self):
        return self.error_massage
     
     

if __name__=="__main__":
    try:
        a=1/0
    except Exception as e:
        logging.info('0 division error')
        raise CustomException(e,sys)