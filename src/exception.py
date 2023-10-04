import sys
from src.logger import logging ## importing the logger file from the source folder

## create function for error message details
def error_msg_details(error,error_details:sys):
    exc_type,exc_obj,exc_tb = error_details.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename

    error_msg =  "Error occured in pyhton script name[{0}] line number [{1}] error message [{2}]".format(file_name,exc_tb.tb_lineno,str(error))
    print(exc_type,file_name,exc_tb.tb_lineno)
    return error_msg

## creating the CustomException Class 
class CustomException(Exception):
    def __init__(self,error_msg,error_details:sys):
        super().__init__(error_msg)
        self.error_msg = error_msg_details(error_msg,error_details=error_details)

    def __str__(self):
        logging.error(self.error_msg)
        return self.error_msg
        
## 
if __name__ == "__main__":
    logging.info('Logging has Started')

    try:
        a=1/0
    except Exception as e:
        logging.error('Occures an error in ecxeption')
        raise CustomException(e,sys)