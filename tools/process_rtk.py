import os
from GNSSLogger_convert import convert_RTKLite_log

data_path = 'IGR230415'

def process_rtk_files():
    rtk_path = os.path.join('origin','rtklite')
    rtk_files = list(filter(lambda x: x.endswith('txt'), os.listdir(rtk_path)))

    for rtk_file in rtk_files:
        rtk_file_name = os.path.join(rtk_path, rtk_file)
        csv_name = os.path.join(rtk_path, rtk_file.replace('txt','csv'))
        print(rtk_file_name, csv_name)
        convert_RTKLite_log(rtk_file_name=rtk_file_name, csv_name=csv_name)

if __name__ == "__main__":
    os.chdir(data_path)
    process_rtk_files()