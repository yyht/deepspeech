import re

end_time_pattern = re.compile(u'(?<=(endTime\\\\"\:))(.+)')
start_time_pattern = re.compile(u'(?<=(startTime\\\\"\:))(.+)')
text_time_pattern = re.compile(u'(?<=(text\\\\"\:\\\\"))(.+)(?=(\\\\"}))')

import csv
import numpy as np
output_list = []
with open('/data/albert/asr/gaode_asr/fanbinshixiaogou_20210518.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count >= 1:
           
            ordered_id = row[0]
            region = row[1]
            date = row[2]
            url = row[-1]
            
            t_dict = {
                "ordered_id":ordered_id,
                 "region":region,
                 "date":date,
                 "url":url,
                 "transcript":[]
            }
            
            transcript = row[3:-5]
            
            end_time_flag = -1
            start_time_flag = -1
            text_flag = -1
            
            tmp_dict = {
                        "start":0,
                        "end":0,
                        "transcript":""
                    }
            
            for index, item in enumerate(transcript):
                end_time = end_time_pattern.search(item)
                if end_time:
                    tmp_dict['end'] = end_time.group()
                    end_time_flag += 1
                start_time = start_time_pattern.search(item)
                if start_time:
                    tmp_dict['start'] = start_time.group()
                    start_time_flag += 1
                text = text_time_pattern.search(item)
                if text:
                    tmp_dict['transcript'] = text.group()
                    text_flag += 1
                if end_time_flag == 0 and start_time_flag == 0 and text_flag == 0:
                    t_dict['transcript'].append(tmp_dict)
                    tmp_dict = {
                        "start":0,
                        "end":0,
                        "transcript":""
                    }
                    end_time_flag = -1
                    start_time_flag = -1
                    text_flag = -1
            output_list.append(t_dict)
        line_count += 1
        
        
fwobj = open("/data/albert/asr/gaode_asr/fanbinshixiaogou_20210518.json", "w")
for item in output_list:
    fwobj.write(json.dumps(item, ensure_ascii=False)+"\n")
fwobj.close()

fwobj = open("/data/albert/asr/gaode_asr/fanbinshixiaogou_20210518_url.json", "w")
for item in output_list:
    fwobj.write(item['ordered_id']+"\n")
    fwobj.write(item['url']+"\n")
fwobj.close()

import os
data_path = "/data/albert/asr/gaode_asr/fanbinshixiaogou_20210518"
speech_command = 'wget "{}" -O {}'
with open("/data/albert/asr/gaode_asr/fanbinshixiaogou_20210518.sh", "w") as fwobj:
    for item in output_list:
        url = item['url']
        order_id = item['ordered_id']

        my_command = speech_command.format(url, os.path.join(data_path, order_id+".mp3"))
        fwobj.write(my_command+"\n")

import csv
import numpy as np
output_list = []
with open('/data/albert/asr/gaode_asr/10000_20210519.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count >= 1:
           
            ordered_id = row[0]
            region = row[1]
            date = row[2]
            url = row[-1]
            
            t_dict = {
                "ordered_id":ordered_id,
                 "region":region,
                 "date":date,
                 "url":url,
                 "transcript":[]
            }
            
            transcript = row[3:-5]
            
            end_time_flag = -1
            start_time_flag = -1
            text_flag = -1
            
            tmp_dict = {
                        "start":0,
                        "end":0,
                        "transcript":""
                    }
            
            for index, item in enumerate(transcript):
                end_time = end_time_pattern.search(item)
                if end_time:
                    tmp_dict['end'] = end_time.group()
                    end_time_flag += 1
                start_time = start_time_pattern.search(item)
                if start_time:
                    tmp_dict['start'] = start_time.group()
                    start_time_flag += 1
                text = text_time_pattern.search(item)
                if text:
                    tmp_dict['transcript'] = text.group()
                    text_flag += 1
                if end_time_flag == 0 and start_time_flag == 0 and text_flag == 0:
                    t_dict['transcript'].append(tmp_dict)
                    tmp_dict = {
                        "start":0,
                        "end":0,
                        "transcript":""
                    }
                    end_time_flag = -1
                    start_time_flag = -1
                    text_flag = -1
            output_list.append(t_dict)
        line_count += 1
        
        
fwobj = open("/data/albert/asr/gaode_asr/10000_20210519.json", "w")
for item in output_list:
    fwobj.write(json.dumps(item, ensure_ascii=False)+"\n")
fwobj.close()


fwobj = open("/data/albert/asr/gaode_asr/10000_20210519_url.json", "w")
for item in output_list:
    fwobj.write(item['ordered_id']+"\n")
    fwobj.write(item['url']+"\n")
fwobj.close()


import os
data_path = "/data/albert/asr/gaode_asr/10000_20210519"
speech_command = 'wget "{}" -O {}'
with open("/data/albert/asr/gaode_asr/10000_20210519.sh", "w") as fwobj:
    for item in output_list:
        url = item['url']
        order_id = item['ordered_id']

        my_command = speech_command.format(url, os.path.join(data_path, order_id+".mp3"))
        fwobj.write(my_command+"\n")