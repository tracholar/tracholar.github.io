---
layout: post
title: "Python ����ʱ�����ģ���ܽ�"
description: ""
category: "techology"
tags: ["python","datetime"]
---
##���

Python��׼���ṩ�˶����ڡ�ʱ�䡢�������в�����ģ��`time, datetime, calendar`��
����`time`ģ��������Ƕ�Unixʱ����Ĳ����ʹ����Լ��漰������ϵͳ
��صĲ�����`datetime`ģ�����Ƕ����ں�ʱ����еĴ����װ��֧��ʱ��
֮������㣬�ڶ����ں�ʱ��Ĵ����ϱ�`time`ģ��Ҫ���㡣


## time ģ��
`time`ģ����python��׼���У���������ͨ�ò���ϵͳ����Ŀ¼�£�
�ɴ˿ɼ����ģ�������ϵͳ���źܴ�Ĺ�ϵ��
��Ϊ���ԭ��ĳЩ�����������ϵͳƽ̨�йصġ�
�����ǻ���Unixʱ���������ʱ���ʾ�ķ�Χ
���޶���1970-2038��֮�䡣
���ģ���еĻ������ݽ����`struct_time`��ʵ������һ�������ֵ�Ԫ�顣

���ģ���ṩ��ʱ�����������Ҫ��ʱ�����ʱ���ַ�����`struct_time`���������е��໥ת����
����һЩ����ϵͳ��ʱ���йص�ϵͳ���á�

### �����÷�

����Unixʱ�������λ����

`
>>> ts = time.time()
>>> ts
1445495655.495
`

ʱ�����һ��������������ͨ�����ú���ת��Ϊ������ʽ��

ʱ���ת��Ϊ`struct_time`

`
>>> time.gmtime(ts)
time.struct_time(tm_year=2015, tm_mon=10, tm_mday=22, tm_hour=6, tm_min=34, tm_sec=15, tm_wday=3, tm_yday=295, tm_isdst=0)

>>> time.localtime(ts)
time.struct_time(tm_year=2015, tm_mon=10, tm_mday=22, tm_hour=14, tm_min=34, tm_sec=15, tm_wday=3, tm_yday=295, tm_isdst=0)
```

ʱ���ת��Ϊ�����Ķ����ַ���   

`
>>> time.ctime(ts)
'Thu Oct 22 14:34:15 2015'
`

`struct_time`ת��Ϊ�ַ���   
`
>>> st = time.localtime(ts)
>>> time.asctime(st)
'Thu Oct 22 14:34:15 2015'

>>> time.strftime('%Y-%m-%d', st)
'2015-10-22'
`

`struct_time`ת��Ϊʱ���  
`
>>> time.mktime(st)
1445495655.0
`

ʱ���ַ���ת`struct_time`    
```
>>> time.strptime('2015-10-22','%Y-%m-%d')
time.struct_time(tm_year=2015, tm_mon=10, tm_mday=22, tm_hour=0, tm_min=0, tm_sec=0, tm_wday=3, tm_yday=295, tm_isdst=-1)
```

��ʱ���йص�ϵͳ���ã���`time.sleep`��



## datetime ģ��
datetimeģ���ṩ��ʱ������ڵķ�װ�����ṩ����֮�����ѧ���㡣
��ģ�����4���࣬�õö����`datetime.timedelta`��`datetime.datetime`��    
```
object
    timedelta     # ��Ҫ���ڼ���ʱ����
    tzinfo        # ʱ�����
    time          # ֻ��עʱ��
    date          # ֻ��ע����
        datetime  # ͬʱ��ʱ�������
```

dateֻ����������3�����ԣ�datetime������ʱ���֡��롢���롣

��ȡ����ʱ��   
```
>>> datetime.today()
datetime.datetime(2015, 10, 22, 15, 8, 54, 88000)
>>> datetime.now()
datetime.datetime(2015, 10, 22, 15, 8, 41, 304000)
```

��ʱ�����ת��    
```
>>> datetime.fromtimestamp(ts)
datetime.datetime(2015, 10, 22, 14, 34, 15, 495000)
```

����`datetime.combine(date,time)`���Խ�date��time���Ϊdatetime��
����`datetime.strptime`��`datetime.strftime`
������ʱ���ַ�����datetime������໥ת����   
```
>>> dt = datetime.strptime('2015-10-12','%Y-%m-%d')
>>> dt
datetime.datetime(2015, 10, 12, 0, 0)
>>> dt.strftime('%d/%m/%Y')
'12/10/2015'
```

��datetime�����������޸�    
```
>>> dt.replace(year=2016)
datetime.datetime(2016, 10, 12, 0, 0)
```

ת��ΪtimetupleҲ����time.struct_time   
```
>>> dt.timetuple
time.struct_time(tm_year=2015, tm_mon=10, tm_mday=12, tm_hour=0, tm_min=0, tm_sec=0, tm_wday=0, tm_yday=285, tm_isdst=-1)
```

`datetime.timedelta`�����`datetime.time`��������������Ƶģ�ֻ����ǰ����ʱ��
`total_seconds`���������ܵ�������

`timedelta`��`datetime`֮�����ѧ������Թ���Ϊ    
```
timedelta = datetime - datetime
datetime = datetime + timedelta
```
����   
```
>>> delta = timedelta(hours=1)
>>> delta
datetime.timedelta(0, 3600)
>>> 
>>> dt + delta
datetime.datetime(2015, 10, 12, 1, 0)
```

## calendar ģ��
���ģ����Ҫ�ṩ������һЩ���������Ժܷ�������һ���ı�������    
```
>>> print calendar.month(2015,10)

    October 2015
Mo Tu We Th Fr Sa Su
          1  2  3  4
 5  6  7  8  9 10 11
12 13 14 15 16 17 18
19 20 21 22 23 24 25
26 27 28 29 30 31
```

calendar����һ��`Calendar`�����������������ݵĽṹ��һЩ���ò�����
����ʽ�����񽻸�����������`TextCalendar`��`HTMLCalendar`��


## �ܽ�
ʱ��ģ��ıȽϣ�
�����ʱ����������ַ�����ʱ�����ת���Ļ�����timeģ�顣
�����Ҫ��ʱ����бȽϸ��ӵ���ѧ����Ļ�����datetimeģ�顣

