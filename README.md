
## Code Overview

### Directory Structure
```
IGRProcess
|  [data]
|  tools
```

### Description
* `process.py`          <p>获取数据集信息，信息写入info.yaml文件
* `get_nav.py`          <p>下载广播星历
* `get_rinex.py`        <p>将GNSS raw data转化为rinex格式， 生成的rinex文件统一命名为gnss_log.obs
* `run_pntpos.py`       <p>单次运行单点定位
* `run_pntpos_all.py`   <p>批量运行单点定位
* `run_doppler.py`      <p>单次运行多普勒测速
* `run_doppler_all.py`  <p>批量运行多普勒测速 
* `run_tdcp.py`         <p>单次运行TDCP
* `run_tdcp_all.py`     <p>批量运行TDCP 
* `calAllTDCPErr.py`    <p>批量计算TDCP误差

### Using our code
* `python get_*.py`     <p>下载广播星历 / 转化rinex文件，二者不互相依赖可同时进行
* `python run_*_all.py` <p>批量运行 单点定位 / 多普勒测速 / TDCP (需修改数据目录)
* `python get_*.py`         <p>下载广播星历 / 转化rinex文件，二者不互相依赖可同时进行
* `python run_*_all.py`     <p>批量运行 单点定位 / 多普勒测速 / TDCP (需修改数据目录)
* `python calAllTDCPErr.py` <p>批量计算TDCP误差, 结果保存在process路径下(tdcp_err.csv), 终端同时打印各trace的平均误差和TDCP覆盖率 

Raw: utcTimeMillis, ChipsetElapsedRealtimeNanos
IMU: utcTimeMillis, elapsedRealtimeNanos
Fix: UnixTimeMillis, elapsedRealtimeNanos
GNGGA: utcTimeMillis

GNGGA: 
0: GNSS fix not available
1: GNSS fix valid
4: RTK fixed ambiquities    (mm)
5: RTK float ambiguities    (cm)