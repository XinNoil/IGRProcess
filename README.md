
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

### Using our code
* `python get_*.py`     <p>下载广播星历 / 转化rinex文件，二者不互相依赖可同时进行
* `python run_*_all.py` <p>批量运行 单点定位 / 多普勒测速 / TDCP (需修改数据目录)

