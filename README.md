
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

### 步骤
1. 数据解压：在Origin目录下存放各手机数据和rtklite数据
2. 配置文件：path_list.txt, devices.txt
3. 目录重构：执行`python process.py`
4. 计算多普勒速度（可选）：
    - python get_nav.py （windows）
    - python get_rinex.py
    - python run_doppler_all.py (windows)
5. 转换CSV文件：
    - python GNSSLogger_convert.py, python plot_trip.py
    - python AllSensorLogger_convert.py
6. 生成H5文件：python train_data.py
7. 数据统计：
    - python plots/gen_route_type_dict.py
    - GroundTruth 精度统计：1) python export_data.py phone_fix 2) plot_pos.ipynb
    - PhoneLabel 精度统计：1) python export_data.py phone_fix 2) plot_err.ipynb
    - IMU 画图：1) python export_data.py phone_fix 2) plot_imu.ipynb
8. 同步文件：
```
rsync -arKL --info=progress2 --include="*/" --include="*.h5" --exclude="*" IGR230415/processed/* IGRProcessed
sh data_sync.sh
```

注：下载广播星历单点定位等需要在windows下运行

Raw: utcTimeMillis, ChipsetElapsedRealtimeNanos
IMU: utcTimeMillis, elapsedRealtimeNanos
Fix: UnixTimeMillis, elapsedRealtimeNanos
GNGGA: utcTimeMillis

GNGGA: 
0: GNSS fix not available
1: GNSS fix valid
4: RTK fixed ambiquities    (mm)
5: RTK float ambiguities    (cm)