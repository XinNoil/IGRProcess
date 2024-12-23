export subtype=none
export allsensor=0
export orisource=0
export filter_groundtruth=0
export filter_provider=0
# export path=IGR_cjy
# export path=IGR230307
# export path=IGR230312
# export path=IGR230415
# export path=IGR230419
# export path=IGR230425
# export path=IGR230426
# export path=IGR230429
# export path=IGR230503
# export subtype=people
# export path=IGR230510
# export path=IGR230626
# export path=IGR231231
# export path=IGR240111
export path=IGR241012
# todo: 0415, 0419, 0425, 0426

export debuglevel=1
export overwrite=1
export imu_overwrite=0
export sync=1

function pause(){
   read -p "$*"
}

echo 'python process.py -d' $path '-st' $subtype
echo 'python get_rinex.py -d' $path
echo 'python GNSSLogger_convert.py -d' $path
if [ $allsensor = 1 ]; then
    echo 'python AllSensorLogger_convert.py -d' $path
fi
# echo 'python get_nav.py' $path
# echo 'python run_doppler_all.py' $path
echo 'python train_data.py  -d' $path '-dl' $debuglevel '-s' $allsensor '-os' $orisource
# echo 'python export_data.py -d' $path '-e phone_err'
# echo 'python export_data.py -d' $path '-e phone_fix'
# echo 'python export_data.py -d' $path '-e imu_data'

# python process.py -d $path -st $subtype
# python get_rinex.py -d $path
# python GNSSLogger_convert.py -d $path
# if [ $allsensor = 1 ]; then
#     python AllSensorLogger_convert.py -d $path -o $overwrite -io $imu_overwrite
# fi
# pause 'run get_nav.py and run_doppler_all.py on Windows [Press Enter key to continue...]'
python train_data.py -d $path  -dl $debuglevel -s $allsensor -os $orisource -fg $filter_groundtruth -fp $filter_provider
# python export_data.py -d $path -e phone_err
# python export_data.py $path -e phone_fix
# python export_data.py $path -e imu_data

if [ $sync = 1 ]; then
    rsync -arKL --info=progress2 --include="*/" --include="*.h5" --exclude="*" IGRData/$path/processed/* IGRProcessed
    rsync -arKL --info=progress2 --include="*/" --include="*.yaml" --exclude="*" IGRData/$path/processed/* IGRProcessed
fi