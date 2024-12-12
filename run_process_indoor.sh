# export path=IGR_indoor_241116_hsk
# export people=hushunkang
# export path=IGR_indoor_241116_ljl
# export people=lijialin
# export path=IGR_indoor_241116_lmy
# export people=limingyang
# export path=IGR_indoor_241116_zyp
# export people=zhangyupeng
export debuglevel=1
export sync=1

function pause(){
   read -p "$*"
}

# echo 'python config.py -d' $path ' -t Indoor -i True -st area -p ' $people
# echo 'python process.py -d' $path ' -st area'
# echo 'python AllSensorLogger_convert.py -d' $path
# echo 'python train_data.py -d' $path  ' -i 1 -dl' $debuglevel
# # echo 'python export_data.py -d' $path ' -e imu_data'

# python config.py -d $path -t Indoor -i True -st area -p $people
# python process.py -d $path -st area
# python AllSensorLogger_convert.py -d $path
# python train_data.py -d $path -i 1 -dl $debuglevel
# # python export_data.py $path -e imu_data

if [ $sync = 1 ]; then
    rsync -arKL --info=progress2 --include="*/" --include="*.h5" --exclude="*" IGRData/$path/processed/* IGRProcessed
    rsync -arKL --info=progress2 --include="*/" --include="*.yaml" --exclude="*" IGRData/$path/processed/* IGRProcessed
fi