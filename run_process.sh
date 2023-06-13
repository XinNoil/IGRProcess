# export path=IGR230307
# export subtype=none
# export indoor=0
# export allsensor=0
# export orisource=0

# export path=IGR230312
# export subtype=none
# export indoor=0
# export allsensor=0
# export orisource=2

# export path=IGR_cjy
# export subtype=none
# export indoor=0
# export allsensor=0
# export orisource=0

# export path=IGR230503
# export subtype=people
# export indoor=0
# export allsensor=0
# export orisource=0

export path=IGR230510
export subtype=none
export indoor=0
export allsensor=1
export orisource=0

export debuglevel=1
export overwrite=1
export imu_overwrite=0
export sync=1

function pause(){
   read -p "$*"
}

if [ $indoor = 0 ]; then
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
    # # pause 'run get_nav.py and run_doppler_all.py on Windows [Press Enter key to continue...]'
    python train_data.py -d $path  -dl $debuglevel -s $allsensor -os $orisource
    # python export_data.py -d $path -e phone_err
    # python export_data.py $path -e phone_fix
    # python export_data.py $path -e imu_data
else
    echo 'python process.py -d' $path
    echo 'python AllSensorLogger_convert.py -d' $path
    echo 'python train_data.py -d' $path  '-dl' $debuglevel
    # echo 'python export_data.py -d' $path ' -e imu_data'

    python process.py -d $path
    python AllSensorLogger_convert.py -d $path
    python train_data.py $path -i 1 -d $debuglevel
    # python export_data.py $path -e imu_data
fi

if [ $sync = 1 ]; then
    rsync -arKL --info=progress2 --include="*/" --include="*.h5" --exclude="*" IGRData/$path/processed/* IGRProcessed
    rsync -arKL --info=progress2 --include="*/" --include="*.yaml" --exclude="*" IGRData/$path/processed/* IGRProcessed
fi