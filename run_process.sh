export path=IGR230510
export indoor=0
export subtype=0
export debuglevel=1

function pause(){
   read -p "$*"
}

if [ $indoor = 0 ]; then
    echo 'python process.py ' $path '-i' $indoor '-s' $subtype
    echo 'python get_rinex.py' $path
    # echo 'python get_nav.py ' $path
    # echo 'python run_doppler_all.py' $path
    echo 'python GNSSLogger_convert.py' $path
    echo 'python train_data.py' $path  '-d' $debuglevel
    echo 'python export_data.py ' $path ' -e phone_err'
    echo 'python export_data.py ' $path ' -e phone_fix'
    echo 'python export_data.py ' $path ' -e imu_data'

    # python process.py $path '-i' $indoor '-s' $subtype
    # python get_rinex.py $path
    # python GNSSLogger_convert.py $path
    # # pause 'run get_nav.py and run_doppler_all.py on Windows [Press Enter key to continue...]'
    # python train_data.py $path  '-d' $debuglevel
    python export_data.py $path -e phone_err
    # python export_data.py $path -e phone_fix
    # python export_data.py $path -e imu_data
    # rsync -arKL --info=progress2 --include="*/" --include="*.h5" --exclude="*" IGRData/$path/processed/* IGRProcessed
    # rsync -arKL --info=progress2 --include="*/" --include="*.yaml" --exclude="*" IGRData/$path/processed/* IGRProcessed
else
    echo 'python process.py ' $path '-i' $indoor '-s' $subtype
    echo 'python AllSensorLogger_convert.py' $path
    echo 'python train_data.py' $path  '-d' $debuglevel
    echo 'python export_data.py ' $path ' -e imu_data'

    python process.py $path '-i' $indoor '-s' $subtype
    python AllSensorLogger_convert.py $path
    python train_data.py $path '-i' $indoor '-d' $debuglevel
    # python export_data.py $path -e imu_data
    rsync -arKL --info=progress2 --include="*/" --include="*.h5" --exclude="*" IGRData/$path/processed/* IGRProcessed
    rsync -arKL --info=progress2 --include="*/" --include="*.yaml" --exclude="*" IGRData/$path/processed/* IGRProcessed
fi