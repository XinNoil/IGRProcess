export path=IGR230422
export indoor=1

function pause(){
   read -p "$*"
}

if [ $indoor = 0 ]; then
    echo 'python process.py ' $path '-i' $indoor
    echo 'python get_rinex.py' $path
    echo 'python get_nav.py ' $path
    echo 'python run_doppler_all.py' $path
    echo 'python GNSSLogger_convert.py' $path
    echo 'python train_data.py' $path
    echo 'python export_data.py ' $path ' -e phone_err'
    echo 'python export_data.py ' $path ' -e phone_fix'
    echo 'python export_data.py ' $path ' -e imu_data'

    python process.py $path '-i' $indoor
    python get_rinex.py $path
    python GNSSLogger_convert.py $path
    # pause 'run get_nav.py and run_doppler_all.py on Windows [Press Enter key to continue...]'
    python train_data.py $path
    python export_data.py $path -e phone_err
    python export_data.py $path -e phone_fix
    python export_data.py $path -e imu_data
    rsync -arKL --info=progress2 --include="*/" --include="*.h5" --exclude="*" IGRData/$path/processed/* IGRProcessed
else
    echo 'python AllSensorLogger_convert2.py' $path
    python AllSensorLogger_convert2.py $path
fi
