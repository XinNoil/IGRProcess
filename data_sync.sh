# rsync -arKL --info=progress2 --include="*/" --include="*.h5" --exclude="*" IGR/processed/* IGRProcessed
rsync -arKL --info=progress2 --include="*/" --include="*.h5" --exclude="*" IGR230307/processed/* IGRProcessed
rsync -arKL --info=progress2 --include="*/" --include="*.h5" --exclude="*" IGR230312/processed/* IGRProcessed
rsync -arKL --info=progress2 IGRProcessed/* tjubd5:/mnt/lun1/cjy/workspace/IMU-pr32/Data/IGR
rsync -arKL --info=progress2 IGRProcessed/* stone:/data1/wjk/IMU-pr32/Data/IGR