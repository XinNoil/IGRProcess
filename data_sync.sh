# rsync -arKL --info=progress2 --include="*/" --include="*.yaml" --exclude="*" IGRData/IGR230307/processed/* IGRProcessed
# rsync -arKL --info=progress2 --include="*/" --include="*.yaml" --exclude="*" IGRData/IGR230312/processed/* IGRProcessed
# rsync -arKL --info=progress2 --include="*/" --include="*.yaml" --exclude="*" IGRData/IGR230415/processed/* IGRProcessed
# rsync -arKL --info=progress2 --include="*/" --include="*.yaml" --exclude="*" IGRData/IGR230419/processed/* IGRProcessed
# rsync -arKL --info=progress2 --include="*/" --include="*.yaml" --exclude="*" IGRData/IGR230425/processed/* IGRProcessed
# rsync -arKL --info=progress2 --include="*/" --include="*.yaml" --exclude="*" IGRData/IGR230426/processed/* IGRProcessed
# rsync -arKL --info=progress2 --include="*/" --include="*.yaml" --exclude="*" IGRData/IGR_cjy/processed/* IGRProcessed

rsync -arKL --info=progress2 IGRProcessed/* tjubd5:/mnt/lun1/cjy/workspace/IMU-pr32/Data/IGR
rsync -arKL --info=progress2 IGRProcessed/* stone:/data1/wjk/IMU-pr32/Data/IGR