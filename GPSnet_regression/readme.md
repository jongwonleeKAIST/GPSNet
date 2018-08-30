1. ./streetview_api/에 raw_dataset_location.csv 파일 저장.
2. ./streetview_api/의 
processing_raw_dataset.py (>>> python processing_raw_dataset.py location), 
download_imgs.py (>>> python download_imgs.py location seqname), 
make_spherical_image.py (>>> python make_spherical_image.py seqname)
순으로 실행. 단, CartesianToOmni.m을 matlab에서 실행 중인 상태여야 함.
3. ./img_files/의 label_data.py 실행. (>>> python label_data.py seqname)
4. 다시 원래 디렉토리로 돌아와 test.py와 train.py를 차례로 실행. (실행하기 전 helper.py의 dataset 파일명을 수정할 것.)
