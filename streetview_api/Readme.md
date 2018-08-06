# streetview.py와 streetview.pyc는 google map으로부터 streetview panorama image를 다운로드받는 데 사용되는 모듈이므로 건드리지 않을 것.

1. google map 방문 히스토리가 저장된 csv 파일을 다운로드 받아 'raw_dataset_LOCATIONNAME.csv' 형식으로 이름 저장. (mozila firefox의 history master라는 app이 이를 지원)
2. processing_raw_dataset.py 실행. 3번째 줄에서 process할 raw csv 파일의 이름을 명시할 것.
3. download_imgs.py 실행. 이는 실제로 웹으로부터 streetview를 파노라마 이미지로서 다운로드받음. 55번째 줄에서 불러올 csv 파일의 이름을 반드시 명시할 것. 이로부터 다운로드된 이미지는 ./download_imgs_results/ 라는 경로에 저장됨.
4. make_spherical_image.py 실행. 이는 ./download_imgs_results/ 하위에 있는 파노라마 이미지를 CartesianToOmni.m 의 인자로서 전달하여 ./omni_imgs/ 하위에 omnidirectional image를 생성 및 저장.

