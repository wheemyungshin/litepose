conda create --name litepose python=3.8 -y
conda activate litepose
conda install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cudatoolkit=10.1 -c pytorch
../miniconda3/envs/litepose/bin/pip install -r requirements.txt
git clone https://github.com/cocodataset/cocoapi.git ../cocoapi
cd ../cocoapi/PythonAPI
make install
cd ../..

git clone https://github.com/Jeff-sjtu/CrowdPose.git crowdposeapi
cd crowdposeapi/crowdpose-api/PythonAPI
make install
cd ../../..
