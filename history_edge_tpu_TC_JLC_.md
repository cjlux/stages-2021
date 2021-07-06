## Creation de l'EVP (tf1):

conda create -n tf1 python=3.6
conda activate tf1
mkdir tod_tf1
cd tod_tf1/
pip install tensorflow-gpu==1.15
pip install absl nets
sudo apt-get install -y python python-tk
pip install Cython contextlib2 pillow lxml jupyter matplotlib

## Installation des models tensorflow et de google-coral
git clone https://github.com/tensorflow/models.git
cd models && git checkout f788046ca876a8820e05b0b48c1fc2e16b0955bc
cd ..
git clone https://github.com/google-coral/tutorials.git
cp -r tutorials/docker/object_detection/scripts/* models/research/

## Installer protoc:

wget https://www.github.com/google/protobuf/releases/download/v3.0.0/protoc-3.0.0-linux-x86_64.zip
unzip protoc-3.0.0-linux-x86_64.zip -d proto3
mkdir -p local/bin && mkdir -p local/include
mv proto3/bin/* local/bin
mv proto3/include/* local/include
rm -rf proto3 protoc-3.0.0-linux-x86_64.zip

## Installer cocoapi:

git clone --depth 1 https://github.com/cocodataset/cocoapi.git
(cd cocoapi/PythonAPI && make -j8)
cp -r cocoapi/PythonAPI/pycocotools/ models/research/
rm -rf cocoapi

## Vérifier obtject detection 

cd models/research/
../../local/bin/protoc object_detection/protos/*.proto --python_out=.
echo $PYTHONPATH
vim ~/.bashrc =>
	export TOD_ROOT="/home/jlc/tod_tf1"
	export PYTHONPATH=$TOD_ROOT/models:$TOD_ROOT/models/research:$TOD_ROOT/models/research/slim:$PYTHONPATH
python object_detection/builders/model_builder_test.py

## Entraîner le réseau

python object_detection/model_main.py   --pipeline_config_path="${CKPT_DIR}/pipeline.config"   --model_dir="${TRAIN_DIR}"   --num_train_steps="${num_training_steps}"   --num_eval_steps="${num_eval_steps}"
cd models/research
./retrain_detection_model.sh --num_training_steps 5000 --num_eval_steps 10

## Exporter le réseau du format .pb au formet .tflite:
export ckpt_number=5000
python object_detection/export_tflite_ssd_graph.py   --pipeline_config_path="${CKPT_DIR}/pipeline.config"   --trained_checkpoint_prefix="${TRAIN_DIR}/model.ckpt-${ckpt_number}"   --output_directory="${OUTPUT_DIR}"   --add_postprocessing_op=true
cd ~/
tflite_convert   --output_file="${OUTPUT_DIR}/output_tflite_graph.tflite"   --graph_def_file="${OUTPUT_DIR}/tflite_graph.pb"   --inference_type=QUANTIZED_UINT8   --input_arrays="${INPUT_TENSORS}"   --output_arrays="${OUTPUT_TENSORS}"   --mean_values=128   --std_dev_values=128   --input_shapes=1,300,300,3   --change_concat_input_ranges=false   --allow_nudging_weights_to_use_fast_gemm_kernel=true   --allow_custom_ops

## Ajout de edgetpu-compile

curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
sudo apt-get update
sudo apt-get install edgetpu-compiler

## Transformer le tflite en version compatible edgeTPU:

cd ~/tod_tf1/training/
cd form_detection/ssd_mobilenet_v1_quantized_300x300_coco14_sync_2018_07_18/tflite/
edgetpu_compiler output_tflite_graph.tflite

