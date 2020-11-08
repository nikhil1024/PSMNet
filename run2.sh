python finetune.py --maxdisp 192 \
                   --model stackhourglass \
                   --datatype 2015 \
                   --datapath ../data_scene_flow_2015/training/ \
                   --epochs 1000 \
                   --loadmodel ./trained_models/pretrained_sceneflow.tar \
                   --savemodel ./trained_models2/
