## Usage

### app.py
NOTE : Run docker command (Multi-model or Multi-model + Batching config) before running this
```
curl -H "Content-Type: application/json" -X POST -d '{"essay":"xyz"}' http://0.0.0.0:12000/API_NAME
```

### test_tfserving.py

#### Start Docker
```
sudo service docker start
```

#### Single model -
```
sudo docker run -p 8500:8500 -p 8501:8501 --mount type=\
bind,source=$(pwd)/models/normalised_rater2_domain1,\
target=/models/normalised_rater2_domain1/ -e MODEL_NAME=normalised_rater2_domain1 \
-t tensorflow/serving:latest-gpu
```

#### Multi-model -
```
docker run -t --rm --gpus all --name=tf-serving -p  8501:8501     \
-v "$(pwd)/models:/models/" tensorflow/serving:latest-gpu   \
     --model_config_file=/models/models.config
```

#### Multi-Model + Batching Config
```
docker run -t --rm --gpus all --name=tf-serving -p  8501:8501 \
-v "$(pwd)/models:/models/" tensorflow/serving:latest-gpu \
    --model_config_file=/models/models.config --enable_batching=true \
--batching_parameters_file=/models/batch_config/batching_parameters.txt
```
