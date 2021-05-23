# Image classification reference training scripts
Tested on `pytorch:1.8.1`.

## docker
* python: 3.8
* pytorch: 1.8.1

```sh
docker pull flystarhe/torch:1.8.1-cuda10.2-dev
docker tag flystarhe/torch:1.8.1-cuda10.2-dev torch:1.8.1-cuda10.2-dev

docker save -o torch1.8.1-cuda10.2-dev.tar torch:1.8.1-cuda10.2-dev
docker load -i torch1.8.1-cuda10.2-dev.tar

n=torch18_cu10
t=torch:1.8.1-cuda10.2-dev
docker run --gpus all -d -p 7000:9000 --ipc=host --name ${n} -v "$(pwd)"/${n}:/workspace ${t}
```
