all:
	mkdir -pv build
	if [ ! -f build/inception-v3.tar.gz ]; then cd build && wget http://data.dmlc.ml/mxnet/models/imagenet/inception-v3.tar.gz && tar xf inception-v3.tar.gz; fi
	cd build && cmake .. && make
