# Multi-class-segmentation
Multiclass segmentation 

Reference work: https://github.com/xiaoyufenfei/Efficient-Segmentation-Networks#Dataset-Setting \
@misc{Efficient-Segmentation-Networks,
  author = {Yu Wang},
  title = {Efficient-Segmentation-Networks Pytorch Implementation},
  year = {2019},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/xiaoyufenfei/Efficient-Segmentation-Networks}},
  commit = {master}
}

## Prerequisite
cuda 11.7\
libtorch 11.7\
pytorch 2.0.1\
boost 1.80.0 \
opencv 4.8.0

## Links
cuda 11.7 --> https://forums.developer.nvidia.com/t/installing-nvidia-cuda-toolkit/265957 \
libtorch 11.7 --> wget https://download.pytorch.org/libtorch/cu117/libtorch-cxx11-abi-shared-with-deps-2.0.1%2Bcu117.zip \
pytorch 2.0.1 --> https://pytorch.org/get-started/locally/ \
boost 1.80.0 --> https://www.boost.org/users/history/version_1_80_0.html \
*** Once boost is install, set target link library for boost in cmake text file \
opencv 4.8.0 --> https://www.geeksforgeeks.org/how-to-install-opencv-in-c-on-linux/ \

## To compile libtorch

Link: https://pytorch.org/tutorials/advanced/cpp_export.html \
Change nvcc version --> https://askubuntu.com/questions/885610/nvcc-version-command-says-nvcc-is-not-installed \
If there is compilation error after changing nvcc --> https://github.com/NVlabs/instant-ngp/issues/747 \
***Make sure cuda compiler is up-to-date 


## To run python file
python3 train.py

## To test the model
python3 test.py

## To test the model in cpp
Go into the folder that has the same name as your cpp file \
mkdir build \
cd build \
cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch .. \
cmake --build . --config Release \



