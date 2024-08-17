# Drone Swarm using Motion Capture - Bachelor's Degree Final Project

This repository contains all the content related to my Bachelor's Degree Final Project.


# Dependencies

Python 3.8  
OpenCV's Extra Modules  
Numpy   
[Pseyepy](https://github.com/bensondaled/pseyepy)  

## OpenCV's Extra Modules Installation
Step 1: Install required dependencies
```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev
sudo apt-get install -y python3.8-dev python3-numpy libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libdc1394-22-dev
```
[Structure from Motion Dependencies Installation](https://docs.opencv.org/4.x/db/db8/tutorial_sfm_installation.html)  

Step 2: Clone openCV and openCV contrib repositories
```bash
git clone https://github.com/opencv/opencv.git
git clone https://github.com/opencv/opencv_contrib.git
```

Step 3: Create a build directory
Create a build directory within the OpenCV directory. This is where the compilation will take place.
```bash
cd opencv
mkdir build
cd builds
```

Step 4: Run CMake

Configure the build with CMake. Point OPENCV_EXTRA_MODULES_PATH to the modules directory within the cloned opencv_contrib repository. Make sure to replace \<opencv_contrib\> with the actual path to your opencv_contrib directory.
```bash
cmake -D CMAKE_BUILD_TYPE=RELEASE \
      -D CMAKE_INSTALL_PREFIX=$(python -c "import sys; print(sys.prefix)") \
      -D PYTHON_EXECUTABLE=$(which python) \
      -D OPENCV_EXTRA_MODULES_PATH=<opencv_contrib>/modules \
      -D BUILD_EXAMPLES=ON ..
```

Step 5: Compile and Install

Compile OpenCV with the contrib modules. Adjust the -j flag according to the number of cores in your processor to speed up the compilation process.
```bash
make -j$(nproc)
```

After the compilation is complete, install OpenCV into your virtual environment.
```bash
make install
```

## To start PyQt Designer

```bash
pyqt6-tools designer
```