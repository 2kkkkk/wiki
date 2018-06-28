---
title: "Mac os 安装lightgbm"
layout: page
date: 2018-06-27 00:00
---

[TOC]
Mac 系统真的麻烦，在Windows上一个pip install lightgbm就可以了，但在Mac上不行，百度了下，原因是：

    LightGBM depends on OpenMP for compiling, which isn't supported by Apple Clang.
    
    Please install gcc/g++ by using the following commands

**stackverflow真是个好网站！！！！！！！！！**

## 成功
心想用英文搜一下问题，看看老外是怎么解决的，果然找出了stackoverflow上的[解决方案](https://stackoverflow.com/questions/44937698/lightgbm-oserror-library-not-loaded)，一看就觉得靠谱，试了之后果然成功了，老外真厉害！

    Build LightGBM in Mac:
    brew install cmake  
    brew install gcc --without-multilib  
    git clone --recursive https://github.com/Microsoft/LightGBM ; cd LightGBM  
    mkdir build ; cd build  
    cmake ..   
    make -j  
    cd ../python-packages  
    sudo python setup.py install --precompile   #注意后面要加 --precompile ，不加会报错，原因待查
    As stated by @ecodan, you might need to force Mac to use GCC and G++ instead of the default compiler. So instead of building with cmake .., try:
    
    cmake -DCMAKE_C_COMPILER=/usr/local/Cellar/gcc/6.1.0/bin/gcc-6 -DCMAKE_CXX_COMPILER=/usr/local/Cellar/gcc/6.1.0/bin/g++-6 ..       #这句是关键，强行指定编译器，也就解决了软连接的问题
    #注意要修改gcc版本为自己的版本
关于make，cmake，可以理解为make需要使用Makefile，而cmake是产生Makefile的工具
## 失败
百度搜索到的方法基本都是按照github官网来的：

    brew install cmake
    brew install gcc
    git clone --recursive https://github.com/Microsoft/LightGBM ; cd LightGBM
    export CXX=g++-7 CC=gcc-7  # replace 7 with version of gcc installed on your machine
    mkdir build ; cd build
    cmake ..
    make -j4
提示 permission denied ，一般是因为权限问题，在命令前加sudo即可

cmake这步出错：`Apple clang is not supported` ,百度后，原因应该是Mac OS X 系统默认将 GCC软连接到Clang ，然后百度修改软连接的方法：

    $sudo vim ~/.bash_profile
    alias gcc='gcc-4.7'
    alias cc='gcc-4.7'
    alias g++='g++-4.7'
    alias c++='c++-4.7'
    source ~/.bash_profile
修改配置后然并卵。。。
