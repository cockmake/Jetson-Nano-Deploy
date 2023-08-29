#-------------------------------------------------
#
# Project created by QtCreator 2023-08-29T18:08:47
#
#-------------------------------------------------

QT += core gui
CONFIG += c++17
greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = p1-test
TEMPLATE = app

# The following define makes your compiler emit warnings if you use
# any feature of Qt which has been marked as deprecated (the exact warnings
# depend on your compiler). Please consult the documentation of the
# deprecated API in order to know how to port your code away from it.
DEFINES += QT_DEPRECATED_WARNINGS

# You can also make your code fail to compile if you use deprecated APIs.
# In order to do so, uncomment the following line.
# You can also select to disable deprecated APIs only up to a certain version of Qt.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0


SOURCES += \
        main.cpp \
        mainwindow.cpp

HEADERS += \
        mainwindow.h \
    yolov8-pose.hpp

FORMS += \
        mainwindow.ui


#TensorRT | Cudnn
INCLUDEPATH += \
    /usr/src/tensorrt/samples/common \
    /usr/include/aarch64-linux-gnu
LIBS += /usr/lib/aarch64-linux-gnu/libnv*.so

#Cuda
INCLUDEPATH += /usr/local/cuda-10.2/targets/aarch64-linux/include
LIBS += /usr/local/cuda-10.2/targets/aarch64-linux/lib/*.so

#OpenCV4.5.4
INCLUDEPATH += \
    /usr/include/opencv4/opencv2 \
    /usr/include/opencv4
LIBS += /usr/lib/aarch64-linux-gnu/libopencv*.so.4.5.4
