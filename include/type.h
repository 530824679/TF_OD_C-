//
// Created by chenwei on 20-12-28.
//

#ifndef TF_OD_C_TYPE_H
#define TF_OD_C_TYPE_H

enum Classes
{
    CAR = 0,
    BUS = 1,
    TRUCK = 2,
    PEDESTRIAN = 4,
    UNKNOWN = 5,
};

struct BBox
{
    // center point
    float x;
    float y;
    float z;

    // dimension
    float dx;
    float dy;
    float dz;

    // angle
    float yaw;

    // confidence
    float score;

    Classes cls;
};

#endif //TF_OD_C_TYPE_H
