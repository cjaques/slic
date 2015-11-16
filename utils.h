#ifndef SLIC_UTILS_H
#define SLIC_UTILS_H 

#include <string.h>
#include <stdio.h>
#include <assert.h>

#include <vector>
//#include <sstream>
#include <iostream>
#include <stdlib.h>
#include <string>
#include <errno.h>


#include "LKM.h"

using namespace std;

void splitpath(string path, char* fname, char* ext);
string getNameFromPathWithoutExtension(string path);
void SaveImage(
               UINT*	ubuff,				// RGB buffer
               const int&			width,				// size
               const int&			height,
               const string&		fileName);			// filename to be given; even if whole path is given, it is still the filename that is used

void DrawContoursAroundSegments(
                                UINT*     img,//contours will be drawn on this image
                                sidType*      labels,
                                const int&        width,
                                const int&        height,
                                const UINT&       color );
void DrawContoursAroundVoxels(
                                double***     img,    //contours will be drawn on this image
                                sidType***    labels,
                                const int&    width,
                                const int&    height,
                                const int&    depth,
                                const UINT&   color );

#endif //SLIC_UTILS_H
