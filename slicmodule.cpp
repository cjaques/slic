#include <Python.h>

#define PY_ARRAY_UNIQUE_SYMBOL slicmodule_PyArray_API
// #define NO_IMPORT_ARRAY # <-- this has to be put in all the files where there's NO module declaration. We have a module declaration here.

#include "vigra/numpy_array.hxx"
#include "numpy/arrayobject.h"

#include "LKM.h"
#include "utils.h"

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"

// #define DEBUG

using namespace vigra;

static PyObject * slic_Compute2DSlic(PyObject *self, PyObject *args)
{
  PyObject *inputArray;

  int dimX;
  int dimY;
  int STEP;
  float M;
  int MAX_NUM_ITERATIONS;
  if (!PyArg_ParseTuple(args, "Oifi", &inputArray,&STEP,&M,&MAX_NUM_ITERATIONS)) // Getting arrays in PyObjects
    return NULL;

  #ifdef DEBUG
  printf("[slicmodule.cpp] Arrays dimensions : x: %d, y: %d \nSuperpixel parameters : STEP: %d, M: %f\n",dimX,dimY,STEP,M);
  #endif

  double ***inputVolume; // 3 dimensionnal double array
  int * dimensions;

  //Create C arrays from numpy objects:
  int typenum = NPY_DOUBLE;
  PyArray_Descr *descr;
  descr = PyArray_DescrFromType(typenum);
  npy_intp dims[3];
  if (PyArray_AsCArray(&inputArray, (void ***)&inputVolume, dims, 3, descr) < 0 ){ 
    PyErr_SetString(PyExc_TypeError, "[slicmodule.cpp] Error converting to c array");
    return NULL; 
  }

  int numlabels = 100;
  // TODO CHris - how to avoid this? Creation of unnecessary new array
  dimX = dims[0];
  dimY = dims[1];
  int imgLength = dimX*dimY;
  UINT* ubuff = new UINT[imgLength];
  UINT* outbuff = new UINT[imgLength];
  sidType* labels = new sidType[imgLength]; 
  LKM lkm;

  #ifdef DEBUG
  // The following lines copy and save the input image. Used to debug inverted axis from numpy to C++
  // idx1=0;
  // cv::Mat im2 = cv::Mat::zeros(dimX,dimY, CV_8UC1);
  // for(int i=0;i<dimY;i++)
  // {
  //   uchar* rowi = im2.ptr(i);
  //   for(int j=0;j<dimX;j++) 
  //     {
  //       rowi[j] = ubuff[idx1];
  //       idx1 ++;
  //     }
  // }
  // cv::imwrite("/Users/Chris/Code/Images/deeeeAfterContours.jpg",im2);
  printf("[slicmodule.cpp] Generating superpixels. STEP=%d, M=%f\n", STEP, M);
  #endif

  // Pass data from input to ubuff
  int idx =0;
  for(int i=0;i<dimY;i++)
  {
    for(int j=0;j<dimX;j++) 
      {
        ubuff[idx] = inputVolume[j][i][0];
        idx ++;
      }
  }
  
  lkm.DoSuperpixelSegmentation(ubuff, dimX, dimY, labels, numlabels, STEP, M,MAX_NUM_ITERATIONS);

  UINT color = 0xff0000; //0xff0000 draws red contours
  
  #ifdef DEBUG
  printf("[slicmodule.cpp] Drawing contours around segments ...\n");
  #endif
  DrawContoursAroundSegments(ubuff, labels, dimX, dimY, color);
  
  dims[0] = dimX;
  dims[1] = dimY;
  dims[2] = 1;
  double COLORBOUNDARY = 255.0;
  double COLORBACKGROUND = 0.0;
  
  PyObject* ret = PyArray_SimpleNew(3, dims, NPY_DOUBLE);
  PyArrayObject * returnval = (PyArrayObject*)PyArray_FROM_OTF(ret,NPY_DOUBLE, NPY_ARRAY_OUT_ARRAY);
  npy_intp index[3];
  idx =0;
  index[2] = 0; 

  for(int i=0;i<dimY;i++)
  {
    for(int j=0;j<dimX;j++) 
    {
      index[0] = j; 
      index[1] = i;
      if(ubuff[idx] == color)
          { 
            
            if(PyArray_SETITEM((PyArrayObject*)returnval, (char*)PyArray_GetPtr(returnval,index), (PyObject*)Py_BuildValue("d",COLORBOUNDARY)) < 0)
              {  
                printf("[slicmodule.cpp] Error while setting items in output array.\n");
                return NULL;
              }
          }
          else
          {
            if(PyArray_SETITEM((PyArrayObject*)returnval, (char*)PyArray_GetPtr(returnval,index), (PyObject*)Py_BuildValue("d",COLORBACKGROUND)) < 0)
              {  
                printf("[slicmodule.cpp] Error while setting items in output array.\n");
                return NULL;
              }
          }
      idx++;
    }

  }
  
  #ifdef DEBUG
  printf("[slicmodule.cpp] Output array ready, casting PyArray to PyObject\n");
  #endif 

  Py_INCREF(returnval);
  return (PyObject*)returnval;
}

static PyObject * slic_Compute3DSlic(PyObject *self, PyObject *args)
{
  PyObject *inputArray;

  int dimX;
  int dimY;
  int STEP;
  float M;
  int MAX_NUM_ITERATIONS;
  if (!PyArg_ParseTuple(args, "O", &inputArray)) // Getting arrays in PyObjects
    return NULL;

  #ifdef DEBUG
  printf("[slicmodule.cpp] Arrays dimensions : x: %d, y: %d \nSupervoxel parameters : STEP: %d, M: %f\n",dimX,dimY,STEP,M);
  #endif

  double ****inputVolume; // 4 dimensionnal double array
  int * dimensions;

  //Create C arrays from numpy objects:
  int typenum = NPY_DOUBLE;
  PyArray_Descr *descr;
  descr = PyArray_DescrFromType(typenum);
  npy_intp dims[4];
  if (PyArray_AsCArray(&inputArray, (void ****)&inputVolume, dims, 4, descr) < 0 ){ 
    PyErr_SetString(PyExc_TypeError, "[slicmodule.cpp] Error converting to c array");
    return NULL; 
  }


  printf("In compute3D slic...\n");
  return inputArray;
}

// Defining module methods
static PyMethodDef SlicMethods[] = {
     // ...
     {"Compute2DSlic",  slic_Compute2DSlic, METH_VARARGS,
     "Computes 2D slic on the input image."},
     {"Compute3DSlic",  slic_Compute3DSlic, METH_VARARGS,
     "Computes 3D slic on the input volume."},
    // ...
    {NULL, NULL, 0, NULL}        /* Sentinel */
};


// Python module INIT function
PyMODINIT_FUNC
initslic(void)
{

    PyObject *m;
    import_array();
    m = Py_InitModule("slic", SlicMethods);
    if (m == NULL)
        return;

}