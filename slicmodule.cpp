#include <Python.h>

#define PY_ARRAY_UNIQUE_SYMBOL slicmodule_PyArray_API
// #define NO_IMPORT_ARRAY # <-- this has to be put in all the files where there's NO module declaration.

#include "vigra/numpy_array.hxx"
#include "numpy/arrayobject.h"

#include "LKM.h"
#include "utils.h"

// #define DEBUG

using namespace vigra;

static PyObject * slic_NumpyArgs(PyObject *self, PyObject *args)
{
  PyObject *list2_obj;
  PyObject *list3_obj;
  int dimX;
  int dimY;
  int STEP;
  float M;
  if (!PyArg_ParseTuple(args, "Oiiif", &list2_obj, &dimX, &dimY,&STEP,&M)) // Getting arrays in PyObjects
    return NULL;

  #ifdef DEBUG
  printf("[slicmodule.cpp] Arrays dimensions : x: %d, y: %d \nSuperpixel parameters : STEP: %d, M: %d",dimX,dimY,STEP,M);
  #endif

  double ***inputVolume; // 3dims double array
  int * dimensions;

  //Create C arrays from numpy objects:
  int typenum = NPY_DOUBLE;
  PyArray_Descr *descr;
  descr = PyArray_DescrFromType(typenum);
  npy_intp dims[3];
  if (PyArray_AsCArray(&list2_obj, (void ***)&inputVolume, dims, 3, descr) < 0 ){ //|| PyArray_AsCArray(&list3_obj, (void ***)&list3, dims, 3, descr) < 0) {
    PyErr_SetString(PyExc_TypeError, "[slicmodule.cpp] Error converting to c array");
    return NULL; 
  }

  int numlabels = 100;
  // TODO CHris - how to avoid this? Creation of unnecessary new array
  int imgLength = dimX*dimY;
  UINT* ubuff = new UINT[imgLength];
  UINT* outbuff = new UINT[imgLength];
  sidType* labels = new sidType[imgLength]; 
  LKM lkm;

  int idx =0;
  for(int i=0;i<dimY;i++)
    for(int j=0;j<dimX;j++)
    {
      ubuff[idx] = inputVolume[i][j][0]; 
      idx++;
    }

  #ifdef DEBUG
  printf("[slicmodule.cpp] Generating superpixels. STEP=%d, M=%f\n", STEP, M);
  #endif
  lkm.DoSuperpixelSegmentation(ubuff, dimX, dimY, labels, numlabels, STEP, M);
  
  UINT color = 0xff0000;
  
  #ifdef DEBUG
  printf("[slicmodule.cpp] Drawing contours around segments ...\n");
  #endif
  DrawContoursAroundSegments(ubuff, labels, dimX, dimY,  color);//0xff0000 draws red contours
  
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
    for(int j=0;j<dimX;j++)
    {
      index[0] = i;
      index[1] = j;
      if(ubuff[idx] == color)
          { 
            
            if(PyArray_SETITEM((PyArrayObject*)returnval, (char*)PyArray_GetPtr(returnval,index), (PyObject*)Py_BuildValue("d",COLORBOUNDARY)) < 0)
              {  
                printf("[slicmodule.cpp] Error while setting items in output array.\n");
                return NULL;
              }
              // else
              //   printf("SETITEM OK in PyArray at location i : %d  j : %d \n",i,j );
          }
          else
            if(PyArray_SETITEM((PyArrayObject*)returnval, (char*)PyArray_GetPtr(returnval,index), (PyObject*)Py_BuildValue("d",COLORBACKGROUND)) < 0)
              {  
                printf("[slicmodule.cpp] Error while setting items in output array.\n");
                return NULL;
              }


      idx++;
    }

  #ifdef DEBUG
  printf("[slicmodule.cpp] Output array ready, casting PyArray to PyObject\n");
  #endif 

  return (PyObject*)returnval;
}

// Defining module methods
static PyMethodDef SlicMethods[] = {
     // ...
     {"ArgsTest",  slic_NumpyArgs, METH_VARARGS,
     "Passing numpy arrays as test."},
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