#include <Python.h>

#define PY_ARRAY_UNIQUE_SYMBOL slicmodule_PyArray_API
// #define NO_IMPORT_ARRAY # <-- this has to be put in all the files where there's NO module declaration. We have a module declaration here.

#include "numpy/arrayobject.h"

#include "LKM.h"
#include "utils.h"

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"

// #define DEBUG 

/* -----------------------------------------
        Functions
 ---------------------------------------*/
void Extract_array2D(PyArrayObject *returnval,npy_intp *dims,UINT *ubuff, UINT color);
template<typename T> void Extract_array3D(PyArrayObject * returnval,npy_intp *dims,T **labels);

/* -----------------------------------------
        Python callbacks
 ---------------------------------------*/
static PyObject *PyCallback_setBoundaries = NULL;

/* -----------------------------------------
        COMPUTE 2D SUPERPIXELS 
--------------------------------------------*/
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
  printf("[slicmodule.cpp] Arrays dimensions : x: %d, y: %d \n[slicmodule.cpp] Superpixel parameters : STEP: %d, M: %f\n",dimX,dimY,STEP,M);
  #endif

  double ***inputVolume;
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

  int numlabels = STEP*STEP;
  // TODO CHris - avoid this? Creation of unnecessary new array
  dimX = dims[0];
  dimY = dims[1];
  int imgLength = dimX*dimY;
  UINT* ubuff = new UINT[imgLength];
  sidType* labels = new sidType[imgLength]; 
  LKM lkm;

  #ifdef DEBUG
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
  
  PyObject* ret = PyArray_SimpleNew(3, dims, NPY_DOUBLE);
  PyArrayObject * returnval = (PyArrayObject*)PyArray_FROM_OTF(ret,NPY_DOUBLE, NPY_ARRAY_OUT_ARRAY);
  Extract_array2D(returnval,dims,ubuff, color);
  
  #ifdef DEBUG
  printf("[slicmodule.cpp] Output array ready, casting PyArray to PyObject\n");
  #endif 

  Py_INCREF(returnval);
  return (PyObject*)returnval;
}

void Extract_array2D(PyArrayObject * returnval,npy_intp *dims,UINT *ubuff, UINT color)
{
  double COLORBOUNDARY = 255.0;
  double COLORBACKGROUND = 0.0;
  npy_intp index[3];
  int idx =0;
  index[2] = 0; 
  for(int i=0;i<dims[1];i++)
  {
    for(int j=0;j<dims[0];j++) 
    {
      index[0] = j; 
      index[1] = i;
      if(ubuff[idx] == color)
          { 
            
            if(PyArray_SETITEM((PyArrayObject*)returnval, (char*)PyArray_GetPtr(returnval,index), (PyObject*)Py_BuildValue("d",COLORBOUNDARY)) < 0)
              {  
                printf("[slicmodule.cpp] Error while setting items in output array.\n");
                return;
              }
          }
          else
          {
            if(PyArray_SETITEM((PyArrayObject*)returnval, (char*)PyArray_GetPtr(returnval,index), (PyObject*)Py_BuildValue("d",COLORBACKGROUND)) < 0)
              {  
                printf("[slicmodule.cpp] Error while setting items in output array.\n");
                return;
              }
          }
      idx++;
    }

  }

}
/* -----------------------------------------
        COMPUTE 3D SUPERVOXELS 
--------------------------------------------*/
static PyObject * slic_Compute3DSlic(PyObject *self, PyObject *args)
{
  PyObject *inputArray;
  int STEP;
  float M;
  int MAX_NUM_ITERATIONS;

  // Parsing arguments
  if (!PyArg_ParseTuple(args, "Oifi", &inputArray,&STEP, &M,&MAX_NUM_ITERATIONS)) 
    return NULL;

  double ***inputVolume;
  int * dimensions;

  //C arrays from numpy objects:
  int typenum = NPY_DOUBLE;
  PyArray_Descr *descr;
  descr = PyArray_DescrFromType(typenum);
  npy_intp dims[3];
  if (PyArray_AsCArray(&inputArray, (void ***)&inputVolume, dims, 3, descr) < 0 ){ 
    PyErr_SetString(PyExc_TypeError, "[slicmodule.cpp] Error converting to c array");
    return NULL; 
  }

  #ifdef DEBUG
  printf("[slicmodule.cpp] Arrays dimensions : x: %d, y: %d z: %d\n[slicmodule.cpp] Supervoxel parameters : STEP: %d, M: %.1f\n",dims[0],dims[1],dims[2],STEP,M);
  #endif

  int dimZ = dims[0]; 
  int dimY = dims[1];
  int dimX = dims[2];
  int imgLength = dimX*dimY; 
  int imgDepth = dimZ;
  int numlabels;
  double *** ubuff = new double*[imgDepth];
  sidType *** labels = new sidType*[imgDepth]; 
  LKM lkm;

  // // Copy data from input to ubuff --> avoid this?
  // int idx =0;
  // for(int k=0;k<dimZ;k++)
  // {
  //   ubuff[k] = new double[imgLength];
  //   labels[k] = new sidType[imgLength];
  //   idx = 0;
  //   for(int j=0;j<dimY;j++)
  //   {
  //     for(int i=0;i<dimX;i++) 
  //       {
  //         ubuff[k][idx] = (double)inputVolume[k][j][i];
  //         idx ++;
  //       }
  //   }
  // }

  UINT color = 255;
  lkm.DoSupervoxelSegmentationForGrayVolume(inputVolume, dimX, dimY, dimZ, labels, numlabels, STEP, M);
  DrawContoursAroundVoxels(ubuff,labels,dimX,dimY,dimZ,color);
  
  #ifdef DEBUG
  printf("[slicmodule.cpp] Output array ready, casting PyArray to PyObject\n");
  #endif

  PyObject* ret = PyArray_SimpleNew(3, dims, NPY_DOUBLE);
  PyObject* bnd = PyArray_SimpleNew(3, dims, NPY_DOUBLE);
  PyArrayObject * returnval = (PyArrayObject*)PyArray_FROM_OTF(ret,NPY_DOUBLE, NPY_ARRAY_OUT_ARRAY);
  PyArrayObject * boundaries = (PyArrayObject*)PyArray_FROM_OTF(bnd,NPY_DOUBLE, NPY_ARRAY_OUT_ARRAY);

  Extract_array3D<sidType>(returnval,dims,labels); // Gets labels into returnval
  Extract_array3D<double>(boundaries,dims,ubuff); // Gets boundaries
  
  PyObject* callbackResult;
  if(PyCallback_setBoundaries!=NULL)
     callbackResult = PyObject_CallObject(PyCallback_setBoundaries, (PyObject*)Py_BuildValue("(O)", boundaries ));

  #ifdef DEBUG
  printf("[slicmodule.cpp] Returning outputs\n");
  #endif 
  
  /* -------------------------
       CLEAN UP 
  ---------------------------*/
  for(int k=0;k<dimZ;k++)
  {
    delete [] ubuff[k] ;  
    delete [] labels[k] ;
  }
  delete[] ubuff;  
  delete [] labels; 
  
  /* -------------------------
       RETURN VAL
  ---------------------------*/
  Py_INCREF(returnval);
  return (PyObject*)returnval;
}

template<typename T> void Extract_array3D(PyArrayObject * returnval,npy_intp *dims,T **labels)
{
  PyObject* labelValue;
  npy_intp index[3];
  int idx =0;

  double SIZE = dims[0]*dims[1]*dims[2];

  #ifdef DEBUG
  printf("[slicmodule.cpp] Extract3Darray - Dims : %d - %d - %d \n",dims[0],dims[1],dims[2]);
  #endif 

  for(int i =0;i<dims[0];i++)
  { 
    idx =0;

    for(int j=0;j<dims[1];j++)
    {
      for(int k=0;k<dims[2];k++) 
      {
        index[2] = k;
        index[1] = j;
        index[0] = i; 

        labelValue = (PyObject*)Py_BuildValue("d",(double)(labels[i][idx]) );
        idx++;

        if(PyArray_SETITEM((PyArrayObject*)returnval, (char*)PyArray_GetPtr(returnval,index),labelValue ) < 0) 
          {  
            printf("[slicmodule.cpp] Error while setting items in output array.\n");
            return;
          }
      }
    }
  }
}

static PyObject * setPythonBoundariesCallback(PyObject *dummy, PyObject *args)
{
    PyObject *result = NULL;
    PyObject *temp;
    
    if (PyArg_ParseTuple(args, "O:set_callback", &temp)) {
        if (!PyCallable_Check(temp)) {
            PyErr_SetString(PyExc_TypeError, "parameter must be callable");
            return NULL;
        }
        Py_XINCREF(temp);         /* Add a reference to new callback */
        Py_XDECREF(PyCallback_setBoundaries);  /* Dispose of previous callback */
        PyCallback_setBoundaries = temp;       /* Remember new callback */

        /* Boilerplate to return "None" */
        Py_INCREF(Py_None);
        result = Py_None;
    }
    
    return result;
}


// Defining module methods
static PyMethodDef SlicMethods[] = {
     {"Compute2DSlic",  slic_Compute2DSlic, METH_VARARGS,
     "Computes 2D slic on the input image."},
     {"Compute3DSlic",  slic_Compute3DSlic, METH_VARARGS,
     "Computes 3D slic on the input volume."},
     {"SetPyCallback",  setPythonBoundariesCallback, METH_VARARGS,
     "Sets a Python callback function."},
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


/*
  ---------------------------- 
        DEBUG SECTION 
  ----------------------------
  ---------------------------*/

// #ifdef DEBUG
//         val = (double)(labels[i][idx]);
//         if(val >= max)
//           if(val == max)
//             countMax ++;
//           else
//           {
//             countMax =0;
//             max = val;
//           }
//         if(val <= min)
//           if(val == min)
//             countMin ++;
//           else
//           {
//             countMin =0;
//             min = val;
//           }
//         #endif

  // ---------------


  // int idx1=0;
  // int dimz = dims[0];
  // int dimy = dims[1];
  // int dimx = dims[2];
  // UINT color = 0xff0000;
  // int numLabelss = STEP*STEP;

  // cv::Mat im2 = cv::Mat::zeros(dimy,dimx, CV_8UC1);
  // cv::Mat im3 = cv::Mat::zeros(dimy,dimx, CV_8UC1);

  // sidType* labelsMM = new sidType[dimx*dimy];
  // UINT* ubb = new UINT[dimx*dimy];
  
  // printf("About to process volume slice by slice\n");
  
  // for(int k=0;k<dimz;k++)
  //  {
  //   idx1=0;

  //   // Pass data from input to ubuff
  //   int idx2 =0;
  //   for(int i=0;i<dimy;i++)
  //   {
  //     for(int j=0;j<dimx;j++) 
  //       {
  //         ubb[idx2] = (UINT)inputVolume[k][i][j];
  //         idx2 ++;
  //       }
  //   }
    
  //   lkm.DoSuperpixelSegmentation(ubb,dimx,dimy,labelsMM,numLabelss, 5, 2.0, 6);
  //   DrawContoursAroundSegments(ubb, labelsMM, dimx,dimy, color);

  //   for(int i=0;i<dimy;i++)
  //   {
  //     uchar* rowi = im2.ptr(i);
  //     uchar* rowM = im3.ptr(i);
  //     for(int j=0;j<dimx;j++) 
  //       {
  //         rowi[j] = labels[k][idx1];
          
  //         if(ubb[idx1] == color)
  //           rowM[j] = 255;
  //         else
  //           rowM[j] = 0;
  //         idx1++;
  //       }
  //   }
  //   string name = "/Users/Chris/Code/Images/stacks/superVox" + std::to_string(k)+ ".jpg";
  //   string nameMM = "/Users/Chris/Code/Images/stacks/superPix" + std::to_string(k)+ ".jpg";
  //   cv::imwrite(name,im2);
  //   cv::imwrite(nameMM,im3);
  // }

  /*
  ---------------------------- 
       END OF DEBUG SECTION 
  ----------------------------
  ---------------------------*/
// DEBUG 
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