#include <Python.h>

#define PY_ARRAY_UNIQUE_SYMBOL slicmodule_PyArray_API
// #define NO_IMPORT_ARRAY # <-- this has to be put in all the files where there's NO module declaration.
#include "numpy/arrayobject.h"
#include "LKM.h"

static PyObject *
slic_NumpyArgs(PyObject *self, PyObject *args)
{
  PyObject *list2_obj;
  PyObject *list3_obj;
  int dimX;
  int dimY;
  if (!PyArg_ParseTuple(args, "OOii", &list2_obj, &list3_obj, &dimX, &dimY)) // Getting arrays in PyObjects
    return NULL;
     


  printf("Arrays dimensions : x: %d, y: %d \n",dimX,dimY);

  double ***list2;
  double ***list3;
  int * dimensions;

  //Create C arrays from numpy objects:
  int typenum = NPY_DOUBLE;
  PyArray_Descr *descr;

  descr = PyArray_DescrFromType(typenum);

  npy_intp dims[3];
  if (PyArray_AsCArray(&list2_obj, (void ***)&list2, dims, 3, descr) < 0 || PyArray_AsCArray(&list3_obj, (void ***)&list3, dims, 3, descr) < 0) {
    PyErr_SetString(PyExc_TypeError, "error converting to c array");
    return NULL; 
  }
  printf("Input image: %f, Output (not set yet): %f.\n", list2[3][1][0], list3[1][0][0]);
  printf("Size of each element : %d \n",sizeof(list2[0]));
  // Stupid copy of arrays to test SLIC algo.
  double ** input1 = new double[dimX][dimY];
  double ** result = new double[dimX][dimY];

  for(int i=0;i<dimX;i++)
    for(int j=0;j<dimY;j++)
    {
      input1[i][j] = list2[i][j][0];
    }



  Py_INCREF(Py_None); // without this Py_INCREF, the python interpreter may try to delete Py_None --> segfault
  return Py_None;
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
