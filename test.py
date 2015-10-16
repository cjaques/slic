import slic, numpy
l2 = numpy.array([[1.0,2.0,3.0],
                  [4.0,5.0,6.0],
                  [7.0,8.0,9.0],
                  [6.0, 5.0, 0.0]], dtype="double")

l3 = numpy.array([[[2,7, 1, 11], [6, 3, 9, 12]],
                  [[1, 10, 13, 15], [4, 2, 6, 2]]], dtype="double")

slic.ArgsTest(l2, l3)
