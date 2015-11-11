#include "utils.h"

// #include <cv.h>
// #include <highgui.h>



void SaveImage(
               UINT*  ubuff,        // RGB buffer
               const int&     width,        // size
               const int&     height,
               const string&    fileName)     // filename to be given; even if whole path is given, it is still the filename that is used
{
  // IplImage* img= 0; //cvCreateImage(cvSize(width,height),IPL_DEPTH_8U,3); 
  // uchar* pValue;
  // int idx = 0;

  // for(int j=0;j<img->height;j++)
  //   for(int i=0;i<img->width;i++)
  //     {
  //       pValue = &((uchar*)(img->imageData + img->widthStep*(j)))[(i)*img->nChannels];
  //       pValue[0] = ubuff[idx] & 0xff;
  //       pValue[1] = (ubuff[idx] >> 8) & 0xff;
  //       pValue[2] = (ubuff[idx] >>16) & 0xff;
  //       idx++;
  //     }

  //cvSaveImage(fileName.c_str(),img);
}


//=================================================================================
/// DrawContoursAroundSegments
///
/// Internal contour drawing option exists. One only needs to comment the 'if'
/// statement inside the loop that looks at neighbourhood.
//=================================================================================
void DrawContoursAroundSegments(
                                UINT*     img,//contours will be drawn on this image
                                sidType*      labels,
                                const int&        width,
                                const int&        height,
                                const UINT&       color )
{
  const int dx8[8] = {-1, -1,  0,  1, 1, 1, 0, -1};
  const int dy8[8] = { 0, -1, -1, -1, 0, 1, 1,  1};

  int sz = width*height;

  vector<bool> istaken(sz, false);

  int mainindex(0);
  int cind(0);
  for( int j = 0; j < height; j++ )
    {
      for( int k = 0; k < width; k++ )
        {
          int np(0);
          for( int i = 0; i < 8; i++ )
            {
              int x = k + dx8[i];
              int y = j + dy8[i];

              if( (x >= 0 && x < width) && (y >= 0 && y < height) )
                {
                  int index = y*width + x;

                  if( false == istaken[index] )//comment this to obtain internal contours
                    {
                      if( labels[mainindex] != labels[index] ) np++;
                    }
                }
            }
          if( np > 1 )
            {
              istaken[mainindex] = true;
              img[mainindex] = color;
              cind++;
            }
          mainindex++;
        }
    }
}

//=================================================================================
/// DrawContoursAroundSegments
///
/// Internal contour drawing option exists. One only needs to comment the 'if'
/// statement inside the loop that looks at neighbourhood.
//=================================================================================
void DrawContoursAroundVoxels(
                                double**      img,    //contours will be drawn on this image
                                sidType**     labels,
                                const int&    width,
                                const int&    height,
                                const int&    depth,
                                const UINT&   color )
{
  const int dx10[10] = {-1,  0,  1,  0, -1,  1,  1, -1,  0, 0};
  const int dy10[10] = { 0, -1,  0,  1, -1, -1,  1,  1,  0, 0};
  const int dz10[10] = { 0,  0,  0,  0,  0,  0,  0,  0, -1, 1};

  int sz = width*height;
  vector< vector<bool> > istaken(depth, vector<bool>(sz,false));

  int mainindex(0);
  int cind(0);
  for(int i =0; i<depth;i++)
  {
    mainindex = 0;
    for( int j = 0; j < height; j++ )
      {
        for( int k = 0; k < width; k++ )
          {
            int np(0);

            for( int ix = 0; ix < 10; ix++ )
              {
                int x = k + dx10[ix];
                int y = j + dy10[ix];
                int z = i + dz10[ix];

                if( (x >= 0 && x < width) && (y >= 0 && y < height) && (z >= 0 && z < depth))
                  {
                    int index = y*width + x;

                    if( false == istaken[z][index] )//comment this to obtain internal contours
                      {
                        if( labels[z][mainindex] != labels[z][index] ) np++;
                      }
                  }
              }
            if( np > 1 )
              {
                istaken[i][mainindex] = true;
                img[i][mainindex] = (double)color;
                cind++;
              }
              else
              {
                img[i][mainindex] = 0;
              }
            mainindex++;
          }
      }
    }
}


void splitpath(string path, char* fname, char* ext)
{
  string nameWithExt =  path.substr(path.find_last_of("/\\")+1);
  int dotIdx = nameWithExt.find_last_of(".");
  string nameWithoutExt = nameWithExt.substr(0,dotIdx);
  strcpy(ext, nameWithoutExt.c_str());

  string sExt = nameWithExt.substr(0,dotIdx);
  strcpy(ext, sExt.c_str());
}

string getNameFromPathWithoutExtension(string path){
  string nameWith =  path.substr(path.find_last_of("/\\")+1);
  string nameWithout = nameWith.substr(0,nameWith.find_last_of("."));
  return nameWithout;
}
/*
string getNameFromPathWithoutExtension(string path){
  string nameWith =  path.substr(path.find_last_of("/\\")+1);
  string nameWithout = nameWith.substr(0,nameWith.find_last_of("."));
  return nameWithout;
}

string getExtension(string path){
  return path.substr(path.find_last_of(".")+1);
}
*/