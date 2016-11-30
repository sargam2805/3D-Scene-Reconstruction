# pragma once

#include "geometryfusion_mipmap_cpu1.hpp"
#include "mesh_interleaved1.hpp"
#include "mesh_interleaved_meshcell.hpp"
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudafilters.hpp>
#include <pmmintrin.h>

using namespace cv;
using namespace std;




void add_brickz_caller(treeinfo info,
                    sidetype ox, sidetype oy, sidetype oz,
                    sidetype size,
                    volumetype lastleaf0, volumetype lastleaf1,
                    const ParentArray &leafParent,
                    const MarchingCubesIndexed &mc,
                    MeshInterleaved *pmesh
        );
