#include "geometryfusion_mipmap_cpu1.hpp"
#include "mesh_interleaved1.hpp"
#include "mesh_interleaved_meshcell.hpp"
#include "mesh1.hpp"
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudafilters.hpp>
#include <pmmintrin.h>
#include <stdio.h>
#include <iostream>
#include <vector>
#include <list>
#include "brickz_cuda.hpp"
#include "brick.hpp"

using namespace cv;
using namespace std;


__global__ void add_brickz_kernel (treeinfo info,
                    sidetype ox, sidetype oy, sidetype oz,
                    sidetype size,
                    volumetype lastleaf0, volumetype lastleaf1,
                    const ParentArray &leafParent,
                    const MarchingCubesIndexed &mc,
                    MeshInterleaved *pmesh
        ){
    sidetype &bl = info.brickLength;
	sidetype &brickSize = info.brickSize;
	const sidetype *leafScale = info.leafScale;
	const sidetype3 *leafPos = info.leafPos;
	const float *distance = info.distance;
	const weighttype *weights = info.weights;
	const colortype3 *color = info.color;
	float minWeight = info.minWeight;
//	unsigned int &degenerate_faces = *info.degenerate_faces;
	float3 offset = info.offset;
	float scale = info.scale;
	MeshInterleaved &mesh = *pmesh;

	sidetype bs = bl*bl;

	float *d = new float[brickSize];
	weighttype *w = new weighttype[brickSize];
	VertexColor *c = new VertexColor[brickSize];
	size_t *indices = new size_t[brickSize*3];
	bool *vertexIsSet = new bool[brickSize*3];
	for(unsigned int i=0;i<brickSize*3;i++) vertexIsSet[i] = false;
	bool *faceIsSet = new bool[brickSize*3];
	for(unsigned int i=0;i<brickSize*3;i++) faceIsSet[i] = false;
	int **tables = new int*[brickSize];
	for(unsigned int i=0;i<brickSize;i++) tables[i] = NULL;

	weighttype wf[4]; float df[4]; colortype3 cf[4];

	volumetype lastleaves[2] = {lastleaf0,lastleaf1};
    sidetype sizeMin;
    if(leafScale[lastleaf0]>leafScale[lastleaf1]){
        sizeMin=leafScale[lastleaf1];
    }
    else{
        sizeMin=leafScale[lastleaf0];
    }
	//sidetype sizeMin = std::min(leafScale[lastleaf0],leafScale[lastleaf1]);
	sidetype sizeStretch = leafScale[lastleaf0];

	if(sizeStretch>size){
    //fprintf(stderr,"\nWARNING Wall Z: The Leaf is too large: %i > %i",sizeStretch,size);
		return;
	}

	sidetype z[2] = {(sidetype)(oz+size-sizeStretch) , (sidetype)(oz+size)};
	
	int x1 = threadIdx.x + blockIdx.x*blockDim.x;
	int y1 = threadIdx.y + blockIdx.y*blockDim.y;
	sidetype x=ox+x1*size,y=oy+y1*size;

	if (y<oy+size){
		sidetype by = (y-oy)/sizeMin;
		if(x<ox+size){
			sidetype bx = (x-ox)/sizeMin;

			for(unsigned int bz=0;bz<2;bz++){
				volumetype idx = (bz*bl+by)*bl+bx;
				w[idx] = 0;
				for(volumetype leaf=lastleaves[bz];leaf<BRANCHINIT
#ifdef ADD_WEIGHTS_TRANSITION_140424
				&& w[idx]<=MIN_WEIGHT_FOR_SURFACE;
#else
				&& w[idx]<=0.0f;
#endif
				leaf=leafParent[leaf]
	//				leaf=BRANCHINIT
				 ){
					volumetype start = leaf*brickSize;
					sidetype3 lo = leafPos[leaf];
					sidetype ls = leafScale[leaf];

					sidetype lz = (z[bz]-lo.z)/ls;

					sidetype lxl = (x-lo.x)/ls;
					sidetype lxr = (x-lo.x)%ls;
					float rx = (float)lxr/(float)ls;
					sidetype lxh = lxl+(rx>0.0 && lxl<bl-1);

					sidetype lyl = (y-lo.y)/ls;
					sidetype lyr = (y-lo.y)%ls;
					float ry = (float)lyr/(float)ls;
					sidetype lyh = lyl+(ry>0.0 && lyl<bl-1);

					volumetype idxLeaf[4] = {
							(volumetype)((lz*bl+lyl)*bl+lxl),
							(volumetype)((lz*bl+lyl)*bl+lxh),
							(volumetype)((lz*bl+lyh)*bl+lxl),
							(volumetype)((lz*bl+lyh)*bl+lxh)
					};

					for(volumetype i=0;i<4;i++) {
						wf[i] = weights[start+idxLeaf[i]];
						df[i] = distance[start+idxLeaf[i]];
						if(color) cf[i] = color[start+idxLeaf[i]];
					}

					float rxInv = 1.0f-rx;
					float ryInv = 1.0f-ry;

#ifndef WEIGHT_MINIMUM
#ifdef ADD_WEIGHTS_TRANSITION_140424
					w[idx] += ryInv*rxInv*(float)wf[0]+
									  ryInv*rx   *(float)wf[1]+
									  ry   *rxInv*(float)wf[2]+
									  ry   *rx   *(float)wf[3];
#else
					w[idx] = ryInv*rxInv*(float)wf[0]+
									 ryInv*rx   *(float)wf[1]+
									 ry   *rxInv*(float)wf[2]+
									 ry   *rx   *(float)wf[3];
#endif
#else
					w[idx] = std::min(std::min(wf[0],wf[1]),std::min(wf[2],wf[3]));
#endif

					d[idx] = ryInv*rxInv*df[0]+
									 ryInv*rx   *df[1]+
									 ry   *rxInv*df[2]+
									 ry   *rx   *df[3];

					if(color)
#ifndef BRICKVISUALIZATION
#ifndef COLORINVERSION
					c[idx] = VertexColor(
									 ryInv*rxInv*(float)cf[0].x+
									 ryInv*rx   *(float)cf[1].x+
									 ry   *rxInv*(float)cf[2].x+
									 ry   *rx   *(float)cf[3].x,
									 ryInv*rxInv*(float)cf[0].y+
									 ryInv*rx   *(float)cf[1].y+
									 ry   *rxInv*(float)cf[2].y+
									 ry   *rx   *(float)cf[3].y,
									 ryInv*rxInv*(float)cf[0].z+
									 ryInv*rx   *(float)cf[1].z+
									 ry   *rxInv*(float)cf[2].z+
									 ry   *rx   *(float)cf[3].z);
#else
					c[idx] = VertexColor(
									 ryInv*rxInv*(float)cf[0].z+
									 ryInv*rx   *(float)cf[1].z+
									 ry   *rxInv*(float)cf[2].z+
									 ry   *rx   *(float)cf[3].z,
									 ryInv*rxInv*(float)cf[0].y+
									 ryInv*rx   *(float)cf[1].y+
									 ry   *rxInv*(float)cf[2].y+
									 ry   *rx   *(float)cf[3].y,
									 ryInv*rxInv*(float)cf[0].x+
									 ryInv*rx   *(float)cf[1].x+
									 ry   *rxInv*(float)cf[2].x+
									 ry   *rx   *(float)cf[3].x);
#endif
#else
						c[idx] = VertexColor(0,0,65280);
#endif
				}
			}
		}
	}

	if(y<oy+size-sizeMin){
		sidetype by = (y-oy)/sizeMin;
		if(x<ox+size-sizeMin){
			sidetype bx = (x-ox)/sizeMin;

			volumetype idx = by*bl+bx;
			if(weightInfluence(minWeight,
					w[idx],w[idx+1],
					w[idx+bl+1],w[idx+bl],
					w[idx+bs],w[idx+bs+1],
					w[idx+bs+bl+1],w[idx+bs+bl])){
				int *table = mc.offsetTable[mc.getCubeIndex(
						d[idx],d[idx+1],
						d[idx+bl+1],d[idx+bl],
						d[idx+bs],d[idx+bs+1],
						d[idx+bs+bl+1],d[idx+bs+bl],
						w[idx],w[idx+1],
						w[idx+bl+1],w[idx+bl],
						w[idx+bs],w[idx+bs+1],
						w[idx+bs+bl+1],w[idx+bs+bl])];

				for (unsigned int i=0;table[i]!=-1;i+=3) {
					faceIsSet[3*idx+table[i  ]] = true;
					faceIsSet[3*idx+table[i+1]] = true;
					faceIsSet[3*idx+table[i+2]] = true;
				}
				tables[idx] = table;
			}
		}
	}

	size_t runningIndex = mesh.vertices.size();
	if(y<oy+size){
		sidetype by = (y-oy)/sizeMin;
		if(x<ox+size){
			sidetype bx = (x-ox)/sizeMin;
			for(sidetype bz=0;bz<2;bz++){
				volumetype idx = (bz*bl+by)*bl+bx;

				if(
						x<ox+size-sizeMin &&
						faceIsSet[3*idx+0]
				 	&& w[idx] && w[idx+1] && ((d[idx]<0)!=(d[idx+1]<0))
				){
					Vertex3f ver = MarchingCubes::VertexInterp(
							Vertex3f(offset.x+(x        )*scale,offset.y+y*scale,offset.z+z[bz]*scale),
							Vertex3f(offset.x+(x+sizeMin)*scale,offset.y+y*scale,offset.z+z[bz]*scale),
							d[idx],d[idx+1]);
					VertexColor col = MarchingCubes::VertexInterp(c[idx],c[idx+1],d[idx],d[idx+1]);
					mesh.vertices.push_back(ver);
					mesh.colors.push_back(Color3b(col.x/COLOR_MULTIPLICATOR,col.y/COLOR_MULTIPLICATOR,col.z/COLOR_MULTIPLICATOR));
					indices[3*idx+0] = runningIndex++;
					vertexIsSet[3*idx+0] = true;
				}
				if(
						y<oy+size-sizeMin &&
						faceIsSet[3*idx+1]
						&& w[idx] && w[idx+bl] && ((d[idx]<0)!=(d[idx+bl]<0))
				){
					Vertex3f ver = MarchingCubes::VertexInterp(
							Vertex3f(offset.x+x*scale,offset.y+(y        )*scale,offset.z+z[bz]*scale),
							Vertex3f(offset.x+x*scale,offset.y+(y+sizeMin)*scale,offset.z+z[bz]*scale),
							d[idx],d[idx+bl]);
					VertexColor col = MarchingCubes::VertexInterp(c[idx],c[idx+bl],d[idx],d[idx+bl]);
					mesh.vertices.push_back(ver);
					mesh.colors.push_back(Color3b(col.x/COLOR_MULTIPLICATOR,col.y/COLOR_MULTIPLICATOR,col.z/COLOR_MULTIPLICATOR));
					indices[3*idx+1] = runningIndex++;
					vertexIsSet[3*idx+1] = true;
				}
				if(
					 faceIsSet[3*idx+2]
						&& w[idx] && w[idx+bs] &&((d[idx]<0)!=(d[idx+bs]<0))
				){
					Vertex3f ver = MarchingCubes::VertexInterp(
							Vertex3f(offset.x+x*scale,offset.y+y*scale,offset.z+(z[bz]            )*scale),
							Vertex3f(offset.x+x*scale,offset.y+y*scale,offset.z+(z[bz]+sizeStretch)*scale),
							d[idx],d[idx+bs]);
					VertexColor col = MarchingCubes::VertexInterp(c[idx],c[idx+bs],d[idx],d[idx+bs]);
					mesh.vertices.push_back(ver);
					mesh.colors.push_back(Color3b(col.x/COLOR_MULTIPLICATOR,col.y/COLOR_MULTIPLICATOR,col.z/COLOR_MULTIPLICATOR));
					indices[3*idx+2] = runningIndex++;
					vertexIsSet[3*idx+2] = true;
				}
			}
		}
	}

	if(y<oy+size-sizeMin){
		sidetype by = (y-oy)/sizeMin;
		if(x<ox+size-sizeMin){
			sidetype bx = (x-ox)/sizeMin;

			volumetype idx = by*bl+bx;
			int *table = tables[idx];
			if(table){
				for (unsigned int i=0;table[i]!=-1;i+=3) {
					mesh.faces.push_back(indices[3*idx+table[i  ]]);
					mesh.faces.push_back(indices[3*idx+table[i+1]]);
					mesh.faces.push_back(indices[3*idx+table[i+2]]);
				}
			}
		}
	}

	bool loneVertices = false;
	bool wrongIndices = false;
	int lastWrongIndex = -1;
	for(unsigned int i=0;i<brickSize*3;i++) loneVertices |= (vertexIsSet[i]&& !faceIsSet[i]);
	for(unsigned int i=0;i<brickSize*3;i++) {
		wrongIndices |= (!vertexIsSet[i]&& faceIsSet[i]);
		if(!vertexIsSet[i]&& faceIsSet[i]) lastWrongIndex = i;
	}

    //if(loneVertices) fprintf(stderr,"\nERROR: There were lone Vertices at [%i %i %i]",ox,oy,oz);
	//if(wrongIndices) fprintf(stderr,"\nERROR: There were wrong Indices at [%i %i %i]:%i - > [%i %i %i]%i",
			//ox,oy,oz,lastWrongIndex/3,
			//(lastWrongIndex/3)%bl,((lastWrongIndex/3)/bl)%bl,(lastWrongIndex/3)/bs,lastWrongIndex%3);

	delete [] d; delete [] w; delete [] c;
	delete [] indices;
	delete [] vertexIsSet; delete [] faceIsSet;
	delete [] tables;

            
}






void add_brickz_caller(treeinfo info,
                    sidetype ox, sidetype oy, sidetype oz,
                    sidetype size,
                    volumetype lastleaf0, volumetype lastleaf1,
                    const ParentArray &leafParent,
                    const MarchingCubesIndexed &mc,
                    MeshInterleaved *pmesh
        ){
        dim3 block(16, 16);
        const sidetype *leafScale = info.leafScale;
        sidetype sizeMin = std::min(leafScale[lastleaf0],leafScale[lastleaf1]);
        dim3 grid(((size/sizeMin)+ block.x-1)/ block.x , ((size/sizeMin) + block.y-1)/ block.y);
        add_brickz_kernel <<<grid, block>>>(info, ox, oy, oz,size,lastleaf0,lastleaf1, leafParent,mc, pmesh );
} 
