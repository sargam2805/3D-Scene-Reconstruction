
#include "mesh1.hpp"

#include <set>
#include <map>
#include <list>
#include <iostream>
#include "boost/tuple/tuple.hpp"
#include <boost/tuple/tuple_comparison.hpp>
#include <math.h>

#ifdef __CUDACC__
#define CUDA_HOSTDEV __device__
#else
#define CUDA_HOSTDEV
#endif

CUDA_HOSTDEV
VertexColor MarchingCubes::VertexInterp(VertexColor p1,VertexColor p2,float valp1,float valp2, float isolevel)
{
   float mu;
   VertexColor p;
   if (fabs(isolevel-valp1) < DISTZEROEPSILON){
     return(p1);
   }
   if (fabs(isolevel-valp2) < DISTZEROEPSILON){
      return(p2);
   }
   if (fabs(valp1-valp2) < DISTZEROEPSILON){
      return(p1);
   }
   mu = (isolevel - valp1) / (valp2 - valp1);
   float px = ((float)p1.x + mu * ((float)p2.x - (float)p1.x));
   float py = ((float)p1.y + mu * ((float)p2.y - (float)p1.y));
   float pz = ((float)p1.z + mu * ((float)p2.z - (float)p1.z));
   p.x = px<=255.0f*COLOR_MULTIPLICATOR ? (px>=0.0f ? (colortype)px : 0) : 255*COLOR_MULTIPLICATOR;
   p.y = py<=255.0f*COLOR_MULTIPLICATOR ? (py>=0.0f ? (colortype)py : 0) : 255*COLOR_MULTIPLICATOR;
   p.z = pz<=255.0f*COLOR_MULTIPLICATOR ? (pz>=0.0f ? (colortype)pz : 0) : 255*COLOR_MULTIPLICATOR;

   return(p);
}

CUDA_HOSTDEV
int MarchingCubes::getCubeIndex(float d000, float d001, float d010, float d011,
		float d100, float d101, float d110, float d111,
		float isolevel)
{

	int cubeindex = 0;
	if (d000 < isolevel) cubeindex |= 1;
	if (d001 < isolevel) cubeindex |= 2;
	if (d010 < isolevel) cubeindex |= 4;
	if (d011 < isolevel) cubeindex |= 8;
	if (d100 < isolevel) cubeindex |= 16;
	if (d101 < isolevel) cubeindex |= 32;
	if (d110 < isolevel) cubeindex |= 64;
	if (d111 < isolevel) cubeindex |= 128;
 return cubeindex;
}

CUDA_HOSTDEV
bool weightInfluence(float minWeight,
weighttype w0, weighttype w1, weighttype w2, weighttype w3,
weighttype w4, weighttype w5, weighttype w6, weighttype w7
)
{
	bool result = w0>=minWeight

//			&& w1>0.0f
//			&& w2>0.0f
//			&& w3>0.0f
//			&& w4>0.0f
//			&& w5>0.0f
//			&& w6>0.0f
//			&& w7>0.0f

//			&& w1>=minWeight
//			&& w2>=minWeight
//			&& w3>=minWeight
//			&& w4>=minWeight
//			&& w5>=minWeight
//			&& w6>=minWeight
//			&& w7>=minWeight
			;

//	if(!result) fprintf(stderr," w(%f %f %f %f %f %f %f %f)",w0,w1,w2,w3,w4,w5,w6,w7);
	return result;
//	return true;
}

CUDA_HOSTDEV
volumetype &OwnParentArray_::operator[](size_t pos){
	if(pos < _size){
		return _array[pos];
	}
	else{
		//fprintf(stderr,"\nERROR: Wrong Index in MeshCellNeighborhood Array! %li >= %li",pos,_size);
		return _dummy;
	}
}

CUDA_HOSTDEV
const volumetype &OwnParentArray_::operator[](size_t pos) const {
	if(pos < _size){
		return _array[pos];
	}
	else{
		//fprintf(stderr,"\nERROR: Wrong Index in MeshCellNeighborhood Array! %li >= %li",pos,_size);
		return _dummy;
	}
}

CUDA_HOSTDEV
Color3b::Color3b(uchar r_p, uchar g_p, uchar b_p):r(r_p),g(g_p),b(b_p){}

