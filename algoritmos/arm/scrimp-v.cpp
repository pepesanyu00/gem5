#include <iostream>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <vector>
#include <algorithm>
#include <random>
#include <string>
#include <sstream>
#include <chrono>
#include <omp.h>
#include <unistd.h> //For getpid(), used to get the pid to generate a unique filename
#include <typeinfo> //To obtain type name as string
#include <array>
#include <assert.h> //Including assert for checking invariants and conditions
//#include <immintrin.h>
#include <arm_sve.h>     // Include SVE intrinsics

#ifdef ENABLE_PARSEC_HOOKS
#include <hooks.h>
#endif 

#ifdef ENABLE_GEM5_ROI
#include <gem5/m5ops.h>
#endif

#define SHUFFLE_DIAGS true
#define DTYPE double        /* DATA TYPE */
#define ITYPE uint64_t /* INDEX TYPE: using long long int so that both double and int are 64 bits (facilitates vectorization) */

/*******************************************************************/ 
#define VDTYPE svfloat64_t
#define VITYPE svuint64_t
#define VMTYPE svbool_t


// Gets the maximum number of 64-bit elements (number of lanes for double)
uint64_t VLMAX = svcntd();

//  Broadcasts a value across a vector register
#define SETZERO_PD() svdup_f64(0.0)


// Creates a double vector with all elements equal to "a"
static inline svfloat64_t set1_pd(double a,svbool_t vl) {
  // Vector where element 'a' will be stored
  VDTYPE inactive = svdup_f64(0.0);
  // The vector is created with value 'a' and predicate to process 'vl' elements
  return svdup_f64_m(inactive,vl,a);

}
#define SET1_PD(a, vl) set1_pd(a,vl)

// Creates a 64-bit integer vector with all elements equal to "a"
static inline svuint64_t set1_epi(double a, svbool_t vl) {
  // Vector where element 'a' will be stored
  VITYPE inactive = svdup_u64(0.0);
  // The vector is created with value 'a' and predicate to process 'vl' elements
  return svdup_u64_m(inactive,vl,a);

}
#define SET1_EPI(a, vl) set1_epi(a,vl)

//  Loads elements into a register (float and int)
#define LOADU_PD(a, vl) svld1_f64(vl, a)
#define LOADU_SI(a, vl) svld1_u64(vl, a)

// Extracts the first value of a vector register and puts it into a float64_t scalar (get), and vice versa (set), takes a float64_t scalar and puts it into the first value of the vector.
// Extracts the first value of an SVE vector of doubles using an inline function
static inline double get_first_pd(svfloat64_t vec) {
    int n = svcntd(); // number of lanes for double
    DTYPE tmp[n];
    // The complete vector is stored; we use the "all active" mask
    svst1_f64(svptrue_b64(), tmp, vec);
    return tmp[0];
}
#define GETFIRST_PD(a) get_first_pd(a)

// Extracts the minimum from a vector register and puts it into a float64_t scalar, variable 'b' is the initial maximum value.
#define REDMIN_PD(a, b, vl) svminv_f64(vl, svmin_f64_m(vl, a, svdup_f64(b)))

// Gets the index of the first true value in a mask, if there are no 1s, returns -1
static inline uint64_t get_first_mask(svbool_t mask, svbool_t vl) {
  svbool_t first_true = svbrkb_z(vl, mask);
  ITYPE index = svcntp_b64(vl, first_true);
  if (index == svcntp_b64(vl,vl)) // If index equals the count of active elements in vl, it means no true was found before end of active elements
    return -1;
  return index;
}
#define GETFIRST_MASK(mask, vl) get_first_mask(mask, vl)

//  Stores elements from a register (float and int) to memory. For unaligned, SVE does not require alignment for most of its instructions
#define STORE_PD(a, b, vl) svst1_f64(vl, a, b)
#define STORE_SI(a, b, vl) svst1_u64(vl, a, b)
//  performs multiply-add of two vectors and stores it in a third one
#define FMADD_PD(a, b, c, vl) svmla_f64_z(vl, c, a, b)
#define FMSUB_PD(a,b,c, vl) svsub_f64_z(vl, svmul_f64_z(vl, a, b), c)
// sum, subtraction, and multiplication of two vectors
#define SUB_PD(a, b, vl) svsub_f64_z(vl, a, b)
#define ADD_PD(a, b, vl) svadd_f64_z(vl, a, b)
#define MUL_PD(a, b, vl) svmul_f64_z(vl, a, b)
#define DIV_PD(a, b, vl) svdiv_f64_z(vl, a, b)
// Compares two vectors element by element (a less than b) and returns a mask with 1s in elements that satisfy the condition
#define CMP_PD_LT(a, b, vl) svcmplt_f64(vl,a, b)
// Compares two vectors element by element (a equal to b - scalar duplicated)
#define CMP_PD_EQ(a, b, vl) svcmpeq_f64(vl ,a, svdup_f64(b))
// Combines two operands using a mask
#define BLEND_EPI(a, b, mask) svsel(mask, b, a)
#define BLEND_PD(a, b, mask) svsel(mask, b, a)
// Stores 64-bit elements in memory, but only those that satisfy the mask (PD for floating point and EPI for integers)
#define MASKSTOREU_PD(mask, a, b) svst1_f64(mask, a, b)
#define MASKSTOREU_EPI(mask, a, b) svst1_u64(mask, a, b)



// Macros for array creation
#define ARRAY_NEW(_type, _var, _elem) _var = new _type[_elem];


#define ARRAY_DEL(_var)      \
  assert(_var != NULL); \
  delete[] _var;

using namespace std;

ITYPE numThreads, exclusionZone;
ITYPE windowSize, tSeriesLength, profileLength, percent_diags;


// Private structures
DTYPE *profile_tmp = NULL;
ITYPE *profileIndex_tmp = NULL;

void preprocess(DTYPE *tSeries, DTYPE *means,
                DTYPE *devs)
{
  DTYPE *ACumSum = new DTYPE[tSeriesLength];
  DTYPE *ASqCumSum = new DTYPE[tSeriesLength];
  DTYPE *ASum = new DTYPE[profileLength];
  DTYPE *ASumSq = new DTYPE[profileLength];
  DTYPE *ASigmaSq = new DTYPE[profileLength];

  ACumSum[0] = tSeries[0];
  ASqCumSum[0] = tSeries[0] * tSeries[0];

  // No need to initialize them because they are overwritten

  // Cumulative sum
  for (ITYPE i = 1; i < tSeriesLength; i++)
  {
    ACumSum[i] = tSeries[i] + ACumSum[i - 1];
    ASqCumSum[i] = tSeries[i] * tSeries[i] + ASqCumSum[i - 1];
  }

  ASum[0] = ACumSum[windowSize - 1];
  ASumSq[0] = ASqCumSum[windowSize - 1];

  for (ITYPE i = 0; i < tSeriesLength - windowSize; i++)
  {
    ASum[i + 1] = ACumSum[windowSize + i] - ACumSum[i];
    ASumSq[i + 1] = ASqCumSum[windowSize + i] - ASqCumSum[i];
  }

  for (ITYPE i = 0; i < profileLength; i++)
  {
    means[i] = (ASum[i] / windowSize);
    ASigmaSq[i] = ASumSq[i] / windowSize - means[i] * means[i];
    devs[i] = sqrt(ASigmaSq[i]);
  }

  delete[] ACumSum;
  delete[] ASqCumSum;
  delete[] ASum;
  delete[] ASumSq;
  delete[] ASigmaSq;
}

void scrimp(DTYPE *tSeries, vector<ITYPE> &idx,
            DTYPE *means, DTYPE *devs, DTYPE *profile,
            ITYPE *profileIndex)
{

#pragma omp parallel // Example of processor binding strategy
  {
    ITYPE my_offset = omp_get_thread_num() * (profileLength);
    svbool_t vlOuter = svptrue_b64(), vlInner = svptrue_b64(), vlRed = svptrue_b64();
    ITYPE Ndiags = (ITYPE)idx.size()*percent_diags/100;

// Go through diagonals (dynamic scheduling)
#pragma omp for schedule(dynamic)
    for (ITYPE ri = 0; ri < Ndiags; ri++)
    {
      ITYPE diag = idx[ri];
      vlOuter = svwhilelt_b64((ITYPE)0,min((ITYPE)VLMAX, profileLength - diag));
      VDTYPE dotProd_v = SETZERO_PD();


      for (ITYPE j = diag; j < windowSize + diag; j++)
      {
        VDTYPE tSeriesj_v = LOADU_PD(&tSeries[j], vlOuter);
        VDTYPE tSeriesMinusDiag_v = SET1_PD(tSeries[j - diag], vlOuter);
        dotProd_v = FMADD_PD(tSeriesj_v, tSeriesMinusDiag_v, dotProd_v, vlOuter);
      }

      // j is the column index, i is the row index of the current distance value in the distance matrix
      ITYPE j = diag;
      ITYPE i = 0;

      VDTYPE distance_v;
      VDTYPE meansj_v = LOADU_PD(&means[j], vlOuter),
             devsj_v = LOADU_PD(&devs[j], vlOuter),
             meansi_v = SET1_PD(means[i], vlOuter),
             devsi_v = SET1_PD(devs[i], vlOuter);
      VDTYPE windowSize_v = SET1_PD((double)windowSize, vlOuter);

      // Evaluate the distance based on the dot product
      VDTYPE prod_devs_v = MUL_PD(devsi_v, devsj_v, vlOuter);
      VDTYPE triple_product_v = MUL_PD(windowSize_v, MUL_PD(meansi_v, meansj_v, vlOuter), vlOuter);
      VDTYPE division_v = DIV_PD(SUB_PD(dotProd_v, triple_product_v, vlOuter), prod_devs_v, vlOuter);
      distance_v = MUL_PD(SET1_PD(2.0, vlOuter), SUB_PD(windowSize_v, division_v, vlOuter), vlOuter);


      // Horizontal search for the maximum
      // Gets the first value of the resulting vector by applying the minimum between the correlation vector and the profile at position i+myoffset
      DTYPE corr_max = REDMIN_PD(distance_v, profile_tmp[i + my_offset], vlOuter);
      VMTYPE mask = CMP_PD_EQ(distance_v, corr_max, vlOuter);
      long index_max = GETFIRST_MASK(mask, vlInner);

      if(index_max != -1) {
        profile_tmp[i + my_offset] = corr_max;
        profileIndex_tmp[i + my_offset] = j + index_max;      
      }

      VDTYPE profilej_v = LOADU_PD(&profile_tmp[j + my_offset], vlOuter);
      mask = CMP_PD_LT(distance_v, profilej_v, vlOuter);
      MASKSTOREU_PD(mask, &profile_tmp[j + my_offset], distance_v);
      MASKSTOREU_EPI(mask, &profileIndex_tmp[j + my_offset],  SET1_EPI(i, vlOuter));

      i = 1;
      for (ITYPE j = diag + 1; j < profileLength; j++)
      {
        vlInner = svwhilelt_b64((ITYPE)0,min((ITYPE)VLMAX, profileLength - j));
        VDTYPE tSeriesj0_v = LOADU_PD(&tSeries[j + windowSize - 1], vlInner),
               tSeriesj1_v = LOADU_PD(&tSeries[j - 1], vlInner),
               tSeriesi0_v = SET1_PD(tSeries[i + windowSize - 1], vlInner),
               tSeriesi1_v = SET1_PD(tSeries[i - 1], vlInner);

        dotProd_v = ADD_PD(FMSUB_PD(tSeriesj0_v, tSeriesi0_v,
                                    MUL_PD(tSeriesj1_v, tSeriesi1_v, vlInner), vlInner),
                           dotProd_v, vlInner);

        meansj_v = LOADU_PD(&means[j], vlInner);
        devsj_v = LOADU_PD(&devs[j], vlInner);
        meansi_v = SET1_PD(means[i], vlInner);
        devsi_v = SET1_PD(devs[i], vlInner);
        triple_product_v = MUL_PD(windowSize_v, MUL_PD(meansi_v, meansj_v, vlInner), vlInner);
        prod_devs_v = MUL_PD(devsi_v, devsj_v, vlInner);
        division_v = DIV_PD(SUB_PD(dotProd_v, triple_product_v, vlInner), prod_devs_v, vlInner);
        distance_v = MUL_PD(SET1_PD(2.0, vlInner), SUB_PD(windowSize_v, division_v, vlInner), vlInner);


        // Gets the first value of the resulting vector by applying the maximum (minimum distance) between the correlation vector and the profile at position i+myoffset
        corr_max = REDMIN_PD(distance_v, profile_tmp[i + my_offset], vlInner);
        mask = CMP_PD_EQ(distance_v, corr_max, vlInner);
        index_max = GETFIRST_MASK(mask, vlInner);
        
        if(index_max != -1) {
          profile_tmp[i + my_offset] = corr_max;
          profileIndex_tmp[i + my_offset] = j + index_max;      
        }

        profilej_v = LOADU_PD(&profile_tmp[j + my_offset], vlInner);
        mask = CMP_PD_LT(distance_v, profilej_v, vlInner);
        MASKSTOREU_PD(mask, &profile_tmp[j + my_offset], distance_v);
        MASKSTOREU_EPI(mask, &profileIndex_tmp[j + my_offset], SET1_EPI(i, vlInner));

        i++;
      }

    } //'pragma omp for' places here a barrier unless 'nowait' is specified

// Reduction
#pragma omp for schedule(static)
    for (ITYPE colum = 0; colum < profileLength; colum += VLMAX)
    {
      vlRed = svwhilelt_b64((ITYPE)0,min((ITYPE)VLMAX, profileLength - colum));
      VDTYPE min_dist_v = SET1_PD(numeric_limits<DTYPE>::infinity(), vlRed);
      VITYPE min_indices_v = SET1_EPI(1, vlRed);
      for (ITYPE th = 0; th < numThreads; th++)
      {
        VDTYPE profile_tmp_v = LOADU_PD(&profile_tmp[colum + (th * profileLength)], vlRed);
        VITYPE profileIndex_tmp_v = LOADU_SI(&profileIndex_tmp[colum + (th * profileLength)], vlRed);
        VMTYPE mask = CMP_PD_LT(profile_tmp_v, min_dist_v, vlRed);
        min_indices_v = BLEND_EPI(min_indices_v, profileIndex_tmp_v, mask); // Update indices with mask
        min_dist_v = BLEND_PD(min_dist_v, profile_tmp_v, mask);             // Update distances with mask
      }
      STORE_PD(&profile[colum], min_dist_v, vlRed);
      STORE_SI(&profileIndex[colum], min_indices_v, vlRed);
    }
  }
}

int main(int argc, char *argv[])
{
  try
  {
    chrono::steady_clock::time_point tstart, tprogstart, tend;
    chrono::duration<double> telapsed;

    if (argc != 6)
    {
      cout << "[ERROR] usage: ./scrimp input_file win_size num_threads percent_diags out_directory" << endl;
      return 0;
    }

    windowSize = atoi(argv[2]);
    numThreads = atoi(argv[3]);
    percent_diags = atoi(argv[4]);
    string outdir = argv[5];
    exclusionZone = (ITYPE)(windowSize * 0.25);
    omp_set_num_threads(numThreads);

    string inputfilename = argv[1];
    string alg = argv[0];
    alg = alg.substr(2);
    stringstream tmp;
    tmp << outdir << alg.substr(alg.rfind('/') +1) << "_" << inputfilename.substr(inputfilename.rfind('/') + 1, inputfilename.size() - 4 - inputfilename.rfind('/') - 1) << "_w" << windowSize << "_t" << numThreads << "_pdiags" << percent_diags << "_" << getpid() << ".csv";
    string outfilename = tmp.str();

    cout << endl;
    cout << "############################################################" << endl;
    cout << "///////////////////////// SCRIMP ///////////////////////////" << endl;
    cout << "############################################################" << endl;
    cout << endl;
    cout << "[>>] Reading File: " << inputfilename << "..." << endl;

    tstart = chrono::steady_clock::now();

    fstream tSeriesFile(inputfilename, ios_base::in);
    tSeriesLength = 0;
    cout << "[>>] Counting lines ... " << endl;
    string line;
    while (getline(tSeriesFile, line))
      tSeriesLength++;

    tend = chrono::steady_clock::now();
    telapsed = tend - tstart;
    cout << "[OK] Lines: " << tSeriesLength << " Time: " << telapsed.count() << "s." << endl;

    cout << "[>>] Reading values..." << endl;
    tstart = chrono::steady_clock::now();
    tprogstart = tstart;
    tSeriesFile.clear();
    tSeriesFile.seekg(tSeriesFile.beg);
    DTYPE *tSeries = NULL;
    DTYPE tempval, tSeriesMin = numeric_limits<DTYPE>::infinity(),
                   tSeriesMax = -numeric_limits<double>::infinity();
    ARRAY_NEW(DTYPE, tSeries, tSeriesLength);

    for (int i = 0; tSeriesFile >> tempval; i++)
    {
      tSeries[i] = tempval;
      if (tempval < tSeriesMin)
        tSeriesMin = tempval;
      if (tempval > tSeriesMax)
        tSeriesMax = tempval;
    }
    tSeriesFile.close();
    for (ITYPE i = tSeriesLength; i < tSeriesLength; i++)
      tSeries[i] = numeric_limits<DTYPE>::quiet_NaN();

    tend = chrono::steady_clock::now();
    telapsed = tend - tstart;
    cout << "[OK] Read File Time: " << setprecision(numeric_limits<double>::digits10 + 2) << telapsed.count() << " seconds." << endl;

    profileLength = tSeriesLength - windowSize + 1;
    DTYPE *means = NULL, *devs = NULL, *profile = NULL;
    vector<ITYPE> diags;
    ITYPE *profileIndex = NULL;
    ARRAY_NEW(DTYPE, means, profileLength);
    ARRAY_NEW(DTYPE, devs, profileLength);
    ARRAY_NEW(DTYPE, profile, profileLength);
    ARRAY_NEW(ITYPE, profileIndex, profileLength);

    ARRAY_NEW(DTYPE, profile_tmp, profileLength * numThreads);
    ARRAY_NEW(ITYPE, profileIndex_tmp, profileLength * numThreads);
    for (ITYPE i = 0; i < (profileLength * numThreads); i++)
      profile_tmp[i] = numeric_limits<DTYPE>::infinity();

    cout << endl;
    cout << "------------------------------------------------------------" << endl;
    cout << "************************** INFO ****************************" << endl;
    cout << endl;
    cout << " Series/MP data type: " << typeid(tSeries[0]).name() << "(" << sizeof(tSeries[0]) << "B)" << endl;
    cout << " Index data type:     " << typeid(profileIndex[0]).name() << "(" << sizeof(profileIndex[0]) << "B)" << endl;
    cout << " Time series length:  " << tSeriesLength << endl;
    cout << " Window size:         " << windowSize << endl;
    cout << " Time series min:     " << tSeriesMin << endl;
    cout << " Time series max:     " << tSeriesMax << endl;
    cout << " Number of threads:   " << numThreads << endl;
    cout << " Exclusion zone:      " << exclusionZone << endl;
    cout << " Profile length:      " << profileLength << endl;
    cout << "------------------------------------------------------------" << endl;
    cout << endl;


    cout << "[>>] Preprocessing..." << endl;
    tstart = chrono::steady_clock::now();

    preprocess(tSeries, means, devs);

    tend = chrono::steady_clock::now();
    telapsed = tend - tstart;
    cout << "[OK] Preprocessing Time:         " << setprecision(numeric_limits<double>::digits10 + 2) << telapsed.count() << " seconds." << endl;
    
    diags.clear();
    for (ITYPE i = exclusionZone + 1; i < profileLength; i+= VLMAX)
      diags.push_back(i);

    if (SHUFFLE_DIAGS){
      mt19937 g(0);
      shuffle(diags.begin(), diags.end(), g);
    }

    cout << "[>>] Executing SCRIMP..." << endl;
    tstart = chrono::steady_clock::now();

    #ifdef ENABLE_PARSEC_HOOKS
      __parsec_roi_begin();
    #endif

    #ifdef ENABLE_GEM5_ROI
    m5_work_begin(0,0);
    #endif

    scrimp(tSeries, diags, means, devs, profile, profileIndex);

    #ifdef ENABLE_GEM5_ROI
    m5_work_end(0,0);
    #endif
    
    #ifdef ENABLE_PARSEC_HOOKS
      __parsec_roi_end();
    #endif

    tend = chrono::steady_clock::now();
    telapsed = tend - tstart;
    cout << "[OK] SCRIMP Time:             " << setprecision(numeric_limits<DTYPE>::digits10 + 2) << telapsed.count() << " seconds." << endl;

    cout << "[>>] Saving Profile..." << endl;
    tstart = chrono::steady_clock::now();

    fstream statsFile(outfilename, ios_base::out);
    statsFile << "# Time (s)" << endl;
    statsFile << setprecision(9) << telapsed.count() << endl;
    statsFile << "# Profile Length" << endl;
    statsFile << profileLength << endl;
    statsFile << "# i,tseries,profile,index" << endl;
    for (ITYPE i = 0; i < profileLength; i++)
    {
      statsFile << i << "," << tSeries[i] << "," << (DTYPE)sqrt(profile[i]) << "," << profileIndex[i] << endl;
    }
    statsFile.close();

    ARRAY_DEL(tSeries);
    ARRAY_DEL(means);
    ARRAY_DEL(devs);
    ARRAY_DEL(profile);
    ARRAY_DEL(profileIndex);
    ARRAY_DEL(profile_tmp);
    ARRAY_DEL(profileIndex_tmp);

    tend = chrono::steady_clock::now();
    telapsed = tend - tstart;
    cout << "[OK] Saving Profile Time:       " << setprecision(numeric_limits<DTYPE>::digits10 + 2) << telapsed.count() << " seconds." << endl;

    telapsed = tend - tprogstart;
    cout << "[OK] Total Time:              " << setprecision(numeric_limits<DTYPE>::digits10 + 2) << telapsed.count() << " seconds." << endl;
    cout << "[OK] Filename: " << outfilename << endl;
    cout << endl;
  }
  catch (exception &e)
  {
    cout << "Exception: " << e.what() << endl;
  }
}
