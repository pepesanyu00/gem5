#include <iostream>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <algorithm>
#include <vector>
#include <string>
#include <sstream>
#include <chrono>
#include <random>
#include <omp.h>
#include <unistd.h>      // For getpid()
#include <typeinfo>      // For typeid().name()
#include <array>
#include <cassert>
#include <arm_sve.h>     // Include SVE intrinsics

#ifdef ENABLE_PARSEC_HOOKS
#include <hooks.h>
#endif 

#ifdef ENABLE_GEM5_ROI
#include <gem5/m5ops.h>
#endif

#define SHUFFLE_DIAGS true
#define DTYPE double              /* DATA TYPE */
#define ITYPE uint64_t       /* INDEX TYPE */

// SVE types: for double, 64-bit unsigned integers, and masks
#define VDTYPE svfloat64_t
#define VITYPE svuint64_t
#define VMTYPE svbool_t

// ================================================================
// Auxiliary functions for SVE

// Gets the maximum number of 64-bit elements (number of lanes for double)
uint64_t VLMAX = svcntd();

// Creates a double vector with all elements equal to 0.0
#define SETZERO_PD() svdup_f64(0.0)

// Creates a double vector with all elements equal to "a"
static inline svfloat64_t set1_pd(double a, svbool_t vl) {
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

// Extracts the first value of an SVE vector of doubles using an inline function
static inline double get_first_pd(svfloat64_t vec) {
    int n = svcntd(); // number of lanes for double
    DTYPE tmp[n];
    // The complete vector is stored; we use the "all active" mask
    svst1_f64(svptrue_b64(), tmp, vec);
    return tmp[0];
}
#define GETFIRST_PD(a) get_first_pd(a)



// Performs a horizontal maximum reduction; combines vector "a" with a duplicated vector of "b" and reduces
#define REDMAX_PD(a, b, vl) svmaxv_f64(vl, svmax_f64_m(vl, a, svdup_f64(b)))

// Auxiliary function to extract the first active index (value 1) from an SVE mask.
// Since SVE does not offer direct mask extraction, the mask is stored in an array and traversed. (Note: svbrkb_z and svcntp_b64 provide a more direct way)
static inline uint64_t get_first_mask(svbool_t mask, svbool_t vl) {
  svbool_t first_true = svbrkb_z(vl, mask); // Break to first true element
  ITYPE index = svcntp_b64(vl, first_true); // Count true elements before or at the first break
  if (index == svcntp_b64(vl,vl)) // If count equals total active elements in vl, no true found
    return -1;
  return index;
}
#define GETFIRST_MASK(mask, vl) get_first_mask(mask, vl)

// Loads into an SVE vector of doubles (unaligned load) using a mask that activates "vl" elements
#define LOADU_PD(a, vl) svld1_f64(vl, a)

// Loads into an SVE vector of 64-bit integers
#define LOADU_SI(a, vl) svld1_u64(vl, a)

// Stores to memory from an SVE vector of doubles
#define STORE_PD(a, b, vl) svst1_f64(vl, a, b)

// Stores to memory from an SVE vector of 64-bit integers
#define STORE_SI(a, b, vl) svst1_u64(vl, a, b)

// Performs the fused multiply-add operation: calculates (a * b + c)
#define FMADD_PD(a, b, c, vl) svmla_f64_z(vl, c, a, b)

// Element-wise sum, subtraction, and multiplication of two double vectors
#define ADD_PD(a, b, vl) svadd_f64_z(vl, a, b)
#define SUB_PD(a, b, vl) svsub_f64_z(vl, a, b)
#define MUL_PD(a, b, vl) svmul_f64_z(vl, a, b)

// Element-wise comparison of two double vectors: greater than
#define CMP_PD_GT(a, b, vl) svcmpgt_f64(vl,a, b)

// Element-wise comparison: equality with a scalar (scalar "b" is duplicated)
#define CMP_PD_EQ(a, b, vl) svcmpeq_f64(vl ,a, svdup_f64(b))

// Combines two vectors according to a mask: for integers
#define BLEND_EPI(a, b, mask) svsel(mask, b, a)
// Combines two vectors according to a mask: for doubles
#define BLEND_PD(a, b, mask) svsel(mask, b, a)

// Stores to memory only the elements of a double vector that satisfy the mask
#define MASKSTOREU_PD(mask, a, b) svst1_f64(mask, a, b)
// Stores to memory only the elements of an integer vector that satisfy the mask
#define MASKSTOREU_EPI(mask, a, b) svst1_u64(mask, a, b)


// Macros for array creation and deletion (used the same way)
#define ARRAY_NEW(_type, _var, _elem) _var = new _type[_elem];
#define ARRAY_DEL(_var)      \
  assert(_var != NULL);      \
  delete[] _var;

using namespace std;

// ------------------------------------------------------------------
// Global variables
ITYPE numThreads, exclusionZone, windowSize, tSeriesLength, profileLength, percent_diags;

// Private temporary variables
DTYPE *profile_tmp = NULL;
ITYPE *profileIndex_tmp = NULL;

// ------------------------------------------------------------------
// Preprocessing function: calculates statistics needed for SCAMP
void preprocess(DTYPE *tSeries, DTYPE *means, DTYPE *norms, DTYPE *df, DTYPE *dg, ITYPE tSeriesLength)
{
  vector<DTYPE> prefix_sum(tSeriesLength);
  vector<DTYPE> prefix_sum_sq(tSeriesLength);

  prefix_sum[0] = tSeries[0];
  prefix_sum_sq[0] = tSeries[0] * tSeries[0];
  for (ITYPE i = 1; i < tSeriesLength; ++i) {
    prefix_sum[i] = tSeries[i] + prefix_sum[i - 1];
    prefix_sum_sq[i] = tSeries[i] * tSeries[i] + prefix_sum_sq[i - 1];
  }

  means[0] = prefix_sum[windowSize - 1] / static_cast<DTYPE>(windowSize);
  for (ITYPE i = 1; i < profileLength; ++i)
    means[i] = (prefix_sum[i + windowSize - 1] - prefix_sum[i - 1]) / static_cast<DTYPE>(windowSize);

  DTYPE sum = 0;
  for (ITYPE i = 0; i < windowSize; ++i) {
    DTYPE val = tSeries[i] - means[0];
    sum += val * val;
  }
  norms[0] = sum;

  for (ITYPE i = 1; i < profileLength; ++i)
    norms[i] = norms[i - 1] + ((tSeries[i - 1] - means[i - 1]) + (tSeries[i + windowSize - 1] - means[i])) *
                              (tSeries[i + windowSize - 1] - tSeries[i - 1]);
  for (ITYPE i = 0; i < profileLength; ++i)
    norms[i] = 1.0 / sqrt(norms[i]);

  for (ITYPE i = 0; i < profileLength - 1; ++i) {
    df[i] = (tSeries[i + windowSize] - tSeries[i]) / 2.0;
    dg[i] = (tSeries[i + windowSize] - means[i + 1]) + (tSeries[i] - means[i]);
  }
}

// ------------------------------------------------------------------
// SCAMP function: vectorized calculation with SVE intrinsics
void scamp(DTYPE *tSeries, vector<ITYPE> &diags, DTYPE *means, DTYPE *norms, DTYPE *df, DTYPE *dg, DTYPE *profile, ITYPE *profileIndex)
{
#pragma omp parallel
  {
    ITYPE my_offset = omp_get_thread_num() * profileLength;
    svbool_t vlOuter = svptrue_b64(), vlInner = svptrue_b64(), vlRed = svptrue_b64(); // Predicates, initialized to all true
    ITYPE Ndiags = (ITYPE)diags.size() * percent_diags / 100;

    // Iterate over diagonals (dynamically scheduled)
#pragma omp for schedule(dynamic)
    for (ITYPE ri = 0; ri < Ndiags; ri++) {
      ITYPE diag = diags[ri];
      vlOuter = svwhilelt_b64((ITYPE)0,min((ITYPE)VLMAX, profileLength - diag)); // Predicate for outer loop operations
      
      VDTYPE meansWinDiag_v   = LOADU_PD(&means[diag], vlOuter);
      VDTYPE meansWin0_v      = SET1_PD(means[0], vlOuter);      
      
      VDTYPE covariance_v = SETZERO_PD();
      for (ITYPE i = 0; i < windowSize; i++) {
        VDTYPE tSeriesWinDiag_v = LOADU_PD(&tSeries[diag + i], vlOuter);
        VDTYPE tSeriesWin0_v    = SET1_PD(tSeries[i], vlOuter);
        covariance_v = FMADD_PD( SUB_PD(tSeriesWinDiag_v, meansWinDiag_v, vlOuter),
                                 SUB_PD(tSeriesWin0_v, meansWin0_v, vlOuter),
                                 covariance_v, vlOuter);
      }

      ITYPE i = 0;
      // j is initialized with diag
      ITYPE j = diag;
      VDTYPE normsi_v = SET1_PD(norms[i], vlOuter);
      VDTYPE normsj_v = LOADU_PD(&norms[j], vlOuter);
      VDTYPE correlation_v = MUL_PD( MUL_PD(covariance_v, normsi_v, vlOuter), normsj_v, vlOuter);

      // // Horizontal reduction: find the maximum
      DTYPE corr_max = REDMAX_PD(correlation_v, profile_tmp[i + my_offset], vlOuter);
      VMTYPE mask = CMP_PD_EQ(correlation_v, corr_max, vlOuter);
      long index_max = GETFIRST_MASK(mask, vlOuter); // Get index of first max

      if(index_max != -1) {
        profile_tmp[i + my_offset] = corr_max;
        profileIndex_tmp[i + my_offset] = j + index_max;      
      }

      VDTYPE profilej_v = LOADU_PD(&profile_tmp[j + my_offset], vlOuter);
      mask = CMP_PD_GT(correlation_v, profilej_v, vlOuter);
      MASKSTOREU_PD(mask, &profile_tmp[j + my_offset], correlation_v);
      MASKSTOREU_EPI(mask, &profileIndex_tmp[j + my_offset], SET1_EPI(i, vlOuter));

      i = 1;
      for (ITYPE j = diag + 1; j < profileLength; j++) {
        vlInner = svwhilelt_b64((ITYPE)0,min((ITYPE)VLMAX, profileLength - j)); // Predicate for inner loop operations
        VDTYPE dfj_v = LOADU_PD(&df[j - 1], vlInner);
        VDTYPE dgj_v = LOADU_PD(&dg[j - 1], vlInner);
        VDTYPE dfi_v = SET1_PD(df[i - 1], vlInner);
        VDTYPE dgi_v = SET1_PD(dg[i - 1], vlInner);
        covariance_v = ADD_PD(covariance_v,
                              FMADD_PD(dfi_v, dgj_v, MUL_PD(dfj_v, dgi_v, vlInner), vlInner),
                              vlInner);
        normsi_v = SET1_PD(norms[i], vlInner);
        normsj_v = LOADU_PD(&norms[j], vlInner);
        correlation_v = MUL_PD( MUL_PD(covariance_v, normsi_v, vlInner), normsj_v, vlInner);

         corr_max = REDMAX_PD(correlation_v, profile_tmp[i + my_offset], vlInner);
         mask = CMP_PD_EQ(correlation_v, corr_max, vlInner);
         index_max = GETFIRST_MASK(mask, vlInner);
         
         if(index_max != -1) {
           profile_tmp[i + my_offset] = corr_max;
           profileIndex_tmp[i + my_offset] = j + index_max;      
         }

        profilej_v = LOADU_PD(&profile_tmp[j + my_offset], vlInner);
        mask = CMP_PD_GT(correlation_v, profilej_v, vlInner); // Compare correlation with existing profile value
        MASKSTOREU_PD(mask, &profile_tmp[j + my_offset], correlation_v); // Store if greater
        MASKSTOREU_EPI(mask, &profileIndex_tmp[j + my_offset], SET1_EPI(i, vlInner)); // Store index if greater

        i++;
      }
    } // End of omp for (implicit barrier)

    // Final reduction in parallel: iterate over columns in VLMAX steps
#pragma omp for schedule(static)
    for (ITYPE colum = 0; colum < profileLength; colum += VLMAX) {
      vlRed = svwhilelt_b64((ITYPE)0,min((ITYPE)VLMAX, profileLength - colum)); // Predicate for reduction
      VDTYPE max_corr_v = SET1_PD(-numeric_limits<DTYPE>::infinity(), vlRed);
      VITYPE max_indices_v = SET1_EPI(-1, vlRed); // Initialize with invalid index
      for (ITYPE th = 0; th < numThreads; th++) {
        VDTYPE profile_tmp_v = LOADU_PD(&profile_tmp[colum + (th * profileLength)], vlRed);
        VITYPE profileIndex_tmp_v = LOADU_SI(&profileIndex_tmp[colum + (th * profileLength)], vlRed);
        VMTYPE mask = CMP_PD_GT(profile_tmp_v, max_corr_v, vlRed); // Find max correlation among threads
        max_indices_v = BLEND_EPI(max_indices_v, profileIndex_tmp_v, mask); // Update index based on max
        max_corr_v = BLEND_PD(max_corr_v, profile_tmp_v, mask); // Update max correlation
      }
      STORE_PD(&profile[colum], max_corr_v, vlRed); // Store final max correlation
      STORE_SI(&profileIndex[colum], max_indices_v, vlRed); // Store final index
    }
  }
}

// ------------------------------------------------------------------
// Main function
int main(int argc, char *argv[])
{
  try {
    chrono::steady_clock::time_point tstart, tend;
    chrono::duration<double> telapsed;

    if (argc != 6) {
      cout << "[ERROR] usage: ./scamp input_file win_size num_threads percent_diags out_directory" << endl;
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
    tmp << outdir << alg.substr(alg.rfind('/') +1) << "_" 
        << inputfilename.substr(inputfilename.rfind('/') + 1, inputfilename.size() - 4 - inputfilename.rfind('/') - 1)
        << "_w" << windowSize << "_t" << numThreads << "_pdiags" << percent_diags 
        << "_" << getpid() << ".csv";
    string outfilename = tmp.str();
    cout << "VLMAX" << VLMAX << endl;
    cout << endl;
    cout << "############################################################" << endl;
    cout << "///////////////////////// SCAMP ////////////////////////////" << endl;
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

    cout << "[>>] Reading values ... " << endl;
    tstart = chrono::steady_clock::now();
    DTYPE *tSeries = NULL;
    ARRAY_NEW(DTYPE, tSeries, tSeriesLength);
    tSeriesFile.clear();
    tSeriesFile.seekg(tSeriesFile.beg);
    DTYPE tempval, tSeriesMin = numeric_limits<DTYPE>::infinity(), tSeriesMax = -numeric_limits<DTYPE>::infinity();
    for (int i = 0; tSeriesFile >> tempval; i++) {
      tSeries[i] = tempval;
      if (tempval < tSeriesMin)
        tSeriesMin = tempval;
      if (tempval > tSeriesMax)
        tSeriesMax = tempval;
    }
    tSeriesFile.close();
    tend = chrono::steady_clock::now();
    telapsed = tend - tstart;

    profileLength = tSeriesLength - windowSize + 1;

    DTYPE *norms = NULL, *means = NULL, *df = NULL, *dg = NULL, *profile = NULL;
    ITYPE *profileIndex = NULL;
    vector<ITYPE> diags;
    ARRAY_NEW(DTYPE, norms, profileLength);
    ARRAY_NEW(DTYPE, means, profileLength);
    ARRAY_NEW(DTYPE, df, profileLength);
    ARRAY_NEW(DTYPE, dg, profileLength);
    ARRAY_NEW(DTYPE, profile, profileLength);
    ARRAY_NEW(ITYPE, profileIndex, profileLength);
    ARRAY_NEW(DTYPE, profile_tmp, profileLength * numThreads);
    ARRAY_NEW(ITYPE, profileIndex_tmp, profileLength * numThreads);

    for (ITYPE i = 0; i < (profileLength * numThreads); i++)
      profile_tmp[i] = -numeric_limits<DTYPE>::infinity();

    cout << "[OK] Time: " << telapsed.count() << "s." << endl;
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
    preprocess(tSeries, means, norms, df, dg, tSeriesLength);
    tend = chrono::steady_clock::now();
    telapsed = tend - tstart;
    cout << "[OK] Preprocessing Time:         " << setprecision(numeric_limits<double>::digits10 + 2) << telapsed.count() << " seconds." << endl;

    diags.clear();
    for (ITYPE i = exclusionZone + 1; i < profileLength; i+=VLMAX)
      diags.push_back(i);
    if (SHUFFLE_DIAGS) {
      //random_device rd;
      mt19937 g(0);
      shuffle(diags.begin(), diags.end(), g);
    }
    
    cout << "[>>] Executing SCAMP..." << endl;
    tstart = chrono::steady_clock::now();

#ifdef ENABLE_PARSEC_HOOKS
    __parsec_roi_begin();
#endif


#ifdef ENABLE_GEM5_ROI
    m5_work_begin(0,0);
#endif

    scamp(tSeries, diags, means, norms, df, dg, profile, profileIndex);
    
#ifdef ENABLE_GEM5_ROI
    m5_work_end(0,0);
#endif
#ifdef ENABLE_PARSEC_HOOKS
    __parsec_roi_end();
#endif

    tend = chrono::steady_clock::now();
    telapsed = tend - tstart;
    cout << "[OK] SCAMP Time:              " << setprecision(numeric_limits<double>::digits10 + 2) << telapsed.count() << " seconds." << endl;

    cout << "[>>] Saving result: " << outfilename << " ..." << endl;
    fstream statsFile(outfilename, ios_base::out);
    statsFile << "# Time (s)" << endl;
    statsFile << setprecision(9) << telapsed.count() << endl;
    statsFile << "# Profile Length" << endl;
    statsFile << profileLength << endl;
    statsFile << "# i,tseries,profile,index" << endl;
    for (ITYPE i = 0; i < profileLength; i++)
      statsFile << i << "," << tSeries[i] << ","  << (DTYPE)sqrt(2 * windowSize * (1 - profile[i])) << "," << profileIndex[i] << endl;
    statsFile.close();

    cout << endl;

    ARRAY_DEL(tSeries);
    ARRAY_DEL(norms);
    ARRAY_DEL(means);
    ARRAY_DEL(df);
    ARRAY_DEL(dg);
    ARRAY_DEL(profile);
    ARRAY_DEL(profileIndex);
    ARRAY_DEL(profile_tmp);
    ARRAY_DEL(profileIndex_tmp);
  }
  catch (exception &e) {
    cout << "Exception: " << e.what() << endl;
  }
}

