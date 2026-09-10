#include <iostream>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <limits>
#include <vector>
#include <string>
#include <random>
#include <sstream>
#include <chrono>
#include <assert.h>
#include <omp.h>
#include <unistd.h> //For getpid(), used to get the pid to generate a unique filename
#include <typeinfo> //To obtain type name as string

#ifdef ENABLE_PARSEC_HOOKS
#include <hooks.h>
#endif 

#ifdef ENABLE_GEM5_ROI
#include <gem5/m5ops.h>
#endif

#define SHUFFLE_DIAGS true
#define DTYPE double        /* DATA TYPE */
#define ITYPE long long int /* INDEX TYPE */

using namespace std;

ITYPE numThreads, exclusionZone, windowSize, tSeriesLength, profileLength, percent_diags;

// Private structures
DTYPE *profile_tmp;
ITYPE *profileIndex_tmp;

void print_binding_info()
{
  int my_place = omp_get_place_num();
  int place_num_procs = omp_get_place_num_procs(my_place);

  string prtstr("Thread " + to_string(omp_get_thread_num()) + " bound to place " + to_string(my_place) + " which consists of " + to_string(place_num_procs) + " processors.");

  cout << prtstr << endl;
  /*int *place_processors = new int[place_num_procs];
  omp_get_place_proc_ids(my_place, place_processors);

  for (int i = 0; i < place_num_procs; i++)
    cout << place_processors[i] << " ";
  cout << endl;

  delete[] place_processors;*/
}

// Computes all required statistics for SCAMP, populating info with these values
void preprocess(vector<DTYPE> &tSeries, vector<DTYPE> &means, vector<DTYPE> &norms,
                vector<DTYPE> &df, vector<DTYPE> &dg)
{

  vector<DTYPE> prefix_sum(tSeries.size());
  vector<DTYPE> prefix_sum_sq(tSeries.size());

  // Calculates prefix sum and square sum vectors
  prefix_sum[0] = tSeries[0];
  prefix_sum_sq[0] = tSeries[0] * tSeries[0];
  for (ITYPE i = 1; i < tSeriesLength; ++i)
  {
    prefix_sum[i] = tSeries[i] + prefix_sum[i - 1];
    prefix_sum_sq[i] = tSeries[i] * tSeries[i] + prefix_sum_sq[i - 1];
  }

  // Prefix sum value is used to calculate mean value of a given window, taking last value
  // of the window minus the first one and dividing by window size
  means[0] = prefix_sum[windowSize - 1] / static_cast<DTYPE>(windowSize);
  for (ITYPE i = 1; i < profileLength; ++i)
    means[i] = (prefix_sum[i + windowSize - 1] - prefix_sum[i - 1]) / static_cast<DTYPE>(windowSize);

  DTYPE sum = 0;
  for (ITYPE i = 0; i < windowSize; ++i)
  {
    DTYPE val = tSeries[i] - means[0];
    sum += val * val;
  }
  norms[0] = sum;

  // Calculates L2-norms (euclidean norm, euclidean distance)
  for (ITYPE i = 1; i < profileLength; ++i)
    norms[i] = norms[i - 1] + ((tSeries[i - 1] - means[i - 1]) + (tSeries[i + windowSize - 1] - means[i])) *
                                  (tSeries[i + windowSize - 1] - tSeries[i - 1]);
  for (ITYPE i = 0; i < profileLength; ++i)
    norms[i] = 1.0 / sqrt(norms[i]);

  // Calculates df and dg vectors
  for (ITYPE i = 0; i < profileLength - 1; ++i)
  {
    df[i] = (tSeries[i + windowSize] - tSeries[i]) / 2.0;
    dg[i] = (tSeries[i + windowSize] - means[i + 1]) + (tSeries[i] - means[i]);
  }
}

void scamp(vector<DTYPE> &tSeries, vector<ITYPE> &diags, vector<DTYPE> &means, vector<DTYPE> &norms,
           vector<DTYPE> &df, vector<DTYPE> &dg, vector<DTYPE> &profile, vector<ITYPE> &profileIndex)
{

#pragma omp parallel //proc_bind(spread)
  {
    // Suppossing ITYPE as uint32_t (we could index series up to 4G elements), to index profile_tmp we need more bits (uint64_t)
    uint64_t my_offset = omp_get_thread_num() * profileLength;
    DTYPE covariance, correlation;
    print_binding_info();
    ITYPE Ndiags = (ITYPE)diags.size()*percent_diags/100;

// Go through diagonals (dynamic)
#pragma omp for schedule(dynamic)
    for (ITYPE ri = 0; ri < Ndiags; ri++)
    {
      // Select a diagonal
      ITYPE diag = diags[ri];
      covariance = 0;

      for (ITYPE i = 0; i < windowSize; i++)
        covariance += ((tSeries[diag + i] - means[diag]) * (tSeries[i] - means[0]));

      ITYPE i = 0;
      ITYPE j = diag;

      correlation = covariance * norms[i] * norms[j];

      if (correlation > profile_tmp[i + my_offset])
      {
        profile_tmp[i + my_offset] = correlation;
        profileIndex_tmp[i + my_offset] = j;
      }
      if (correlation > profile_tmp[j + my_offset])
      {
        profile_tmp[j + my_offset] = correlation;
        profileIndex_tmp[j + my_offset] = i;
      }

      i = 1;

      for (ITYPE j = diag + 1; j < profileLength; j++)
      {
        covariance += (df[i - 1] * dg[j - 1] + df[j - 1] * dg[i - 1]);
        correlation = covariance * norms[i] * norms[j];

        if (correlation > profile_tmp[i + my_offset])
        {
          profile_tmp[i + my_offset] = correlation;
          profileIndex_tmp[i + my_offset] = j;
        }

        if (correlation > profile_tmp[j + my_offset])
        {
          profile_tmp[j + my_offset] = correlation;
          profileIndex_tmp[j + my_offset] = i;
        }
        i++;
      }
    } //'pragma omp for' places here a barrier unless 'no wait' is specified

    DTYPE max_corr;
    ITYPE max_index = 0;
// Reduction
#pragma omp for schedule(static)
    for (ITYPE colum = 0; colum < profileLength; colum++)
    {
      max_corr = -numeric_limits<DTYPE>::infinity();
      for (ITYPE th = 0; th < numThreads; th++)
      { // uint64_t counter to promote the index of profile_tmp to uint64_t
        if (profile_tmp[colum + (th * profileLength)] > max_corr)
        {
          max_corr = profile_tmp[colum + (th * profileLength)];
          max_index = profileIndex_tmp[colum + (th * profileLength)];
        }
      }
      profile[colum] = max_corr;
      profileIndex[colum] = max_index;
    }
  }
}

int main(int argc, char *argv[])
{
  try
  {
    // Creation of time meassure structures
    chrono::steady_clock::time_point tstart, tend;
    chrono::duration<double> telapsed;

    if (argc != 6)
    {
      cout << "[ERROR] usage: ./scamp input_file win_size num_threads percent_diags out_directory" << endl;
      return 0;
    }

    windowSize = atoi(argv[2]);
    numThreads = atoi(argv[3]);
    percent_diags = atoi(argv[4]);
    string outdir = argv[5];
    // Set the exclusion zone to 0.25
    exclusionZone = (ITYPE)(windowSize * 0.25);
    omp_set_num_threads(numThreads);

    vector<DTYPE> tSeries;
    string inputfilename = argv[1];
    string alg = argv[0];
    alg = alg.substr(2);
    stringstream tmp;
    tmp << outdir << alg.substr(alg.rfind('/') +1) << "_" << inputfilename.substr(inputfilename.rfind('/') + 1, inputfilename.size() - 4 - inputfilename.rfind('/') - 1) << "_w" << windowSize << "_t" << numThreads << "_pdiags" << percent_diags << "_" << getpid() << ".csv";
    string outfilename = tmp.str();

    // Display info through console
    cout << endl;
    cout << "############################################################" << endl;
    cout << "///////////////////////// SCAMP ////////////////////////////" << endl;
    cout << "############################################################" << endl;
    cout << endl;
    cout << "[>>] Reading File: " << inputfilename << "..." << endl;

    /* ------------------------------------------------------------------ */
    /* Read time series file */
    tstart = chrono::steady_clock::now();

    fstream tSeriesFile(inputfilename, ios_base::in);

    DTYPE tempval, tSeriesMin = numeric_limits<DTYPE>::infinity(), tSeriesMax = -numeric_limits<double>::infinity();

    tSeriesLength = 0;
    while (tSeriesFile >> tempval)
    {
      tSeries.push_back(tempval);

      if (tempval < tSeriesMin)
        tSeriesMin = tempval;
      if (tempval > tSeriesMax)
        tSeriesMax = tempval;
      tSeriesLength++;
    }
    tSeriesFile.close();

    tend = chrono::steady_clock::now();
    telapsed = tend - tstart;
    cout << "[OK] Read File Time: " << setprecision(numeric_limits<double>::digits10 + 2) << telapsed.count() << " seconds." << endl;

    // Set Matrix Profile Length
    profileLength = tSeriesLength - windowSize + 1;

    // Auxiliary vectors
    vector<DTYPE> norms(profileLength), means(profileLength), df(profileLength), dg(profileLength);
    vector<DTYPE> profile(profileLength);
    vector<ITYPE> diags, profileIndex(profileLength);

    profile_tmp = new DTYPE[profileLength * numThreads];
    profileIndex_tmp = new ITYPE[profileLength * numThreads];
    // Private profile initialization
    for (ITYPE i = 0; i < (profileLength * numThreads); i++)
      profile_tmp[i] = -numeric_limits<DTYPE>::infinity();

    // Display info through console
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

    /***************** Preprocess ******************/
    cout << "[>>] Preprocessing..." << endl;
    tstart = chrono::steady_clock::now();
    preprocess(tSeries, means, norms, df, dg);
    tend = chrono::steady_clock::now();
    telapsed = tend - tstart;
    cout << "[OK] Preprocessing Time:         " << setprecision(numeric_limits<double>::digits10 + 2) << telapsed.count() << " seconds." << endl;
    /***********************************************/
    
    // Random shuffle the diagonals
    diags.clear();
    for (ITYPE i = exclusionZone + 1; i < profileLength; i++)
      diags.push_back(i);

    if (SHUFFLE_DIAGS){
      //random_device rd;
      mt19937 g(0);
      shuffle(diags.begin(), diags.end(), g);
    }

    /******************** SCAMP ********************/
    cout << "[>>] Executing SCAMP..." << endl;
    tstart = chrono::steady_clock::now();

        // ROI de Iván
    #ifdef ENABLE_PARSEC_HOOKS
      __parsec_roi_begin();
    #endif


    // Establish begining of ROI
    #ifdef ENABLE_GEM5_ROI
    m5_work_begin(0,0);
    #endif


    scamp(tSeries, diags, means, norms, df, dg, profile, profileIndex);

        // Establish end of ROI    
    #ifdef ENABLE_GEM5_ROI
    m5_work_end(0,0);
    #endif
    
    // ROI de Iván
    #ifdef ENABLE_PARSEC_HOOKS
      __parsec_roi_end();
    #endif


    tend = chrono::steady_clock::now();
    telapsed = tend - tstart;
    cout << "[OK] SCAMP Time:              " << setprecision(numeric_limits<double>::digits10 + 2) << telapsed.count() << " seconds." << endl;
    /***********************************************/

    cout << "[>>] Saving result: " << outfilename << " ..." << endl;
    fstream statsFile(outfilename, ios_base::out);
    statsFile << "# Time (s)" << endl;
    statsFile << setprecision(9) << telapsed.count() << endl;
    statsFile << "# Profile Length" << endl;
    statsFile << profileLength << endl;
    statsFile << "# i,tseries,profile,index" << endl;
    for (ITYPE i = 0; i < profileLength; i++)
    {
      statsFile << i << "," << tSeries[i] << "," << (DTYPE)sqrt(2 * windowSize * (1 - profile[i])) << "," << profileIndex[i] << endl;
    }
    statsFile.close();

    delete[] profile_tmp;
    delete[] profileIndex_tmp;
    cout << endl;
  }
  catch (exception &e)
  {
    cout << "Exception: " << e.what() << endl;
  }
}
