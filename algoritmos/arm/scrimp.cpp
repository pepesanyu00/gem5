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

void preprocess(vector<DTYPE> &tSeries, vector<DTYPE> &means, vector<DTYPE> &devs)
{
  DTYPE *ACumSum = new DTYPE[tSeriesLength];
  DTYPE *ASqCumSum = new DTYPE[tSeriesLength];
  DTYPE *ASum = new DTYPE[profileLength];
  DTYPE *ASumSq = new DTYPE[profileLength];
  DTYPE *ASigmaSq = new DTYPE[profileLength];

  ACumSum[0] = tSeries[0];
  ASqCumSum[0] = tSeries[0] * tSeries[0];

  means.clear();
  devs.clear();
  // Cummulate sum
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
    means.push_back(ASum[i] / windowSize);
    ASigmaSq[i] = ASumSq[i] / windowSize - means[i] * means[i];
    devs.push_back(sqrt(ASigmaSq[i]));
  }

}

void scrimp(vector<DTYPE> &tSeries, vector<ITYPE> &idx, vector<DTYPE> &means, vector<DTYPE> &devs, vector<DTYPE> &profile, vector<ITYPE> &profileIndex)
{

#pragma omp parallel //proc_bind(spread)
  {
    // Suppossing ITYPE as uint32_t (we could index series up to 4G elements), to index profile_tmp we need more bits (uint64_t)
    uint64_t my_offset = omp_get_thread_num() * profileLength;
    print_binding_info();
    ITYPE Ndiags = (ITYPE)idx.size()*percent_diags/100;

// Go through diagonals (dynamic)
#pragma omp for schedule(dynamic)
    for (ITYPE ri = 0; ri < Ndiags; ri++)
    {
      // Select a diagonal
      ITYPE diag = idx[ri];
      DTYPE dotProd = 0;

      // Calculate the dot product
      for (ITYPE j = diag; j < windowSize + diag; j++)
        dotProd += tSeries[j] * tSeries[j - diag];

      // j is the column index, i is the row index of the current distance value in the distance matrix
      ITYPE j = diag;
      ITYPE i = 0;

      // Evaluate the distance based on the dot product
      DTYPE distance = 2 * (windowSize - (dotProd - windowSize * means[j] * means[i]) / (devs[j] * devs[i]));

      // update matrix profile and matrix profile index if the current distance value is smaller
      if (distance < profile_tmp[my_offset + j])
      {
        profile_tmp[my_offset + j] = distance;
        profileIndex_tmp[my_offset + j] = i;
      }

      if (distance < profile_tmp[my_offset + i])
      {
        profile_tmp[my_offset + i] = distance;
        profileIndex_tmp[my_offset + i] = j;
      }

      i = 1;

      for (ITYPE j = diag + 1; j < profileLength; j++)
      {
        dotProd += (tSeries[j + windowSize - 1] * tSeries[i + windowSize - 1]) - (tSeries[j - 1] * tSeries[i - 1]);
        distance = 2 * (windowSize - (dotProd - means[j] * means[i] * windowSize) / (devs[j] * devs[i]));

        if (distance < profile_tmp[my_offset + j])
        {
          profile_tmp[my_offset + j] = distance;
          profileIndex_tmp[my_offset + j] = i;
        }

        if (distance < profile_tmp[my_offset + i])
        {
          profile_tmp[my_offset + i] = distance;
          profileIndex_tmp[my_offset + i] = j;
        }
        i++;
      }
    } //'pragma omp for' places here a barrier unless 'no wait' is specified

    DTYPE min_distance;
    ITYPE min_index;
// Reduction
#pragma omp for schedule(static)
    for (ITYPE colum = 0; colum < profileLength; colum++)
    {
      min_distance = numeric_limits<DTYPE>::infinity();
      min_index = 0;
      for (ITYPE th = 0; th < numThreads; th++)
      { // uint64_t counter to promote the index of profile_tmp to uint64_t
        if (profile_tmp[colum + (th * profileLength)] < min_distance)
        {
          min_distance = profile_tmp[colum + (th * profileLength)];
          min_index = profileIndex_tmp[colum + (th * profileLength)];
        }
      }
      profile[colum] = min_distance;
      profileIndex[colum] = min_index;
    }
  }
}

int main(int argc, char *argv[])
{
  try
  {
    // Creation of time meassure structures
    chrono::steady_clock::time_point tstart, tprogstart, tend;
    chrono::duration<double> telapsed;

    if (argc != 6)
    {
      cout << "[ERROR] usage: ./scrimp input_file win_size num_threads percent_diags out_directory" << endl;
      return 0;
    }

    windowSize = atoi(argv[2]);
    numThreads = atoi(argv[3]);
    string outdir = argv[5];
    percent_diags = atoi(argv[4]);
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
    cout << "///////////////////////// SCRIMP ///////////////////////////" << endl;
    cout << "############################################################" << endl;
    cout << endl;
    cout << "[>>] Reading File: " << inputfilename << "..." << endl;

    /* ------------------------------------------------------------------ */
    /* Read time series file */
    tstart = chrono::steady_clock::now();
    tprogstart = tstart;
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

    vector<DTYPE> means, devs, profile(profileLength);
    vector<ITYPE> diags, profileIndex(profileLength);

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
    cout << endl;
    cout << "------------------------------------------------------------" << endl;
    cout << endl;

    // Initialize Matrix Profile and Matrix Profile Index
    cout << "[>>] Initializing Profile..." << endl;
    tstart = chrono::steady_clock::now();

    for (ITYPE i = 0; i < profileLength; i++)
      profile.push_back(numeric_limits<DTYPE>::infinity());

    profile_tmp = new DTYPE[profileLength * numThreads];
    profileIndex_tmp = new ITYPE[profileLength * numThreads];
    // Private profile initialization
    for (ITYPE i = 0; i < (profileLength * numThreads); i++)
      profile_tmp[i] = numeric_limits<DTYPE>::infinity();

    tend = chrono::steady_clock::now();
    telapsed = tend - tstart;
    cout << "[OK] Initializing Profile Time: " << setprecision(numeric_limits<DTYPE>::digits10 + 2) << telapsed.count() << " seconds." << endl;

    // Preprocess, statistics, get the mean and standard deviation of every subsequence in the time series
    cout << "[>>] Preprocessing..." << endl;
    tstart = chrono::steady_clock::now();

    preprocess(tSeries, means, devs);

    tend = chrono::steady_clock::now();
    telapsed = tend - tstart;
    cout << "[OK] Preprocessing Time:         " << setprecision(numeric_limits<double>::digits10 + 2) << telapsed.count() << " seconds." << endl;

    // Random shuffle the diagonals
    diags.clear();
    for (ITYPE i = exclusionZone + 1; i < profileLength; i++)
      diags.push_back(i);

    if (SHUFFLE_DIAGS) {
      //random_device rd;
      mt19937 g(0);
      shuffle(diags.begin(), diags.end(), g);
    }
    
    // Print the first 30 elements of diags
    // cout << "First 30 elements of diags: ";
    // for (size_t i = 0; i < 30 && i < diags.size(); ++i) {
    //   cout << diags[i] << " ";
    // }
    // cout << endl;

    /******************** SCRIMP ********************/
    cout << "[>>] Executing SCRIMP..." << endl;
    tstart = chrono::steady_clock::now();
    
    // ROI de Iván
    #ifdef ENABLE_PARSEC_HOOKS
      __parsec_roi_begin();
    #endif


    // Establish begining of ROI
    #ifdef ENABLE_GEM5_ROI
    m5_work_begin(0,0);
    #endif

    scrimp(tSeries, diags, means, devs, profile, profileIndex);
    
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
    cout << "[OK] SCRIMP Time:             " << setprecision(numeric_limits<DTYPE>::digits10 + 2) << telapsed.count() << " seconds." << endl;

    // Save profile to file
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

    delete[] profile_tmp;
    delete[] profileIndex_tmp;
    tend = chrono::steady_clock::now();
    telapsed = tend - tstart;
    cout << "[OK] Saving Profile Time:       " << setprecision(numeric_limits<DTYPE>::digits10 + 2) << telapsed.count() << " seconds." << endl;

    // Calculate total time
    telapsed = tend - tprogstart;
    cout << "[OK] Total Time:              " << setprecision(numeric_limits<DTYPE>::digits10 + 2) << telapsed.count() << " seconds." << endl;
    cout << endl;
  }
  catch (exception &e)
  {
    cout << "Exception: " << e.what() << endl;
  }
}
