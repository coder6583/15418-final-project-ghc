#include <mpi.h>
#include <fstream>
#include <iostream>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>
#include <unistd.h>
#include <climits>
#include <cmath>
#include <chrono>

#define SOUND_OF_SPEED 343
#define MAX_LAG 25

#define X_MIN -1.0f
#define X_MAX 1.0f
#define Y_MIN -1.0f
#define Y_MAX 1.0f
#define GRID_WIDTH 10000
#define GRID_HEIGHT 10000

#define DX (X_MAX - X_MIN) / (GRID_WIDTH - 1)
#define DY (Y_MAX - Y_MIN) / (GRID_HEIGHT - 1)

#define ENERGY_THRESHOLD 20000000000


struct TimeLag {
  float dist_diff;
  uint8_t silent;
  uint8_t src;
};

struct Result {
  float error;
  float x;
  float y;
  uint8_t src;
};

int get_next_rank(int rank, int nproc) {
  return (rank + 1) % nproc;
}

int get_prev_rank(int rank, int nproc) {
  if (rank == 0) {
    return nproc - 1;
  } else {
    return rank - 1;
  }
}

void sounddetect_ref_pair(
  std::vector<int32_t> &mic_data,
  int num_samples,
  int sample_freq,
  const std::vector<std::vector<float>> &mic_positions,
  int nproc,
  int rank
) {
  const auto compute_start = std::chrono::steady_clock::now();
  std::vector<int16_t> audio_data;

  // Simplifying from 24 bits to 16 bits
  for (auto &data: mic_data) {
    audio_data.push_back((int16_t)(data >> 8));
  }

  std::vector<int16_t> ref_data;
  ref_data.resize(num_samples);

  if (rank == 0) {
    for (int i = 0; i < num_samples; i++) {
      ref_data[i] = audio_data[i];
    }
  }

  MPI_Bcast(ref_data.data(), num_samples * sizeof(int16_t), MPI_BYTE, 0, MPI_COMM_WORLD);

  // Find Time Lag
  float best_dist_diff = 0.0f;
  uint8_t silent = 1;

  int64_t energy = 0;
  int best_lag = 0;
  int64_t best_score = INT64_MIN;
  for (int lag = -MAX_LAG; lag < MAX_LAG; lag++) {
    int64_t score = 0;
    for (int j = 0; j < num_samples; j++) {
      energy += (int64_t)ref_data[j] * (int64_t)ref_data[j];
      if (j + lag < 0) {
        continue;
      }
      if (j + lag >= num_samples) {
        continue;
      }
      score += (int64_t)audio_data[j + lag] * (int64_t)ref_data[j];
    }
    if (score > best_score) {
      best_score = score;
      best_lag = lag;
    }
  }
  if (energy > ENERGY_THRESHOLD) {
    silent = 0;
  }
  best_dist_diff = -(float)best_lag * (1.0f/sample_freq) * 343.0f;
  if (rank == 1) {
    // printf("energy: %ld\n", energy);
    // printf("pair (0, %d): best_lag=%d dist_diff=%f\n", i, best_lag, best_dist_diff[i]);
  }

  std::vector<struct TimeLag> best_dist_diffs(nproc);

  struct TimeLag timelag_result = {
    best_dist_diff,
    silent,
    rank
  };

  int next_rank = get_next_rank(rank, nproc);
  int prev_rank = get_prev_rank(rank, nproc);
  // Send data to all nodes
  // initiate sending data
  MPI_Send(&timelag_result, sizeof(struct TimeLag),
           MPI_BYTE, next_rank, rank, MPI_COMM_WORLD);
  int done = 0;
  while (!done) {
    struct TimeLag recv_buffer;
    MPI_Recv(&recv_buffer, sizeof(struct TimeLag),
             MPI_BYTE, prev_rank, MPI_ANY_TAG, MPI_COMM_WORLD, NULL);

    int orig_src = recv_buffer.src;
    if (orig_src == rank) {
      done = 1;
    } else {
      memcpy(&best_dist_diffs[orig_src], &recv_buffer,
             sizeof(struct TimeLag));
      MPI_Send(&recv_buffer, sizeof(struct TimeLag),
               MPI_BYTE, next_rank, rank, MPI_COMM_WORLD);
      done = 0;
    }
  }
  MPI_Barrier(MPI_COMM_WORLD);

  int nx = GRID_WIDTH / nproc;
  int start_ix = rank * nx;
  int end_ix = (rank + 1) * nx;

  // Find intersection
  float best_error = INFINITY;
  float best_x = 0.0f;
  float best_y = 0.0f;
  for (int ix = start_ix; ix < end_ix; ix ++) {
    float x = X_MIN + ix * DX;
    for (float y = Y_MIN; y < Y_MAX; y += DY) {
      float total_error = 0.0f;

      float dx_0 = x - mic_positions[0][0];
      float dy_0 = y - mic_positions[0][1];
      float dist_0_sq = dx_0 * dx_0 + dy_0 * dy_0;

      for (int i = 1; i < nproc; i++) {
        float dx_i = x - mic_positions[i][0];
        float dy_i = y - mic_positions[i][1];
        float dist_i_sq = dx_i * dx_i + dy_i * dy_i;

        float error = dist_0_sq - dist_i_sq - best_dist_diffs[i].dist_diff;
        total_error += error * error;
      }

      if (total_error < best_error) {
        best_error = total_error;
        best_x = x;
        best_y = y;
      }
    }
  }

  Result local_result = {
    best_error,
    best_x,
    best_y,
    rank,
  };

  std::vector<Result> all_results;
  all_results.resize(nproc);

  MPI_Send(&local_result, sizeof(Result), MPI_BYTE,
           next_rank, rank, MPI_COMM_WORLD);
  all_results[rank] = local_result;
  int gather_done = 0;
  while(!gather_done) {
    Result recv_buffer;
    MPI_Recv(&recv_buffer, sizeof(Result), MPI_BYTE,
             prev_rank, MPI_ANY_TAG, MPI_COMM_WORLD, NULL);

    int orig_src = recv_buffer.src;
    if (orig_src == rank) {
      gather_done = 1;
    } else {
      memcpy(&all_results[orig_src], &recv_buffer, sizeof(Result));
      MPI_Send(&recv_buffer, sizeof(Result), MPI_BYTE,
               next_rank, rank, MPI_COMM_WORLD);
      gather_done = 0;
    }
  }
  MPI_Barrier(MPI_COMM_WORLD);

  float global_best_error = INFINITY;
  float global_x = 0.0f;
  float global_y = 0.0f;
  for (int i = 0; i < nproc; i++) {
    if (all_results[i].error < global_best_error) {
      global_best_error = all_results[i].error;
      global_x = all_results[i].x;
      global_y = all_results[i].y;
    }
  }

  if (rank == 1) {
    if (silent) {
      std::cout << "silent" << std::endl;
    } else {
      std::cout << "mic position: " << mic_positions[1][0] << ", " << mic_positions[1][1] << std::endl;
      std::cout << "coord: " << global_x << ", " << global_y << std::endl;
      const double compute_time = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now() - compute_start).count();
      std::cout << "time: " << std::fixed << std::setprecision(10) << compute_time << std::endl;
    }
    // std::cout << "dx, dy: " << DX << ", " << DY << std::endl;
  }
}

int main(int argc, char *argv[]) {
  int pid;
  int nproc;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &pid);
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);

  std::string input_filename;
  int sample_freq = 16000;
  int num_samples = 4000;
  int batch_size = 256;

  int opt;
  while ((opt = getopt(argc, argv, "f:s:n:b:")) != -1) {
    switch (opt) {
      case 'f':
        input_filename = optarg;
        break;
      case 's':
        sample_freq = atoi(optarg);
        break;
      case 'n':
        num_samples = atoi(optarg);
        break;
      case 'b':
        batch_size = atoi(optarg);
        break;
      default:
        if (pid == 0) {
          std::cerr << "Usage: " << argv[0] << "-f input_filename [-s sample_freq]\n";
        }

        MPI_Finalize();
        exit(EXIT_FAILURE);
    }
  }

  if (empty(input_filename) || sample_freq <= 0) {
    if (pid == 0) {
      std::cerr << "Usage: " << argv[0] << "-f input_filename [-s sample_freq]\n";
    }

    MPI_Finalize();
    exit(EXIT_FAILURE);
  }

  std::vector<std::vector<int32_t>> raw_data;
  raw_data = std::vector(num_samples, std::vector<int>(nproc));
  std::ifstream fin(input_filename);

  if (!fin) {
    std::cerr << "Unable to open file: " << input_filename << ".\n";
    exit(EXIT_FAILURE);
  }

  int num_mics = 0;
  fin >> num_mics;
  std::vector<std::vector<float>> mic_positions;
  mic_positions = std::vector(num_mics, std::vector(2, 0.0f));
  for (int i = 0; i < num_mics; i++) {
    fin >> mic_positions[i][0] >> mic_positions[i][1];
  }
  for (auto &line: raw_data) {
    for (auto &proc_data: line) {
      fin >> proc_data;
    }
  }

  for (int offset = 0; offset < num_samples; offset += batch_size) {
    std::vector<int32_t> mic_data = std::vector(batch_size, 0);
    for (int i = offset; i < offset + batch_size && i < num_samples; i++) {
      mic_data[i - offset] = raw_data[i][pid];
    }

    // mic_data should have the expected data when running on STM32
    sounddetect_ref_pair(mic_data, batch_size, sample_freq, mic_positions, nproc, pid);
  }
  MPI_Finalize();
  return 0;
}
