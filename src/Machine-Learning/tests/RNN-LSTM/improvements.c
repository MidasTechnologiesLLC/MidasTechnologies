/* Enhanced stock_predictor.c with advanced features */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define MAX_SAMPLES       10000
#define INPUT_SIZE        13
#define HIDDEN_SIZE       16
#define OUTPUT_SIZE        1
#define EPOCHS           1000
#define LEARNING_RATE   0.001f
#define VALIDATION_SIZE    30
#define CLIP_VALUE         5.0f
#define DROPOUT_RATE       0.2f
#define REDUCE_LR_FACTOR   0.5f
#define PATIENCE           10

/* Structures for LSTM parameters, gradients, and cache */
#include "lstm_structs.h" // Keep all LSTM-related structures in a separate file

/* Xavier initialization */
#include "init_params.h" // Contains init_params_xavier function

/* Technical indicators */
#include "technical_indicators.h" // Contains OBV, ADX, RSI, Aroon, MACD functions

/* Functions for CSV parsing */
#include "csv_parser.h"

/* Utility functions for normalization and data preparation */
#include "data_utils.h"

/* Dropout implementation */
static void apply_dropout(float *layer, int size, float rate) {
    for (int i = 0; i < size; i++) {
        if ((float)rand() / RAND_MAX < rate) {
            layer[i] = 0.0f;
        }
    }
}

/* Dynamic learning rate adjustment */
static void adjust_learning_rate(float *learning_rate, float current_loss, float *best_loss, int *no_improvement_steps) {
    if (current_loss < *best_loss) {
        *best_loss = current_loss;
        *no_improvement_steps = 0;
    } else {
        (*no_improvement_steps)++;
        if (*no_improvement_steps >= PATIENCE) {
            *learning_rate *= REDUCE_LR_FACTOR;
            *no_improvement_steps = 0;
            printf("Learning rate reduced to %.6f\n", *learning_rate);
        }
    }
}

/* Batch normalization */
static void batch_normalization(float *layer, int size) {
    float mean = 0.0f, variance = 0.0f;
    for (int i = 0; i < size; i++) {
        mean += layer[i];
    }
    mean /= size;
    for (int i = 0; i < size; i++) {
        variance += powf(layer[i] - mean, 2);
    }
    variance /= size;
    float stddev = sqrtf(variance + 1e-7f);
    for (int i = 0; i < size; i++) {
        layer[i] = (layer[i] - mean) / stddev;
    }
}

/* Enhanced LSTM forward pass with dropout and batch normalization */
static void lstm_forward(LSTMParams *params, float inputs[][INPUT_SIZE], int seq_len, LSTMCache *cache) {
    float h_prev[HIDDEN_SIZE] = {0};
    float c_prev[HIDDEN_SIZE] = {0};

    for (int t = 0; t < seq_len; t++) {
        for (int i = 0; i < INPUT_SIZE; i++) {
            cache->x[t][i] = inputs[t][i];
        }

        float i_in[HIDDEN_SIZE] = {0}, f_in[HIDDEN_SIZE] = {0}, o_in[HIDDEN_SIZE] = {0}, c_in[HIDDEN_SIZE] = {0};

        for (int i = 0; i < HIDDEN_SIZE; i++) {
            i_in[i] = params->b_i[i];
            f_in[i] = params->b_f[i];
            o_in[i] = params->b_o[i];
            c_in[i] = params->b_c[i];
        }

        for (int i = 0; i < HIDDEN_SIZE; i++) {
            for (int j = 0; j < INPUT_SIZE; j++) {
                float val = inputs[t][j];
                i_in[i] += params->W_ix[i][j] * val;
                f_in[i] += params->W_fx[i][j] * val;
                o_in[i] += params->W_ox[i][j] * val;
                c_in[i] += params->W_cx[i][j] * val;
            }
            for (int j = 0; j < HIDDEN_SIZE; j++) {
                float hval = h_prev[j];
                i_in[i] += params->W_ih[i][j] * hval;
                f_in[i] += params->W_fh[i][j] * hval;
                o_in[i] += params->W_oh[i][j] * hval;
                c_in[i] += params->W_ch[i][j] * hval;
            }
        }

        for (int i = 0; i < HIDDEN_SIZE; i++) {
            cache->i_gate[t][i] = sigmoid(i_in[i]);
            cache->f_gate[t][i] = sigmoid(f_in[i]);
            cache->o_gate[t][i] = sigmoid(o_in[i]);
            cache->c_hat[t][i] = tanh_approx(c_in[i]);
        }

        for (int i = 0; i < HIDDEN_SIZE; i++) {
            cache->c_state[t][i] = cache->f_gate[t][i] * c_prev[i] + cache->i_gate[t][i] * cache->c_hat[t][i];
            cache->h_state[t][i] = cache->o_gate[t][i] * tanh_approx(cache->c_state[t][i]);
        }

        batch_normalization(cache->h_state[t], HIDDEN_SIZE);
        apply_dropout(cache->h_state[t], HIDDEN_SIZE, DROPOUT_RATE);

        for (int i = 0; i < OUTPUT_SIZE; i++) {
            float sum = params->b_y[i];
            for (int k = 0; k < HIDDEN_SIZE; k++) {
                sum += params->W_hy[i][k] * cache->h_state[t][k];
            }
            cache->y_pred[t][i] = sum;
        }

        memcpy(h_prev, cache->h_state[t], sizeof(h_prev));
        memcpy(c_prev, cache->c_state[t], sizeof(c_prev));
    }
}

/* Training function with enhancements */
static void train_model(float inputs[MAX_SAMPLES][INPUT_SIZE], float targets[MAX_SAMPLES][OUTPUT_SIZE], int validCount) {
    LSTMParams params;
    init_params_xavier(&params, INPUT_SIZE, HIDDEN_SIZE);

    LSTMGrads grads;
    zero_grads(&grads);

    LSTMCache cache;
    float learning_rate = LEARNING_RATE;
    float best_loss = INFINITY;
    int no_improvement_steps = 0;

    for (int epoch = 1; epoch <= EPOCHS; epoch++) {
        zero_grads(&grads);
        lstm_forward(&params, inputs, validCount, &cache);
        float loss = lstm_backward(&params, &cache, targets, validCount, &grads);

        clip_grads(&grads, CLIP_VALUE);
        update_params_custom(&params, &grads, learning_rate);

        adjust_learning_rate(&learning_rate, loss, &best_loss, &no_improvement_steps);

        if (epoch % 100 == 0 || epoch == 1) {
            printf("Epoch %4d, Loss=%.6f, Learning Rate=%.6f\n", epoch, loss / validCount, learning_rate);
        }

        if (learning_rate < 1e-6f) {
            printf("Learning rate too low, stopping early.\n");
            break;
        }
    }
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s path/to/stock_data.csv\n", argv[0]);
        return 1;
    }
    srand((unsigned)time(NULL));

    DailyBar dailyData[MAX_SAMPLES];
    int rawCount = load_and_aggregate_daily(argv[1], dailyData, MAX_SAMPLES);

    if (rawCount <= 1) {
        fprintf(stderr, "No valid data found.\n");
        return 1;
    }

    TechnicalIndicators indicators;
    calculate_technical_indicators(dailyData, rawCount, &indicators);

    float inputs[MAX_SAMPLES][INPUT_SIZE];
    float targets[MAX_SAMPLES][OUTPUT_SIZE];
    int validCount = prepare_data(dailyData, rawCount, &indicators, inputs, targets);

    train_model(inputs, targets, validCount);

    return 0;
}

