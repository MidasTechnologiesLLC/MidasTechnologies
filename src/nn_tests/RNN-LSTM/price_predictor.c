/*******************************************************************************
 * price_predictor.c (Advanced Implementation)
 *
 * Description:
 *   A C program that implements a multi-layer LSTM with Adam optimizer, learning
 *   rate scheduling, L2 regularization, and early stopping for predicting the
 *   next day's stock close price based on technical indicators calculated from
 *   historical stock data.
 *
 * Key Enhancements:
 *   1. Adam Optimizer
 *   2. Stacked LSTM Layers
 *   3. Learning Rate Scheduling
 *   4. L2 Regularization
 *   5. Early Stopping
 *   6. Improved Output Formatting
 *
 * Compile:
 *   gcc price_predictor.c -o price_predictor -lm
 *
 * Run:
 *   ./price_predictor path/to/stock_data.csv
 *******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define MAX_SAMPLES        10000  /* Maximum number of data samples */
#define INPUT_SIZE          13    /* Technical indicators: 13 features */
#define HIDDEN_SIZE        16     /* Number of hidden units per LSTM layer */
#define OUTPUT_SIZE          1    /* Next-day close prediction */
#define EPOCHS             1000    /* Number of training epochs */
#define INITIAL_LEARNING_RATE 0.001f /* Initial learning rate */
#define VALIDATION_SIZE      30    /* Number of days for validation */
#define CLIP_VALUE           5.0f  /* Gradient clipping threshold */
#define L2_LAMBDA          0.0001f /* L2 Regularization parameter */
#define EARLY_STOPPING_PATIENCE 50 /* Epochs to wait before early stopping */
#define NUM_LSTM_LAYERS       2    /* Number of stacked LSTM layers */

/***************************************
 * Data Structures
 ***************************************/

/* LSTM parameters for one layer */
typedef struct {
    /* Input gate */
    float W_ix[HIDDEN_SIZE][INPUT_SIZE];
    float W_ih[HIDDEN_SIZE][HIDDEN_SIZE];
    float b_i[HIDDEN_SIZE];
    /* Forget gate */
    float W_fx[HIDDEN_SIZE][INPUT_SIZE];
    float W_fh[HIDDEN_SIZE][HIDDEN_SIZE];
    float b_f[HIDDEN_SIZE];
    /* Output gate */
    float W_ox[HIDDEN_SIZE][INPUT_SIZE];
    float W_oh[HIDDEN_SIZE][HIDDEN_SIZE];
    float b_o[HIDDEN_SIZE];
    /* Candidate gate */
    float W_cx[HIDDEN_SIZE][INPUT_SIZE];
    float W_ch[HIDDEN_SIZE][HIDDEN_SIZE];
    float b_c[HIDDEN_SIZE];
    /* Output layer */
    float W_hy[OUTPUT_SIZE][HIDDEN_SIZE];
    float b_y[OUTPUT_SIZE];
} LSTMLayerParams;

/* Gradient for LSTM parameters */
typedef struct {
    /* Input gate */
    float W_ix[HIDDEN_SIZE][INPUT_SIZE];
    float W_ih[HIDDEN_SIZE][HIDDEN_SIZE];
    float b_i[HIDDEN_SIZE];
    /* Forget gate */
    float W_fx[HIDDEN_SIZE][INPUT_SIZE];
    float W_fh[HIDDEN_SIZE][HIDDEN_SIZE];
    float b_f[HIDDEN_SIZE];
    /* Output gate */
    float W_ox[HIDDEN_SIZE][INPUT_SIZE];
    float W_oh[HIDDEN_SIZE][HIDDEN_SIZE];
    float b_o[HIDDEN_SIZE];
    /* Candidate gate */
    float W_cx[HIDDEN_SIZE][INPUT_SIZE];
    float W_ch[HIDDEN_SIZE][HIDDEN_SIZE];
    float b_c[HIDDEN_SIZE];
    /* Output layer */
    float W_hy[OUTPUT_SIZE][HIDDEN_SIZE];
    float b_y[OUTPUT_SIZE];
} LSTMLayerGrads;

/* Adam optimizer parameters for one layer */
typedef struct {
    /* Input gate */
    float m_W_ix[HIDDEN_SIZE][INPUT_SIZE];
    float v_W_ix[HIDDEN_SIZE][INPUT_SIZE];
    float m_W_ih[HIDDEN_SIZE][HIDDEN_SIZE];
    float v_W_ih[HIDDEN_SIZE][HIDDEN_SIZE];
    float m_b_i[HIDDEN_SIZE];
    float v_b_i[HIDDEN_SIZE];
    /* Forget gate */
    float m_W_fx[HIDDEN_SIZE][INPUT_SIZE];
    float v_W_fx[HIDDEN_SIZE][INPUT_SIZE];
    float m_W_fh[HIDDEN_SIZE][HIDDEN_SIZE];
    float v_W_fh[HIDDEN_SIZE][HIDDEN_SIZE];
    float m_b_f[HIDDEN_SIZE];
    float v_b_f[HIDDEN_SIZE];
    /* Output gate */
    float m_W_ox[HIDDEN_SIZE][INPUT_SIZE];
    float v_W_ox[HIDDEN_SIZE][INPUT_SIZE];
    float m_W_oh[HIDDEN_SIZE][HIDDEN_SIZE];
    float v_W_oh[HIDDEN_SIZE][HIDDEN_SIZE];
    float m_b_o[HIDDEN_SIZE];
    float v_b_o[HIDDEN_SIZE];
    /* Candidate gate */
    float m_W_cx[HIDDEN_SIZE][INPUT_SIZE];
    float v_W_cx[HIDDEN_SIZE][INPUT_SIZE];
    float m_W_ch[HIDDEN_SIZE][HIDDEN_SIZE];
    float v_W_ch[HIDDEN_SIZE][HIDDEN_SIZE];
    float m_b_c[HIDDEN_SIZE];
    float v_b_c[HIDDEN_SIZE];
    /* Output layer */
    float m_W_hy[OUTPUT_SIZE][HIDDEN_SIZE];
    float v_W_hy[OUTPUT_SIZE][HIDDEN_SIZE];
    float m_b_y[OUTPUT_SIZE];
    float v_b_y[OUTPUT_SIZE];
} LSTMLayerAdam;

/* Forward-pass caches for one layer */
typedef struct {
    float i_gate[MAX_SAMPLES][HIDDEN_SIZE];
    float f_gate[MAX_SAMPLES][HIDDEN_SIZE];
    float o_gate[MAX_SAMPLES][HIDDEN_SIZE];
    float c_hat[MAX_SAMPLES][HIDDEN_SIZE];
    float c_state[MAX_SAMPLES][HIDDEN_SIZE];
    float h_state[MAX_SAMPLES][HIDDEN_SIZE];
    float y_pred[MAX_SAMPLES][OUTPUT_SIZE];
} LSTMLayerCache;

/* All layers */
typedef struct {
    LSTMLayerParams layers[NUM_LSTM_LAYERS];
    LSTMLayerGrads grads[NUM_LSTM_LAYERS];
    LSTMLayerAdam adam[NUM_LSTM_LAYERS];
    LSTMLayerCache cache[NUM_LSTM_LAYERS];
} LSTMModel;

/* Daily aggregated data */
typedef struct {
    char date[11];  /* "YYYY-MM-DD" */
    float open;
    float high;
    float low;
    float close;
    float volume;
} DailyBar;

/***************************************
 * Technical Indicators Structures
 ***************************************/
typedef struct {
    float obv[MAX_SAMPLES];
    float ad[MAX_SAMPLES];
    float adx[MAX_SAMPLES];
    float aroonUp[MAX_SAMPLES];
    float aroonDown[MAX_SAMPLES];
    float macd[MAX_SAMPLES];
    float rsi[MAX_SAMPLES];
} TechnicalIndicators;

/***************************************
 * Early Stopping Structure
 ***************************************/
typedef struct {
    float best_validation_loss;
    int epochs_no_improve;
    int stop;
} EarlyStopping;

/***************************************
 * Utility Functions
 ***************************************/

/* Generate a random float in [-range, range] */
static inline float randf(float range) {
    float r = (float)rand() / (float)RAND_MAX;
    return (r * 2.0f - 1.0f) * range;
}

/* Sigmoid activation */
static inline float sigmoid_act(float x) {
    return 1.0f / (1.0f + expf(-x));
}

/* Derivative of sigmoid */
static inline float dsigmoid(float s) {
    return s * (1.0f - s);
}

/* Tanh activation */
static inline float tanh_act(float x) {
    return tanhf(x);
}

/* Derivative of tanh */
static inline float dtanh_act(float tval) {
    return 1.0f - tval * tval;
}

/* Initialize LSTM parameters with Xavier Initialization */
static void init_lstm_params(LSTMModel *model, float input_size, float hidden_size) {
    for(int layer = 0; layer < NUM_LSTM_LAYERS; layer++) {
        float current_input_size = (layer == 0) ? input_size : hidden_size;
        float limit = sqrtf(6.0f / (current_input_size + hidden_size));
        for(int i = 0; i < hidden_size; i++) {
            for(int j = 0; j < current_input_size; j++) {
                model->layers[layer].W_ix[i][j] = randf(limit);
                model->layers[layer].W_fx[i][j] = randf(limit);
                model->layers[layer].W_ox[i][j] = randf(limit);
                model->layers[layer].W_cx[i][j] = randf(limit);
            }
            for(int j = 0; j < hidden_size; j++) {
                model->layers[layer].W_ih[i][j] = randf(limit);
                model->layers[layer].W_fh[i][j] = randf(limit);
                model->layers[layer].W_oh[i][j] = randf(limit);
                model->layers[layer].W_ch[i][j] = randf(limit);
            }
            model->layers[layer].b_i[i] = 0.0f;
            model->layers[layer].b_f[i] = 1.0f; // Initialize forget gate bias to 1.0f
            model->layers[layer].b_o[i] = 0.0f;
            model->layers[layer].b_c[i] = 0.0f;
        }
        /* Initialize output layer */
        for(int i = 0; i < OUTPUT_SIZE; i++) {
            for(int j = 0; j < hidden_size; j++) {
                model->layers[layer].W_hy[i][j] = randf(limit);
            }
            model->layers[layer].b_y[i] = 0.0f;
        }
    }
}

/* Zero out gradients */
static void zero_grads(LSTMModel *model) {
    for(int layer = 0; layer < NUM_LSTM_LAYERS; layer++) {
        memset(&model->grads[layer], 0, sizeof(LSTMLayerGrads));
    }
}

/* Apply gradient clipping */
static void clip_grads(LSTMModel *model, float clip_value) {
    for(int layer = 0; layer < NUM_LSTM_LAYERS; layer++) {
        LSTMLayerGrads *g = &model->grads[layer];
        /* Clip weights */
        for(int i = 0; i < HIDDEN_SIZE; i++) {
            for(int j = 0; j < INPUT_SIZE; j++) {
                /* Input gate */
                if(g->W_ix[i][j] > clip_value) g->W_ix[i][j] = clip_value;
                if(g->W_ix[i][j] < -clip_value) g->W_ix[i][j] = -clip_value;
                /* Forget gate */
                if(g->W_fx[i][j] > clip_value) g->W_fx[i][j] = clip_value;
                if(g->W_fx[i][j] < -clip_value) g->W_fx[i][j] = -clip_value;
                /* Output gate */
                if(g->W_ox[i][j] > clip_value) g->W_ox[i][j] = clip_value;
                if(g->W_ox[i][j] < -clip_value) g->W_ox[i][j] = -clip_value;
                /* Candidate gate */
                if(g->W_cx[i][j] > clip_value) g->W_cx[i][j] = clip_value;
                if(g->W_cx[i][j] < -clip_value) g->W_cx[i][j] = -clip_value;
            }
            for(int j = 0; j < HIDDEN_SIZE; j++) {
                /* Input gate */
                if(g->W_ih[i][j] > clip_value) g->W_ih[i][j] = clip_value;
                if(g->W_ih[i][j] < -clip_value) g->W_ih[i][j] = -clip_value;
                /* Forget gate */
                if(g->W_fh[i][j] > clip_value) g->W_fh[i][j] = clip_value;
                if(g->W_fh[i][j] < -clip_value) g->W_fh[i][j] = -clip_value;
                /* Output gate */
                if(g->W_oh[i][j] > clip_value) g->W_oh[i][j] = clip_value;
                if(g->W_oh[i][j] < -clip_value) g->W_oh[i][j] = -clip_value;
                /* Candidate gate */
                if(g->W_ch[i][j] > clip_value) g->W_ch[i][j] = clip_value;
                if(g->W_ch[i][j] < -clip_value) g->W_ch[i][j] = -clip_value;
            }
            /* Biases */
            if(g->b_i[i] > clip_value) g->b_i[i] = clip_value;
            if(g->b_i[i] < -clip_value) g->b_i[i] = -clip_value;
            if(g->b_f[i] > clip_value) g->b_f[i] = clip_value;
            if(g->b_f[i] < -clip_value) g->b_f[i] = -clip_value;
            if(g->b_o[i] > clip_value) g->b_o[i] = clip_value;
            if(g->b_o[i] < -clip_value) g->b_o[i] = -clip_value;
            if(g->b_c[i] > clip_value) g->b_c[i] = clip_value;
            if(g->b_c[i] < -clip_value) g->b_c[i] = -clip_value;
        }
        /* Clip output layer gradients */
        for(int layer = 0; layer < NUM_LSTM_LAYERS; layer++) {
            LSTMLayerGrads *g_out = &model->grads[layer];
            for(int i = 0; i < OUTPUT_SIZE; i++) {
                for(int j = 0; j < HIDDEN_SIZE; j++) {
                    if(g_out->W_hy[i][j] > clip_value) g_out->W_hy[i][j] = clip_value;
                    if(g_out->W_hy[i][j] < -clip_value) g_out->W_hy[i][j] = -clip_value;
                }
                if(g_out->b_y[i] > clip_value) g_out->b_y[i] = clip_value;
                if(g_out->b_y[i] < -clip_value) g_out->b_y[i] = -clip_value;
            }
        }
    }

/* Initialize LSTM cache */
static void init_cache(LSTMModel *model) {
    for(int layer = 0; layer < NUM_LSTM_LAYERS; layer++) {
        memset(&model->cache[layer], 0, sizeof(LSTMLayerCache));
    }
}

/* Initialize Adam optimizer parameters */
static void init_adam_parameters(LSTMModel *model) {
    for(int layer = 0; layer < NUM_LSTM_LAYERS; layer++) {
        memset(&model->adam[layer], 0, sizeof(LSTMLayerAdam));
    }
}

/* Update parameters using Adam optimizer */
static void update_parameters_adam(LSTMModel *model, int epoch) {
    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float epsilon = 1e-8f;
    
    for(int layer = 0; layer < NUM_LSTM_LAYERS; layer++) {
        LSTMLayerParams *params = &model->layers[layer];
        LSTMLayerGrads *grads = &model->grads[layer];
        LSTMLayerAdam *adam = &model->adam[layer];
        
        /* Update weights and biases */
        for(int i = 0; i < HIDDEN_SIZE; i++) {
            for(int j = 0; j < INPUT_SIZE; j++) {
                /* Input gate W_ix */
                adam->m_W_ix[i][j] = beta1 * adam->m_W_ix[i][j] + (1 - beta1) * grads->W_ix[i][j];
                adam->v_W_ix[i][j] = beta2 * adam->v_W_ix[i][j] + (1 - beta2) * grads->W_ix[i][j] * grads->W_ix[i][j];
                float m_hat = adam->m_W_ix[i][j] / (1 - powf(beta1, epoch));
                float v_hat = adam->v_W_ix[i][j] / (1 - powf(beta2, epoch));
                params->W_ix[i][j] -= (INITIAL_LEARNING_RATE * m_hat) / (sqrtf(v_hat) + epsilon);
                
                /* Forget gate W_fx */
                adam->m_W_fx[i][j] = beta1 * adam->m_W_fx[i][j] + (1 - beta1) * grads->W_fx[i][j];
                adam->v_W_fx[i][j] = beta2 * adam->v_W_fx[i][j] + (1 - beta2) * grads->W_fx[i][j] * grads->W_fx[i][j];
                m_hat = adam->m_W_fx[i][j] / (1 - powf(beta1, epoch));
                v_hat = adam->v_W_fx[i][j] / (1 - powf(beta2, epoch));
                params->W_fx[i][j] -= (INITIAL_LEARNING_RATE * m_hat) / (sqrtf(v_hat) + epsilon);
                
                /* Output gate W_ox */
                adam->m_W_ox[i][j] = beta1 * adam->m_W_ox[i][j] + (1 - beta1) * grads->W_ox[i][j];
                adam->v_W_ox[i][j] = beta2 * adam->v_W_ox[i][j] + (1 - beta2) * grads->W_ox[i][j] * grads->W_ox[i][j];
                m_hat = adam->m_W_ox[i][j] / (1 - powf(beta1, epoch));
                v_hat = adam->v_W_ox[i][j] / (1 - powf(beta2, epoch));
                params->W_ox[i][j] -= (INITIAL_LEARNING_RATE * m_hat) / (sqrtf(v_hat) + epsilon);
                
                /* Candidate gate W_cx */
                adam->m_W_cx[i][j] = beta1 * adam->m_W_cx[i][j] + (1 - beta1) * grads->W_cx[i][j];
                adam->v_W_cx[i][j] = beta2 * adam->v_W_cx[i][j] + (1 - beta2) * grads->W_cx[i][j] * grads->W_cx[i][j];
                m_hat = adam->m_W_cx[i][j] / (1 - powf(beta1, epoch));
                v_hat = adam->v_W_cx[i][j] / (1 - powf(beta2, epoch));
                params->W_cx[i][j] -= (INITIAL_LEARNING_RATE * m_hat) / (sqrtf(v_hat) + epsilon);
            }
            for(int j = 0; j < HIDDEN_SIZE; j++) {
                /* Input gate W_ih */
                adam->m_W_ih[i][j] = beta1 * adam->m_W_ih[i][j] + (1 - beta1) * grads->W_ih[i][j];
                adam->v_W_ih[i][j] = beta2 * adam->v_W_ih[i][j] + (1 - beta2) * grads->W_ih[i][j] * grads->W_ih[i][j];
                float m_hat_ih = adam->m_W_ih[i][j] / (1 - powf(beta1, epoch));
                float v_hat_ih = adam->v_W_ih[i][j] / (1 - powf(beta2, epoch));
                params->W_ih[i][j] -= (INITIAL_LEARNING_RATE * m_hat_ih) / (sqrtf(v_hat_ih) + epsilon);
                
                /* Forget gate W_fh */
                adam->m_W_fh[i][j] = beta1 * adam->m_W_fh[i][j] + (1 - beta1) * grads->W_fh[i][j];
                adam->v_W_fh[i][j] = beta2 * adam->v_W_fh[i][j] + (1 - beta2) * grads->W_fh[i][j] * grads->W_fh[i][j];
                float m_hat_fh = adam->m_W_fh[i][j] / (1 - powf(beta1, epoch));
                float v_hat_fh = adam->v_W_fh[i][j] / (1 - powf(beta2, epoch));
                params->W_fh[i][j] -= (INITIAL_LEARNING_RATE * m_hat_fh) / (sqrtf(v_hat_fh) + epsilon);
                
                /* Output gate W_oh */
                adam->m_W_oh[i][j] = beta1 * adam->m_W_oh[i][j] + (1 - beta1) * grads->W_oh[i][j];
                adam->v_W_oh[i][j] = beta2 * adam->v_W_oh[i][j] + (1 - beta2) * grads->W_oh[i][j] * grads->W_oh[i][j];
                float m_hat_oh = adam->m_W_oh[i][j] / (1 - powf(beta1, epoch));
                float v_hat_oh = adam->v_W_oh[i][j] / (1 - powf(beta2, epoch));
                params->W_oh[i][j] -= (INITIAL_LEARNING_RATE * m_hat_oh) / (sqrtf(v_hat_oh) + epsilon);
                
                /* Candidate gate W_ch */
                adam->m_W_ch[i][j] = beta1 * adam->m_W_ch[i][j] + (1 - beta1) * grads->W_ch[i][j];
                adam->v_W_ch[i][j] = beta2 * adam->v_W_ch[i][j] + (1 - beta2) * grads->W_ch[i][j] * grads->W_ch[i][j];
                float m_hat_ch = adam->m_W_ch[i][j] / (1 - powf(beta1, epoch));
                float v_hat_ch = adam->v_W_ch[i][j] / (1 - powf(beta2, epoch));
                params->W_ch[i][j] -= (INITIAL_LEARNING_RATE * m_hat_ch) / (sqrtf(v_hat_ch) + epsilon);
            }
            /* Biases */
            for(int j = 0; j < HIDDEN_SIZE; j++) {
                /* Input gate b_i */
                adam->m_b_i[j] = beta1 * adam->m_b_i[j] + (1 - beta1) * grads->b_i[j];
                adam->v_b_i[j] = beta2 * adam->v_b_i[j] + (1 - beta2) * grads->b_i[j] * grads->b_i[j];
                float m_hat_bi = adam->m_b_i[j] / (1 - powf(beta1, epoch));
                float v_hat_bi = adam->v_b_i[j] / (1 - powf(beta2, epoch));
                params->b_i[j] -= (INITIAL_LEARNING_RATE * m_hat_bi) / (sqrtf(v_hat_bi) + epsilon);
                
                /* Forget gate b_f */
                adam->m_b_f[j] = beta1 * adam->m_b_f[j] + (1 - beta1) * grads->b_f[j];
                adam->v_b_f[j] = beta2 * adam->v_b_f[j] + (1 - beta2) * grads->b_f[j] * grads->b_f[j];
                float m_hat_bf = adam->m_b_f[j] / (1 - powf(beta1, epoch));
                float v_hat_bf = adam->v_b_f[j] / (1 - powf(beta2, epoch));
                params->b_f[j] -= (INITIAL_LEARNING_RATE * m_hat_bf) / (sqrtf(v_hat_bf) + epsilon);
                
                /* Output gate b_o */
                adam->m_b_o[j] = beta1 * adam->m_b_o[j] + (1 - beta1) * grads->b_o[j];
                adam->v_b_o[j] = beta2 * adam->v_b_o[j] + (1 - beta2) * grads->b_o[j] * grads->b_o[j];
                float m_hat_bo = adam->m_b_o[j] / (1 - powf(beta1, epoch));
                float v_hat_bo = adam->v_b_o[j] / (1 - powf(beta2, epoch));
                params->b_o[j] -= (INITIAL_LEARNING_RATE * m_hat_bo) / (sqrtf(v_hat_bo) + epsilon);
                
                /* Candidate gate b_c */
                adam->m_b_c[j] = beta1 * adam->m_b_c[j] + (1 - beta1) * grads->b_c[j];
                adam->v_b_c[j] = beta2 * adam->v_b_c[j] + (1 - beta2) * grads->b_c[j] * grads->b_c[j];
                float m_hat_bc = adam->m_b_c[j] / (1 - powf(beta1, epoch));
                float v_hat_bc = adam->v_b_c[j] / (1 - powf(beta2, epoch));
                params->b_c[j] -= (INITIAL_LEARNING_RATE * m_hat_bc) / (sqrtf(v_hat_bc) + epsilon);
            }
        }
    }

/* Apply L2 regularization to gradients */
static void apply_l2_regularization(LSTMModel *model, float lambda) {
    for(int layer = 0; layer < NUM_LSTM_LAYERS; layer++) {
        LSTMLayerGrads *g = &model->grads[layer];
        LSTMLayerParams *p = &model->layers[layer];
        /* Input gate weights */
        for(int i = 0; i < HIDDEN_SIZE; i++) {
            for(int j = 0; j < INPUT_SIZE; j++) {
                g->W_ix[i][j] += lambda * p->W_ix[i][j];
                g->W_fx[i][j] += lambda * p->W_fx[i][j];
                g->W_ox[i][j] += lambda * p->W_ox[i][j];
                g->W_cx[i][j] += lambda * p->W_cx[i][j];
            }
            for(int j = 0; j < HIDDEN_SIZE; j++) {
                g->W_ih[i][j] += lambda * p->W_ih[i][j];
                g->W_fh[i][j] += lambda * p->W_fh[i][j];
                g->W_oh[i][j] += lambda * p->W_oh[i][j];
                g->W_ch[i][j] += lambda * p->W_ch[i][j];
            }
            /* Biases */
            g->b_i[i] += lambda * p->b_i[i];
            g->b_f[i] += lambda * p->b_f[i];
            g->b_o[i] += lambda * p->b_o[i];
            g->b_c[i] += lambda * p->b_c[i];
        }
        /* Output layer weights */
        for(int i = 0; i < OUTPUT_SIZE; i++) {
            for(int j = 0; j < HIDDEN_SIZE; j++) {
                g->W_hy[i][j] += lambda * p->W_hy[i][j];
            }
            g->b_y[i] += lambda * p->b_y[i];
        }
    }
}

/* Implement learning rate schedule (e.g., step decay) */
static float get_learning_rate(int epoch) {
    float initial_lr = INITIAL_LEARNING_RATE;
    float decay_rate = 0.5f;
    int decay_step = 200;
    return initial_lr * powf(decay_rate, floorf((float)epoch / decay_step));
}

/* Initialize early stopping */
static void init_early_stopping(EarlyStopping *es) {
    es->best_validation_loss = INFINITY;
    es->epochs_no_improve = 0;
    es->stop = 0;
}

/* Update early stopping */
static void update_early_stopping(EarlyStopping *es, float current_loss) {
    if(current_loss < es->best_validation_loss) {
        es->best_validation_loss = current_loss;
        es->epochs_no_improve = 0;
    }
    else {
        es->epochs_no_improve += 1;
        if(es->epochs_no_improve >= EARLY_STOPPING_PATIENCE) {
            es->stop = 1;
        }
    }
}

/***************************************
 * CSV Parsing and Data Aggregation
 ***************************************/

/* Load intraday CSV, aggregate into daily bars */
static int load_and_aggregate_daily(
    const char *csv_file,
    DailyBar *daily,
    int max_days
){
    FILE *fp = fopen(csv_file, "r");
    if(!fp){
        fprintf(stderr, "Could not open file: %s\n", csv_file);
        return -1;
    }

    char line[1024];
    int firstRow = 1;
    char currentDate[11] = {0};
    DailyBar curDay;
    int haveCurrent = 0;
    int dayCount = 0;

    while(fgets(line, sizeof(line), fp)){
        if(firstRow){
            /* Detect and skip header */
            if(strstr(line, "time") || strstr(line, "Timestamp")){
                firstRow = 0;
                continue;
            }
            firstRow = 0; /* No header present */
        }

        /* Tokenize the CSV line */
        char *ts   = strtok(line, ",");
        char *oStr = strtok(NULL, ",");
        char *hStr = strtok(NULL, ",");
        char *lStr = strtok(NULL, ",");
        char *cStr = strtok(NULL, ",");
        char *vStr = strtok(NULL, ",");
        if(!ts || !oStr || !hStr || !lStr || !cStr || !vStr) {
            /* Skip malformed line */
            continue;
        }

        /* Extract date (YYYY-MM-DD) from Timestamp */
        char dateBuf[11];
        strncpy(dateBuf, ts, 10);
        dateBuf[10] = '\0';

        char *endptr;
        float o_val = strtof(oStr, &endptr);
        if(endptr == oStr) continue; /* parse fail */
        float h_val = strtof(hStr, &endptr);
        if(endptr == hStr) continue;
        float l_val = strtof(lStr, &endptr);
        if(endptr == lStr) continue;
        float c_val = strtof(cStr, &endptr);
        if(endptr == cStr) continue;
        float v_val = strtof(vStr, &endptr);
        if(endptr == vStr) continue;

        /* If it's a new date, finalize the old day (if any) */
        if(!haveCurrent || strcmp(dateBuf, currentDate) != 0){
            /* Finalize the previous day */
            if(haveCurrent){
                /* Check if curDay is valid: open>0, high>=open, etc. */
                if(curDay.open > 0 && curDay.high >= curDay.open && curDay.low <= curDay.open
                   && curDay.close > 0 && curDay.volume >= 0
                   && dayCount < max_days)
                {
                    daily[dayCount++] = curDay;
                }
            }
            /* Start new day */
            strncpy(currentDate, dateBuf, 11);
            curDay.open   = o_val;
            curDay.high   = h_val;
            curDay.low    = l_val;
            curDay.close  = c_val;
            curDay.volume = v_val;
            strncpy(curDay.date, dateBuf, 11);
            haveCurrent = 1;
        } else {
            /* Same day => update H, L, C, Vol */
            if(h_val > curDay.high) curDay.high = h_val;
            if(l_val < curDay.low)  curDay.low  = l_val;
            curDay.close  = c_val;
            curDay.volume += v_val; 
        }
    }
    /* Finalize last day if valid */
    if(haveCurrent && dayCount < max_days) {
        if(curDay.open > 0 && curDay.high >= curDay.open && curDay.low <= curDay.open
           && curDay.close > 0 && curDay.volume >= 0)
        {
            daily[dayCount++] = curDay;
        }
    }

    fclose(fp);
    return dayCount;
}

/***************************************
 * Technical Indicators Calculation
 ***************************************/

/* On-Balance Volume (OBV) */
static void calculate_obv(DailyBar data[], int count, float obvArr[]) {
    obvArr[0] = 0.0f;
    for(int i = 1; i < count; i++) {
        if(data[i].close > data[i-1].close)
            obvArr[i] = obvArr[i-1] + data[i].volume;
        else if(data[i].close < data[i-1].close)
            obvArr[i] = obvArr[i-1] - data[i].volume;
        else
            obvArr[i] = obvArr[i-1];
    }
}

/* Accumulation/Distribution (AD) */
static void calculate_ad(DailyBar data[], int count, float adArr[]) {
    for(int i = 0; i < count; i++) {
        float money_flow_multiplier = 0.0f;
        if(data[i].high != data[i].low) {
            money_flow_multiplier = ((data[i].close - data[i].low) - (data[i].high - data[i].close)) / (data[i].high - data[i].low);
        }
        float money_flow_volume = money_flow_multiplier * data[i].volume;
        if(i == 0)
            adArr[i] = money_flow_volume;
        else
            adArr[i] = adArr[i-1] + money_flow_volume;
    }
}

/* Relative Strength Index (RSI) */
static void calculate_rsi(DailyBar data[], int count, float rsiArr[]) {
    int period = 14;
    float gains = 0.0f, losses = 0.0f;

    /* Initial calculation */
    for(int i = 1; i <= period; i++) {
        float change = data[i].close - data[i-1].close;
        if(change > 0)
            gains += change;
        else
            losses -= change;
    }

    float average_gain = gains / period;
    float average_loss = losses / period;
    rsiArr[period] = (average_loss == 0) ? 100.0f : 100.0f - (100.0f / (1.0f + (average_gain / average_loss)));

    /* Subsequent calculations */
    for(int i = period + 1; i < count; i++) {
        float change = data[i].close - data[i-1].close;
        if(change > 0) {
            average_gain = ((average_gain * (period - 1)) + change) / period;
            average_loss = (average_loss * (period - 1)) / period;
        } else {
            average_gain = (average_gain * (period - 1)) / period;
            average_loss = ((average_loss * (period - 1)) - change) / period;
        }
        rsiArr[i] = (average_loss == 0) ? 100.0f : 100.0f - (100.0f / (1.0f + (average_gain / average_loss)));
    }

    /* Fill the initial periods with 50 (neutral RSI) */
    for(int i = 0; i < period; i++) {
        rsiArr[i] = 50.0f;
    }
}

/* Aroon Up and Aroon Down */
static void calculate_aroon(DailyBar data[], int count, float aroonUpArr[], float aroonDownArr[]) {
    int period = 25;
    for(int i = 0; i < count; i++) {
        if(i < period - 1){
            aroonUpArr[i] = 50.0f;
            aroonDownArr[i] = 50.0f;
            continue;
        }
        int highest = 0, lowest = 0;
        for(int j = 0; j < period; j++) {
            if(data[i - j].high > data[i - highest].high)
                highest = j;
            if(data[i - j].low < data[i - lowest].low)
                lowest = j;
        }
        aroonUpArr[i] = ((float)(period - highest) / period) * 100.0f;
        aroonDownArr[i] = ((float)(period - lowest) / period) * 100.0f;
    }
}

/* Moving Average Convergence Divergence (MACD) */
static void calculate_macd(DailyBar data[], int count, float macdArr[]) {
    int short_period = 12;
    int long_period = 26;
    int signal_period = 9;
    float ema_short = 0.0f, ema_long = 0.0f;
    float multiplier_short = 2.0f / (short_period + 1);
    float multiplier_long = 2.0f / (long_period + 1);
    float macd_signal = 0.0f;
    float multiplier_signal = 2.0f / (signal_period + 1);

    /* Initialize EMA_short and EMA_long */
    for(int i = 0; i < count; i++) {
        if(i == short_period - 1) {
            float sum = 0.0f;
            for(int j = 0; j < short_period; j++) {
                sum += data[i - j].close;
            }
            ema_short = sum / short_period;
            macdArr[i] = 0.0f; /* MACD undefined for first calculation */
        }
        else if(i >= short_period) {
            ema_short = (data[i].close - ema_short) * multiplier_short + ema_short;
            if(i == long_period - 1) {
                float sum = 0.0f;
                for(int j = 0; j < long_period; j++) {
                    sum += data[i - j].close;
                }
                ema_long = sum / long_period;
                macdArr[i] = ema_short - ema_long;
            }
            else if(i >= long_period) {
                ema_long = (data[i].close - ema_long) * multiplier_long + ema_long;
                float macd = ema_short - ema_long;
                macdArr[i] = macd;
            }
            else {
                macdArr[i] = 0.0f;
            }
        }
        else {
            macdArr[i] = 0.0f;
        }
    }

    /* Signal line (EMA of MACD) */
    float signal_line = 0.0f;
    for(int i = 0; i < count; i++) {
        if(macdArr[i] == 0.0f){
            /* Do nothing */
        }
        else {
            if(i < long_period + signal_period - 1){
                /* Not enough data for signal line */
                macdArr[i] = 0.0f;
            }
            else if(i == long_period + signal_period - 1){
                /* Initialize signal line */
                float sum = 0.0f;
                for(int j = 0; j < signal_period; j++) {
                    sum += macdArr[i - j];
                }
                signal_line = sum / signal_period;
                macdArr[i] = signal_line;
            }
            else {
                /* Update signal line */
                signal_line = (macdArr[i] - signal_line) * multiplier_signal + signal_line;
                macdArr[i] = signal_line;
            }
        }
    }
}

/* Average Directional Index (ADX) - Simplified version */
static void calculate_adx(DailyBar data[], int count, float adxArr[]) {
    int period = 14;
    float tr = 0.0f, plus_dm = 0.0f, minus_dm = 0.0f;
    float atr = 0.0f, plus_di = 0.0f, minus_di = 0.0f, dx = 0.0f, adx = 0.0f;

    for(int i = 1; i < count; i++) {
        float high_diff = data[i].high - data[i-1].high;
        float low_diff = data[i-1].low - data[i].low;
        float plus_dm_val = (high_diff > low_diff && high_diff > 0) ? high_diff : 0.0f;
        float minus_dm_val = (low_diff > high_diff && low_diff > 0) ? low_diff : 0.0f;
        float tr_val = fmaxf(data[i].high - data[i].low, fmaxf(fabsf(data[i].high - data[i-1].close), fabsf(data[i].low - data[i-1].close)));

        plus_dm += plus_dm_val;
        minus_dm += minus_dm_val;
        tr += tr_val;

        if(i >= period) {
            if(i > period){
                plus_dm = plus_dm - (plus_dm / period) + plus_dm_val;
                minus_dm = minus_dm - (minus_dm / period) + minus_dm_val;
                tr = tr - (tr / period) + tr_val;
            }

            atr = tr / period;
            plus_di = (atr == 0.0f) ? 0.0f : (plus_dm / atr) * 100.0f;
            minus_di = (atr == 0.0f) ? 0.0f : (minus_dm / atr) * 100.0f;
            float di_diff = fabsf(plus_di - minus_di);
            float di_sum = plus_di + minus_di;
            dx = (di_sum == 0.0f) ? 0.0f : (di_diff / di_sum) * 100.0f;

            if(i == period){
                adx = dx;
            }
            else {
                adx = ((adx * (period - 1)) + dx) / period;
            }

            adxArr[i] = adx;
        }
        else {
            adxArr[i] = 0.0f;
        }
    }

    /* Fill the initial periods with 0 */
    for(int i = 0; i < period; i++) {
        adxArr[i] = 0.0f;
    }
}

/* Normalize Data */
static void normalize_data(float inputs[][INPUT_SIZE], float targets[][OUTPUT_SIZE], int count,
                           float minVal[], float maxVal[],
                           float *min_target, float *max_target,
                           float normalized_inputs[][INPUT_SIZE],
                           float normalized_targets[][OUTPUT_SIZE]) {
    /* Find min and max for inputs */
    for(int j = 0; j < INPUT_SIZE; j++){
        minVal[j] = 1e9f;
        maxVal[j] = -1e9f;
    }
    for(int i = 0; i < count; i++){
        for(int j = 0; j < INPUT_SIZE; j++){
            float v = inputs[i][j];
            if(v < minVal[j]) minVal[j] = v;
            if(v > maxVal[j]) maxVal[j] = v;
        }
    }

    /* Find min and max for targets */
    *min_target = 1e9f;
    *max_target = -1e9f;
    for(int i = 0; i < count; i++){
        float target = targets[i][0];
        if(target < *min_target) *min_target = target;
        if(target > *max_target) *max_target = target;
    }

    /* Normalize inputs */
    for(int i = 0; i < count; i++){
        for(int j = 0; j < INPUT_SIZE; j++){
            float denom = (maxVal[j] - minVal[j]);
            if(denom < 1e-6f) denom = 1.0f; /* Prevent division by zero */
            normalized_inputs[i][j] = (inputs[i][j] - minVal[j]) / denom;
            if(isnan(normalized_inputs[i][j]) || isinf(normalized_inputs[i][j])){
                fprintf(stderr, "Normalization resulted in invalid value at sample %d, feature %d.\n", i, j);
                exit(1);
            }
        }
    }

    /* Normalize targets */
    for(int i = 0; i < count; i++){
        normalized_targets[i][0] = (targets[i][0] - *min_target) / (*max_target - *min_target);
        if(isnan(normalized_targets[i][0]) || isinf(normalized_targets[i][0])){
            fprintf(stderr, "Normalization resulted in invalid target at sample %d.\n", i);
            exit(1);
        }
    }
}

/***************************************
 * LSTM Forward Pass
 ***************************************/
static void lstm_forward_pass(LSTMModel *model, float normalized_inputs[][INPUT_SIZE], int seq_len) {
    for(int layer = 0; layer < NUM_LSTM_LAYERS; layer++) {
        LSTMLayerParams *params = &model->layers[layer];
        LSTMLayerCache *cache = &model->cache[layer];
        int current_input_size = (layer == 0) ? INPUT_SIZE : HIDDEN_SIZE; /* Previous layer's hidden size */

        float h_prev[HIDDEN_SIZE];
        float c_prev[HIDDEN_SIZE];
        memset(h_prev, 0, sizeof(h_prev));
        memset(c_prev, 0, sizeof(c_prev));

        for(int t = 0; t < seq_len; t++) {
            /* Get input for this layer */
            float *current_input_ptr;
            if(layer == 0)
                current_input_ptr = normalized_inputs[t];
            else
                current_input_ptr = model->cache[layer - 1].h_state[t];

            /* Compute gate inputs */
            float i_in[HIDDEN_SIZE], f_in[HIDDEN_SIZE], o_in[HIDDEN_SIZE], c_in[HIDDEN_SIZE];
            for(int i = 0; i < HIDDEN_SIZE; i++) {
                i_in[i] = params->b_i[i];
                f_in[i] = params->b_f[i];
                o_in[i] = params->b_o[i];
                c_in[i] = params->b_c[i];
            }

            /* Input and hidden contributions */
            for(int i = 0; i < HIDDEN_SIZE; i++) {
                for(int j = 0; j < current_input_size; j++) {
                    i_in[i] += params->W_ix[i][j] * current_input_ptr[j];
                    f_in[i] += params->W_fx[i][j] * current_input_ptr[j];
                    o_in[i] += params->W_ox[i][j] * current_input_ptr[j];
                    c_in[i] += params->W_cx[i][j] * current_input_ptr[j];
                }
                for(int j = 0; j < HIDDEN_SIZE; j++) {
                    i_in[i] += params->W_ih[i][j] * h_prev[j];
                    f_in[i] += params->W_fh[i][j] * h_prev[j];
                    o_in[i] += params->W_oh[i][j] * h_prev[j];
                    c_in[i] += params->W_ch[i][j] * h_prev[j];
                }
            }

            /* Activation functions */
            for(int i = 0; i < HIDDEN_SIZE; i++) {
                cache->i_gate[t][i] = sigmoid_act(i_in[i]);
                cache->f_gate[t][i] = sigmoid_act(f_in[i]);
                cache->o_gate[t][i] = sigmoid_act(o_in[i]);
                cache->c_hat[t][i] = tanh_act(c_in[i]);
            }

            /* Cell and hidden states */
            for(int i = 0; i < HIDDEN_SIZE; i++) {
                cache->c_state[t][i] = cache->f_gate[t][i] * c_prev[i] + cache->i_gate[t][i] * cache->c_hat[t][i];
                cache->h_state[t][i] = tanh_act(cache->c_state[t][i]) * cache->o_gate[t][i];
            }

            /* Output layer */
            for(int i = 0; i < OUTPUT_SIZE; i++) {
                float sum = params->b_y[i];
                for(int j = 0; j < HIDDEN_SIZE; j++) {
                    sum += params->W_hy[i][j] * cache->h_state[t][j];
                }
                cache->y_pred[t][i] = sum; /* Linear activation */
            }

            /* Update previous states */
            for(int i = 0; i < HIDDEN_SIZE; i++) {
                h_prev[i] = cache->h_state[t][i];
                c_prev[i] = cache->c_state[t][i];
            }

            /* Debug: Print activations for first few time steps and layers */
            if(t < 5 && layer == 0){
                printf("Layer %d, Time Step %d:\n", layer+1, t);
                printf("  i_gate[0]=%.3f, f_gate[0]=%.3f, o_gate[0]=%.3f, c_hat[0]=%.3f\n",
                    cache->i_gate[t][0], cache->f_gate[t][0], cache->o_gate[t][0], cache->c_hat[t][0]);
                printf("  c_state[0]=%.3f, h_state[0]=%.3f\n",
                    cache->c_state[t][0], cache->h_state[t][0]);
            }
        }
    }
}

/***************************************
 * LSTM Backward Pass
 ***************************************/
static float lstm_backward_pass(LSTMModel *model, float targets[][OUTPUT_SIZE], int seq_len) {
    float total_loss = 0.0f;
    /* Initialize gradients for all layers */
    zero_grads(model);

    /* Initialize variables for backpropagation */
    float dh_next[NUM_LSTM_LAYERS][HIDDEN_SIZE];
    float dc_next[NUM_LSTM_LAYERS][HIDDEN_SIZE];
    memset(dh_next, 0, sizeof(dh_next));
    memset(dc_next, 0, sizeof(dc_next));

    /* Iterate over time steps in reverse */
    for(int t = seq_len -1; t >=0; t--) {
        /* Calculate loss */
        float y_pred = model->cache[NUM_LSTM_LAYERS-1].y_pred[t][0];
        float y_true = targets[t][0];
        float error = y_pred - y_true;
        float loss = 0.5f * error * error;
        total_loss += loss;

        /* Output layer gradients */
        for(int layer = NUM_LSTM_LAYERS -1; layer >=0; layer--) {
            LSTMLayerParams *params = &model->layers[layer];
            LSTMLayerGrads *grads = &model->grads[layer];
            LSTMLayerCache *cache = &model->cache[layer];

            /* Gradient of loss w.r.t y_pred */
            float dy = error;

            /* Gradient w.r.t W_hy and b_y */
            for(int i = 0; i < OUTPUT_SIZE; i++) {
                for(int j = 0; j < HIDDEN_SIZE; j++) {
                    grads->W_hy[i][j] += dy * cache->h_state[t][j];
                }
                grads->b_y[i] += dy;
            }

            /* Gradient w.r.t hidden state */
            float dh[HIDDEN_SIZE];
            for(int j = 0; j < HIDDEN_SIZE; j++) {
                dh[j] = params->W_hy[0][j] * dy + dh_next[layer][j];
            }

            /* Backprop through output gate */
            float do_[HIDDEN_SIZE];
            for(int j = 0; j < HIDDEN_SIZE; j++) {
                do_[j] = tanh_act(cache->c_state[t][j]) * dh[j];
            }

            /* Backprop through cell state */
            float dc[HIDDEN_SIZE];
            for(int j = 0; j < HIDDEN_SIZE; j++) {
                float tanhc = tanh_act(cache->c_state[t][j]);
                dc[j] = params->W_hy[0][j] * dy * cache->o_gate[t][j] * dtanh_act(tanhc);
                dc[j] += dh[j] * cache->o_gate[t][j] * dtanh_act(tanhc);
                dc[j] += dc_next[layer][j];
            }

            /* Backprop through gates */
            float di[HIDDEN_SIZE], df[HIDDEN_SIZE], dc_hat[HIDDEN_SIZE];
            for(int j = 0; j < HIDDEN_SIZE; j++) {
                di[j] = dc[j] * cache->c_hat[t][j] * dsigmoid(cache->i_gate[t][j]);
                df[j] = dc[j] * cache->f_gate[t][j] * dsigmoid(cache->f_gate[t][j]);
                dc_hat[j] = dc[j] * cache->i_gate[t][j] * dtanh_act(cache->c_hat[t][j]);
            }

            /* Accumulate gradients */
            for(int j = 0; j < HIDDEN_SIZE; j++) {
                for(int k = 0; k < INPUT_SIZE; k++) {
                    grads->W_ix[j][k] += di[j] * ((layer == 0) ? 0 : model->cache[layer-1].h_state[t][k]);
                    grads->W_fx[j][k] += df[j] * ((layer == 0) ? 0 : model->cache[layer-1].h_state[t][k]);
                    grads->W_ox[j][k] += do_[j] * ((layer == 0) ? 0 : model->cache[layer-1].h_state[t][k]);
                    grads->W_cx[j][k] += dc_hat[j] * ((layer == 0) ? 0 : model->cache[layer-1].h_state[t][k]);
                }
                for(int k = 0; k < HIDDEN_SIZE; k++) {
                    grads->W_ih[j][k] += di[j] * ((t > 0) ? model->cache[layer].h_state[t-1][k] : 0.0f);
                    grads->W_fh[j][k] += df[j] * ((t > 0) ? model->cache[layer].h_state[t-1][k] : 0.0f);
                    grads->W_oh[j][k] += do_[j] * ((t > 0) ? model->cache[layer].h_state[t-1][k] : 0.0f);
                    grads->W_ch[j][k] += dc_hat[j] * ((t > 0) ? model->cache[layer].h_state[t-1][k] : 0.0f);
                }
                grads->b_i[j] += di[j];
                grads->b_f[j] += df[j];
                grads->b_o[j] += do_[j];
                grads->b_c[j] += dc_hat[j];
            }

            /* Update dh_next and dc_next */
            for(int j = 0; j < HIDDEN_SIZE; j++) {
                dh_next[layer][j] = 0.0f;
                dc_next[layer][j] = 0.0f;
                for(int k = 0; k < HIDDEN_SIZE; k++) {
                    dh_next[layer][j] += model->layers[layer].W_ih[j][k] * di[k];
                    dh_next[layer][j] += model->layers[layer].W_fh[j][k] * df[k];
                    dh_next[layer][j] += model->layers[layer].W_oh[j][k] * do_[k];
                    dh_next[layer][j] += model->layers[layer].W_ch[j][k] * dc_hat[k];
                }
            }

            /* Debug: Print gradients for first few time steps and layers */
            if(t < 5 && layer == 0){
                printf("Backward Layer %d, Time Step %d:\n", layer+1, t);
                printf("  Gradient di[0]=%.3f, df[0]=%.3f, dc_hat[0]=%.3f\n",
                    di[0], df[0], dc_hat[j]);
                printf("  Gradient do_[0]=%.3f\n", do_[0]);
            }
        }
    }

    return total_loss;
}

/***************************************
 * LSTM Forward Pass for Validation
 ***************************************/
static void lstm_forward_validation(LSTMModel *model, float normalized_inputs[][INPUT_SIZE], int seq_len) {
    lstm_forward_pass(model, normalized_inputs, seq_len);
}

/***************************************
 * Main Function
 ***************************************/
int main(int argc, char *argv[]){
    if(argc < 2){
        fprintf(stderr, "Usage: %s path/to/stock_data.csv\n", argv[0]);
        return 1;
    }
    srand((unsigned)time(NULL));

    /* 1) Load intraday CSV, aggregate daily bars */
    DailyBar dailyData[MAX_SAMPLES];
    int rawCount = load_and_aggregate_daily(argv[1], dailyData, MAX_SAMPLES);
    if(rawCount <= 1){
        fprintf(stderr, "No valid daily bars found in CSV.\n");
        return 1;
    }

    /* 2) Calculate Technical Indicators */
    TechnicalIndicators indicators;
    calculate_obv(dailyData, rawCount, indicators.obv);
    calculate_ad(dailyData, rawCount, indicators.ad);
    calculate_adx(dailyData, rawCount, indicators.adx);
    calculate_aroon(dailyData, rawCount, indicators.aroonUp, indicators.aroonDown);
    calculate_macd(dailyData, rawCount, indicators.macd);
    calculate_rsi(dailyData, rawCount, indicators.rsi);

    /* 3) Build input features and targets */
    float inputs[MAX_SAMPLES][INPUT_SIZE];
    float targets_raw[MAX_SAMPLES][OUTPUT_SIZE]; // Raw targets
    int validCount = 0;
    for(int i = 0; i < rawCount; i++){
        /* Basic check: ensure all indicators are calculated */
        if(i < 26 || indicators.adx[i] == 0.0f || indicators.macd[i] == 0.0f || indicators.rsi[i] == 0.0f){
            continue;
        }
        inputs[validCount][0]  = dailyData[i].open;
        inputs[validCount][1]  = dailyData[i].high;
        inputs[validCount][2]  = dailyData[i].low;
        inputs[validCount][3]  = dailyData[i].close;
        inputs[validCount][4]  = dailyData[i].volume;
        inputs[validCount][5]  = dailyData[i].high - dailyData[i].low; /* Range */
        inputs[validCount][6]  = indicators.obv[i];
        inputs[validCount][7]  = indicators.ad[i];
        inputs[validCount][8]  = indicators.adx[i];
        inputs[validCount][9]  = indicators.aroonUp[i];
        inputs[validCount][10] = indicators.aroonDown[i];
        inputs[validCount][11] = indicators.macd[i];
        inputs[validCount][12] = indicators.rsi[i];

        /* Target is the next day's close */
        if(i < rawCount - 1){
            targets_raw[validCount][0] = dailyData[i+1].close;
        }
        else{
            targets_raw[validCount][0] = dailyData[i].close; /* Last target */
        }
        validCount++;
    }

    if(validCount <= VALIDATION_SIZE){
        fprintf(stderr, "Not enough valid data after processing.\n");
        return 1;
    }

    printf("Total valid daily bars used: %d\n", validCount);
    printf("First day: %s O=%.2f H=%.2f L=%.2f C=%.2f V=%.0f\n",
        dailyData[0].date, dailyData[0].open, dailyData[0].high, dailyData[0].low,
        dailyData[0].close, dailyData[0].volume);
    printf("Last day:  %s O=%.2f H=%.2f L=%.2f C=%.2f V=%.0f\n",
        dailyData[validCount-1].date, dailyData[validCount-1].open,
        dailyData[validCount-1].high, dailyData[validCount-1].low,
        dailyData[validCount-1].close, dailyData[validCount-1].volume);

    /* 4) Normalize Inputs and Targets */
    float minVal[INPUT_SIZE], maxVal[INPUT_SIZE];
    float min_target, max_target;
    float normalized_inputs[MAX_SAMPLES][INPUT_SIZE];
    float normalized_targets[MAX_SAMPLES][OUTPUT_SIZE];
    normalize_data(inputs, targets_raw, validCount, minVal, maxVal, &min_target, &max_target,
                  normalized_inputs, normalized_targets);

    /* Debug: Print min and max targets */
    printf("\nTarget Min: %.2f, Target Max: %.2f\n", min_target, max_target);

    /* Debug: Print first 5 normalized targets */
    printf("\nNormalized Targets (First 5 Samples):\n");
    for(int i = 0; i < 5 && i < validCount; i++) {
        printf("Sample %d: %.3f\n", i, normalized_targets[i][0]);
    }

    /* 5) Initialize LSTM Model */
    LSTMModel model;
    init_lstm_params(&model, INPUT_SIZE, HIDDEN_SIZE);
    init_adam_parameters(&model);
    init_cache(&model);

    /* 6) Split data into training and validation */
    int trainLen = validCount - VALIDATION_SIZE;
    if(trainLen < 2){
        trainLen = validCount; /* Fallback */
    }

    /* 7) Training Loop with Advanced Features */
    EarlyStopping es;
    init_early_stopping(&es);

    for(int epoch = 1; epoch <= EPOCHS; epoch++){
        /* Forward Pass */
        lstm_forward_pass(&model, normalized_inputs, trainLen);

        /* Compute Loss and Backward Pass */
        float epoch_loss = lstm_backward_pass(&model, normalized_targets, trainLen);

        /* Apply L2 Regularization */
        apply_l2_regularization(&model, L2_LAMBDA);

        /* Update Parameters with Adam */
        float current_lr = get_learning_rate(epoch);
        update_parameters_adam(&model, epoch);

        /* Apply Gradient Clipping */
        clip_grads(&model, CLIP_VALUE);

        /* Calculate Average Loss */
        float avg_loss = epoch_loss / trainLen;

        /* Early Stopping based on Validation Loss */
        /* Forward pass on validation set */
        lstm_forward_validation(&model, normalized_inputs, validCount);
        float validation_loss = 0.0f;
        for(int t = trainLen; t < validCount; t++) {
            float y_pred_norm = model.cache[NUM_LSTM_LAYERS-1].y_pred[t][0];
            float y_true_norm = normalized_targets[t][0];
            float diff = y_pred_norm - y_true_norm;
            validation_loss += 0.5f * diff * diff;
        }
        float avg_val_loss = validation_loss / VALIDATION_SIZE;
        update_early_stopping(&es, avg_val_loss);

        /* Print training progress */
        if(epoch % 100 == 0 || epoch == 1){
            printf("Epoch %4d, Train Loss=%.6f, Val Loss=%.6f, LR=%.6f\n", epoch, avg_loss, avg_val_loss, current_lr);
        }

        /* Early stopping condition */
        if(es.stop){
            printf("Early stopping triggered at epoch %d\n", epoch);
            break;
        }
    }

    /* 8) Validation */
    lstm_forward_validation(&model, normalized_inputs, validCount);
    printf("\nValidation (Last %d Days):\n", VALIDATION_SIZE);
    printf("-------------------------------------------------------------\n");
    printf("| Day |     Date     | Predicted Close | Actual Close | Error |\n");
    printf("-------------------------------------------------------------\n");
    int valStart = trainLen;
    float total_mae = 0.0f;
    float total_rmse = 0.0f;
    for(int t = valStart; t < validCount; t++){
        float y_pred_norm = model.cache[NUM_LSTM_LAYERS-1].y_pred[t][0];
        /* Denormalize prediction */
        float y_pred_raw = y_pred_norm * (max_target - min_target) + min_target;
        float y_true = dailyData[t].close;
        float error = fabsf(y_pred_raw - y_true);
        total_mae += error;
        total_rmse += error * error;
        printf("| %3d | %10s |     %8.2f     |    %8.2f   | %5.2f |\n",
            t, dailyData[t].date, y_pred_raw, y_true, error);
    }
    printf("-------------------------------------------------------------\n");

    /* 9) Calculate Validation Metrics */
    float mae = total_mae / VALIDATION_SIZE;
    float rmse = sqrtf(total_rmse / VALIDATION_SIZE);
    printf("\nValidation Metrics:\n");
    printf("Mean Absolute Error (MAE): %.2f\n", mae);
    printf("Root Mean Squared Error (RMSE): %.2f\n", rmse);

    /* 10) Pretty Output for All Data */
    printf("\nDetailed Predictions for All Data:\n");
    printf("--------------------------------------------------------------------\n");
    printf("| Day |     Date     | Predicted Close | Actual Close | Error |\n");
    printf("--------------------------------------------------------------------\n");
    for(int t = 0; t < validCount; t++){
        float y_pred_norm = model.cache[NUM_LSTM_LAYERS-1].y_pred[t][0];
        /* Denormalize prediction */
        float y_pred_raw = y_pred_norm * (max_target - min_target) + min_target;
        float y_true = dailyData[t].close;
        float error = fabsf(y_pred_raw - y_true);
        printf("| %3d | %10s |     %8.2f     |    %8.2f   | %5.2f |\n",
            t, dailyData[t].date, y_pred_raw, y_true, error);
    }
    printf("--------------------------------------------------------------------\n");

    return 0;
}

