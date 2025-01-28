#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>

/********************************************************
 * Hyperparameters
 ********************************************************/
#define BATCH_SIZE   1    /* We'll do one sequence at a time for simplicity */
#define MAX_SEQ_LEN  50   /* Maximum length of the time series */
#define INPUT_SIZE   8    /* # of input features (macro/micro/technical + current price) */
#define HIDDEN_SIZE  6
#define OUTPUT_SIZE  1    /* Predict next-day price */
#define EPOCHS       200
#define LEARNING_RATE 0.01f

/********************************************************
 * LSTM Parameter Structures
 ********************************************************/
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
    /* Output layer (h->y) */
    float W_hy[OUTPUT_SIZE][HIDDEN_SIZE];
    float b_y[OUTPUT_SIZE];
} LSTMParams;

typedef struct {
    float W_ix[HIDDEN_SIZE][INPUT_SIZE];
    float W_ih[HIDDEN_SIZE][HIDDEN_SIZE];
    float b_i[HIDDEN_SIZE];
    float W_fx[HIDDEN_SIZE][INPUT_SIZE];
    float W_fh[HIDDEN_SIZE][HIDDEN_SIZE];
    float b_f[HIDDEN_SIZE];
    float W_ox[HIDDEN_SIZE][INPUT_SIZE];
    float W_oh[HIDDEN_SIZE][HIDDEN_SIZE];
    float b_o[HIDDEN_SIZE];
    float W_cx[HIDDEN_SIZE][INPUT_SIZE];
    float W_ch[HIDDEN_SIZE][HIDDEN_SIZE];
    float b_c[HIDDEN_SIZE];
    float W_hy[OUTPUT_SIZE][HIDDEN_SIZE];
    float b_y[OUTPUT_SIZE];
} LSTMGrads;

typedef struct {
    float x[MAX_SEQ_LEN][INPUT_SIZE];
    float i[MAX_SEQ_LEN][HIDDEN_SIZE];
    float f[MAX_SEQ_LEN][HIDDEN_SIZE];
    float o[MAX_SEQ_LEN][HIDDEN_SIZE];
    float c_hat[MAX_SEQ_LEN][HIDDEN_SIZE];
    float c[MAX_SEQ_LEN][HIDDEN_SIZE];
    float h[MAX_SEQ_LEN][HIDDEN_SIZE];
    float y[MAX_SEQ_LEN][OUTPUT_SIZE];
} LSTMCache;

/********************************************************
 * Utilities
 ********************************************************/
static inline float randf(float range) {
    float r = (float)rand() / (float)RAND_MAX;
    return (r*2.0f - 1.0f)*range;
}
static inline float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}
static inline float dsigmoid_from_val(float val) {
    return val*(1.0f - val);
}
static inline float tanh_approx(float x) {
    return tanhf(x);
}
static inline float dtanh_from_val(float val) {
    return 1.0f - val*val;
}

static void init_params(LSTMParams *p, float range) {
    float *pf = (float*)p;
    int count = sizeof(LSTMParams)/sizeof(float);
    for(int i=0; i<count; i++){
        pf[i] = randf(range);
    }
}

static void zero_grads(LSTMGrads *g) {
    memset(g, 0, sizeof(LSTMGrads));
}

static void update_params(LSTMParams *params, LSTMGrads *grads) {
    float *p = (float*)params;
    float *r = (float*)grads;
    int count = sizeof(LSTMParams)/sizeof(float);
    for(int i=0; i<count; i++){
        p[i] -= LEARNING_RATE * r[i];
    }
}

/********************************************************
 * Forward Pass for a single sequence
 ********************************************************/
static void lstm_forward(
    LSTMParams *params,
    float inputs[][INPUT_SIZE], /* shape [seq_len][INPUT_SIZE] */
    int seq_len,
    LSTMCache *cache,
    float *h0, float *c0
){
    float h_prev[HIDDEN_SIZE], c_prev[HIDDEN_SIZE];
    memcpy(h_prev, h0, sizeof(float)*HIDDEN_SIZE);
    memcpy(c_prev, c0, sizeof(float)*HIDDEN_SIZE);

    for(int t=0; t<seq_len; t++){
        memcpy(cache->x[t], inputs[t], INPUT_SIZE*sizeof(float));

        /* Gate pre-activations */
        float i_in[HIDDEN_SIZE], f_in[HIDDEN_SIZE], o_in[HIDDEN_SIZE], c_in[HIDDEN_SIZE];
        for(int i=0; i<HIDDEN_SIZE; i++){
            i_in[i] = params->b_i[i];
            f_in[i] = params->b_f[i];
            o_in[i] = params->b_o[i];
            c_in[i] = params->b_c[i];
        }

        /* Add x->(gate) and h_prev->(gate) contributions */
        for(int i=0; i<HIDDEN_SIZE; i++){
            for(int j=0; j<INPUT_SIZE; j++){
                float x_val = inputs[t][j];
                i_in[i] += params->W_ix[i][j]*x_val;
                f_in[i] += params->W_fx[i][j]*x_val;
                o_in[i] += params->W_ox[i][j]*x_val;
                c_in[i] += params->W_cx[i][j]*x_val;
            }
            for(int j=0; j<HIDDEN_SIZE; j++){
                float h_val = h_prev[j];
                i_in[i] += params->W_ih[i][j]*h_val;
                f_in[i] += params->W_fh[i][j]*h_val;
                o_in[i] += params->W_oh[i][j]*h_val;
                c_in[i] += params->W_ch[i][j]*h_val;
            }
        }

        /* Activations */
        for(int i=0; i<HIDDEN_SIZE; i++){
            cache->i[t][i] = sigmoid(i_in[i]);
            cache->f[t][i] = sigmoid(f_in[i]);
            cache->o[t][i] = sigmoid(o_in[i]);
            cache->c_hat[t][i] = tanh_approx(c_in[i]);
        }

        /* Cell state */
        for(int i=0; i<HIDDEN_SIZE; i++){
            cache->c[t][i] = cache->f[t][i]*c_prev[i] + cache->i[t][i]*cache->c_hat[t][i];
        }

        /* Hidden state */
        for(int i=0; i<HIDDEN_SIZE; i++){
            cache->h[t][i] = cache->o[t][i]*tanh_approx(cache->c[t][i]);
        }

        /* Output layer */
        for(int out_i=0; out_i<OUTPUT_SIZE; out_i++){
            float sum = params->b_y[out_i];
            for(int j=0; j<HIDDEN_SIZE; j++){
                sum += params->W_hy[out_i][j]*cache->h[t][j];
            }
            cache->y[t][out_i] = sum;
        }

        /* Update h_prev, c_prev */
        for(int i=0; i<HIDDEN_SIZE; i++){
            h_prev[i] = cache->h[t][i];
            c_prev[i] = cache->c[t][i];
        }
    }
}

/********************************************************
 * Backprop (MSE loss)
 * Next-day price => target[t] = actual price at t+1
 ********************************************************/
static float lstm_backward(
    LSTMParams *params,
    LSTMCache *cache,
    float targets[][OUTPUT_SIZE],
    int seq_len,
    LSTMGrads *grads
){
    float loss = 0.f;
    /* We'll store dh_next, dc_next for each time */
    float dh_next[HIDDEN_SIZE], dc_next[HIDDEN_SIZE];
    memset(dh_next, 0, sizeof(dh_next));
    memset(dc_next, 0, sizeof(dc_next));

    for(int t=seq_len-1; t>=0; t--){
        if(t == seq_len-1){
            /* last step has no next-day target */
            continue;
        }

        /* MSE derivative wrt output */
        float y_pred = cache->y[t][0];
        float y_true = targets[t+1][0]; /* next-day price */
        float diff = (y_pred - y_true);
        float dy = diff;
        loss += 0.5f*diff*diff;

        /* Output layer grads => dW_hy, db_y, dh(t) */
        float dh[HIDDEN_SIZE];
        memset(dh, 0, sizeof(dh));
        for(int i=0; i<HIDDEN_SIZE; i++){
            grads->W_hy[0][i] += dy*cache->h[t][i];
            dh[i] = dy*params->W_hy[0][i];
        }
        grads->b_y[0] += dy;

        /* Add dh_next from future time step */
        for(int i=0; i<HIDDEN_SIZE; i++){
            dh[i] += dh_next[i];
        }

        /* h(t) = o(t)*tanh(c(t)) => do(t), dc(t) */
        float do_[HIDDEN_SIZE], dc[HIDDEN_SIZE];
        for(int i=0; i<HIDDEN_SIZE; i++){
            float o_val = cache->o[t][i];
            float c_val = cache->c[t][i];
            float tanhc = tanh_approx(c_val);
            do_[i] = tanhc*dh[i];
            dc[i] = o_val*(1.0f - tanhc*tanhc)*dh[i];
        }

        /* plus dc_next */
        for(int i=0; i<HIDDEN_SIZE; i++){
            dc[i] += dc_next[i];
        }

        /* c(t)=f(t)*c(t-1)+i(t)*c_hat(t) => df, di, dc_hat */
        float di[HIDDEN_SIZE], df[HIDDEN_SIZE], dc_hat[HIDDEN_SIZE];
        for(int i=0; i<HIDDEN_SIZE; i++){
            float c_prev = (t==0)? 0.0f : cache->c[t-1][i];
            di[i] = dc[i]*cache->c_hat[t][i];
            df[i] = dc[i]*c_prev;
            dc_hat[i] = dc[i]*cache->i[t][i];
        }

        /* convert to pre-activation space => multiply by derivative of sigmoid or tanh */
        float do_in[HIDDEN_SIZE], di_in[HIDDEN_SIZE], df_in[HIDDEN_SIZE], dc_in[HIDDEN_SIZE];
        for(int i=0; i<HIDDEN_SIZE; i++){
            do_in[i] = do_[i]*dsigmoid_from_val(cache->o[t][i]);
            di_in[i] = di[i]*dsigmoid_from_val(cache->i[t][i]);
            df_in[i] = df[i]*dsigmoid_from_val(cache->f[t][i]);
            float ch_val = cache->c_hat[t][i];
            float dch = dc_hat[i]*(1.0f - ch_val*ch_val);
            dc_in[i] = dch;
        }

        /* Accumulate gate weight grads => W_ix, W_ih, etc. */
        float dh_prev[HIDDEN_SIZE];
        memset(dh_prev, 0, sizeof(dh_prev));

        float *x_t = cache->x[t];
        float *h_t_1 = (t==0)? NULL : cache->h[t-1];

        #define ACCUM_GATE(Wx,Wh,bias, d_in)                                   \
            do {                                                               \
                for(int i=0; i<HIDDEN_SIZE; i++){                              \
                    float dval = d_in[i];                                      \
                    for(int j=0; j<INPUT_SIZE; j++){                           \
                        grads->Wx[i][j] += dval*x_t[j];                        \
                    }                                                          \
                    if(h_t_1 != NULL){                                         \
                        for(int j=0; j<HIDDEN_SIZE; j++){                      \
                            grads->Wh[i][j] += dval*h_t_1[j];                  \
                            dh_prev[j] += dval*params->Wh[i][j];              \
                        }                                                      \
                    }                                                          \
                    grads->bias[i] += dval;                                    \
                }                                                              \
            } while(0)

        ACCUM_GATE(W_ix, W_ih, b_i, di_in);
        ACCUM_GATE(W_fx, W_fh, b_f, df_in);
        ACCUM_GATE(W_ox, W_oh, b_o, do_in);
        ACCUM_GATE(W_cx, W_ch, b_c, dc_in);

        /* handle dh_prev if no h(t-1). We skip for t=0 => no real backprop to h(-1). */

        /* c(t-1) => pass to dc_next */
        float dc_prev[HIDDEN_SIZE];
        for(int i=0; i<HIDDEN_SIZE; i++){
            float f_val = cache->f[t][i];
            dc_prev[i] = dc[i]*f_val;
        }

        memcpy(dh_next, dh_prev, sizeof(dh_prev));
        memcpy(dc_next, dc_prev, sizeof(dc_prev));
    }
    return loss;
}

/********************************************************
 * Synthetic Data Generation
 * - We add cyclical (daily) signals plus random noise
 * - "Price" evolves with random walk
 ********************************************************/
static void generate_synthetic_data(
    float inputs[][INPUT_SIZE],
    float targets[][OUTPUT_SIZE],
    int seq_len
){
    float price = 10.0f + (rand()%100)/10.f;

    for(int t=0; t<seq_len; t++){
        /* Macro/Micro/Tech factors (7 of them) */
        float dayCycle = sinf(2.0f*3.14159f*t/30.0f)*0.5f + 0.5f; // daily-ish cycle
        for(int i=0; i<INPUT_SIZE-1; i++){
            /* random in [0..1], modulated by dayCycle */
            inputs[t][i] = ((float)rand()/RAND_MAX + dayCycle)/2.f;
        }

        /* Store current price as last feature */
        inputs[t][INPUT_SIZE-1] = price;

        /* Random walk step for next price */
        float noise = (float)(rand()%100 - 50)/100.0f; // in [-0.5..0.5]
        float next_price = price + noise;
        if(next_price < 0) next_price = 0;

        targets[t][0] = next_price;  /* This is the "future" price, used at time t-1 -> t */
        price = next_price;
    }
}

/********************************************************
 * Main
 ********************************************************/
int main(){
    srand((unsigned)time(NULL));

    LSTMParams params;
    init_params(&params, 0.1f);

    LSTMGrads grads;
    zero_grads(&grads);

    LSTMCache cache;
    memset(&cache, 0, sizeof(cache));

    /* We'll train on a single sequence (BATCH_SIZE=1), length=MAX_SEQ_LEN. */
    int seq_len = MAX_SEQ_LEN;
    float inputs[MAX_SEQ_LEN][INPUT_SIZE];
    float targets[MAX_SEQ_LEN][OUTPUT_SIZE];

    generate_synthetic_data(inputs, targets, seq_len);

    /* Hidden, cell init */
    float h0[HIDDEN_SIZE], c0[HIDDEN_SIZE];
    memset(h0, 0, sizeof(h0));
    memset(c0, 0, sizeof(c0));

    /* Training */
    for(int e=1; e<=EPOCHS; e++){
        zero_grads(&grads);

        /* Forward */
        lstm_forward(&params, inputs, seq_len, &cache, h0, c0);

        /* Backward (MSE) */
        float loss = lstm_backward(&params, &cache, targets, seq_len, &grads);

        /* Update */
        update_params(&params, &grads);

        if(e%20==0){
            printf("Epoch %3d, Loss=%.4f\n", e, loss);
        }
    }

    /* Final predictions */
    printf("\nFinal Predictions:\n");
    for(int t=0; t<seq_len; t++){
        float y_pred = cache.y[t][0];
        if(t < seq_len-1){
            float y_true = targets[t+1][0];
            printf(" t=%2d: pred=%.3f, next-day-true=%.3f\n", t, y_pred, y_true);
        } else {
            printf(" t=%2d: pred=%.3f, next-day-true=N/A (last step)\n", t, y_pred);
        }
    }

    return 0;
}

