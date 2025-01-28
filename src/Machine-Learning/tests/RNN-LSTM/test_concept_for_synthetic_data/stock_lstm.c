/*******************************************************************************
 * stock_lstm_extended.c
 *
 * Demonstrates:
 *   - Synthetic data for multiple sequences
 *   - Next-day price prediction (shifted targets)
 *   - Variable sequence lengths
 *   - A single-layer LSTM with BPTT
 *
 * Compile:
 *   cc -o stock_lstm_extended stock_lstm_extended.c -lm
 *
 * Run:
 *   ./stock_lstm_extended
 *******************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

/***********************************
 * Hyperparameters
 ***********************************/
#define MAX_SEQ_LEN   50    // max length of a sequence
#define BATCH_SIZE    3     // how many sequences in this batch
#define INPUT_SIZE    8     // e.g., G(t), I(t), R(t), E(t), S(t), C(t), M(t), V(t)
#define HIDDEN_SIZE   6     // number of LSTM hidden units
#define OUTPUT_SIZE   1     // predict next-day price as a single value
#define EPOCHS        200
#define LR            0.01f // learning rate

/***********************************
 * Data Structures
 ***********************************/
// LSTM parameters
typedef struct {
    // input gate
    float W_ix[HIDDEN_SIZE][INPUT_SIZE];
    float W_ih[HIDDEN_SIZE][HIDDEN_SIZE];
    float b_i[HIDDEN_SIZE];
    // forget gate
    float W_fx[HIDDEN_SIZE][INPUT_SIZE];
    float W_fh[HIDDEN_SIZE][HIDDEN_SIZE];
    float b_f[HIDDEN_SIZE];
    // output gate
    float W_ox[HIDDEN_SIZE][INPUT_SIZE];
    float W_oh[HIDDEN_SIZE][HIDDEN_SIZE];
    float b_o[HIDDEN_SIZE];
    // candidate gate
    float W_cx[HIDDEN_SIZE][INPUT_SIZE];
    float W_ch[HIDDEN_SIZE][HIDDEN_SIZE];
    float b_c[HIDDEN_SIZE];
    // output layer (h -> y)
    float W_hy[OUTPUT_SIZE][HIDDEN_SIZE];
    float b_y[OUTPUT_SIZE];
} LSTMParams;

// LSTM parameter gradients
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

/* We store intermediate values for each time step **per sequence**.
   So dimension: [batch_size][max_seq_len].
*/
typedef struct {
    // input (copied for convenience)
    float x[BATCH_SIZE][MAX_SEQ_LEN][INPUT_SIZE];
    // gates
    float i[BATCH_SIZE][MAX_SEQ_LEN][HIDDEN_SIZE];
    float f[BATCH_SIZE][MAX_SEQ_LEN][HIDDEN_SIZE];
    float o[BATCH_SIZE][MAX_SEQ_LEN][HIDDEN_SIZE];
    float c_hat[BATCH_SIZE][MAX_SEQ_LEN][HIDDEN_SIZE];
    // states
    float c[BATCH_SIZE][MAX_SEQ_LEN][HIDDEN_SIZE];
    float h[BATCH_SIZE][MAX_SEQ_LEN][HIDDEN_SIZE];
    // output
    float y[BATCH_SIZE][MAX_SEQ_LEN][OUTPUT_SIZE];
} LSTMCache;

/***********************************
 * Math helpers
 ***********************************/
static inline float randf(float range) {
    // random in [-range, range]
    float r = (float)rand() / (float)RAND_MAX;
    return (r*2.0f - 1.0f)*range;
}

static inline float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}
static inline float dsigmoid_from_val(float s) {
    // derivative given the *sigmoid value* s, i.e. s*(1-s)
    return s*(1.0f - s);
}
static inline float tanh_approx(float x) {
    return tanhf(x);
}
static inline float dtanh_from_val(float tval) {
    // derivative given tanh(t) = tval => 1 - tval^2
    return 1.0f - tval*tval;
}

/***********************************
 * Parameter Initialization
 ***********************************/
void init_params(LSTMParams *p, float range) {
    float *pf = (float*)p;
    int count = sizeof(LSTMParams)/sizeof(float);
    for(int i=0; i<count; i++){
        pf[i] = randf(range);
    }
}
void zero_grads(LSTMGrads *g) {
    memset(g, 0, sizeof(LSTMGrads));
}

/***********************************
 * Forward Pass (per sequence)
 *   We handle each sequence in the batch independently.
 ***********************************/
void lstm_forward(
    LSTMParams *params,
    LSTMCache *cache,
    float **inputs,       // inputs[s] is a pointer to array of (seq_len_s * INPUT_SIZE)
    int *seq_len,         // lengths of each sequence
    float **h0, float **c0, // initial hidden and cell states, shape [batch_size][HIDDEN_SIZE]
    int batch_size
)
{
    // For each sequence in the batch
    for(int s=0; s<batch_size; s++){
        int L = seq_len[s];
        // copy in h0, c0
        float h_prev[HIDDEN_SIZE], c_prev[HIDDEN_SIZE];
        for(int i=0; i<HIDDEN_SIZE; i++){
            h_prev[i] = h0[s][i];
            c_prev[i] = c0[s][i];
        }

        // Unroll for L time steps
        for(int t=0; t<L; t++){
            // 1. Copy input to cache
            for(int i=0; i<INPUT_SIZE; i++){
                cache->x[s][t][i] = inputs[s][t*INPUT_SIZE + i];
            }

            // 2. Compute gates
            // We'll do i(t) = sigmoid( W_ix x(t) + W_ih h_prev + b_i ), etc.
            float i_in[HIDDEN_SIZE], f_in[HIDDEN_SIZE], o_in[HIDDEN_SIZE], c_in[HIDDEN_SIZE];
            // Zero them first
            for(int i=0; i<HIDDEN_SIZE; i++){
                i_in[i] = params->b_i[i];
                f_in[i] = params->b_f[i];
                o_in[i] = params->b_o[i];
                c_in[i] = params->b_c[i];
            }

            // input->gate: i_in, f_in, o_in, c_in
            // i_in += W_ix x(t)
            for(int i=0; i<HIDDEN_SIZE; i++){
                for(int j=0; j<INPUT_SIZE; j++){
                    float x_val = cache->x[s][t][j];
                    i_in[i] += params->W_ix[i][j]*x_val;
                    f_in[i] += params->W_fx[i][j]*x_val;
                    o_in[i] += params->W_ox[i][j]*x_val;
                    c_in[i] += params->W_cx[i][j]*x_val;
                }
            }
            // i_in += W_ih h_prev
            for(int i=0; i<HIDDEN_SIZE; i++){
                for(int j=0; j<HIDDEN_SIZE; j++){
                    float h_val = h_prev[j];
                    i_in[i] += params->W_ih[i][j]*h_val;
                    f_in[i] += params->W_fh[i][j]*h_val;
                    o_in[i] += params->W_oh[i][j]*h_val;
                    c_in[i] += params->W_ch[i][j]*h_val;
                }
            }
            // Activation
            for(int i=0; i<HIDDEN_SIZE; i++){
                cache->i[s][t][i] = sigmoid(i_in[i]);
                cache->f[s][t][i] = sigmoid(f_in[i]);
                cache->o[s][t][i] = sigmoid(o_in[i]);
                cache->c_hat[s][t][i] = tanh_approx(c_in[i]);
            }

            // 3. c(t) = f(t)*c_prev + i(t)*c_hat(t)
            for(int i=0; i<HIDDEN_SIZE; i++){
                cache->c[s][t][i] = cache->f[s][t][i]*c_prev[i]
                                    + cache->i[s][t][i]*cache->c_hat[s][t][i];
            }

            // 4. h(t) = o(t)*tanh(c(t))
            for(int i=0; i<HIDDEN_SIZE; i++){
                cache->h[s][t][i] = cache->o[s][t][i] * tanh_approx(cache->c[s][t][i]);
            }

            // 5. output layer: y = W_hy h(t) + b_y
            for(int out_i=0; out_i<OUTPUT_SIZE; out_i++){
                float sum= params->b_y[out_i];
                for(int j=0; j<HIDDEN_SIZE; j++){
                    sum += params->W_hy[out_i][j]*cache->h[s][t][j];
                }
                cache->y[s][t][out_i] = sum;
            }

            // 6. Update h_prev, c_prev
            for(int i=0; i<HIDDEN_SIZE; i++){
                h_prev[i] = cache->h[s][t][i];
                c_prev[i] = cache->c[s][t][i];
            }
        } // end for t
    } // end for s
}

/***********************************
 * Backprop (BPTT) + MSE Loss
 *
 * We'll shift the target by +1 for "next-day" prediction:
 *   => The prediction for time t is used to match target[t+1].
 *   => There's no target for the last time step in each sequence.
 ***********************************/
float lstm_backward(
    LSTMParams *params, LSTMCache *cache, LSTMGrads *grads,
    float **targets, // targets[s][t*OUTPUT_SIZE + ?]
    int *seq_len,    // for each sequence
    int batch_size
)
{
    float total_loss = 0.0f;
    // For storing d h(t-1), d c(t-1) across time
    static float dh_next[BATCH_SIZE][HIDDEN_SIZE];
    static float dc_next[BATCH_SIZE][HIDDEN_SIZE];
    memset(dh_next, 0, sizeof(dh_next));
    memset(dc_next, 0, sizeof(dc_next));

    // Work backwards
    for(int s=0; s<batch_size; s++){
        int L = seq_len[s];
        for(int t=L-1; t>=0; t--){
            // Because we predict "next-day" price:
            //   The target for time t is actually the data at t+1 => if (t == L-1), no target
            if(t == L-1){
                // no next-day target for the very last step
                continue;
            }
            // 1. MSE derivative wrt y(t)
            float dy[OUTPUT_SIZE];
            float y_pred = cache->y[s][t][0];
            float y_true = targets[s][(t+1)*OUTPUT_SIZE + 0];  // next-day target
            float diff = (y_pred - y_true);
            dy[0] = diff; // dL/dy
            total_loss += 0.5f * diff*diff;

            // 2. Backprop to output layer => W_hy, b_y, and dh(t)
            float dh[HIDDEN_SIZE];
            for(int i=0; i<HIDDEN_SIZE; i++){
                dh[i] = 0.0f;
            }
            // single output dimension => out_i=0
            for(int i=0; i<HIDDEN_SIZE; i++){
                grads->W_hy[0][i] += dy[0]*cache->h[s][t][i];
                dh[i] += dy[0]*params->W_hy[0][i];
            }
            grads->b_y[0] += dy[0];

            // add dh_next from future time step
            for(int i=0; i<HIDDEN_SIZE; i++){
                dh[i] += dh_next[s][i];
            }

            // 3. backprop through h(t) = o(t)*tanh(c(t))
            float *o_t = cache->o[s][t];
            float *c_t = cache->c[s][t];
            float *i_t = cache->i[s][t];
            float *f_t = cache->f[s][t];
            float *c_hat_t = cache->c_hat[s][t];

            float do_t[HIDDEN_SIZE], dc_t[HIDDEN_SIZE];
            for(int i=0; i<HIDDEN_SIZE; i++){
                float tanhc = tanh_approx(c_t[i]);
                do_t[i] = tanhc * dh[i]; // partial w.r.t. o_in
                // partial w.r.t. c(t) from dh
                dc_t[i] = o_t[i]*(1.0f - tanhc*tanhc)*dh[i];
            }
            // plus dc_next
            for(int i=0; i<HIDDEN_SIZE; i++){
                dc_t[i] += dc_next[s][i];
            }

            // 4. c(t) = f(t)*c(t-1) + i(t)*c_hat(t)
            // => partial wrt f(t) is c(t-1), partial wrt i(t) is c_hat(t), partial wrt c_hat(t) is i(t)
            // We need c(t-1). If t=0, c(t-1)=0. We'll handle that carefully:
            float dc_prev_val[HIDDEN_SIZE];
            for(int i=0; i<HIDDEN_SIZE; i++){
                float c_prev = (t==0) ? 0.0f : cache->c[s][t-1][i];
                dc_prev_val[i] = dc_t[i]*f_t[i]; // for next iteration
            }
            float di_t[HIDDEN_SIZE], df_t[HIDDEN_SIZE], do_in[HIDDEN_SIZE], dc_hat_in[HIDDEN_SIZE];
            for(int i=0; i<HIDDEN_SIZE; i++){
                di_t[i] = dc_t[i]*c_hat_t[i]; // derivative wrt i(t)
                df_t[i] = dc_t[i]*((t==0)?0.0f:cache->c[s][t-1][i]); // wrt f(t)
            }
            // c_hat(t) = tanh(c_in)
            for(int i=0; i<HIDDEN_SIZE; i++){
                dc_hat_in[i] = dc_t[i]*i_t[i]*(1.0f - c_hat_t[i]*c_hat_t[i]);
            }
            // do_in: do_t[i] => derivative wrt logistic pre-activation => multiply by dsigmoid(o_t[i])
            for(int i=0; i<HIDDEN_SIZE; i++){
                do_in[i] = do_t[i]*dsigmoid_from_val(o_t[i]);
            }
            // same for di_t, df_t => multiply by dsigmoid
            float di_in[HIDDEN_SIZE], df_in[HIDDEN_SIZE];
            for(int i=0; i<HIDDEN_SIZE; i++){
                di_in[i] = di_t[i]*dsigmoid_from_val(i_t[i]);
                df_in[i] = df_t[i]*dsigmoid_from_val(f_t[i]);
            }

            // 5. Now we accumulate parameter grads for (i_in, f_in, o_in, c_hat_in).
            // We'll unify c_hat_in under "dc_in" as if it's a separate gate with tanh activation.
            float dc_in[HIDDEN_SIZE];
            for(int i=0; i<HIDDEN_SIZE; i++){
                dc_in[i] = dc_hat_in[i]; // already includes tanh derivative
            }

            float dh_prev[HIDDEN_SIZE], dc_prev[HIDDEN_SIZE];
            for(int i=0; i<HIDDEN_SIZE; i++){
                dh_prev[i] = 0.f;
                dc_prev[i] = dc_prev_val[i];
            }

            // x(t) and h(t-1)
            float *x_t = cache->x[s][t];
            float *h_prev;
            if(t == 0) {
                // no real "h(t-1)", assume 0
                static float zero_h[HIDDEN_SIZE];
                h_prev = zero_h;
            } else {
                h_prev = cache->h[s][t-1];
            }

            // For each gate: i_in, f_in, o_in, c_in
            // i_in => W_ix, W_ih, b_i
            for(int i=0; i<HIDDEN_SIZE; i++){
                float dval = di_in[i];
                // W_ix
                for(int j=0; j<INPUT_SIZE; j++){
                    grads->W_ix[i][j] += dval*x_t[j];
                }
                // W_ih
                for(int j=0; j<HIDDEN_SIZE; j++){
                    grads->W_ih[i][j] += dval*h_prev[j];
                    // accumulate dh_prev
                    dh_prev[j] += dval*params->W_ih[i][j];
                }
                // b_i
                grads->b_i[i] += dval;
            }
            // f_in => W_fx, W_fh, b_f
            for(int i=0; i<HIDDEN_SIZE; i++){
                float dval = df_in[i];
                for(int j=0; j<INPUT_SIZE; j++){
                    grads->W_fx[i][j] += dval*x_t[j];
                }
                for(int j=0; j<HIDDEN_SIZE; j++){
                    grads->W_fh[i][j] += dval*h_prev[j];
                    dh_prev[j] += dval*params->W_fh[i][j];
                }
                grads->b_f[i] += dval;
            }
            // o_in => W_ox, W_oh, b_o
            for(int i=0; i<HIDDEN_SIZE; i++){
                float dval = do_in[i];
                for(int j=0; j<INPUT_SIZE; j++){
                    grads->W_ox[i][j] += dval*x_t[j];
                }
                for(int j=0; j<HIDDEN_SIZE; j++){
                    grads->W_oh[i][j] += dval*h_prev[j];
                    dh_prev[j] += dval*params->W_oh[i][j];
                }
                grads->b_o[i] += dval;
            }
            // c_in => W_cx, W_ch, b_c
            for(int i=0; i<HIDDEN_SIZE; i++){
                float dval = dc_in[i];
                for(int j=0; j<INPUT_SIZE; j++){
                    grads->W_cx[i][j] += dval*x_t[j];
                }
                for(int j=0; j<HIDDEN_SIZE; j++){
                    grads->W_ch[i][j] += dval*h_prev[j];
                    dh_prev[j] += dval*params->W_ch[i][j];
                }
                grads->b_c[i] += dval;
            }

            // 6. set dh_next, dc_next for next iteration
            for(int i=0; i<HIDDEN_SIZE; i++){
                dh_next[s][i] = dh_prev[i];
                dc_next[s][i] = dc_prev[i];
            }
        } // end for t
    } // end for s

    return total_loss;
}

/***********************************
 * Grad Update (simple SGD)
 ***********************************/
void update_params(LSTMParams *params, LSTMGrads *grads) {
    float *p = (float*)params;
    float *g = (float*)grads;
    int N = sizeof(LSTMParams)/sizeof(float);
    for(int i=0; i<N; i++){
        p[i] -= LR*g[i];
    }
}

/***********************************
 * Main: Demonstration
 ***********************************/
int main(){
    srand((unsigned)time(NULL));

    // We will have BATCH_SIZE sequences, each up to MAX_SEQ_LEN.
    // Let's define variable lengths for each.
    int seq_len[BATCH_SIZE];
    seq_len[0] = 10;   // shorter
    seq_len[1] = 15;   // medium
    seq_len[2] = 20;   // longer

    // We store the "inputs" and "targets" in dynamic arrays:
    // For each sequence s, we have seq_len[s]*INPUT_SIZE, and seq_len[s]*OUTPUT_SIZE (although the last step has no target).
    float *inputs[BATCH_SIZE];
    float *targets[BATCH_SIZE];

    for(int s=0; s<BATCH_SIZE; s++){
        inputs[s] = (float*)malloc(seq_len[s]*INPUT_SIZE*sizeof(float));
        targets[s] = (float*)malloc(seq_len[s]*OUTPUT_SIZE*sizeof(float));
    }

    // Synthetic data: 
    // - We'll create "true price" P(t) that random-walks. 
    // - We'll create random macro factors in [0..1] and let the "price" evolve.
    // - We'll store them in inputs[s], and the "price" in a separate array. Then the "price" at t will be an input. Next-day price is the target for time t.
    for(int s=0; s<BATCH_SIZE; s++){
        float cur_price = 10.0f + (rand()%100)/10.0f;  // random start in [10..20]
        for(int t=0; t<seq_len[s]; t++){
            // macro/micro/tech
            for(int i=0; i<INPUT_SIZE-1; i++){
                inputs[s][t*INPUT_SIZE + i] = (float)(rand()%100)/100.0f;
            }
            // let's store current price as the last input factor
            inputs[s][t*INPUT_SIZE + (INPUT_SIZE-1)] = cur_price;

            // next day price (target) we do a random walk step
            float noise = (float)(rand()%100 - 50)/100.0f; // in [-0.5..0.5]
            float next_price = cur_price + noise;  
            if(next_price < 0.0f) next_price = 0.0f; // clamp

            // The "target" for time t is next day's price
            targets[s][t*OUTPUT_SIZE + 0] = next_price;

            // Update cur_price for next iteration
            cur_price = next_price;
        }
    }

    // LSTM parameters & grads
    LSTMParams params;
    LSTMGrads grads;
    init_params(&params, 0.1f); // random init
    zero_grads(&grads);

    // LSTM cache
    static LSTMCache cache;

    // initial states (h0, c0) for each sequence
    float *h0[BATCH_SIZE], *c0[BATCH_SIZE];
    for(int s=0; s<BATCH_SIZE; s++){
        h0[s] = (float*)calloc(HIDDEN_SIZE, sizeof(float));
        c0[s] = (float*)calloc(HIDDEN_SIZE, sizeof(float));
    }

    // Training
    for(int e=0; e<EPOCHS; e++){
        zero_grads(&grads);

        // Forward
        lstm_forward(&params, &cache, inputs, seq_len, h0, c0, BATCH_SIZE);

        // Backward
        float loss = lstm_backward(&params, &cache, &grads, targets, seq_len, BATCH_SIZE);

        // Update
        update_params(&params, &grads);

        if((e+1)%20==0){
            printf("Epoch %3d, Loss=%.4f\n", e+1, loss);
        }
    }

    // Check final predictions
    printf("\nFinal Predictions vs. Targets (showing last sequence):\n");
    int s = BATCH_SIZE-1; // show the longest sequence
    for(int t=0; t<seq_len[s]; t++){
        float y_pred = cache.y[s][t][0];
        // target is next-day => if t==seq_len[s]-1, no real next day
        if(t < seq_len[s]-1){
            float y_true = targets[s][(t+1)*OUTPUT_SIZE + 0]; 
            printf(" t=%2d: pred=%.3f, next-day-true=%.3f\n", t, y_pred, y_true);
        } else {
            printf(" t=%2d: pred=%.3f, next-day-true=N/A (last step)\n", t, y_pred);
        }
    }

    // Cleanup
    for(int s=0; s<BATCH_SIZE; s++){
        free(inputs[s]);
        free(targets[s]);
        free(h0[s]);
        free(c0[s]);
    }
    return 0;
}

