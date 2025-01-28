/*******************************************************************************
 * gitlab_lstm_nextday.c (Revised)
 *
 * Key changes from previous version:
 *   1. Improved CSV parsing & daily aggregation:
 *      - Skips malformed rows (missing columns or can't parse float).
 *      - Skips final daily bar if it has invalid data.
 *   2. Basic data validation:
 *      - If a daily bar has open<=0, high<=0, low<=0, close<=0, or volume<0,
 *        we discard that day from the final dataset.
 *   3. Prints a summary of final daily bars used.
 *   4. (Optional) Simple min/max normalization for daily features to help
 *      prevent exploding values.
 *
 * Compile:
 *   cc gitlab_lstm_nextday.c -o gitlab_lstm_nextday -lm
 * Run:
 *   ./gitlab_lstm_nextday path/to/gitlab_5min.csv
 *******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

/***************************************
 * Hyperparams & Dimensions
 ***************************************/
#define MAX_DAYS       600  /* Safety upper limit, can handle ~2 years if needed */
#define INPUT_SIZE     6    /* daily (Open,High,Low,Close,Volume, e.g., Range) */
#define HIDDEN_SIZE    8
#define OUTPUT_SIZE    1    /* next-day close prediction */
#define EPOCHS         300
#define LEARNING_RATE  0.01f

/***************************************
 * Data Structures
 ***************************************/

/* LSTM parameters */
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
} LSTMParams;

/* Parameter gradients, same shape */
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

/* Forward-pass caches for each day t */
typedef struct {
    float x[MAX_DAYS][INPUT_SIZE];
    float i[MAX_DAYS][HIDDEN_SIZE];
    float f[MAX_DAYS][HIDDEN_SIZE];
    float o[MAX_DAYS][HIDDEN_SIZE];
    float c_hat[MAX_DAYS][HIDDEN_SIZE];
    float c[MAX_DAYS][HIDDEN_SIZE];
    float h[MAX_DAYS][HIDDEN_SIZE];
    float y[MAX_DAYS][OUTPUT_SIZE];
} LSTMCache;

/***************************************
 * Utility Functions
 ***************************************/
static inline float randf(float range) {
    float r = (float)rand() / (float)RAND_MAX;
    return (r*2.0f - 1.0f)*range;
}

static inline float sigmoid(float x) {
    return 1.0f/(1.0f + expf(-x));
}
static inline float dsigmoid_from_val(float s) {
    return s*(1.0f - s); /* derivative if s = sigmoid(...) */
}

static inline float tanh_approx(float x) {
    return tanhf(x);
}

static inline float dtanh_from_val(float tval) {
    return 1.0f - tval*tval; 
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
        p[i] -= LEARNING_RATE*r[i];
    }
}

/***************************************
 * Aggregated daily bars from intraday
 ***************************************/
typedef struct {
    char date[11];  /* "YYYY-MM-DD" */
    float open;
    float high;
    float low;
    float close;
    float volume;
} DailyBar;

/* Load intraday 5-min bars, group them by date, produce daily bars.
   Skips malformed lines or days with obviously invalid data. */
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
            /* skip header if present */
            firstRow = 0;
            if(strstr(line, "Timestamp")){
                continue; 
            }
        }

        char *ts   = strtok(line, ",");
        char *oStr = strtok(NULL, ",");
        char *hStr = strtok(NULL, ",");
        char *lStr = strtok(NULL, ",");
        char *cStr = strtok(NULL, ",");
        char *vStr = strtok(NULL, ",");
        if(!ts || !oStr || !hStr || !lStr || !cStr || !vStr) {
            /* skip malformed line */
            continue;
        }

        /* extract date (YYYY-MM-DD) from Timestamp */
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
            /* finalize the previous day */
            if(haveCurrent){
                /* check if curDay is valid: open>0, high>=open, ... etc. 
                   You can customize. We'll do a minimal sanity check. */
                if(curDay.open > 0 && curDay.high > 0 && curDay.low > 0 
                   && curDay.close > 0 && curDay.volume >= 0 
                   && dayCount < max_days)
                {
                    daily[dayCount++] = curDay;
                }
            }
            /* start new day */
            strncpy(currentDate, dateBuf, 11);
            curDay.open   = o_val;
            curDay.high   = h_val;
            curDay.low    = l_val;
            curDay.close  = c_val;
            curDay.volume = v_val;
            strncpy(curDay.date, dateBuf, 11);
            haveCurrent = 1;
        } else {
            /* same day => update H,L,C,Vol */
            if(h_val > curDay.high) curDay.high = h_val;
            if(l_val < curDay.low)  curDay.low  = l_val;
            curDay.close  = c_val;
            curDay.volume += v_val; 
        }
    }
    /* finalize last day if valid */
    if(haveCurrent && dayCount < max_days) {
        if(curDay.open > 0 && curDay.high > 0 && curDay.low > 0 
           && curDay.close > 0 && curDay.volume >= 0)
        {
            daily[dayCount++] = curDay;
        }
    }

    fclose(fp);
    return dayCount;
}

/***************************************
 * LSTM Forward
 ***************************************/
static void lstm_forward(
    LSTMParams *params,
    float inputs[][INPUT_SIZE],
    int seq_len,
    LSTMCache *cache
){
    float h_prev[HIDDEN_SIZE];
    float c_prev[HIDDEN_SIZE];
    memset(h_prev, 0, sizeof(h_prev));
    memset(c_prev, 0, sizeof(c_prev));

    for(int t=0; t<seq_len; t++){
        /* store x in cache */
        for(int i=0; i<INPUT_SIZE; i++){
            cache->x[t][i] = inputs[t][i];
        }
        float i_in[HIDDEN_SIZE], f_in[HIDDEN_SIZE], o_in[HIDDEN_SIZE], c_in[HIDDEN_SIZE];
        for(int i=0; i<HIDDEN_SIZE; i++){
            i_in[i] = params->b_i[i];
            f_in[i] = params->b_f[i];
            o_in[i] = params->b_o[i];
            c_in[i] = params->b_c[i];
        }
        /* Add input->gate and h_prev->gate */
        for(int i=0; i<HIDDEN_SIZE; i++){
            for(int j=0; j<INPUT_SIZE; j++){
                float val = inputs[t][j];
                i_in[i] += params->W_ix[i][j]*val;
                f_in[i] += params->W_fx[i][j]*val;
                o_in[i] += params->W_ox[i][j]*val;
                c_in[i] += params->W_cx[i][j]*val;
            }
            for(int j=0; j<HIDDEN_SIZE; j++){
                float hval = h_prev[j];
                i_in[i] += params->W_ih[i][j]*hval;
                f_in[i] += params->W_fh[i][j]*hval;
                o_in[i] += params->W_oh[i][j]*hval;
                c_in[i] += params->W_ch[i][j]*hval;
            }
        }
        /* Activations */
        for(int i=0; i<HIDDEN_SIZE; i++){
            cache->i[t][i] = sigmoid(i_in[i]);
            cache->f[t][i] = sigmoid(f_in[i]);
            cache->o[t][i] = sigmoid(o_in[i]);
            cache->c_hat[t][i] = tanh_approx(c_in[i]);
        }
        /* c(t), h(t) */
        for(int i=0; i<HIDDEN_SIZE; i++){
            cache->c[t][i] = cache->f[t][i]*c_prev[i] + cache->i[t][i]*cache->c_hat[t][i];
        }
        for(int i=0; i<HIDDEN_SIZE; i++){
            cache->h[t][i] = cache->o[t][i]*tanh_approx(cache->c[t][i]);
        }
        /* output y(t) */
        {
            float sum = params->b_y[0];
            for(int k=0; k<HIDDEN_SIZE; k++){
                sum += params->W_hy[0][k]*cache->h[t][k];
            }
            cache->y[t][0] = sum;
        }
        /* update h_prev, c_prev */
        for(int i=0; i<HIDDEN_SIZE; i++){
            h_prev[i] = cache->h[t][i];
            c_prev[i]  = cache->c[t][i];
        }
    }
}

/***************************************
 * BPTT (MSE on next-day close)
 ***************************************/
static float lstm_backward(
    LSTMParams *params,
    LSTMCache *cache,
    float targets[][OUTPUT_SIZE],
    int seq_len,
    LSTMGrads *grads
){
    float total_loss = 0.f;
    float dh_next[HIDDEN_SIZE];
    float dc_next[HIDDEN_SIZE];
    memset(dh_next, 0, sizeof(dh_next));
    memset(dc_next, 0, sizeof(dc_next));

    for(int t=seq_len-1; t>=0; t--){
        if(t == seq_len-1){
            /* no next day for the final day -> skip */
            continue;
        }
        float pred  = cache->y[t][0];
        float truth = targets[t+1][0]; 
        float diff  = (pred - truth);
        float dy    = diff;
        total_loss += 0.5f*diff*diff;

        /* Output layer grads => W_hy, b_y, dh(t) */
        float dh[HIDDEN_SIZE];
        memset(dh, 0, sizeof(dh));
        for(int i=0; i<HIDDEN_SIZE; i++){
            grads->W_hy[0][i] += dy*cache->h[t][i];
            dh[i] = dy*params->W_hy[0][i];
        }
        grads->b_y[0] += dy;

        /* add dh_next from future time step */
        for(int i=0; i<HIDDEN_SIZE; i++){
            dh[i] += dh_next[i];
        }

        /* h(t) = o(t)*tanh(c(t)) => do, dc */
        float do_[HIDDEN_SIZE], dc[HIDDEN_SIZE];
        for(int i=0; i<HIDDEN_SIZE; i++){
            float o_val = cache->o[t][i];
            float c_val = cache->c[t][i];
            float tanhc = tanh_approx(c_val);
            do_[i] = tanhc*dh[i];
            dc[i]  = o_val*(1.f - tanhc*tanhc)*dh[i];
        }
        /* plus dc_next */
        for(int i=0; i<HIDDEN_SIZE; i++){
            dc[i] += dc_next[i];
        }

        /* c(t)=f(t)*c(t-1) + i(t)*c_hat(t) => di, df, dc_hat */
        float di[HIDDEN_SIZE], df[HIDDEN_SIZE], dc_hat[HIDDEN_SIZE];
        for(int i=0; i<HIDDEN_SIZE; i++){
            float c_prev = (t==0)? 0.f : cache->c[t-1][i];
            di[i]      = dc[i]*cache->c_hat[t][i];
            df[i]      = dc[i]*c_prev;
            dc_hat[i]  = dc[i]*cache->i[t][i];
        }

        /* gate pre-activation grads */
        float do_in[HIDDEN_SIZE], di_in[HIDDEN_SIZE], df_in[HIDDEN_SIZE], dc_in[HIDDEN_SIZE];
        for(int i=0; i<HIDDEN_SIZE; i++){
            do_in[i] = do_[i]*dsigmoid_from_val(cache->o[t][i]);
            di_in[i] = di[i]*dsigmoid_from_val(cache->i[t][i]);
            df_in[i] = df[i]*dsigmoid_from_val(cache->f[t][i]);
            float ch  = cache->c_hat[t][i];
            dc_in[i]  = dc_hat[i]*(1.f - ch*ch);
        }

        float dh_prev[HIDDEN_SIZE];
        memset(dh_prev, 0, sizeof(dh_prev));

        #define ACCUM_GATE(Wx,Wh,bias,d_in)                                    
            do {                                                               
                for(int i=0; i<HIDDEN_SIZE; i++){                              
                    float dval = d_in[i];                                      
                    /* dW_x  => x(t) */                                        
                    for(int j=0; j<INPUT_SIZE; j++){                           
                        grads->Wx[i][j] += dval*cache->x[t][j];                
                    }                                                          
                    /* dW_h => h(t-1) if t>0 */                                
                    if(t>0) {                                                 
                        for(int j=0; j<HIDDEN_SIZE; j++){                      
                            grads->Wh[i][j] += dval*cache->h[t-1][j];          
                            dh_prev[j]       += dval*params->Wh[i][j];        
                        }                                                      
                    }                                                          
                    grads->bias[i] += dval;                                    
                }                                                              
            } while(0)

        ACCUM_GATE(W_ix, W_ih, b_i, di_in);
        ACCUM_GATE(W_fx, W_fh, b_f, df_in);
        ACCUM_GATE(W_ox, W_oh, b_o, do_in);
        ACCUM_GATE(W_cx, W_ch, b_c, dc_in);

        float dc_prev[HIDDEN_SIZE];
        for(int i=0; i<HIDDEN_SIZE; i++){
            dc_prev[i] = dc[i]*cache->f[t][i];
        }

        memcpy(dh_next, dh_prev, sizeof(dh_prev));
        memcpy(dc_next, dc_prev, sizeof(dc_prev));
    }

    return total_loss;
}

/***************************************
 * Main
 ***************************************/
int main(int argc, char *argv[]){
    if(argc < 2){
        fprintf(stderr, "Usage: %s path/to/gitlab_5min.csv\n", argv[0]);
        return 1;
    }
    srand((unsigned)time(NULL));

    /* 1) Load intraday CSV, aggregate daily bars */
    DailyBar dailyData[MAX_DAYS];
    int rawCount = load_and_aggregate_daily(argv[1], dailyData, MAX_DAYS);
    if(rawCount <= 1){
        fprintf(stderr,"No valid daily bars found in CSV.\n");
        return 1;
    }

    /* 2) Build final array of valid days only 
          - Possibly do more checks or normalizations
    */
    float inputs[MAX_DAYS][INPUT_SIZE];
    float targets[MAX_DAYS][OUTPUT_SIZE];
    int dayCount = 0;
    for(int i=0; i<rawCount; i++){
        DailyBar *d = &dailyData[i];
        /* Basic check again (some days might slip if partial): */
        if(d->open <= 0.f || d->high <= 0.f || d->low <=0.f || d->close<=0.f || d->volume<0.f){
            continue;
        }
        inputs[dayCount][0] = d->open;
        inputs[dayCount][1] = d->high;
        inputs[dayCount][2] = d->low;
        inputs[dayCount][3] = d->close;
        inputs[dayCount][4] = d->volume;
        inputs[dayCount][5] = d->high - d->low; /* example range */

        /* target = today's close, for reference. Next-day logic used in backprop. */
        targets[dayCount][0] = d->close;
        dayCount++;
    }

    if(dayCount<=1){
        fprintf(stderr,"After validation, we have only %d daily bars.\n",dayCount);
        return 1;
    }

    printf("Final daily bars used: %d\n", dayCount);
    printf("First day: %s O=%.2f H=%.2f L=%.2f C=%.2f V=%.0f\n",
        dailyData[0].date, dailyData[0].open, dailyData[0].high, dailyData[0].low,
        dailyData[0].close, dailyData[0].volume);
    printf("Last day:  %s O=%.2f H=%.2f L=%.2f C=%.2f V=%.0f\n",
        dailyData[dayCount-1].date, dailyData[dayCount-1].open,
        dailyData[dayCount-1].high, dailyData[dayCount-1].low,
        dailyData[dayCount-1].close, dailyData[dayCount-1].volume);

    /* Optional: Basic min-max or standard normalization to avoid large volumes, etc. */
    
    float minVal[INPUT_SIZE], maxVal[INPUT_SIZE];
    for(int j=0; j<INPUT_SIZE; j++){
        minVal[j] = 1e9f; maxVal[j] = -1e9f;
    }
    for(int i=0; i<dayCount; i++){
        for(int j=0; j<INPUT_SIZE; j++){
            float v = inputs[i][j];
            if(v<minVal[j]) minVal[j]=v;
            if(v>maxVal[j]) maxVal[j]=v;
        }
    }
    for(int i=0; i<dayCount; i++){
        for(int j=0; j<INPUT_SIZE; j++){
            float denom = (maxVal[j]-minVal[j]);
            if(denom<1e-9f) denom=1.f;
            inputs[i][j] = (inputs[i][j] - minVal[j]) / denom;
        }
    }
    

    /* 3) LSTM Setup */
    LSTMParams params;
    init_params(&params, 0.1f);
    LSTMGrads grads;
    zero_grads(&grads);
    static LSTMCache cache;

    /* 4) Train/Validation split: e.g. last 30 days as validation */
    int trainLen = dayCount - 30;
    if(trainLen < 2) {
        trainLen = dayCount; /* fallback if not enough days */
    }

    /* 5) Training */
    for(int e=1; e<=EPOCHS; e++){
        zero_grads(&grads);
        lstm_forward(&params, inputs, trainLen, &cache);
        float loss = lstm_backward(&params, &cache, targets, trainLen, &grads);
        update_params(&params, &grads);

        if(e%50==0){
            printf("Epoch %3d, TrainLoss=%.4f\n", e, loss);
            if(isnan(loss)) {
                fprintf(stderr, "Encountered NaN at epoch %d, aborting.\n", e);
                break;
            }
        }
    }

    /* 6) Evaluate one-day-behind on entire dataset */
    lstm_forward(&params, inputs, dayCount, &cache);
    printf("\nValidation (last 30 days):\n");
    int valStart = trainLen-1; 
    if(valStart<0) valStart=0;
    for(int t=valStart; t<dayCount-1; t++){
        float y_pred = cache.y[t][0];
        float y_true = targets[t+1][0];
        printf(" Day %3d => Predict=%.3f, Actual=%.3f [Day %s -> next day %s]\n",
            t, y_pred, y_true,
            dailyData[t].date, dailyData[t+1].date);
    }

    return 0;
}

