/*******************************************************************************
 * price_predictor.c (Enhanced and Corrected)
 *
 * Description:
 *   A C program that implements an LSTM for predicting the next day's stock close price
 *   based on technical indicators calculated from historical stock data.
 *
 * Key Enhancements:
 *   1. Corrected usage of pointers.
 *   2. Increased model capacity by doubling hidden units.
 *   3. Adjusted learning rate and increased epochs.
 *   4. Enhanced debugging statements for better monitoring.
 *   5. Improved output formatting for readability.
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

#define MAX_SAMPLES       10000  /* Maximum number of data samples */
#define INPUT_SIZE        13     /* Technical indicators: 6 original + 7 indicators */
#define HIDDEN_SIZE       16     /* Increased hidden units from 8 to 16 */
#define OUTPUT_SIZE        1     /* Next-day close prediction */
#define EPOCHS           1000    /* Increased epochs from 300 to 1000 */
#define LEARNING_RATE   0.001f   /* Increased learning rate from 0.0001 to 0.001 */
#define VALIDATION_SIZE    30    /* Number of days for validation */
#define CLIP_VALUE         5.0f  /* Gradient clipping threshold */

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
    float x[MAX_SAMPLES][INPUT_SIZE];
    float i_gate[MAX_SAMPLES][HIDDEN_SIZE];
    float f_gate[MAX_SAMPLES][HIDDEN_SIZE];
    float o_gate[MAX_SAMPLES][HIDDEN_SIZE];
    float c_hat[MAX_SAMPLES][HIDDEN_SIZE];
    float c_state[MAX_SAMPLES][HIDDEN_SIZE];
    float h_state[MAX_SAMPLES][HIDDEN_SIZE];
    float y_pred[MAX_SAMPLES][OUTPUT_SIZE];
} LSTMCache;

typedef struct {
    char date[11];  /* "YYYY-MM-DD" */
    float open;
    float high;
    float low;
    float close;
    float volume;
} DailyBar;

typedef struct {
    float obv[MAX_SAMPLES];
    float ad[MAX_SAMPLES];
    float adx[MAX_SAMPLES];
    float aroonUp[MAX_SAMPLES];
    float aroonDown[MAX_SAMPLES];
    float macd[MAX_SAMPLES];
    float rsi[MAX_SAMPLES];
} TechnicalIndicators;

static inline float randf(float range) {
    float r = (float)rand() / (float)RAND_MAX;
    return (r * 2.0f - 1.0f) * range;
}

static inline float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

static inline float dsigmoid(float s) {
    return s * (1.0f - s); /* derivative if s = sigmoid(...) */
}

static inline float tanh_approx(float x) {
    return tanhf(x);
}

static inline float dtanh(float tval) {
    return 1.0f - tval * tval;
}

/* Initialize LSTM parameters with Xavier Initialization */
static void init_params_xavier(LSTMParams *p, float input_size, float hidden_size) {
    float limit_i = sqrtf(6.0f / (input_size + hidden_size));
    float limit_f = sqrtf(6.0f / (input_size + hidden_size));
    float limit_o = sqrtf(6.0f / (input_size + hidden_size));
    float limit_c = sqrtf(6.0f / (input_size + hidden_size));
    float limit_y = sqrtf(6.0f / (hidden_size + OUTPUT_SIZE));

    for(int i = 0; i < HIDDEN_SIZE; i++) {
        for(int j = 0; j < INPUT_SIZE; j++) {
            p->W_ix[i][j] = ((float)rand() / RAND_MAX) * 2 * limit_i - limit_i;
            p->W_fx[i][j] = ((float)rand() / RAND_MAX) * 2 * limit_f - limit_f;
            p->W_ox[i][j] = ((float)rand() / RAND_MAX) * 2 * limit_o - limit_o;
            p->W_cx[i][j] = ((float)rand() / RAND_MAX) * 2 * limit_c - limit_c;
        }
        for(int j = 0; j < HIDDEN_SIZE; j++) {
            p->W_ih[i][j] = ((float)rand() / RAND_MAX) * 2 * limit_i - limit_i;
            p->W_fh[i][j] = ((float)rand() / RAND_MAX) * 2 * limit_f - limit_f;
            p->W_oh[i][j] = ((float)rand() / RAND_MAX) * 2 * limit_o - limit_o;
            p->W_ch[i][j] = ((float)rand() / RAND_MAX) * 2 * limit_c - limit_c;
        }
        p->b_i[i] = 0.0f;
        p->b_f[i] = 1.0f; // Initialize forget gate bias to 1.0f to encourage remembering
        p->b_o[i] = 0.0f;
        p->b_c[i] = 0.0f;
    }

    /* Initialize output layer */
    for(int i = 0; i < OUTPUT_SIZE; i++) {
        for(int j = 0; j < HIDDEN_SIZE; j++) {
            p->W_hy[i][j] = ((float)rand() / RAND_MAX) * 2 * limit_y - limit_y;
        }
        p->b_y[i] = 0.0f;
    }
}

/* Zero out gradients */
static void zero_grads(LSTMGrads *g) {
    memset(g, 0, sizeof(LSTMGrads));
}

/* Apply gradient clipping */
static void clip_grads(LSTMGrads *g, float clip_value) {
    // Iterate through all gradients and clip them
    for(int i = 0; i < HIDDEN_SIZE; i++) {
        for(int j = 0; j < INPUT_SIZE; j++) {
            if(g->W_ix[i][j] > clip_value) g->W_ix[i][j] = clip_value;
            if(g->W_ix[i][j] < -clip_value) g->W_ix[i][j] = -clip_value;

            if(g->W_fx[i][j] > clip_value) g->W_fx[i][j] = clip_value;
            if(g->W_fx[i][j] < -clip_value) g->W_fx[i][j] = -clip_value;

            if(g->W_ox[i][j] > clip_value) g->W_ox[i][j] = clip_value;
            if(g->W_ox[i][j] < -clip_value) g->W_ox[i][j] = -clip_value;

            if(g->W_cx[i][j] > clip_value) g->W_cx[i][j] = clip_value;
            if(g->W_cx[i][j] < -clip_value) g->W_cx[i][j] = -clip_value;
        }

        for(int j = 0; j < HIDDEN_SIZE; j++) {
            if(g->W_ih[i][j] > clip_value) g->W_ih[i][j] = clip_value;
            if(g->W_ih[i][j] < -clip_value) g->W_ih[i][j] = -clip_value;

            if(g->W_fh[i][j] > clip_value) g->W_fh[i][j] = clip_value;
            if(g->W_fh[i][j] < -clip_value) g->W_fh[i][j] = -clip_value;

            if(g->W_oh[i][j] > clip_value) g->W_oh[i][j] = clip_value;
            if(g->W_oh[i][j] < -clip_value) g->W_oh[i][j] = -clip_value;

            if(g->W_ch[i][j] > clip_value) g->W_ch[i][j] = clip_value;
            if(g->W_ch[i][j] < -clip_value) g->W_ch[i][j] = -clip_value;
        }

        // Biases
        if(g->b_i[i] > clip_value) g->b_i[i] = clip_value;
        if(g->b_i[i] < -clip_value) g->b_i[i] = -clip_value;

        if(g->b_f[i] > clip_value) g->b_f[i] = clip_value;
        if(g->b_f[i] < -clip_value) g->b_f[i] = -clip_value;

        if(g->b_o[i] > clip_value) g->b_o[i] = clip_value;
        if(g->b_o[i] < -clip_value) g->b_o[i] = -clip_value;

        if(g->b_c[i] > clip_value) g->b_c[i] = clip_value;
        if(g->b_c[i] < -clip_value) g->b_c[i] = -clip_value;
    }

    // Output layer gradients
    for(int i = 0; i < OUTPUT_SIZE; i++) {
        for(int j = 0; j < HIDDEN_SIZE; j++) {
            if(g->W_hy[i][j] > clip_value) g->W_hy[i][j] = clip_value;
            if(g->W_hy[i][j] < -clip_value) g->W_hy[i][j] = -clip_value;
        }
        if(g->b_y[i] > clip_value) g->b_y[i] = clip_value;
        if(g->b_y[i] < -clip_value) g->b_y[i] = -clip_value;
    }
}

/* Update LSTM parameters using gradients */
static void update_params_custom(LSTMParams *params, LSTMGrads *grads, float learning_rate) {
    float *p = (float*)params;
    float *g = (float*)grads;
    int count = sizeof(LSTMParams) / sizeof(float);
    for(int i = 0; i < count; i++) {
        p[i] -= learning_rate * g[i];
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

/***************************************
 * LSTM Forward Pass
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

    for(int t = 0; t < seq_len; t++){
        /* Store x in cache */
        for(int i = 0; i < INPUT_SIZE; i++){
            cache->x[t][i] = inputs[t][i];
        }

        float i_in[HIDDEN_SIZE], f_in[HIDDEN_SIZE], o_in[HIDDEN_SIZE], c_in[HIDDEN_SIZE];
        for(int i = 0; i < HIDDEN_SIZE; i++){
            i_in[i] = params->b_i[i];
            f_in[i] = params->b_f[i];
            o_in[i] = params->b_o[i];
            c_in[i] = params->b_c[i];
        }

        /* Add input->gate and h_prev->gate */
        for(int i = 0; i < HIDDEN_SIZE; i++){
            for(int j = 0; j < INPUT_SIZE; j++){
                float val = inputs[t][j];
                i_in[i] += params->W_ix[i][j] * val;
                f_in[i] += params->W_fx[i][j] * val;
                o_in[i] += params->W_ox[i][j] * val;
                c_in[i] += params->W_cx[i][j] * val;
            }
            for(int j = 0; j < HIDDEN_SIZE; j++){
                float hval = h_prev[j];
                i_in[i] += params->W_ih[i][j] * hval;
                f_in[i] += params->W_fh[i][j] * hval;
                o_in[i] += params->W_oh[i][j] * hval;
                c_in[i] += params->W_ch[i][j] * hval;
            }
        }

        /* Activations */
        for(int i = 0; i < HIDDEN_SIZE; i++){
            cache->i_gate[t][i] = sigmoid(i_in[i]);
            cache->f_gate[t][i] = sigmoid(f_in[i]);
            cache->o_gate[t][i] = sigmoid(o_in[i]);
            cache->c_hat[t][i] = tanh_approx(c_in[i]);
        }

        /* c(t) and h(t) */
        for(int i = 0; i < HIDDEN_SIZE; i++){
            cache->c_state[t][i] = cache->f_gate[t][i] * c_prev[i] + cache->i_gate[t][i] * cache->c_hat[t][i];
            cache->h_state[t][i] = cache->o_gate[t][i] * tanh_approx(cache->c_state[t][i]);
        }

        /* Output y(t) */
        for(int i = 0; i < OUTPUT_SIZE; i++){
            float sum = params->b_y[i];
            for(int k = 0; k < HIDDEN_SIZE; k++){
                sum += params->W_hy[i][k] * cache->h_state[t][k];
            }
            cache->y_pred[t][i] = sum; // Linear activation
        }

        /* Update h_prev and c_prev */
        for(int i = 0; i < HIDDEN_SIZE; i++){
            h_prev[i] = cache->h_state[t][i];
            c_prev[i] = cache->c_state[t][i];
        }

        /* Debug: Print activations for first few time steps */
        if(t < 5){
            printf("Time Step %d:\n", t);
            printf("  i_gate[0] = %.3f, f_gate[0] = %.3f, o_gate[0] = %.3f, c_hat[0] = %.3f\n",
                cache->i_gate[t][0], cache->f_gate[t][0], cache->o_gate[t][0], cache->c_hat[t][0]);
            printf("  c_state[0] = %.3f, h_state[0] = %.3f\n",
                cache->c_state[t][0], cache->h_state[t][0]);
        }
    }
}

/***************************************
 * LSTM Backward Pass (Simplified)
 ***************************************/
static float lstm_backward(
    LSTMParams *params,
    LSTMCache *cache,
    float targets[][OUTPUT_SIZE],
    int seq_len,
    LSTMGrads *grads
){
    float total_loss = 0.0f;
    float dh_next[HIDDEN_SIZE];
    float dc_next[HIDDEN_SIZE];
    memset(dh_next, 0, sizeof(dh_next));
    memset(dc_next, 0, sizeof(dc_next));

    for(int t = seq_len - 1; t >= 0; t--){
        /* Skip the last time step since there's no next day target */
        if(t == seq_len - 1){
            continue;
        }

        float y_pred_norm = cache->y_pred[t][0];
        float y_true_norm = targets[t+1][0];
        float diff = y_pred_norm - y_true_norm;
        float loss = 0.5f * diff * diff;
        total_loss += loss;

        /* Output layer gradients */
        grads->W_hy[0][0] += diff * cache->h_state[t][0];
        grads->b_y[0] += diff;

        float dy = diff;
        float dh[HIDDEN_SIZE];
        for(int i = 0; i < HIDDEN_SIZE; i++){
            dh[i] = params->W_hy[0][i] * dy + dh_next[i];
        }

        /* Gradients for output gate */
        float do_[HIDDEN_SIZE];
        for(int i = 0; i < HIDDEN_SIZE; i++){
            do_[i] = tanh_approx(cache->c_state[t][i]) * dh[i];
        }

        /* Gradients for c_state */
        float dc[HIDDEN_SIZE];
        for(int i = 0; i < HIDDEN_SIZE; i++){
            float tanhc = tanh_approx(cache->c_state[t][i]);
            dc[i] = params->W_hy[0][i] * dy * cache->o_gate[t][i] * dtanh(tanhc);
            dc[i] += dh[i] * cache->o_gate[t][i] * dtanh(tanhc);
            dc[i] += dc_next[i];
        }

        /* Gradients for gates */
        float di[HIDDEN_SIZE], df[HIDDEN_SIZE], dc_hat[HIDDEN_SIZE];
        for(int i = 0; i < HIDDEN_SIZE; i++){
            di[i] = dc[i] * cache->c_hat[t][i] * dsigmoid(cache->i_gate[t][i]);
            df[i] = dc[i] * cache->f_gate[t][i] * dsigmoid(cache->f_gate[t][i]);
            dc_hat[i] = dc[i] * cache->i_gate[t][i] * dtanh(cache->c_hat[t][i]);
        }

        /* Accumulate gradients for input, forget, output, and candidate gates */
        for(int i = 0; i < HIDDEN_SIZE; i++){
            for(int j = 0; j < INPUT_SIZE; j++){
                grads->W_ix[i][j] += di[i] * cache->x[t][j];
                grads->W_fx[i][j] += df[i] * cache->x[t][j];
                grads->W_ox[i][j] += do_[i] * cache->x[t][j];
                grads->W_cx[i][j] += dc_hat[i] * cache->x[t][j];
            }
            for(int j = 0; j < HIDDEN_SIZE; j++){
                grads->W_ih[i][j] += di[i] * ((t > 0) ? cache->h_state[t-1][j] : 0.0f);
                grads->W_fh[i][j] += df[i] * ((t > 0) ? cache->h_state[t-1][j] : 0.0f);
                grads->W_oh[i][j] += do_[i] * ((t > 0) ? cache->h_state[t-1][j] : 0.0f);
                grads->W_ch[i][j] += dc_hat[i] * ((t > 0) ? cache->h_state[t-1][j] : 0.0f);
            }
            grads->b_i[i] += di[i];
            grads->b_f[i] += df[i];
            grads->b_o[i] += do_[i];
            grads->b_c[i] += dc_hat[i];
        }

        /* Update dh_next and dc_next for the previous time step */
        for(int i = 0; i < HIDDEN_SIZE; i++){
            dh_next[i] = 0.0f;
            dc_next[i] = 0.0f;
            for(int j = 0; j < HIDDEN_SIZE; j++){
                dh_next[i] += params->W_ih[i][j] * di[j];
                dh_next[i] += params->W_fh[i][j] * df[j];
                dh_next[i] += params->W_oh[i][j] * do_[j];
                dh_next[i] += params->W_ch[i][j] * dc_hat[j];
            }
        }

        /* Debug: Print gradients for first few time steps */
        if(t < 5){
            printf("Backward Time Step %d:\n", t);
            printf("  Gradient di[0] = %.3f, df[0] = %.3f, dc_hat[0] = %.3f\n",
                di[0], df[0], dc_hat[0]);
            printf("  Gradient do_[0] = %.3f\n", do_[0]);
        }
    }

    return total_loss;
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
    float min_target = 1e9f, max_target = -1e9f;

    /* Find min and max for inputs */
    for(int j = 0; j < INPUT_SIZE; j++){
        minVal[j] = 1e9f;
        maxVal[j] = -1e9f;
    }
    for(int i = 0; i < validCount; i++){
        for(int j = 0; j < INPUT_SIZE; j++){
            float v = inputs[i][j];
            if(v < minVal[j]) minVal[j] = v;
            if(v > maxVal[j]) maxVal[j] = v;
        }
    }

    /* Find min and max for targets */
    for(int i = 0; i < validCount; i++){
        float target = targets_raw[i][0];
        if(target < min_target) min_target = target;
        if(target > max_target) max_target = target;
    }

    /* Debug: Print min and max targets */
    printf("\nTarget Min: %.2f, Target Max: %.2f\n", min_target, max_target);

    /* Normalize inputs */
    for(int i = 0; i < validCount; i++){
        for(int j = 0; j < INPUT_SIZE; j++){
            float denom = (maxVal[j] - minVal[j]);
            if(denom < 1e-6f) denom = 1.0f; /* Prevent division by zero */
            inputs[i][j] = (inputs[i][j] - minVal[j]) / denom;
            if(isnan(inputs[i][j]) || isinf(inputs[i][j])){
                fprintf(stderr, "Normalization resulted in invalid value at sample %d, feature %d.\n", i, j);
                return 1;
            }
        }
    }

    /* Normalize targets */
    float targets_norm[MAX_SAMPLES][OUTPUT_SIZE];
    for(int i = 0; i < validCount; i++){
        targets_norm[i][0] = (targets_raw[i][0] - min_target) / (max_target - min_target);
        if(isnan(targets_norm[i][0]) || isinf(targets_norm[i][0])){
            fprintf(stderr, "Normalization resulted in invalid target at sample %d.\n", i);
            return 1;
        }
    }

    /* Debug: Print first 5 normalized targets */
    printf("\nNormalized Targets (First 5 Samples):\n");
    for(int i = 0; i < 5 && i < validCount; i++) {
        printf("Sample %d: %.3f\n", i, targets_norm[i][0]);
    }

    /* 5) Initialize LSTM */
    LSTMParams params;
    init_params_xavier(&params, INPUT_SIZE, HIDDEN_SIZE);
    LSTMGrads grads;
    zero_grads(&grads);
    LSTMCache cache;

    /* 6) Split data into training and validation */
    int trainLen = validCount - VALIDATION_SIZE;
    if(trainLen < 2){
        trainLen = validCount; /* Fallback */
    }

    /* 7) Training Loop */
    for(int epoch = 1; epoch <= EPOCHS; epoch++){
        zero_grads(&grads);
        lstm_forward(&params, inputs, trainLen, &cache);
        float loss = lstm_backward(&params, &cache, targets_norm, trainLen, &grads);

        /* Gradient Clipping */
        clip_grads(&grads, CLIP_VALUE);

        /* Update Parameters */
        update_params_custom(&params, &grads, LEARNING_RATE);

        /* Check for NaN in loss */
        if(isnan(loss)){
            fprintf(stderr, "Encountered NaN in loss at epoch %d, aborting.\n", epoch);
            return 1;
        }

        /* Print training progress */
        if(epoch % 100 == 0 || epoch == 1){
            /* Calculate weight norm for monitoring */
            float weight_norm = 0.0f;
            for(int i = 0; i < HIDDEN_SIZE; i++) {
                for(int j = 0; j < INPUT_SIZE; j++) {
                    weight_norm += params.W_ix[i][j] * params.W_ix[i][j];
                    weight_norm += params.W_fx[i][j] * params.W_fx[i][j];
                    weight_norm += params.W_ox[i][j] * params.W_ox[i][j];
                    weight_norm += params.W_cx[i][j] * params.W_cx[i][j];
                }
                for(int j = 0; j < HIDDEN_SIZE; j++) {
                    weight_norm += params.W_ih[i][j] * params.W_ih[i][j];
                    weight_norm += params.W_fh[i][j] * params.W_fh[i][j];
                    weight_norm += params.W_oh[i][j] * params.W_oh[i][j];
                    weight_norm += params.W_ch[i][j] * params.W_ch[i][j];
                }
                weight_norm += params.b_i[i] * params.b_i[i];
                weight_norm += params.b_f[i] * params.b_f[i];
                weight_norm += params.b_o[i] * params.b_o[i];
                weight_norm += params.b_c[i] * params.b_c[i];
            }
            /* Output layer weights */
            for(int i = 0; i < OUTPUT_SIZE; i++) {
                for(int j = 0; j < HIDDEN_SIZE; j++) {
                    weight_norm += params.W_hy[i][j] * params.W_hy[i][j];
                }
                weight_norm += params.b_y[i] * params.b_y[i];
            }
            weight_norm = sqrtf(weight_norm);
            printf("Epoch %4d, Train Loss=%.6f, Weight Norm=%.6f\n", epoch, loss / trainLen, weight_norm);

            /* Print a few sample predictions */
            printf("Sample Predictions at Epoch %d:\n", epoch);
            printf("-------------------------------------------------------------\n");
            printf("| Day |     Date     | Predicted Close | Actual Close | Error |\n");
            printf("-------------------------------------------------------------\n");
            for(int t = trainLen - 5; t < trainLen && t < validCount -1; t++){
                float y_pred_norm = cache.y_pred[t][0];
                float y_pred_raw = y_pred_norm * (max_target - min_target) + min_target;
                float y_true = targets_raw[t+1][0];
                float error = fabsf(y_pred_raw - y_true);
                printf("| %3d | %10s |     %8.2f     |    %8.2f   | %5.2f |\n",
                    t, dailyData[t].date, y_pred_raw, y_true, error);
            }
            printf("-------------------------------------------------------------\n");
        }
    }

    /* 8) Validation */
    lstm_forward(&params, inputs, validCount, &cache);
    printf("\nValidation (Last %d Days):\n", VALIDATION_SIZE);
    printf("-------------------------------------------------------------\n");
    printf("| Day |     Date     | Predicted Close | Actual Close | Error |\n");
    printf("-------------------------------------------------------------\n");
    int valStart = trainLen;
    for(int t = valStart; t < validCount - 1; t++){
        float y_pred_norm = cache.y_pred[t][0];
        /* Denormalize prediction */
        float y_pred_raw = y_pred_norm * (max_target - min_target) + min_target;
        float y_true = targets_raw[t+1][0];
        float error = fabsf(y_pred_raw - y_true);
        printf("| %3d | %10s |     %8.2f     |    %8.2f   | %5.2f |\n",
            t, dailyData[t].date, y_pred_raw, y_true, error);
    }
    printf("-------------------------------------------------------------\n");

    /* 9) Final Summary */
    // Calculate overall validation metrics
    float total_mae = 0.0f;
    float total_rmse = 0.0f;
    for(int t = valStart; t < validCount - 1; t++){
        float y_pred_norm = cache.y_pred[t][0];
        /* Denormalize prediction */
        float y_pred_raw = y_pred_norm * (max_target - min_target) + min_target;
        float y_true = targets_raw[t+1][0];
        float error = y_pred_raw - y_true;
        total_mae += fabsf(error);
        total_rmse += error * error;
    }
    int val_samples = validCount - 1 - valStart;
    float mae = total_mae / val_samples;
    float rmse = sqrtf(total_rmse / val_samples);
    printf("\nValidation Metrics:\n");
    printf("Mean Absolute Error (MAE): %.2f\n", mae);
    printf("Root Mean Squared Error (RMSE): %.2f\n", rmse);

    /* 10) Pretty Output for All Data */
    printf("\nDetailed Predictions for All Data:\n");
    printf("--------------------------------------------------------------------\n");
    printf("| Day |     Date     | Predicted Close | Actual Close | Error |\n");
    printf("--------------------------------------------------------------------\n");
    for(int t = 0; t < validCount -1; t++){
        float y_pred_norm = cache.y_pred[t][0];
        /* Denormalize prediction */
        float y_pred_raw = y_pred_norm * (max_target - min_target) + min_target;
        float y_true = targets_raw[t+1][0];
        float error = fabsf(y_pred_raw - y_true);
        printf("| %3d | %10s |     %8.2f     |    %8.2f   | %5.2f |\n",
            t, dailyData[t].date, y_pred_raw, y_true, error);
    }
    printf("--------------------------------------------------------------------\n");

    return 0;
}

