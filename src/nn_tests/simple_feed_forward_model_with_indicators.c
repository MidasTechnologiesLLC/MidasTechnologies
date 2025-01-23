/*****************************************************************************
 * simple_feed_forward_model_with_indicators.c
 *
 * A feed-forward neural network in C that:
 *   - Reads CSV with OHLCV data
 *   - Computes 7 indicators internally
 *   - Uses [Open,High,Low,Volume] + 7 indicators = 11 features
 *   - Trains to predict Close
 *   - Prints extended info: predictions, errors, feature importances
 *****************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

// ---------------------------------------------------------------------------
// CONFIG & GLOBALS
// ---------------------------------------------------------------------------
#define MAX_SAMPLES   100000

// 4 base + 7 technical = 11 inputs
#define INPUT_SIZE    11
#define OUTPUT_SIZE   1

static double openArr[MAX_SAMPLES];
static double highArr[MAX_SAMPLES];
static double lowArr[MAX_SAMPLES];
static double closeArr[MAX_SAMPLES];
static double volumeArr[MAX_SAMPLES];

static double X_raw[MAX_SAMPLES][INPUT_SIZE];
static double y_raw[MAX_SAMPLES];

static int    NUM_SAMPLES = 0;  // actual number of rows loaded

// 7 indicator arrays
static double obvArr[MAX_SAMPLES];
static double adArr[MAX_SAMPLES];
static double adxArr[MAX_SAMPLES];
static double aroonUpArr[MAX_SAMPLES];
static double aroonDownArr[MAX_SAMPLES];
static double macdArr[MAX_SAMPLES];
static double rsiArr[MAX_SAMPLES];
static double stochArr[MAX_SAMPLES]; // if you want to use it

// For normalization
static double X_mean[INPUT_SIZE];
static double X_std[INPUT_SIZE];
static double y_mean;
static double y_std;

// Architecture
#define NUM_HIDDEN_LAYERS 2
static const int HIDDEN_SIZES[NUM_HIDDEN_LAYERS] = {8, 6};

static int layer_sizes[NUM_HIDDEN_LAYERS + 2];

// Weights & Biases
static double **W[NUM_HIDDEN_LAYERS + 1];
static double  *b[NUM_HIDDEN_LAYERS + 1];

// Adam buffers
static double **mW[NUM_HIDDEN_LAYERS + 1], **vW[NUM_HIDDEN_LAYERS + 1];
static double  *mb[NUM_HIDDEN_LAYERS + 1],  *vb[NUM_HIDDEN_LAYERS + 1];

// Adam hyperparams
#define ADAM_LEARNING_RATE 0.01
#define BETA1 0.9
#define BETA2 0.999
#define EPSILON 1e-8

// L2 regularization
#define L2_LAMBDA 0.001

// Training
#define MAX_EPOCHS  1000
#define BATCH_SIZE  32
#define PRINT_INTERVAL 100

// ---------------------------------------------------------------------------
// UTILS
// ---------------------------------------------------------------------------
static inline double relu(double x) {
    return x > 0.0 ? x : 0.0;
}
static inline double relu_derivative(double x) {
    return x > 0.0 ? 1.0 : 0.0;
}

// 2D allocation
static double **alloc_2d(int rows, int cols) {
    double **ptr = (double **)malloc(rows * sizeof(double *));
    if(!ptr) {
        fprintf(stderr, "Memory alloc error\n");
        exit(1);
    }
    for(int i = 0; i < rows; i++) {
        ptr[i] = (double *)calloc(cols, sizeof(double));
        if(!ptr[i]) {
            fprintf(stderr, "Memory alloc error\n");
            exit(1);
        }
    }
    return ptr;
}
static void free_2d(double **arr, int rows) {
    if(!arr) return;
    for(int i = 0; i < rows; i++){
        free(arr[i]);
    }
    free(arr);
}

// ---------------------------------------------------------------------------
// 1) LOAD CSV
// ---------------------------------------------------------------------------
static void load_csv_data(const char *filename)
{
    FILE *fp = fopen(filename, "r");
    if (!fp) {
        perror("Error opening CSV file");
        exit(1);
    }

    char line[1024];
    if(!fgets(line, sizeof(line), fp)) {
        fprintf(stderr, "CSV file seems empty\n");
        fclose(fp);
        exit(1);
    }

    int row = 0;
    while (fgets(line, sizeof(line), fp)) {
        // Avoid overflow
        if (row >= MAX_SAMPLES) {
            fprintf(stderr, "Reached MAX_SAMPLES limit.\n");
            break;
        }

        // Expect: idx, date, open, high, low, close, volume
        char *token = strtok(line, ","); // index
        if(!token) continue;

        token = strtok(NULL, ","); // date
        if(!token) continue;

        token = strtok(NULL, ","); // open
        if(!token) continue;
        double openVal = atof(token);

        token = strtok(NULL, ","); // high
        if(!token) continue;
        double highVal = atof(token);

        token = strtok(NULL, ","); // low
        if(!token) continue;
        double lowVal = atof(token);

        token = strtok(NULL, ","); // close
        if(!token) continue;
        double closeVal = atof(token);

        token = strtok(NULL, ","); // volume
        if(!token) continue;
        double volumeVal = atof(token);

        openArr[row]   = openVal;
        highArr[row]   = highVal;
        lowArr[row]    = lowVal;
        closeArr[row]  = closeVal;
        volumeArr[row] = volumeVal;

        row++;
    }
    fclose(fp);

    NUM_SAMPLES = row;
    printf("Loaded %d samples from CSV.\n", NUM_SAMPLES);
}

// ---------------------------------------------------------------------------
// 2A) TECHNICAL INDICATORS
// ---------------------------------------------------------------------------
static void compute_obv()
{
    obvArr[0] = 0.0;
    for(int i = 1; i < NUM_SAMPLES; i++){
        if(closeArr[i] > closeArr[i-1]) {
            obvArr[i] = obvArr[i-1] + volumeArr[i];
        } else if(closeArr[i] < closeArr[i-1]) {
            obvArr[i] = obvArr[i-1] - volumeArr[i];
        } else {
            obvArr[i] = obvArr[i-1];
        }
    }
}

static void compute_ad_line()
{
    adArr[0] = 0.0;
    for(int i=1; i<NUM_SAMPLES; i++){
        double high  = highArr[i];
        double low   = lowArr[i];
        double close = closeArr[i];
        double range = (high - low) + 1e-9;

        double mfm = ((close - low) - (high - close)) / range;
        double mfv = mfm * volumeArr[i];
        adArr[i]   = adArr[i-1] + mfv;
    }
}

#define ADX_PERIOD 14
static void compute_adx()
{
    double *posDM = calloc(NUM_SAMPLES, sizeof(double));
    double *negDM = calloc(NUM_SAMPLES, sizeof(double));
    double *tr    = calloc(NUM_SAMPLES, sizeof(double));
    // Compute +DM, -DM, TR
    for(int i=1; i<NUM_SAMPLES; i++){
        double upMove   = highArr[i] - highArr[i-1];
        double downMove = lowArr[i-1] - lowArr[i];
        posDM[i] = (upMove > downMove && upMove > 0)   ? upMove : 0.0;
        negDM[i] = (downMove > upMove && downMove > 0) ? downMove : 0.0;

        double highLow   = highArr[i] - lowArr[i];
        double highClose = fabs(highArr[i] - closeArr[i-1]);
        double lowClose  = fabs(lowArr[i] - closeArr[i-1]);
        double trueRange = fmax(highLow, fmax(highClose, lowClose));
        tr[i] = trueRange;
    }

    // Smooth them
    double *smoothPosDM = calloc(NUM_SAMPLES, sizeof(double));
    double *smoothNegDM = calloc(NUM_SAMPLES, sizeof(double));
    double *smoothTR    = calloc(NUM_SAMPLES, sizeof(double));

    double sumPos=0, sumNeg=0, sumTR_=0;
    for(int i=1; i<=ADX_PERIOD && i<NUM_SAMPLES; i++){
        sumPos += posDM[i];
        sumNeg += negDM[i];
        sumTR_ += tr[i];
    }
    if(ADX_PERIOD < NUM_SAMPLES){
        smoothPosDM[ADX_PERIOD] = sumPos;
        smoothNegDM[ADX_PERIOD] = sumNeg;
        smoothTR[ADX_PERIOD]    = sumTR_;
    }
    for(int i=ADX_PERIOD+1; i<NUM_SAMPLES; i++){
        smoothPosDM[i] = smoothPosDM[i-1] - (smoothPosDM[i-1]/ADX_PERIOD) + posDM[i];
        smoothNegDM[i] = smoothNegDM[i-1] - (smoothNegDM[i-1]/ADX_PERIOD) + negDM[i];
        smoothTR[i]    = smoothTR[i-1]    - (smoothTR[i-1]/ADX_PERIOD)    + tr[i];
    }

    double *pdi = calloc(NUM_SAMPLES, sizeof(double));
    double *ndi = calloc(NUM_SAMPLES, sizeof(double));
    for(int i=ADX_PERIOD; i<NUM_SAMPLES; i++){
        if(smoothTR[i] < 1e-9){
            pdi[i] = 0.0;
            ndi[i] = 0.0;
        } else {
            pdi[i] = (smoothPosDM[i]/smoothTR[i])*100.0;
            ndi[i] = (smoothNegDM[i]/smoothTR[i])*100.0;
        }
    }

    double *dx = calloc(NUM_SAMPLES, sizeof(double));
    for(int i=ADX_PERIOD; i<NUM_SAMPLES; i++){
        double sum_ = pdi[i] + ndi[i];
        if(sum_<1e-9) dx[i] = 0.0;
        else dx[i] = (fabs(pdi[i]-ndi[i]) / sum_) * 100.0;
    }

    for(int i=0; i<ADX_PERIOD && i<NUM_SAMPLES; i++){
        adxArr[i] = 0.0;
    }
    for(int i=ADX_PERIOD; i<NUM_SAMPLES; i++){
        double sumDX=0.0; int count=0;
        for(int k=i-ADX_PERIOD+1; k<=i; k++){
            sumDX += dx[k];
            count++;
        }
        adxArr[i] = sumDX / count;
    }

    free(posDM); free(negDM); free(tr);
    free(smoothPosDM); free(smoothNegDM); free(smoothTR);
    free(pdi); free(ndi); free(dx);
}

#define AROON_PERIOD 14
static void compute_aroon()
{
    for(int i=0; i<NUM_SAMPLES; i++){
        aroonUpArr[i]=0.0; aroonDownArr[i]=0.0;
    }
    for(int i=AROON_PERIOD; i<NUM_SAMPLES; i++){
        double highestHigh = highArr[i];
        double lowestLow   = lowArr[i];
        int highestIndex=i, lowestIndex=i;
        for(int back=1; back<AROON_PERIOD; back++){
            int idx = i-back;
            if(highArr[idx]>highestHigh){
                highestHigh=highArr[idx];
                highestIndex=idx;
            }
            if(lowArr[idx]<lowestLow){
                lowestLow=lowArr[idx];
                lowestIndex=idx;
            }
        }
        double up   = ((AROON_PERIOD-(i-highestIndex))/(double)AROON_PERIOD)*100.0;
        double down = ((AROON_PERIOD-(i-lowestIndex)) /(double)AROON_PERIOD)*100.0;
        aroonUpArr[i]   = up;
        aroonDownArr[i] = down;
    }
}

#define FAST_PERIOD 12
#define SLOW_PERIOD 26
static void compute_macd()
{
    double *emaFast = calloc(NUM_SAMPLES, sizeof(double));
    double *emaSlow = calloc(NUM_SAMPLES, sizeof(double));

    emaFast[0] = closeArr[0];
    emaSlow[0] = closeArr[0];

    double alphaFast = 2.0/(FAST_PERIOD+1.0);
    double alphaSlow = 2.0/(SLOW_PERIOD+1.0);

    for(int i=1; i<NUM_SAMPLES; i++){
        emaFast[i] = alphaFast*closeArr[i] + (1-alphaFast)*emaFast[i-1];
        emaSlow[i] = alphaSlow*closeArr[i] + (1-alphaSlow)*emaSlow[i-1];
    }
    for(int i=0; i<NUM_SAMPLES; i++){
        macdArr[i] = emaFast[i] - emaSlow[i];
    }
    free(emaFast); free(emaSlow);
}

#define RSI_PERIOD 14
static void compute_rsi()
{
    double *gain=calloc(NUM_SAMPLES,sizeof(double));
    double *loss=calloc(NUM_SAMPLES,sizeof(double));

    for(int i=1; i<NUM_SAMPLES; i++){
        double change = closeArr[i] - closeArr[i-1];
        gain[i] = (change>0)? change:0.0;
        loss[i] = (change<0)? -change:0.0;
    }

    double avgGain=0, avgLoss=0;
    for(int i=1; i<=RSI_PERIOD && i<NUM_SAMPLES; i++){
        avgGain+=gain[i];
        avgLoss+=loss[i];
    }
    avgGain/=RSI_PERIOD; avgLoss/=RSI_PERIOD;

    if(RSI_PERIOD<NUM_SAMPLES){
        double rs = (avgLoss<1e-9)?0.0:(avgGain/avgLoss);
        rsiArr[RSI_PERIOD] = 100.0-(100.0/(1.0+rs));
    }

    for(int i=RSI_PERIOD+1; i<NUM_SAMPLES; i++){
        avgGain=(avgGain*(RSI_PERIOD-1)+gain[i])/(double)RSI_PERIOD;
        avgLoss=(avgLoss*(RSI_PERIOD-1)+loss[i])/(double)RSI_PERIOD;
        double rs=(avgLoss<1e-9)?0.0: (avgGain/avgLoss);
        rsiArr[i] = 100.0-(100.0/(1.0+rs));
    }
    for(int i=0; i<RSI_PERIOD && i<NUM_SAMPLES; i++){
        rsiArr[i]=0.0;
    }
    free(gain); free(loss);
}

#define STOCH_PERIOD 14
static void compute_stochastic()
{
    for(int i=0; i<NUM_SAMPLES; i++){
        stochArr[i]=0.0;
    }
    for(int i=STOCH_PERIOD; i<NUM_SAMPLES; i++){
        double highestHigh=highArr[i];
        double lowestLow=lowArr[i];
        for(int back=1; back<STOCH_PERIOD; back++){
            int idx = i-back;
            if(highArr[idx]>highestHigh) highestHigh=highArr[idx];
            if(lowArr[idx]<lowestLow)   lowestLow=lowArr[idx];
        }
        double denom=(highestHigh-lowestLow)+1e-9;
        stochArr[i]=((closeArr[i]-lowestLow)/denom)*100.0;
    }
}

static void compute_all_indicators()
{
    compute_obv();
    compute_ad_line();
    compute_adx();
    compute_aroon();
    compute_macd();
    compute_rsi();
    compute_stochastic(); // if you want to use it
}

// ---------------------------------------------------------------------------
// 2B) ASSEMBLE FEATURES
// ---------------------------------------------------------------------------
static void assemble_features()
{
    for(int i=0; i<NUM_SAMPLES; i++){
        // 4 base
        X_raw[i][0] = openArr[i];
        X_raw[i][1] = highArr[i];
        X_raw[i][2] = lowArr[i];
        X_raw[i][3] = volumeArr[i];

        // 7 indicators => total 11 features
        X_raw[i][4]  = obvArr[i];
        X_raw[i][5]  = adArr[i];
        X_raw[i][6]  = adxArr[i];
        X_raw[i][7]  = aroonUpArr[i];
        X_raw[i][8]  = aroonDownArr[i];
        X_raw[i][9]  = macdArr[i];
        X_raw[i][10] = rsiArr[i];
        // If you prefer to swap in stochArr, do it as needed

        y_raw[i]     = closeArr[i];
    }
}

// ---------------------------------------------------------------------------
// 3) NORMALIZATION
// ---------------------------------------------------------------------------
static void compute_mean_std()
{
    for(int j=0; j<INPUT_SIZE; j++){
        double sum=0, sum_sq=0;
        for(int i=0; i<NUM_SAMPLES; i++){
            sum    += X_raw[i][j];
            sum_sq += X_raw[i][j]*X_raw[i][j];
        }
        double mean = sum/NUM_SAMPLES;
        double var  = (sum_sq/NUM_SAMPLES)-(mean*mean);
        double sd   = sqrt(var+1e-9);
        X_mean[j]=mean; X_std[j]=sd;
    }

    double sumY=0, sumYsq=0;
    for(int i=0; i<NUM_SAMPLES; i++){
        sumY  += y_raw[i];
        sumYsq+= y_raw[i]*y_raw[i];
    }
    double meanY = sumY/NUM_SAMPLES;
    double varY  = (sumYsq/NUM_SAMPLES) - (meanY*meanY);
    double sdY   = sqrt(varY+1e-9);
    y_mean=meanY; y_std=sdY;
}

static void normalize_data(double X_norm[][INPUT_SIZE], double y_norm[])
{
    for(int i=0; i<NUM_SAMPLES; i++){
        for(int j=0; j<INPUT_SIZE; j++){
            X_norm[i][j] = (X_raw[i][j] - X_mean[j]) / (X_std[j]);
        }
        y_norm[i] = (y_raw[i] - y_mean)/y_std;
    }
}

// ---------------------------------------------------------------------------
// 4) INIT NETWORK
// ---------------------------------------------------------------------------
static double **alloc_2d(int rows, int cols);
static void init_network()
{
    layer_sizes[0] = INPUT_SIZE;  // 11
    for(int i=0; i<NUM_HIDDEN_LAYERS; i++){
        layer_sizes[i+1]=HIDDEN_SIZES[i];
    }
    layer_sizes[NUM_HIDDEN_LAYERS+1]=OUTPUT_SIZE; // 1

    srand((unsigned int)time(NULL));
    for(int L=0; L<NUM_HIDDEN_LAYERS+1; L++){
        int in_dim = layer_sizes[L];
        int out_dim= layer_sizes[L+1];

        W[L]=alloc_2d(out_dim, in_dim);
        b[L]=(double*)calloc(out_dim,sizeof(double));

        mW[L]=alloc_2d(out_dim, in_dim);
        vW[L]=alloc_2d(out_dim, in_dim);
        mb[L]=(double*)calloc(out_dim,sizeof(double));
        vb[L]=(double*)calloc(out_dim,sizeof(double));

        for(int i=0; i<out_dim; i++){
            b[L][i]=0.0;
            for(int j=0; j<in_dim; j++){
                W[L][i][j] = ((double)rand()/RAND_MAX -0.5);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// 5) FORWARD/BACKWARD
// ---------------------------------------------------------------------------
static void forward_pass(const double *x, double **acts)
{
    // input
    for(int i=0; i<layer_sizes[0]; i++){
        acts[0][i] = x[i];
    }
    // hidden -> out
    for(int L=0; L<NUM_HIDDEN_LAYERS+1; L++){
        int in_dim=layer_sizes[L];
        int out_dim=layer_sizes[L+1];
        for(int o=0; o<out_dim; o++){
            double z = b[L][o];
            for(int in_i=0; in_i<in_dim; in_i++){
                z += W[L][o][in_i]*acts[L][in_i];
            }
            if(L<NUM_HIDDEN_LAYERS) {
                // relu
                acts[L+1][o] = relu(z);
            } else {
                // linear
                acts[L+1][o] = z;
            }
        }
    }
}

static void backward_pass(const double *y_true, double **acts, double **deltas)
{
    int L_out=NUM_HIDDEN_LAYERS;
    int out_dim=layer_sizes[L_out+1];

    // output delta
    for(int i=0; i<out_dim; i++){
        double y_pred=acts[L_out+1][i];
        double error=y_pred-y_true[i];
        deltas[L_out+1][i]=error;
    }
    // hidden deltas
    for(int L=L_out; L>=1; L--){
        int out_dim_L=layer_sizes[L];
        int out_dim_Lplus=layer_sizes[L+1];

        for(int i=0; i<out_dim_L; i++){
            double d_act=relu_derivative(acts[L][i]);
            double sum_=0.0;
            for(int k=0; k<out_dim_Lplus; k++){
                sum_ += W[L][k][i]*deltas[L+1][k];
            }
            deltas[L][i]=sum_*d_act;
        }
    }
}

// ---------------------------------------------------------------------------
// 6) ADAM UPDATE
// ---------------------------------------------------------------------------
static double pow_beta1_t=1.0, pow_beta2_t=1.0;

static void adam_update_weights(double **acts, double **deltas,int batch_size,int t)
{
    pow_beta1_t*=BETA1; 
    pow_beta2_t*=BETA2;

    for(int L=0; L<NUM_HIDDEN_LAYERS+1; L++){
        int in_dim=layer_sizes[L];
        int out_dim=layer_sizes[L+1];
        for(int o=0; o<out_dim; o++){
            double grad_b=deltas[L+1][o];
            mb[L][o] = BETA1*mb[L][o] + (1.0-BETA1)*grad_b;
            vb[L][o] = BETA2*vb[L][o] + (1.0-BETA2)*(grad_b*grad_b);

            double m_hat_b= mb[L][o]/(1.0-pow_beta1_t);
            double v_hat_b= vb[L][o]/(1.0-pow_beta2_t);

            b[L][o] -= ADAM_LEARNING_RATE*(m_hat_b/(sqrt(v_hat_b)+EPSILON)) / batch_size;

            for(int in_i=0; in_i<in_dim; in_i++){
                double grad_w = deltas[L+1][o]*acts[L][in_i];
                grad_w += L2_LAMBDA*W[L][o][in_i];

                mW[L][o][in_i] = BETA1*mW[L][o][in_i] + (1.0-BETA1)*grad_w;
                vW[L][o][in_i] = BETA2*vW[L][o][in_i] + (1.0-BETA2)*(grad_w*grad_w);

                double m_hat= mW[L][o][in_i]/(1.0-pow_beta1_t);
                double v_hat= vW[L][o][in_i]/(1.0-pow_beta2_t);

                W[L][o][in_i] -= ADAM_LEARNING_RATE*(m_hat/(sqrt(v_hat)+EPSILON))/batch_size;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Shuffle
// ---------------------------------------------------------------------------
static void shuffle_indices(int *indices, int n)
{
    for(int i=0; i<n; i++){
        int r = i + rand()%(n-i);
        int temp=indices[i];
        indices[i]=indices[r];
        indices[r]=temp;
    }
}

// ---------------------------------------------------------------------------
// 7) TRAIN
// ---------------------------------------------------------------------------
static void train_network(double X_norm[][INPUT_SIZE], double y_norm[])
{
    double **activations=malloc((NUM_HIDDEN_LAYERS+2)*sizeof(double*));
    double **deltas=     malloc((NUM_HIDDEN_LAYERS+2)*sizeof(double*));
    for(int L=0; L<NUM_HIDDEN_LAYERS+2; L++){
        activations[L]=calloc(layer_sizes[L],sizeof(double));
        deltas[L]=calloc(layer_sizes[L],sizeof(double));
    }
    int *indices=malloc(NUM_SAMPLES*sizeof(int));
    for(int i=0; i<NUM_SAMPLES; i++){
        indices[i]=i;
    }

    int steps=0;
    for(int epoch=0; epoch<MAX_EPOCHS; epoch++){
        shuffle_indices(indices,NUM_SAMPLES);

        double epoch_loss=0.0;
        int num_batches=(NUM_SAMPLES+BATCH_SIZE-1)/BATCH_SIZE;

        for(int b_i=0; b_i<num_batches; b_i++){
            int start=b_i*BATCH_SIZE;
            int end=start+BATCH_SIZE;
            if(end>NUM_SAMPLES) end=NUM_SAMPLES;

            for(int L=0; L<NUM_HIDDEN_LAYERS+2; L++){
                memset(deltas[L],0, layer_sizes[L]*sizeof(double));
            }

            double batch_loss=0.0;
            for(int m_i=start; m_i<end; m_i++){
                int idx=indices[m_i];
                // forward
                forward_pass(X_norm[idx],activations);
                // MSE
                double y_pred=activations[NUM_HIDDEN_LAYERS+1][0];
                double error=y_pred-y_norm[idx];
                batch_loss += 0.5*error*error;

                // backward
                backward_pass(&y_norm[idx], activations, deltas);
            }
            steps++;
            adam_update_weights(activations, deltas, (end-start), steps);

            epoch_loss+=batch_loss;
        }

        if(epoch%PRINT_INTERVAL==0){
            double avg_loss=epoch_loss/NUM_SAMPLES;
            printf("Epoch %d, Loss = %f\n", epoch, avg_loss);
        }
    }

    free(indices);
    for(int L=0; L<NUM_HIDDEN_LAYERS+2; L++){
        free(activations[L]);
        free(deltas[L]);
    }
    free(activations);
    free(deltas);
}

// ---------------------------------------------------------------------------
// 8) PREDICT
// ---------------------------------------------------------------------------
static double predict(const double *x_input)
{
    double **acts=malloc((NUM_HIDDEN_LAYERS+2)*sizeof(double*));
    for(int L=0; L<NUM_HIDDEN_LAYERS+2; L++){
        acts[L]=calloc(layer_sizes[L],sizeof(double));
    }
    forward_pass(x_input, acts);

    double y_pred_norm=acts[NUM_HIDDEN_LAYERS+1][0];

    for(int L=0; L<NUM_HIDDEN_LAYERS+2; L++){
        free(acts[L]);
    }
    free(acts);

    double y_pred=(y_pred_norm*y_std)+y_mean;
    return y_pred;
}

// ---------------------------------------------------------------------------
// EXTRA: PRINT PREDICTIONS & METRICS
// ---------------------------------------------------------------------------
static void print_predictions_and_metrics(double X_norm[][INPUT_SIZE], double y_norm[])
{
    // We'll predict the entire dataset, compute errors, and print some rows
    double mse=0.0, mae=0.0;
    double *predictions=malloc(NUM_SAMPLES*sizeof(double));

    for(int i=0; i<NUM_SAMPLES; i++){
        predictions[i] = predict(X_norm[i]);
    }

    // Let's print the first 20 or so:
    int rows_to_print = (NUM_SAMPLES<20)? NUM_SAMPLES : 20;
    printf("\n-----------------------------------------------\n");
    printf(" Index |  Actual   | Predicted | Abs Error \n");
    printf("-----------------------------------------------\n");
    for(int i=0; i<rows_to_print; i++){
        double actual = y_raw[i];
        double pred   = predictions[i];
        double err    = fabs(pred-actual);
        printf(" %5d | %9.4f | %9.4f | %9.4f\n",
               i, actual, pred, err);
    }
    printf("-----------------------------------------------\n");
    if(NUM_SAMPLES>rows_to_print) {
        printf("  (... omitted %d rows ...)\n", NUM_SAMPLES-rows_to_print);
    }

    // Compute MSE, MAE
    for(int i=0; i<NUM_SAMPLES; i++){
        double err=(predictions[i]-y_raw[i]);
        mse += err*err;
        mae += fabs(err);
    }
    mse /= NUM_SAMPLES;
    mae /= NUM_SAMPLES;
    double rmse=sqrt(mse);

    printf("\nError Metrics (all %d samples):\n", NUM_SAMPLES);
    printf(" - MSE : %f\n", mse);
    printf(" - RMSE: %f\n", rmse);
    printf(" - MAE : %f\n", mae);

    free(predictions);
}

// ---------------------------------------------------------------------------
// EXTRA: FEATURE IMPORTANCE (NAIVE)
// ---------------------------------------------------------------------------
// We'll do a simple sum of absolute weights from the first layer
// i.e. feature_importance[j] = sum over hidden_neurons of |W[0][neuron][j]|
static void print_feature_importance()
{
    int in_dim = layer_sizes[0];     // 11
    int out_dim= layer_sizes[1];     // 8 if HIDDEN_SIZES[0]=8
    double *importance=calloc(in_dim,sizeof(double));

    // Sum abs value of weights for the FIRST layer only
    for(int j=0; j<in_dim; j++){
        double sum_=0.0;
        for(int neuron=0; neuron<out_dim; neuron++){
            double w = W[0][neuron][j];
            sum_ += fabs(w);
        }
        importance[j]=sum_;
    }

    // We'll define some labels:
    // 0:Open,1:High,2:Low,3:Volume,4:OBV,5:AD,6:ADX,7:AroonUp,8:AroonDown,9:MACD,10:RSI
    const char* feature_names[INPUT_SIZE] = {
        "Open", "High", "Low", "Volume","OBV","A/D","ADX","AroonUp","AroonDown","MACD","RSI"
        // or rename if you put stoch in the array
    };

    // We want to print them sorted by importance
    // Let's create an array of indices and sort
    int *idxs=malloc(in_dim*sizeof(int));
    for(int i=0; i<in_dim; i++){
        idxs[i]=i;
    }
    // simple bubble sort or so
    for(int i=0; i<in_dim-1; i++){
        for(int j=0; j<in_dim-1-i; j++){
            if(importance[idxs[j]]<importance[idxs[j+1]]) {
                int tmp=idxs[j];
                idxs[j]=idxs[j+1];
                idxs[j+1]=tmp;
            }
        }
    }

    printf("\nNaive Feature Importance (sum of abs W in first layer):\n");
    printf("-------------------------------------------------------\n");
    for(int i=0; i<in_dim; i++){
        int f=idxs[i];
        printf(" %10s: %.6f\n", feature_names[f], importance[f]);
    }
    printf("-------------------------------------------------------\n");
    free(importance);
    free(idxs);
}

// ---------------------------------------------------------------------------
// MAIN
// ---------------------------------------------------------------------------
int main(void)
{
    // 1) Load CSV
    load_csv_data("3_month_testing_data.csv");
    if(NUM_SAMPLES<2) {
        fprintf(stderr,"Not enough data.\n");
        return 1;
    }

    // 2) Compute indicators
    compute_all_indicators();

    // 3) Assemble features
    assemble_features();

    // 4) Normalize
    compute_mean_std();
    double (*X_norm)[INPUT_SIZE] = malloc(NUM_SAMPLES*sizeof(*X_norm));
    double *y_norm=malloc(NUM_SAMPLES*sizeof(double));
    normalize_data(X_norm,y_norm);

    // 5) Init & Train
    init_network();
    printf("Starting training on %d samples...\n",NUM_SAMPLES);
    train_network(X_norm,y_norm);
    printf("Training complete.\n");

    // 6) Print extended predictions & metrics
    print_predictions_and_metrics(X_norm,y_norm);

    // 7) Print naive feature importance
    print_feature_importance();

    // 8) Cleanup
    for(int L=0; L<NUM_HIDDEN_LAYERS+1; L++){
        int out_dim=layer_sizes[L+1];
        free_2d(W[L],out_dim);
        free_2d(mW[L],out_dim);
        free_2d(vW[L],out_dim);
        free(b[L]);
        free(mb[L]);
        free(vb[L]);
    }
    free(X_norm);
    free(y_norm);

    return 0;
}

