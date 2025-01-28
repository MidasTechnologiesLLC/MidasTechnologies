/*****************************************************************************
 * stock_predictor_enhanced.c
 *
 * Updated to parse real CSV data:
 *   - Reads 3_month_testing_data.csv
 *   - Uses columns [Open, High, Low, Volume] as inputs
 *   - Uses column [Close] as target
 *****************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

/***************************************
 * Configuration and Globals           *
 ***************************************/

// Adjust these if your CSV is large
#define MAX_SAMPLES   100000  // max number of rows to read from CSV
#define INPUT_SIZE    4       // [Open, High, Low, Volume]
#define OUTPUT_SIZE   1       // we're predicting [Close]
#define NUM_HIDDEN_LAYERS 2
static const int HIDDEN_SIZES[NUM_HIDDEN_LAYERS] = {8, 6};

// Adam hyperparams
#define ADAM_LEARNING_RATE 0.01
#define BETA1 0.9
#define BETA2 0.999
#define EPSILON 1e-8

// L2 regularization term
#define L2_LAMBDA 0.001

// Training settings
#define MAX_EPOCHS  1000
#define BATCH_SIZE  32        // you can change this
#define PRINT_INTERVAL 100    // print loss every 100 epochs

// We’ll fill these from the CSV file
static int NUM_SAMPLES = 0;

// Global arrays to hold raw data from CSV
static double X_raw[MAX_SAMPLES][INPUT_SIZE];
static double y_raw[MAX_SAMPLES];

/***************************************
 * Normalization Buffers (Global)      *
 ***************************************/
static double X_mean[INPUT_SIZE];
static double X_std[INPUT_SIZE];
static double y_mean;
static double y_std;

/***************************************
 * Weights & Biases Data Structures    *
 ***************************************/

// layer_sizes[0] = INPUT_SIZE
// layer_sizes[1] = HIDDEN_SIZES[0]
// layer_sizes[2] = HIDDEN_SIZES[1]
// layer_sizes[3] = OUTPUT_SIZE
static int layer_sizes[NUM_HIDDEN_LAYERS + 2];

// Weight arrays: W[L] with shape [layer_sizes[L+1]][layer_sizes[L]]
// Bias arrays:   b[L] with shape [layer_sizes[L+1]]
static double **W[NUM_HIDDEN_LAYERS + 1];
static double  *b[NUM_HIDDEN_LAYERS + 1];

// For Adam, we store momentum (m) and velocity (v) for each weight & bias
static double **mW[NUM_HIDDEN_LAYERS + 1], **vW[NUM_HIDDEN_LAYERS + 1];
static double  *mb[NUM_HIDDEN_LAYERS + 1],  *vb[NUM_HIDDEN_LAYERS + 1];

/***************************************
 * Utilities                           *
 ***************************************/

static inline double relu(double x) {
    return x > 0.0 ? x : 0.0;
}
static inline double relu_derivative(double x) {
    return x > 0.0 ? 1.0 : 0.0;
}

// Allocate a 2D array of size rows x cols
static double **alloc_2d(int rows, int cols)
{
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
static void free_2d(double **arr, int rows)
{
    if(!arr) return;
    for(int i = 0; i < rows; i++){
        free(arr[i]);
    }
    free(arr);
}

/***************************************
 * Step 1: CSV Parsing                 *
 ***************************************/

// We expect CSV with a header like:
// Index,Date,Open,High,Low,Close,Volume
// We'll read up to MAX_SAMPLES lines
static void load_csv_data(const char *filename)
{
    FILE *fp = fopen(filename, "r");
    if (!fp) {
        perror("Error opening CSV file");
        exit(1);
    }

    // Read and discard the header line
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
            fprintf(stderr, "Reached MAX_SAMPLES limit (%d), some data not read.\n", MAX_SAMPLES);
            break;
        }

        // Tokenize by comma
        char *token = strtok(line, ","); // index
        if(!token) continue; // skip malformed line

        // skip index (already in token)
        token = strtok(NULL, ",");       // Date (we skip or ignore for now)
        if(!token) continue;

        // next token = Open
        token = strtok(NULL, ","); 
        if(!token) continue;
        double openVal = atof(token);

        // next token = High
        token = strtok(NULL, ",");
        if(!token) continue;
        double highVal = atof(token);

        // next token = Low
        token = strtok(NULL, ",");
        if(!token) continue;
        double lowVal = atof(token);

        // next token = Close
        token = strtok(NULL, ",");
        if(!token) continue;
        double closeVal = atof(token);

        // next token = Volume
        token = strtok(NULL, ",");
        if(!token) continue;
        double volumeVal = atof(token);

        // Store in global arrays
        X_raw[row][0] = openVal;
        X_raw[row][1] = highVal;
        X_raw[row][2] = lowVal;
        X_raw[row][3] = volumeVal;   // 4 features: open, high, low, volume
        y_raw[row]    = closeVal;    // target = close

        row++;
    }

    fclose(fp);

    // The actual number of samples read
    NUM_SAMPLES = row;
    printf("Loaded %d samples from CSV.\n", NUM_SAMPLES);
}

/***************************************
 * Step 2: Data Normalization          *
 ***************************************/

// We'll standardize inputs: x' = (x - mean) / std
// We'll also standardize targets: y' = (y - mean) / y_std
static void compute_mean_std()
{
    // Compute mean & std for each input dimension
    for(int j = 0; j < INPUT_SIZE; j++) {
        double sum = 0.0, sum_sq = 0.0;
        for(int i = 0; i < NUM_SAMPLES; i++) {
            sum    += X_raw[i][j];
            sum_sq += X_raw[i][j] * X_raw[i][j];
        }
        double mean = sum / NUM_SAMPLES;
        double var  = (sum_sq / NUM_SAMPLES) - (mean * mean);
        double std  = sqrt(var + 1e-9); // add small epsilon

        X_mean[j] = mean;
        X_std[j]  = std;
    }

    // Compute mean & std for target
    {
        double sum = 0.0, sum_sq = 0.0;
        for(int i = 0; i < NUM_SAMPLES; i++) {
            sum    += y_raw[i];
            sum_sq += y_raw[i] * y_raw[i];
        }
        double mean = sum / NUM_SAMPLES;
        double var  = (sum_sq / NUM_SAMPLES) - (mean * mean);
        double std  = sqrt(var + 1e-9);

        y_mean = mean;
        y_std  = std;
    }
}

static void normalize_data(double X_norm[][INPUT_SIZE], double y_norm[])
{
    // Standardize X
    for(int i = 0; i < NUM_SAMPLES; i++){
        for(int j = 0; j < INPUT_SIZE; j++){
            X_norm[i][j] = (X_raw[i][j] - X_mean[j]) / X_std[j];
        }
    }
    // Standardize y
    for(int i = 0; i < NUM_SAMPLES; i++){
        y_norm[i] = (y_raw[i] - y_mean) / y_std;
    }
}

/***************************************
 * Step 3: Init the Network            *
 ***************************************/

static void init_network()
{
    // Fill layer_sizes array
    layer_sizes[0] = INPUT_SIZE;               // e.g. 4
    for(int i = 0; i < NUM_HIDDEN_LAYERS; i++){
        layer_sizes[i+1] = HIDDEN_SIZES[i];    // e.g. 8, 6
    }
    layer_sizes[NUM_HIDDEN_LAYERS + 1] = OUTPUT_SIZE; // 1

    // Initialize random
    srand((unsigned int)time(NULL));

    // For each layer L
    for(int L = 0; L < NUM_HIDDEN_LAYERS + 1; L++){
        int in_dim  = layer_sizes[L];
        int out_dim = layer_sizes[L+1];

        // Allocate W[L] & b[L]
        W[L] = alloc_2d(out_dim, in_dim);
        b[L] = (double *)calloc(out_dim, sizeof(double));

        // Allocate Adam buffers
        mW[L] = alloc_2d(out_dim, in_dim);
        vW[L] = alloc_2d(out_dim, in_dim);
        mb[L] = (double *)calloc(out_dim, sizeof(double));
        vb[L] = (double *)calloc(out_dim, sizeof(double));

        // Random initialization
        for(int i = 0; i < out_dim; i++){
            b[L][i] = 0.0; // or small random
            for(int j = 0; j < in_dim; j++){
                // He initialization example:
                // double scale = sqrt(2.0 / in_dim);
                // W[L][i][j] = scale * ((double)rand() / RAND_MAX - 0.5);

                // Uniform in [-0.5, 0.5]
                W[L][i][j] = ((double)rand() / RAND_MAX - 0.5);
            }
        }
    }
}

/***************************************
 * Forward Pass                        *
 ***************************************/

static void forward_pass(const double *x, double **activations)
{
    // Copy x into activations[0]
    for(int i = 0; i < layer_sizes[0]; i++){
        activations[0][i] = x[i];
    }

    // For each layer L
    for(int L = 0; L < NUM_HIDDEN_LAYERS + 1; L++){
        int in_dim  = layer_sizes[L];
        int out_dim = layer_sizes[L+1];

        for(int out_i = 0; out_i < out_dim; out_i++){
            double z = b[L][out_i];
            // Weighted sum
            for(int in_i = 0; in_i < in_dim; in_i++){
                z += W[L][out_i][in_i] * activations[L][in_i];
            }
            // Activation
            if(L < NUM_HIDDEN_LAYERS){
                // Hidden layer -> ReLU
                activations[L+1][out_i] = relu(z);
            } else {
                // Output layer -> linear
                activations[L+1][out_i] = z;
            }
        }
    }
}

/***************************************
 * Backward Pass (Compute Deltas)      *
 ***************************************/

static void backward_pass(const double *y_true, double **activations, double **deltas)
{
    // Output layer delta
    // MSE: dL/dy_pred = (y_pred - y_true)
    int L_out = NUM_HIDDEN_LAYERS; // index of last hidden layer
    int out_dim = layer_sizes[L_out + 1];
    for(int i = 0; i < out_dim; i++){
        double y_pred = activations[L_out + 1][i];
        double error  = y_pred - y_true[i];
        deltas[L_out + 1][i] = error; // linear activation => direct
    }

    // Hidden layers
    for(int L = L_out; L >= 1; L--){
        int out_dim_L = layer_sizes[L];
        int out_dim_Lplus = layer_sizes[L+1];

        for(int i = 0; i < out_dim_L; i++){
            double d_act = relu_derivative(activations[L][i]);
            // sum over next layer
            double sum_ = 0.0;
            for(int k = 0; k < out_dim_Lplus; k++){
                sum_ += W[L][k][i] * deltas[L+1][k];
            }
            deltas[L][i] = sum_ * d_act;
        }
    }
}

/***************************************
 * Step 4: Adam Update (Mini-Batch)    *
 ***************************************/

static double pow_beta1_t = 1.0;
static double pow_beta2_t = 1.0;

static void adam_update_weights(
    double **activations, double **deltas,
    int batch_size, int t /*current step*/)
{
    // Increment these for bias correction in Adam
    pow_beta1_t *= BETA1;
    pow_beta2_t *= BETA2;

    // For each layer
    for(int L = 0; L < NUM_HIDDEN_LAYERS + 1; L++){
        int in_dim  = layer_sizes[L];
        int out_dim = layer_sizes[L+1];

        for(int out_i = 0; out_i < out_dim; out_i++){
            // Grad for bias
            double grad_b = deltas[L+1][out_i];

            // Accumulate in m, v
            mb[L][out_i] = BETA1 * mb[L][out_i] + (1.0 - BETA1) * grad_b;
            vb[L][out_i] = BETA2 * vb[L][out_i] + (1.0 - BETA2) * (grad_b * grad_b);

            // Bias-corrected m, v
            double m_hat_b = mb[L][out_i] / (1.0 - pow_beta1_t);
            double v_hat_b = vb[L][out_i] / (1.0 - pow_beta2_t);

            // Update bias
            b[L][out_i] -= ADAM_LEARNING_RATE * (m_hat_b / (sqrt(v_hat_b) + EPSILON))
                           / batch_size; // average over batch

            for(int in_i = 0; in_i < in_dim; in_i++){
                // Weight gradient
                double grad_w = deltas[L+1][out_i] * activations[L][in_i];
                // Add L2 regularization
                grad_w += L2_LAMBDA * W[L][out_i][in_i];

                // Accumulate in m, v
                mW[L][out_i][in_i] = BETA1 * mW[L][out_i][in_i] + (1.0 - BETA1) * grad_w;
                vW[L][out_i][in_i] = BETA2 * vW[L][out_i][in_i] + (1.0 - BETA2) * (grad_w * grad_w);

                double m_hat = mW[L][out_i][in_i] / (1.0 - pow_beta1_t);
                double v_hat = vW[L][out_i][in_i] / (1.0 - pow_beta2_t);

                // Weight update
                W[L][out_i][in_i] -= ADAM_LEARNING_RATE * (m_hat / (sqrt(v_hat) + EPSILON))
                                     / batch_size;
            }
        }
    }
}

/***************************************
 * Shuffling Utility                   *
 ***************************************/

static void shuffle_indices(int *indices, int n)
{
    for(int i = 0; i < n; i++){
        int r = i + rand() % (n - i);
        int temp = indices[i];
        indices[i] = indices[r];
        indices[r] = temp;
    }
}

/***************************************
 * Step 5: Train Loop                  *
 ***************************************/

static void train_network(double X_norm[][INPUT_SIZE], double y_norm[])
{
    // Prepare activation & delta arrays
    // For each of (NUM_HIDDEN_LAYERS+2) layers, we store up to layer_sizes[L] values.
    double **activations = (double **)malloc((NUM_HIDDEN_LAYERS+2) * sizeof(double *));
    double **deltas      = (double **)malloc((NUM_HIDDEN_LAYERS+2) * sizeof(double *));
    for(int L = 0; L < NUM_HIDDEN_LAYERS+2; L++){
        activations[L] = (double *)calloc(layer_sizes[L], sizeof(double));
        deltas[L]      = (double *)calloc(layer_sizes[L], sizeof(double));
    }

    // Indices for shuffling
    int *indices = (int *)malloc(NUM_SAMPLES * sizeof(int));
    for(int i = 0; i < NUM_SAMPLES; i++) indices[i] = i;

    int steps = 0; // Adam steps

    for(int epoch = 0; epoch < MAX_EPOCHS; epoch++){
        // Shuffle each epoch
        shuffle_indices(indices, NUM_SAMPLES);

        double epoch_loss = 0.0;

        // Process mini-batches
        int num_batches = (NUM_SAMPLES + BATCH_SIZE - 1) / BATCH_SIZE;

        for(int b_i = 0; b_i < num_batches; b_i++){
            int start = b_i * BATCH_SIZE;
            int end   = start + BATCH_SIZE;
            if(end > NUM_SAMPLES) end = NUM_SAMPLES;

            // We'll accumulate deltas in this batch
            // First, zero out the deltas for all layers
            for(int L = 0; L < NUM_HIDDEN_LAYERS+2; L++){
                for(int k = 0; k < layer_sizes[L]; k++){
                    deltas[L][k] = 0.0;
                }
            }

            double batch_loss = 0.0;

            // Forward+Backward for each sample in this batch
            for(int m_i = start; m_i < end; m_i++){
                int idx = indices[m_i];
                // 1) Forward
                forward_pass(X_norm[idx], activations);

                // 2) Compute loss (MSE = 0.5 * error^2)
                double y_pred = activations[NUM_HIDDEN_LAYERS+1][0];
                double error  = y_pred - y_norm[idx];
                double loss   = 0.5 * error * error;
                batch_loss   += loss;

                // 3) Backward
                backward_pass(&y_norm[idx], activations, deltas);
            }

            // 4) Update weights once for the entire batch
            steps++;
            adam_update_weights(activations, deltas, (end - start), steps);

            epoch_loss += batch_loss;
        }

        // Print progress
        if(epoch % PRINT_INTERVAL == 0){
            double avg_loss = epoch_loss / NUM_SAMPLES;
            printf("Epoch %d, Loss = %f\n", epoch, avg_loss);
        }
    }

    // Cleanup
    free(indices);
    for(int L = 0; L < NUM_HIDDEN_LAYERS+2; L++){
        free(activations[L]);
        free(deltas[L]);
    }
    free(activations);
    free(deltas);
}

/***************************************
 * Step 6: Inference (Predict)         *
 ***************************************/

static double predict(const double *x_input)
{
    // Create temporary activation array
    double **activations = (double **)malloc((NUM_HIDDEN_LAYERS+2)*sizeof(double *));
    for(int L=0; L < NUM_HIDDEN_LAYERS+2; L++){
        activations[L] = (double *)calloc(layer_sizes[L], sizeof(double));
    }

    // Forward pass
    forward_pass(x_input, activations);

    // Grab output
    double y_pred_norm = activations[NUM_HIDDEN_LAYERS+1][0];

    // Free
    for(int L=0; L < NUM_HIDDEN_LAYERS+2; L++){
        free(activations[L]);
    }
    free(activations);

    // "Denormalize" the prediction
    double y_pred = (y_pred_norm * y_std) + y_mean;
    return y_pred;
}

/***************************************
 * Main                                *
 ***************************************/

int main(void)
{
    // 1) Load CSV data
    load_csv_data("3_month_testing_data.csv");
    if (NUM_SAMPLES < 2) {
        fprintf(stderr, "Not enough samples to train.\n");
        return 1;
    }

    // 2) Compute data stats & normalize
    compute_mean_std();

    // We'll create arrays to hold the normalized data
    double (*X_norm)[INPUT_SIZE] = malloc(NUM_SAMPLES * sizeof(*X_norm));
    double *y_norm = malloc(NUM_SAMPLES * sizeof(double));
    if(!X_norm || !y_norm){
        fprintf(stderr, "Allocation error for normalized data\n");
        return 1;
    }

    normalize_data(X_norm, y_norm);

    // 3) Initialize network
    init_network();

    // 4) Train
    printf("Starting training on %d samples...\n", NUM_SAMPLES);
    train_network(X_norm, y_norm);
    printf("Training complete.\n");

    // 5) Predict on a couple of “test” points
    // For example, let's pick row 0 and row NUM_SAMPLES-1 in raw scale
    // (Or any custom data you'd like.)

    // We'll do it manually here:
    if(NUM_SAMPLES > 1) {
        double sample_raw[INPUT_SIZE] = {
            X_raw[0][0], // open
            X_raw[0][1], // high
            X_raw[0][2], // low
            X_raw[0][3]  // volume
        };
        // Normalize
        double sample_norm[INPUT_SIZE];
        for(int i=0; i<INPUT_SIZE; i++){
            sample_norm[i] = (sample_raw[i] - X_mean[i]) / X_std[i];
        }
        double pred1 = predict(sample_norm);
        printf("Prediction for row 0's close = %.4f (actual=%.4f)\n", pred1, y_raw[0]);
    }

    if(NUM_SAMPLES > 10) {
        // arbitrary row e.g. 10
        double sample_raw2[INPUT_SIZE] = {
            X_raw[10][0],
            X_raw[10][1],
            X_raw[10][2],
            X_raw[10][3]
        };
        double sample_norm2[INPUT_SIZE];
        for(int i=0; i<INPUT_SIZE; i++){
            sample_norm2[i] = (sample_raw2[i] - X_mean[i]) / X_std[i];
        }
        double pred2 = predict(sample_norm2);
        printf("Prediction for row 10's close = %.4f (actual=%.4f)\n", pred2, y_raw[10]);
    }

    // 6) Cleanup all W & b
    for(int L=0; L < NUM_HIDDEN_LAYERS + 1; L++){
        int out_dim = layer_sizes[L+1];
        free_2d(W[L], out_dim);
        free_2d(mW[L], out_dim);
        free_2d(vW[L], out_dim);
        free(b[L]);
        free(mb[L]);
        free(vb[L]);
    }

    free(X_norm);
    free(y_norm);

    return 0;
}

