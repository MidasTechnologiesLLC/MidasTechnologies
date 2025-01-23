Below is a **minimalist** example of how one might **program an LSTM forward pass in C** (and do a very rough training loop) to handle a **sequence** of inputs, such as your macro/micro/technical factors plus past prices. This is **not production-ready**—it’s a demonstration of the core ideas. In real practice, you’d want to handle:

1. **Memory management** carefully (possibly dynamic allocation).  
2. **Vectorized operations** for speed (e.g., SSE, AVX, CUDA, etc.).  
3. **Gradient computation** (backpropagation through time) if you actually want to *train* the LSTM in C.  
4. **Data loading** from files or other sources.  

Nevertheless, this skeleton code shows how to **structure** an LSTM in pure C, perform a **forward pass** over a time series, and do a *very basic* gradient step. You could adapt it for your multi-factor stock model by constructing the input vector \(\mathbf{x}(t)\) from your macro/micro/technical variables.

---

## Table of Contents
1. **Definitions and Structures**  
2. **Helper Functions (Activation, Matrix Ops)**  
3. **LSTM Forward Pass**  
4. **Basic MSE Loss & Naive Training Loop**  
5. **Example `main()`**  

---

# 1. Definitions and Structures

### 1.1 LSTM Hyperparameters

```c
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define INPUT_SIZE  8   // e.g., number of features: macro + micro + tech + maybe past price
#define HIDDEN_SIZE 4   // size of LSTM hidden state
#define OUTPUT_SIZE 1   // e.g. predict a single scalar: P(t)

// For toy example:
#define TIME_STEPS  5   // length of a single sequence
#define BATCH_SIZE  1   // for simplicity

// A small learning rate for naive training:
#define LEARNING_RATE 0.001
```

Here,
- `INPUT_SIZE` might be 8 if you combine, say, GDP, inflation, interest, earnings, sales, competition, momentum, etc.  
- `HIDDEN_SIZE` is how many hidden units (cells) the LSTM has.  
- `OUTPUT_SIZE` is the dimension of the output (predict 1D stock price).  
- `TIME_STEPS` is the length of the sequence we unroll over.  
- `LEARNING_RATE` is just for a naive gradient update.

### 1.2 LSTM Parameter Structure

We’ll store weights and biases for **all gates** in one struct. Recall each gate (input, forget, output, candidate) has its own weight matrices and biases. A naive layout:

- **W_x**: Weight matrix for input-to-hidden (dimensions \(\mathrm{HIDDEN\_SIZE} \times \mathrm{INPUT\_SIZE}\))  
- **W_h**: Weight matrix for hidden-to-hidden (dimensions \(\mathrm{HIDDEN\_SIZE} \times \mathrm{HIDDEN\_SIZE}\))  
- **b**: Bias vector (dimensions \(\mathrm{HIDDEN\_SIZE}\))

We do this for the input, forget, output gates, and the candidate cell state. That’s 4 sets of parameters.

```c
typedef struct {
    // Input gate
    float W_ix[HIDDEN_SIZE][INPUT_SIZE]; 
    float W_ih[HIDDEN_SIZE][HIDDEN_SIZE];
    float b_i[HIDDEN_SIZE];
    
    // Forget gate
    float W_fx[HIDDEN_SIZE][INPUT_SIZE];
    float W_fh[HIDDEN_SIZE][HIDDEN_SIZE];
    float b_f[HIDDEN_SIZE];
    
    // Output gate
    float W_ox[HIDDEN_SIZE][INPUT_SIZE];
    float W_oh[HIDDEN_SIZE][HIDDEN_SIZE];
    float b_o[HIDDEN_SIZE];

    // Candidate gate (cell update)
    float W_cx[HIDDEN_SIZE][INPUT_SIZE];
    float W_ch[HIDDEN_SIZE][HIDDEN_SIZE];
    float b_c[HIDDEN_SIZE];
    
    // Output layer (hidden->output)
    float W_hy[OUTPUT_SIZE][HIDDEN_SIZE]; 
    float b_y[OUTPUT_SIZE];
    
} LSTMParams;
```

We will also store the **hidden state** \(\mathbf{h}(t)\) and **cell state** \(\mathbf{c}(t)\) in a separate struct for clarity:

```c
typedef struct {
    float h[HIDDEN_SIZE];  // hidden state
    float c[HIDDEN_SIZE];  // cell state
} LSTMState;
```

---

# 2. Helper Functions (Activation, Matrix Ops)

We need a **sigmoid** function and possibly **tanh**. We also need some small matrix or vector routines.

```c
// Sigmoid activation
float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// Tanh activation
float tanh_approx(float x) {
    // We can just use the standard tanh from math.h
    return tanhf(x);
}

// Vector + Bias -> In-place addition
void add_bias(float *y, float *b, int size) {
    for(int i = 0; i < size; i++) {
        y[i] += b[i];
    }
}

// y = W * x  (simple matrix-vector multiply), W is [out_dim x in_dim], x is [in_dim]
void matvec(const float *W, const float *x, float *y, int out_dim, int in_dim) {
    // We'll assume W is laid out row-major
    for(int i = 0; i < out_dim; i++) {
        float sum = 0.0f;
        for(int j = 0; j < in_dim; j++) {
            sum += W[i*in_dim + j] * x[j];
        }
        y[i] = sum;
    }
}

// Elementwise multiply of two vectors
void vec_mul(float *out, const float *a, const float *b, int size) {
    for(int i = 0; i < size; i++) {
        out[i] = a[i] * b[i];
    }
}

// Elementwise add
void vec_add(float *out, const float *a, const float *b, int size) {
    for(int i=0; i<size; i++){
        out[i] = a[i] + b[i];
    }
}
```

> **Note**: In the code above, `matvec()` expects a flat array for `W`. But in our struct, `W_ix` is a 2D array. We’ll cast it or flatten it carefully.

---

# 3. LSTM Forward Pass

We implement a **single time-step** of LSTM. The equations are:

\[
\begin{aligned}
\mathbf{i}(t) &= \sigma\bigl(W_{ix} \mathbf{x}(t) + W_{ih} \mathbf{h}(t-1) + \mathbf{b}_i\bigr),\\
\mathbf{f}(t) &= \sigma\bigl(W_{fx} \mathbf{x}(t) + W_{fh} \mathbf{h}(t-1) + \mathbf{b}_f\bigr),\\
\mathbf{o}(t) &= \sigma\bigl(W_{ox} \mathbf{x}(t) + W_{oh} \mathbf{h}(t-1) + \mathbf{b}_o\bigr),\\
\mathbf{\tilde{c}}(t) &= \tanh\bigl(W_{cx} \mathbf{x}(t) + W_{ch} \mathbf{h}(t-1) + \mathbf{b}_c\bigr),\\
\mathbf{c}(t) &= \mathbf{f}(t) \odot \mathbf{c}(t-1) \;+\; \mathbf{i}(t) \odot \mathbf{\tilde{c}}(t),\\
\mathbf{h}(t) &= \mathbf{o}(t) \odot \tanh(\mathbf{c}(t)).
\end{aligned}
\]

```c
// Forward pass for ONE time step
void lstm_step(LSTMParams *params, LSTMState *state, float *x_t) {
    float i_t[HIDDEN_SIZE];
    float f_t[HIDDEN_SIZE];
    float o_t[HIDDEN_SIZE];
    float c_hat_t[HIDDEN_SIZE];

    // Temporary vectors for mat-vec results
    float tmp_i[HIDDEN_SIZE], tmp_f[HIDDEN_SIZE], tmp_o[HIDDEN_SIZE], tmp_c[HIDDEN_SIZE];

    // 1. i(t) = sigmoid(W_ix*x_t + W_ih*h(t-1) + b_i)
    matvec((float*)params->W_ix, x_t, tmp_i, HIDDEN_SIZE, INPUT_SIZE);
    matvec((float*)params->W_ih, state->h, tmp_i, HIDDEN_SIZE, HIDDEN_SIZE);
    add_bias(tmp_i, (float*)params->b_i, HIDDEN_SIZE);
    for(int i=0; i<HIDDEN_SIZE; i++) {
        i_t[i] = sigmoid(tmp_i[i]);
    }

    // 2. f(t) = sigmoid(W_fx*x_t + W_fh*h(t-1) + b_f)
    matvec((float*)params->W_fx, x_t, tmp_f, HIDDEN_SIZE, INPUT_SIZE);
    matvec((float*)params->W_fh, state->h, tmp_f, HIDDEN_SIZE, HIDDEN_SIZE);
    add_bias(tmp_f, (float*)params->b_f, HIDDEN_SIZE);
    for(int i=0; i<HIDDEN_SIZE; i++) {
        f_t[i] = sigmoid(tmp_f[i]);
    }

    // 3. o(t) = sigmoid(W_ox*x_t + W_oh*h(t-1) + b_o)
    matvec((float*)params->W_ox, x_t, tmp_o, HIDDEN_SIZE, INPUT_SIZE);
    matvec((float*)params->W_oh, state->h, tmp_o, HIDDEN_SIZE, HIDDEN_SIZE);
    add_bias(tmp_o, (float*)params->b_o, HIDDEN_SIZE);
    for(int i=0; i<HIDDEN_SIZE; i++) {
        o_t[i] = sigmoid(tmp_o[i]);
    }

    // 4. c_hat(t) = tanh(W_cx*x_t + W_ch*h(t-1) + b_c)
    matvec((float*)params->W_cx, x_t, tmp_c, HIDDEN_SIZE, INPUT_SIZE);
    matvec((float*)params->W_ch, state->h, tmp_c, HIDDEN_SIZE, HIDDEN_SIZE);
    add_bias(tmp_c, (float*)params->b_c, HIDDEN_SIZE);
    for(int i=0; i<HIDDEN_SIZE; i++){
        c_hat_t[i] = tanh_approx(tmp_c[i]);
    }

    // 5. c(t) = f(t)*c(t-1) + i(t)*c_hat(t)
    for(int i=0; i<HIDDEN_SIZE; i++){
        state->c[i] = f_t[i]*state->c[i] + i_t[i]*c_hat_t[i];
    }

    // 6. h(t) = o(t)*tanh(c(t))
    for(int i=0; i<HIDDEN_SIZE; i++){
        state->h[i] = o_t[i] * tanh_approx(state->c[i]);
    }
}
```

### 3.1 LSTM Output Layer

After we get \(\mathbf{h}(t)\), we do a final linear layer:

\[
\hat{y}(t) = W_{hy} \, \mathbf{h}(t) + b_y.
\]

```c
float lstm_output(LSTMParams *params, float *h) {
    // Single output dimension => do matvec of dimension [1 x HIDDEN_SIZE] * [HIDDEN_SIZE]
    float y = 0.0f;
    for(int i=0; i<HIDDEN_SIZE; i++){
        y += params->W_hy[0][i] * h[i];
    }
    y += params->b_y[0];
    return y;
}
```

---

# 4. Basic MSE Loss & Naive Training Loop

For a single sequence \(\{x(1), \dots, x(T)\}\) and targets \(\{y(1), \dots, y(T)\}\), we do:

1. Initialize LSTM state: \(\mathbf{h}(0) = 0, \mathbf{c}(0) = 0\).  
2. For each \(t\) in \(\{1, \dots, T\}\):  
   - Compute one LSTM step using `lstm_step()`.  
   - Get prediction \(\hat{y}(t)\) via `lstm_output()`.  
3. Compute MSE vs. target.  
4. (Naive) Use partial derivatives w.r.t. each weight to do gradient updates.  
   - **Below** we only show a dummy gradient update to illustrate the approach. Doing real BPTT in C is quite involved (we need partial derivatives for each gate, accumulate them across timesteps, etc.).  

```c
float forward_sequence(LSTMParams *params, float *input_seq, float *target_seq, int seq_len) {
    // We'll store the LSTM state
    LSTMState state;
    // Initialize hidden and cell to zero
    for(int i=0; i<HIDDEN_SIZE; i++){ 
        state.h[i] = 0.0f; 
        state.c[i] = 0.0f; 
    }

    float mse = 0.0f;
    // For each timestep in the sequence
    for(int t=0; t<seq_len; t++){
        // input_seq has shape [seq_len * INPUT_SIZE]
        float *x_t = &input_seq[t * INPUT_SIZE];
        float y_true = target_seq[t];

        // 1. LSTM step
        lstm_step(params, &state, x_t);

        // 2. Output
        float y_pred = lstm_output(params, state.h);

        // 3. Accumulate MSE
        float err = (y_true - y_pred);
        mse += err * err;
        
        // 4. Naive gradient step (just as a placeholder)
        //   Real backprop requires partial derivatives w.r.t all gates, etc.
        //   This is just to show how you might do a simplistic update
        float grad = -2.0f * err;  // d/dy_pred of MSE
        for(int i=0; i<HIDDEN_SIZE; i++){
            params->W_hy[0][i] -= LEARNING_RATE * grad * state.h[i];
        }
        params->b_y[0] -= LEARNING_RATE * grad;
    }

    mse /= (float)seq_len;
    return mse;
}
```

> **Important**: The code above is *not* a correct BPTT implementation. We’re only applying a gradient step to the final linear layer for demonstration. Proper training would require storing the internal gate values and partial derivatives at each time step, then backpropagating.  

---

# 5. Example `main()` Function

We can **initialize** the LSTM parameters randomly, create a **dummy** input sequence, a **dummy** target sequence, and run `forward_sequence()`.

```c
int main() {
    srand((unsigned int)time(NULL));

    // 1. Allocate parameters
    LSTMParams params;
    
    // 2. Randomly initialize
    //    For simplicity, small random floats in [-0.1, 0.1]
    float range = 0.1f;
    float scale = (float)RAND_MAX;
    #define RAND_WEIGHT (range * ( (float)rand()/scale*2.0f - 1.0f ))
    
    // Macro to fill an array with random numbers
    #define FILL_MATRIX(mat) do {                     \
        for (int i = 0; i < (int)(sizeof(mat)/sizeof(mat[0][0])); i++){  \
            ((float*)mat)[i] = RAND_WEIGHT;           \
        }                                             \
    } while(0)
    
    FILL_MATRIX(params.W_ix); FILL_MATRIX(params.W_ih); 
    FILL_MATRIX(params.W_fx); FILL_MATRIX(params.W_fh);
    FILL_MATRIX(params.W_ox); FILL_MATRIX(params.W_oh);
    FILL_MATRIX(params.W_cx); FILL_MATRIX(params.W_ch);
    FILL_MATRIX(params.W_hy);
    
    // Bias arrays
    for(int i=0; i<HIDDEN_SIZE; i++){
        params.b_i[i] = RAND_WEIGHT;
        params.b_f[i] = RAND_WEIGHT;
        params.b_o[i] = RAND_WEIGHT;
        params.b_c[i] = RAND_WEIGHT;
    }
    params.b_y[0] = RAND_WEIGHT;

    // 3. Create a dummy input sequence:
    //    Suppose we have TIME_STEPS=5, each with INPUT_SIZE=8
    float input_seq[TIME_STEPS * INPUT_SIZE];
    float target_seq[TIME_STEPS];
    
    for(int t=0; t<TIME_STEPS; t++){
        for(int i=0; i<INPUT_SIZE; i++){
            input_seq[t*INPUT_SIZE + i] = (float)(rand() % 100) / 100.0f; 
            // e.g., random macro/micro/tech features
        }
        // Suppose the "target" price is also random
        target_seq[t] = (float)(rand() % 100) / 10.0f; 
    }

    // 4. A very naive training loop
    int epochs = 1000;
    for(int e=0; e<epochs; e++){
        float mse = forward_sequence(&params, input_seq, target_seq, TIME_STEPS);
        if((e+1) % 100 == 0){
            printf("Epoch %d, MSE = %f\n", e+1, mse);
        }
    }

    // 5. Check final predictions
    {
        LSTMState s;
        for(int i=0; i<HIDDEN_SIZE; i++){
            s.h[i] = 0.0f; s.c[i] = 0.0f;
        }
        printf("\nFinal predictions:\n");
        for(int t=0; t<TIME_STEPS; t++){
            float *x_t = &input_seq[t*INPUT_SIZE];
            lstm_step(&params, &s, x_t);
            float y_pred = lstm_output(&params, s.h);
            printf(" t=%d, target=%.3f, pred=%.3f\n", t, target_seq[t], y_pred);
        }
    }

    return 0;
}
```

**Compile and run** (on a Unix-like system, for instance):

```bash
cc -o lstm_example lstm_example.c -lm
./lstm_example
```

You should see it print MSE decreasing over epochs (though with the naive partial update, it might not do much). Then it prints final predictions vs. targets.

---

# Extending to Your Multi-Factor Stock Model

1. **Input Construction**:  
   - Instead of random data, fill `input_seq` with your real macro/micro/technical variables for each time \(t\). For instance:
     \[
     \mathbf{x}(t) = [\,G(t),\; I(t),\; R(t),\; E(t),\; S(t),\; C(t),\; M(t),\; V(t)\,].
     \]
   - Possibly also include the past price \(P(t-1)\) or log-return \(r(t-1)\) if you want the LSTM to learn from that.

2. **Targets**:  
   - If you want to predict next price \(P(t)\), store that in `target_seq[t]`.  
   - Or if you want to predict the change \(\Delta P(t)\) or log-return, store that. The code is essentially the same; you just feed different targets.

3. **Full BPTT**:  
   - Implement the backward pass for the LSTM, summing gradients from each time step, etc. This is non-trivial but well-documented in many references on building RNNs from scratch.

4. **Longer Sequences & Mini-Batches**:  
   - Increase `TIME_STEPS`.  
   - Use more advanced data handling for multiple sequences (batch size > 1).  

5. **Production Considerations**:  
   - For real forecasting, you’d likely want to shift your target so that \(\hat{P}(t)\) uses data up to \(t-1\).  
   - Do train/test splitting, cross-validation, hyperparameter tuning, etc.

---

## Final Thoughts

This code demonstrates the **core LSTM mechanics** in pure C:

- A **parameter struct** holding weight matrices for each gate.  
- A **forward pass** function that calculates gate activations and updates hidden/cell states.  
- A **rudimentary training loop** (with only partial gradient updates in the example).

To incorporate your **modular stock model** (macro, micro, technical factors), you’d:

1. **Generate or load** those time series for each time \(t\).  
2. **Feed** them into the LSTM as the input vector.  
3. **Train** the LSTM to predict \(\Delta P(t)\) or \(P(t)\).

A fully correct solution requires implementing **backpropagation through time** so that the network learns the best gating and transformations for your data. However, this skeleton shows that **yes, you can indeed** implement an LSTM in C to model a complex, factor-based stock-price time series—**it just takes a fair amount of low-level coding** compared to using high-level libraries (Python/TensorFlow/PyTorch, etc.).
