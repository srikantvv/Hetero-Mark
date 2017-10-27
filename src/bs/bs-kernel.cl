float Phi(float X) {
  float y, absX, t;

  // the coefficients
  const float c1 = 0.319381530f;
  const float c2 = -0.356563782f;
  const float c3 = 1.781477937f;
  const float c4 = -1.821255978f;
  const float c5 = 1.330274429f;

  const float oneBySqrt2pi = 0.398942280f;

  absX = fabs(X);
  t = 1.0f / (1.0f + 0.2316419f * absX);

  y = 1.0f -
      oneBySqrt2pi * exp(-X * X / 2.0f) * t *
          (c1 + t * (c2 + t * (c3 + t * (c4 + t * c5))));

  return (X < 0) ? (1.0f - y) : y;
}

__kernel void blackscholes(__global float *rand_array, __global float *d_call_price_,
                        __global float *d_put_price_,
                        __global volatile int *locPtr,
                        unsigned long sigAddr) {
  uint tid = get_global_id(0);

  // the variable representing the value in the array[i]
  float i_rand = rand_array[tid];

  // calculating the initial S,K,T, and R
  float s = 10.0 * i_rand + 100.0 * (1.0f - i_rand);
  float k = 10.0 * i_rand + 100.0 * (1.0f - i_rand);
  float t = 1.0 * i_rand + 10.0 * (1.0f - i_rand);
  float r = 0.01 * i_rand + 0.05 * (1.0f - i_rand);
  float sigma = 0.01 * i_rand + 0.10 * (1.0f - i_rand);

  // Calculating the sigmaSqrtT
  float sigma_sqrt_t_ = sigma * sqrt(t);

  // Calculating the derivatives
  float d1 = (log(s / k) + (r + sigma * sigma / 2.0f) * t) / sigma_sqrt_t_;
  float d2 = d1 - sigma_sqrt_t_;

  // Calculating exponent
  float k_exp_minus_rt_ = k * exp(-r * t);

  // Getting the output call and put prices
  d_call_price_[tid] = s * Phi(d1) - k_exp_minus_rt_ * Phi(d2);
  d_put_price_[tid] = k_exp_minus_rt_ * Phi(-d2) - s * Phi(-d1);

  int gsize = get_global_size(0);
  atomic_fetch_add((atomic_int *) locPtr, 1);
  if (tid == (gsize - 1)) {
      while(1) {
          if(*locPtr == gsize) {
              break;
          }
      }
      __global unsigned long long *sigFinish = (__global unsigned long long *)sigAddr;
      *sigFinish = 1;
  }
}

