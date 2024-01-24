/* Copyright (c) by CryptoLab Inc.
 * This library is licensed under a
 * Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.
 * You should have received a copy of the license along with this
 * work. If not, see <http://creativecommons.org/licenses/by-nc-sa/4.0/>.
 */
#include <iomanip>

#include "public/Test.h"

using namespace ckks;
using namespace std;

class Timer {
 public:
  Timer(const string& name) : name{name} {
    cudaDeviceSynchronize();
    CudaNvtxStart(name);
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
  }

  ~Timer() {
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    CudaNvtxStop();
    cout << setprecision(3);
    cout << name << ", " << fixed << setprecision(3) << milliseconds << " ms"
         << endl;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
  }

  string name;
  cudaEvent_t start, stop;
};

void ModUpBench() {}

class Benchmark {
 public:
  Benchmark(const Parameter& param) : ckks{param}, param{param} {
    // std::cout<<primes.size()std::endl;
    ckks.context.EnableMemoryPool();
    ModUpBench();
    ModDownBench();
    // warmup();
    ADDBench();
    PmultBench();
    RotateBench();
    KeyswitchBench();
    PtxtCtxtBatchBench();
    
  }

  template <typename F, typename R, class... Args>
  void Run(const string& message, R(F::*mf), Args&&... args) {
    for (int i = 0; i < iters; i++) {
      Timer marker(message);
      (ckks.context.*mf)(std::forward<Args>(args)...);
    }
  }

  template <typename Callable, class... Args>
  void Run(const string& message, Callable C, Args&&... args) {
    for (int i = 0; i < iters; i++) {
      Timer marker(message);
      C(std::forward<Args>(args)...);
    }
  }

    void RotateBench() {
      auto key = ckks.GetRandomKey();
      int sn=1;
      Ciphertext in=ckks.GetRandomCiphertext();;
      Ciphertext out;
      Run("Rotate", &Context::rotate,in,key,1,out);
    }

  void ModUpBench() {
    auto from = ckks.GetRandomPoly();
    ckks.context.is_modup_batched = false;
    Run("ModUp", &Context::ModUp, from);
    ckks.context.is_modup_batched = true;
    Run("FusedModUp", &Context::ModUp, from);
  }

  void ModDownBench() {
    const int num_moduli_after_moddown = param.chain_length_;  // PQ -> Q
    auto from = ckks.GetRandomPolyRNS(param.max_num_moduli_);
    DeviceVector to;
    ckks.context.is_moddown_fused = false;
    Run("ModDown", &Context::ModDown, from, to, num_moduli_after_moddown);
    ckks.context.is_moddown_fused = true;
    Run("FusedModDown", &Context::ModDown, from, to, num_moduli_after_moddown);
  }
//从一个密钥域转换到另一个密钥域
  void KeyswitchBench() {
    auto key = ckks.GetRandomKey();
    auto in = ckks.GetRandomPolyAfterModUp(param.dnum_);  // beta = dnum case
    DeviceVector ax, bx;
    ckks.context.is_keyswitch_fused = false;
    Run("KeySwitch", &Context::KeySwitch, in, key, ax, bx);
    // ckks.context.is_keyswitch_fused = true;
    // Run("FusedKeySwitch", &Context::KeySwitch, in, key, ax, bx);
  }
  void warmup() {
    Ciphertext op1,op2;
    op1=ckks.GetRandomCiphertext();
    op2=ckks.GetRandomCiphertext();
    auto ADD = [&](const auto& op1, const auto& op2) {
      Ciphertext accum;
      ckks.context.Add(op2, op1, accum);
    };

    Run("warmup", ADD, op1, op2);
  }
  void ADDBench() {
    Ciphertext op1,op2;
    op1=ckks.GetRandomCiphertext();
    op2=ckks.GetRandomCiphertext();
    auto ADD = [&](const auto& op1, const auto& op2) {
      Ciphertext accum;
      ckks.context.Add(op2, op1, accum);
    };

    Run("ADD", ADD, op1, op2);
  }
  void PmultBench() {
    int batch_size = 1;
    Ciphertext op1;
    Plaintext op2;
    op1=ckks.GetRandomCiphertext();
    op2=ckks.GetRandomPlaintext();
    auto PMult = [&](const auto& op1, const auto& op2) {
      Ciphertext accum;
      ckks.context.PMult(op1, op2, accum);
    };

    Run("PMult", PMult, op1, op2);
  }

  void PtxtCtxtBatchBench() {
    int batch_size = 1;
    vector<Ciphertext> op1(batch_size);
    vector<Plaintext> op2(batch_size);
    // setup
    for (int i = 0; i < batch_size; i++) {
      op1[i] = ckks.GetRandomCiphertext();
      op2[i] = ckks.GetRandomPlaintext();
    }
    auto MAD = [&](const auto& op1, const auto& op2) {
      Ciphertext accum, out;
      ckks.context.PMult(op1[0], op2[0], accum);
      ckks.context.Add(accum, op1[0], accum);
      // for (int i = 1; i < batch_size; i++) {
      //   ckks.context.PMult(op1[i], op2[i], out);
      //   ckks.context.Add(accum, out, accum);
      // }
    };
    auto BatchMAD = [&](const auto& op1, const auto& op2) {
      MultPtxtBatch batcher(&ckks.context);
      Ciphertext accum;
      for (int i = 0; i < batch_size; i++) {
        batcher.push(op1[i], op2[i]);
      }
      batcher.flush(accum);
    };
    Run("PtxtCtxtMAD", MAD, op1, op2);
    // Run("BatchedPtxtCtxtMAD", BatchMAD, op1, op2);
  }

 private:
  Test ckks;
  Parameter param;
  int iters = 1;
};

int main() {
  std::cout<<"N=10: "<<std::endl;
  Benchmark bench0(PARAM_SMALL_DNUM10);
  std::cout<<std::endl;
  std::cout<<"N=11: "<<std::endl;
  Benchmark bench1(PARAM_SMALL_DNUM11);
  std::cout<<std::endl;
  std::cout<<"N=12: "<<std::endl;
  Benchmark bench2(PARAM_SMALL_DNUM12);
  std::cout<<std::endl;
  std::cout<<"N=13: "<<std::endl;
  Benchmark bench3(PARAM_SMALL_DNUM13);
  std::cout<<std::endl;
  std::cout<<"N=14: "<<std::endl;
  Benchmark bench4(PARAM_SMALL_DNUM14);
  std::cout<<std::endl;
  std::cout<<"N=15: "<<std::endl;
  Benchmark bench5(PARAM_SMALL_DNUM15);
  std::cout<<std::endl;
  std::cout<<"N=16: "<<std::endl;
  Benchmark bench6(PARAM_SMALL_DNUM16);
  std::cout<<std::endl;
  std::cout<<"N=17: "<<std::endl;
  Benchmark bench7(PARAM_SMALL_DNUM17);
  std::cout<<std::endl;

  // std::cout<<"level=44: "<<std::endl;
  // Benchmark bench0(PARAM_44);
  // std::cout<<std::endl;
  // std::cout<<"level=143: "<<std::endl;
  // Benchmark bench1(PARAM_143);
  // std::cout<<std::endl;
  // std::cout<<"level=242: "<<std::endl;
  // Benchmark bench2(PARAM_242);
  // std::cout<<std::endl;
  // std::cout<<"level=341: "<<std::endl;
  // Benchmark bench3(PARAM_341);
  // std::cout<<std::endl;
  // std::cout<<"level=440: "<<std::endl;
  // Benchmark bench4(PARAM_440);
  // std::cout<<std::endl;
  // std::cout<<"level=539: "<<std::endl;
  // Benchmark bench5(PARAM_539);
  // std::cout<<std::endl;
  // std::cout<<"level=638: "<<std::endl;
  // Benchmark bench6(PARAM_638);
  // std::cout<<std::endl;
  // std::cout<<"level=737: "<<std::endl;
  // Benchmark bench7(PARAM_737);
  // std::cout<<"level=836: "<<std::endl;
  // Benchmark bench8(PARAM_836);
  // std::cout<<"level=935: "<<std::endl;
  // Benchmark bench9(PARAM_935);
  return 0;
}