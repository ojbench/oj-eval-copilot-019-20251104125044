#pragma once
#include "simulator.hpp"
namespace sjtu {

void Calculate(std::vector<Matrix *> keys, std::vector<Matrix *> values,
               Rater &rater, GpuSimulator &gpu_sim,
               MatrixMemoryAllocator matrix_memory_allocator) {
  assert(keys.size() == values.size());
  for (size_t i = 0; i < keys.size(); ++i) {
    Matrix *current_query = rater.GetNextQuery();

    // Ensure operands are in SRAM for compute
    if (current_query->GetPosition() != kInSharedMemory)
      gpu_sim.MoveMatrixToSharedMem(current_query);
    for (size_t j = 0; j <= i; ++j) {
      if (keys[j]->GetPosition() != kInSharedMemory)
        gpu_sim.MoveMatrixToSharedMem(keys[j]);
      if (values[j]->GetPosition() != kInSharedMemory)
        gpu_sim.MoveMatrixToSharedMem(values[j]);
    }

    // Stack keys into K (i+1 x d)
    Matrix *k_stack = matrix_memory_allocator.Allocate("k_stack");
    gpu_sim.Copy(keys[0], k_stack, kInSharedMemory);
    for (size_t j = 1; j <= i; ++j) {
      Matrix *tmp = matrix_memory_allocator.Allocate("k_stack_tmp");
      gpu_sim.Concat(k_stack, keys[j], tmp, /*axis=*/0, kInSharedMemory);
      gpu_sim.ReleaseMatrix(k_stack);
      k_stack = tmp;
    }

    // Stack values into V (i+1 x d)
    Matrix *v_stack = matrix_memory_allocator.Allocate("v_stack");
    gpu_sim.Copy(values[0], v_stack, kInSharedMemory);
    for (size_t j = 1; j <= i; ++j) {
      Matrix *tmp = matrix_memory_allocator.Allocate("v_stack_tmp");
      gpu_sim.Concat(v_stack, values[j], tmp, /*axis=*/0, kInSharedMemory);
      gpu_sim.ReleaseMatrix(v_stack);
      v_stack = tmp;
    }

    // scores = Q * K^T  -> (i+1 x i+1)
    gpu_sim.Transpose(k_stack, kInSharedMemory);
    Matrix *scores = matrix_memory_allocator.Allocate("scores");
    gpu_sim.MatMul(current_query, k_stack, scores);

    // For each row: softmax(row) * V -> output
    Matrix *ans = nullptr;
    for (size_t r = 0; r <= i; ++r) {
      Matrix *row = matrix_memory_allocator.Allocate("row_scores");
      gpu_sim.GetRow(scores, r, row, kInSharedMemory);  // 1 x (i+1)
      Matrix *row_exp = matrix_memory_allocator.Allocate("row_exp");
      gpu_sim.MatExp(row, row_exp);
      Matrix *den = matrix_memory_allocator.Allocate("den");
      gpu_sim.Sum(row_exp, den);                         // 1 x 1
      Matrix *weights = matrix_memory_allocator.Allocate("weights");
      gpu_sim.MatDiv(row_exp, den, weights);             // 1 x (i+1)
      Matrix *row_out = matrix_memory_allocator.Allocate("row_out");
      gpu_sim.MatMul(weights, v_stack, row_out);         // 1 x d

      if (r == 0) {
        ans = matrix_memory_allocator.Allocate("ans");
        gpu_sim.Copy(row_out, ans, kInSharedMemory);
      } else {
        Matrix *tmp_ans = matrix_memory_allocator.Allocate("ans_tmp");
        gpu_sim.Concat(ans, row_out, tmp_ans, /*axis=*/0, kInSharedMemory);
        gpu_sim.ReleaseMatrix(ans);
        ans = tmp_ans;
      }
      gpu_sim.ReleaseMatrix(row);
      gpu_sim.ReleaseMatrix(row_exp);
      gpu_sim.ReleaseMatrix(den);
      gpu_sim.ReleaseMatrix(weights);
      gpu_sim.ReleaseMatrix(row_out);
    }

    // Cleanup
    gpu_sim.ReleaseMatrix(k_stack);
    gpu_sim.ReleaseMatrix(v_stack);
    gpu_sim.ReleaseMatrix(scores);

    // Move answer to HBM, run, then commit
    gpu_sim.MoveMatrixToGpuHbm(ans);
    gpu_sim.Run(false, &matrix_memory_allocator);
    rater.CommitAnswer(*ans);
  }
}

void Test(Rater &rater, GpuSimulator &gpu_sim,
          MatrixMemoryAllocator &matrix_memory_allocator) {
  Calculate(rater.keys_, rater.values_, rater, gpu_sim,
            matrix_memory_allocator);
  rater.PrintResult(gpu_sim);
}

} // namespace sjtu