#pragma once
#include "simulator.hpp"
namespace sjtu {

void Calculate(std::vector<Matrix *> keys, std::vector<Matrix *> values,
               Rater &rater, GpuSimulator &gpu_sim,
               MatrixMemoryAllocator matrix_memory_allocator) {
  assert(keys.size() == values.size());
  // Maintain growing stacks across rounds to avoid rebuilding
  Matrix *k_stack = nullptr;
  Matrix *v_stack = nullptr;
  for (size_t i = 0; i < keys.size(); ++i) {
    Matrix *current_query = rater.GetNextQuery();

    // Ensure operands are in SRAM for compute
    if (current_query->GetPosition() != kInSharedMemory)
      gpu_sim.MoveMatrixToSharedMem(current_query);
    if (keys[i]->GetPosition() != kInSharedMemory)
      gpu_sim.MoveMatrixToSharedMem(keys[i]);
    if (values[i]->GetPosition() != kInSharedMemory)
      gpu_sim.MoveMatrixToSharedMem(values[i]);

    // Grow K and V stacks incrementally (each is (i+1) x d)
    if (i == 0) {
      k_stack = matrix_memory_allocator.Allocate("k_stack");
      gpu_sim.Copy(keys[0], k_stack, kInSharedMemory);
      v_stack = matrix_memory_allocator.Allocate("v_stack");
      gpu_sim.Copy(values[0], v_stack, kInSharedMemory);
    } else {
      Matrix *tmp_k = matrix_memory_allocator.Allocate("k_stack_tmp");
      gpu_sim.Concat(k_stack, keys[i], tmp_k, /*axis=*/0, kInSharedMemory);
      gpu_sim.ReleaseMatrix(k_stack);
      k_stack = tmp_k;

      Matrix *tmp_v = matrix_memory_allocator.Allocate("v_stack_tmp");
      gpu_sim.Concat(v_stack, values[i], tmp_v, /*axis=*/0, kInSharedMemory);
      gpu_sim.ReleaseMatrix(v_stack);
      v_stack = tmp_v;
    }

    // scores = Q * K^T  -> (i+1 x i+1)
    Matrix *k_t = matrix_memory_allocator.Allocate("k_t");
    gpu_sim.Copy(k_stack, k_t, kInSharedMemory);
    gpu_sim.Transpose(k_t, kInSharedMemory);
    Matrix *scores = matrix_memory_allocator.Allocate("scores");
    gpu_sim.MatMul(current_query, k_t, scores);

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

    // Cleanup temps of this round
    gpu_sim.ReleaseMatrix(k_t);
    gpu_sim.ReleaseMatrix(scores);

    // Move answer to HBM, run, then commit
    gpu_sim.MoveMatrixToGpuHbm(ans);
    gpu_sim.Run(false, &matrix_memory_allocator);
    rater.CommitAnswer(*ans);
  }
  // Release persistent stacks
  if (k_stack) gpu_sim.ReleaseMatrix(k_stack);
  if (v_stack) gpu_sim.ReleaseMatrix(v_stack);
}

void Test(Rater &rater, GpuSimulator &gpu_sim,
          MatrixMemoryAllocator &matrix_memory_allocator) {
  Calculate(rater.keys_, rater.values_, rater, gpu_sim,
            matrix_memory_allocator);
  rater.PrintResult(gpu_sim);
}

} // namespace sjtu