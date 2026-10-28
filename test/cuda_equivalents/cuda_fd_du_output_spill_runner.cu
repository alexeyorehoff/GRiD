#include <algorithm>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>

#include "grid.cuh"

// Exercise the codegen-selected PERF rung, including on floating robots. The
// general equivalence runner deliberately uses MINIMAL for floating gradients,
// which would compare the output-spill rung to itself in this test.
template <typename T>
using gradient_kernel_t = void (*)(T *, unsigned char *, const T *, int, T *,
                                  const grid::robotModel<T> *, T, int);

template <typename T>
int run(int threads) {
    cudaStream_t *streams = grid::init_grid_streams<T>();
    gpuErrchk(grid::init_grid_kernel_attr_forward_dynamics_gradient_checked<T>());
    auto *model = grid::init_robotModel<T>();
    auto *data = grid::init_gridData<T, 1>();

    // The kernel uses three nq-wide slots, with nv values in the last two.
    std::fill(data->h_q_qd_u, data->h_q_qd_u + 3 * grid::NUM_JOINTS, T(0));
    for (int i = 0; i < grid::NUM_JOINTS; ++i) std::cin >> data->h_q_qd_u[i];
    for (int slot = 1; slot < 3; ++slot)
        for (int i = 0; i < grid::NUM_VEL; ++i)
            std::cin >> data->h_q_qd_u[slot * grid::NUM_JOINTS + i];
    if (!std::cin) return 1;
    gpuErrchk(cudaMemcpy(data->d_q_qd_u, data->h_q_qd_u,
                        3 * grid::NUM_JOINTS * sizeof(T), cudaMemcpyHostToDevice));

    threads = std::min(threads, grid::MAX_PERF_LEVEL_THREADS);
    const size_t bytes = grid::FORWARD_DYNAMICS_GRADIENT_DYNAMIC_SHARED_MEM_BYTES<T>();
    std::cerr << "fd_du threads=" << threads << " shared_bytes=" << bytes << '\n';
    grid::forward_dynamics_gradient_kernel<T><<<1, threads, bytes>>>(
        data->d_df_du, data->d_workspace, data->d_q_qd_u,
        3 * grid::NUM_JOINTS, nullptr, model, T(-9.81), 1);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    gpuErrchk(cudaMemcpy(data->h_df_du, data->d_df_du,
                        2 * grid::NUM_VEL * grid::NUM_VEL * sizeof(T), cudaMemcpyDeviceToHost));

    const char *names[] = {"forward_dynamics_gradient_q", "forward_dynamics_gradient_qd"};
    std::cout << std::setprecision(10);
    for (int band = 0; band < 2; ++band) {
        std::cout << "BEGIN " << names[band] << ' ' << grid::NUM_VEL
                  << ' ' << grid::NUM_VEL << '\n';
        for (int i = 0; i < grid::NUM_VEL * grid::NUM_VEL; ++i)
            std::cout << data->h_df_du[band * grid::NUM_VEL * grid::NUM_VEL + i] << ' ';
        std::cout << "\nEND " << names[band] << '\n';
    }
    grid::close_grid<T>(streams, model, data);
    return 0;
}

int main(int argc, char **argv) {
    if (argc > 1 && std::string(argv[1]) == "--shared-memory-capacity") {
        size_t capacity = 0;
        gpuErrchk(grid::grid_get_max_dynamic_shared_memory_bytes(&capacity));
        cudaFuncAttributes attrs;
        gpuErrchk(cudaFuncGetAttributes(&attrs, static_cast<gradient_kernel_t<float>>(
            &grid::forward_dynamics_gradient_kernel<float>)));
        std::cout << capacity - attrs.sharedSizeBytes << '\n';
        return 0;
    }
    return run<float>(argc > 1 ? std::atoi(argv[1]) : 32);
}
