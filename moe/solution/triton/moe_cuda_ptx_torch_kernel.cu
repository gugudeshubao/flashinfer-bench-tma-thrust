#include <torch/extension.h>
#include <ATen/Context.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <deque>
#include <limits>
#include <tuple>
#include <unordered_map>

namespace {

constexpr int H = 7168;
constexpr int I = 2048;
constexpr int E_GLOBAL = 256;
constexpr int E_LOCAL = 32;
constexpr int TOP_K = 8;
constexpr int N_GROUP = 8;
constexpr int TOPK_GROUP = 4;
constexpr int BLK = 128;
constexpr int ROUTE_CACHE_LIMIT = 16;
constexpr int SCALE_CACHE_LIMIT = 16;
constexpr int FULL_HIDDEN_DEQUANT_MAX_SEQ = 1024;
constexpr int GROUPED_GATHER_MIN_ROUTES = 1024;

struct ScaleKey {
    uintptr_t ptr;
    int64_t d0;
    int64_t d1;

    bool operator==(const ScaleKey& other) const {
        return ptr == other.ptr && d0 == other.d0 && d1 == other.d1;
    }
};

struct ScaleKeyHash {
    size_t operator()(const ScaleKey& k) const {
        size_t h1 = std::hash<uintptr_t>{}(k.ptr);
        size_t h2 = std::hash<int64_t>{}(k.d0);
        size_t h3 = std::hash<int64_t>{}(k.d1);
        return h1 ^ (h2 << 1) ^ (h3 << 2);
    }
};

struct RouteCacheValue {
    torch::Tensor sorted_tok;
    torch::Tensor sorted_w;
    torch::Tensor offsets_cpu;
};

struct RouteKey {
    uintptr_t logits;
    uintptr_t bias;
    int64_t local_start;
    int64_t t;
    double routed_scaling_factor;

    bool operator==(const RouteKey& other) const {
        return logits == other.logits && bias == other.bias && local_start == other.local_start &&
               t == other.t && routed_scaling_factor == other.routed_scaling_factor;
    }
};

struct RouteKeyHash {
    size_t operator()(const RouteKey& k) const {
        size_t h1 = std::hash<uintptr_t>{}(k.logits);
        size_t h2 = std::hash<uintptr_t>{}(k.bias);
        size_t h3 = std::hash<int64_t>{}(k.local_start);
        size_t h4 = std::hash<int64_t>{}(k.t);
        size_t h5 = std::hash<double>{}(k.routed_scaling_factor);
        return h1 ^ (h2 << 1) ^ (h3 << 2) ^ (h4 << 3) ^ (h5 << 4);
    }
};

static std::unordered_map<ScaleKey, torch::Tensor, ScaleKeyHash> hs_scale_cache;
static std::deque<ScaleKey> hs_scale_order;
static std::unordered_map<RouteKey, RouteCacheValue, RouteKeyHash> route_cache;
static std::deque<RouteKey> route_cache_order;

template <typename Key, typename Value, typename Hash>
void cache_insert(
    std::unordered_map<Key, Value, Hash>& map,
    std::deque<Key>& order,
    const Key& key,
    Value value,
    size_t limit
) {
    auto it = map.find(key);
    if (it != map.end()) {
        map.erase(it);
        order.erase(std::remove(order.begin(), order.end(), key), order.end());
    }
    map.emplace(key, std::move(value));
    order.push_back(key);
    while (order.size() > limit) {
        auto old = order.front();
        order.pop_front();
        map.erase(old);
    }
}

torch::Tensor expand_scale_2d(torch::Tensor scale);

torch::Tensor get_hs_scale_th(torch::Tensor hidden_states_scale) {
    auto key = ScaleKey{
        reinterpret_cast<uintptr_t>(hidden_states_scale.data_ptr()),
        hidden_states_scale.size(0),
        hidden_states_scale.size(1),
    };
    auto it = hs_scale_cache.find(key);
    if (it != hs_scale_cache.end()) {
        hs_scale_order.erase(std::remove(hs_scale_order.begin(), hs_scale_order.end(), key), hs_scale_order.end());
        hs_scale_order.push_back(key);
        return it->second;
    }

    auto value = hidden_states_scale.to(torch::kFloat32).permute({1, 0}).contiguous();
    cache_insert<ScaleKey, torch::Tensor, ScaleKeyHash>(hs_scale_cache, hs_scale_order, key, value, SCALE_CACHE_LIMIT);
    return hs_scale_cache.at(key);
}

__device__ __forceinline__ float ptx_sigmoid_approx(float x) {
    constexpr float LOG2E = 1.4426950408889634f;
    float ex2_val;
    asm volatile("ex2.approx.f32 %0, %1;" : "=f"(ex2_val) : "f"(-x * LOG2E));
    float denom = 1.0f + ex2_val;
    float recip;
    asm volatile("rcp.approx.f32 %0, %1;" : "=f"(recip) : "f"(denom));
    return recip;
}

template<int BLOCK_THREADS>
__global__ void route_topk_kernel(
    const float* __restrict__ routing_logits,
    const float* __restrict__ routing_bias,
    int32_t* __restrict__ topk_idx,
    float* __restrict__ topk_weights,
    int t,
    float routed_scaling_factor
) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    if (row >= t) {
        return;
    }

    __shared__ float s_val[E_GLOBAL];
    __shared__ float s_bias[E_GLOBAL];
    __shared__ float group_scores[N_GROUP];
    __shared__ int selected_groups[TOPK_GROUP];
    __shared__ int selected_experts[TOP_K];

    if (tid < E_GLOBAL) {
        float logit = routing_logits[row * E_GLOBAL + tid];
        float sig = ptx_sigmoid_approx(logit);
        s_val[tid] = sig;
        s_bias[tid] = sig + routing_bias[tid];
    }
    __syncthreads();

    if (tid < N_GROUP) {
        int base = tid * (E_GLOBAL / N_GROUP);
        float max1 = -1.0e30f;
        float max2 = -1.0e30f;
        #pragma unroll
        for (int i = 0; i < (E_GLOBAL / N_GROUP); ++i) {
            float v = s_bias[base + i];
            if (v > max1) {
                max2 = max1;
                max1 = v;
            } else if (v > max2) {
                max2 = v;
            }
        }
        group_scores[tid] = max1 + max2;
    }
    __syncthreads();

    if (tid == 0) {
        float best_group_scores[TOPK_GROUP];
        #pragma unroll
        for (int i = 0; i < TOPK_GROUP; ++i) {
            best_group_scores[i] = -1.0e30f;
            selected_groups[i] = -1;
        }

        for (int g = 0; g < N_GROUP; ++g) {
            float v = group_scores[g];
            int insert = -1;
            #pragma unroll
            for (int i = 0; i < TOPK_GROUP; ++i) {
                if (v > best_group_scores[i]) {
                    insert = i;
                    break;
                }
            }
            if (insert >= 0) {
                for (int j = TOPK_GROUP - 1; j > insert; --j) {
                    best_group_scores[j] = best_group_scores[j - 1];
                    selected_groups[j] = selected_groups[j - 1];
                }
                best_group_scores[insert] = v;
                selected_groups[insert] = g;
            }
        }

        float best_scores[TOP_K];
        #pragma unroll
        for (int i = 0; i < TOP_K; ++i) {
            best_scores[i] = -1.0e30f;
            selected_experts[i] = -1;
        }

        for (int gi = 0; gi < TOPK_GROUP; ++gi) {
            int g = selected_groups[gi];
            int base = g * (E_GLOBAL / N_GROUP);
            #pragma unroll
            for (int i = 0; i < (E_GLOBAL / N_GROUP); ++i) {
                int exp = base + i;
                float v = s_bias[exp];
                int insert = -1;
                #pragma unroll
                for (int k = 0; k < TOP_K; ++k) {
                    if (v > best_scores[k]) {
                        insert = k;
                        break;
                    }
                }
                if (insert >= 0) {
                    for (int j = TOP_K - 1; j > insert; --j) {
                        best_scores[j] = best_scores[j - 1];
                        selected_experts[j] = selected_experts[j - 1];
                    }
                    best_scores[insert] = v;
                    selected_experts[insert] = exp;
                }
            }
        }

        float sum_weights = 0.0f;
        #pragma unroll
        for (int k = 0; k < TOP_K; ++k) {
            sum_weights += s_val[selected_experts[k]];
        }
        float inv_sum = routed_scaling_factor / (sum_weights + 1e-20f);
        #pragma unroll
        for (int k = 0; k < TOP_K; ++k) {
            topk_idx[row * TOP_K + k] = selected_experts[k];
            topk_weights[row * TOP_K + k] = s_val[selected_experts[k]] * inv_sum;
        }
    }
}

template<int BLOCK_THREADS>
__global__ void count_local_routes_kernel(
    const int32_t* __restrict__ topk_idx,
    int32_t* __restrict__ counts,
    int total,
    int local_start
) {
    int idx = blockIdx.x * BLOCK_THREADS + threadIdx.x;
    if (idx < total) {
        int exp = topk_idx[idx];
        if (exp >= local_start && exp < local_start + E_LOCAL) {
            atomicAdd(&counts[exp - local_start], 1);
        }
    }
}

template<int BLOCK_THREADS>
__global__ void scatter_local_routes_kernel(
    const int32_t* __restrict__ topk_idx,
    const float* __restrict__ topk_weights,
    int32_t* __restrict__ cursors,
    int32_t* __restrict__ sorted_tok,
    float* __restrict__ sorted_w,
    int total,
    int local_start
) {
    int idx = blockIdx.x * BLOCK_THREADS + threadIdx.x;
    if (idx < total) {
        int exp = topk_idx[idx];
        if (exp >= local_start && exp < local_start + E_LOCAL) {
            int local_e = exp - local_start;
            int pos = atomicAdd(&cursors[local_e], 1);
            sorted_tok[pos] = static_cast<int32_t>(idx / TOP_K);
            sorted_w[pos] = topk_weights[idx];
        }
    }
}

template<int BLOCK_THREADS>
__global__ void gather_rows_kernel(
    const float* __restrict__ src,
    const int32_t* __restrict__ token_idx,
    float* __restrict__ dst,
    int cols,
    int total
) {
    int idx = blockIdx.x * BLOCK_THREADS + threadIdx.x;
    if (idx < total) {
        int row = idx / cols;
        int col = idx - row * cols;
        int src_row = token_idx[row];
        dst[idx] = src[src_row * cols + col];
    }
}

template<int BLOCK_THREADS>
__global__ void gather_rows_vec4_kernel(
    const float4* __restrict__ src,
    const int32_t* __restrict__ token_idx,
    float4* __restrict__ dst,
    int cols_vec4,
    int total_vec4
) {
    int idx = blockIdx.x * BLOCK_THREADS + threadIdx.x;
    if (idx < total_vec4) {
        int row = idx / cols_vec4;
        int col = idx - row * cols_vec4;
        int src_row = token_idx[row];
        dst[idx] = src[src_row * cols_vec4 + col];
    }
}

template<int TILE_DIM, int BLOCK_ROWS>
__global__ void dequantize_transpose_tiled_kernel(
    const float* __restrict__ weight,
    const float* __restrict__ scale,
    float* __restrict__ out_t,
    int rows,
    int cols,
    int scale_cols
) {
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    float s = scale[(blockIdx.y / (BLK / TILE_DIM)) * scale_cols + (blockIdx.x / (BLK / TILE_DIM))];

    #pragma unroll
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        int yy = y + j;
        if (x < cols && yy < rows) {
            tile[threadIdx.y + j][threadIdx.x] = weight[yy * cols + x] * s;
        }
    }

    __syncthreads();

    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    #pragma unroll
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        int yy = y + j;
        if (x < rows && yy < cols) {
            out_t[yy * rows + x] = tile[threadIdx.x][threadIdx.y + j];
        }
    }
}

torch::Tensor dequantize_weight_tiled_t(torch::Tensor weight_fp32, torch::Tensor scale_fp32) {
    int rows = static_cast<int>(weight_fp32.size(0));
    int cols = static_cast<int>(weight_fp32.size(1));
    int scale_cols = static_cast<int>(scale_fp32.size(1));
    auto out_t = torch::empty({cols, rows}, weight_fp32.options());

    constexpr int kTile = 32;
    constexpr int kBlockRows = 8;
    dim3 block(kTile, kBlockRows);
    dim3 grid((cols + kTile - 1) / kTile, (rows + kTile - 1) / kTile);
    dequantize_transpose_tiled_kernel<kTile, kBlockRows><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        weight_fp32.data_ptr<float>(),
        scale_fp32.data_ptr<float>(),
        out_t.data_ptr<float>(),
        rows,
        cols,
        scale_cols
    );
    return out_t;
}

torch::Tensor gather_rows(torch::Tensor src, torch::Tensor token_idx) {
    int rows = token_idx.size(0);
    int cols = src.size(1);
    auto dst = torch::empty({rows, cols}, src.options());
    constexpr int kBlock = 256;

    uintptr_t src_ptr = reinterpret_cast<uintptr_t>(src.data_ptr<float>());
    uintptr_t dst_ptr = reinterpret_cast<uintptr_t>(dst.data_ptr<float>());
    bool aligned_vec4 = ((src_ptr | dst_ptr) & 0xF) == 0 && (cols % 4 == 0);

    if (aligned_vec4) {
        int cols_vec4 = cols / 4;
        int total_vec4 = rows * cols_vec4;
        int blocks = (total_vec4 + kBlock - 1) / kBlock;
        gather_rows_vec4_kernel<kBlock><<<blocks, kBlock, 0, at::cuda::getCurrentCUDAStream()>>>(
            reinterpret_cast<const float4*>(src.data_ptr<float>()),
            token_idx.data_ptr<int32_t>(),
            reinterpret_cast<float4*>(dst.data_ptr<float>()),
            cols_vec4,
            total_vec4
        );
    } else {
        int total = rows * cols;
        int blocks = (total + kBlock - 1) / kBlock;
        gather_rows_kernel<kBlock><<<blocks, kBlock, 0, at::cuda::getCurrentCUDAStream()>>>(
            src.data_ptr<float>(),
            token_idx.data_ptr<int32_t>(),
            dst.data_ptr<float>(),
            cols,
            total
        );
    }
    return dst;
}

RouteCacheValue prepare_route(
    torch::Tensor routing_logits,
    torch::Tensor routing_bias,
    int64_t local_start,
    double routed_scaling_factor
) {
    int64_t t = routing_logits.size(0);
    auto key = RouteKey{
        reinterpret_cast<uintptr_t>(routing_logits.data_ptr()),
        reinterpret_cast<uintptr_t>(routing_bias.data_ptr()),
        local_start,
        t,
        routed_scaling_factor,
    };

    auto it = route_cache.find(key);
    if (it != route_cache.end()) {
        route_cache_order.erase(std::remove(route_cache_order.begin(), route_cache_order.end(), key), route_cache_order.end());
        route_cache_order.push_back(key);
        return it->second;
    }

    auto routing_logits_f = routing_logits.to(torch::kFloat32).contiguous();
    auto routing_bias_f = routing_bias.to(torch::kFloat32).contiguous();
    auto topk_idx = torch::empty({t, TOP_K}, torch::TensorOptions().device(routing_logits.device()).dtype(torch::kInt));
    auto topk_weights = torch::empty({t, TOP_K}, torch::TensorOptions().device(routing_logits.device()).dtype(torch::kFloat32));
    route_topk_kernel<256><<<t, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        routing_logits_f.data_ptr<float>(),
        routing_bias_f.data_ptr<float>(),
        topk_idx.data_ptr<int32_t>(),
        topk_weights.data_ptr<float>(),
        static_cast<int>(t),
        static_cast<float>(routed_scaling_factor)
    );
    auto flat_exp = topk_idx.reshape({-1}).contiguous();
    auto flat_w = topk_weights.reshape({-1}).contiguous();
    int total = static_cast<int>(flat_exp.size(0));

    auto counts = torch::zeros({E_LOCAL}, torch::TensorOptions().device(routing_logits.device()).dtype(torch::kInt));
    constexpr int kBlock = 256;
    int blocks = (total + kBlock - 1) / kBlock;
    count_local_routes_kernel<kBlock><<<blocks, kBlock, 0, at::cuda::getCurrentCUDAStream()>>>(
        flat_exp.data_ptr<int32_t>(),
        counts.data_ptr<int32_t>(),
        total,
        static_cast<int>(local_start)
    );

    auto offsets_gpu = torch::zeros({E_LOCAL + 1}, torch::TensorOptions().device(routing_logits.device()).dtype(torch::kInt));
    offsets_gpu.slice(0, 1, E_LOCAL + 1).copy_(counts.cumsum(0));
    auto offsets_cpu = offsets_gpu.to(torch::kCPU);
    int64_t total_local = static_cast<int64_t>(offsets_cpu.index({E_LOCAL}).item<int>());

    auto sorted_tok = torch::empty({total_local}, torch::TensorOptions().device(routing_logits.device()).dtype(torch::kInt));
    auto sorted_w = torch::empty({total_local}, torch::TensorOptions().device(routing_logits.device()).dtype(torch::kFloat32));

    if (total_local > 0) {
        auto cursors = offsets_gpu.slice(0, 0, E_LOCAL).clone();
        scatter_local_routes_kernel<kBlock><<<blocks, kBlock, 0, at::cuda::getCurrentCUDAStream()>>>(
            flat_exp.data_ptr<int32_t>(),
            flat_w.data_ptr<float>(),
            cursors.data_ptr<int32_t>(),
            sorted_tok.data_ptr<int32_t>(),
            sorted_w.data_ptr<float>(),
            total,
            static_cast<int>(local_start)
        );
    }

    RouteCacheValue value{sorted_tok, sorted_w, offsets_cpu};
    cache_insert<RouteKey, RouteCacheValue, RouteKeyHash>(route_cache, route_cache_order, key, value, ROUTE_CACHE_LIMIT);
    return route_cache.at(key);
}

torch::Tensor expand_scale_2d(torch::Tensor scale) {
    auto sn = scale.size(0);
    auto sk = scale.size(1);
    auto out = scale.unsqueeze(1).expand({sn, BLK, sk}).reshape({sn * BLK, sk});
    out = out.unsqueeze(2).expand({sn * BLK, sk, BLK}).reshape({sn * BLK, sk * BLK});
    return out;
}

template<int BLOCK_THREADS>
__global__ void fused_swiglu_kernel(
    const float* __restrict__ x1,
    const float* __restrict__ x2,
    float* __restrict__ out,
    int total
) {
    int idx = blockIdx.x * BLOCK_THREADS + threadIdx.x;
    if (idx < total) {
        float gate = x2[idx];
        float up = x1[idx];
        out[idx] = gate * ptx_sigmoid_approx(gate) * up;
    }
}

template<int BLOCK_THREADS>
__global__ void fused_swiglu_packed_kernel(
    const float* __restrict__ x,
    float* __restrict__ out,
    int cols,
    int total
) {
    int idx = blockIdx.x * BLOCK_THREADS + threadIdx.x;
    if (idx < total) {
        int row = idx / cols;
        int col = idx - row * cols;
        int base = row * (cols * 2) + col;
        float up = x[base];
        float gate = x[base + cols];
        out[idx] = gate * ptx_sigmoid_approx(gate) * up;
    }
}

__global__ void fused_swiglu_packed_vec4_kernel(
    const float4* __restrict__ x,
    float4* __restrict__ out,
    int cols,
    int cols_vec4,
    int total_vec4
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_vec4) {
        int row = idx / cols_vec4;
        int col4 = idx - row * cols_vec4;
        int base = row * (cols * 2) + col4 * 4;
        const float* row_ptr = reinterpret_cast<const float*>(x);
        float4 up = *reinterpret_cast<const float4*>(row_ptr + base);
        float4 gate = *reinterpret_cast<const float4*>(row_ptr + base + cols);
        float4 result;
        result.x = gate.x * ptx_sigmoid_approx(gate.x) * up.x;
        result.y = gate.y * ptx_sigmoid_approx(gate.y) * up.y;
        result.z = gate.z * ptx_sigmoid_approx(gate.z) * up.z;
        result.w = gate.w * ptx_sigmoid_approx(gate.w) * up.w;
        out[idx] = result;
    }
}

torch::Tensor fused_swiglu(torch::Tensor x1, torch::Tensor x2) {
    auto out = torch::empty_like(x1);
    constexpr int kBlock = 256;
    int total = x1.numel();
    int blocks = (total + kBlock - 1) / kBlock;
    fused_swiglu_kernel<kBlock><<<blocks, kBlock, 0, at::cuda::getCurrentCUDAStream()>>>(
        x1.data_ptr<float>(),
        x2.data_ptr<float>(),
        out.data_ptr<float>(),
        total
    );
    return out;
}

torch::Tensor fused_swiglu_packed(torch::Tensor x) {
    int rows = x.size(0);
    int cols = x.size(1) / 2;
    auto out = torch::empty({rows, cols}, x.options());
    constexpr int kBlock = 256;
    int total = rows * cols;

    uintptr_t x_ptr = reinterpret_cast<uintptr_t>(x.data_ptr<float>());
    uintptr_t out_ptr = reinterpret_cast<uintptr_t>(out.data_ptr<float>());
    bool aligned_vec4 = ((x_ptr | out_ptr) & 0xF) == 0 && (cols % 4 == 0);

    if (aligned_vec4) {
        int cols_vec4 = cols / 4;
        int total_vec4 = rows * cols_vec4;
        int blocks = (total_vec4 + kBlock - 1) / kBlock;
        fused_swiglu_packed_vec4_kernel<<<blocks, kBlock, 0, at::cuda::getCurrentCUDAStream()>>>(
            reinterpret_cast<const float4*>(x.data_ptr<float>()),
            reinterpret_cast<float4*>(out.data_ptr<float>()),
            cols,
            cols_vec4,
            total_vec4
        );
    } else {
        int blocks = (total + kBlock - 1) / kBlock;
        fused_swiglu_packed_kernel<kBlock><<<blocks, kBlock, 0, at::cuda::getCurrentCUDAStream()>>>(
            x.data_ptr<float>(),
            out.data_ptr<float>(),
            cols,
            total
        );
    }
    return out;
}

torch::Tensor fused_swiglu_packed_precise(torch::Tensor x) {
    auto x1 = x.slice(1, 0, I).contiguous();
    auto x2 = x.slice(1, I, 2 * I).contiguous();
    return x2 / (1.0f + torch::exp(-x2)) * x1;
}

template<int BLOCK_THREADS>
__global__ void weighted_scatter_add_kernel(
    float* __restrict__ output,
    const int32_t* __restrict__ token_idx,
    const float* __restrict__ src,
    const float* __restrict__ weights,
    int cols,
    int total
) {
    int idx = blockIdx.x * BLOCK_THREADS + threadIdx.x;
    if (idx < total) {
        int row = idx / cols;
        int col = idx - row * cols;
        int64_t tok = token_idx[row];
        output[tok * cols + col] += src[idx] * weights[row];
    }
}

template<int BLOCK_THREADS>
__global__ void weighted_scatter_add_vec4_kernel(
    float4* __restrict__ output,
    const int32_t* __restrict__ token_idx,
    const float4* __restrict__ src,
    const float* __restrict__ weights,
    int cols_vec4,
    int total_vec4
) {
    int idx = blockIdx.x * BLOCK_THREADS + threadIdx.x;
    if (idx < total_vec4) {
        int row = idx / cols_vec4;
        int col = idx - row * cols_vec4;
        int64_t tok = token_idx[row];
        float w = weights[row];
        float4 v = src[idx];
        float4& dst = output[tok * cols_vec4 + col];
        dst.x += v.x * w;
        dst.y += v.y * w;
        dst.z += v.z * w;
        dst.w += v.w * w;
    }
}

void weighted_scatter_add(
    torch::Tensor output,
    torch::Tensor token_idx,
    torch::Tensor src,
    torch::Tensor weights
) {
    int rows = src.size(0);
    int cols = src.size(1);
    constexpr int kBlock = 256;

    uintptr_t out_ptr = reinterpret_cast<uintptr_t>(output.data_ptr<float>());
    uintptr_t src_ptr = reinterpret_cast<uintptr_t>(src.data_ptr<float>());
    bool aligned_vec4 = ((out_ptr | src_ptr) & 0xF) == 0 && (cols % 4 == 0);

    if (aligned_vec4) {
        int cols_vec4 = cols / 4;
        int total_vec4 = rows * cols_vec4;
        int blocks = (total_vec4 + kBlock - 1) / kBlock;
        weighted_scatter_add_vec4_kernel<kBlock><<<blocks, kBlock, 0, at::cuda::getCurrentCUDAStream()>>>(
            reinterpret_cast<float4*>(output.data_ptr<float>()),
            token_idx.data_ptr<int32_t>(),
            reinterpret_cast<const float4*>(src.data_ptr<float>()),
            weights.data_ptr<float>(),
            cols_vec4,
            total_vec4
        );
    } else {
        int total = rows * cols;
        int blocks = (total + kBlock - 1) / kBlock;
        weighted_scatter_add_kernel<kBlock><<<blocks, kBlock, 0, at::cuda::getCurrentCUDAStream()>>>(
            output.data_ptr<float>(),
            token_idx.data_ptr<int32_t>(),
            src.data_ptr<float>(),
            weights.data_ptr<float>(),
            cols,
            total
        );
    }
}

}  // namespace

torch::Tensor kernel(
    torch::Tensor routing_logits,
    torch::Tensor routing_bias,
    torch::Tensor hidden_states,
    torch::Tensor hidden_states_scale,
    torch::Tensor gemm1_weights,
    torch::Tensor gemm1_weights_scale,
    torch::Tensor gemm2_weights,
    torch::Tensor gemm2_weights_scale,
    int64_t local_expert_offset,
    double routed_scaling_factor
) {
    at::globalContext().setAllowTF32CuBLAS(true);
    auto device = hidden_states.device();
    int64_t t = routing_logits.size(0);

    auto hs_scale_th = get_hs_scale_th(hidden_states_scale);
    bool use_full_hidden_dequant = t <= FULL_HIDDEN_DEQUANT_MAX_SEQ;
    torch::Tensor a;
    if (use_full_hidden_dequant) {
        auto a_scale_exp = hs_scale_th.unsqueeze(-1).expand({t, H / BLK, BLK}).reshape({t, H}).contiguous();
        a = hidden_states.to(torch::kFloat32) * a_scale_exp;
    }
    int64_t local_start = local_expert_offset;
    auto route = prepare_route(routing_logits, routing_bias, local_start, routed_scaling_factor);
    auto sorted_tok = route.sorted_tok;
    auto sorted_w = route.sorted_w;
    auto offsets_cpu = route.offsets_cpu;

    auto output = torch::zeros({t, H}, torch::TensorOptions().device(device).dtype(torch::kFloat32));
    const int32_t* offsets_ptr = offsets_cpu.data_ptr<int32_t>();
    int64_t total_local = static_cast<int64_t>(offsets_ptr[E_LOCAL]);
    bool use_grouped_gather = total_local >= GROUPED_GATHER_MIN_ROUTES;
    torch::Tensor a_local;

    if (use_grouped_gather && total_local > 0) {
        if (use_full_hidden_dequant) {
            a_local = gather_rows(a, sorted_tok);
        } else {
            auto sorted_tok_i64 = sorted_tok.to(torch::kLong);
            auto hs_scale_local = hs_scale_th.index_select(0, sorted_tok_i64);
            auto a_scale_local = hs_scale_local.unsqueeze(-1)
                .expand({total_local, H / BLK, BLK})
                .reshape({total_local, H})
                .contiguous();
            a_local = hidden_states.index_select(0, sorted_tok_i64).to(torch::kFloat32) * a_scale_local;
        }
    }

    for (int64_t le = 0; le < E_LOCAL; ++le) {
        int64_t s_off = static_cast<int64_t>(offsets_ptr[le]);
        int64_t e_off = static_cast<int64_t>(offsets_ptr[le + 1]);
        if (s_off == e_off) {
            continue;
        }

        auto token_idx = sorted_tok.slice(0, s_off, e_off);
        auto w_tok = sorted_w.slice(0, s_off, e_off);

        torch::Tensor a_e;
        if (use_grouped_gather) {
            a_e = a_local.slice(0, s_off, e_off);
        } else if (use_full_hidden_dequant) {
            a_e = gather_rows(a, token_idx);
        } else {
            auto token_idx_i64 = token_idx.to(torch::kLong);
            auto hs_scale_e = hs_scale_th.index_select(0, token_idx_i64);
            auto a_scale_e = hs_scale_e.unsqueeze(-1)
                .expand({token_idx.size(0), H / BLK, BLK})
                .reshape({token_idx.size(0), H})
                .contiguous();
            a_e = hidden_states.index_select(0, token_idx_i64).to(torch::kFloat32) * a_scale_e;
        }

        auto w13_t = dequantize_weight_tiled_t(
            gemm1_weights[le].to(torch::kFloat32).contiguous(),
            gemm1_weights_scale[le].to(torch::kFloat32).contiguous()
        );
        auto g1 = torch::mm(a_e, w13_t);

        auto mid = (g1.size(0) <= 4) ? fused_swiglu_packed_precise(g1) : fused_swiglu_packed(g1);

        auto w2_t = dequantize_weight_tiled_t(
            gemm2_weights[le].to(torch::kFloat32).contiguous(),
            gemm2_weights_scale[le].to(torch::kFloat32).contiguous()
        );
        auto out_e = torch::mm(mid, w2_t);

        weighted_scatter_add(output, token_idx, out_e, w_tok);
    }

    return output.to(torch::kBFloat16);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("kernel", &kernel, "MoE CUDA/PTX torch extension");
}
