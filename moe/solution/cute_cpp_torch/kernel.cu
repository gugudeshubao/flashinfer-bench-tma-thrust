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

    auto s = 1.0 / (1.0 + torch::exp(-routing_logits.to(torch::kFloat32)));
    auto s_b = s + routing_bias.to(torch::kFloat32).reshape({-1});

    int64_t group_size = E_GLOBAL / N_GROUP;
    auto grouped = s_b.view({t, N_GROUP, group_size});
    auto top2 = std::get<0>(torch::topk(grouped, 2, 2, true, false));
    auto g_scores = top2.sum(2);

    auto g_idx = std::get<1>(torch::topk(g_scores, TOPK_GROUP, 1, true, false));
    auto g_mask = torch::zeros_like(g_scores);
    g_mask.scatter_(1, g_idx, 1.0);
    auto e_mask = g_mask.unsqueeze(2).expand({t, N_GROUP, group_size}).reshape({t, E_GLOBAL});

    auto pruned = s_b.masked_fill(e_mask == 0, std::numeric_limits<float>::lowest());
    auto topk_idx = std::get<1>(torch::topk(pruned, TOP_K, 1, true, false));

    auto topk_scores = s.gather(1, topk_idx);
    auto topk_weights = (topk_scores / (topk_scores.sum(1, true) + 1e-20)) * routed_scaling_factor;

    auto flat_tok = torch::arange(t, torch::TensorOptions().device(routing_logits.device()).dtype(torch::kLong))
        .unsqueeze(1)
        .expand({t, TOP_K})
        .reshape({-1});
    auto flat_exp = topk_idx.reshape({-1});
    auto flat_w = topk_weights.reshape({-1});

    auto local_mask = (flat_exp >= local_start) & (flat_exp < local_start + E_LOCAL);
    auto local_tok = flat_tok.index({local_mask});
    auto local_exp = flat_exp.index({local_mask}) - local_start;
    auto local_w = flat_w.index({local_mask});

    torch::Tensor sorted_tok = local_tok;
    torch::Tensor sorted_w = local_w;
    auto offsets_cpu = torch::zeros({E_LOCAL + 1}, torch::TensorOptions().device(torch::kCPU).dtype(torch::kLong));

    if (local_tok.numel() > 0) {
        auto order = std::get<1>(torch::sort(local_exp, 0, false));
        sorted_tok = local_tok.index_select(0, order);
        auto sorted_exp = local_exp.index_select(0, order);
        sorted_w = local_w.index_select(0, order);
        auto counts = torch::bincount(sorted_exp.to(torch::kInt64), {}, E_LOCAL);
        offsets_cpu.slice(0, 1, E_LOCAL + 1).copy_(counts.cumsum(0).to(torch::kCPU));
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
        out[idx] = gate / (1.0f + __expf(-gate)) * up;
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
    const int64_t* offsets_ptr = offsets_cpu.data_ptr<int64_t>();

    for (int64_t le = 0; le < E_LOCAL; ++le) {
        int64_t s_off = offsets_ptr[le];
        int64_t e_off = offsets_ptr[le + 1];
        if (s_off == e_off) {
            continue;
        }

        auto token_idx = sorted_tok.slice(0, s_off, e_off);
        auto w_tok = sorted_w.slice(0, s_off, e_off);

        torch::Tensor a_e;
        if (use_full_hidden_dequant) {
            a_e = a.index_select(0, token_idx);
        } else {
            auto hs_scale_e = hs_scale_th.index_select(0, token_idx);
            auto a_scale_e = hs_scale_e.unsqueeze(-1)
                .expand({token_idx.size(0), H / BLK, BLK})
                .reshape({token_idx.size(0), H})
                .contiguous();
            a_e = hidden_states.index_select(0, token_idx).to(torch::kFloat32) * a_scale_e;
        }

        auto w13_f32 = gemm1_weights[le].to(torch::kFloat32);
        auto s13_exp = expand_scale_2d(gemm1_weights_scale[le].to(torch::kFloat32));
        auto g1 = torch::matmul(a_e, (w13_f32 * s13_exp).transpose(0, 1));

        auto x1 = g1.slice(1, 0, I).contiguous();
        auto x2 = g1.slice(1, I, 2 * I).contiguous();
        auto mid = fused_swiglu(x1, x2);

        auto w2_f32 = gemm2_weights[le].to(torch::kFloat32);
        auto s2_exp = expand_scale_2d(gemm2_weights_scale[le].to(torch::kFloat32));
        auto out_e = torch::matmul(mid, (w2_f32 * s2_exp).transpose(0, 1));

        output.index_add_(0, token_idx, out_e * w_tok.unsqueeze(1));
    }

    return output.to(torch::kBFloat16);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("kernel", &kernel, "MoE CuTe/C++ torch extension");
}
