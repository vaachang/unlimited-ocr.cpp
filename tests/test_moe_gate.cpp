#include "test_main.h"
#include "uocr/moe_decoder.h"

#include <cmath>
#include <vector>

using namespace uocr;

UOCR_TEST(moe_gate_topk) {
    ModelConfig cfg;
    cfg.hidden_size = 4;
    cfg.n_routed_experts = 4;
    cfg.num_experts_per_tok = 2;
    cfg.norm_topk_prob = true;
    cfg.routed_scaling_factor = 1.0f;
    cfg.scoring_func = "softmax";
    cfg.topk_method = "greedy";

    WeightMatrix router;
    router.rows = 4;
    router.cols = 4;
    router.fmt = WeightFormat::F32_OWNED;
    // expert 0 strongly matches dimension 0, etc.
    router.f32 = {5, 0, 0, 0,
                  0, 5, 0, 0,
                  0, 0, 5, 0,
                  0, 0, 0, 5};

    std::vector<float> x = {2.0f, 0.1f, 0.1f, 0.1f};
    std::vector<int> ids;
    std::vector<float> weights;
    moe_detail::moe_gate(x.data(), router, cfg, 1, ids, weights);

    CHECK_EQ(static_cast<int>(ids.size()), 2);
    CHECK_EQ(ids[0], 0);
    // normalized weights sum to routed_scaling_factor
    CHECK_NEAR(weights[0] + weights[1], 1.0, 1e-4);
    CHECK(weights[0] > weights[1]);
}

UOCR_TEST(moe_gate_scaling) {
    ModelConfig cfg;
    cfg.hidden_size = 4;
    cfg.n_routed_experts = 4;
    cfg.num_experts_per_tok = 2;
    cfg.norm_topk_prob = false;
    cfg.routed_scaling_factor = 2.0f;

    WeightMatrix router;
    router.rows = 4;
    router.cols = 4;
    router.fmt = WeightFormat::F32_OWNED;
    router.f32.assign(16, 0.0f);
    for (int i = 0; i < 4; ++i) router.f32[i * 4 + i] = 1.0f;

    std::vector<float> x = {1.0f, 0.0f, 0.0f, 0.0f};
    std::vector<int> ids;
    std::vector<float> weights;
    moe_detail::moe_gate(x.data(), router, cfg, 1, ids, weights);
    // raw softmax probabilities for the top-2 experts, scaled by 2 (no renormalization)
    // softmax([1,0,0,0]) -> [0.4754, 0.1749, 0.1749, 0.1749]; top2 sum = 0.6503
    CHECK_NEAR(weights[0], 0.4754 * 2.0, 2e-3);
    CHECK_NEAR(weights[0] + weights[1], 0.6503 * 2.0, 2e-3);
}
