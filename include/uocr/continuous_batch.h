#pragma once

// Continuous batching scheduler.
//
// Maintains a waiting queue and a running set.  Every scheduler step it fills
// a batch up to `max_batch_size` by admitting new requests (prefill) and
// continuing running requests (decode).  Requests that finish immediately
// release their KV cache and memory-pool blocks.

#include <deque>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "uocr/common.h"
#include "uocr/kv_cache.h"

namespace uocr {

enum class ReqState { Waiting, Prefilling, Decoding, Finished };

struct Request {
    int id = -1;
    std::string doc_key;                  // for visual prefix sharing
    std::vector<int> prompt_tokens;
    std::vector<int> output_tokens;
    ReqState state = ReqState::Waiting;
    int max_new_tokens = 4096;
    bool stopped = false;

    std::shared_ptr<RSWACache> cache;

    // scheduling / metrics
    double arrival_time = 0.0;
    double prefill_time = 0.0;
    double first_token_time = 0.0;
    double finish_time = 0.0;
    int decode_steps = 0;

    bool is_finished() const { return state == ReqState::Finished; }
    int seq_len() const {
        return static_cast<int>(prompt_tokens.size() + output_tokens.size());
    }
};

struct Batch {
    std::vector<int> prefill;  // requests that need a (re)prefill this step
    std::vector<int> decode;   // requests that decode one token this step
    bool empty() const { return prefill.empty() && decode.empty(); }
};

class ContinuousBatchScheduler {
public:
    struct Stats {
        std::uint64_t total_requests = 0;
        std::uint64_t finished_requests = 0;
        std::uint64_t prefill_tokens = 0;
        std::uint64_t decode_tokens = 0;
        std::size_t waiting = 0;
        std::size_t running = 0;
        double last_batch_size = 0.0;
    };

    explicit ContinuousBatchScheduler(int max_batch_size = 16, int min_batch_size = 4)
        : max_batch_size_(max_batch_size), min_batch_size_(min_batch_size) {}

    // Adds a request; returns its id.
    int add_request(Request r);

    // Builds the next batch.  New waiting requests are admitted until the batch
    // reaches max_batch_size, the rest of the batch continues running requests.
    Batch build_batch();

    void mark_prefilled(int id);
    void mark_prefilling(int id);

    // Register a generated token.  Returns true when the request is finished.
    bool add_token(int id, int token);

    void finish(int id);

    Request* get(int id);
    const Request* get(int id) const;

    bool empty() const { return reqs_.empty(); }
    bool has_work() const { return !waiting_.empty() || !running_.empty(); }

    int max_batch_size() const { return max_batch_size_; }
    int min_batch_size() const { return min_batch_size_; }
    void set_max_batch_size(int v) { max_batch_size_ = v; }
    void set_min_batch_size(int v) { min_batch_size_ = v; }

    Stats stats() const;
    const std::vector<int>& running() const { return running_; }

private:
    int max_batch_size_;
    int min_batch_size_;
    int next_id_ = 0;
    std::deque<int> waiting_;
    std::vector<int> running_;
    std::unordered_map<int, Request> reqs_;
};

}  // namespace uocr
