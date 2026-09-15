#include "uocr/continuous_batch.h"

#include <algorithm>
#include <ctime>

#include "uocr/log.h"

namespace uocr {

int ContinuousBatchScheduler::add_request(Request r) {
    r.id = next_id_++;
    if (r.arrival_time == 0.0) {
        r.arrival_time = static_cast<double>(::clock()) / CLOCKS_PER_SEC;
    }
    waiting_.push_back(r.id);
    reqs_.emplace(r.id, std::move(r));
    UOCR_DEBUG("scheduler: queued request %d", r.id);
    return next_id_ - 1;
}

Batch ContinuousBatchScheduler::build_batch() {
    Batch b;

    // Continue running decodes first.
    for (int id : running_) {
        auto it = reqs_.find(id);
        if (it == reqs_.end() || it->second.is_finished()) continue;
        if (static_cast<int>(b.prefill.size() + b.decode.size()) >= max_batch_size_) break;
        if (it->second.state == ReqState::Decoding) b.decode.push_back(id);
        else if (it->second.state == ReqState::Prefilling) b.prefill.push_back(id);
    }

    // Admit waiting requests.
    while (!waiting_.empty() &&
           static_cast<int>(b.prefill.size() + b.decode.size()) < max_batch_size_) {
        const int id = waiting_.front();
        waiting_.pop_front();
        auto it = reqs_.find(id);
        if (it == reqs_.end()) continue;
        it->second.state = ReqState::Prefilling;
        running_.push_back(id);
        b.prefill.push_back(id);
    }

    // Remove finished requests from running set.
    running_.erase(std::remove_if(running_.begin(), running_.end(),
                                  [&](int id) {
                                      auto it = reqs_.find(id);
                                      return it == reqs_.end() || it->second.is_finished();
                                  }),
                   running_.end());

    return b;
}

void ContinuousBatchScheduler::mark_prefilling(int id) {
    auto it = reqs_.find(id);
    if (it != reqs_.end()) it->second.state = ReqState::Prefilling;
}

void ContinuousBatchScheduler::mark_prefilled(int id) {
    auto it = reqs_.find(id);
    if (it == reqs_.end()) return;
    it->second.state = ReqState::Decoding;
    it->second.prefill_time = static_cast<double>(::clock()) / CLOCKS_PER_SEC;
}

bool ContinuousBatchScheduler::add_token(int id, int token) {
    auto it = reqs_.find(id);
    if (it == reqs_.end()) return true;
    Request& r = it->second;
    r.output_tokens.push_back(token);
    r.decode_steps += 1;
    if (r.first_token_time == 0.0)
        r.first_token_time = static_cast<double>(::clock()) / CLOCKS_PER_SEC;

    if (static_cast<int>(r.output_tokens.size()) >= r.max_new_tokens) {
        finish(id);
        return true;
    }
    return false;
}

void ContinuousBatchScheduler::finish(int id) {
    auto it = reqs_.find(id);
    if (it == reqs_.end()) return;
    it->second.state = ReqState::Finished;
    it->second.finish_time = static_cast<double>(::clock()) / CLOCKS_PER_SEC;
    it->second.cache.reset();
    running_.erase(std::remove(running_.begin(), running_.end(), id), running_.end());
}

Request* ContinuousBatchScheduler::get(int id) {
    auto it = reqs_.find(id);
    return it == reqs_.end() ? nullptr : &it->second;
}

const Request* ContinuousBatchScheduler::get(int id) const {
    auto it = reqs_.find(id);
    return it == reqs_.end() ? nullptr : &it->second;
}

ContinuousBatchScheduler::Stats ContinuousBatchScheduler::stats() const {
    Stats s;
    for (const auto& kv : reqs_) {
        s.total_requests += 1;
        if (kv.second.is_finished()) {
            s.finished_requests += 1;
            s.prefill_tokens += kv.second.prompt_tokens.size();
            s.decode_tokens += kv.second.output_tokens.size();
        }
    }
    s.waiting = waiting_.size();
    s.running = running_.size();
    s.last_batch_size = static_cast<double>(running_.size());
    return s;
}

}  // namespace uocr
