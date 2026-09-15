#include "test_main.h"
#include "uocr/continuous_batch.h"

using namespace uocr;

UOCR_TEST(scheduler_continuous_batching) {
    ContinuousBatchScheduler sched(4, 2);
    for (int i = 0; i < 6; ++i) {
        Request r;
        r.prompt_tokens = {1, 2, 3};
        r.max_new_tokens = 2;
        sched.add_request(r);
    }
    CHECK_EQ(sched.stats().waiting, 6u);

    // first step admits up to max_batch_size (4)
    Batch b1 = sched.build_batch();
    CHECK_EQ(b1.prefill.size(), 4u);
    CHECK_EQ(b1.decode.size(), 0u);
    for (int id : b1.prefill) sched.mark_prefilled(id);

    // batch is full with decodes, so no new request is admitted this step
    Batch b2 = sched.build_batch();
    CHECK_EQ(b2.decode.size(), 4u);
    CHECK_EQ(b2.prefill.size(), 0u);

    // generate first token (not finished yet)
    for (int id : b2.decode) CHECK(!sched.add_token(id, 10));
    CHECK_EQ(sched.stats().finished_requests, 0u);

    // second token finishes the four running requests
    Batch b3 = sched.build_batch();
    CHECK_EQ(b3.decode.size(), 4u);
    for (int id : b3.decode) CHECK(sched.add_token(id, 11));
    CHECK_EQ(sched.stats().finished_requests, 4u);

    // freed slots are reused by the two waiting requests
    Batch b4 = sched.build_batch();
    CHECK_EQ(b4.prefill.size(), 2u);
    for (int id : b4.prefill) sched.mark_prefilled(id);
    Batch b5 = sched.build_batch();
    for (int id : b5.decode) CHECK(!sched.add_token(id, 10));
    Batch b6 = sched.build_batch();
    for (int id : b6.decode) CHECK(sched.add_token(id, 11));

    CHECK_EQ(sched.stats().finished_requests, 6u);
    CHECK(!sched.has_work());
}

UOCR_TEST(scheduler_min_batch_size) {
    ContinuousBatchScheduler sched(16, 4);
    Request r;
    r.prompt_tokens = {1};
    sched.add_request(r);
    Batch b = sched.build_batch();
    CHECK_EQ(b.prefill.size(), 1u);
    CHECK(sched.min_batch_size() == 4);
}
