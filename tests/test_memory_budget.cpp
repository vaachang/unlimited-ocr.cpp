#include "test_main.h"
#include "uocr/block_manager.h"
#include "uocr/memory_pool.h"

using namespace uocr;

UOCR_TEST(memory_pool_basic) {
    MemoryPool pool(1024, 64);
    CHECK_EQ(pool.total(), 1024u);
    CHECK_EQ(pool.used(), 0u);

    i64 a = pool.allocate(100);
    CHECK(a >= 0);
    CHECK_EQ(static_cast<std::size_t>(a) % 64, 0u);
    i64 b = pool.allocate(200);
    CHECK(b > a);
    // `used` accounts for alignment padding as well (28 bytes before b)
    CHECK_EQ(pool.used(), 328u);

    pool.free(a, 100);
    CHECK_EQ(pool.used(), 228u);

    // freed + adjacent region coalesces; a new allocation should fit in place
    i64 c = pool.allocate(50);
    CHECK(c >= 0);
    CHECK_EQ(pool.num_free_blocks() <= 3, true);
}

UOCR_TEST(memory_pool_fragmentation) {
    MemoryPool pool(1000, 1);
    std::vector<i64> offs;
    for (int i = 0; i < 4; ++i) offs.push_back(pool.allocate(200));
    // free every other block -> fragmented (holes at 0 and 400, tail at 800)
    pool.free(offs[0], 200);
    pool.free(offs[2], 200);
    CHECK(pool.fragmentation() > 0.5);
    // a 300-byte alloc cannot fit in 200-byte holes
    CHECK_EQ(pool.allocate(300), -1);
    // but a 200-byte one can
    CHECK(pool.allocate(200) >= 0);
}

UOCR_TEST(block_manager_prefix_sharing) {
    BlockManager mgr(1 << 20, 1 << 20);
    auto* e1 = mgr.acquire_prefix("docA", 100, 4096);
    CHECK(e1 != nullptr);
    CHECK_EQ(e1->refcount, 1);
    CHECK_EQ(mgr.stats().prefix_shared_bytes, 0u);

    auto* e2 = mgr.acquire_prefix("docA", 100, 4096);
    CHECK(e2 != nullptr);
    CHECK_EQ(e2->refcount, 2);
    CHECK_EQ(mgr.stats().live_prefixes, 1u);
    CHECK_EQ(mgr.stats().prefix_shared_bytes, 4096u);

    auto* e3 = mgr.acquire_prefix("docB", 50, 2048);
    CHECK(e3 != nullptr);
    CHECK_EQ(mgr.stats().live_prefixes, 2u);

    mgr.release_prefix("docA");
    CHECK_EQ(mgr.stats().live_prefixes, 2u);
    mgr.release_prefix("docA");
    CHECK_EQ(mgr.stats().live_prefixes, 1u);
    mgr.release_prefix("docB");
    CHECK_EQ(mgr.stats().live_prefixes, 0u);
}
