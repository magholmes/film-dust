// One persistent set of worker threads for every native function. A job's work (rows, columns, objects) is cut
// into chunks that whichever thread is free takes next, the calling thread included, so starting a job costs
// microseconds instead of creating and joining threads, and uneven work evens out. Jobs come one at a time from
// the JS thread; a parallel call made from inside a running chunk simply runs inline. A job's settings change
// only while no worker is inside work() (`active` == 0): a worker that woke late for a finished job could
// otherwise take a chunk of the next one with the old chunk count or function.
// An exception in any chunk (memory running out) stops the job's remaining chunks and is rethrown on the
// calling thread, where the binding turns it into a JS error; it never escapes a worker thread.
// shutdown_pool() stops and joins the workers (the Photoshop addon calls it when UXP unloads the addon, so no
// thread outlives the DLL's code).
#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <exception>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

#include "filters.h"

namespace fd {

namespace {

thread_local bool tl_busy = false;

struct Pool {
    std::mutex mu;
    std::condition_variable cv_job, cv_done;
    std::vector<std::thread> workers;
    const std::function<void(int, int)>* job = nullptr;
    int n = 0, chunks = 0, chunk = 0, pending = 0, active = 0;
    std::atomic<int> next{0};
    std::atomic<bool> failed{false};
    std::exception_ptr error;
    unsigned long long gen = 0;
    bool quit = false;

    void loop() {
        tl_busy = true;
        unsigned long long seen = 0;
        for (;;) {
            {
                std::unique_lock<std::mutex> lk(mu);
                cv_job.wait(lk, [&] { return quit || gen != seen; });
                if (quit) return;
                seen = gen;
                active++;
            }
            work();
            std::lock_guard<std::mutex> lk(mu);
            if (--active == 0) cv_done.notify_all();
        }
    }

    void work() {
        for (;;) {
            const int c = next.fetch_add(1);
            if (c >= chunks) return;
            const int a = c * chunk, b = std::min(n, a + chunk);
            if (a < b && !failed.load()) {
                try {
                    (*job)(a, b);
                } catch (...) {
                    std::lock_guard<std::mutex> lk(mu);
                    if (!error) error = std::current_exception();
                    failed.store(true);
                }
            }
            std::lock_guard<std::mutex> lk(mu);
            if (--pending == 0) cv_done.notify_all();
        }
    }
};

Pool* g_pool = nullptr;

}  // namespace

void run_parallel(int n, int t, const std::function<void(int, int)>& fn, int chunks) {
    if (t <= 1 || n <= 1 || tl_busy) {
        fn(0, n);
        return;
    }
    if (!g_pool) g_pool = new Pool();
    Pool& P = *g_pool;
    while ((int)P.workers.size() < t - 1) P.workers.emplace_back([&P] { P.loop(); });
    // more chunks than threads: whoever is free takes the next one, so uneven work (objects of every size, slower
    // cores) evens out
    chunks = std::max(t, std::min(chunks, n));
    {
        std::unique_lock<std::mutex> lk(P.mu);
        P.cv_done.wait(lk, [&] { return P.active == 0; });   // nobody still inside the last job
        P.job = &fn;
        P.n = n;
        P.chunk = (n + chunks - 1) / chunks;
        P.chunks = (n + P.chunk - 1) / P.chunk;
        P.pending = P.chunks;
        P.next.store(0);
        P.failed.store(false);
        P.error = nullptr;
        P.gen++;
    }
    P.cv_job.notify_all();
    tl_busy = true;
    P.work();
    tl_busy = false;
    std::unique_lock<std::mutex> lk(P.mu);
    P.cv_done.wait(lk, [&] { return P.pending == 0 && P.active == 0; });
    P.job = nullptr;
    std::exception_ptr e = P.error;
    P.error = nullptr;
    lk.unlock();
    if (e) std::rethrow_exception(e);
}

void parallel_rows(int n, int64_t cost, const std::function<void(int, int)>& fn, int min_items) {
    const int64_t by_work = std::max<int64_t>(1, (int64_t)n * std::max<int64_t>(1, cost) >> 16);
    const int64_t by_items = std::max(1, n / std::max(1, min_items));
    const int t = (int)std::min<int64_t>({(int64_t)threads(), by_items, by_work});
    if (t <= 1) fn(0, n);
    else run_parallel(n, t, fn);
}

void shutdown_pool() {
    if (!g_pool) return;
    {
        std::lock_guard<std::mutex> lk(g_pool->mu);
        g_pool->quit = true;
    }
    g_pool->cv_job.notify_all();
    for (auto& th : g_pool->workers)
        if (th.joinable()) th.join();
    delete g_pool;
    g_pool = nullptr;
}

}  // namespace fd
