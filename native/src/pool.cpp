// One persistent set of worker threads for every native function: a call hands out chunks of its rows
// and the calling thread works too, so a call on a small tile costs microseconds to start instead of
// creating and joining up to 20 threads. Calls come one at a time from the JS thread; a parallel call
// made from inside a running chunk simply runs inline. A job's settings change only while no worker is
// inside work() (`active` == 0): a worker that woke late for a finished job could otherwise take a chunk
// of the next one with the old chunk count or function (it deadlocked and crashed under load, 10-18).
// shutdown_pool() stops and joins the workers
// (the Photoshop addon calls it when UXP unloads the addon, so no thread outlives the DLL's code).
#include <atomic>
#include <condition_variable>
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
            if (a < b) (*job)(a, b);
            std::lock_guard<std::mutex> lk(mu);
            if (--pending == 0) cv_done.notify_all();
        }
    }
};

Pool* g_pool = nullptr;

}  // namespace

void run_parallel(int n, int t, const std::function<void(int, int)>& fn) {
    if (t <= 1 || n <= 1 || tl_busy) {
        fn(0, n);
        return;
    }
    if (!g_pool) g_pool = new Pool();
    Pool& P = *g_pool;
    while ((int)P.workers.size() < t - 1) P.workers.emplace_back([&P] { P.loop(); });
    {
        std::unique_lock<std::mutex> lk(P.mu);
        P.cv_done.wait(lk, [&] { return P.active == 0; });   // nobody still inside the last job
        P.job = &fn;
        P.n = n;
        P.chunks = t;
        P.chunk = (n + t - 1) / t;
        P.pending = t;
        P.next.store(0);
        P.gen++;
    }
    P.cv_job.notify_all();
    tl_busy = true;
    P.work();
    tl_busy = false;
    std::unique_lock<std::mutex> lk(P.mu);
    P.cv_done.wait(lk, [&] { return P.pending == 0 && P.active == 0; });
    P.job = nullptr;
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
