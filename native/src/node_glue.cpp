// Node-API binding (filmdust.node): the same functions as the Photoshop addon, so the engine's
// native path can be tested and timed in node against the plain JS engine.
#define NAPI_VERSION 8
#include <node_api.h>

#include <cstddef>

#include "filters.h"

#define ENV napi_env
#define VAL napi_value
#define CBINFO napi_callback_info
#define ARGS(env, info, n)        \
    size_t argc = n;              \
    napi_value argv[n] = {};      \
    napi_get_cb_info(env, info, &argc, argv, nullptr, nullptr)

template <typename T>
static T* ab_ptr(napi_env env, napi_value v, size_t off, size_t n) {
    void* data = nullptr;
    size_t len = 0;
    if (napi_get_arraybuffer_info(env, v, &data, &len) != napi_ok || !data) return nullptr;
    if (off + n * sizeof(T) > len) return nullptr;
    return (T*)((char*)data + off);
}
#define AB(env, v, off, T, n) ab_ptr<T>(env, v, off, n)

static int i32(napi_env env, napi_value v) {
    int32_t r = 0;
    napi_get_value_int32(env, v, &r);
    return r;
}
static double f64(napi_env env, napi_value v) {
    double r = 0;
    napi_get_value_double(env, v, &r);
    return r;
}
#define I32(env, v) i32(env, v)
#define F64(env, v) f64(env, v)

static napi_value new_ab(napi_env env, size_t bytes, void** ptr) {
    napi_value r = nullptr;
    napi_create_arraybuffer(env, bytes, ptr, &r);
    return r;
}
#define NEW_AB(env, bytes, pptr) new_ab(env, bytes, pptr)

static napi_value num(napi_env env, double x) {
    napi_value r = nullptr;
    napi_create_double(env, x, &r);
    return r;
}
#define NUM(env, x) num(env, x)
#define THROW(env, msg) napi_throw_error(env, nullptr, msg)

#include "glue.inc"

NAPI_MODULE_INIT() {
    for (const auto& e : FD_FUNCTIONS) {
        napi_value fn;
        napi_create_function(env, e.name, NAPI_AUTO_LENGTH, e.fn, nullptr, &fn);
        napi_set_named_property(env, exports, e.name, fn);
    }
    return exports;
}
